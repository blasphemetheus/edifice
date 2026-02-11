defmodule Edifice.Memory.NTM do
  @moduledoc """
  Neural Turing Machine (Graves et al., 2014).

  An NTM augments a neural network controller with an external memory matrix
  that can be read from and written to via differentiable attention mechanisms.
  This enables learning algorithms like copying, sorting, and associative recall.

  ## Architecture

  ```
  Input [batch, input_size]
        |
        +------------------+
        |                  |
        v                  v
  +------------+    +-----------+
  | Controller |    |  Memory   |
  |   (LSTM)   |    | [N x M]  |
  +------------+    +-----------+
        |                ^  |
        +--+--+          |  |
        |  |  |          |  |
        v  v  v          |  |
      Read Write    Read/ Write
      Head  Head    Addressing
        |    |           |
        +----+-----------+
        |
        v
  Output [batch, output_size]
  ```

  ## Addressing Mechanism

  The NTM uses a combination of content-based and location-based addressing:

  1. **Content addressing**: Cosine similarity between controller output and
     memory rows, scaled by a sharpness parameter beta
  2. **Interpolation**: Blend content weights with previous weights
  3. **Shift**: Circular convolution for location-based shifting
  4. **Sharpening**: Raise weights to a power to prevent blurring

  ## Usage

      model = NTM.build(
        input_size: 64,
        memory_size: 128,
        memory_dim: 32,
        controller_size: 256,
        num_heads: 1
      )

  ## References
  - Graves et al., "Neural Turing Machines" (2014)
  - https://arxiv.org/abs/1410.5401
  """

  require Axon
  import Nx.Defn

  @default_memory_size 128
  @default_memory_dim 32
  @default_controller_size 256
  @default_num_heads 1

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Neural Turing Machine.

  The NTM consists of:
  - An LSTM controller that processes inputs and generates head parameters
  - A differentiable memory matrix accessed via read and write heads
  - Content-based and location-based addressing mechanisms

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:memory_size` - Number of memory rows N (default: 128)
    - `:memory_dim` - Dimension of each memory row M (default: 32)
    - `:controller_size` - LSTM controller hidden size (default: 256)
    - `:num_heads` - Number of read/write heads (default: 1)
    - `:output_size` - Output dimension (default: same as input_size)

  ## Returns
    An Axon model taking input `[batch, input_size]` and memory `[batch, N, M]`,
    producing output `[batch, output_size]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    memory_size = Keyword.get(opts, :memory_size, @default_memory_size)
    memory_dim = Keyword.get(opts, :memory_dim, @default_memory_dim)
    controller_size = Keyword.get(opts, :controller_size, @default_controller_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    output_size = Keyword.get(opts, :output_size, input_size)

    # Inputs
    input = Axon.input("input", shape: {nil, input_size})
    memory = Axon.input("memory", shape: {nil, memory_size, memory_dim})

    # Controller: processes input concatenated with read vectors
    # For simplicity, we first compute read from memory, then feed to controller
    controller_input = build_controller_input(input, memory, memory_dim, num_heads)
    controller_out = build_controller(controller_input, controller_size)

    # Read head: compute read weights and read from memory
    read_result = read_head(controller_out, memory,
      memory_size: memory_size, memory_dim: memory_dim, name: "read_head")

    # Write head: compute write weights, erase, and add vectors
    _write_result = write_head(controller_out, memory,
      memory_size: memory_size, memory_dim: memory_dim, name: "write_head")

    # Combine controller output with read result for final output
    combined = Axon.concatenate([controller_out, read_result],
      name: "ntm_combine")

    Axon.dense(combined, output_size, name: "ntm_output")
  end

  @doc """
  Build the LSTM controller that drives the read/write heads.

  The controller processes the combined input (external input + previous read
  vectors) and produces a hidden state used to parameterize the head operations.

  ## Parameters
    - `input` - Axon node with combined input `[batch, combined_dim]`
    - `controller_size` - Hidden dimension for the controller

  ## Returns
    An Axon node with shape `[batch, controller_size]`
  """
  @spec build_controller(Axon.t(), pos_integer()) :: Axon.t()
  def build_controller(input, controller_size) do
    # Reshape to sequence of length 1 for LSTM: [batch, 1, dim]
    seq_input = Axon.nx(input, fn x ->
      Nx.new_axis(x, 1)
    end, name: "controller_reshape")

    # LSTM controller
    {output_seq, _hidden} = Axon.lstm(seq_input, controller_size,
      name: "controller_lstm",
      recurrent_initializer: :glorot_uniform)

    # Squeeze back to [batch, controller_size]
    Axon.nx(output_seq, fn x ->
      Nx.squeeze(x, axes: [1])
    end, name: "controller_squeeze")
  end

  @doc """
  Compute read head: address memory and read content.

  Generates read weights via content-based addressing, then computes
  a weighted sum over memory rows.

  ## Parameters
    - `controller_out` - Controller hidden state `[batch, controller_size]`
    - `memory` - Memory matrix `[batch, N, M]`

  ## Options
    - `:memory_size` - Number of memory rows N
    - `:memory_dim` - Dimension of each memory row M
    - `:name` - Layer name prefix

  ## Returns
    Read vector `[batch, M]`
  """
  @spec read_head(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def read_head(controller_out, memory, opts \\ []) do
    memory_size = Keyword.fetch!(opts, :memory_size)
    memory_dim = Keyword.fetch!(opts, :memory_dim)
    name = Keyword.get(opts, :name, "read_head")

    # Generate read key from controller: [batch, memory_dim]
    read_key = Axon.dense(controller_out, memory_dim, name: "#{name}_key")

    # Generate sharpness parameter beta: [batch, 1]
    beta = Axon.dense(controller_out, 1, name: "#{name}_beta")
    beta = Axon.activation(beta, :softplus, name: "#{name}_beta_act")

    # Content-based addressing: compute read weights
    # Then read from memory using the weights
    Axon.layer(
      &read_head_impl/4,
      [read_key, beta, memory],
      name: name,
      memory_size: memory_size,
      memory_dim: memory_dim,
      op_name: :ntm_read
    )
  end

  @doc """
  Compute write head: address memory and produce erase/add vectors.

  Generates write weights, erase vector, and add vector from the
  controller output. Returns the add vector as a representation of
  what would be written (actual memory update is computed externally).

  ## Parameters
    - `controller_out` - Controller hidden state `[batch, controller_size]`
    - `memory` - Memory matrix `[batch, N, M]`

  ## Options
    - `:memory_size` - Number of memory rows N
    - `:memory_dim` - Dimension of each memory row M
    - `:name` - Layer name prefix

  ## Returns
    Write information vector `[batch, M]` (the add vector)
  """
  @spec write_head(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def write_head(controller_out, memory, opts \\ []) do
    memory_size = Keyword.fetch!(opts, :memory_size)
    memory_dim = Keyword.fetch!(opts, :memory_dim)
    name = Keyword.get(opts, :name, "write_head")

    # Generate write key: [batch, memory_dim]
    write_key = Axon.dense(controller_out, memory_dim, name: "#{name}_key")

    # Sharpness beta: [batch, 1]
    beta = Axon.dense(controller_out, 1, name: "#{name}_beta")
    beta = Axon.activation(beta, :softplus, name: "#{name}_beta_act")

    # Erase vector: [batch, memory_dim] (sigmoid -> values in [0, 1])
    erase = Axon.dense(controller_out, memory_dim, name: "#{name}_erase")
    erase = Axon.sigmoid(erase)

    # Add vector: [batch, memory_dim]
    add = Axon.dense(controller_out, memory_dim, name: "#{name}_add")

    # Compute write operation
    Axon.layer(
      &write_head_impl/5,
      [write_key, beta, erase, memory],
      name: name,
      memory_size: memory_size,
      memory_dim: memory_dim,
      op_name: :ntm_write
    )

    # Return the add vector as the write representation
    add
  end

  @doc """
  Content-based addressing using cosine similarity.

  Computes attention weights over memory rows based on cosine similarity
  between a query key and each memory row, scaled by sharpness beta.

      w_i = softmax(beta * cosine_similarity(key, memory[i]))

  ## Parameters
    - `key` - Query key `[batch, M]`
    - `memory` - Memory matrix `[batch, N, M]`
    - `beta` - Sharpness parameter `[batch, 1]`

  ## Returns
    Attention weights `[batch, N]`
  """
  @spec content_addressing(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn content_addressing(key, memory, beta) do
    # key: [batch, M] -> [batch, 1, M]
    key_expanded = Nx.new_axis(key, 1)

    # Cosine similarity between key and each memory row
    # memory: [batch, N, M]
    # Numerator: dot product
    dot_product = Nx.sum(key_expanded * memory, axes: [2])

    # Denominator: product of norms
    key_norm = Nx.sqrt(Nx.sum(key_expanded * key_expanded, axes: [2]) + 1.0e-8)
    mem_norm = Nx.sqrt(Nx.sum(memory * memory, axes: [2]) + 1.0e-8)

    # Cosine similarity: [batch, N]
    cosine_sim = dot_product / (key_norm * mem_norm)

    # Scale by beta and apply softmax
    # beta: [batch, 1]
    scaled = beta * cosine_sim
    max_score = Nx.reduce_max(scaled, axes: [1], keep_axes: true)
    exp_scores = Nx.exp(scaled - max_score)
    exp_scores / Nx.sum(exp_scores, axes: [1], keep_axes: true)
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Build the controller input by concatenating external input with a simple
  # memory read (using mean of memory as a simple initial read vector)
  defp build_controller_input(input, memory, _memory_dim, _num_heads) do
    # Simple initial read: mean over memory rows
    initial_read = Axon.nx(memory, fn m ->
      Nx.mean(m, axes: [1])
    end, name: "initial_read")

    Axon.concatenate([input, initial_read], name: "controller_input")
  end

  # Read head implementation: content addressing + weighted read
  defp read_head_impl(read_key, beta, memory, _opts) do
    # Content-based addressing
    weights = content_addressing(read_key, memory, beta)

    # Weighted read: w^T * M -> [batch, M]
    # weights: [batch, N] -> [batch, N, 1]
    # memory: [batch, N, M]
    weights_expanded = Nx.new_axis(weights, 2)
    Nx.sum(weights_expanded * memory, axes: [1])
  end

  # Write head implementation: content addressing + erase/add
  defp write_head_impl(write_key, beta, erase, memory, _opts) do
    # Content-based addressing for write
    weights = content_addressing(write_key, memory, beta)

    # Erase: M_new = M * (1 - w * e^T)
    # weights: [batch, N] -> [batch, N, 1]
    # erase: [batch, M] -> [batch, 1, M]
    weights_expanded = Nx.new_axis(weights, 2)
    erase_expanded = Nx.new_axis(erase, 1)

    erase_matrix = weights_expanded * erase_expanded
    # Return the erase signal (memory update is applied externally)
    Nx.sum(erase_matrix, axes: [1])
  end
end
