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

  The NTM uses a 4-stage addressing pipeline for each head:

  1. **Content addressing**: Cosine similarity between controller key and
     memory rows, scaled by sharpness parameter beta → softmax
  2. **Interpolation**: `w = g * w_content + (1-g) * w_prev` blends
     content-based weights with previous location weights
  3. **Circular shift**: Convolves weights with a learned 3-element kernel
     `[shift_left, stay, shift_right]` to move focus
  4. **Sharpening**: `w = w^gamma / sum(w^gamma)` concentrates the
     distribution (gamma >= 1 prevents blurring)

  ## Write Mechanism

  The write head updates memory via erase-then-add:

      M_new = M * (1 - w * e^T) + w * a^T

  where w is the address weights, e is the erase vector (sigmoid, [0,1]),
  and a is the add vector.

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
  - Full 4-stage addressing: content → interpolation → shift → sharpening

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
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:controller_size, pos_integer()}
          | {:input_size, pos_integer()}
          | {:memory_dim, pos_integer()}
          | {:memory_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:output_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
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

    # Controller: processes input concatenated with initial read vectors
    controller_input = build_controller_input(input, memory, memory_dim, num_heads)
    controller_out = build_controller(controller_input, controller_size)

    # Write head: full addressing pipeline → erase/add → updated memory
    updated_memory =
      write_head(controller_out, memory,
        memory_size: memory_size,
        memory_dim: memory_dim,
        name: "write_head"
      )

    # Read head: full addressing pipeline → read from updated memory
    read_result =
      read_head(controller_out, updated_memory,
        memory_size: memory_size,
        memory_dim: memory_dim,
        name: "read_head"
      )

    # Combine controller output with read result for final output
    combined =
      Axon.concatenate([controller_out, read_result],
        name: "ntm_combine"
      )

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
    seq_input =
      Axon.nx(
        input,
        fn x ->
          Nx.new_axis(x, 1)
        end,
        name: "controller_reshape"
      )

    # LSTM controller
    {output_seq, _hidden} =
      Axon.lstm(seq_input, controller_size,
        name: "controller_lstm",
        recurrent_initializer: :glorot_uniform
      )

    # Squeeze back to [batch, controller_size]
    Axon.nx(
      output_seq,
      fn x ->
        Nx.squeeze(x, axes: [1])
      end,
      name: "controller_squeeze"
    )
  end

  @doc """
  Compute read head: full addressing pipeline + weighted read from memory.

  Uses the 4-stage addressing pipeline (content → interpolation → shift →
  sharpening) to compute read weights, then reads a weighted sum from memory.

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

    # Full 4-stage addressing pipeline
    weights =
      build_addressing(controller_out, memory,
        memory_size: memory_size,
        memory_dim: memory_dim,
        name: name
      )

    # Weighted read: w^T * M -> [batch, M]
    Axon.layer(
      fn w, mem, _opts ->
        w_expanded = Nx.new_axis(w, 2)
        Nx.sum(Nx.multiply(w_expanded, mem), axes: [1])
      end,
      [weights, memory],
      name: "#{name}_read",
      op_name: :ntm_read
    )
  end

  @doc """
  Compute write head: full addressing pipeline + erase/add memory update.

  Uses the 4-stage addressing pipeline to compute write weights, then
  updates memory via the erase-then-add mechanism:

      M_new = M * (1 - w * e^T) + w * a^T

  ## Parameters
    - `controller_out` - Controller hidden state `[batch, controller_size]`
    - `memory` - Memory matrix `[batch, N, M]`

  ## Options
    - `:memory_size` - Number of memory rows N
    - `:memory_dim` - Dimension of each memory row M
    - `:name` - Layer name prefix

  ## Returns
    Updated memory `[batch, N, M]`
  """
  @spec write_head(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def write_head(controller_out, memory, opts \\ []) do
    memory_size = Keyword.fetch!(opts, :memory_size)
    memory_dim = Keyword.fetch!(opts, :memory_dim)
    name = Keyword.get(opts, :name, "write_head")

    # Full 4-stage addressing pipeline
    weights =
      build_addressing(controller_out, memory,
        memory_size: memory_size,
        memory_dim: memory_dim,
        name: name
      )

    # Erase vector: sigmoid -> values in [0, 1]
    erase =
      controller_out
      |> Axon.dense(memory_dim, name: "#{name}_erase")
      |> Axon.activation(:sigmoid, name: "#{name}_erase_act")

    # Add vector
    add = Axon.dense(controller_out, memory_dim, name: "#{name}_add")

    # Update memory: M_new = M * (1 - w * e^T) + w * a^T
    Axon.layer(
      &write_memory_update/5,
      [weights, erase, add, memory],
      name: "#{name}_update",
      op_name: :ntm_write
    )
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
    dot_product = Nx.sum(Nx.multiply(key_expanded, memory), axes: [2])

    # Denominator: product of norms
    key_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(key_expanded, key_expanded), axes: [2]), 1.0e-8))
    mem_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(memory, memory), axes: [2]), 1.0e-8))

    # Cosine similarity: [batch, N]
    cosine_sim = Nx.divide(dot_product, Nx.multiply(key_norm, mem_norm))

    # Scale by beta and apply softmax
    scaled = Nx.multiply(beta, cosine_sim)
    max_score = Nx.reduce_max(scaled, axes: [1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scaled, max_score))
    Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [1], keep_axes: true), 1.0e-8))
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Build the controller input by concatenating external input with a simple
  # memory read (uniform-weighted mean of memory rows as initial read vector)
  defp build_controller_input(input, memory, _memory_dim, _num_heads) do
    # Initial read: mean over memory rows (= uniform-weighted read)
    initial_read =
      Axon.nx(
        memory,
        fn m ->
          Nx.mean(m, axes: [1])
        end,
        name: "initial_read"
      )

    Axon.concatenate([input, initial_read], name: "controller_input")
  end

  # Build the full 4-stage addressing pipeline for a head:
  # content addressing → interpolation → circular shift → sharpening
  defp build_addressing(controller_out, memory, opts) do
    name = opts[:name]
    memory_size = opts[:memory_size]
    memory_dim = opts[:memory_dim]

    # Generate head parameters from controller via separate projections
    # Key for content addressing: [batch, M]
    key = Axon.dense(controller_out, memory_dim, name: "#{name}_key")

    # Beta (content sharpness): softplus ensures > 0
    beta_raw = Axon.dense(controller_out, 1, name: "#{name}_beta")

    # Interpolation gate: sigmoid ensures in [0, 1]
    gate_raw = Axon.dense(controller_out, 1, name: "#{name}_gate")

    # Shift kernel: 3 elements for {left, stay, right}, softmax applied in impl
    shift_raw = Axon.dense(controller_out, 3, name: "#{name}_shift")

    # Sharpening gamma: softplus + 1 ensures >= 1
    gamma_raw = Axon.dense(controller_out, 1, name: "#{name}_gamma")

    # Compute the full addressing pipeline
    Axon.layer(
      &addressing_pipeline_impl/7,
      [key, beta_raw, gate_raw, shift_raw, gamma_raw, memory],
      name: "#{name}_address",
      memory_size: memory_size,
      op_name: :ntm_addressing
    )
  end

  # Full 4-stage addressing pipeline implementation
  # Returns addressing weights [batch, N]
  defp addressing_pipeline_impl(key, beta_raw, gate_raw, shift_raw, gamma_raw, memory, _opts) do
    memory_size = Nx.axis_size(memory, 1)

    # Apply activations to raw head parameters
    # Beta > 0 (content addressing sharpness)
    beta = Nx.log(Nx.add(Nx.exp(beta_raw), 1.0))

    # Gate in [0, 1] (interpolation between content and previous location)
    gate = Nx.sigmoid(gate_raw)

    # Gamma >= 1 (sharpening, softplus + 1)
    gamma = Nx.add(Nx.log(Nx.add(Nx.exp(gamma_raw), 1.0)), 1.0)

    # Shift kernel: softmax over 3 elements [left, stay, right]
    shift_max = Nx.reduce_max(shift_raw, axes: [1], keep_axes: true)
    shift_exp = Nx.exp(Nx.subtract(shift_raw, shift_max))
    shift = Nx.divide(shift_exp, Nx.add(Nx.sum(shift_exp, axes: [1], keep_axes: true), 1.0e-8))

    # --- Stage 1: Content addressing ---
    key_expanded = Nx.new_axis(key, 1)
    dot_product = Nx.sum(Nx.multiply(key_expanded, memory), axes: [2])
    key_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(key_expanded, key_expanded), axes: [2]), 1.0e-8))
    mem_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(memory, memory), axes: [2]), 1.0e-8))
    cosine_sim = Nx.divide(dot_product, Nx.multiply(key_norm, mem_norm))
    scaled = Nx.multiply(beta, cosine_sim)
    max_score = Nx.reduce_max(scaled, axes: [1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scaled, max_score))
    w_c = Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [1], keep_axes: true), 1.0e-8))

    # --- Stage 2: Interpolation ---
    # w_g = g * w_c + (1 - g) * w_prev
    # For single-step model, w_prev = uniform(1/N)
    w_prev = Nx.divide(Nx.broadcast(1.0, w_c), memory_size)
    w_g = Nx.add(Nx.multiply(gate, w_c), Nx.multiply(Nx.subtract(1.0, gate), w_prev))

    # --- Stage 3: Circular convolution shift ---
    w_s = circular_conv(w_g, shift)

    # --- Stage 4: Sharpening ---
    # w = w^gamma / sum(w^gamma), gamma >= 1
    w_sharp = Nx.pow(Nx.add(w_s, 1.0e-8), gamma)
    w_sum = Nx.sum(w_sharp, axes: [1], keep_axes: true)
    Nx.divide(w_sharp, Nx.add(w_sum, 1.0e-8))
  end

  # Circular convolution: convolve weight vector with 3-element shift kernel
  # shift: [batch, 3] = [s_{-1}, s_0, s_{+1}]
  # w: [batch, N]
  # Returns: [batch, N] where w_tilde[i] = s[-1]*w[i+1] + s[0]*w[i] + s[1]*w[i-1]
  defp circular_conv(w, shift) do
    n = Nx.axis_size(w, 1)

    # Split shift kernel into components
    # [batch, 1]
    s_left = Nx.slice_along_axis(shift, 0, 1, axis: 1)
    # [batch, 1]
    s_stay = Nx.slice_along_axis(shift, 1, 1, axis: 1)
    # [batch, 1]
    s_right = Nx.slice_along_axis(shift, 2, 1, axis: 1)

    # Roll left: [w[1], w[2], ..., w[N-1], w[0]]
    w_rolled_left =
      Nx.concatenate(
        [
          Nx.slice_along_axis(w, 1, n - 1, axis: 1),
          Nx.slice_along_axis(w, 0, 1, axis: 1)
        ],
        axis: 1
      )

    # Roll right: [w[N-1], w[0], w[1], ..., w[N-2]]
    w_rolled_right =
      Nx.concatenate(
        [
          Nx.slice_along_axis(w, n - 1, 1, axis: 1),
          Nx.slice_along_axis(w, 0, n - 1, axis: 1)
        ],
        axis: 1
      )

    # w_tilde[i] = s_{-1} * w[i+1] + s_0 * w[i] + s_{+1} * w[i-1]
    Nx.add(
      Nx.add(
        Nx.multiply(s_left, w_rolled_left),
        Nx.multiply(s_stay, w)
      ),
      Nx.multiply(s_right, w_rolled_right)
    )
  end

  # Write memory update: M_new = M * (1 - w * e^T) + w * a^T
  defp write_memory_update(weights, erase, add, memory, _opts) do
    # weights: [batch, N] -> [batch, N, 1]
    # erase: [batch, M] -> [batch, 1, M]
    # add: [batch, M] -> [batch, 1, M]
    w_expanded = Nx.new_axis(weights, 2)
    erase_expanded = Nx.new_axis(erase, 1)
    add_expanded = Nx.new_axis(add, 1)

    # Erase: M * (1 - w * e^T)
    erase_matrix = Nx.multiply(w_expanded, erase_expanded)
    erased = Nx.multiply(memory, Nx.subtract(1.0, erase_matrix))

    # Add: + w * a^T
    add_matrix = Nx.multiply(w_expanded, add_expanded)
    Nx.add(erased, add_matrix)
  end
end
