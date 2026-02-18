defmodule Edifice.Attention.InfiniAttention do
  @moduledoc """
  Infini-Attention: local windowed attention + compressive memory.

  Extends standard multi-head attention with a compressive memory system
  that enables effectively unbounded context length. Each layer maintains
  a learnable memory matrix that accumulates information from past segments.

  ## Key Innovation

  For each segment of the input:
  1. Standard local attention within the segment (captures fine-grained patterns)
  2. Memory retrieval: sigma(Q) @ M / (sigma(Q) @ z) where sigma = ELU + 1
  3. Memory update: M += sigma(K)^T @ V, z += sum(sigma(K))
  4. A learnable gate blends local and memory outputs

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection to hidden_size
        |
  +----------------------------------------------+
  |   Infini-Attention Block (x num_layers)      |
  |                                              |
  |   LayerNorm -> Infini-Attention              |
  |     Split into segments of segment_size      |
  |     Per segment:                             |
  |       Local multi-head attention             |
  |       Memory retrieval + update              |
  |       Gated blend of local + memory          |
  |   -> Residual                                |
  |   LayerNorm -> FFN -> Residual               |
  +----------------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = InfiniAttention.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        segment_size: 32,
        num_layers: 4
      )

  ## References
  - "Leave No Context Behind: Efficient Infinite Context Transformers with
    Infini-attention" (Munkhdalai et al., 2024)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_segment_size 32
  @default_num_layers 4
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:segment_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build an Infini-Attention model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:segment_size` - Size of each local attention segment (default: 32)
    - `:num_layers` - Number of transformer blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    segment_size = Keyword.get(opts, :segment_size, @default_segment_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "infini_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_infini_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              segment_size: segment_size,
              name: attn_name
            )
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  @doc """
  Build the Infini-Attention layer with segmented local attention and compressive memory.
  """
  @spec build_infini_attention(Axon.t(), keyword()) :: Axon.t()
  def build_infini_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    segment_size = Keyword.get(opts, :segment_size, @default_segment_size)
    name = Keyword.get(opts, :name, "infini_attn")

    head_dim = div(hidden_size, num_heads)

    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Learnable gate parameter for blending local and memory attention
    gate = Axon.param("#{name}_gate", {num_heads, head_dim}, initializer: :zeros)

    output =
      Axon.layer(
        &infini_attention_impl/5,
        [q_proj, k_proj, v_proj, gate],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        segment_size: segment_size,
        op_name: :infini_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Infini-attention implementation
  defp infini_attention_impl(q, k, v, gate, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    segment_size = opts[:segment_size]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Sigmoid gate for blending: [num_heads, head_dim] -> broadcast as needed
    gate_sigmoid = Nx.sigmoid(gate)

    # Number of segments (pad last segment if needed)
    num_segments = div(seq_len + segment_size - 1, segment_size)

    # Initialize compressive memory
    # M: [batch, heads, head_dim, head_dim] - memory matrix
    # z: [batch, heads, head_dim] - normalization vector
    tensor_type = Nx.type(q)

    init_memory =
      Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, num_heads, head_dim, head_dim})

    init_z = Nx.broadcast(Nx.tensor(1.0e-6, type: tensor_type), {batch, num_heads, head_dim})

    # Process each segment
    {segment_outputs, _final_memory, _final_z} =
      Enum.reduce(0..(num_segments - 1), {[], init_memory, init_z}, fn seg_idx,
                                                                       {outputs, memory, z_norm} ->
        start = seg_idx * segment_size
        actual_seg_size = min(segment_size, seq_len - start)

        # Extract segment: [batch, heads, seg_size, head_dim]
        q_seg = Nx.slice_along_axis(q, start, actual_seg_size, axis: 2)
        k_seg = Nx.slice_along_axis(k, start, actual_seg_size, axis: 2)
        v_seg = Nx.slice_along_axis(v, start, actual_seg_size, axis: 2)

        # 1. Local attention within segment
        local_out = local_attention(q_seg, k_seg, v_seg)

        # 2. Memory retrieval: sigma(Q) @ M / (sigma(Q) @ z)
        # sigma = ELU + 1
        sigma_q = Nx.add(Nx.max(q_seg, 0.0), Nx.multiply(Nx.exp(Nx.min(q_seg, 0.0)), 1.0))

        # sigma_q: [batch, heads, seg_size, head_dim]
        # memory: [batch, heads, head_dim, head_dim]
        # retrieval: [batch, heads, seg_size, head_dim]
        retrieval = Nx.dot(sigma_q, [3], [0, 1], memory, [2], [0, 1])

        # Normalize: sigma_q @ z -> [batch, heads, seg_size]
        # z_norm: [batch, heads, head_dim]
        retrieval_norm =
          Nx.dot(sigma_q, [3], [0, 1], Nx.new_axis(z_norm, 3), [2], [0, 1])
          |> Nx.squeeze(axes: [3])

        retrieval_norm = Nx.add(Nx.abs(retrieval_norm), 1.0e-6)
        retrieval = Nx.divide(retrieval, Nx.new_axis(retrieval_norm, -1))

        # 3. Gated blend: gate * memory_output + (1 - gate) * local_output
        # gate_sigmoid: [num_heads, head_dim] -> broadcast to [1, num_heads, 1, head_dim]
        gate_b =
          gate_sigmoid
          |> Nx.reshape({1, num_heads, 1, head_dim})
          |> Nx.broadcast({batch, num_heads, actual_seg_size, head_dim})

        blended =
          Nx.add(Nx.multiply(gate_b, retrieval), Nx.multiply(Nx.subtract(1.0, gate_b), local_out))

        # 4. Update memory: M += sigma(K)^T @ V
        sigma_k = Nx.add(Nx.max(k_seg, 0.0), Nx.multiply(Nx.exp(Nx.min(k_seg, 0.0)), 1.0))

        # sigma_k: [batch, heads, seg_size, head_dim]
        # v_seg: [batch, heads, seg_size, head_dim]
        # Contract seq (axis 2), batch on [batch, heads] => [batch, heads, head_dim, head_dim]
        memory_update = Nx.dot(sigma_k, [2], [0, 1], v_seg, [2], [0, 1])
        new_memory = Nx.add(memory, memory_update)

        # z += sum(sigma(K), axis=seq)
        z_update = Nx.sum(sigma_k, axes: [2])
        new_z = Nx.add(z_norm, z_update)

        {outputs ++ [blended], new_memory, new_z}
      end)

    # Concatenate all segment outputs: [batch, heads, seq_len, head_dim]
    output = Nx.concatenate(segment_outputs, axis: 2)

    # Take only seq_len positions (in case of padding from last segment)
    output =
      if num_segments * segment_size > seq_len do
        Nx.slice_along_axis(output, 0, seq_len, axis: 2)
      else
        output
      end

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Standard scaled dot-product attention for a segment
  # Q, K, V: [batch, heads, seg_size, head_dim]
  defp local_attention(q, k, v) do
    head_dim = Nx.axis_size(q, 3)
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # scores: [batch, heads, seg_size, seg_size]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask within segment
    seg_len = Nx.axis_size(q, 2)
    rows = Nx.iota({seg_len, seg_len}, axis: 0)
    cols = Nx.iota({seg_len, seg_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    batch = Nx.axis_size(q, 0)
    heads = Nx.axis_size(q, 1)

    causal_mask =
      causal_mask
      |> Nx.reshape({1, 1, seg_len, seg_len})
      |> Nx.broadcast({batch, heads, seg_len, seg_len})

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

    # weighted sum: [batch, heads, seg_size, head_dim]
    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  @doc """
  Get the output size of an Infini-Attention model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
