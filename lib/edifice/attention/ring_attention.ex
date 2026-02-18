defmodule Edifice.Attention.RingAttention do
  @moduledoc """
  Ring Attention: chunked attention simulating ring-distributed computation
  (Liu et al., 2023).

  Splits the sequence into chunks and processes attention in a rotating pattern,
  where each query chunk attends to key/value chunks in a ring communication
  order. On a single device, this is equivalent to memory-efficient chunked
  attention but structured as a ring pattern for educational purposes and
  future distributed scaling.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +-----v--------------------+
  | Input Projection          |  Dense to hidden_size
  +---------------------------+
        |
        v
  +-----v--------------------+
  | Ring Attention Block x N  |
  |                           |
  | 1. LayerNorm              |
  | 2. QKV projection         |
  | 3. Split into num_chunks  |
  | 4. Ring attention:        |
  |    For each Q chunk:      |
  |      attend to all K,V    |
  |      chunks in ring order |
  | 5. Residual               |
  | 6. LayerNorm + FFN        |
  | 7. Residual               |
  +---------------------------+
        |
        v
  +---------------------------+
  | Final LayerNorm           |
  +---------------------------+
        |
        v
  [batch, hidden_size]
  ```

  ## Key Insight

  Ring attention enables processing sequences much longer than what fits in
  memory on a single device. The ring pattern naturally maps to distributed
  settings where each device holds one chunk and passes K,V to the next
  device in a ring topology.

  ## Usage

      model = RingAttention.build(
        embed_dim: 288,
        hidden_size: 256,
        num_heads: 4,
        num_chunks: 4,
        num_layers: 4
      )

  ## References

  - Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
  - https://arxiv.org/abs/2310.01889
  """

  alias Edifice.Blocks.{FFN, ModelBuilder}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_chunks 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Ring Attention model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_chunks` - Number of ring chunks to split sequence into (default: 4)
    - `:num_layers` - Number of ring attention layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]` from the last timestep.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_chunks, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_chunks = Keyword.get(opts, :num_chunks, @default_num_chunks)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    block_builder = fn input, block_opts ->
      layer_idx = Keyword.get(block_opts, :layer_idx, 1)

      ring_attention_block(input, hidden_size, num_heads, num_chunks, dropout,
        name: "ring_#{layer_idx}",
        seq_len: window_size
      )
    end

    ModelBuilder.build_sequence_model(
      embed_dim: embed_dim,
      hidden_size: hidden_size,
      num_layers: num_layers,
      block_builder: block_builder,
      window_size: window_size,
      dropout: dropout,
      output_mode: :last_timestep
    )
  end

  @doc """
  Get the output size of a Ring Attention model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  # Ring attention block: norm -> ring_attn -> residual -> norm -> FFN -> residual
  defp ring_attention_block(input, hidden_size, num_heads, num_chunks, dropout, opts) do
    name = Keyword.get(opts, :name, "ring")
    seq_len = Keyword.get(opts, :seq_len, @default_window_size)

    head_dim = div(hidden_size, num_heads)

    # Attention sublayer
    normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    # QKV projection
    qkv = Axon.dense(normed, hidden_size * 3, name: "#{name}_qkv")

    # Ring attention computation
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, actual_seq, _} = Nx.shape(qkv_tensor)

          # Split QKV
          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

          # Reshape to multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
          query = reshape_to_heads(query, batch, actual_seq, num_heads, head_dim)
          key = reshape_to_heads(key, batch, actual_seq, num_heads, head_dim)
          value = reshape_to_heads(value, batch, actual_seq, num_heads, head_dim)

          # Ring attention: process Q in chunks, each attending to all K,V
          output = ring_attention_compute(query, key, value, num_chunks, seq_len)

          # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
          reshape_from_heads(output, batch, actual_seq, num_heads, head_dim)
        end,
        name: "#{name}_ring_compute"
      )

    # Output projection + dropout
    attended = Axon.dense(attended, hidden_size, name: "#{name}_out_proj")

    attended =
      if dropout > 0.0 do
        Axon.dropout(attended, rate: dropout, name: "#{name}_attn_dropout")
      else
        attended
      end

    x = Axon.add(input, attended, name: "#{name}_attn_residual")

    # FFN sublayer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = FFN.layer(ffn_normed, hidden_size: hidden_size, name: "#{name}_ffn")

    ffn_out =
      if dropout > 0.0 do
        Axon.dropout(ffn_out, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn_out
      end

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # Ring attention: split into chunks and process in ring pattern
  # Q, K, V: [batch, heads, seq, head_dim]
  defp ring_attention_compute(query, key, value, num_chunks, seq_len) do
    {batch, heads, _seq, dim} = Nx.shape(query)
    chunk_size = div(seq_len, num_chunks)
    scale = Nx.sqrt(dim) |> Nx.as_type(Nx.type(query))

    # Process each query chunk
    chunk_results =
      for q_idx <- 0..(num_chunks - 1) do
        q_start = q_idx * chunk_size

        q_chunk =
          Nx.slice_along_axis(query, q_start, chunk_size, axis: 2)

        # Ring pattern: attend to K,V chunks in order starting from current position
        # Use online softmax to combine results from each K,V chunk
        tensor_type = Nx.type(query)
        neg_inf = Nx.Constants.neg_infinity() |> Nx.as_type(tensor_type)

        init_output =
          Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, heads, chunk_size, dim})

        init_max = Nx.broadcast(neg_inf, {batch, heads, chunk_size})

        init_sum =
          Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, heads, chunk_size})

        {chunk_output, _final_max, final_sum} =
          Enum.reduce(0..(num_chunks - 1), {init_output, init_max, init_sum}, fn ring_step,
                                                                                 {acc_output,
                                                                                  acc_max,
                                                                                  acc_sum} ->
            # Ring order: (q_idx + ring_step) mod num_chunks
            kv_idx = rem(q_idx + ring_step, num_chunks)
            kv_start = kv_idx * chunk_size

            k_chunk = Nx.slice_along_axis(key, kv_start, chunk_size, axis: 2)
            v_chunk = Nx.slice_along_axis(value, kv_start, chunk_size, axis: 2)

            # Attention scores: [batch, heads, chunk_size, chunk_size]
            scores = Nx.dot(q_chunk, [3], [0, 1], k_chunk, [3], [0, 1])
            scores = Nx.divide(scores, scale)

            # Apply causal mask: query positions can only attend to key positions <= them
            q_positions = Nx.iota({chunk_size, 1}, axis: 0) |> Nx.add(q_start)
            k_positions = Nx.iota({1, chunk_size}, axis: 1) |> Nx.add(kv_start)
            causal_mask = Nx.greater_equal(q_positions, k_positions)

            causal_mask =
              causal_mask
              |> Nx.new_axis(0)
              |> Nx.new_axis(0)
              |> Nx.broadcast({batch, heads, chunk_size, chunk_size})

            scores =
              Nx.select(causal_mask, scores, Nx.broadcast(neg_inf, Nx.shape(scores)))

            # Online softmax update
            chunk_max = Nx.reduce_max(scores, axes: [-1])
            new_max = Nx.max(acc_max, chunk_max)
            old_scale_factor = Nx.exp(Nx.subtract(acc_max, new_max))
            exp_scores = Nx.exp(Nx.subtract(scores, Nx.new_axis(new_max, -1)))
            chunk_sum = Nx.sum(exp_scores, axes: [-1])
            new_sum = Nx.add(Nx.multiply(acc_sum, old_scale_factor), chunk_sum)

            chunk_out = Nx.dot(exp_scores, [3], [0, 1], v_chunk, [2], [0, 1])

            new_output =
              Nx.add(
                Nx.multiply(acc_output, Nx.new_axis(old_scale_factor, -1)),
                chunk_out
              )

            {new_output, new_max, new_sum}
          end)

        # Normalize
        Nx.divide(chunk_output, Nx.new_axis(final_sum, -1))
      end

    # Concatenate all query chunk results
    Nx.concatenate(chunk_results, axis: 2)
  end

  # Reshape [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end
end
