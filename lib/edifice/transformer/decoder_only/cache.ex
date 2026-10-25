defmodule Edifice.Transformer.DecoderOnly.Cache do
  @moduledoc """
  Fixed-capacity KV decoding for DecoderOnly's GQA backbone.

  `build/1` returns `%{init: fn batch -> state end, predict: fn params, frames,
  state -> {outputs, state} end, max_seq_len: capacity}`. Frames and outputs
  have shape `{batch, new_frames, features}`. Only new frames pass through
  projections and feed-forward layers; attention reads the cached keys/values.
  Pass `compiler: EXLA` to compile each chunk shape once (including the
  single-frame decode path). State is a plain tensor container suitable for
  `Edifice.Stateful.snapshot/1` and serialization for rollback.

  Uses the same parameters as `DecoderOnly.build/1`, in inference mode.
  Supports GQA, RoPE/NoPE, interleaved RoPE, input projection and optional final
  normalization. YaRN and other attention types are rejected. The caller must
  reset and prefill on overflow; evicting KV entries would change the meaning
  of the retained higher-layer activations relative to a fresh sliding window.
  """
  alias Edifice.Blocks.{RMSNorm, RoPE}
  alias Edifice.Stateful.Ops

  def build(opts) do
    hidden = Keyword.get(opts, :hidden_size, 256)
    heads = Keyword.get(opts, :num_heads, 8)
    kv_heads = Keyword.get(opts, :num_kv_heads, 2)
    layers = Keyword.get(opts, :num_layers, 4)
    capacity = Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60))

    unless Enum.all?([hidden, heads, kv_heads, layers, capacity], &(is_integer(&1) and &1 > 0)) and
             rem(hidden, heads) == 0 and rem(heads, kv_heads) == 0 do
      raise ArgumentError, "invalid GQA cache dimensions"
    end

    unless Keyword.get(opts, :attention_type, :gqa) == :gqa and
             not Keyword.get(opts, :yarn, false),
           do: raise(ArgumentError, "cached decoding supports GQA without YaRN")

    if (Keyword.get(opts, :use_rope, true) or Keyword.get(opts, :interleave_rope, false)) and
         rem(div(hidden, heads), 2) != 0,
       do: raise(ArgumentError, "RoPE requires an even head dimension")

    config =
      Keyword.merge(opts,
        hidden_size: hidden,
        num_heads: heads,
        num_kv_heads: kv_heads,
        num_layers: layers,
        seq_len: capacity
      )

    before = Keyword.get(opts, :input_transform, fn _params, frames -> frames end)
    after_fn = Keyword.get(opts, :output_transform, fn _params, output -> output end)

    compiled =
      Nx.Defn.jit(
        fn params, frames, state ->
          {output, state} = forward(params, before.(params, frames), state, config)
          {after_fn.(params, output), state}
        end,
        Keyword.take(opts, [:compiler, :client])
      )

    init = fn batch ->
      buffers =
        for i <- 1..layers, into: %{} do
          zero = Nx.broadcast(Nx.tensor(0.0), {batch, kv_heads, capacity, div(hidden, heads)})
          {i, {zero, zero}}
        end

      %{layers: buffers, position: Nx.tensor(0, type: :s64)}
    end

    predict = fn params, frames, state ->
      if Nx.to_number(state.position) + Nx.axis_size(frames, 1) > capacity,
        do:
          raise(
            ArgumentError,
            "KV cache capacity exceeded; reset and prefill the retained context"
          )

      compiled.(Ops.unwrap_params(params), frames, state)
    end

    %{init: init, predict: predict, max_seq_len: capacity}
  end

  defp forward(params, frames, state, opts) do
    hidden = opts[:hidden_size]
    heads = opts[:num_heads]
    kv_heads = opts[:num_kv_heads]
    dim = div(hidden, heads)
    {batch, count, _} = Nx.shape(frames)
    capacity = opts[:seq_len]

    x =
      if Map.has_key?(params, "input_projection"),
        do: Ops.dense(frames, params["input_projection"]),
        else: frames

    {x, buffers} =
      Enum.reduce(1..opts[:num_layers], {x, state.layers}, fn i, {x, buffers} ->
        name = "decoder_block_#{i}"
        norm = rms(x, params, name <> "_attn_norm")

        project = fn suffix, n ->
          Ops.dense(norm, Map.fetch!(params, name <> "_attn_" <> suffix <> "_proj"))
          |> Nx.reshape({batch, count, n, dim})
          |> Nx.transpose(axes: [0, 2, 1, 3])
        end

        q = project.("q", heads)
        k = project.("k", kv_heads)
        v = project.("v", kv_heads)

        rope? =
          if Keyword.get(opts, :interleave_rope, false),
            do: rem(i, 2) == 0,
            else: Keyword.get(opts, :use_rope, true)

        {q, k} =
          if rope? do
            {qr, kr} =
              RoPE.apply_rotary_offset(
                Nx.reshape(q, {batch * heads, count, dim}),
                Nx.reshape(k, {batch * kv_heads, count, dim}),
                state.position
              )

            {Nx.reshape(qr, Nx.shape(q)), Nx.reshape(kr, Nx.shape(k))}
          else
            {q, k}
          end

        {kb, vb} = Map.fetch!(buffers, i)
        kb = Nx.put_slice(kb, [0, 0, state.position, 0], k)
        vb = Nx.put_slice(vb, [0, 0, state.position, 0], v)

        repeat = fn t ->
          t
          |> Nx.reshape({batch, kv_heads, 1, capacity, dim})
          |> Nx.broadcast({batch, kv_heads, div(heads, kv_heads), capacity, dim})
          |> Nx.reshape({batch, heads, capacity, dim})
        end

        scores = Nx.dot(q, [3], [0, 1], repeat.(kb), [3], [0, 1]) |> Nx.divide(:math.sqrt(dim))

        mask =
          Nx.less_equal(
            Nx.iota({1, 1, count, capacity}, axis: 3),
            Nx.add(Nx.iota({1, 1, count, capacity}, axis: 2), state.position)
          )

        weights =
          Nx.select(Nx.broadcast(mask, Nx.shape(scores)), scores, -1.0e9)
          |> Axon.Activations.softmax(axis: -1)

        attention =
          Nx.dot(weights, [3], [0, 1], repeat.(vb), [2], [0, 1])
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, count, hidden})
          |> Ops.dense(Map.fetch!(params, name <> "_attn_out_proj"))

        x = Nx.add(x, attention)
        norm = rms(x, params, name <> "_ffn_norm")
        gate = Ops.dense(norm, params[name <> "_ffn_gate"]) |> Ops.silu()
        up = Ops.dense(norm, params[name <> "_ffn_up"])
        x = Nx.add(x, Ops.dense(Nx.multiply(gate, up), params[name <> "_ffn_down"]))
        {x, Map.put(buffers, i, {kb, vb})}
      end)

    x =
      if Keyword.get(opts, :final_norm, true),
        do: Ops.layer_norm(x, params["final_norm"]),
        else: x

    {x, %{layers: buffers, position: Nx.add(state.position, count)}}
  end

  defp rms(x, params, name),
    do: RMSNorm.apply(x, Map.fetch!(Map.fetch!(params, name), name <> "_gamma"))
end
