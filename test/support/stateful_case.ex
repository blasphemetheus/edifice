defmodule Edifice.StatefulCase do
  @moduledoc """
  Shared harness for `Edifice.Stateful` correctness pins.

  Two properties matter (see HANDOFF_BOT_RUNTIME.md task 2):

    1. **Step/forward equivalence** — iterating `step/3` over a sequence
       reproduces the full-sequence forward outputs. Verified at EVERY
       prefix length k (not just the final step) by running the forward on
       `x[:, 0..k-1, :]`: O(N²) but cheap at test sizes, and it needs no
       per-timestep output plumbing in the builders.
    2. **Rollback** — snapshot at frame k, continue to N, restore, replay
       k..N with the same inputs → bitwise-identical outputs. This is the
       netplay rollback primitive.
  """

  import ExUnit.Assertions

  @doc """
  Assert iterated `step/3` matches the full-sequence forward at every prefix.

  Options:
    * `:seq_len` - frames to test (default 8)
    * `:batch` - batch size (default 1)
    * `:atol` - absolute tolerance (default 1.0e-5)
    * `:seed` - RNG seed (default 42)
  """
  def assert_step_matches_forward(arch, build_opts, opts \\ []) do
    seq_len = Keyword.get(opts, :seq_len, 8)
    batch = Keyword.get(opts, :batch, 1)
    atol = Keyword.get(opts, :atol, 1.0e-5)
    seed = Keyword.get(opts, :seed, 42)
    step_opts = Keyword.get(opts, :step_opts, [])

    {params, predict_fn, x} = build_forward(arch, build_opts, seq_len, batch, seed)

    state =
      Edifice.init_state(
        arch,
        params,
        build_opts ++ [batch_size: batch] ++ Keyword.take(step_opts, [:compiler])
      )

    Enum.reduce(1..seq_len, state, fn k, state ->
      frame = x |> Nx.slice_along_axis(k - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      {step_out, state} = Edifice.step(arch, params, state, frame, step_opts)

      prefix = Nx.slice_along_axis(x, 0, k, axis: 1)
      full_out = predict_fn.(params, prefix)

      diff =
        step_out
        |> Nx.subtract(full_out)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      assert diff <= atol,
             "#{inspect(arch)} step output diverged from full forward at " <>
               "prefix k=#{k}/#{seq_len}: max |diff| = #{diff} > atol #{atol} " <>
               "(build_opts: #{inspect(build_opts)})"

      state
    end)

    :ok
  end

  @doc """
  Assert snapshot/restore rollback determinism.

  Runs `seq_len` steps, snapshots (and serialize/deserialize round-trips)
  the state at `rollback_at`, then replays frames `rollback_at+1..seq_len`
  from the restored state and asserts bitwise-identical outputs.
  """
  def assert_rollback_deterministic(arch, build_opts, opts \\ []) do
    seq_len = Keyword.get(opts, :seq_len, 12)
    rollback_at = Keyword.get(opts, :rollback_at, 5)
    batch = Keyword.get(opts, :batch, 1)
    seed = Keyword.get(opts, :seed, 7)
    step_opts = Keyword.get(opts, :step_opts, [])

    {params, _predict_fn, x} = build_forward(arch, build_opts, seq_len, batch, seed)

    frames =
      for t <- 0..(seq_len - 1) do
        x |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end

    state0 = Edifice.init_state(arch, params, build_opts ++ [batch_size: batch])

    # First pass: run to the end, snapshotting at rollback_at
    {outputs, snapshot} =
      frames
      |> Enum.with_index(1)
      |> Enum.reduce({[], nil}, fn {frame, t}, {outs, snap} ->
        state = if outs == [], do: state0, else: elem(hd(outs), 1)
        {out, new_state} = Edifice.step(arch, params, state, frame, step_opts)

        snap =
          if t == rollback_at do
            # The netplay wire-format guarantee: snapshot survives
            # serialize/deserialize and continues identically
            Edifice.Stateful.serialize(new_state)
          else
            snap
          end

        {[{out, new_state} | outs], snap}
      end)

    outputs = Enum.reverse(outputs)
    tail_outputs = outputs |> Enum.drop(rollback_at) |> Enum.map(&elem(&1, 0))

    # Rollback: restore at k, replay k+1..N with identical inputs
    restored = Edifice.Stateful.deserialize(snapshot)

    {replayed, _state} =
      frames
      |> Enum.drop(rollback_at)
      |> Enum.reduce({[], restored}, fn frame, {outs, state} ->
        {out, new_state} = Edifice.step(arch, params, state, frame, step_opts)
        {[out | outs], new_state}
      end)

    replayed = Enum.reverse(replayed)

    Enum.zip(tail_outputs, replayed)
    |> Enum.with_index(rollback_at + 1)
    |> Enum.each(fn {{original, replay}, t} ->
      assert Nx.to_binary(original) == Nx.to_binary(replay),
             "#{inspect(arch)} rollback replay diverged at frame #{t} " <>
               "(snapshot at #{rollback_at}); rollback must be bitwise-deterministic"
    end)

    :ok
  end

  @doc """
  Build a model with a dynamic sequence axis, init params, and return
  `{params, predict_fn, input [batch, seq_len, embed]}`.
  """
  def build_forward(arch, build_opts, seq_len, batch, seed) do
    embed_dim = Keyword.fetch!(build_opts, :embed_dim)

    # seq_len: nil -> dynamic sequence axis, so one build serves every
    # prefix length
    model = Edifice.build(arch, Keyword.put(build_opts, :seq_len, nil))
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    template = %{"state_sequence" => Nx.template({batch, seq_len, embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    key = Nx.Random.key(seed)
    {x, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, embed_dim})

    {params, predict_fn, x}
  end
end
