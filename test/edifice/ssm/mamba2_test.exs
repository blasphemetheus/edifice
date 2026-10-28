defmodule Edifice.SSM.Mamba2Test do
  use ExUnit.Case, async: false
  import Nx.Testing
  alias Edifice.SSM.Mamba2

  setup_all do
    %{
      fixture:
        Safetensors.read!(
          Path.expand("../../fixtures/numerical/mamba2_reference.safetensors", __DIR__)
        )
    }
  end

  defp arguments(f, name) do
    keys =
      ~w(x h0 conv0 in_proj.weight conv1d.weight out_proj.weight A_log dt_bias D norm.weight in_proj.bias conv1d.bias out_proj.bias)

    Map.new(Enum.filter(keys, &Map.has_key?(f, "#{name}.#{&1}")), fn key ->
      {key, Nx.as_type(f["#{name}.#{key}"], :f32)}
    end)
  end

  defp run(p) do
    Mamba2.forward(Mamba2.from_reference(p), p["x"], %{h: p["h0"], conv: p["conv0"]})
  end

  for name <- ~w(plain grouped short pointwise) do
    test "reference output, carried state and every gradient: #{name}", %{fixture: f} do
      name = unquote(name)
      p = arguments(f, name)
      {y, state} = Nx.Defn.jit(&run/1).(p)
      assert_all_close(y, f["#{name}.output"], atol: 2.0e-6, rtol: 2.0e-5)
      assert_all_close(state.h, f["#{name}.h"], atol: 2.0e-6, rtol: 2.0e-5)
      assert_all_close(state.conv, f["#{name}.conv"], atol: 2.0e-6, rtol: 2.0e-5)
      weights = Nx.as_type(f["#{name}.loss_weights"], :f32)

      grads =
        Nx.Defn.jit(fn p, weights ->
          Nx.Defn.grad(p, fn p ->
            {y, _} = run(p)
            Nx.sum(Nx.multiply(y, weights))
          end)
        end).(p, weights)

      for {key, grad} <- grads do
        if name == "pointwise" and key == "conv0" do
          assert_all_close(grad, Nx.broadcast(0.0, Nx.shape(grad)))
        else
          assert_all_close(grad, f["#{name}.grad.#{key}"], atol: 1.0e-5, rtol: 2.0e-4)
        end
      end

      step = Nx.Defn.jit(&Mamba2.block_step/3)

      {ys, carried} =
        Enum.map_reduce(0..(Nx.axis_size(p["x"], 1) - 1), %{h: p["h0"], conv: p["conv0"]}, fn t,
                                                                                              s ->
          step.(Mamba2.from_reference(p), s, Nx.take(p["x"], t, axis: 1))
        end)

      assert_all_close(Nx.stack(ys, axis: 1), y, atol: 2.0e-6, rtol: 2.0e-5)
      assert_all_close(carried.h, state.h, atol: 2.0e-6, rtol: 2.0e-5)
      assert_all_close(carried.conv, state.conv)
    end
  end

  @tag :exla_only
  test "compiled host output and learned decay gradients", %{fixture: f} do
    p = arguments(f, "grouped")
    {y, state} = Nx.Defn.jit(&run/1, compiler: EXLA, client: :host).(p)
    assert_all_close(y, f["grouped.output"], atol: 2.0e-6, rtol: 2.0e-5)
    assert_all_close(state.h, f["grouped.h"], atol: 2.0e-6, rtol: 2.0e-5)
    weights = Nx.as_type(f["grouped.loss_weights"], :f32)

    grads =
      Nx.Defn.jit(
        fn p, weights ->
          Nx.Defn.grad(p, fn p ->
            {y, _} = run(p)
            Nx.sum(Nx.multiply(y, weights))
          end)
        end,
        compiler: EXLA,
        client: :host
      ).(p, weights)

    for {key, grad} <- grads do
      assert_all_close(grad, f["grouped.grad.#{key}"], atol: 1.0e-5, rtol: 2.0e-4)
    end
  end

  test "standalone Axon mixer accepts converted reference weights", %{fixture: f} do
    p = arguments(f, "plain")

    model =
      Mamba2.build_block(Axon.input("x", shape: {nil, nil, 4}),
        hidden_size: 4,
        head_dim: 2,
        state_size: 3,
        conv_bias: false
      )

    {init, predict} = Axon.build(model)
    params = init.(p["x"], Axon.ModelState.empty())
    params = %{params | data: Map.put(params.data, "mamba2", Mamba2.from_reference(p))}
    assert_all_close(predict.(params, p["x"]), f["plain.output"], atol: 2.0e-6, rtol: 2.0e-5)
  end

  test "two-layer backbone agrees with carried stepping and split replay" do
    opts = [
      embed_dim: 3,
      hidden_size: 4,
      head_dim: 2,
      state_size: 3,
      num_groups: 2,
      num_layers: 2,
      output_mode: :all
    ]

    model = Mamba2.build(opts)
    {init, predict} = Axon.build(model)
    x = Nx.sin(Nx.iota({2, 7, 3}))
    params = init.(x, Axon.ModelState.empty())
    expected = predict.(params, x)
    initial = Mamba2.init_state(params, opts ++ [batch_size: 2])
    step = Nx.Defn.jit(&Mamba2.step/3)

    {ys, state} =
      Enum.map_reduce(0..6, initial, fn t, s -> step.(params, s, Nx.take(x, t, axis: 1)) end)

    assert_all_close(Nx.stack(ys, axis: 1), expected, atol: 1.0e-5, rtol: 1.0e-4)
    assert Nx.shape(state.h) == {2, 2, 4, 2, 3}
    # Restore a snapshot and replay the suffix; the next output must be identical.
    {_, snapshot} =
      Enum.map_reduce(0..2, initial, fn t, s -> step.(params, s, Nx.take(x, t, axis: 1)) end)

    snapshot = snapshot |> Nx.serialize() |> Nx.deserialize()

    {suffix, restored} =
      Enum.map_reduce(3..6, snapshot, fn t, s -> step.(params, s, Nx.take(x, t, axis: 1)) end)

    assert_all_close(Nx.stack(suffix, axis: 1), Nx.slice_along_axis(expected, 3, 4, axis: 1),
      atol: 1.0e-5,
      rtol: 1.0e-4
    )

    assert_all_close(restored.h, state.h)
    assert_all_close(restored.conv, state.conv)
  end

  test "unsupported variants fail explicitly" do
    base = [embed_dim: 4, hidden_size: 4, head_dim: 2]

    for override <- [
          [head_dim: 3],
          [num_groups: 3],
          [state_size: 0],
          [d_ssm: 4],
          [rmsnorm: false],
          [norm_before_gate: true],
          [d_has_hdim: true],
          [dt_limit: {0.0, 0.1}],
          [dropout: 0.1]
        ] do
      assert_raise ArgumentError, fn -> Mamba2.build(Keyword.merge(base, override)) end
    end
  end

  @tag :exla_only
  test "registry dispatch and compiled chunk replay preserve all state" do
    opts = [embed_dim: 3, hidden_size: 4, head_dim: 2, state_size: 3, num_layers: 2]
    assert Edifice.module_for(:mamba2) == Mamba2
    assert Edifice.module_for(:mamba_ssd) == Edifice.SSM.MambaSSD
    {init, _} = Axon.build(Edifice.build(:mamba2, opts))
    x = Nx.sin(Nx.iota({2, 7, 3}))
    params = init.(x, Axon.ModelState.empty())
    initial = Edifice.init_state(:mamba2, params, opts ++ [batch_size: 2])
    step = Edifice.Stateful.jit_step(Mamba2, EXLA, client: :host)
    replay = Edifice.Stateful.jit_replay(Mamba2, EXLA, client: :host)
    {_, carry} = step.(params, initial, Nx.take(x, 0, axis: 1))

    expected =
      Enum.reduce(0..6, carry, fn t, s -> elem(step.(params, s, Nx.take(x, t, axis: 1)), 1) end)

    actual = replay.(params, carry, x)
    split = replay.(params, carry, Nx.slice_along_axis(x, 0, 2, axis: 1))
    split = replay.(params, split, Nx.slice_along_axis(x, 2, 5, axis: 1))

    for key <- [:h, :conv] do
      assert_all_close(actual[key], expected[key], atol: 1.0e-5, rtol: 1.0e-4)
      assert_all_close(split[key], expected[key], atol: 1.0e-5, rtol: 1.0e-4)
    end
  end
end
