defmodule Edifice.Pretrained.RecurrentReferenceTest do
  use ExUnit.Case, async: false
  import Nx.Testing
  alias Edifice.SSM.Common
  alias Edifice.CUDA.FusedScan

  # AI-generated. Reference revisions and limits: docs/recurrent_reference_validation.md.
  setup_all do
    fixture =
      Safetensors.read!(
        Path.expand("../../fixtures/numerical/recurrent_reference.safetensors", __DIR__)
      )

    %{fixture: fixture}
  end

  setup do
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  defp inputs(f, name, keys) do
    Map.new(keys, fn key -> {key, Nx.as_type(f["#{name}.#{key}"], :f32)} end)
  end

  defp validate(f, name, inputs, predict, opts \\ []) do
    run = Nx.Defn.jit(predict, opts)
    {output, state} = run.(inputs)
    assert_all_close(output, f["#{name}.output"], atol: 1.0e-5, rtol: 1.0e-4)
    assert_all_close(state, f["#{name}.state"], atol: 1.0e-5, rtol: 1.0e-4)
    weights = Nx.as_type(f["#{name}.loss_weights"], :f32)

    gradient =
      Nx.Defn.jit(
        fn args, weights ->
          Nx.Defn.grad(args, fn p ->
            {y, _} = predict.(p)
            Nx.sum(Nx.multiply(y, weights))
          end)
        end,
        opts
      ).(inputs, weights)

    for {key, actual} <- gradient do
      assert_all_close(actual, f["#{name}.grad.#{key}"], atol: 2.0e-5, rtol: 1.0e-4)
    end
  end

  defp gru_params(p) do
    hidden = Nx.axis_size(p["wh"], 1)
    split = fn tensor, i -> Nx.slice_along_axis(tensor, i * hidden, hidden, axis: 0) end

    %{
      "gru_1" => %{
        "input_kernel" =>
          Map.new(Enum.with_index(["wir", "wiz", "win"]), fn {key, i} ->
            {key, Nx.transpose(split.(p["wi"], i))}
          end),
        "hidden_kernel" =>
          Map.new(Enum.with_index(["whr", "whz", "whn"]), fn {key, i} ->
            {key, Nx.transpose(split.(p["wh"], i))}
          end),
        "bias" => %{
          "br" => Nx.add(split.(p["bi"], 0), split.(p["bh"], 0)),
          "bz" => Nx.add(split.(p["bi"], 1), split.(p["bh"], 1)),
          "bin" => split.(p["bi"], 2),
          "bhn" => split.(p["bh"], 2)
        }
      }
    }
  end

  defp gru_steps(p) do
    params = gru_params(p)
    initial = %{h: Nx.transpose(p["h0"], axes: [1, 0, 2])}

    {ys, state} =
      Enum.map_reduce(0..(Nx.axis_size(p["x"], 1) - 1), initial, fn t, s ->
        Edifice.Recurrent.step(params, s, Nx.take(p["x"], t, axis: 1))
      end)

    {Nx.stack(ys, axis: 1), Nx.transpose(state.h, axes: [1, 0, 2])}
  end

  for name <- ["gru_short", "gru"] do
    test "GRU carried outputs and all input/parameter gradients against ATen: #{name}", %{
      fixture: f
    } do
      name = unquote(name)
      p = inputs(f, name, ~w(x h0 wi wh bi bh))
      validate(f, name, p, &gru_steps/1)
      hidden = Nx.axis_size(p["wh"], 1)
      input = Axon.input("x", shape: {nil, nil, 3})

      model =
        Edifice.Recurrent.build_backbone_with_carry(input,
          hidden_size: hidden,
          num_layers: 1,
          input_layer_norm: false,
          use_layer_norm: false,
          dropout: 0.0
        )

      {init, predict} = Axon.build(model)

      template =
        init.(
          %{"x" => p["x"], "initial_hidden" => Nx.transpose(p["h0"], axes: [1, 0, 2])},
          Axon.ModelState.empty()
        )

      validate(f, name, p, fn p ->
        result =
          predict.(%{template | data: gru_params(p)}, %{
            "x" => p["x"],
            "initial_hidden" => Nx.transpose(p["h0"], axes: [1, 0, 2])
          })

        {result.output, Nx.transpose(result.hidden, axes: [1, 0, 2])}
      end)
    end
  end

  test "GRU scan accepts the reference's nonzero recurrent candidate bias", %{fixture: f} do
    p = inputs(f, "gru", ~w(x h0 wi wh bi bh))

    validate(f, "gru", p, fn p ->
      hidden = Nx.axis_size(p["wh"], 1)
      # r/z recurrent biases can be merged into the input bias; candidate bias cannot.
      merged =
        Nx.concatenate([Nx.slice_along_axis(p["bh"], 0, 2 * hidden), Nx.broadcast(0.0, {hidden})])

      wx = Nx.add(Nx.dot(p["x"], Nx.transpose(p["wi"])), Nx.add(p["bi"], merged))

      y =
        FusedScan.gru_scan(
          wx,
          Nx.transpose(p["wh"]),
          Nx.squeeze(p["h0"], axes: [0]),
          Nx.slice_along_axis(p["bh"], 2 * hidden, hidden)
        )

      {y, Nx.new_axis(Nx.take(y, Nx.axis_size(y, 1) - 1, axis: 1), 0)}
    end)
  end

  defp selective(p) do
    y = FusedScan.selective_scan(p["x"], p["dt"], p["a"], p["b"], p["c"])
    # This API returns output only; use it as the state slot too for validation.
    {y, Nx.take(y, Nx.axis_size(y, 1) - 1, axis: 1)}
  end

  test "selective scan honors arbitrary A and its gradients", %{fixture: f} do
    p = inputs(f, "mamba", ~w(x dt a b c))
    f = Map.put(f, "mamba.state", Nx.take(f["mamba.output"], 8, axis: 1))
    validate(f, "mamba", p, &selective/1)
  end

  test "Mamba carried state matches reference from a nonzero initial state", %{fixture: f} do
    p = inputs(f, "mamba_fixed", ~w(x dt b c h0))

    validate(f, "mamba_fixed", p, fn p ->
      {ys, state} =
        Enum.map_reduce(0..8, p["h0"], fn t, h ->
          Common.step_discretized_ssm(
            Nx.take(p["x"], t, axis: 1),
            Nx.take(p["b"], t, axis: 1),
            Nx.take(p["c"], t, axis: 1),
            Nx.take(p["dt"], t, axis: 1),
            h
          )
        end)

      {Nx.stack(ys, axis: 1), state}
    end)
  end

  defp ssd_predict(chunk, training) do
    # Exercise the actual production scan node without duplicating its algorithm.
    model =
      Edifice.SSM.MambaSSD.build(
        embed_dim: 3,
        hidden_size: 3,
        state_size: 2,
        num_layers: 1,
        chunk_size: chunk,
        training_mode: training
      )

    node = model.nodes |> Map.values() |> Enum.find(&(&1.op_name == :ssd_scan))

    fn p ->
      y = node.op.(p["x"], p["b"], p["dt"], node.opts)
      {y, Nx.take(y, Nx.axis_size(y, 1) - 1, axis: 1)}
    end
  end

  test "Mamba parallel scan matches reference on a non-power-of-two long sequence", %{fixture: f} do
    validate(f, "mamba_long", inputs(f, "mamba_long", ~w(x dt b c)), fn p ->
      {a, bx} = Common.discretize_ssm(p["x"], p["b"], p["dt"], 3)
      h = Common.blelloch_scan(a, bx)
      {Common.compute_ssm_output(h, p["c"]), Nx.take(h, 36, axis: 1)}
    end)
  end

  for chunk <- [3, 5, 16], training <- [false, true] do
    test "SSD scan matches upstream chunk decomposition: chunk #{chunk}, training #{training}", %{
      fixture: f
    } do
      validate(
        f,
        "ssd",
        inputs(f, "ssd", ~w(x dt b)),
        ssd_predict(unquote(chunk), unquote(training))
      )
    end
  end

  test "signed MinGRU matches closed-form recurrence and differs from positive-candidate variant",
       %{fixture: f} do
    p = inputs(f, "mingru", ~w(gates candidates))

    for scan <- [&Edifice.Recurrent.MinGRU.min_gru_scan/2, &FusedScan.mingru/2] do
      validate(f, "mingru", p, fn p ->
        y = scan.(p["gates"], p["candidates"])
        {y, Nx.take(y, 8, axis: 1)}
      end)
    end

    delta =
      Nx.subtract(f["mingru.output"], f["mingru.positive_output"])
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert delta > 0.5

    positive =
      Nx.select(
        Nx.greater_equal(p["candidates"], 0),
        Nx.add(p["candidates"], 0.5),
        Nx.sigmoid(p["candidates"])
      )

    assert_all_close(
      Edifice.Recurrent.MinGRU.min_gru_scan(p["gates"], positive),
      f["mingru.positive_output"],
      atol: 1.0e-5,
      rtol: 1.0e-4
    )
  end

  @tag :exla_only
  test "compiled reference checks retain outputs and gradients", %{fixture: f} do
    opts = [compiler: EXLA, client: EXLA.Client.default_name(), precision: :highest]
    validate(f, "gru", inputs(f, "gru", ~w(x h0 wi wh bi bh)), &gru_steps/1, opts)
    f = Map.put(f, "mamba.state", Nx.take(f["mamba.output"], 8, axis: 1))
    validate(f, "mamba", inputs(f, "mamba", ~w(x dt a b c)), &selective/1, opts)
    validate(f, "ssd", inputs(f, "ssd", ~w(x dt b)), ssd_predict(5, true), opts)

    validate(
      f,
      "mingru",
      inputs(f, "mingru", ~w(gates candidates)),
      fn p ->
        y = FusedScan.mingru(p["gates"], p["candidates"])
        {y, Nx.take(y, 8, axis: 1)}
      end,
      opts
    )
  end
end
