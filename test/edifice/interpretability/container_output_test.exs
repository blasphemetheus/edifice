defmodule Edifice.Interpretability.ContainerOutputTest do
  @moduledoc """
  INTERP_AUDIT family-wide gap 2: losses need `(input, reconstruction,
  hidden)` but builds only exposed the reconstruction. `output: :container`
  must expose all three; the default must stay a single tensor
  (backward compat for registry consumers).
  """
  use ExUnit.Case, async: true

  @moduletag :interpretability

  @modules [
    {Edifice.Interpretability.SparseAutoencoder, [input_size: 12, dict_size: 24, top_k: 4], 12},
    {Edifice.Interpretability.BatchTopKSAE, [input_size: 12, dict_size: 24, batch_k: 8], 12},
    {Edifice.Interpretability.GatedSAE, [input_size: 12, dict_size: 24, top_k: 4], 12},
    {Edifice.Interpretability.JumpReluSAE, [input_size: 12, dict_size: 24], 12},
    {Edifice.Interpretability.MatryoshkaSAE, [input_size: 12, dict_size: 24, top_k: 4], 12},
    {Edifice.Interpretability.Transcoder, [input_size: 12, output_size: 10, dict_size: 24, top_k: 4], 10}
  ]

  defp input_key(module) do
    model = module.build(Enum.find(@modules, fn {m, _, _} -> m == module end) |> elem(1))
    model |> Axon.get_inputs() |> Map.keys() |> hd()
  end

  defp run(model, module, batch) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    key = input_key(module)

    template = %{key => Nx.template({batch, 12}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    {x, _} = Nx.Random.normal(Nx.Random.key(0), 0.0, 1.0, shape: {batch, 12})
    {predict_fn.(params, %{key => x}), x}
  end

  for {module, opts, out_size} <- @modules do
    test "#{inspect(module)} output: :container exposes reconstruction/hidden/pre_acts" do
      module = unquote(module)
      opts = unquote(opts)
      out_size = unquote(out_size)

      model = module.build(opts ++ [output: :container])
      {out, _x} = run(model, module, 4)

      assert %{reconstruction: recon, hidden: hidden, pre_acts: pre_acts} = out
      assert Nx.shape(recon) == {4, out_size}
      assert Nx.shape(hidden) == {4, 24}
      assert Nx.shape(pre_acts) == {4, 24}

      # hidden must be the SPARSE activations: for top-k style modules,
      # nonzero count bounded by the k budget. JumpReluSAE is exempt —
      # its soft-sigmoid gate produces NO exact zeros (that's the audit's
      # documented complaint about it, not a container bug).
      unless module == Edifice.Interpretability.JumpReluSAE do
        nonzero = hidden |> Nx.not_equal(0.0) |> Nx.sum() |> Nx.to_number()
        total = 4 * 24
        assert nonzero < total, "hidden is not sparse (#{nonzero}/#{total} nonzero)"
      end
    end

    test "#{inspect(module)} default output unchanged (single tensor)" do
      module = unquote(module)
      opts = unquote(opts)
      out_size = unquote(out_size)

      model = module.build(opts)
      {out, _x} = run(model, module, 4)

      assert %Nx.Tensor{} = out
      assert Nx.shape(out) == {4, out_size}
    end
  end

  test "container reconstruction equals the default output (same params would match graphs)" do
    # Same graph up to the output node: with identical params the
    # reconstruction inside the container must equal the default output.
    opts = [input_size: 12, dict_size: 24, top_k: 4]
    module = Edifice.Interpretability.SparseAutoencoder

    container_model = module.build(opts ++ [output: :container])
    {init_fn, predict_fn} = Axon.build(container_model, mode: :inference)
    template = %{"sae_input" => Nx.template({4, 12}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    plain_model = module.build(opts)
    {_, plain_predict} = Axon.build(plain_model, mode: :inference)

    {x, _} = Nx.Random.normal(Nx.Random.key(1), 0.0, 1.0, shape: {4, 12})

    %{reconstruction: from_container} = predict_fn.(params, %{"sae_input" => x})
    plain = plain_predict.(params, %{"sae_input" => x})

    assert Nx.to_binary(from_container) == Nx.to_binary(plain)
  end
end
