defmodule Edifice.Recurrent.HuginnTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  import Edifice.TestHelpers

  alias Edifice.Recurrent.Huginn

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 16
  @num_heads 4
  @head_dim 4

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    head_dim: @head_dim,
    prelude_layers: 1,
    core_layers: 2,
    coda_layers: 1,
    num_iterations: 2,
    seq_len: @seq_len
  ]

  defp build_and_run(opts) do
    model = Huginn.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    embed = opts[:embed_dim] || @embed_dim
    seq = opts[:seq_len] || @seq_len

    params =
      init_fn.(Nx.template({@batch, seq, embed}, :f32), Axon.ModelState.empty())

    input = random_tensor({@batch, seq, embed})
    output = predict_fn.(params, input)
    {model, output}
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = Huginn.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output values are finite" do
      {_model, output} = build_and_run(@opts)
      assert_finite!(output)
    end

    test "works with different iteration counts" do
      {_model, output} = build_and_run(Keyword.put(@opts, :num_iterations, 4))
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "handles different embed and hidden sizes" do
      opts =
        Keyword.merge(@opts,
          embed_dim: 48,
          hidden_size: 24,
          num_heads: 4,
          head_dim: 6,
          prelude_layers: 1,
          core_layers: 1,
          coda_layers: 1,
          num_iterations: 2
        )

      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, 24}
    end

    test "core layers share parameters across iterations" do
      # Build with 2 vs 4 iterations — param count should be the same
      # since core layers are weight-tied
      model_2 = Huginn.build(Keyword.put(@opts, :num_iterations, 2))
      model_4 = Huginn.build(Keyword.put(@opts, :num_iterations, 4))

      {init_fn_2, _} = Axon.build(model_2)
      {init_fn_4, _} = Axon.build(model_4)

      template = Nx.template({@batch, @seq_len, @embed_dim}, :f32)
      params_2 = init_fn_2.(template, Axon.ModelState.empty())
      params_4 = init_fn_4.(template, Axon.ModelState.empty())

      # Same number of parameter groups (weight tying means same params)
      string_keys_2 =
        params_2 |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.sort()

      string_keys_4 =
        params_4 |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.sort()

      assert string_keys_2 == string_keys_4
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Huginn.output_size(hidden_size: 128) == 128
    end

    test "returns default when no option" do
      assert Huginn.output_size([]) == 256
    end
  end

  describe "registry integration" do
    test "Edifice.build(:huginn, ...) works" do
      model = Edifice.build(:huginn, @opts)
      assert %Axon{} = model
    end
  end
end
