defmodule Edifice.Attention.DiffTransformerTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.DiffTransformer

  @batch 4
  @seq_len 12
  @embed_dim 64
  @hidden_size 32
  @num_heads 4
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    window_size: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = DiffTransformer.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = DiffTransformer.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = DiffTransformer.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      refute output |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
      refute output |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 1
    end

    test "single layer variant" do
      opts = Keyword.put(@opts, :num_layers, 1)
      model = DiffTransformer.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "different number of heads" do
      opts = Keyword.merge(@opts, num_heads: 2, hidden_size: 32)
      model = DiffTransformer.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, 32}
    end

    test "lambda parameters exist in model params" do
      model = DiffTransformer.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      # Flatten all parameter keys from the nested ModelState
      all_keys =
        params
        |> Map.get(:data)
        |> Enum.flat_map(fn {top_key, inner} ->
          case inner do
            %{} -> Enum.map(Map.keys(inner), &"#{top_key}/#{&1}")
            _ -> [to_string(top_key)]
          end
        end)

      # Lambda parameters should be present for each block
      # Implementation uses a single combined lambda param per block
      assert Enum.any?(all_keys, &String.contains?(&1, "lambda"))
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DiffTransformer.output_size(@opts) == @hidden_size
    end

    test "returns default when no opts" do
      assert DiffTransformer.output_size([]) == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns a keyword list" do
      defaults = DiffTransformer.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_layers)
    end
  end
end
