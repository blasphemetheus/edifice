defmodule Edifice.Recurrent.TransformerLikeTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.TransformerLike

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 64

  defp build_and_run(opts) do
    model = TransformerLike.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  defp base_opts do
    [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_layers: 2,
      seq_len: @seq_len,
      dropout: 0.0
    ]
  end

  describe "build/1 with defaults" do
    test "returns an Axon model" do
      model = TransformerLike.build(base_opts())
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with GRU cell" do
    test "produces correct output shape" do
      opts = Keyword.put(base_opts(), :cell_type, :gru)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      opts = Keyword.put(base_opts(), :cell_type, :gru)
      {_model, output} = build_and_run(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with custom options" do
    test "different num_layers" do
      opts = Keyword.put(base_opts(), :num_layers, 1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "different ffn_multiplier" do
      opts = Keyword.put(base_opts(), :ffn_multiplier, 4)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "embed_dim == hidden_size skips projection" do
      opts = Keyword.merge(base_opts(), embed_dim: @hidden_size, hidden_size: @hidden_size)

      model = TransformerLike.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})

      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 with recurrent_norm" do
    test "applies pre-norm before recurrence" do
      opts = Keyword.put(base_opts(), :recurrent_norm, true)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite with recurrent_norm" do
      opts = Keyword.put(base_opts(), :recurrent_norm, true)
      {_model, output} = build_and_run(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with dropout" do
    test "produces correct shape in inference mode" do
      opts = Keyword.put(base_opts(), :dropout, 0.1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite with dropout" do
      opts = Keyword.put(base_opts(), :dropout, 0.1)
      {_model, output} = build_and_run(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert TransformerLike.output_size(hidden_size: 256) == 256
    end

    test "returns default when no option given" do
      assert TransformerLike.output_size([]) == 512
    end
  end

  describe "registry integration" do
    test "Edifice.build(:transformer_like, ...) works" do
      model =
        Edifice.build(:transformer_like,
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end
  end
end
