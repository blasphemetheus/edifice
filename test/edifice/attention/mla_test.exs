defmodule Edifice.Attention.MLATest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.MLA

  @batch 4
  @seq_len 12
  @embed_dim 64
  @hidden_size 32
  @num_heads 4
  @head_dim 8
  @kv_latent_dim 8
  @q_latent_dim 24
  @rope_dim 8
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    head_dim: @head_dim,
    kv_latent_dim: @kv_latent_dim,
    q_latent_dim: @q_latent_dim,
    rope_dim: @rope_dim,
    num_layers: @num_layers,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = MLA.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MLA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = MLA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    @tag timeout: 120_000
    test "works with default options" do
      model =
        MLA.build(
          embed_dim: @embed_dim,
          hidden_size: 64,
          num_heads: 2,
          head_dim: 16,
          kv_latent_dim: 16,
          q_latent_dim: 48,
          rope_dim: 8,
          num_layers: 1,
          seq_len: 8,
          dropout: 0.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {input, _} = Nx.Random.uniform(key, shape: {2, 8, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 64}
    end

    test "single layer model works" do
      model = MLA.build(Keyword.put(@opts, :num_layers, 1))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build_mla_block/2" do
    test "builds a single MLA block" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      block =
        MLA.build_mla_block(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          kv_latent_dim: @kv_latent_dim,
          q_latent_dim: @q_latent_dim,
          rope_dim: @rope_dim,
          dropout: 0.0,
          name: "test_block"
        )

      assert %Axon{} = block

      {init_fn, predict_fn} = Axon.build(block)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(123)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Block preserves sequence shape
      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MLA.output_size(@opts) == @hidden_size
    end

    test "returns default when no opts" do
      assert MLA.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MLA.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :head_dim)
      assert Keyword.has_key?(defaults, :kv_latent_dim)
      assert Keyword.has_key?(defaults, :rope_dim)
    end
  end
end
