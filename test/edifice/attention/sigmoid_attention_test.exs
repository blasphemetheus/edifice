defmodule Edifice.Attention.SigmoidAttentionTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.SigmoidAttention

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  @small_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: 4,
    num_layers: 2,
    dropout: 0.0,
    window_size: @seq_len,
    seq_len: @seq_len,
    layer_scale: true,
    qk_norm: false
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "SigmoidAttention.build/1" do
    test "returns an Axon model" do
      model = SigmoidAttention.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = SigmoidAttention.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = SigmoidAttention.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "SigmoidAttention.compute/1" do
    test "applies sigmoid with sequence-length bias" do
      # Simple 2x2 logits
      logits = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = SigmoidAttention.compute(logits)

      # b = -log(2) ≈ -0.693
      # sigmoid(1 - 0.693) ≈ sigmoid(0.307) ≈ 0.576
      assert Nx.shape(result) == {2, 2}

      # All values should be in (0, 1)
      min_val = Nx.reduce_min(result) |> Nx.to_number()
      max_val = Nx.reduce_max(result) |> Nx.to_number()
      assert min_val > 0.0
      assert max_val < 1.0
    end

    test "values decrease relative to softmax for longer sequences" do
      # With more keys, bias = -log(n) grows more negative, pulling values down
      short = Nx.tensor([[1.0, 2.0]])
      long = Nx.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      short_max = SigmoidAttention.compute(short) |> Nx.reduce_max() |> Nx.to_number()
      long_max = SigmoidAttention.compute(long) |> Nx.reduce_max() |> Nx.to_number()

      # Longer sequence should have smaller max attention weight
      assert long_max < short_max
    end
  end

  describe "configuration variants" do
    test "works with QK-norm enabled" do
      opts = Keyword.put(@small_opts, :qk_norm, true)
      model = SigmoidAttention.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "works without LayerScale" do
      opts = Keyword.put(@small_opts, :layer_scale, false)
      model = SigmoidAttention.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build via registry" do
      model = Edifice.build(:sigmoid_attention, @small_opts)
      assert %Axon{} = model
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert SigmoidAttention.output_size(hidden_size: 128) == 128
      assert SigmoidAttention.output_size() == 256
    end
  end
end
