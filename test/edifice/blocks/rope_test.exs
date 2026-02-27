defmodule Edifice.Blocks.RoPETest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.RoPE

  @batch 2
  @seq_len 8
  @hidden 32

  describe "precompute_freqs/2" do
    test "returns correct shape" do
      {cos_table, sin_table} = RoPE.precompute_freqs(32, 64)
      assert Nx.shape(cos_table) == {64, 16}
      assert Nx.shape(sin_table) == {64, 16}
    end

    test "cos and sin tables are bounded in [-1, 1]" do
      {cos_t, sin_t} = RoPE.precompute_freqs(32, 64)

      assert Nx.to_number(Nx.reduce_max(cos_t)) <= 1.0
      assert Nx.to_number(Nx.reduce_min(cos_t)) >= -1.0
      assert Nx.to_number(Nx.reduce_max(sin_t)) <= 1.0
      assert Nx.to_number(Nx.reduce_min(sin_t)) >= -1.0
    end

    test "tables are finite" do
      {cos_t, sin_t} = RoPE.precompute_freqs(64, 128)

      refute Nx.any(Nx.is_nan(cos_t)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_nan(sin_t)) |> Nx.to_number() == 1
    end

    test "different dim/max_len combos" do
      {cos_t, sin_t} = RoPE.precompute_freqs(8, 16)
      assert Nx.shape(cos_t) == {16, 4}
      assert Nx.shape(sin_t) == {16, 4}
    end
  end

  describe "apply_rotary/2" do
    test "preserves vector norms (rotation property)" do
      key = Nx.Random.key(42)
      {x, key} = Nx.Random.normal(key, shape: {1, @seq_len, @hidden})
      {y, _} = Nx.Random.normal(key, shape: {1, @seq_len, @hidden})

      {x_rot, _y_rot} = RoPE.apply_rotary(x, y)

      x_norms = Nx.sqrt(Nx.sum(Nx.pow(x, 2), axes: [2]))
      x_rot_norms = Nx.sqrt(Nx.sum(Nx.pow(x_rot, 2), axes: [2]))

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_norms, x_rot_norms))))
      assert diff < 1.0e-4
    end

    test "position 0 is close to identity" do
      x = Nx.tensor([[[1.0, 2.0, 3.0, 4.0]]])
      y = Nx.tensor([[[1.0, 1.0, 1.0, 1.0]]])

      {x_rot, _} = RoPE.apply_rotary(x, y)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(x_rot[0][0], x[0][0]))))
      assert diff < 1.0e-5
    end

    test "different positions produce different embeddings" do
      x = Nx.broadcast(Nx.tensor([1.0, 2.0, 3.0, 4.0]), {1, 4, 4})
      y = Nx.broadcast(Nx.tensor([1.0, 1.0, 1.0, 1.0]), {1, 4, 4})

      {x_rot, _} = RoPE.apply_rotary(x, y)

      diff = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(x_rot[0][0], x_rot[0][1]))))
      assert diff > 0.01
    end
  end

  describe "layer/2" do
    test "applies rotation in Axon model" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = RoPE.layer(input, dim: @hidden, name: "test_rope")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = RoPE.layer(input, dim: @hidden, name: "test_rope")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, @seq_len, @hidden}))

      assert Nx.shape(output) == {1, @seq_len, @hidden}
    end
  end
end
