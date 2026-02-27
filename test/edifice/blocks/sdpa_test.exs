defmodule Edifice.Blocks.SDPATest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.{CausalMask, SDPA}

  @batch 2
  @seq_len 8
  @hidden 32
  @num_heads 4
  @head_dim 8

  describe "compute/5 (no mask)" do
    test "produces correct output shape" do
      q = Nx.broadcast(0.5, {@batch, @seq_len, @hidden})
      k = Nx.broadcast(0.5, {@batch, @seq_len, @hidden})
      v = Nx.broadcast(0.5, {@batch, @seq_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "output values are finite" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end

    test "handles batch_size=1" do
      q = Nx.broadcast(0.5, {1, @seq_len, @hidden})
      k = Nx.broadcast(0.5, {1, @seq_len, @hidden})
      v = Nx.broadcast(0.5, {1, @seq_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim)

      assert Nx.shape(output) == {1, @seq_len, @hidden}
    end

    test "handles seq_len=1" do
      q = Nx.broadcast(0.5, {@batch, 1, @hidden})
      k = Nx.broadcast(0.5, {@batch, 1, @hidden})
      v = Nx.broadcast(0.5, {@batch, 1, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim)

      assert Nx.shape(output) == {@batch, 1, @hidden}
    end

    test "different Q/KV lengths (cross-attention pattern)" do
      q_len = 4
      kv_len = 12

      q = Nx.broadcast(0.5, {@batch, q_len, @hidden})
      k = Nx.broadcast(0.5, {@batch, kv_len, @hidden})
      v = Nx.broadcast(0.5, {@batch, kv_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim)

      assert Nx.shape(output) == {@batch, q_len, @hidden}
    end

    test "uniform input produces uniform attention (output near mean of values)" do
      # With identical Q, K, V, attention weights should be uniform
      v = Nx.broadcast(1.0, {@batch, @seq_len, @hidden})
      q = Nx.broadcast(1.0, {@batch, @seq_len, @hidden})
      k = Nx.broadcast(1.0, {@batch, @seq_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim)

      # With uniform V = 1.0 and uniform attention, output should be close to 1.0
      mean = Nx.to_number(Nx.mean(output))
      assert_in_delta(mean, 1.0, 0.1)
    end
  end

  describe "compute/6 (with mask)" do
    test "produces correct shape with causal mask" do
      mask = CausalMask.causal(@seq_len)

      q = Nx.broadcast(0.5, {@batch, @seq_len, @hidden})
      k = Nx.broadcast(0.5, {@batch, @seq_len, @hidden})
      v = Nx.broadcast(0.5, {@batch, @seq_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim, mask)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "causal mask produces finite output" do
      mask = CausalMask.causal(@seq_len)

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      output = SDPA.compute(q, k, v, @num_heads, @head_dim, mask)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "nil mask is same as no mask" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      out_5 = SDPA.compute(q, k, v, @num_heads, @head_dim)
      out_6 = SDPA.compute(q, k, v, @num_heads, @head_dim, nil)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(out_5, out_6))))
      assert diff < 1.0e-5
    end

    test "masked output differs from unmasked output" do
      mask = CausalMask.causal(@seq_len)

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      out_unmasked = SDPA.compute(q, k, v, @num_heads, @head_dim)
      out_masked = SDPA.compute(q, k, v, @num_heads, @head_dim, mask)

      diff = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(out_unmasked, out_masked))))
      assert diff > 0.01, "Causal mask should change attention output"
    end
  end
end
