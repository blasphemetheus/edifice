defmodule Edifice.Blocks.FlashAttentionTest do
  @moduledoc """
  Correctness tests for flash attention.

  Compares FusedScan.flash_attention output against the standard SDPA
  fallback path on random inputs. Flash attention should produce
  numerically equivalent results.
  """
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.CUDA.FusedScan

  @batch 2
  @num_heads 4
  @seq_len 32
  @head_dim 64

  defp assert_all_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    assert Nx.shape(actual) == Nx.shape(expected),
           "Shape mismatch: #{inspect(Nx.shape(actual))} vs #{inspect(Nx.shape(expected))}"

    diff = actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff < atol,
           "Max absolute difference #{diff} exceeds tolerance #{atol}"
  end

  defp reference_sdpa(q, k, v, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)

    scale = Nx.sqrt(Nx.tensor(head_dim, type: {:f, 32}))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    scores =
      if causal == 1 do
        rows = Nx.iota({seq_len, seq_len}, axis: 0)
        cols = Nx.iota({seq_len, seq_len}, axis: 1)
        mask = Nx.greater_equal(rows, cols)

        mask =
          mask
          |> Nx.reshape({1, 1, seq_len, seq_len})
          |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

        neg_inf = Nx.Constants.neg_infinity({:f, 32})
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    # Stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    weights = Nx.exp(Nx.subtract(scores, max_scores))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  describe "flash_attention_fallback/4" do
    test "no mask: produces correct output shape" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})

      output = FusedScan.flash_attention(q, k, v, causal: false)

      assert Nx.shape(output) == {@batch, @num_heads, @seq_len, @head_dim}
    end

    test "no mask: matches reference SDPA" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})

      flash_out = FusedScan.flash_attention(q, k, v, causal: false)
      ref_out = reference_sdpa(q, k, v, 0)

      assert_all_close(flash_out, ref_out, atol: 1.0e-5)
    end

    test "causal mask: produces correct output shape" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})

      output = FusedScan.flash_attention(q, k, v, causal: true)

      assert Nx.shape(output) == {@batch, @num_heads, @seq_len, @head_dim}
    end

    test "causal mask: matches reference SDPA" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})

      flash_out = FusedScan.flash_attention(q, k, v, causal: true)
      ref_out = reference_sdpa(q, k, v, 1)

      assert_all_close(flash_out, ref_out, atol: 1.0e-5)
    end

    test "causal vs non-causal produce different outputs" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})

      causal_out = FusedScan.flash_attention(q, k, v, causal: true)
      full_out = FusedScan.flash_attention(q, k, v, causal: false)

      diff = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(causal_out, full_out))))
      assert diff > 0.01, "Causal vs full attention should differ"
    end

    test "output values are finite" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, @head_dim})

      output = FusedScan.flash_attention(q, k, v, causal: false)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end

    test "uniform input produces uniform output" do
      q = Nx.broadcast(1.0, {@batch, @num_heads, @seq_len, @head_dim})
      k = Nx.broadcast(1.0, {@batch, @num_heads, @seq_len, @head_dim})
      v = Nx.broadcast(1.0, {@batch, @num_heads, @seq_len, @head_dim})

      output = FusedScan.flash_attention(q, k, v, causal: false)

      mean = Nx.to_number(Nx.mean(output))
      assert_in_delta(mean, 1.0, 0.01)
    end

    test "seq_len=1 works correctly" do
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, 1, @head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, 1, @head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, 1, @head_dim})

      output = FusedScan.flash_attention(q, k, v, causal: false)
      assert Nx.shape(output) == {@batch, @num_heads, 1, @head_dim}

      # With seq=1, output should equal v (softmax of single element = 1.0)
      assert_all_close(output, v, atol: 1.0e-5)
    end

    test "head_dim=32 works correctly" do
      head_dim = 32
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, head_dim})
      {k, key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {@batch, @num_heads, @seq_len, head_dim})

      flash_out = FusedScan.flash_attention(q, k, v, causal: false)
      ref_out = reference_sdpa(q, k, v, 0)

      assert_all_close(flash_out, ref_out, atol: 1.0e-5)
    end

    test "head_dim=128 works correctly" do
      head_dim = 128
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {1, 2, 16, head_dim})
      {k, key} = Nx.Random.normal(key, shape: {1, 2, 16, head_dim})
      {v, _key} = Nx.Random.normal(key, shape: {1, 2, 16, head_dim})

      flash_out = FusedScan.flash_attention(q, k, v, causal: false)
      ref_out = reference_sdpa(q, k, v, 0)

      assert_all_close(flash_out, ref_out, atol: 1.0e-5)
    end
  end
end
