defmodule Edifice.Attention.PrimitivesTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.Primitives
  alias Edifice.Blocks.CausalMask

  @batch 2
  @seq_len 8
  @dim 16

  describe "scaled_dot_product_attention/4" do
    test "computes attention without mask" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)

      result = Primitives.scaled_dot_product_attention(q, k, v)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.bitwise_not()) |> Nx.to_number() == 1
    end

    test "computes attention with 2D mask" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      mask = CausalMask.causal(@seq_len)

      result = Primitives.scaled_dot_product_attention(q, k, v, mask: mask)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end

    test "computes attention with 3D mask" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)

      mask =
        CausalMask.causal(@seq_len) |> Nx.broadcast({@batch, @seq_len, @seq_len})

      result = Primitives.scaled_dot_product_attention(q, k, v, mask: mask)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end
  end

  describe "chunked_attention/4" do
    test "produces same shape as standard attention" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)

      result = Primitives.chunked_attention(q, k, v, chunk_size: 4)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end

    test "with mask" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      mask = CausalMask.causal(@seq_len)

      result = Primitives.chunked_attention(q, k, v, chunk_size: 4, mask: mask)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end
  end

  describe "memory_efficient_attention/4" do
    test "produces correct shape" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)

      result = Primitives.memory_efficient_attention(q, k, v, chunk_size: 4)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end

    test "with causal masking" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)

      result = Primitives.memory_efficient_attention(q, k, v, chunk_size: 4, causal: true)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end
  end

  describe "multi_head_sdpa/3" do
    test "computes 4D attention" do
      q = Nx.broadcast(0.1, {@batch, 2, @seq_len, 8}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, 2, @seq_len, 8}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, 2, @seq_len, 8}) |> Nx.as_type(:f32)

      result = Primitives.multi_head_sdpa(q, k, v)
      assert Nx.shape(result) == {@batch, 2, @seq_len, 8}
    end

    test "with 2D mask" do
      q = Nx.broadcast(0.1, {@batch, 2, @seq_len, 8}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, 2, @seq_len, 8}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, 2, @seq_len, 8}) |> Nx.as_type(:f32)
      mask = CausalMask.causal(@seq_len)

      result = Primitives.multi_head_sdpa(q, k, v, mask: mask)
      assert Nx.shape(result) == {@batch, 2, @seq_len, 8}
    end
  end

  describe "reshape_to_heads/5 and reshape_from_heads/5" do
    test "round-trips correctly" do
      x = Nx.iota({@batch, @seq_len, @dim}, type: :f32)
      heads = Primitives.reshape_to_heads(x, @batch, @seq_len, 2, 8)
      assert Nx.shape(heads) == {@batch, 2, @seq_len, 8}

      back = Primitives.reshape_from_heads(heads, @batch, @seq_len, 2, 8)
      assert Nx.shape(back) == {@batch, @seq_len, @dim}
      assert Nx.all(Nx.equal(x, back)) |> Nx.to_number() == 1
    end
  end

  describe "qk_layer_norm/1" do
    test "normalizes tensor" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32)
      result = Primitives.qk_layer_norm(x)
      assert Nx.shape(result) == {1, 4}
      # Mean should be approximately 0
      mean = Nx.mean(result) |> Nx.to_number()
      assert abs(mean) < 0.01
    end
  end

  describe "add_positional_encoding/2" do
    test "adds positional encoding to input" do
      input = Axon.input("input", shape: {nil, @seq_len, @dim})
      layer = Primitives.add_positional_encoding(input)
      assert %Axon{} = layer
    end
  end
end
