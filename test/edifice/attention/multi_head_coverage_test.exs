defmodule Edifice.Attention.MultiHeadCoverageTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.MultiHead

  @batch 2
  @seq_len 8
  @dim 16

  describe "causal_mask/1 (delegated)" do
    test "produces lower-triangular mask" do
      mask = MultiHead.causal_mask(4)
      assert Nx.shape(mask) == {4, 4}
      # Top-left should be true (position 0 can attend to position 0)
      assert Nx.to_number(mask[0][0]) == 1
      # Top-right should be false (position 0 cannot attend to position 3)
      assert Nx.to_number(mask[0][3]) == 0
      # Bottom-left should be true (position 3 can attend to position 0)
      assert Nx.to_number(mask[3][0]) == 1
    end
  end

  describe "window_mask/2 (delegated)" do
    test "produces windowed causal mask" do
      mask = MultiHead.window_mask(6, 3)
      assert Nx.shape(mask) == {6, 6}
      # Position 4 can attend to positions 2, 3, 4 but not 0, 1
      assert Nx.to_number(mask[4][2]) == 1
      assert Nx.to_number(mask[4][4]) == 1
      assert Nx.to_number(mask[4][1]) == 0
      # Future positions always masked
      assert Nx.to_number(mask[4][5]) == 0
    end
  end

  describe "scaled_dot_product_attention/4 (delegated)" do
    test "delegates to Primitives" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @dim}) |> Nx.as_type(:f32)

      result = MultiHead.scaled_dot_product_attention(q, k, v)
      assert Nx.shape(result) == {@batch, @seq_len, @dim}
    end
  end

  describe "self_attention/2" do
    test "builds Axon layer" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @dim})
      layer = MultiHead.self_attention(input, hidden_size: @dim, num_heads: 1, causal: false)
      assert %Axon{} = layer
    end

    test "with qk_layernorm" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @dim})

      layer =
        MultiHead.self_attention(input,
          hidden_size: @dim,
          num_heads: 1,
          qk_layernorm: true,
          causal: false
        )

      assert %Axon{} = layer
    end
  end

  describe "add_positional_encoding/2 (delegated)" do
    test "adds positional encoding to input" do
      input = Axon.input("input", shape: {nil, @seq_len, @dim})
      layer = MultiHead.add_positional_encoding(input)
      assert %Axon{} = layer
    end
  end

  describe "output_size/1" do
    test "returns num_heads * head_dim" do
      assert MultiHead.output_size(num_heads: 4, head_dim: 64) == 256
    end

    test "defaults" do
      assert MultiHead.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = MultiHead.recommended_defaults()
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :num_heads)
    end
  end
end
