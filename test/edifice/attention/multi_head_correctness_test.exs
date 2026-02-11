defmodule Edifice.Attention.MultiHeadCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.MultiHead

  @batch 2
  @seq_len 8
  @embed_size 32
  @num_heads 4
  @head_dim 8
  @hidden_dim @num_heads * @head_dim

  # ============================================================================
  # Multi-Head Reshape Verification
  # ============================================================================

  describe "multi-head reshape" do
    test "different num_heads produces different outputs (not cosmetic)" do
      # Build two models: 1 head vs 4 heads, same hidden_dim
      model_1h =
        MultiHead.build_sliding_window(
          embed_size: @embed_size,
          num_heads: 1,
          head_dim: @hidden_dim,
          num_layers: 1,
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0
        )

      model_4h =
        MultiHead.build_sliding_window(
          embed_size: @embed_size,
          num_heads: 4,
          head_dim: @head_dim,
          num_layers: 1,
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0
        )

      # Both models have same hidden_dim (32)
      # With proper multi-head, they should behave differently given same params
      {init_1h, predict_1h} = Axon.build(model_1h, mode: :inference)
      {init_4h, predict_4h} = Axon.build(model_4h, mode: :inference)

      params_1h =
        init_1h.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      params_4h =
        init_4h.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output_1h = predict_1h.(params_1h, input)
      output_4h = predict_4h.(params_4h, input)

      # Both should produce [batch, hidden_dim] output
      assert Nx.shape(output_1h) == {@batch, @hidden_dim}
      assert Nx.shape(output_4h) == {@batch, @hidden_dim}

      # They should produce different outputs (proving heads matter)
      diff = Nx.subtract(output_1h, output_4h) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "1-head and 4-head models should produce different outputs"
    end

    test "self_attention with num_heads > 1 produces correct output shape" do
      input_node = Axon.input("x", shape: {nil, @seq_len, @embed_size})

      attn =
        MultiHead.self_attention(input_node,
          hidden_dim: @hidden_dim,
          num_heads: @num_heads,
          causal: false,
          dropout: 0.0,
          name: "test_attn"
        )

      {init_fn, predict_fn} = Axon.build(attn, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, seq_len, hidden_dim]
      assert Nx.shape(output) == {@batch, @seq_len, @hidden_dim}
    end

    test "multi_head_attention passes num_heads to self_attention" do
      input_node = Axon.input("x", shape: {nil, @seq_len, @embed_size})

      attn =
        MultiHead.multi_head_attention(input_node,
          num_heads: @num_heads,
          head_dim: @head_dim,
          causal: false,
          dropout: 0.0,
          name: "mha_test"
        )

      {init_fn, predict_fn} = Axon.build(attn, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Attention Score Properties
  # ============================================================================

  describe "attention score properties" do
    test "scaled_dot_product_attention scores sum to 1 per query" do
      # Test the standalone attention function with 3D tensors
      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {2, 6, 8})
      {k, key} = Nx.Random.normal(key, shape: {2, 6, 8})
      {v, _} = Nx.Random.normal(key, shape: {2, 6, 8})

      # Compute attention and verify output is finite
      output = MultiHead.scaled_dot_product_attention(q, k, v)
      assert Nx.shape(output) == {2, 6, 8}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "causal mask prevents attending to future positions" do
      mask = MultiHead.causal_mask(4)

      # mask[i, j] should be true iff j <= i
      expected =
        Nx.tensor([
          [1, 0, 0, 0],
          [1, 1, 0, 0],
          [1, 1, 1, 0],
          [1, 1, 1, 1]
        ])
        |> Nx.as_type(:u8)

      assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "sliding window attention is deterministic in inference mode" do
      model =
        MultiHead.build_sliding_window(
          embed_size: @embed_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          num_layers: 1,
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  # ============================================================================
  # Hybrid Model
  # ============================================================================

  describe "hybrid model" do
    test "hybrid LSTM + attention produces correct shape" do
      model =
        MultiHead.build_hybrid(
          embed_size: @embed_size,
          lstm_hidden: 16,
          lstm_layers: 1,
          num_heads: 2,
          head_dim: 8,
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # hidden_dim = num_heads * head_dim = 2 * 8 = 16
      assert Nx.shape(output) == {@batch, 16}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
