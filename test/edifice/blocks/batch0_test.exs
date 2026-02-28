defmodule Edifice.Blocks.Batch0Test do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Attention.GQA
  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.CausalMask
  alias Edifice.Blocks.DepthwiseConv
  alias Edifice.Blocks.TransformerBlock
  alias Edifice.Recurrent.TTT

  @batch_size 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 64

  # ============================================================================
  # CausalMask Tests
  # ============================================================================

  describe "CausalMask" do
    test "causal/1 produces correct shape and lower-triangular pattern" do
      mask = CausalMask.causal(4)
      assert Nx.shape(mask) == {4, 4}

      # First row: only position 0
      assert Nx.to_number(mask[0][0]) == 1
      assert Nx.to_number(mask[0][1]) == 0

      # Last row: all positions
      assert Nx.to_number(mask[3][0]) == 1
      assert Nx.to_number(mask[3][3]) == 1
    end

    test "window/2 produces correct windowed mask" do
      mask = CausalMask.window(6, 3)
      assert Nx.shape(mask) == {6, 6}

      # Position 4 should attend to positions 2, 3, 4 (window=3)
      assert Nx.to_number(mask[4][1]) == 0
      assert Nx.to_number(mask[4][2]) == 1
      assert Nx.to_number(mask[4][3]) == 1
      assert Nx.to_number(mask[4][4]) == 1
      assert Nx.to_number(mask[4][5]) == 0
    end

    test "to_binary_backend/1 copies to BinaryBackend" do
      mask = CausalMask.causal(4) |> CausalMask.to_binary_backend()
      assert Nx.shape(mask) == {4, 4}
    end
  end

  # ============================================================================
  # DepthwiseConv Tests
  # ============================================================================

  describe "DepthwiseConv" do
    test "layer/4 produces correct output shape" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})
      output = DepthwiseConv.layer(input, @hidden_size, 3, name: "test_dw")

      {init_fn, predict_fn} = Axon.build(output)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @seq_len, @hidden_size}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "layer/4 with different out_channels" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})
      output = DepthwiseConv.layer(input, @hidden_size, 5, out_channels: 32, name: "test_dw2")

      {init_fn, predict_fn} = Axon.build(output)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @seq_len, 32}
    end
  end

  # ============================================================================
  # TransformerBlock :custom_ffn Tests
  # ============================================================================

  describe "TransformerBlock :custom_ffn" do
    test "custom_ffn callback replaces standard FFN" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      custom_ffn_fn = fn x, name ->
        x
        |> Axon.dense(@hidden_size * 2, name: "#{name}_custom_up")
        |> Axon.activation(:silu, name: "#{name}_custom_act")
        |> Axon.dense(@hidden_size, name: "#{name}_custom_down")
      end

      output =
        TransformerBlock.layer(input,
          attention_fn: fn x, name ->
            MultiHead.self_attention(x,
              hidden_size: @hidden_size,
              num_heads: 4,
              name: name,
              causal: false
            )
          end,
          hidden_size: @hidden_size,
          custom_ffn: custom_ffn_fn,
          name: "custom_ffn_block"
        )

      {init_fn, predict_fn} = Axon.build(output)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @seq_len, @hidden_size}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # MultiHead :rope Tests
  # ============================================================================

  describe "MultiHead :rope option" do
    test "self_attention with rope produces correct shape" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      output =
        MultiHead.self_attention(input,
          hidden_size: @hidden_size,
          num_heads: 4,
          rope: true,
          causal: true,
          name: "rope_attn"
        )

      {init_fn, predict_fn} = Axon.build(output)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @seq_len, @hidden_size}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # GQA :rope Tests
  # ============================================================================

  describe "GQA :rope option" do
    test "build with rope produces correct shape" do
      model =
        GQA.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 8,
          num_kv_heads: 2,
          num_layers: 1,
          rope: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # TTT :variant Tests
  # ============================================================================

  describe "TTT :variant option" do
    test "linear variant (default) produces correct shape" do
      model =
        TTT.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          inner_size: 16,
          num_layers: 1,
          variant: :linear
        )

      {init_fn, predict_fn} = Axon.build(model)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "mlp variant produces correct shape" do
      model =
        TTT.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          inner_size: 16,
          num_layers: 1,
          variant: :mlp
        )

      {init_fn, predict_fn} = Axon.build(model)

      inp = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      result = predict_fn.(params, %{"state_sequence" => inp})
      assert Nx.shape(result) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
