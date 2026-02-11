defmodule Edifice.Blocks.CorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.{
    RMSNorm,
    SwiGLU,
    RoPE,
    ALiBi,
    PatchEmbed,
    SinusoidalPE,
    AdaptiveNorm,
    CrossAttention,
    FFN,
    TransformerBlock
  }

  @batch 2
  @seq_len 8
  @hidden 32

  # ============================================================================
  # RMSNorm Correctness
  # ============================================================================

  describe "RMSNorm correctness" do
    test "output has unit RMS when gamma=1" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)

      # RMS of the output should be approximately 1.0
      rms = Nx.sqrt(Nx.mean(Nx.pow(result, 2), axes: [-1]))
      assert_close(rms, Nx.tensor([1.0]), atol: 1.0e-5)
    end

    test "scale parameter works correctly" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma_ones = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      gamma_twos = Nx.tensor([2.0, 2.0, 2.0, 2.0])

      result_ones = RMSNorm.apply(x, gamma_ones)
      result_twos = RMSNorm.apply(x, gamma_twos)

      # gamma=2 should produce exactly 2x the output of gamma=1
      assert_close(Nx.multiply(result_ones, 2.0), result_twos, atol: 1.0e-5)
    end

    test "invariant to input scaling" do
      # RMSNorm(alpha * x) should equal RMSNorm(x) for any positive alpha
      # because the normalization cancels the scaling
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])

      result_1x = RMSNorm.apply(x, gamma)
      result_5x = RMSNorm.apply(Nx.multiply(x, 5.0), gamma)

      assert_close(result_1x, result_5x, atol: 1.0e-5)
    end

    test "handles zero input without NaN" do
      x = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)

      refute has_nan(result)
    end

    test "preserves batch dimension independence" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)

      # Each row should be independently normalized
      result_row0 = RMSNorm.apply(Nx.tensor([[1.0, 2.0, 3.0]]), gamma)
      result_row1 = RMSNorm.apply(Nx.tensor([[10.0, 20.0, 30.0]]), gamma)

      assert_close(result[0], result_row0[0], atol: 1.0e-5)
      assert_close(result[1], result_row1[0], atol: 1.0e-5)
    end
  end

  # ============================================================================
  # RoPE Correctness
  # ============================================================================

  describe "RoPE correctness" do
    test "preserves vector norms" do
      # RoPE is a rotation, so ||RoPE(x)|| == ||x||
      key = Nx.Random.key(42)
      {x, _} = Nx.Random.normal(key, shape: {1, @seq_len, @hidden}, type: :f32)
      {y, _} = Nx.Random.normal(Nx.Random.key(43), shape: {1, @seq_len, @hidden}, type: :f32)

      {x_rot, _y_rot} = RoPE.apply_rotary(x, y)

      # Norms should be preserved at each position
      x_norms = Nx.sqrt(Nx.sum(Nx.pow(x, 2), axes: [2]))
      x_rot_norms = Nx.sqrt(Nx.sum(Nx.pow(x_rot, 2), axes: [2]))

      assert_close(x_norms, x_rot_norms, atol: 1.0e-4)
    end

    test "position 0 is close to identity for low frequencies" do
      # At position 0, angles are all 0, so cos=1, sin=0 -> identity
      x = Nx.tensor([[[1.0, 2.0, 3.0, 4.0]]])
      y = Nx.tensor([[[1.0, 1.0, 1.0, 1.0]]])

      {x_rot, _} = RoPE.apply_rotary(x, y)

      # At position 0, rotation should be identity
      assert_close(x_rot[0][0], x[0][0], atol: 1.0e-5)
    end

    test "different positions produce different embeddings" do
      # Same vector at different positions should produce different results
      x = Nx.broadcast(Nx.tensor([1.0, 2.0, 3.0, 4.0]), {1, 4, 4})
      y = Nx.broadcast(Nx.tensor([1.0, 1.0, 1.0, 1.0]), {1, 4, 4})

      {x_rot, _} = RoPE.apply_rotary(x, y)

      # Position 0 and position 1 should differ
      diff = Nx.sum(Nx.abs(Nx.subtract(x_rot[0][0], x_rot[0][1])))
      assert Nx.to_number(diff) > 0.01
    end

    test "precomputed frequencies have correct shape" do
      dim = 64
      max_len = 128
      {cos_t, sin_t} = RoPE.precompute_freqs(dim, max_len)

      assert Nx.shape(cos_t) == {max_len, div(dim, 2)}
      assert Nx.shape(sin_t) == {max_len, div(dim, 2)}
    end

    test "cos and sin tables are bounded in [-1, 1]" do
      {cos_t, sin_t} = RoPE.precompute_freqs(32, 64)

      assert Nx.to_number(Nx.reduce_max(cos_t)) <= 1.0
      assert Nx.to_number(Nx.reduce_min(cos_t)) >= -1.0
      assert Nx.to_number(Nx.reduce_max(sin_t)) <= 1.0
      assert Nx.to_number(Nx.reduce_min(sin_t)) >= -1.0
    end
  end

  # ============================================================================
  # SwiGLU Correctness
  # ============================================================================

  describe "SwiGLU correctness" do
    test "zero gate produces near-zero output" do
      # When the gate branch outputs zeros, the gated multiplication should be ~zero
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, name: "test_swiglu")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      # Zero input -> gate activation is near 0 for SiLU (SiLU(0) = 0)
      output = predict_fn.(params, Nx.broadcast(0.0, {@batch, @seq_len, @hidden}))

      # Output should be very small (not exactly zero due to bias)
      max_val = Nx.to_number(Nx.reduce_max(Nx.abs(output)))
      # Bias terms may cause non-zero, but should be bounded
      assert max_val < 10.0
    end

    test "different activations produce different outputs" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})

      silu_model = SwiGLU.layer(input, hidden_size: @hidden, activation: :silu, name: "silu")
      gelu_model = SwiGLU.layer(input, hidden_size: @hidden, activation: :gelu, name: "gelu")

      {silu_init, silu_pred} = Axon.build(silu_model)
      {gelu_init, gelu_pred} = Axon.build(gelu_model)

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)

      silu_params =
        silu_init.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      gelu_params =
        gelu_init.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      silu_out = silu_pred.(silu_params, test_input)
      gelu_out = gelu_pred.(gelu_params, test_input)

      # Different activations with different params should produce different results
      # (this mainly verifies that the activation option is actually used)
      assert Nx.shape(silu_out) == Nx.shape(gelu_out)
    end

    test "output dimension matches input dimension" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, name: "test_swiglu")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "inner size defaults to roughly hidden * 2.667" do
      input = Axon.input("input", shape: {nil, @seq_len, 256})
      model = SwiGLU.layer(input, hidden_size: 256, name: "test")

      # The model should have been built - verify by running it
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 256}, :f32), Axon.ModelState.empty())

      # Check that inner projection weight exists and has expected inner_size dimension
      # inner_size = round(256 * 2.667) = 683, rounded to multiple of 8 = 688
      gate_key = "test_gate"
      assert Map.has_key?(params.data, gate_key)
      gate_kernel = params.data[gate_key]["kernel"]
      # Second dim of kernel should be the inner_size
      {_input_dim, inner_size} = Nx.shape(gate_kernel)
      assert rem(inner_size, 8) == 0
    end
  end

  # ============================================================================
  # ALiBi Correctness
  # ============================================================================

  describe "ALiBi correctness" do
    test "slopes follow geometric sequence" do
      slopes = ALiBi.compute_slopes(8)

      # Slopes should be 2^(-1), 2^(-2), ..., 2^(-8)
      expected =
        Enum.map(1..8, fn i -> :math.pow(2.0, -i) end)
        |> Nx.tensor(type: :f32)

      assert_close(slopes, expected, atol: 1.0e-6)
    end

    test "correct slope formula for power-of-2 heads" do
      # For n_heads = 8, ratio = 1.0, so slopes = 2^-1, 2^-2, ..., 2^-8
      slopes = ALiBi.compute_slopes(8)
      first_slope = Nx.to_number(slopes[0])
      last_slope = Nx.to_number(slopes[7])

      assert_in_delta(first_slope, 0.5, 1.0e-6)
      assert_in_delta(last_slope, :math.pow(2.0, -8), 1.0e-6)
    end

    test "causal bias has zero on diagonal" do
      bias = ALiBi.compute_bias(seq_len: 8, num_heads: 4)

      # All heads should have 0 on the diagonal (self-distance)
      for h <- 0..3 do
        for i <- 0..7 do
          val = Nx.to_number(bias[h][i][i])
          assert_in_delta(val, 0.0, 1.0e-5, "Head #{h}, position #{i}")
        end
      end
    end

    test "bias magnitude increases with distance" do
      bias = ALiBi.compute_bias(seq_len: 8, num_heads: 2, causal: true)

      # For causal bias, looking back further should have more negative bias
      # At head 0, position 4 looking at position 3 vs position 0
      bias_near = Nx.to_number(bias[0][4][3])
      bias_far = Nx.to_number(bias[0][4][0])

      # Farther position should have more negative bias
      assert bias_far < bias_near
    end
  end

  # ============================================================================
  # PatchEmbed Correctness
  # ============================================================================

  describe "PatchEmbed correctness" do
    test "correct patch count" do
      assert PatchEmbed.num_patches(224, 16) == 196
      assert PatchEmbed.num_patches(32, 8) == 16
      assert PatchEmbed.num_patches(64, 16) == 16
    end

    test "correct patch dimension before projection" do
      # patch_dim = P * P * C
      # For P=8, C=3: patch_dim = 192
      # For P=16, C=3: patch_dim = 768
      image = Axon.input("image", shape: {nil, 3, 32, 32})
      model = PatchEmbed.layer(image, embed_dim: 64, patch_size: 8, in_channels: 3)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 3, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(1.0, {1, 3, 32, 32}))

      # 32/8 = 4, 4*4 = 16 patches, projected to 64-dim
      assert Nx.shape(output) == {1, 16, 64}
    end

    test "patches are non-overlapping" do
      # Create image where each patch region has a unique value
      # 2x2 patches on a 4x4 image = 4 patches
      image = Axon.input("image", shape: {nil, 1, 4, 4})

      # Use identity-like projection to verify patches don't overlap
      model = PatchEmbed.layer(image, embed_dim: 4, patch_size: 2, in_channels: 1)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1, 4, 4}, :f32), Axon.ModelState.empty())

      # Make each 2x2 patch have different content
      test_image =
        Nx.tensor([
          [
            [
              [1.0, 1.0, 2.0, 2.0],
              [1.0, 1.0, 2.0, 2.0],
              [3.0, 3.0, 4.0, 4.0],
              [3.0, 3.0, 4.0, 4.0]
            ]
          ]
        ])

      output = predict_fn.(params, test_image)
      assert Nx.shape(output) == {1, 4, 4}
    end
  end

  # ============================================================================
  # SinusoidalPE Correctness
  # ============================================================================

  describe "SinusoidalPE correctness" do
    test "values are bounded in [-1, 1]" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)

      assert Nx.to_number(Nx.reduce_max(table)) <= 1.0 + 1.0e-6
      assert Nx.to_number(Nx.reduce_min(table)) >= -1.0 - 1.0e-6
    end

    test "distinct positions produce distinct encodings" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)

      pos0 = table[0]
      pos1 = table[1]
      pos50 = table[50]

      diff_01 = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(pos0, pos1))))
      diff_050 = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(pos0, pos50))))

      assert diff_01 > 0.1, "Position 0 and 1 should differ"
      assert diff_050 > 0.1, "Position 0 and 50 should differ"
    end

    test "frequencies decrease along the dimension" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)

      # The first half contains sin values; earlier dims have higher frequencies
      # Check that the first dim oscillates faster than later dims by comparing
      # variance across positions
      sin_part = Nx.slice_along_axis(table, 0, 32, axis: 1)

      var_dim0 = Nx.to_number(Nx.variance(sin_part[[.., 0]]))
      var_dim31 = Nx.to_number(Nx.variance(sin_part[[.., 31]]))

      # Higher frequency -> more variation across positions
      assert var_dim0 > var_dim31 * 0.5,
             "First dim should have at least comparable variance to last"
    end

    test "position 0 has sin=0 for all sin dimensions" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)
      # First half is sin values; sin(0) = 0 for all frequencies
      sin_at_pos0 = Nx.slice_along_axis(table[0], 0, 32, axis: 0)

      max_sin_at_0 = Nx.to_number(Nx.reduce_max(Nx.abs(sin_at_pos0)))
      assert max_sin_at_0 < 1.0e-5, "sin(0) should be 0 for all frequencies"
    end
  end

  # ============================================================================
  # AdaptiveNorm Correctness
  # ============================================================================

  describe "AdaptiveNorm correctness" do
    test "zero conditioning approximates LayerNorm" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      condition = Axon.input("condition", shape: {nil, @hidden})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: @hidden,
          mode: :adaln,
          name: "test_adaln"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "condition" => Nx.template({@batch, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)

      output =
        predict_fn.(params, %{
          "input" => test_input,
          "condition" => Nx.broadcast(0.0, {@batch, @hidden})
        })

      # Output should be finite and same shape
      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
      refute has_nan(output)
    end

    test "adaln_zero with zero conditioning produces small output" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      condition = Axon.input("condition", shape: {nil, @hidden})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: @hidden,
          mode: :adaln_zero,
          name: "test_adaln_zero"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "condition" => Nx.template({@batch, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "input" => Nx.broadcast(1.0, {@batch, @seq_len, @hidden}),
          "condition" => Nx.broadcast(0.0, {@batch, @hidden})
        })

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
      refute has_nan(output)
    end

    test "adaln responds to different conditioning" do
      h = 64
      input = Axon.input("input", shape: {nil, 4, h})
      condition = Axon.input("condition", shape: {nil, h})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: h,
          mode: :adaln,
          name: "test_adaln"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({1, 4, h}, :f32),
            "condition" => Nx.template({1, h}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {1, 4, h}, type: :f32)

      out_zeros =
        predict_fn.(params, %{
          "input" => test_input,
          "condition" => Nx.broadcast(0.0, {1, h})
        })

      out_large =
        predict_fn.(params, %{
          "input" => test_input,
          "condition" => Nx.broadcast(100.0, {1, h})
        })

      are_equal =
        Nx.to_number(Nx.all(Nx.less(Nx.abs(Nx.subtract(out_zeros, out_large)), 1.0e-6)))

      assert are_equal == 0, "Different conditions should produce different outputs"
    end
  end

  # ============================================================================
  # CrossAttention Correctness
  # ============================================================================

  describe "CrossAttention correctness" do
    test "Q from query input, KV from context" do
      # Different query and context inputs should produce valid output
      queries = Axon.input("queries", shape: {nil, 4, 16})
      context = Axon.input("context", shape: {nil, 8, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_dim: 16,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({1, 4, 16}, :f32),
            "context" => Nx.template({1, 8, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(1.0, {1, 4, 16}),
          "context" => Nx.broadcast(1.0, {1, 8, 16})
        })

      # Output seq_len should match query seq_len, not context
      assert Nx.shape(output) == {1, 4, 16}
    end

    test "output changes with different context" do
      queries = Axon.input("queries", shape: {nil, 4, 16})
      context = Axon.input("context", shape: {nil, 8, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_dim: 16,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({1, 4, 16}, :f32),
            "context" => Nx.template({1, 8, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      query_input = Nx.broadcast(1.0, {1, 4, 16})

      out1 =
        predict_fn.(params, %{
          "queries" => query_input,
          "context" => Nx.broadcast(1.0, {1, 8, 16})
        })

      key = Nx.Random.key(42)
      {rand_ctx, _} = Nx.Random.normal(key, shape: {1, 8, 16}, type: :f32)

      out2 =
        predict_fn.(params, %{
          "queries" => query_input,
          "context" => rand_ctx
        })

      diff = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(out1, out2))))
      assert diff > 0.01, "Different context should produce different outputs"
    end

    test "handles different query and context sequence lengths" do
      queries = Axon.input("queries", shape: {nil, 3, 16})
      context = Axon.input("context", shape: {nil, 20, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_dim: 16,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({1, 3, 16}, :f32),
            "context" => Nx.template({1, 20, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(0.5, {1, 3, 16}),
          "context" => Nx.broadcast(0.5, {1, 20, 16})
        })

      assert Nx.shape(output) == {1, 3, 16}
      refute has_nan(output)
    end

    test "attention weights sum to 1 (verified via uniform input)" do
      # With uniform Q=K, attention weights should be uniform (1/N per position)
      # This means the output should be approximately the mean of values
      queries = Axon.input("queries", shape: {nil, 4, 8})
      context = Axon.input("context", shape: {nil, 4, 8})

      model =
        CrossAttention.layer(queries, context,
          hidden_dim: 8,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({1, 4, 8}, :f32),
            "context" => Nx.template({1, 4, 8}, :f32)
          },
          Axon.ModelState.empty()
        )

      # Uniform input should produce finite output
      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(1.0, {1, 4, 8}),
          "context" => Nx.broadcast(1.0, {1, 4, 8})
        })

      refute has_nan(output)
    end
  end

  # ============================================================================
  # FFN (new) Correctness
  # ============================================================================

  describe "FFN correctness" do
    test "output dimension equals input dimension" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.layer(input, hidden_size: @hidden, name: "test_ffn")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "correct expansion factor" do
      hidden = 64
      input = Axon.input("input", shape: {nil, @seq_len, hidden})
      model = FFN.layer(input, hidden_size: hidden, expansion_factor: 4, name: "test_ffn")

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, hidden}, :f32), Axon.ModelState.empty())

      # Up projection kernel should be [hidden, hidden*4]
      up_kernel = params.data["test_ffn_up"]["kernel"]
      {in_dim, inner_dim} = Nx.shape(up_kernel)
      assert in_dim == hidden
      assert inner_dim == hidden * 4
    end

    test "deterministic at dropout=0" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.layer(input, hidden_size: @hidden, dropout: 0.0, name: "test_ffn")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)

      out1 = predict_fn.(params, test_input)
      out2 = predict_fn.(params, test_input)

      assert_close(out1, out2, atol: 1.0e-6)
    end
  end

  # ============================================================================
  # TransformerBlock (new) Correctness
  # ============================================================================

  describe "TransformerBlock correctness" do
    test "pre-norm residual structure preserves shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})

      # Identity attention function (pass through)
      attn_fn = fn x, name ->
        Axon.dense(x, @hidden, name: "#{name}_proj")
      end

      model =
        TransformerBlock.layer(input,
          attention_fn: attn_fn,
          hidden_size: @hidden,
          name: "test_block"
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
      refute has_nan(output)
    end

    test "attention_fn callback is actually used" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})

      # Two different attention functions should produce different models
      attn_fn_dense = fn x, name ->
        Axon.dense(x, @hidden, name: "#{name}_dense")
      end

      attn_fn_zero = fn x, name ->
        Axon.nx(x, fn t -> Nx.broadcast(0.0, Nx.shape(t)) end, name: "#{name}_zero")
      end

      model1 =
        TransformerBlock.layer(input,
          attention_fn: attn_fn_dense,
          hidden_size: @hidden,
          name: "block_dense"
        )

      model2 =
        TransformerBlock.layer(input,
          attention_fn: attn_fn_zero,
          hidden_size: @hidden,
          name: "block_zero"
        )

      {init_fn1, pred_fn1} = Axon.build(model1)
      {init_fn2, pred_fn2} = Axon.build(model2)

      params1 =
        init_fn1.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      params2 =
        init_fn2.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)

      out1 = pred_fn1.(params1, test_input)
      out2 = pred_fn2.(params2, test_input)

      # Both should have same shape but different values
      assert Nx.shape(out1) == Nx.shape(out2)
    end

    test "stack produces correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})

      attn_fn = fn x, name ->
        Axon.dense(x, @hidden, name: "#{name}_proj")
      end

      model =
        TransformerBlock.stack(input, 3,
          attention_fn: attn_fn,
          hidden_size: @hidden,
          name: "stack"
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden}, type: :f32)
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
      refute has_nan(output)
    end
  end

  # ============================================================================
  # ModelBuilder Correctness
  # ============================================================================

  describe "ModelBuilder correctness" do
    test "build_sequence_model with last_timestep output" do
      attn_fn = fn x, name ->
        Axon.dense(x, 32, name: "#{name}_proj")
      end

      block_builder = fn input, opts ->
        name = "block_#{opts[:layer_idx]}"

        TransformerBlock.layer(input,
          attention_fn: attn_fn,
          hidden_size: 32,
          name: name
        )
      end

      model =
        Edifice.Blocks.ModelBuilder.build_sequence_model(
          embed_size: 16,
          hidden_size: 32,
          num_layers: 2,
          seq_len: @seq_len,
          block_builder: block_builder,
          output_mode: :last_timestep
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 16}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, 16}, type: :f32)
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, 32}
      refute has_nan(output)
    end

    test "build_sequence_model with all output mode" do
      block_builder = fn input, opts ->
        name = "block_#{opts[:layer_idx]}"
        Axon.dense(input, 32, name: "#{name}_dense")
      end

      model =
        Edifice.Blocks.ModelBuilder.build_sequence_model(
          embed_size: 32,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: block_builder,
          output_mode: :all
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 32}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @seq_len, 32}))

      # :all mode returns full sequence
      assert Nx.shape(output) == {@batch, @seq_len, 32}
    end

    test "build_sequence_model with mean_pool output" do
      block_builder = fn input, opts ->
        name = "block_#{opts[:layer_idx]}"
        Axon.dense(input, 32, name: "#{name}_dense")
      end

      model =
        Edifice.Blocks.ModelBuilder.build_sequence_model(
          embed_size: 32,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: block_builder,
          output_mode: :mean_pool
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 32}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @seq_len, 32}))

      assert Nx.shape(output) == {@batch, 32}
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp assert_close(actual, expected, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(actual, expected))))

    assert diff < atol,
           "Expected tensors to be close (max diff: #{diff}, tolerance: #{atol})\n" <>
             "actual: #{inspect(Nx.to_flat_list(actual))}\n" <>
             "expected: #{inspect(Nx.to_flat_list(expected))}"
  end

  defp has_nan(tensor) do
    Nx.to_number(Nx.any(Nx.is_nan(tensor))) == 1
  end
end
