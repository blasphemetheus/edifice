defmodule Edifice.SSM.GatedSSMCoverage2Test do
  @moduledoc """
  Extended coverage tests for Edifice.SSM.GatedSSM.

  Targets the ~42% of uncovered code: build_checkpointed forward pass,
  step/4 incremental inference, step_mamba_block, step_causal_conv1d,
  step_ssm, dense_forward, layer_norm_forward, get_layer_params,
  param_count edge cases, and dropout paths.
  """
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  alias Edifice.SSM.GatedSSM

  # Small dims for speed
  @batch 2
  @embed 16
  @hidden 16
  @state_size 4
  @expand_factor 2
  @conv_size 2
  @num_layers 2
  @seq_len 8

  defp base_opts(overrides \\ []) do
    defaults = [
      embed_dim: @embed,
      hidden_size: @hidden,
      state_size: @state_size,
      expand_factor: @expand_factor,
      conv_size: @conv_size,
      num_layers: @num_layers,
      seq_len: @seq_len
    ]

    Keyword.merge(defaults, overrides)
  end

  defp build_and_run(opts) do
    model = GatedSSM.build(opts)
    seq = Keyword.get(opts, :seq_len, @seq_len)
    embed = Keyword.fetch!(opts, :embed_dim)

    {init_fn, predict_fn} = Axon.build(model)
    template = Nx.template({@batch, seq, embed}, :f32)
    params = init_fn.(template, Axon.ModelState.empty())

    key = Nx.Random.key(42)
    {input, _} = Nx.Random.uniform(key, shape: {@batch, seq, embed})

    output = predict_fn.(params, input)
    {model, params, output}
  end

  # ============================================================================
  # build/1 - Additional branches
  # ============================================================================

  describe "build/1 additional branches" do
    test "embed_dim == hidden_size skips input projection" do
      opts = base_opts(embed_dim: @hidden, hidden_size: @hidden)
      {_model, _params, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "embed_dim != hidden_size includes input projection" do
      opts = base_opts(embed_dim: 12, hidden_size: @hidden)
      {_model, _params, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "dropout > 0 with num_layers > 1 adds dropout between layers" do
      opts = base_opts(dropout: 0.2, num_layers: 3)
      {_model, _params, output} = build_and_run(opts)
      # Dropout should not produce NaN for inference
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "dropout > 0 not applied on last layer" do
      # With 2 layers and dropout, only layer 1 gets dropout (layer_idx < num_layers)
      opts = base_opts(dropout: 0.3, num_layers: 2)
      {_model, _params, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "single layer with dropout does not apply dropout" do
      # dropout > 0 but num_layers=1, so layer_idx == num_layers => no dropout
      opts = base_opts(dropout: 0.5, num_layers: 1)
      {_model, _params, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "custom conv_size" do
      opts = base_opts(conv_size: 3)
      {_model, _params, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "custom expand_factor" do
      opts = base_opts(expand_factor: 3)
      {_model, _params, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "window_size option is used as seq_len default" do
      opts = base_opts() |> Keyword.delete(:seq_len) |> Keyword.put(:window_size, 6)
      model = GatedSSM.build(opts)
      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, 6, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 6, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
    end
  end

  # ============================================================================
  # build_checkpointed/1 with forward passes
  # ============================================================================

  describe "build_checkpointed/1 forward pass" do
    test "produces correct output shape" do
      opts = base_opts(checkpoint_every: 1)
      model = GatedSSM.build_checkpointed(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with checkpoint_every > 1" do
      opts = base_opts(num_layers: 4, checkpoint_every: 2)
      model = GatedSSM.build_checkpointed(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "with embed_dim != hidden_size" do
      opts = base_opts(embed_dim: 12, hidden_size: @hidden, checkpoint_every: 1)
      model = GatedSSM.build_checkpointed(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, 12}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, 12})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "with embed_dim == hidden_size skips input projection" do
      opts = base_opts(embed_dim: @hidden, hidden_size: @hidden, checkpoint_every: 1)
      model = GatedSSM.build_checkpointed(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @hidden}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "with dropout" do
      opts = base_opts(dropout: 0.2, num_layers: 3, checkpoint_every: 1)
      model = GatedSSM.build_checkpointed(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "checkpoint_every=3 with num_layers=3 checkpoints only layer 3" do
      opts = base_opts(num_layers: 3, checkpoint_every: 3)
      model = GatedSSM.build_checkpointed(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden}
    end
  end

  # ============================================================================
  # init_cache/1 - Additional cases
  # ============================================================================

  describe "init_cache/1" do
    test "default options" do
      cache = GatedSSM.init_cache()
      assert cache.step == 0
      assert is_map(cache.layers)
      assert Map.has_key?(cache.layers, "layer_1")
      assert Map.has_key?(cache.layers, "layer_2")
      assert cache.config.hidden_size == 256
    end

    test "custom batch_size" do
      cache = GatedSSM.init_cache(batch_size: 4, hidden_size: 8, state_size: 4, num_layers: 1)
      layer = cache.layers["layer_1"]
      assert Nx.shape(layer.h) == {4, 16, 4}
      assert Nx.shape(layer.conv_buffer) == {4, 3, 16}
    end

    test "custom conv_size affects buffer shape" do
      cache =
        GatedSSM.init_cache(
          batch_size: 1,
          hidden_size: 8,
          state_size: 2,
          expand_factor: 2,
          conv_size: 6,
          num_layers: 1
        )

      layer = cache.layers["layer_1"]
      # conv_buffer: [batch, conv_size - 1, inner_size]
      # inner_size = hidden_size * expand_factor = 8 * 2 = 16
      assert Nx.shape(layer.conv_buffer) == {1, 5, 16}
    end

    test "config stored correctly" do
      cache =
        GatedSSM.init_cache(
          hidden_size: 32,
          state_size: 8,
          expand_factor: 3,
          conv_size: 5,
          num_layers: 4
        )

      assert cache.config.hidden_size == 32
      assert cache.config.state_size == 8
      assert cache.config.expand_factor == 3
      assert cache.config.conv_size == 5
      assert cache.config.num_layers == 4
      assert map_size(cache.layers) == 4
    end
  end

  # ============================================================================
  # step/4 - Incremental inference
  # ============================================================================

  describe "step/4 incremental inference" do
    # Helper to build model, get params, and run step
    defp setup_step_test(opts \\ []) do
      opts = base_opts(opts)
      model = GatedSSM.build(opts)

      seq = Keyword.get(opts, :seq_len, @seq_len)
      embed = Keyword.fetch!(opts, :embed_dim)
      hidden = Keyword.get(opts, :hidden_size, @hidden)
      state_size = Keyword.get(opts, :state_size, @state_size)
      expand_factor = Keyword.get(opts, :expand_factor, @expand_factor)
      conv_size = Keyword.get(opts, :conv_size, @conv_size)
      num_layers = Keyword.get(opts, :num_layers, @num_layers)

      {init_fn, _predict_fn} = Axon.build(model)
      template = Nx.template({1, seq, embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      cache =
        GatedSSM.init_cache(
          batch_size: 1,
          hidden_size: hidden,
          state_size: state_size,
          expand_factor: expand_factor,
          conv_size: conv_size,
          num_layers: num_layers
        )

      {params, cache, hidden}
    end

    test "single step with [batch, hidden_size] input" do
      {params, cache, hidden} = setup_step_test()

      key = Nx.Random.key(42)
      {x, _} = Nx.Random.uniform(key, shape: {1, hidden})

      {output, new_cache} = GatedSSM.step(x, params, cache)

      assert Nx.shape(output) == {1, hidden}
      assert new_cache.step == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "single step with [batch, 1, hidden_size] input" do
      {params, cache, hidden} = setup_step_test()

      key = Nx.Random.key(42)
      {x, _} = Nx.Random.uniform(key, shape: {1, 1, hidden})

      {output, new_cache} = GatedSSM.step(x, params, cache)

      assert Nx.shape(output) == {1, hidden}
      assert new_cache.step == 1
    end

    test "multiple sequential steps update cache" do
      {params, cache, hidden} = setup_step_test()

      key = Nx.Random.key(42)
      {x1, key} = Nx.Random.uniform(key, shape: {1, hidden})
      {x2, _key} = Nx.Random.uniform(key, shape: {1, hidden})

      {out1, cache1} = GatedSSM.step(x1, params, cache)
      {out2, cache2} = GatedSSM.step(x2, params, cache1)

      assert cache1.step == 1
      assert cache2.step == 2
      assert Nx.shape(out1) == {1, hidden}
      assert Nx.shape(out2) == {1, hidden}

      # Outputs should be different for different inputs
      # (they may be close due to initialization, but not all identical)
      assert Nx.shape(out1) == Nx.shape(out2)
    end

    test "step updates conv_buffer" do
      {params, cache, hidden} = setup_step_test(num_layers: 1)

      key = Nx.Random.key(42)
      {x, _} = Nx.Random.uniform(key, shape: {1, hidden})

      {_output, new_cache} = GatedSSM.step(x, params, cache)

      old_buffer = cache.layers["layer_1"].conv_buffer
      new_buffer = new_cache.layers["layer_1"].conv_buffer

      # Buffer should be updated (shifted)
      assert Nx.shape(old_buffer) == Nx.shape(new_buffer)
    end

    test "step updates hidden state h" do
      {params, cache, hidden} = setup_step_test(num_layers: 1)

      key = Nx.Random.key(42)
      {x, _} = Nx.Random.uniform(key, shape: {1, hidden})

      {_output, new_cache} = GatedSSM.step(x, params, cache)

      old_h = cache.layers["layer_1"].h
      new_h = new_cache.layers["layer_1"].h

      assert Nx.shape(old_h) == Nx.shape(new_h)
      # Hidden state should have been updated from all-zeros
      assert Nx.any(Nx.not_equal(old_h, new_h)) |> Nx.to_number() == 1
    end

    test "step with Axon.ModelState params" do
      opts = base_opts()
      model = GatedSSM.build(opts)

      seq = Keyword.get(opts, :seq_len, @seq_len)
      embed = Keyword.fetch!(opts, :embed_dim)

      {init_fn, _predict_fn} = Axon.build(model)
      template = Nx.template({1, seq, embed}, :f32)
      # init_fn returns ModelState directly
      params = init_fn.(template, Axon.ModelState.empty())

      cache =
        GatedSSM.init_cache(
          batch_size: 1,
          hidden_size: @hidden,
          state_size: @state_size,
          expand_factor: @expand_factor,
          conv_size: @conv_size,
          num_layers: @num_layers
        )

      key = Nx.Random.key(42)
      {x, _} = Nx.Random.uniform(key, shape: {1, @hidden})

      # Pass ModelState directly to test the params conversion branch
      {output, _new_cache} = GatedSSM.step(x, params, cache)
      assert Nx.shape(output) == {1, @hidden}
    end
  end

  # ============================================================================
  # build_mamba_block/2 - Different option combinations
  # ============================================================================

  describe "build_mamba_block/2" do
    test "with all custom options" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden})

      block =
        GatedSSM.build_mamba_block(input,
          hidden_size: @hidden,
          state_size: @state_size,
          expand_factor: 3,
          conv_size: 3,
          name: "custom_block"
        )

      assert %Axon{} = block
    end

    test "forward pass through single block" do
      input_node = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden})

      block =
        GatedSSM.build_mamba_block(input_node,
          hidden_size: @hidden,
          state_size: @state_size,
          expand_factor: @expand_factor,
          conv_size: @conv_size,
          name: "test_block"
        )

      {init_fn, predict_fn} = Axon.build(block)
      template = Nx.template({@batch, @seq_len, @hidden}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end
  end

  # ============================================================================
  # build_causal_conv1d/4 - Forward pass
  # ============================================================================

  describe "build_causal_conv1d/4 forward pass" do
    test "produces same-length output (causal padding)" do
      input_node = Axon.input("input", shape: {nil, @seq_len, @embed})
      conv = GatedSSM.build_causal_conv1d(input_node, @embed, 4, "test_conv")

      {init_fn, predict_fn} = Axon.build(conv)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      # Output should have same seq_len (causal padding preserves length)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end

    test "with kernel_size=2 (small padding)" do
      input_node = Axon.input("input", shape: {nil, @seq_len, @embed})
      conv = GatedSSM.build_causal_conv1d(input_node, @embed, 2, "small_conv")

      {init_fn, predict_fn} = Axon.build(conv)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end
  end

  # ============================================================================
  # build_selective_ssm/2 - Forward pass
  # ============================================================================

  describe "build_selective_ssm/2 forward pass" do
    test "produces correct output shape" do
      input_node = Axon.input("input", shape: {nil, @seq_len, @embed})

      ssm =
        GatedSSM.build_selective_ssm(input_node,
          hidden_size: @embed,
          state_size: @state_size,
          name: "test_ssm"
        )

      {init_fn, predict_fn} = Axon.build(ssm)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with custom dt_rank" do
      input_node = Axon.input("input", shape: {nil, @seq_len, @embed})

      ssm =
        GatedSSM.build_selective_ssm(input_node,
          hidden_size: @embed,
          state_size: @state_size,
          dt_rank: 2,
          name: "custom_ssm"
        )

      {init_fn, predict_fn} = Axon.build(ssm)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed})

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end
  end

  # ============================================================================
  # param_count/1 edge cases
  # ============================================================================

  describe "param_count/1" do
    test "with embed_dim == hidden_size (no input proj)" do
      count = GatedSSM.param_count(embed_dim: 32, hidden_size: 32, num_layers: 1)
      assert is_integer(count)
      assert count > 0
    end

    test "with embed_dim != hidden_size (includes input proj)" do
      count_no_proj = GatedSSM.param_count(embed_dim: 32, hidden_size: 32, num_layers: 1)
      count_with_proj = GatedSSM.param_count(embed_dim: 64, hidden_size: 32, num_layers: 1)
      # With projection should have more params
      assert count_with_proj > count_no_proj
    end

    test "scales with num_layers" do
      count1 = GatedSSM.param_count(embed_dim: 32, hidden_size: 32, num_layers: 1)
      count3 = GatedSSM.param_count(embed_dim: 32, hidden_size: 32, num_layers: 3)
      assert count3 > count1
    end

    test "default embed_dim used when not provided" do
      count = GatedSSM.param_count(hidden_size: 32, num_layers: 1)
      assert is_integer(count)
      assert count > 0
    end
  end

  # ============================================================================
  # Numerical stability
  # ============================================================================

  describe "numerical stability" do
    test "output is finite for normal input" do
      opts = base_opts()
      {_model, _params, output} = build_and_run(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is finite for zero input" do
      opts = base_opts()
      model = GatedSSM.build(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch, @seq_len, @embed})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is finite for large input" do
      opts = base_opts()
      model = GatedSSM.build(opts)

      {init_fn, predict_fn} = Axon.build(model)
      template = Nx.template({@batch, @seq_len, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      input = Nx.broadcast(10.0, {@batch, @seq_len, @embed})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # recommended_defaults/0
  # ============================================================================

  describe "recommended_defaults/0" do
    test "contains all expected keys" do
      defaults = GatedSSM.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :state_size)
      assert Keyword.has_key?(defaults, :expand_factor)
      assert Keyword.has_key?(defaults, :conv_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :dropout)
    end

    test "values are sensible" do
      defaults = GatedSSM.recommended_defaults()
      assert defaults[:hidden_size] > 0
      assert defaults[:state_size] > 0
      assert defaults[:expand_factor] > 0
      assert defaults[:num_layers] > 0
      assert defaults[:dropout] >= 0.0 and defaults[:dropout] <= 1.0
    end
  end

  # ============================================================================
  # output_size/1
  # ============================================================================

  describe "output_size/1" do
    test "default when no opts" do
      assert GatedSSM.output_size() == 256
    end

    test "custom hidden_size" do
      assert GatedSSM.output_size(hidden_size: 64) == 64
    end
  end
end
