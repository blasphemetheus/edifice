defmodule Edifice.Meta.CapsuleCorrectnessTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.Capsule

  @batch 2
  @num_primary_caps 4
  @primary_cap_dim 4
  @num_digit_caps 5
  @digit_cap_dim 8
  @conv_channels 16
  @conv_kernel 3

  # Input must be large enough to survive conv layers:
  # initial_conv (kernel 3, valid) -> 28-3+1=26
  # primary_caps_conv (kernel 9, stride 2, valid) -> (26-9)/2+1=9
  # This gives 9*9*4=324 primary capsules
  @input_height 28
  @input_width 28
  @input_channels 1

  @base_opts [
    input_shape: {nil, @input_height, @input_width, @input_channels},
    num_primary_caps: @num_primary_caps,
    primary_cap_dim: @primary_cap_dim,
    num_digit_caps: @num_digit_caps,
    digit_cap_dim: @digit_cap_dim,
    conv_channels: @conv_channels,
    conv_kernel: @conv_kernel
  ]

  # ============================================================================
  # Per-Pair Transform Weights
  # ============================================================================

  describe "per-pair transform weights" do
    test "params contain routing W key (per-pair weight matrix)" do
      model = Capsule.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @input_height, @input_width, @input_channels}, :f32),
          Axon.ModelState.empty()
        )

      # The routing W param is nested: params.data["digit_caps"]["digit_caps_W"]
      digit_caps_params = params.data["digit_caps"]

      assert is_map(digit_caps_params) and not is_struct(digit_caps_params),
             "Should have 'digit_caps' param group with nested W, got keys: #{inspect(Map.keys(params.data))}"

      w_keys = Map.keys(digit_caps_params) |> Enum.filter(&String.contains?(&1, "_W"))

      assert w_keys != [],
             "digit_caps should contain a '_W' per-pair weight param, got: #{inspect(Map.keys(digit_caps_params))}"
    end

    test "params do NOT contain routing_transform dense layer" do
      model = Capsule.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @input_height, @input_width, @input_channels}, :f32),
          Axon.ModelState.empty()
        )

      param_keys = Map.keys(params.data)

      transform_keys = Enum.filter(param_keys, &String.contains?(&1, "routing_transform"))

      assert transform_keys == [],
             "Should not have a 'routing_transform' dense layer, but found: #{inspect(transform_keys)}"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, num_digit_caps]" do
      model = Capsule.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @input_height, @input_width, @input_channels}, :f32),
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)

      {input, _} =
        Nx.Random.uniform(key, shape: {@batch, @input_height, @input_width, @input_channels})

      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @num_digit_caps}
    end

    test "output values are in [0, 1] (capsule norms represent probabilities)" do
      model = Capsule.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @input_height, @input_width, @input_channels}, :f32),
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)

      {input, _} =
        Nx.Random.uniform(key, shape: {@batch, @input_height, @input_width, @input_channels})

      output = predict_fn.(params, input)

      # Capsule norms should be in [0, 1) due to squash activation
      min_val = Nx.reduce_min(output) |> Nx.to_number()
      max_val = Nx.reduce_max(output) |> Nx.to_number()

      assert min_val >= 0.0,
             "Capsule norms should be >= 0, got min = #{min_val}"

      assert max_val <= 1.0,
             "Capsule norms should be <= 1, got max = #{max_val}"
    end

    test "output is finite (no NaN/Inf)" do
      model = Capsule.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @input_height, @input_width, @input_channels}, :f32),
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)

      {input, _} =
        Nx.Random.uniform(key, shape: {@batch, @input_height, @input_width, @input_channels})

      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is deterministic" do
      model = Capsule.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @input_height, @input_width, @input_channels}, :f32),
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)

      {input, _} =
        Nx.Random.uniform(key, shape: {@batch, @input_height, @input_width, @input_channels})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end
end
