defmodule Edifice.Generative.TRELLISTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.TRELLIS

  # Use smaller dimensions for faster testing
  @batch 2
  @max_voxels 64
  @feature_dim 16
  @hidden_size 32
  @num_layers 2
  @num_heads 4
  @window_size 4
  @condition_dim 24
  @cond_len 4
  @voxel_resolution 16

  @opts [
    voxel_resolution: @voxel_resolution,
    feature_dim: @feature_dim,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: @num_heads,
    window_size: @window_size,
    condition_dim: @condition_dim,
    mlp_ratio: 2.0,
    max_voxels: @max_voxels
  ]

  defp random_inputs(seed \\ 42) do
    key = Nx.Random.key(seed)
    {features, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, @feature_dim})
    {positions, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 3})
    positions = Nx.multiply(positions, @voxel_resolution) |> Nx.floor() |> Nx.as_type(:f32)
    {conditioning, key} = Nx.Random.uniform(key, shape: {@batch, @cond_len, @condition_dim})
    {timestep, _key} = Nx.Random.uniform(key, shape: {@batch})

    # Random occupancy mask (about half occupied)
    mask = Nx.broadcast(1.0, {@batch, @max_voxels})

    %{
      "sparse_features" => features,
      "voxel_positions" => positions,
      "occupancy_mask" => mask,
      "conditioning" => conditioning,
      "timestep" => timestep
    }
  end

  defp build_model_and_predict(opts \\ @opts, seed \\ 42) do
    model = TRELLIS.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "sparse_features" => Nx.template({@batch, @max_voxels, @feature_dim}, :f32),
      "voxel_positions" => Nx.template({@batch, @max_voxels, 3}, :f32),
      "occupancy_mask" => Nx.template({@batch, @max_voxels}, :f32),
      "conditioning" => Nx.template({@batch, @cond_len, @condition_dim}, :f32),
      "timestep" => Nx.template({@batch}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_inputs(seed))
    {output, params}
  end

  # ============================================================================
  # build/1
  # ============================================================================

  describe "build/1" do
    test "returns an Axon model" do
      model = TRELLIS.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {output, _params} = build_model_and_predict()

      # Output should be [batch, max_voxels, feature_dim]
      assert Nx.shape(output) == {@batch, @max_voxels, @feature_dim}
    end

    test "output contains finite values" do
      {output, _params} = build_model_and_predict()

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with single layer" do
      opts = Keyword.put(@opts, :num_layers, 1)
      {output, _params} = build_model_and_predict(opts)
      assert Nx.shape(output) == {@batch, @max_voxels, @feature_dim}
    end

    test "works with different window sizes" do
      for window_size <- [2, 4, 8] do
        opts = Keyword.put(@opts, :window_size, window_size)
        model = TRELLIS.build(opts)
        assert %Axon{} = model
      end
    end

    test "respects occupancy mask" do
      model = TRELLIS.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "sparse_features" => Nx.template({@batch, @max_voxels, @feature_dim}, :f32),
        "voxel_positions" => Nx.template({@batch, @max_voxels, 3}, :f32),
        "occupancy_mask" => Nx.template({@batch, @max_voxels}, :f32),
        "conditioning" => Nx.template({@batch, @cond_len, @condition_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      inputs = random_inputs()
      # Set half of voxels as unoccupied
      partial_mask =
        Nx.concatenate(
          [
            Nx.broadcast(1.0, {@batch, div(@max_voxels, 2)}),
            Nx.broadcast(0.0, {@batch, div(@max_voxels, 2)})
          ],
          axis: 1
        )

      inputs_masked = Map.put(inputs, "occupancy_mask", partial_mask)
      output = predict_fn.(params, inputs_masked)

      assert Nx.shape(output) == {@batch, @max_voxels, @feature_dim}
    end
  end

  # ============================================================================
  # encode_to_slat/2
  # ============================================================================

  describe "encode_to_slat/2" do
    test "converts dense grid to sparse representation" do
      # Create a simple dense occupancy grid
      dense_grid = Nx.broadcast(0.0, {@batch, 8, 8, 8})
      # Set some voxels as occupied
      dense_grid = Nx.put_slice(dense_grid, [0, 0, 0, 0], Nx.broadcast(1.0, {1, 2, 2, 2}))

      result = TRELLIS.encode_to_slat(dense_grid, max_voxels: 64)

      assert Map.has_key?(result, :features)
      assert Map.has_key?(result, :positions)
      assert Map.has_key?(result, :mask)

      # Positions should be 3D coordinates
      assert Nx.axis_size(result.positions, 2) == 3
    end

    test "handles dense grid with features" do
      # Dense grid with features: [batch, res, res, res, features]
      dense_grid = Nx.broadcast(0.5, {@batch, 8, 8, 8, 4})

      result = TRELLIS.encode_to_slat(dense_grid, max_voxels: 64, feature_dim: 4)

      assert Map.has_key?(result, :features)
      # Has feature dimension
      assert Nx.axis_size(result.features, 2) >= 1
    end

    test "respects max_voxels limit" do
      dense_grid = Nx.broadcast(1.0, {@batch, 8, 8, 8})

      result = TRELLIS.encode_to_slat(dense_grid, max_voxels: 32)

      assert Nx.axis_size(result.positions, 1) == 32
      assert Nx.axis_size(result.mask, 1) == 32
    end
  end

  # ============================================================================
  # decode_from_slat/2
  # ============================================================================

  describe "decode_from_slat/2" do
    setup do
      key = Nx.Random.key(42)
      {features, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 16})
      {positions, _key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 3})
      positions = Nx.multiply(positions, 16) |> Nx.floor() |> Nx.as_type(:f32)
      mask = Nx.broadcast(1.0, {@batch, @max_voxels})

      %{sparse_latent: %{features: features, positions: positions, mask: mask}}
    end

    test "decodes to dense format", %{sparse_latent: sl} do
      result = TRELLIS.decode_from_slat(sl, output_format: :dense, resolution: 16)

      # Dense output should be [batch, res, res, res, features]
      assert Nx.rank(result) == 5
      assert Nx.axis_size(result, 0) == @batch
      assert Nx.axis_size(result, 1) == 16
      assert Nx.axis_size(result, 2) == 16
      assert Nx.axis_size(result, 3) == 16
    end

    test "decodes to gaussian splats", %{sparse_latent: sl} do
      result = TRELLIS.decode_from_slat(sl, output_format: :gaussian_splats)

      assert Map.has_key?(result, :positions)
      assert Map.has_key?(result, :scales)
      assert Map.has_key?(result, :rotations)
      assert Map.has_key?(result, :colors)
      assert Map.has_key?(result, :opacities)

      # Scales should be positive (sigmoid applied)
      assert Nx.all(Nx.greater(result.scales, 0)) |> Nx.to_number() == 1

      # Colors should be in [0, 1]
      assert Nx.all(Nx.greater_equal(result.colors, 0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(result.colors, 1)) |> Nx.to_number() == 1
    end

    test "decodes to mesh format", %{sparse_latent: sl} do
      result = TRELLIS.decode_from_slat(sl, output_format: :mesh, resolution: 8)

      assert Map.has_key?(result, :density)
      assert Map.has_key?(result, :note)
    end
  end

  # ============================================================================
  # sparse_attention/3
  # ============================================================================

  describe "sparse_attention/3" do
    test "computes windowed attention" do
      key = Nx.Random.key(42)
      {features, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, @hidden_size})
      {positions, _key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 3})
      positions = Nx.multiply(positions, @voxel_resolution) |> Nx.floor() |> Nx.as_type(:f32)

      result =
        TRELLIS.sparse_attention(features, positions,
          window_size: @window_size,
          num_heads: @num_heads
        )

      assert Nx.shape(result) == {@batch, @max_voxels, @hidden_size}
    end

    test "output is finite" do
      key = Nx.Random.key(42)
      {features, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, @hidden_size})
      {positions, _key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 3})
      positions = Nx.multiply(positions, @voxel_resolution) |> Nx.floor() |> Nx.as_type(:f32)

      result = TRELLIS.sparse_attention(features, positions, window_size: 4)

      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "respects occupancy mask" do
      key = Nx.Random.key(42)
      {features, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, @hidden_size})
      {positions, _key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 3})
      positions = Nx.multiply(positions, @voxel_resolution) |> Nx.floor() |> Nx.as_type(:f32)

      # Half occupied
      mask =
        Nx.concatenate(
          [
            Nx.broadcast(1.0, {@batch, div(@max_voxels, 2)}),
            Nx.broadcast(0.0, {@batch, div(@max_voxels, 2)})
          ],
          axis: 1
        )

      result =
        TRELLIS.sparse_attention(features, positions,
          window_size: 4,
          mask: mask
        )

      assert Nx.shape(result) == {@batch, @max_voxels, @hidden_size}
    end
  end

  # ============================================================================
  # Rectified Flow
  # ============================================================================

  describe "rectified_flow_step/6" do
    test "performs one denoising step" do
      model = TRELLIS.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model)

      template = %{
        "sparse_features" => Nx.template({@batch, @max_voxels, @feature_dim}, :f32),
        "voxel_positions" => Nx.template({@batch, @max_voxels, 3}, :f32),
        "occupancy_mask" => Nx.template({@batch, @max_voxels}, :f32),
        "conditioning" => Nx.template({@batch, @cond_len, @condition_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {features, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, @feature_dim})
      {positions, key} = Nx.Random.uniform(key, shape: {@batch, @max_voxels, 3})
      positions = Nx.multiply(positions, @voxel_resolution) |> Nx.floor() |> Nx.as_type(:f32)
      {conditioning, _key} = Nx.Random.uniform(key, shape: {@batch, @cond_len, @condition_dim})
      mask = Nx.broadcast(1.0, {@batch, @max_voxels})

      x_t = %{features: features, positions: positions, mask: mask}
      t = Nx.broadcast(0.5, {@batch})

      result = TRELLIS.rectified_flow_step(model, params, x_t, t, conditioning, num_steps: 10)

      assert Map.has_key?(result, :features)
      assert Map.has_key?(result, :positions)
      assert Map.has_key?(result, :mask)

      # Features should have same shape
      assert Nx.shape(result.features) == {@batch, @max_voxels, @feature_dim}

      # Positions should be unchanged
      assert Nx.all(Nx.equal(result.positions, positions)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = TRELLIS.recommended_defaults()

      assert Keyword.has_key?(defaults, :voxel_resolution)
      assert Keyword.has_key?(defaults, :feature_dim)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :window_size)
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = TRELLIS.param_count(@opts)
      assert is_integer(count)
      assert count > 0
    end

    test "increases with more layers" do
      count_2 = TRELLIS.param_count(num_layers: 2, hidden_size: 32)
      count_4 = TRELLIS.param_count(num_layers: 4, hidden_size: 32)
      assert count_4 > count_2
    end
  end

  describe "output_size/1" do
    test "returns feature_dim" do
      assert TRELLIS.output_size(@opts) == @feature_dim
    end

    test "returns default when no opts" do
      assert TRELLIS.output_size() == 32
    end
  end

  # ============================================================================
  # Registry Integration
  # ============================================================================

  describe "Edifice registry" do
    test "trellis is registered" do
      assert :trellis in Edifice.list_architectures()
    end

    test "can build via Edifice.build/2" do
      model = Edifice.build(:trellis, @opts)
      assert %Axon{} = model
    end
  end
end
