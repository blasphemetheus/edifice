defmodule Edifice.Vision.GaussianSplatTest do
  use ExUnit.Case, async: true

  alias Edifice.Vision.GaussianSplat

  @num_gaussians 100
  @sh_degree 3

  defp random_point_cloud(num_points) do
    key = Nx.Random.key(42)
    {points, _key} = Nx.Random.uniform(key, shape: {num_points, 3}, type: :f32)
    points
  end

  defp random_point_cloud_with_colors(num_points) do
    key = Nx.Random.key(42)
    {points, key} = Nx.Random.uniform(key, shape: {num_points, 3}, type: :f32)
    {colors, _key} = Nx.Random.uniform(key, shape: {num_points, 3}, type: :f32)
    Nx.concatenate([points, colors], axis: 1)
  end

  defp simple_camera do
    # Identity-ish view and projection for testing
    view_matrix =
      Nx.tensor([
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, -5.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      ])

    proj_matrix =
      Nx.tensor([
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      ])

    camera_position = Nx.tensor([[0.0, 0.0, 5.0]])

    %{
      view_matrix: view_matrix,
      proj_matrix: proj_matrix,
      position: camera_position
    }
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = GaussianSplat.build(num_gaussians: @num_gaussians, sh_degree: @sh_degree)
      assert %Axon{} = model
    end

    test "accepts custom options" do
      model = GaussianSplat.build(num_gaussians: 500, sh_degree: 2, name: "custom_splat")
      assert %Axon{} = model
    end
  end

  describe "initialize_gaussians/2" do
    test "initializes from point cloud" do
      points = random_point_cloud(50)
      gaussians = GaussianSplat.initialize_gaussians(points)

      assert Map.has_key?(gaussians, :positions)
      assert Map.has_key?(gaussians, :rotations)
      assert Map.has_key?(gaussians, :scales)
      assert Map.has_key?(gaussians, :opacities)
      assert Map.has_key?(gaussians, :sh_coeffs)

      assert Nx.shape(gaussians.positions) == {50, 3}
      assert Nx.shape(gaussians.rotations) == {50, 4}
      assert Nx.shape(gaussians.scales) == {50, 3}
      assert Nx.shape(gaussians.opacities) == {50, 1}
    end

    test "initializes with colors from point cloud" do
      points = random_point_cloud_with_colors(50)
      gaussians = GaussianSplat.initialize_gaussians(points)

      # SH coefficients should have colors in DC component
      assert {50, _sh_dim} = Nx.shape(gaussians.sh_coeffs)
    end

    test "rotations are initialized to identity quaternion" do
      points = random_point_cloud(10)
      gaussians = GaussianSplat.initialize_gaussians(points)

      # Identity quaternion is [1, 0, 0, 0]
      w = gaussians.rotations[[.., 0]]
      xyz = gaussians.rotations[[.., 1..3]]

      # W should be 1
      assert Nx.all(Nx.equal(w, 1.0)) |> Nx.to_number() == 1
      # XYZ should be 0
      assert Nx.all(Nx.equal(xyz, 0.0)) |> Nx.to_number() == 1
    end

    test "respects sh_degree option" do
      points = random_point_cloud(10)

      for degree <- 0..3 do
        gaussians = GaussianSplat.initialize_gaussians(points, sh_degree: degree)
        sh_coeffs_expected = [:math.pow(degree + 1, 2) |> trunc()] |> List.first()
        sh_dim_expected = 3 * sh_coeffs_expected

        assert {10, ^sh_dim_expected} = Nx.shape(gaussians.sh_coeffs)
      end
    end
  end

  describe "project_gaussians/2" do
    test "projects Gaussians to 2D" do
      points = random_point_cloud(20)
      gaussians = GaussianSplat.initialize_gaussians(points)
      camera = simple_camera()

      projected = GaussianSplat.project_gaussians(gaussians, camera)

      assert Map.has_key?(projected, :means_2d)
      assert Map.has_key?(projected, :covs_2d)
      assert Map.has_key?(projected, :depths)

      assert Nx.shape(projected.means_2d) == {20, 2}
      assert Nx.shape(projected.covs_2d) == {20, 2, 2}
      assert Nx.shape(projected.depths) == {20}
    end

    test "depths are positive" do
      # Points in front of camera should have positive depth
      points = Nx.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], type: :f32)
      gaussians = GaussianSplat.initialize_gaussians(points)
      camera = simple_camera()

      projected = GaussianSplat.project_gaussians(gaussians, camera)

      # With camera at z=5 looking at origin, depths should be ~5
      assert Nx.all(Nx.greater(projected.depths, 0.0)) |> Nx.to_number() == 1
    end

    test "2D covariances are symmetric" do
      points = random_point_cloud(10)
      gaussians = GaussianSplat.initialize_gaussians(points)
      camera = simple_camera()

      projected = GaussianSplat.project_gaussians(gaussians, camera)

      # Check symmetry: cov[0,1] == cov[1,0]
      c01 = projected.covs_2d[[.., 0, 1]]
      c10 = projected.covs_2d[[.., 1, 0]]

      diff = Nx.subtract(c01, c10) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end
  end

  describe "render/4" do
    test "renders to correct image size" do
      points = random_point_cloud(20)
      gaussians = GaussianSplat.initialize_gaussians(points)
      camera = simple_camera()

      image = GaussianSplat.render(gaussians, camera, {32, 32})

      assert Nx.shape(image) == {32, 32, 3}
    end

    test "output values are in [0, 1]" do
      points = random_point_cloud(20)
      gaussians = GaussianSplat.initialize_gaussians(points)
      camera = simple_camera()

      image = GaussianSplat.render(gaussians, camera, {32, 32})

      min_val = Nx.reduce_min(image) |> Nx.to_number()
      max_val = Nx.reduce_max(image) |> Nx.to_number()

      assert min_val >= 0.0
      assert max_val <= 1.0
    end

    test "output contains finite values" do
      points = random_point_cloud(20)
      gaussians = GaussianSplat.initialize_gaussians(points)
      camera = simple_camera()

      image = GaussianSplat.render(gaussians, camera, {32, 32})

      assert Nx.all(Nx.is_nan(image) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(image) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "densification_step/3" do
    test "prunes low opacity Gaussians" do
      points = random_point_cloud(50)
      gaussians = GaussianSplat.initialize_gaussians(points)

      # Set half to very low opacity
      low_opacity = Nx.broadcast(-10.0, {25, 1})
      high_opacity = Nx.broadcast(0.0, {25, 1})
      opacities = Nx.concatenate([low_opacity, high_opacity], axis: 0)

      gaussians = %{gaussians | opacities: opacities}

      # Create dummy gradients
      gradients = Nx.broadcast(0.0, {50, 3})

      result = GaussianSplat.densification_step(gaussians, gradients, prune_threshold: 0.01)

      # Should have fewer Gaussians
      {new_count, _} = Nx.shape(result.positions)
      assert new_count < 50
    end

    test "preserves Gaussian structure" do
      points = random_point_cloud(20)
      gaussians = GaussianSplat.initialize_gaussians(points)
      gradients = Nx.broadcast(0.0001, {20, 3})

      result = GaussianSplat.densification_step(gaussians, gradients)

      assert Map.has_key?(result, :positions)
      assert Map.has_key?(result, :rotations)
      assert Map.has_key?(result, :scales)
      assert Map.has_key?(result, :opacities)
      assert Map.has_key?(result, :sh_coeffs)
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = GaussianSplat.recommended_defaults()

      assert Keyword.has_key?(defaults, :num_gaussians)
      assert Keyword.has_key?(defaults, :sh_degree)
      assert Keyword.has_key?(defaults, :position_lr)
      assert Keyword.has_key?(defaults, :opacity_lr)
    end
  end

  describe "output_size/1" do
    test "returns image dimensions" do
      assert GaussianSplat.output_size(image_height: 256, image_width: 512) == {256, 512, 3}
      assert GaussianSplat.output_size() == {256, 256, 3}
    end
  end

  describe "param_count/1" do
    test "calculates parameter count correctly" do
      # Per Gaussian: 3 (pos) + 4 (rot) + 3 (scale) + 1 (opacity) + 3*16 (SH degree 3) = 59
      count = GaussianSplat.param_count(num_gaussians: 1000, sh_degree: 3)
      assert count == 1000 * 59
    end

    test "varies with sh_degree" do
      count_0 = GaussianSplat.param_count(num_gaussians: 100, sh_degree: 0)
      count_3 = GaussianSplat.param_count(num_gaussians: 100, sh_degree: 3)

      # SH degree 0: 3*1 = 3, SH degree 3: 3*16 = 48
      # Difference per Gaussian: 48 - 3 = 45
      assert count_3 - count_0 == 100 * 45
    end
  end
end
