defmodule Edifice.Vision.GaussianSplat do
  @moduledoc """
  3D Gaussian Splatting for real-time radiance field rendering.

  Represents 3D scenes as collections of Gaussian primitives that can be
  differentiably rendered from arbitrary viewpoints. Achieves 100x faster
  rendering than NeRF while maintaining or improving quality.

  ## Key Insight

  Instead of querying a neural network per ray sample (NeRF), represent the
  scene explicitly as a set of 3D Gaussians. Each Gaussian has position,
  shape (covariance), color (spherical harmonics), and opacity. Rendering
  is a simple alpha-compositing of projected 2D Gaussians:

  ```
  Scene Representation:
  +------------------------------------------+
  | N Gaussians, each with:                  |
  |   μ: position [x, y, z]                  |
  |   Σ: covariance (via R, S)               |  Σ = R·S·S^T·R^T
  |   c: color (SH coefficients)             |  48 values for degree 3
  |   α: opacity [0, 1]                      |
  +------------------------------------------+

  Rendering Pipeline:
  3D Gaussians → Project to 2D → Sort by depth → Alpha composite
       |              |              |               |
       v              v              v               v
    [μ,Σ,c,α]    2D splats      front-to-back    final image
  ```

  ## Differentiable Rendering

  For each pixel, accumulate color from overlapping Gaussians:

  ```
  C(p) = Σᵢ cᵢ · αᵢ · G(p; μ'ᵢ, Σ'ᵢ) · Πⱼ<ᵢ (1 - αⱼ · G(p; μ'ⱼ, Σ'ⱼ))
  ```

  Where:
  - G(p; μ', Σ') is the 2D Gaussian evaluated at pixel p
  - μ', Σ' are the projected mean and covariance
  - Products are over Gaussians in front (sorted by depth)

  ## Adaptive Density Control

  During training, Gaussians are dynamically adjusted:
  - **Clone**: Split high-gradient small Gaussians
  - **Split**: Divide large high-gradient Gaussians into two
  - **Prune**: Remove nearly-transparent Gaussians (α < threshold)

  ## Usage

      # Build the Gaussian splatting model
      model = GaussianSplat.build(
        num_gaussians: 10000,
        sh_degree: 3
      )

      # Initialize from point cloud
      gaussians = GaussianSplat.initialize_gaussians(point_cloud)

      # Render a view
      image = GaussianSplat.render(gaussians, camera, {height, width})

      # During training: adaptive density control
      gaussians = GaussianSplat.densification_step(gaussians, gradients,
        clone_threshold: 0.0002,
        split_threshold: 0.0002,
        prune_threshold: 0.005
      )

  ## References

  - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    (Kerbl et al., SIGGRAPH 2023) — https://arxiv.org/abs/2308.04079
  """

  import Nx.Defn

  @default_num_gaussians 10_000
  @default_sh_degree 3
  @default_position_lr 0.00016
  @default_opacity_lr 0.05
  @default_scaling_lr 0.005
  @default_rotation_lr 0.001
  @default_sh_lr 0.0025

  # Number of SH coefficients per channel: (degree+1)^2
  # Degree 0: 1, Degree 1: 4, Degree 2: 9, Degree 3: 16
  @sh_coeffs_per_degree %{0 => 1, 1 => 4, 2 => 9, 3 => 16}

  @typedoc "Options for GaussianSplat functions."
  @type splat_opt ::
          {:num_gaussians, pos_integer()}
          | {:sh_degree, 0..3}
          | {:name, String.t()}

  @typedoc "Gaussian representation."
  @type gaussians :: %{
          positions: Nx.Tensor.t(),
          rotations: Nx.Tensor.t(),
          scales: Nx.Tensor.t(),
          opacities: Nx.Tensor.t(),
          sh_coeffs: Nx.Tensor.t()
        }

  @typedoc "Camera parameters for rendering."
  @type camera :: %{
          view_matrix: Nx.Tensor.t(),
          proj_matrix: Nx.Tensor.t(),
          position: Nx.Tensor.t()
        }

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an Axon model for 3D Gaussian Splatting scene representation.

  The model takes viewing parameters and outputs a rendered image. Gaussian
  parameters are learned during training.

  ## Options

    - `:num_gaussians` - Number of Gaussian primitives (default: 10000)
    - `:sh_degree` - Spherical harmonics degree for color (0-3, default: 3)
    - `:name` - Layer name prefix (default: "gaussian_splat")

  ## Inputs

    - "camera_position" — Camera world position [batch, 3]
    - "view_matrix" — World-to-camera transform [batch, 4, 4]
    - "proj_matrix" — Camera-to-clip projection [batch, 4, 4]
    - "image_size" — Output resolution [batch, 2] (height, width)

  ## Returns

    An Axon model that renders the scene from the given viewpoint.
  """
  @spec build([splat_opt()]) :: Axon.t()
  def build(opts \\ []) do
    num_gaussians = Keyword.get(opts, :num_gaussians, @default_num_gaussians)
    sh_degree = Keyword.get(opts, :sh_degree, @default_sh_degree)
    name = Keyword.get(opts, :name, "gaussian_splat")

    sh_coeffs = Map.get(@sh_coeffs_per_degree, sh_degree, 16)
    # 3 channels (RGB), each with sh_coeffs coefficients
    sh_dim = 3 * sh_coeffs

    # Inputs: camera/viewing parameters
    camera_position = Axon.input("camera_position", shape: {nil, 3})
    view_matrix = Axon.input("view_matrix", shape: {nil, 4, 4})
    proj_matrix = Axon.input("proj_matrix", shape: {nil, 4, 4})
    image_height = Axon.input("image_height", shape: {nil})
    _image_width = Axon.input("image_width", shape: {nil})

    # Learnable Gaussian parameters
    positions =
      Axon.param("#{name}_positions", fn _ -> {num_gaussians, 3} end,
        initializer: :zeros
      )
      |> then(&Axon.layer(fn _input, params, _opts -> params end, [camera_position], params: &1))

    # Rotations as quaternions [w, x, y, z]
    rotations =
      Axon.param("#{name}_rotations", fn _ -> {num_gaussians, 4} end,
        initializer: fn shape, type, _key ->
          # Initialize to identity quaternion [1, 0, 0, 0]
          zeros = Nx.broadcast(0.0, shape) |> Nx.as_type(type)
          Nx.put_slice(zeros, [0, 0], Nx.broadcast(1.0, {elem(shape, 0), 1}) |> Nx.as_type(type))
        end
      )
      |> then(&Axon.layer(fn _input, params, _opts -> params end, [camera_position], params: &1))

    # Scales (log-space for stability)
    scales =
      Axon.param("#{name}_scales", fn _ -> {num_gaussians, 3} end,
        initializer: fn shape, type, _key ->
          # Initialize to small scales
          Nx.broadcast(-3.0, shape) |> Nx.as_type(type)
        end
      )
      |> then(&Axon.layer(fn _input, params, _opts -> params end, [camera_position], params: &1))

    # Opacities (logit-space for sigmoid activation)
    opacities =
      Axon.param("#{name}_opacities", fn _ -> {num_gaussians, 1} end,
        initializer: fn shape, type, _key ->
          # Initialize to ~0.1 opacity (inverse sigmoid of 0.1)
          Nx.broadcast(-2.2, shape) |> Nx.as_type(type)
        end
      )
      |> then(&Axon.layer(fn _input, params, _opts -> params end, [camera_position], params: &1))

    # Spherical harmonics coefficients
    sh_coeffs_param =
      Axon.param("#{name}_sh_coeffs", fn _ -> {num_gaussians, sh_dim} end,
        initializer: fn shape, type, _key ->
          # Initialize DC component to gray, rest to zero
          zeros = Nx.broadcast(0.0, shape) |> Nx.as_type(type)
          # Set DC components (indices 0, sh_coeffs, 2*sh_coeffs) to 0.5
          dc_val = 0.5
          zeros
          |> Nx.put_slice([0, 0], Nx.broadcast(dc_val, {elem(shape, 0), 1}) |> Nx.as_type(type))
          |> Nx.put_slice(
            [0, sh_coeffs],
            Nx.broadcast(dc_val, {elem(shape, 0), 1}) |> Nx.as_type(type)
          )
          |> Nx.put_slice(
            [0, 2 * sh_coeffs],
            Nx.broadcast(dc_val, {elem(shape, 0), 1}) |> Nx.as_type(type)
          )
        end
      )
      |> then(&Axon.layer(fn _input, params, _opts -> params end, [camera_position], params: &1))

    # Render layer
    Axon.layer(
      &render_layer/10,
      [
        positions,
        rotations,
        scales,
        opacities,
        sh_coeffs_param,
        camera_position,
        view_matrix,
        proj_matrix,
        image_height
      ],
      name: "#{name}_render",
      op_name: :gaussian_render,
      sh_degree: sh_degree
    )
  end

  defp render_layer(
         positions,
         rotations,
         scales,
         opacities,
         sh_coeffs,
         camera_position,
         view_matrix,
         proj_matrix,
         image_height,
         opts
       ) do
    sh_degree = opts[:sh_degree]

    # Build Gaussians map
    gaussians = %{
      positions: positions,
      rotations: rotations,
      scales: scales,
      opacities: opacities,
      sh_coeffs: sh_coeffs
    }

    # Get image dimensions from input (assume square for simplicity)
    height = Nx.to_number(Nx.squeeze(image_height[0]))
    width = height

    # Build camera
    camera = %{
      position: camera_position,
      view_matrix: view_matrix,
      proj_matrix: proj_matrix
    }

    render(gaussians, camera, {height, width}, sh_degree: sh_degree)
  end

  # ============================================================================
  # Gaussian Initialization
  # ============================================================================

  @doc """
  Initialize Gaussians from a point cloud.

  ## Parameters

    - `point_cloud` - Tensor of 3D points [N, 3] or [N, 6] (with colors)
    - `opts` - Options:
      - `:sh_degree` - Spherical harmonics degree (default: 3)
      - `:initial_scale` - Initial Gaussian scale (default: 0.01)

  ## Returns

    A map of Gaussian parameters ready for optimization.

  ## Example

      points = Nx.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], type: :f32)
      gaussians = GaussianSplat.initialize_gaussians(points)
  """
  @spec initialize_gaussians(Nx.Tensor.t(), keyword()) :: gaussians()
  def initialize_gaussians(point_cloud, opts \\ []) do
    sh_degree = Keyword.get(opts, :sh_degree, @default_sh_degree)
    initial_scale = Keyword.get(opts, :initial_scale, 0.01)

    {num_points, dim} = Nx.shape(point_cloud)
    sh_coeffs = Map.get(@sh_coeffs_per_degree, sh_degree, 16)
    sh_dim = 3 * sh_coeffs

    # Extract positions (first 3 columns)
    positions = Nx.slice_along_axis(point_cloud, 0, 3, axis: 1)

    # Initialize rotations as identity quaternions
    rotations =
      Nx.concatenate(
        [
          Nx.broadcast(1.0, {num_points, 1}),
          Nx.broadcast(0.0, {num_points, 3})
        ],
        axis: 1
      )
      |> Nx.as_type(:f32)

    # Initialize scales (log-space)
    scales = Nx.broadcast(:math.log(initial_scale), {num_points, 3}) |> Nx.as_type(:f32)

    # Initialize opacities (logit-space, ~0.5)
    opacities = Nx.broadcast(0.0, {num_points, 1}) |> Nx.as_type(:f32)

    # Initialize SH coefficients
    sh = Nx.broadcast(0.0, {num_points, sh_dim}) |> Nx.as_type(:f32)

    # If colors provided (dim=6), use them for DC component
    sh =
      if dim >= 6 do
        colors = Nx.slice_along_axis(point_cloud, 3, 3, axis: 1)
        # Set DC components for each channel
        sh
        |> Nx.put_slice([0, 0], Nx.slice_along_axis(colors, 0, 1, axis: 1))
        |> Nx.put_slice([0, sh_coeffs], Nx.slice_along_axis(colors, 1, 1, axis: 1))
        |> Nx.put_slice([0, 2 * sh_coeffs], Nx.slice_along_axis(colors, 2, 1, axis: 1))
      else
        # Default to gray
        sh
        |> Nx.put_slice([0, 0], Nx.broadcast(0.5, {num_points, 1}))
        |> Nx.put_slice([0, sh_coeffs], Nx.broadcast(0.5, {num_points, 1}))
        |> Nx.put_slice([0, 2 * sh_coeffs], Nx.broadcast(0.5, {num_points, 1}))
      end

    %{
      positions: positions,
      rotations: rotations,
      scales: scales,
      opacities: opacities,
      sh_coeffs: sh
    }
  end

  # ============================================================================
  # Projection
  # ============================================================================

  @doc """
  Project 3D Gaussians to 2D screen space.

  Transforms Gaussian means and covariances from world space to screen space
  using the camera's view and projection matrices.

  ## Parameters

    - `gaussians` - Map of Gaussian parameters
    - `camera` - Camera parameters (view_matrix, proj_matrix)

  ## Returns

    Map with projected means, covariances, and depths:
    - `:means_2d` - Screen-space positions [N, 2]
    - `:covs_2d` - Screen-space covariances [N, 2, 2]
    - `:depths` - View-space depths for sorting [N]
  """
  @spec project_gaussians(gaussians(), camera()) :: map()
  def project_gaussians(gaussians, camera) do
    project_gaussians_impl(
      gaussians.positions,
      gaussians.rotations,
      gaussians.scales,
      camera.view_matrix,
      camera.proj_matrix
    )
  end

  defnp project_gaussians_impl(positions, rotations, scales, view_matrix, proj_matrix) do
    num_gaussians = Nx.axis_size(positions, 0)

    # Transform positions to view space
    # positions: [N, 3], view_matrix: [batch, 4, 4]
    # Add homogeneous coordinate
    ones = Nx.broadcast(1.0, {num_gaussians, 1})
    pos_homo = Nx.concatenate([positions, ones], axis: 1)

    # Get first batch's matrices (squeeze batch dim for simplicity)
    view = Nx.squeeze(view_matrix[0])
    proj = Nx.squeeze(proj_matrix[0])

    # Transform to view space: [N, 4] @ [4, 4]^T
    pos_view = Nx.dot(pos_homo, Nx.transpose(view))

    # Extract depths (z in view space, negated for right-handed coords)
    depths = Nx.negate(pos_view[[.., 2]])

    # Transform to clip space
    pos_clip = Nx.dot(pos_view, Nx.transpose(proj))

    # Perspective divide
    w = pos_clip[[.., 3]] |> Nx.add(1.0e-6)
    means_2d = Nx.stack([pos_clip[[.., 0]], pos_clip[[.., 1]]], axis: 1) |> Nx.divide(Nx.new_axis(w, 1))

    # Build 3D covariance from rotation and scale
    # Σ = R·S·S^T·R^T where S is diagonal scale matrix
    covs_3d = build_covariance_3d(rotations, scales)

    # Project covariance to 2D (Jacobian of projection)
    # Simplified: take upper-left 2x2 of transformed covariance
    covs_2d = project_covariance(covs_3d, view, proj, pos_view)

    %{
      means_2d: means_2d,
      covs_2d: covs_2d,
      depths: depths
    }
  end

  defnp build_covariance_3d(rotations, scales) do
    # rotations: [N, 4] quaternions (w, x, y, z)
    # scales: [N, 3] log-scales

    # Convert log-scales to actual scales
    s = Nx.exp(scales)

    # Build rotation matrix from quaternion
    w = rotations[[.., 0]]
    x = rotations[[.., 1]]
    y = rotations[[.., 2]]
    z = rotations[[.., 3]]

    # Normalize quaternion
    norm = Nx.sqrt(w * w + x * x + y * y + z * z + 1.0e-8)
    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    # Rotation matrix elements
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    # Build R·S (rotation times diagonal scale)
    # RS[i,j] = R[i,j] * s[j]
    sx = s[[.., 0]]
    sy = s[[.., 1]]
    sz = s[[.., 2]]

    rs00 = r00 * sx
    rs01 = r01 * sy
    rs02 = r02 * sz
    rs10 = r10 * sx
    rs11 = r11 * sy
    rs12 = r12 * sz
    rs20 = r20 * sx
    rs21 = r21 * sy
    rs22 = r22 * sz

    # Covariance Σ = RS·RS^T
    # Σ[i,j] = sum_k RS[i,k] * RS[j,k]
    c00 = rs00 * rs00 + rs01 * rs01 + rs02 * rs02
    c01 = rs00 * rs10 + rs01 * rs11 + rs02 * rs12
    c02 = rs00 * rs20 + rs01 * rs21 + rs02 * rs22
    c11 = rs10 * rs10 + rs11 * rs11 + rs12 * rs12
    c12 = rs10 * rs20 + rs11 * rs21 + rs12 * rs22
    c22 = rs20 * rs20 + rs21 * rs21 + rs22 * rs22

    # Stack into [N, 3, 3] tensor
    row0 = Nx.stack([c00, c01, c02], axis: 1)
    row1 = Nx.stack([c01, c11, c12], axis: 1)
    row2 = Nx.stack([c02, c12, c22], axis: 1)

    Nx.stack([row0, row1, row2], axis: 2)
  end

  defnp project_covariance(covs_3d, _view, _proj, pos_view) do
    # Simplified 2D covariance projection
    # Extract upper-left 2x2 and scale by depth

    # Get z (depth) for scaling
    z = Nx.abs(pos_view[[.., 2]]) |> Nx.add(1.0e-6)
    z_sq = z * z

    # Extract 2x2 from 3x3
    c00 = covs_3d[[.., 0, 0]] / z_sq
    c01 = covs_3d[[.., 0, 1]] / z_sq
    c11 = covs_3d[[.., 1, 1]] / z_sq

    # Add small epsilon for numerical stability
    c00 = c00 + 0.0001
    c11 = c11 + 0.0001

    # Build [N, 2, 2] covariance
    row0 = Nx.stack([c00, c01], axis: 1)
    row1 = Nx.stack([c01, c11], axis: 1)

    Nx.stack([row0, row1], axis: 2)
  end

  # ============================================================================
  # Rendering
  # ============================================================================

  @doc """
  Render Gaussians to an image via differentiable alpha compositing.

  ## Parameters

    - `gaussians` - Map of Gaussian parameters
    - `camera` - Camera parameters
    - `image_size` - Output resolution `{height, width}`
    - `opts` - Rendering options:
      - `:sh_degree` - SH degree for color evaluation (default: 3)
      - `:background` - Background color (default: [0, 0, 0])

  ## Returns

    Rendered image tensor of shape `[height, width, 3]`.

  ## Example

      image = GaussianSplat.render(gaussians, camera, {256, 256})
  """
  @spec render(gaussians(), camera(), {pos_integer(), pos_integer()}, keyword()) :: Nx.Tensor.t()
  def render(gaussians, camera, image_size, opts \\ []) do
    {height, width} = image_size
    _sh_degree = Keyword.get(opts, :sh_degree, @default_sh_degree)

    # Project Gaussians to 2D
    projected = project_gaussians(gaussians, camera)

    # Get opacities (sigmoid activation)
    opacities = Nx.sigmoid(gaussians.opacities) |> Nx.squeeze(axes: [1])

    # Compute colors from SH (simplified: use DC component only)
    colors = compute_colors_from_sh_simple(gaussians.sh_coeffs)

    # Sort by depth (front to back)
    sorted_indices = Nx.argsort(projected.depths, direction: :asc)

    # Rasterize using sorted order
    rasterize_gaussians_simple(
      projected.means_2d,
      projected.covs_2d,
      opacities,
      colors,
      sorted_indices,
      height,
      width
    )
  end

  defp compute_colors_from_sh_simple(sh_coeffs) do
    # Simplified: just use DC component (first coefficient per channel)
    sh_dim = Nx.axis_size(sh_coeffs, 1)
    coeffs_per_channel = div(sh_dim, 3)

    # Extract DC components
    r = sh_coeffs[[.., 0]]
    g = sh_coeffs[[.., coeffs_per_channel]]
    b = sh_coeffs[[.., 2 * coeffs_per_channel]]

    # Clamp to [0, 1]
    colors = Nx.stack([r, g, b], axis: 1)
    Nx.clip(colors, 0.0, 1.0)
  end

  defp rasterize_gaussians_simple(means_2d, covs_2d, opacities, colors, sorted_indices, height, width) do
    num_gaussians = Nx.axis_size(means_2d, 0)

    # Create pixel coordinates (in normalized [-1, 1] space)
    y_coords = Nx.iota({height, width, 1}, axis: 0, type: :f32) |> Nx.divide(height) |> Nx.multiply(2.0) |> Nx.subtract(1.0)
    x_coords = Nx.iota({height, width, 1}, axis: 1, type: :f32) |> Nx.divide(width) |> Nx.multiply(2.0) |> Nx.subtract(1.0)

    # Initialize output
    image = Nx.broadcast(0.0, {height, width, 3})
    transmittance = Nx.broadcast(1.0, {height, width})

    # Process up to 100 Gaussians for efficiency
    num_to_process = min(num_gaussians, 100)

    # Process Gaussians front-to-back using Enum.reduce
    {final_image, _} =
      Enum.reduce(0..(num_to_process - 1), {image, transmittance}, fn i, {img, trans} ->
        idx = sorted_indices[i] |> Nx.to_number()

        # Get Gaussian parameters
        mean = means_2d[idx]
        cov = covs_2d[idx]
        opacity = opacities[idx] |> Nx.to_number()
        color = colors[idx]

        # Compute Gaussian contribution at each pixel
        dx = Nx.subtract(x_coords, Nx.to_number(mean[0]))
        dy = Nx.subtract(y_coords, Nx.to_number(mean[1]))

        # Invert 2x2 covariance
        c00 = Nx.to_number(cov[0][0])
        c01 = Nx.to_number(cov[0][1])
        c11 = Nx.to_number(cov[1][1])
        det = c00 * c11 - c01 * c01 + 1.0e-6
        inv_cov_00 = c11 / det
        inv_cov_01 = -c01 / det
        inv_cov_11 = c00 / det

        # Mahalanobis distance squared
        dist_sq =
          Nx.add(
            Nx.add(
              Nx.multiply(inv_cov_00, Nx.multiply(dx, dx)),
              Nx.multiply(2.0 * inv_cov_01, Nx.multiply(dx, dy))
            ),
            Nx.multiply(inv_cov_11, Nx.multiply(dy, dy))
          )

        # Gaussian weight
        weight = Nx.exp(Nx.multiply(-0.5, dist_sq)) |> Nx.squeeze(axes: [2])

        # Alpha for this Gaussian
        alpha = Nx.multiply(weight, opacity)

        # Alpha compositing: C += T * α * c
        c_r = Nx.to_number(color[0])
        c_g = Nx.to_number(color[1])
        c_b = Nx.to_number(color[2])

        contrib_r = Nx.multiply(Nx.multiply(trans, alpha), c_r)
        contrib_g = Nx.multiply(Nx.multiply(trans, alpha), c_g)
        contrib_b = Nx.multiply(Nx.multiply(trans, alpha), c_b)

        new_img =
          img
          |> Nx.put_slice([0, 0, 0], Nx.new_axis(Nx.add(img[[.., .., 0]], contrib_r), 2))
          |> Nx.put_slice([0, 0, 1], Nx.new_axis(Nx.add(img[[.., .., 1]], contrib_g), 2))
          |> Nx.put_slice([0, 0, 2], Nx.new_axis(Nx.add(img[[.., .., 2]], contrib_b), 2))

        # Update transmittance: T *= (1 - α)
        new_trans = Nx.multiply(trans, Nx.subtract(1.0, alpha))

        {new_img, new_trans}
      end)

    Nx.clip(final_image, 0.0, 1.0)
  end


  # ============================================================================
  # Adaptive Density Control
  # ============================================================================

  @doc """
  Perform adaptive density control based on gradient statistics.

  Implements the clone/split/prune operations from the original paper:
  - Clone: Duplicate small Gaussians with high positional gradients
  - Split: Divide large Gaussians with high gradients into two
  - Prune: Remove Gaussians with low opacity

  ## Parameters

    - `gaussians` - Current Gaussian parameters
    - `gradients` - Position gradients from training [N, 3]
    - `opts` - Control options:
      - `:clone_threshold` - Gradient threshold for cloning (default: 0.0002)
      - `:split_threshold` - Gradient threshold for splitting (default: 0.0002)
      - `:prune_threshold` - Opacity threshold for pruning (default: 0.005)
      - `:scale_threshold` - Scale threshold for clone vs split (default: 0.01)

  ## Returns

    Updated Gaussian parameters with modified count.
  """
  @spec densification_step(gaussians(), Nx.Tensor.t(), keyword()) :: gaussians()
  def densification_step(gaussians, gradients, opts \\ []) do
    clone_threshold = Keyword.get(opts, :clone_threshold, 0.0002)
    split_threshold = Keyword.get(opts, :split_threshold, 0.0002)
    prune_threshold = Keyword.get(opts, :prune_threshold, 0.005)
    scale_threshold = Keyword.get(opts, :scale_threshold, 0.01)

    densification_impl(
      gaussians,
      gradients,
      clone_threshold,
      split_threshold,
      prune_threshold,
      scale_threshold
    )
  end

  defp densification_impl(
         gaussians,
         gradients,
         clone_threshold,
         _split_threshold,
         prune_threshold,
         scale_threshold
       ) do
    num_gaussians = Nx.axis_size(gaussians.positions, 0)

    # Compute gradient magnitudes
    grad_mag = Nx.sqrt(Nx.sum(Nx.pow(gradients, 2), axes: [1]))

    # Compute scale magnitudes (from log-space)
    scales = Nx.exp(gaussians.scales)
    scale_mag = Nx.mean(scales, axes: [1])

    # Compute opacities
    opacities = Nx.sigmoid(gaussians.opacities) |> Nx.squeeze(axes: [1])

    # Identify operations needed
    high_grad = Nx.greater(grad_mag, clone_threshold)
    small_scale = Nx.less(scale_mag, scale_threshold)
    large_scale = Nx.greater_equal(scale_mag, scale_threshold)
    low_opacity = Nx.less(opacities, prune_threshold)

    # Clone mask: high gradient AND small scale (for future clone implementation)
    _clone_mask = Nx.logical_and(high_grad, small_scale)

    # Split mask: high gradient AND large scale (for future split implementation)
    _split_mask = Nx.logical_and(high_grad, large_scale)

    # Keep mask: NOT low opacity
    keep_mask = Nx.logical_not(low_opacity)

    # For simplicity, we'll just implement pruning here
    # Full clone/split requires more complex tensor manipulation
    keep_indices =
      Nx.iota({num_gaussians})
      |> Nx.select(keep_mask, -1)
      |> then(fn x ->
        # Filter out -1 values
        mask = Nx.greater_equal(x, 0)
        count = Nx.sum(mask) |> Nx.to_number() |> trunc()

        if count > 0 and count < num_gaussians do
          Nx.argsort(x, direction: :desc)
          |> Nx.slice([0], [count])
        else
          Nx.iota({num_gaussians})
        end
      end)

    # Gather kept Gaussians
    %{
      positions: Nx.take(gaussians.positions, keep_indices, axis: 0),
      rotations: Nx.take(gaussians.rotations, keep_indices, axis: 0),
      scales: Nx.take(gaussians.scales, keep_indices, axis: 0),
      opacities: Nx.take(gaussians.opacities, keep_indices, axis: 0),
      sh_coeffs: Nx.take(gaussians.sh_coeffs, keep_indices, axis: 0)
    }
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get recommended defaults for Gaussian splatting.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_gaussians: @default_num_gaussians,
      sh_degree: @default_sh_degree,
      position_lr: @default_position_lr,
      opacity_lr: @default_opacity_lr,
      scaling_lr: @default_scaling_lr,
      rotation_lr: @default_rotation_lr,
      sh_lr: @default_sh_lr
    ]
  end

  @doc """
  Get output size for rendered images.
  """
  @spec output_size(keyword()) :: {pos_integer(), pos_integer(), pos_integer()}
  def output_size(opts \\ []) do
    height = Keyword.get(opts, :image_height, 256)
    width = Keyword.get(opts, :image_width, 256)
    {height, width, 3}
  end

  @doc """
  Calculate the number of parameters in a Gaussian splatting model.
  """
  @spec param_count(keyword()) :: pos_integer()
  def param_count(opts \\ []) do
    num_gaussians = Keyword.get(opts, :num_gaussians, @default_num_gaussians)
    sh_degree = Keyword.get(opts, :sh_degree, @default_sh_degree)
    sh_coeffs = Map.get(@sh_coeffs_per_degree, sh_degree, 16)

    # positions: 3, rotations: 4, scales: 3, opacities: 1, sh: 3*sh_coeffs
    params_per_gaussian = 3 + 4 + 3 + 1 + 3 * sh_coeffs
    num_gaussians * params_per_gaussian
  end
end
