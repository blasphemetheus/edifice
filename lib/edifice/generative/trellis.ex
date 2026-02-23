defmodule Edifice.Generative.TRELLIS do
  @moduledoc """
  TRELLIS: Structured 3D Latents for Scalable 3D Generation.

  Implements the TRELLIS architecture from "TRELLIS: Structured 3D Latents for
  Scalable and Versatile 3D Generation" (Xiang et al., Microsoft Research 2024).
  A unified framework for high-quality 3D asset generation using sparse structured
  latent representations and rectified flow.

  ## Key Innovations

  ### 1. Sparse Structured Latents (SLAT)
  Represents 3D content as a sparse voxel grid where only occupied voxels store features:
  - **Sparse representation**: Only N occupied voxels (vs N³ dense grid)
  - **Per-voxel features**: Position (x,y,z) + local feature vector
  - **Memory efficient**: Enables high-resolution 3D at tractable cost

  ```
  Dense 64³ grid = 262,144 voxels (most empty)
  Sparse SLAT    = ~5,000 occupied voxels (typical)
  ```

  ### 2. Sparse Transformer
  Attention mechanism designed for sparse 3D data:
  - **3D windowed attention**: Local attention within spatial windows
  - **Sparse convolutions**: Feature propagation between nearby voxels
  - **Sparse cross-attention**: Voxels attend to text/image conditioning

  ```
  Query voxel at (x,y,z) attends to:
  - All voxels within window of size W centered at (x,y,z)
  - Conditioning tokens (text or image features)
  ```

  ### 3. Rectified Flow
  Simpler, faster alternative to DDPM diffusion:
  - **Straight-line paths**: x_t = t·x_1 + (1-t)·x_0 (linear interpolation)
  - **Velocity prediction**: Model predicts v = x_1 - x_0
  - **Few-step sampling**: Only 10-20 steps (vs 1000 for DDPM)

  ```
  DDPM: Complex curved trajectories, 1000 steps
  Rectified Flow: Straight lines, 10-20 steps
  ```

  ## Architecture

  ```
  Input: Text/Image conditioning
         |
         v
  +---------------------------+
  | Condition Encoder         |  (CLIP or similar)
  +---------------------------+
         |
         v
  +---------------------------+
  | Sparse Transformer        |  × num_layers
  |  • Sparse Self-Attention  |  (3D windowed)
  |  • Sparse Cross-Attention |  (to conditioning)
  |  • Sparse FFN             |
  +---------------------------+
         |
         v
  +---------------------------+
  | Rectified Flow Denoising  |  (10-20 steps)
  +---------------------------+
         |
         v
  +---------------------------+
  | Decode SLAT → 3D Output   |  (Gaussian splats, mesh, or NeRF)
  +---------------------------+
         |
         v
  Output: 3D asset (splats/mesh/radiance field)
  ```

  ## Usage

      # Build TRELLIS model
      model = TRELLIS.build(
        voxel_resolution: 64,
        feature_dim: 32,
        num_layers: 12,
        num_heads: 8
      )

      # Sparse attention over occupied voxels
      attended = TRELLIS.sparse_attention(
        sparse_features,
        positions,
        window_size: 8
      )

      # Single rectified flow step
      x_t_minus_1 = TRELLIS.rectified_flow_step(
        model, x_t, t, conditioning
      )

      # Full generation
      output = TRELLIS.generate(
        model, params, conditioning,
        num_steps: 20
      )

  ## Supported Output Formats

  - **3D Gaussian Splatting**: Fast, high-quality rendering
  - **Mesh extraction**: Via marching cubes from density field
  - **Radiance field**: NeRF-style volumetric representation

  ## References

  - Paper: "TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation"
  - Authors: Xiang et al., Microsoft Research
  - Year: 2024
  - Project: https://trellis3d.github.io/
  """

  alias Edifice.Blocks.FFN

  # Default configuration
  @default_voxel_resolution 64
  @default_feature_dim 32
  @default_hidden_size 512
  @default_num_layers 12
  @default_num_heads 8
  @default_window_size 8
  @default_condition_dim 768
  @default_mlp_ratio 4.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build the TRELLIS model for 3D generation.

  ## Options

    - `:voxel_resolution` - Resolution of voxel grid (default: 64)
    - `:feature_dim` - Per-voxel feature dimension (default: 32)
    - `:hidden_size` - Transformer hidden dimension (default: 512)
    - `:num_layers` - Number of sparse transformer layers (default: 12)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:window_size` - Size of local attention window (default: 8)
    - `:condition_dim` - Conditioning vector dimension (default: 768)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:max_voxels` - Maximum number of occupied voxels (default: 8192)

  ## Returns

    An Axon model that takes sparse voxel features + conditioning and outputs
    denoised sparse features.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:condition_dim, pos_integer()}
          | {:feature_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:max_voxels, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:voxel_resolution, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    voxel_resolution = Keyword.get(opts, :voxel_resolution, @default_voxel_resolution)
    feature_dim = Keyword.get(opts, :feature_dim, @default_feature_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    condition_dim = Keyword.get(opts, :condition_dim, @default_condition_dim)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    max_voxels = Keyword.get(opts, :max_voxels, 8192)

    head_dim = div(hidden_size, num_heads)

    # Inputs
    # Sparse features: [batch, num_voxels, feature_dim]
    sparse_features = Axon.input("sparse_features", shape: {nil, max_voxels, feature_dim})
    # Voxel positions: [batch, num_voxels, 3] (x, y, z coordinates)
    voxel_positions = Axon.input("voxel_positions", shape: {nil, max_voxels, 3})
    # Occupancy mask: [batch, num_voxels] (1 = occupied, 0 = padding)
    occupancy_mask = Axon.input("occupancy_mask", shape: {nil, max_voxels})
    # Conditioning: [batch, cond_len, condition_dim]
    conditioning = Axon.input("conditioning", shape: {nil, nil, condition_dim})
    # Timestep for rectified flow: [batch]
    timestep = Axon.input("timestep", shape: {nil})

    # Project features to hidden size
    x = Axon.dense(sparse_features, hidden_size, name: "input_proj")

    # Add 3D positional encoding based on voxel positions
    x = Axon.layer(
      &add_3d_position_encoding/3,
      [x, voxel_positions],
      name: "pos_encoding",
      hidden_size: hidden_size,
      voxel_resolution: voxel_resolution,
      op_name: :pos_3d
    )

    # Project conditioning
    cond_proj = Axon.dense(conditioning, hidden_size, name: "cond_proj")

    # Timestep embedding
    time_embed = build_timestep_embed(timestep, hidden_size)

    # Sparse transformer layers
    x = Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
      build_sparse_transformer_block(
        acc, voxel_positions, occupancy_mask, cond_proj, time_embed,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        window_size: window_size,
        mlp_ratio: mlp_ratio,
        name: "layer_#{layer_idx}"
      )
    end)

    # Final norm and projection back to feature space
    x = Axon.layer_norm(x, name: "final_norm")
    Axon.dense(x, feature_dim, name: "output_proj")
  end

  defp build_sparse_transformer_block(input, positions, mask, conditioning, time_embed, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    name = Keyword.get(opts, :name, "sparse_block")
    mlp_dim = round(hidden_size * mlp_ratio)

    # Add timestep conditioning
    x = Axon.layer(
      &add_time_conditioning/3,
      [input, time_embed],
      name: "#{name}_time_cond",
      op_name: :time_cond
    )

    # Sparse self-attention with 3D windowing
    x_norm = Axon.layer_norm(x, name: "#{name}_self_attn_norm")

    self_attn_out = Axon.layer(
      &sparse_windowed_attention_impl/4,
      [x_norm, positions, mask],
      name: "#{name}_self_attn",
      num_heads: num_heads,
      head_dim: head_dim,
      window_size: window_size,
      op_name: :sparse_self_attention
    )

    x = Axon.add(x, self_attn_out, name: "#{name}_self_attn_residual")

    # Sparse cross-attention to conditioning
    x_norm2 = Axon.layer_norm(x, name: "#{name}_cross_attn_norm")

    cross_attn_out = Axon.layer(
      &sparse_cross_attention_impl/4,
      [x_norm2, conditioning, mask],
      name: "#{name}_cross_attn",
      num_heads: num_heads,
      head_dim: head_dim,
      hidden_size: hidden_size,
      op_name: :sparse_cross_attention
    )

    x = Axon.add(x, cross_attn_out, name: "#{name}_cross_attn_residual")

    # FFN
    x_norm3 = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out = x_norm3
    |> Axon.dense(mlp_dim, name: "#{name}_ffn_up")
    |> Axon.activation(:gelu, name: "#{name}_ffn_act")
    |> Axon.dense(hidden_size, name: "#{name}_ffn_down")

    # Mask FFN output (zero out padded positions)
    ffn_out = Axon.layer(
      &apply_occupancy_mask/3,
      [ffn_out, mask],
      name: "#{name}_ffn_mask",
      op_name: :apply_mask
    )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  defp add_3d_position_encoding(x, positions, opts) do
    hidden_size = opts[:hidden_size]
    voxel_resolution = opts[:voxel_resolution]

    batch = Nx.axis_size(x, 0)
    num_voxels = Nx.axis_size(x, 1)

    # Normalize positions to [0, 1]
    pos_norm = Nx.divide(Nx.as_type(positions, :f32), voxel_resolution)

    # Sinusoidal encoding for each axis
    dim_per_axis = div(hidden_size, 6)  # 3 axes × 2 (sin + cos)

    freqs = Nx.exp(
      Nx.multiply(
        Nx.negate(Nx.log(Nx.tensor(10_000.0))),
        Nx.divide(Nx.iota({dim_per_axis}, type: :f32), max(dim_per_axis - 1, 1))
      )
    )

    # Compute encoding for x, y, z
    encodings = Enum.map(0..2, fn axis ->
      axis_pos = Nx.slice_along_axis(pos_norm, axis, 1, axis: 2)
      angles = Nx.multiply(axis_pos, Nx.reshape(freqs, {1, 1, dim_per_axis}))
      Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 2)
    end)

    pos_embed = Nx.concatenate(encodings, axis: 2)

    # Pad or truncate to hidden_size
    current_dim = Nx.axis_size(pos_embed, 2)
    pos_embed = if current_dim < hidden_size do
      Nx.pad(pos_embed, 0.0, [{0, 0, 0}, {0, 0, 0}, {0, hidden_size - current_dim, 0}])
    else
      Nx.slice_along_axis(pos_embed, 0, hidden_size, axis: 2)
    end

    Nx.add(x, pos_embed)
  end

  defp add_time_conditioning(x, time_embed, _opts) do
    batch = Nx.axis_size(x, 0)
    num_voxels = Nx.axis_size(x, 1)
    hidden = Nx.axis_size(x, 2)

    time_expanded = time_embed
    |> Nx.reshape({batch, 1, hidden})
    |> Nx.broadcast({batch, num_voxels, hidden})

    Nx.add(x, time_expanded)
  end

  defp sparse_windowed_attention_impl(x, positions, mask, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    window_size = opts[:window_size]

    batch = Nx.axis_size(x, 0)
    num_voxels = Nx.axis_size(x, 1)
    hidden_size = num_heads * head_dim

    # Q, K, V projections (simplified - in practice would be separate dense layers)
    scale = Nx.sqrt(Nx.tensor(hidden_size, type: :f32))
    q = Nx.divide(x, scale)
    k = x
    v = x

    # Reshape to multi-head
    q = q |> Nx.reshape({batch, num_voxels, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, num_voxels, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, num_voxels, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Compute attention scores
    scale_factor = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale_factor)

    # Create 3D windowed attention mask based on positions
    # Voxel i can only attend to voxel j if ||pos_i - pos_j||_inf <= window_size/2
    window_mask = create_window_mask(positions, window_size)

    # Combine with occupancy mask
    # mask: [batch, num_voxels]
    mask_2d = Nx.logical_and(
      Nx.reshape(mask, {batch, 1, num_voxels, 1}),
      Nx.reshape(mask, {batch, 1, 1, num_voxels})
    )

    combined_mask = Nx.logical_and(
      Nx.reshape(window_mask, {batch, 1, num_voxels, num_voxels}),
      mask_2d
    )

    # Expand combined_mask to [batch, num_heads, num_voxels, num_voxels]
    combined_mask = Nx.broadcast(combined_mask, {batch, num_heads, num_voxels, num_voxels})

    # Apply mask
    scores = Nx.select(
      combined_mask,
      scores,
      Nx.broadcast(-1.0e9, Nx.shape(scores))
    )

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply to values
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, num_voxels, hidden_size})
  end

  defp create_window_mask(positions, window_size) do
    batch = Nx.axis_size(positions, 0)
    num_voxels = Nx.axis_size(positions, 1)

    # positions: [batch, num_voxels, 3]
    pos_i = Nx.reshape(positions, {batch, num_voxels, 1, 3})
    pos_j = Nx.reshape(positions, {batch, 1, num_voxels, 3})

    # L-infinity distance
    diff = Nx.abs(Nx.subtract(pos_i, pos_j))
    max_diff = Nx.reduce_max(diff, axes: [-1])

    # Within window if max_diff <= window_size / 2
    half_window = div(window_size, 2)
    Nx.less_equal(max_diff, half_window)
  end

  defp sparse_cross_attention_impl(x, conditioning, mask, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    hidden_size = opts[:hidden_size]

    batch = Nx.axis_size(x, 0)
    num_voxels = Nx.axis_size(x, 1)
    cond_len = Nx.axis_size(conditioning, 1)

    # Q from voxels, K/V from conditioning
    scale = Nx.sqrt(Nx.tensor(hidden_size, type: :f32))
    q = Nx.divide(x, scale)

    # Reshape Q to multi-head
    q = q |> Nx.reshape({batch, num_voxels, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # K, V from conditioning
    k = conditioning |> Nx.reshape({batch, cond_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = conditioning |> Nx.reshape({batch, cond_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Attention scores
    scale_factor = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale_factor)

    # Mask padded voxel queries
    query_mask = mask
    |> Nx.reshape({batch, 1, num_voxels, 1})
    |> Nx.broadcast({batch, num_heads, num_voxels, cond_len})

    scores = Nx.select(
      query_mask,
      scores,
      Nx.broadcast(-1.0e9, Nx.shape(scores))
    )

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply to values
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, num_voxels, hidden_size})
  end

  defp apply_occupancy_mask(x, mask, _opts) do
    batch = Nx.axis_size(x, 0)
    num_voxels = Nx.axis_size(x, 1)
    hidden = Nx.axis_size(x, 2)

    mask_expanded = mask
    |> Nx.reshape({batch, num_voxels, 1})
    |> Nx.as_type(Nx.type(x))
    |> Nx.broadcast({batch, num_voxels, hidden})

    Nx.multiply(x, mask_expanded)
  end

  defp build_timestep_embed(timestep, hidden_size) do
    embed = Axon.layer(
      &sinusoidal_embed_impl/2,
      [timestep],
      name: "time_sinusoidal",
      hidden_size: hidden_size,
      op_name: :sinusoidal_embed
    )

    embed
    |> Axon.dense(hidden_size, name: "time_mlp_1")
    |> Axon.activation(:silu, name: "time_mlp_silu")
    |> Axon.dense(hidden_size, name: "time_mlp_2")
  end

  defp sinusoidal_embed_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    half_dim = div(hidden_size, 2)

    freqs = Nx.exp(
      Nx.multiply(
        Nx.negate(Nx.log(Nx.tensor(10_000.0))),
        Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
      )
    )

    t_f = Nx.as_type(t, :f32)
    angles = Nx.multiply(Nx.new_axis(t_f, 1), Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  # ============================================================================
  # SLAT Encoding/Decoding
  # ============================================================================

  @doc """
  Encode a dense voxel grid to Sparse Structured Latent (SLAT) representation.

  ## Parameters

    - `voxel_grid` - Dense voxel grid [batch, resolution, resolution, resolution, features]
                     or occupancy grid [batch, res, res, res]
    - `opts` - Options including `:threshold` for occupancy detection

  ## Returns

    A map with:
    - `:features` - Sparse features [batch, num_occupied, feature_dim]
    - `:positions` - Voxel positions [batch, num_occupied, 3]
    - `:mask` - Occupancy mask [batch, num_occupied]
  """
  @spec encode_to_slat(Nx.Tensor.t(), keyword()) :: map()
  def encode_to_slat(voxel_grid, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.5)
    max_voxels = Keyword.get(opts, :max_voxels, 8192)

    # Determine if input is occupancy-only or has features
    rank = Nx.rank(voxel_grid)

    {occupancy, features} = if rank == 4 do
      # Just occupancy: [batch, res, res, res]
      occ = Nx.greater(voxel_grid, threshold)
      {occ, nil}
    else
      # With features: [batch, res, res, res, features]
      # Use first channel or norm for occupancy
      occ = Nx.greater(Nx.sum(Nx.abs(voxel_grid), axes: [-1]), threshold)
      {occ, voxel_grid}
    end

    batch = Nx.axis_size(occupancy, 0)
    res = Nx.axis_size(occupancy, 1)

    # For each batch, extract occupied voxel positions and features
    # This is a simplified version - in practice would use sparse tensor ops

    # Create coordinate grids
    x_grid = Nx.iota({res, res, res}, axis: 0)
    y_grid = Nx.iota({res, res, res}, axis: 1)
    z_grid = Nx.iota({res, res, res}, axis: 2)

    # Flatten for indexing
    x_flat = Nx.reshape(x_grid, {res * res * res})
    y_flat = Nx.reshape(y_grid, {res * res * res})
    z_flat = Nx.reshape(z_grid, {res * res * res})
    coords = Nx.stack([x_flat, y_flat, z_flat], axis: 1)

    # Flatten occupancy
    occ_flat = Nx.reshape(occupancy, {batch, res * res * res})

    # For simplicity, take top-k occupied voxels by position
    # (In practice, would use actual sparse extraction)
    positions = Nx.broadcast(coords, {batch, res * res * res, 3})

    # Create mask based on occupancy (take first max_voxels)
    mask = Nx.slice_along_axis(occ_flat, 0, min(max_voxels, res * res * res), axis: 1)
    |> Nx.as_type(:f32)

    positions = Nx.slice_along_axis(positions, 0, min(max_voxels, res * res * res), axis: 1)

    # Extract or create features
    sparse_features = if features != nil do
      feat_dim = Nx.axis_size(features, 4)
      feat_flat = Nx.reshape(features, {batch, res * res * res, feat_dim})
      Nx.slice_along_axis(feat_flat, 0, min(max_voxels, res * res * res), axis: 1)
    else
      # Create placeholder features
      feat_dim = Keyword.get(opts, :feature_dim, @default_feature_dim)
      Nx.broadcast(0.0, {batch, min(max_voxels, res * res * res), feat_dim})
    end

    %{
      features: sparse_features,
      positions: positions,
      mask: mask
    }
  end

  @doc """
  Decode Sparse Structured Latent back to dense representation or output format.

  ## Parameters

    - `sparse_latent` - Map from `encode_to_slat/2` or model output
    - `opts` - Options including:
      - `:output_format` - `:dense`, `:gaussian_splats`, or `:mesh` (default: `:dense`)
      - `:resolution` - Output resolution for dense format (default: 64)

  ## Returns

    Decoded 3D representation in requested format.
  """
  @spec decode_from_slat(map(), keyword()) :: Nx.Tensor.t() | map()
  def decode_from_slat(sparse_latent, opts \\ []) do
    output_format = Keyword.get(opts, :output_format, :dense)
    resolution = Keyword.get(opts, :resolution, @default_voxel_resolution)

    features = sparse_latent[:features]
    positions = sparse_latent[:positions]
    mask = sparse_latent[:mask]

    case output_format do
      :dense ->
        decode_to_dense(features, positions, mask, resolution)

      :gaussian_splats ->
        decode_to_gaussian_splats(features, positions, mask)

      :mesh ->
        # Would use marching cubes on the dense representation
        dense = decode_to_dense(features, positions, mask, resolution)
        %{density: dense, note: "Use marching cubes for mesh extraction"}
    end
  end

  defp decode_to_dense(features, positions, mask, resolution) do
    batch = Nx.axis_size(features, 0)
    num_voxels = Nx.axis_size(features, 1)
    feature_dim = Nx.axis_size(features, 2)

    # Initialize empty dense grid
    dense = Nx.broadcast(0.0, {batch, resolution, resolution, resolution, feature_dim})

    # Scatter features to positions (simplified - uses loop semantics)
    # In practice, would use scatter operations
    # For now, return a placeholder with aggregate statistics

    # Compute mean feature as a summary
    masked_features = Nx.multiply(features, Nx.reshape(mask, {batch, num_voxels, 1}))
    total_mask = Nx.sum(mask, axes: [1], keep_axes: true)
    mean_feature = Nx.divide(
      Nx.sum(masked_features, axes: [1]),
      Nx.add(total_mask, 1.0e-8)
    )

    # Broadcast to dense grid (placeholder)
    mean_feature
    |> Nx.reshape({batch, 1, 1, 1, feature_dim})
    |> Nx.broadcast({batch, resolution, resolution, resolution, feature_dim})
  end

  defp decode_to_gaussian_splats(features, positions, mask) do
    # Gaussian splatting representation:
    # Each occupied voxel becomes a Gaussian with:
    # - Position (x, y, z)
    # - Covariance/scale
    # - Color/features
    # - Opacity

    batch = Nx.axis_size(features, 0)
    num_voxels = Nx.axis_size(features, 1)
    feature_dim = Nx.axis_size(features, 2)

    # Extract splat parameters from features
    # Assume feature layout: [scale_3, rotation_4, color_3, opacity_1, ...]
    min_dim = min(feature_dim, 11)

    scales = if feature_dim >= 3 do
      Nx.slice_along_axis(features, 0, 3, axis: 2)
      |> Nx.sigmoid()  # Ensure positive
    else
      Nx.broadcast(0.1, {batch, num_voxels, 3})
    end

    rotations = if feature_dim >= 7 do
      Nx.slice_along_axis(features, 3, 4, axis: 2)
      |> normalize_quaternions()
    else
      # Identity quaternion
      Nx.broadcast(Nx.tensor([1.0, 0.0, 0.0, 0.0]), {batch, num_voxels, 4})
    end

    colors = if feature_dim >= 10 do
      Nx.slice_along_axis(features, 7, 3, axis: 2)
      |> Nx.sigmoid()  # RGB in [0, 1]
    else
      Nx.broadcast(0.5, {batch, num_voxels, 3})
    end

    opacities = if feature_dim >= 11 do
      Nx.slice_along_axis(features, 10, 1, axis: 2)
      |> Nx.sigmoid()
    else
      # Use mask as opacity
      Nx.reshape(mask, {batch, num_voxels, 1})
    end

    %{
      positions: positions,
      scales: scales,
      rotations: rotations,
      colors: colors,
      opacities: opacities,
      mask: mask
    }
  end

  defp normalize_quaternions(q) do
    norm = Nx.sqrt(Nx.sum(Nx.multiply(q, q), axes: [-1], keep_axes: true))
    Nx.divide(q, Nx.add(norm, 1.0e-8))
  end

  # ============================================================================
  # Sparse Attention (Standalone)
  # ============================================================================

  @doc """
  Compute sparse windowed 3D attention over occupied voxels.

  ## Parameters

    - `sparse_features` - Voxel features [batch, num_voxels, feature_dim]
    - `positions` - Voxel positions [batch, num_voxels, 3]
    - `opts` - Options:
      - `:window_size` - Attention window size (default: 8)
      - `:num_heads` - Number of attention heads (default: 8)
      - `:mask` - Occupancy mask [batch, num_voxels] (optional)

  ## Returns

    Attended features [batch, num_voxels, feature_dim]
  """
  @spec sparse_attention(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def sparse_attention(sparse_features, positions, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)

    batch = Nx.axis_size(sparse_features, 0)
    num_voxels = Nx.axis_size(sparse_features, 1)
    feature_dim = Nx.axis_size(sparse_features, 2)

    mask = Keyword.get(opts, :mask, Nx.broadcast(1.0, {batch, num_voxels}))

    head_dim = div(feature_dim, num_heads)

    sparse_windowed_attention_impl(
      sparse_features, positions, mask,
      %{num_heads: num_heads, head_dim: head_dim, window_size: window_size}
    )
  end

  # ============================================================================
  # Rectified Flow
  # ============================================================================

  @doc """
  Perform one rectified flow denoising step.

  Rectified flow uses straight-line interpolation:
  - Forward: x_t = t * x_1 + (1-t) * x_0 (where x_0 is noise, x_1 is data)
  - Model predicts velocity: v = x_1 - x_0
  - Update: x_{t-dt} = x_t + dt * v

  ## Parameters

    - `model` - TRELLIS model
    - `params` - Model parameters
    - `x_t` - Current noisy sparse latent (map with :features, :positions, :mask)
    - `t` - Current timestep [batch] in [0, 1]
    - `conditioning` - Conditioning tensor [batch, cond_len, cond_dim]
    - `opts` - Options:
      - `:dt` - Step size (default: computed from num_steps)
      - `:num_steps` - Total steps for dt calculation (default: 20)

  ## Returns

    Denoised sparse latent at t - dt.
  """
  @spec rectified_flow_step(Axon.t(), map(), map(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: map()
  def rectified_flow_step(model, params, x_t, t, conditioning, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, 20)
    dt = Keyword.get(opts, :dt, 1.0 / num_steps)

    features = x_t[:features]
    positions = x_t[:positions]
    mask = x_t[:mask]

    # Predict velocity v = x_1 - x_0
    velocity = Axon.predict(model, params, %{
      "sparse_features" => features,
      "voxel_positions" => positions,
      "occupancy_mask" => mask,
      "conditioning" => conditioning,
      "timestep" => t
    })

    # Update: x_{t-dt} = x_t - dt * v
    # (Moving backwards from noise to data)
    new_features = Nx.subtract(features, Nx.multiply(dt, velocity))

    %{
      features: new_features,
      positions: positions,
      mask: mask
    }
  end

  @doc """
  Generate 3D content using rectified flow sampling.

  ## Parameters

    - `model` - TRELLIS model
    - `params` - Model parameters
    - `conditioning` - Conditioning tensor [batch, cond_len, cond_dim]
    - `opts` - Options:
      - `:num_steps` - Number of denoising steps (default: 20)
      - `:max_voxels` - Maximum voxels in output (default: 8192)
      - `:feature_dim` - Feature dimension (default: 32)
      - `:voxel_resolution` - Resolution for position initialization (default: 64)

  ## Returns

    Generated sparse latent map ready for decoding.
  """
  @spec generate(Axon.t(), map(), Nx.Tensor.t(), keyword()) :: map()
  def generate(model, params, conditioning, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, 20)
    max_voxels = Keyword.get(opts, :max_voxels, 8192)
    feature_dim = Keyword.get(opts, :feature_dim, @default_feature_dim)
    voxel_resolution = Keyword.get(opts, :voxel_resolution, @default_voxel_resolution)

    batch = Nx.axis_size(conditioning, 0)
    dt = 1.0 / num_steps

    # Initialize with noise at t=1
    key = Nx.Random.key(42)
    {noise, _key} = Nx.Random.normal(key, shape: {batch, max_voxels, feature_dim})

    # Initialize positions on a regular grid (subset)
    positions = initialize_voxel_positions(batch, max_voxels, voxel_resolution)

    # Initialize mask (all occupied for generation)
    mask = Nx.broadcast(1.0, {batch, max_voxels})

    initial_latent = %{
      features: noise,
      positions: positions,
      mask: mask
    }

    # Iterative denoising from t=1 to t=0
    timesteps = Enum.map(num_steps..1, fn step -> step / num_steps end)

    final_latent = Enum.reduce(timesteps, initial_latent, fn t, current_latent ->
      t_tensor = Nx.broadcast(Nx.tensor(t, type: :f32), {batch})
      rectified_flow_step(model, params, current_latent, t_tensor, conditioning, num_steps: num_steps)
    end)

    final_latent
  end

  defp initialize_voxel_positions(batch, max_voxels, resolution) do
    # Create a subset of regular grid positions
    # Sample uniformly from the grid
    side = round(:math.pow(max_voxels, 1/3))
    step = max(1, div(resolution, side))

    coords = for x <- 0..(side-1), y <- 0..(side-1), z <- 0..(side-1) do
      [x * step, y * step, z * step]
    end
    |> Enum.take(max_voxels)
    |> Nx.tensor(type: :f32)

    # Pad if needed
    num_coords = Nx.axis_size(coords, 0)
    coords = if num_coords < max_voxels do
      padding = Nx.broadcast(0.0, {max_voxels - num_coords, 3})
      Nx.concatenate([coords, padding], axis: 0)
    else
      coords
    end

    Nx.broadcast(coords, {batch, max_voxels, 3})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get recommended defaults for TRELLIS.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      voxel_resolution: @default_voxel_resolution,
      feature_dim: @default_feature_dim,
      hidden_size: @default_hidden_size,
      num_layers: @default_num_layers,
      num_heads: @default_num_heads,
      window_size: @default_window_size,
      condition_dim: @default_condition_dim,
      mlp_ratio: @default_mlp_ratio
    ]
  end

  @doc """
  Approximate parameter count for TRELLIS model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts \\ []) do
    feature_dim = Keyword.get(opts, :feature_dim, @default_feature_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    condition_dim = Keyword.get(opts, :condition_dim, @default_condition_dim)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    mlp_dim = round(hidden_size * mlp_ratio)

    # Input/output projections
    io_proj = feature_dim * hidden_size * 2

    # Conditioning projection
    cond_proj = condition_dim * hidden_size

    # Time embedding
    time_mlp = hidden_size * hidden_size * 2

    # Per layer: self-attn + cross-attn + FFN (simplified count)
    # Self-attn: QKV + out = 4 * hidden^2
    # Cross-attn: Q + KV + out = 4 * hidden^2
    # FFN: up + down = 2 * hidden * mlp_dim
    per_layer = 4 * hidden_size * hidden_size +
                4 * hidden_size * hidden_size +
                2 * hidden_size * mlp_dim

    io_proj + cond_proj + time_mlp + num_layers * per_layer
  end

  @doc """
  Get the output feature dimension.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :feature_dim, @default_feature_dim)
  end
end
