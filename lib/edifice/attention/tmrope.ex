defmodule Edifice.Attention.TMRoPE do
  @moduledoc """
  TMRoPE: Time-aligned Multimodal RoPE for unified position encoding across modalities.

  Extends RoPE to align different modalities (text, image, video) by their temporal
  position rather than sequence position. This enables coherent cross-modal attention
  where tokens from the same moment in time share compatible positional encodings.

  ## Key Insight

  Standard RoPE assigns positions sequentially (0, 1, 2, ...). For multimodal inputs,
  this breaks semantic alignment: an image at position 100 might represent the same
  moment as text at position 50. TMRoPE instead assigns positions based on timestamps:

  ```
  Text tokens:     "A cat"     -> positions [0.0, 0.0]  (same utterance)
  Image patches:   [64 patches] -> positions [1.0, 1.0, ..., 1.0]  (same frame)
  Video frames:    [3 frames]   -> positions [2.0, 2.5, 3.0]  (temporal sequence)
  ```

  ## Position Assignment

  ```
  Input: [text_tokens, image_patches, video_frames]
         |
  +------v-----------------------------------+
  | Modality Metadata                        |
  |   {:text,  [start: 0, end: 5]}           |  -> positions = [0.0] * 5
  |   {:image, [start: 5, end: 69, time: 1]} |  -> positions = [1.0] * 64
  |   {:video, [start: 69, end: 261,         |  -> positions = [2.0, 2.5, 3.0, ...]
  |             frame_times: [2,2.5,3,...]]} |     per-frame assignment
  +------------------------------------------+
         |
         v
  Position IDs: [0,0,0,0,0, 1,1,...,1, 2,2.5,3,...]
         |
  +------v-----------------------------------+
  | RoPE with temporal positions             |
  +------------------------------------------+
         |
         v
  Time-aligned Q, K tensors
  ```

  ## Formula

  For each token at temporal position t:
  - θᵢ = base^(-2i/d) (standard RoPE frequencies)
  - Rotation angle = t × θᵢ (using temporal position, not sequence index)
  - Optional temporal scaling: θᵢ' = θᵢ × temporal_scaling

  ## Usage

      # Build TMRoPE-wrapped attention
      model = TMRoPE.build(
        embed_dim: 64,
        modalities: [:text, :image, :video],
        temporal_scaling: 1.0
      )

      # Assign positions to multimodal sequence
      metadata = [
        {:text, [start_idx: 0, end_idx: 5, time: 0.0]},
        {:image, [start_idx: 5, end_idx: 69, time: 1.0]},
        {:video, [start_idx: 69, end_idx: 261, frame_times: [2.0, 2.5, 3.0]]}
      ]
      position_ids = TMRoPE.assign_positions(seq_len, metadata)

      # Apply TMRoPE to Q, K
      {q_rot, k_rot} = TMRoPE.apply_tmrope(q, k, position_ids)

  ## References

  - "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
    (Wang et al., 2024) — https://arxiv.org/abs/2409.12191
  - "InternVL: Scaling up Vision Foundation Models and Aligning for Generic
    Visual-Linguistic Tasks" (Chen et al., 2024) — https://arxiv.org/abs/2312.14238
  """

  import Nx.Defn

  @default_base 10_000.0
  @default_max_position 32_768
  @default_temporal_scaling 1.0

  @typedoc "Options for TMRoPE functions."
  @type tmrope_opt ::
          {:embed_dim, pos_integer()}
          | {:modalities, [:text | :image | :video | :audio]}
          | {:max_position, pos_integer()}
          | {:temporal_scaling, number()}
          | {:base, number()}
          | {:name, String.t()}

  @typedoc "Modality metadata for position assignment."
  @type modality_metadata ::
          {:text, keyword()}
          | {:image, keyword()}
          | {:video, keyword()}
          | {:audio, keyword()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an Axon model that applies TMRoPE to query/key inputs.

  ## Options

    - `:embed_dim` - Feature dimension (required, must be even)
    - `:modalities` - List of modality types (default: [:text, :image, :video])
    - `:max_position` - Maximum temporal position (default: 32768)
    - `:temporal_scaling` - Scaling factor for temporal frequencies (default: 1.0)
    - `:base` - RoPE base frequency (default: 10000.0)
    - `:name` - Layer name prefix (default: "tmrope")

  ## Inputs

  The model takes three inputs:
    - "tmrope_query" — Query tensor [batch, seq_len, embed_dim]
    - "tmrope_key" — Key tensor [batch, seq_len, embed_dim]
    - "tmrope_positions" — Position IDs [batch, seq_len] (temporal positions)

  ## Returns

    An Axon container with `{:query, :key}` rotated tensors.
  """
  @spec build([tmrope_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    temporal_scaling = Keyword.get(opts, :temporal_scaling, @default_temporal_scaling)
    base = Keyword.get(opts, :base, @default_base)
    name = Keyword.get(opts, :name, "tmrope")

    query = Axon.input("tmrope_query", shape: {nil, nil, embed_dim})
    key = Axon.input("tmrope_key", shape: {nil, nil, embed_dim})
    positions = Axon.input("tmrope_positions", shape: {nil, nil})

    output =
      Axon.layer(
        &apply_tmrope_layer/4,
        [query, key, positions],
        name: name,
        op_name: :tmrope,
        temporal_scaling: temporal_scaling,
        base: base,
        embed_dim: embed_dim
      )

    # Return container with named outputs
    Axon.container(%{
      query: Axon.nx(output, fn out -> elem(out, 0) end, name: "#{name}_query_out"),
      key: Axon.nx(output, fn out -> elem(out, 1) end, name: "#{name}_key_out")
    })
  end

  defp apply_tmrope_layer(query, key, positions, opts) do
    temporal_scaling = opts[:temporal_scaling]
    base = opts[:base]

    apply_tmrope_impl(query, key, positions, temporal_scaling, base)
  end

  # ============================================================================
  # Frequency Computation
  # ============================================================================

  @doc """
  Compute TMRoPE frequency table with temporal scaling.

  ## Parameters

    - `embed_dim` - Feature dimension (must be even)
    - `opts` - Options:
      - `:temporal_scaling` - Frequency scaling factor (default: 1.0)
      - `:base` - RoPE base frequency (default: 10000.0)

  ## Returns

    Tensor of shape `[embed_dim / 2]` with scaled frequencies.

  ## Example

      freqs = TMRoPE.tmrope_freqs(64, temporal_scaling: 0.5)
  """
  @spec tmrope_freqs(pos_integer(), keyword()) :: Nx.Tensor.t()
  def tmrope_freqs(embed_dim, opts \\ []) do
    temporal_scaling = Keyword.get(opts, :temporal_scaling, @default_temporal_scaling)
    base = Keyword.get(opts, :base, @default_base)

    half_dim = div(embed_dim, 2)

    # Base frequencies: theta_i = base^(-2i/dim)
    base_freqs =
      Nx.pow(
        base,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), embed_dim))
      )
      |> Nx.as_type(:f32)

    # Apply temporal scaling
    Nx.multiply(base_freqs, temporal_scaling)
  end

  # ============================================================================
  # Position Assignment
  # ============================================================================

  @doc """
  Assign temporal position IDs to a multimodal sequence.

  Maps token indices to time-aligned positions based on modality metadata.
  Tokens from the same temporal moment (e.g., same video frame) share positions.

  ## Parameters

    - `seq_len` - Total sequence length
    - `modality_metadata` - List of modality specifications:
      - `{:text, [start_idx: int, end_idx: int, time: float]}` — all tokens get same time
      - `{:image, [start_idx: int, end_idx: int, time: float]}` — all patches get same time
      - `{:video, [start_idx: int, end_idx: int, patches_per_frame: int, frame_times: [float]]}` —
        patches grouped by frame

  ## Returns

    Tensor of shape `[seq_len]` with temporal position IDs.

  ## Example

      metadata = [
        {:text, [start_idx: 0, end_idx: 10, time: 0.0]},
        {:image, [start_idx: 10, end_idx: 74, time: 1.0]},
        {:video, [start_idx: 74, end_idx: 266, patches_per_frame: 64, frame_times: [2.0, 3.0, 4.0]]}
      ]
      positions = TMRoPE.assign_positions(266, metadata)
  """
  @spec assign_positions(pos_integer(), [modality_metadata()]) :: Nx.Tensor.t()
  def assign_positions(seq_len, modality_metadata) do
    # Initialize with zeros
    positions = List.duplicate(0.0, seq_len)

    # Fill in positions based on metadata
    positions =
      Enum.reduce(modality_metadata, positions, fn {modality, opts}, acc ->
        start_idx = Keyword.fetch!(opts, :start_idx)
        end_idx = Keyword.fetch!(opts, :end_idx)

        case modality do
          :text ->
            time = Keyword.get(opts, :time, 0.0)
            fill_range(acc, start_idx, end_idx, time)

          :image ->
            time = Keyword.get(opts, :time, 0.0)
            fill_range(acc, start_idx, end_idx, time)

          :audio ->
            time = Keyword.get(opts, :time, 0.0)
            fill_range(acc, start_idx, end_idx, time)

          :video ->
            patches_per_frame = Keyword.get(opts, :patches_per_frame, 64)
            frame_times = Keyword.fetch!(opts, :frame_times)
            fill_video_positions(acc, start_idx, end_idx, patches_per_frame, frame_times)
        end
      end)

    Nx.tensor(positions, type: :f32)
  end

  defp fill_range(positions, start_idx, end_idx, value) do
    Enum.with_index(positions, fn
      _v, idx when idx >= start_idx and idx < end_idx -> value
      v, _idx -> v
    end)
  end

  defp fill_video_positions(positions, start_idx, end_idx, patches_per_frame, frame_times) do
    num_frames = length(frame_times)

    # Assign each patch to its frame's time
    Enum.with_index(positions, fn v, idx ->
      if idx >= start_idx and idx < end_idx do
        patch_offset = idx - start_idx
        frame_idx = min(div(patch_offset, patches_per_frame), num_frames - 1)
        Enum.at(frame_times, frame_idx, 0.0)
      else
        v
      end
    end)
  end

  # ============================================================================
  # Application Functions
  # ============================================================================

  @doc """
  Apply TMRoPE to query and key tensors using temporal position IDs.

  ## Parameters

    - `query` - Query tensor [batch, seq_len, embed_dim]
    - `key` - Key tensor [batch, seq_len, embed_dim]
    - `position_ids` - Temporal positions [batch, seq_len] or [seq_len]
    - `opts` - Options:
      - `:temporal_scaling` - Frequency scaling (default: 1.0)
      - `:base` - RoPE base frequency (default: 10000.0)

  ## Returns

    `{rotated_query, rotated_key}` with same shapes as input.

  ## Example

      positions = TMRoPE.assign_positions(seq_len, metadata)
      {q_rot, k_rot} = TMRoPE.apply_tmrope(q, k, positions)
  """
  @spec apply_tmrope(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def apply_tmrope(query, key, position_ids, opts \\ []) do
    temporal_scaling = Keyword.get(opts, :temporal_scaling, @default_temporal_scaling)
    base = Keyword.get(opts, :base, @default_base)

    apply_tmrope_impl(query, key, position_ids, temporal_scaling, base)
  end

  defnp apply_tmrope_impl(query, key, position_ids, temporal_scaling, base) do
    embed_dim = Nx.axis_size(query, 2)
    half_dim = div(embed_dim, 2)

    # Compute scaled frequencies: theta_i = base^(-2i/dim) * temporal_scaling
    base_freqs =
      Nx.pow(
        base,
        Nx.negate(Nx.divide(Nx.multiply(2.0, Nx.iota({half_dim}, type: :f32)), embed_dim))
      )

    freqs = Nx.multiply(base_freqs, temporal_scaling)

    # Handle position_ids shape: ensure [batch, seq_len]
    position_ids =
      case Nx.rank(position_ids) do
        1 -> Nx.new_axis(position_ids, 0)
        _ -> position_ids
      end

    # Compute angles: [batch, seq_len, half_dim]
    # position_ids: [batch, seq_len] -> [batch, seq_len, 1]
    # freqs: [half_dim] -> [1, 1, half_dim]
    pos_expanded = Nx.new_axis(position_ids, 2)
    freqs_expanded = Nx.reshape(freqs, {1, 1, half_dim})
    angles = Nx.multiply(pos_expanded, freqs_expanded)

    cos_table = Nx.cos(angles)
    sin_table = Nx.sin(angles)

    # Apply rotation
    q_rotated = rotate_half(query, cos_table, sin_table, half_dim)
    k_rotated = rotate_half(key, cos_table, sin_table, half_dim)

    {q_rotated, k_rotated}
  end

  defnp rotate_half(x, cos_table, sin_table, half_dim) do
    x1 = Nx.slice_along_axis(x, 0, half_dim, axis: 2)
    x2 = Nx.slice_along_axis(x, half_dim, half_dim, axis: 2)

    rotated1 = Nx.subtract(Nx.multiply(x1, cos_table), Nx.multiply(x2, sin_table))
    rotated2 = Nx.add(Nx.multiply(x1, sin_table), Nx.multiply(x2, cos_table))

    Nx.concatenate([rotated1, rotated2], axis: 2)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get recommended defaults for TMRoPE.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      modalities: [:text, :image, :video],
      max_position: @default_max_position,
      temporal_scaling: @default_temporal_scaling,
      base: @default_base
    ]
  end

  @doc """
  Calculate output size (same as input embed_dim).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts) do
    Keyword.fetch!(opts, :embed_dim)
  end

  @doc """
  Generate sequential frame times for video.

  ## Parameters

    - `num_frames` - Number of video frames
    - `opts` - Options:
      - `:start_time` - Starting time (default: 0.0)
      - `:frame_interval` - Time between frames (default: 1.0)

  ## Example

      TMRoPE.frame_times(3, start_time: 2.0, frame_interval: 0.5)
      # => [2.0, 2.5, 3.0]
  """
  @spec frame_times(pos_integer(), keyword()) :: [float()]
  def frame_times(num_frames, opts \\ []) do
    start_time = Keyword.get(opts, :start_time, 0.0)
    frame_interval = Keyword.get(opts, :frame_interval, 1.0)

    for i <- 0..(num_frames - 1) do
      start_time + i * frame_interval
    end
  end
end
