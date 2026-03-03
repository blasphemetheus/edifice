defmodule Edifice.Recurrent.MultiTimescaleRecurrence do
  @moduledoc """
  Multi-timescale recurrence with parallel cores at different temporal strides.

  Processes input sequences through multiple recurrent cores operating at
  different temporal resolutions. Fast cores see every frame; slow cores see
  temporally subsampled (strided) sequences. A merge layer combines all
  timescale representations into a single hidden vector.

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        ├─→ Fast  (stride=1,  process every frame)     → [batch, hidden_fast]
        ├─→ Med   (stride=4,  process every 4th frame)  → [batch, hidden_med]
        └─→ Slow  (stride=16, process every 16th frame) → [batch, hidden_slow]
                    |
                    v
            Concat → Merge MLP → [batch, output_size]
  ```

  Each core is a stacked GRU. The slow cores receive fewer frames, so they
  capture longer-range patterns with less compute.

  ## Usage

      model = MultiTimescaleRecurrence.build(
        embed_dim: 288,
        scales: [
          %{stride: 1, hidden_size: 128, num_layers: 2},
          %{stride: 4, hidden_size: 128, num_layers: 2},
          %{stride: 16, hidden_size: 64, num_layers: 1}
        ],
        output_size: 256
      )

  ## Game AI Context

  Inspired by FTW (Jaderberg et al., 2019 — DeepMind's Capture-the-Flag
  agent). Game environments have extreme temporal hierarchy:

  - **Frame-level** (1/60s): tech windows, DI angles, L-cancel timing
  - **Action-level** (4-8 frames): move execution, hitbox timing
  - **Tactic-level** (30-60 frames): combo sequencing, recovery options
  - **Strategy-level** (seconds-minutes): neutral game, adaptation

  A single backbone at one temporal resolution wastes capacity modeling all
  scales simultaneously. Multi-timescale recurrence dedicates separate
  parameters to each scale.

  ## References

  - Jaderberg et al., "Human-level performance in first-person multiplayer
    games with population-based deep RL" (FTW, 2019)
  - Mujika et al., "Fast-Slow Recurrent Neural Networks" (2017)
  """

  @default_output_size 256
  @default_dropout 0.0

  @default_scales [
    %{stride: 1, hidden_size: 128, num_layers: 2},
    %{stride: 4, hidden_size: 128, num_layers: 2},
    %{stride: 16, hidden_size: 64, num_layers: 1}
  ]

  @typedoc "Configuration for a single timescale core."
  @type scale_config :: %{
          stride: pos_integer(),
          hidden_size: pos_integer(),
          num_layers: pos_integer()
        }

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:scales, [scale_config()]}
          | {:output_size, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a multi-timescale recurrence model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:scales` - List of scale configs, each with `:stride`, `:hidden_size`,
      `:num_layers` (default: fast/med/slow at strides 1/4/16)
    - `:output_size` - Output hidden dimension after merge (default: #{@default_output_size})
    - `:dropout` - Dropout rate (default: #{@default_dropout})
    - `:window_size` - Sequence length hint (default: 60)

  ## Returns

    An Axon model outputting `[batch, output_size]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    scales = Keyword.get(opts, :scales, @default_scales)
    output_size = Keyword.get(opts, :output_size, @default_output_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Build each timescale core
    core_outputs =
      scales
      |> Enum.with_index()
      |> Enum.map(fn {scale, idx} ->
        build_timescale_core(input, embed_dim, scale, dropout, idx)
      end)

    # Concatenate all core outputs → [batch, sum_of_hidden_sizes]
    merged =
      case core_outputs do
        [single] -> single
        multiple -> Axon.concatenate(multiple, axis: 1, name: "timescale_concat")
      end

    # Merge MLP → output_size
    merged
    |> Axon.dense(output_size, name: "merge_dense1")
    |> Axon.activation(:gelu)
    |> Axon.dropout(rate: dropout, name: "merge_dropout")
    |> Axon.dense(output_size, name: "merge_dense2")
    |> Axon.layer_norm(name: "merge_norm")
  end

  @doc """
  Return the output size of the model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :output_size, @default_output_size)
  end

  @doc """
  Recommended defaults for Melee (60 FPS gameplay).
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      embed_dim: 288,
      scales: [
        %{stride: 1, hidden_size: 128, num_layers: 2},
        %{stride: 4, hidden_size: 128, num_layers: 2},
        %{stride: 16, hidden_size: 64, num_layers: 1}
      ],
      output_size: 256,
      dropout: 0.1,
      window_size: 60
    ]
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp build_timescale_core(input, embed_dim, scale, dropout, core_idx) do
    stride = Map.fetch!(scale, :stride)
    hidden_size = Map.fetch!(scale, :hidden_size)
    num_layers = Map.fetch!(scale, :num_layers)
    name = "core_#{core_idx}_s#{stride}"

    # Temporal subsampling: take every `stride`-th frame
    subsampled =
      if stride == 1 do
        input
      else
        Axon.nx(input, fn x ->
          {_b, seq_len, _d} = Nx.shape(x)
          num_samples = div(seq_len, stride)
          indices = Nx.iota({num_samples}) |> Nx.multiply(stride)
          Nx.take(x, indices, axis: 1)
        end, name: "#{name}_subsample")
      end

    # Project to core hidden size if needed
    projected =
      if embed_dim != hidden_size do
        Axon.dense(subsampled, hidden_size, name: "#{name}_proj")
      else
        subsampled
      end

    # Stacked GRU layers
    processed =
      Enum.reduce(1..num_layers, projected, fn layer_idx, acc ->
        {output_seq, _hidden} =
          Axon.gru(acc, hidden_size,
            name: "#{name}_gru_#{layer_idx}",
            recurrent_initializer: :glorot_uniform,
            use_bias: true
          )

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(output_seq, rate: dropout, name: "#{name}_drop_#{layer_idx}")
        else
          output_seq
        end
      end)

    # Take last timestep → [batch, hidden_size]
    Axon.nx(processed, fn x ->
      seq_len = Nx.axis_size(x, 1)
      Nx.squeeze(Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1), axes: [1])
    end, name: "#{name}_last")
  end
end
