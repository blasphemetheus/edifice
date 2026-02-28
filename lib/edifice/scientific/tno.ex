defmodule Edifice.Scientific.TNO do
  @moduledoc """
  Temporal Neural Operator (TNO) for time-dependent PDEs.

  Extends DeepONet with a third **temporal branch** that encodes solution
  history, enabling prediction of time-evolving physical systems. The three
  branches are fused via element-wise (Hadamard) product and projected
  through an output MLP.

  ## Architecture

  ```
  Auxiliary Function     Solution History      Spatio-temporal Queries
  [batch, num_sensors]   [batch, history_dim]   [batch, Q, coord_dim+1]
        |                      |                       |
        v                      v                       v
  +----------------+    +------------------+    +----------------+
  | Branch Network |    | Temporal Branch  |    | Trunk Network  |
  | MLP            |    | MLP              |    | MLP per query  |
  +----------------+    +------------------+    +----------------+
        |                      |                       |
        v                      v                       v
  [batch, p]             [batch, p]              [batch, Q, p]
        |                      |                       |
        +-------- Hadamard product (element-wise) -----+
                          |
                          v
                   [batch, Q, p]
                          |
                   Output MLP (G)
                          |
                          v
              [batch, Q, output_steps * output_dim]
  ```

  ## Key Innovation

  Standard DeepONet learns G(u)(y) where u is an input function and y is
  a spatial query. TNO decomposes the temporal problem:
  - **Branch**: encodes auxiliary inputs (forcing terms, boundary conditions)
  - **Temporal branch**: encodes solution history (past states of the system)
  - **Trunk**: encodes spatio-temporal query coordinates (space + time)

  The three-way Hadamard product requires all branches to agree per latent
  dimension, giving richer interaction than a simple dot product.

  ## Usage

      model = TNO.build(
        num_sensors: 100,
        history_dim: 50,
        coord_dim: 2,
        latent_dim: 64,
        output_steps: 4
      )

  ## References
  - Diab & Al-Kobaisi, "Temporal Neural Operator for Modeling
    Time-Dependent Physical Phenomena" (Nature Sci. Reports, 2025)
  - arXiv: https://arxiv.org/abs/2504.20249
  """

  @default_branch_hidden [128, 128, 128]
  @default_temporal_hidden [128, 128, 128]
  @default_trunk_hidden [128, 128, 128]
  @default_output_hidden [128]
  @default_latent_dim 64
  @default_activation :gelu
  @default_trunk_activation :tanh
  @default_output_steps 1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:num_sensors, pos_integer()}
          | {:history_dim, pos_integer()}
          | {:coord_dim, pos_integer()}
          | {:branch_hidden, [pos_integer()]}
          | {:temporal_hidden, [pos_integer()]}
          | {:trunk_hidden, [pos_integer()]}
          | {:output_hidden, [pos_integer()]}
          | {:latent_dim, pos_integer()}
          | {:activation, atom()}
          | {:trunk_activation, atom()}
          | {:output_dim, pos_integer()}
          | {:output_steps, pos_integer()}
          | {:use_bias, boolean()}

  @doc """
  Build a Temporal Neural Operator model.

  ## Options

    - `:num_sensors` - Number of sensor locations for auxiliary input (required)
    - `:history_dim` - Dimension of flattened solution history input (required)
    - `:coord_dim` - Spatial coordinate dimension (required). Trunk receives coord_dim + 1
      (space + time).
    - `:branch_hidden` - Branch MLP hidden sizes (default: [128, 128, 128])
    - `:temporal_hidden` - Temporal branch MLP hidden sizes (default: [128, 128, 128])
    - `:trunk_hidden` - Trunk MLP hidden sizes (default: [128, 128, 128])
    - `:output_hidden` - Output projection MLP hidden sizes (default: [128])
    - `:latent_dim` - Latent dimension p for all branches (default: 64)
    - `:activation` - Branch/temporal activation (default: :gelu)
    - `:trunk_activation` - Trunk activation (default: :tanh)
    - `:output_dim` - Output variables per query point (default: 1)
    - `:output_steps` - Future timesteps predicted per pass (default: 1)
    - `:use_bias` - Include learnable output bias (default: true)

  ## Inputs

    - `"sensors"`: `[batch, num_sensors]` — auxiliary function values
    - `"history"`: `[batch, history_dim]` — flattened solution history
    - `"queries"`: `[batch, num_queries, coord_dim + 1]` — spatio-temporal queries

  ## Returns

    `[batch, num_queries, output_steps * output_dim]`

  ## Examples

      iex> model = Edifice.Scientific.TNO.build(num_sensors: 10, history_dim: 10, coord_dim: 1, latent_dim: 8)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    num_sensors = Keyword.fetch!(opts, :num_sensors)
    history_dim = Keyword.fetch!(opts, :history_dim)
    coord_dim = Keyword.fetch!(opts, :coord_dim)
    branch_hidden = Keyword.get(opts, :branch_hidden, @default_branch_hidden)
    temporal_hidden = Keyword.get(opts, :temporal_hidden, @default_temporal_hidden)
    trunk_hidden = Keyword.get(opts, :trunk_hidden, @default_trunk_hidden)
    output_hidden = Keyword.get(opts, :output_hidden, @default_output_hidden)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    activation = Keyword.get(opts, :activation, @default_activation)
    trunk_activation = Keyword.get(opts, :trunk_activation, @default_trunk_activation)
    output_dim = Keyword.get(opts, :output_dim, 1)
    output_steps = Keyword.get(opts, :output_steps, @default_output_steps)
    use_bias = Keyword.get(opts, :use_bias, true)

    final_out = output_steps * output_dim

    # Three inputs
    sensors = Axon.input("sensors", shape: {nil, num_sensors})
    history = Axon.input("history", shape: {nil, history_dim})
    queries = Axon.input("queries", shape: {nil, nil, coord_dim + 1})

    # Branch network: [batch, num_sensors] -> [batch, latent_dim]
    branch = build_mlp(sensors, branch_hidden, latent_dim, activation, "branch")

    # Temporal branch: [batch, history_dim] -> [batch, latent_dim]
    tbranch = build_mlp(history, temporal_hidden, latent_dim, activation, "temporal")

    # Trunk network: [batch, Q, coord_dim+1] -> [batch, Q, latent_dim]
    trunk = build_mlp(queries, trunk_hidden, latent_dim, trunk_activation, "trunk")

    # Three-way Hadamard fusion: [batch, Q, latent_dim]
    fused =
      Axon.layer(
        &hadamard_fuse_impl/4,
        [branch, tbranch, trunk],
        name: "tno_fuse",
        op_name: :hadamard_fuse
      )

    # Output projection MLP: [batch, Q, latent_dim] -> [batch, Q, final_out]
    projected =
      output_hidden
      |> Enum.with_index()
      |> Enum.reduce(fused, fn {hidden, idx}, acc ->
        acc
        |> Axon.dense(hidden, name: "output_dense_#{idx}")
        |> Axon.activation(activation, name: "output_act_#{idx}")
      end)
      |> Axon.dense(final_out, name: "output_proj")

    # Optional learnable bias
    if use_bias do
      bias_param = Axon.param("output_bias", {final_out}, initializer: :zeros)

      Axon.layer(
        fn x, bias, _opts -> Nx.add(x, bias) end,
        [projected, bias_param],
        name: "output_bias",
        op_name: :bias_add
      )
    else
      projected
    end
  end

  # ============================================================================
  # MLP builder (shared for branch, temporal branch, trunk)
  # ============================================================================

  defp build_mlp(input, hidden_layers, output_dim, activation, prefix) do
    hidden_layers
    |> Enum.with_index()
    |> Enum.reduce(input, fn {hidden, idx}, acc ->
      acc
      |> Axon.dense(hidden, name: "#{prefix}_dense_#{idx}")
      |> Axon.activation(activation, name: "#{prefix}_act_#{idx}")
    end)
    |> Axon.dense(output_dim, name: "#{prefix}_out")
  end

  # ============================================================================
  # Three-way Hadamard fusion
  # ============================================================================

  # branch: [batch, p], tbranch: [batch, p], trunk: [batch, Q, p]
  # -> [batch, Q, p]
  defp hadamard_fuse_impl(branch, tbranch, trunk, _opts) do
    b = Nx.new_axis(branch, 1)
    tb = Nx.new_axis(tbranch, 1)
    Nx.multiply(Nx.multiply(b, tb), trunk)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a TNO model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    output_dim = Keyword.get(opts, :output_dim, 1)
    output_steps = Keyword.get(opts, :output_steps, @default_output_steps)
    output_steps * output_dim
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      branch_hidden: [128, 128, 128],
      temporal_hidden: [128, 128, 128],
      trunk_hidden: [128, 128, 128],
      output_hidden: [128],
      latent_dim: 64,
      activation: :gelu,
      trunk_activation: :tanh,
      output_dim: 1,
      output_steps: 1,
      use_bias: true
    ]
  end
end
