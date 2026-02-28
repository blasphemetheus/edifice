defmodule Edifice.Scientific.DeepONet do
  @moduledoc """
  DeepONet — Deep Operator Network.

  <!-- verified: true, date: 2026-02-27 -->

  Learns nonlinear operators mapping between infinite-dimensional function
  spaces. The architecture decomposes into a **branch network** (encodes
  the input function) and a **trunk network** (encodes query locations),
  whose outputs are combined via dot product to evaluate the learned
  operator at arbitrary query points.

  ## Architecture

  ```
  Input Function Values    Query Locations
  [batch, num_sensors]     [batch, num_queries, coord_dim]
        |                        |
        v                        v
  +----------------+      +----------------+
  | Branch Network |      | Trunk Network  |
  | MLP            |      | MLP per query  |
  +----------------+      +----------------+
        |                        |
        v                        v
  [batch, p]               [batch, num_queries, p]
        |                        |
        +---------- dot ---------+
                     |
                     v
           [batch, num_queries, 1]
           Operator output G(u)(y)
  ```

  ## Key Insight

  For an operator G mapping input function u to output function G(u):
  - **Branch net** encodes u (sampled at fixed sensor locations) → coefficients
  - **Trunk net** encodes query location y → basis functions
  - **Output**: G(u)(y) = sum_k branch_k(u) * trunk_k(y) + bias

  This is a universal approximation theorem for operators (Chen & Chen, 1995),
  made practical with deep networks (Lu et al., 2021).

  ## Usage

      model = DeepONet.build(
        num_sensors: 100,
        coord_dim: 2,
        branch_hidden: [128, 128, 128],
        trunk_hidden: [128, 128, 128],
        latent_dim: 64
      )

  ## Applications

  - PDE solving (Burgers, Darcy flow, advection-diffusion)
  - Fluid dynamics surrogate modeling
  - Climate/weather model emulation
  - Material property prediction

  ## References

  - Lu et al., "Learning nonlinear operators via DeepONet based on the
    universal approximation theorem of operators" (Nature Machine Intelligence, 2021)
  - https://arxiv.org/abs/1910.03193
  """

  @default_branch_hidden [128, 128, 128]
  @default_trunk_hidden [128, 128, 128]
  @default_latent_dim 64
  @default_activation :gelu

  @doc """
  Build a DeepONet model.

  ## Options

    - `:num_sensors` - Number of sensor locations for input function sampling (required)
    - `:coord_dim` - Dimension of query coordinates, e.g. 1, 2, or 3 (required)
    - `:branch_hidden` - Hidden layer sizes for branch network (default: [128, 128, 128])
    - `:trunk_hidden` - Hidden layer sizes for trunk network (default: [128, 128, 128])
    - `:latent_dim` - Dimension of the latent space (p) (default: 64)
    - `:activation` - Activation function (default: :gelu)
    - `:output_dim` - Output dimension per query point (default: 1)
    - `:use_bias` - Whether to include a learnable output bias (default: true)

  ## Returns

  An Axon model with inputs:
    - "sensors": Input function values at sensor locations `[batch, num_sensors]`
    - "queries": Query coordinates `[batch, num_queries, coord_dim]`

  Returns `[batch, num_queries, output_dim]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:branch_hidden, [pos_integer()]}
          | {:coord_dim, pos_integer()}
          | {:latent_dim, pos_integer()}
          | {:num_sensors, pos_integer()}
          | {:output_dim, pos_integer()}
          | {:trunk_hidden, [pos_integer()]}
          | {:use_bias, boolean()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    num_sensors = Keyword.fetch!(opts, :num_sensors)
    coord_dim = Keyword.fetch!(opts, :coord_dim)
    branch_hidden = Keyword.get(opts, :branch_hidden, @default_branch_hidden)
    trunk_hidden = Keyword.get(opts, :trunk_hidden, @default_trunk_hidden)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    activation = Keyword.get(opts, :activation, @default_activation)
    output_dim = Keyword.get(opts, :output_dim, 1)
    use_bias = Keyword.get(opts, :use_bias, true)

    # Inputs
    sensors = Axon.input("sensors", shape: {nil, num_sensors})
    queries = Axon.input("queries", shape: {nil, nil, coord_dim})

    # Branch network: [batch, num_sensors] → [batch, latent_dim * output_dim]
    branch =
      Enum.with_index(branch_hidden, fn hidden, idx ->
        {hidden, idx}
      end)
      |> Enum.reduce(sensors, fn {hidden, idx}, acc ->
        acc
        |> Axon.dense(hidden, name: "branch_dense_#{idx}")
        |> Axon.activation(activation, name: "branch_act_#{idx}")
      end)
      |> Axon.dense(latent_dim * output_dim, name: "branch_out")

    # Trunk network: [batch, num_queries, coord_dim] → [batch, num_queries, latent_dim * output_dim]
    trunk =
      Enum.with_index(trunk_hidden, fn hidden, idx ->
        {hidden, idx}
      end)
      |> Enum.reduce(queries, fn {hidden, idx}, acc ->
        acc
        |> Axon.dense(hidden, name: "trunk_dense_#{idx}")
        |> Axon.activation(activation, name: "trunk_act_#{idx}")
      end)
      |> Axon.dense(latent_dim * output_dim, name: "trunk_out")

    # Optional learnable bias
    bias_param =
      if use_bias do
        Axon.param("output_bias", {output_dim}, initializer: :zeros)
      else
        nil
      end

    # Combine: dot product of branch and trunk → [batch, num_queries, output_dim]
    if use_bias do
      Axon.layer(
        &combine_with_bias/4,
        [branch, trunk, bias_param],
        name: "deeponet_combine",
        latent_dim: latent_dim,
        output_dim: output_dim,
        op_name: :deeponet_combine
      )
    else
      Axon.layer(
        &combine_no_bias/3,
        [branch, trunk],
        name: "deeponet_combine",
        latent_dim: latent_dim,
        output_dim: output_dim,
        op_name: :deeponet_combine
      )
    end
  end

  @doc "Get the output size of a DeepONet model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :output_dim, 1)
  end

  # Combine branch and trunk via dot product (with bias)
  defp combine_with_bias(branch, trunk, bias, opts) do
    result = dot_product(branch, trunk, opts)
    Nx.add(result, bias)
  end

  # Combine branch and trunk via dot product (no bias)
  defp combine_no_bias(branch, trunk, opts) do
    dot_product(branch, trunk, opts)
  end

  defp dot_product(branch, trunk, opts) do
    latent_dim = opts[:latent_dim]
    output_dim = opts[:output_dim]
    {batch, num_queries, _} = Nx.shape(trunk)

    # Reshape branch: [batch, latent_dim * output_dim] → [batch, 1, output_dim, latent_dim]
    branch_r = Nx.reshape(branch, {batch, 1, output_dim, latent_dim})

    # Reshape trunk: [batch, Q, latent_dim * output_dim] → [batch, Q, output_dim, latent_dim]
    trunk_r = Nx.reshape(trunk, {batch, num_queries, output_dim, latent_dim})

    # Dot product over latent_dim: [batch, Q, output_dim]
    Nx.sum(Nx.multiply(branch_r, trunk_r), axes: [3])
  end
end
