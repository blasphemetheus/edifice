defmodule Edifice.Sets.DeepSets do
  @moduledoc """
  Permutation-invariant set processing (Zaheer et al., 2017).

  DeepSets processes sets of elements where the output is invariant to the
  ordering of inputs. This is achieved by processing each element independently
  through a shared network (phi), aggregating with a permutation-invariant
  operation (sum), and post-processing the aggregate (rho).

  ## Architecture

  ```
  Input Set [batch, set_size, element_dim]
        |
        v
  +---------------------------+
  | phi (per-element MLP):    |
  |   For each x_i in set:    |
  |     z_i = phi(x_i)        |
  +---------------------------+
        |
        v
  +---------------------------+
  | Aggregate (sum):          |
  |   z = SUM_i phi(x_i)      |
  +---------------------------+
        |
        v
  +---------------------------+
  | rho (post-aggregation):   |
  |   output = rho(z)         |
  +---------------------------+
        |
        v
  Output [batch, output_dim]
  ```

  ## Key Property

  The architecture output = rho(SUM(phi(x_i))) is provably a universal
  approximator for permutation-invariant functions on sets.

  ## Usage

      # Build a DeepSets model for set classification
      model = DeepSets.build(
        input_dim: 3,
        hidden_size: 64,
        output_dim: 10,
        phi_sizes: [64, 64],
        rho_sizes: [64, 32]
      )

      # Process a batch of sets
      # Input: {batch=4, set_size=20, element_dim=3}
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({4, 20, 3}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, %{"input" => set_data})

  ## References

  - "Deep Sets" (Zaheer et al., NeurIPS 2017)
  """

  @default_hidden_size 64
  @default_phi_sizes [64, 64]
  @default_rho_sizes [64]
  @default_activation :relu
  @default_dropout 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a DeepSets model for permutation-invariant set processing.

  ## Options

  - `:input_dim` - Dimension of each set element (required)
  - `:hidden_size` - Intermediate dimension for phi output (default: 64)
  - `:output_dim` - Final output dimension (required)
  - `:phi_sizes` - Hidden layer sizes for per-element network (default: [64, 64])
  - `:rho_sizes` - Hidden layer sizes for post-aggregation network (default: [64])
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:aggregation` - Set aggregation: :sum, :mean, :max (default: :sum)

  ## Returns

  An Axon model. Input shape: `{batch, set_size, input_dim}`.
  Output shape: `{batch, output_dim}`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:aggregation, :sum | :mean | :max}
          | {:dropout, float()}
          | {:hidden_size, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:output_dim, pos_integer()}
          | {:phi_sizes, pos_integer()}
          | {:rho_sizes, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    output_dim = Keyword.fetch!(opts, :output_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    phi_sizes = Keyword.get(opts, :phi_sizes, @default_phi_sizes)
    rho_sizes = Keyword.get(opts, :rho_sizes, @default_rho_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    aggregation = Keyword.get(opts, :aggregation, :sum)

    # Input: [batch, set_size, element_dim]
    input = Axon.input("input", shape: {nil, nil, input_dim})

    # Phi network: process each element independently
    # Dense layers operate on the last dimension, so they are shared across set elements
    phi_output =
      phi_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        layer =
          acc
          |> Axon.dense(size, name: "phi_dense_#{idx}")
          |> Axon.activation(activation, name: "phi_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "phi_drop_#{idx}")
        else
          layer
        end
      end)

    # Final phi projection to hidden_size
    phi_final = Axon.dense(phi_output, hidden_size, name: "phi_output")

    # Permutation-invariant aggregation: [batch, set_size, hidden_size] -> [batch, hidden_size]
    aggregated =
      Axon.nx(
        phi_final,
        fn features ->
          case aggregation do
            :sum -> Nx.sum(features, axes: [1])
            :mean -> Nx.mean(features, axes: [1])
            :max -> Nx.reduce_max(features, axes: [1])
          end
        end,
        name: "set_aggregate_#{aggregation}"
      )

    # Rho network: post-aggregation processing
    rho_output =
      rho_sizes
      |> Enum.with_index()
      |> Enum.reduce(aggregated, fn {size, idx}, acc ->
        layer =
          acc
          |> Axon.dense(size, name: "rho_dense_#{idx}")
          |> Axon.activation(activation, name: "rho_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "rho_drop_#{idx}")
        else
          layer
        end
      end)

    # Final output projection
    Axon.dense(rho_output, output_dim, name: "output")
  end
end
