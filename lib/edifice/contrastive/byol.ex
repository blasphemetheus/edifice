defmodule Edifice.Contrastive.BYOL do
  @moduledoc """
  BYOL - Bootstrap Your Own Latent.

  Implements BYOL from "Bootstrap Your Own Latent: A New Approach to
  Self-Supervised Learning" (Grill et al., NeurIPS 2020). BYOL learns
  representations without negative pairs by using two networks: an online
  network that is trained, and a target network that is an exponential
  moving average (EMA) of the online network.

  ## Key Innovations

  - **No negative pairs needed**: Avoids mode collapse through asymmetric design
  - **Online/target architecture**: Target network provides stable regression targets
  - **Predictor head**: The online network has an extra predictor that the target lacks
  - **EMA update**: Target parameters are a slow-moving average of online parameters

  ## Architecture

  ```
  Augmented View 1              Augmented View 2
        |                             |
        v                             v
  +============+               +============+
  |  Online    |               |   Target   |
  |  Encoder   |               |   Encoder  |  (EMA of online)
  +============+               +============+
        |                             |
        v                             v
  +============+               +============+
  |  Online    |               |   Target   |
  | Projector  |               |  Projector |  (EMA of online)
  +============+               +============+
        |                             |
        v                             |
  +============+                      |
  |  Predictor |                      |
  | (online    |                      |
  |    only)   |                      |
  +============+                      |
        |                             |
        v                             v
       p_i          MSE Loss         z_j
        |                             |
        +----------->.<---------------+
  ```

  ## Usage

      # Build online and target networks
      {online_model, target_model} = BYOL.build(
        encoder_dim: 287,
        projection_dim: 256,
        predictor_dim: 64
      )

      # After each training step, update target via EMA
      target_params = BYOL.ema_update(online_params, target_params, momentum: 0.996)

  ## References
  - Paper: https://arxiv.org/abs/2006.07733
  """

  require Axon
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default projection dimension"
  @spec default_projection_dim() :: pos_integer()
  def default_projection_dim, do: 256

  @doc "Default predictor hidden dimension"
  @spec default_predictor_dim() :: pos_integer()
  def default_predictor_dim, do: 64

  @doc "Default EMA momentum"
  @spec default_momentum() :: float()
  def default_momentum, do: 0.996

  @doc "Default encoder hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build both the online and target BYOL networks.

  The online network includes encoder + projector + predictor.
  The target network includes encoder + projector (no predictor).

  ## Options
    - `:encoder_dim` - Input feature dimension (required)
    - `:projection_dim` - Projector output dimension (default: 256)
    - `:predictor_dim` - Predictor hidden dimension (default: 64)
    - `:hidden_size` - Encoder hidden dimension (default: 256)

  ## Returns
    `{online_model, target_model}` tuple of Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:encoder_dim, pos_integer()}
          | {:projection_dim, pos_integer()}
          | {:predictor_dim, pos_integer()}
          | {:hidden_size, pos_integer()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    online = build_online(opts)
    target = build_target(opts)
    {online, target}
  end

  @doc """
  Build the online network (encoder + projector + predictor).

  ## Options
    - `:encoder_dim` - Input feature dimension (required)
    - `:projection_dim` - Projector output dimension (default: 256)
    - `:predictor_dim` - Predictor hidden dimension (default: 64)
    - `:hidden_size` - Encoder hidden dimension (default: 256)

  ## Returns
    An Axon model mapping inputs to predictor output.
  """
  @spec build_online(keyword()) :: Axon.t()
  def build_online(opts \\ []) do
    encoder_dim = Keyword.fetch!(opts, :encoder_dim)
    projection_dim = Keyword.get(opts, :projection_dim, default_projection_dim())
    predictor_dim = Keyword.get(opts, :predictor_dim, default_predictor_dim())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())

    input = Axon.input("features", shape: {nil, encoder_dim})

    # Encoder
    encoded =
      input
      |> Axon.dense(hidden_size, name: "online_encoder_fc1")
      |> Axon.activation(:relu, name: "online_encoder_relu1")
      |> Axon.layer_norm(name: "online_encoder_norm")
      |> Axon.dense(hidden_size, name: "online_encoder_fc2")
      |> Axon.activation(:relu, name: "online_encoder_relu2")

    # Projector
    projected =
      encoded
      |> Axon.dense(projection_dim, name: "online_proj_fc1")
      |> Axon.activation(:relu, name: "online_proj_relu")
      |> Axon.layer_norm(name: "online_proj_norm")
      |> Axon.dense(projection_dim, name: "online_proj_fc2")

    # Predictor (asymmetric - only in online network)
    projected
    |> Axon.dense(predictor_dim, name: "predictor_fc1")
    |> Axon.activation(:relu, name: "predictor_relu")
    |> Axon.dense(projection_dim, name: "predictor_fc2")
  end

  @doc """
  Build the target network (encoder + projector, no predictor).

  Target network weights should be initialized as a copy of the online
  network (excluding the predictor) and updated via EMA.

  ## Options
    - `:encoder_dim` - Input feature dimension (required)
    - `:projection_dim` - Projector output dimension (default: 256)
    - `:hidden_size` - Encoder hidden dimension (default: 256)

  ## Returns
    An Axon model mapping inputs to projection output.
  """
  @spec build_target(keyword()) :: Axon.t()
  def build_target(opts \\ []) do
    encoder_dim = Keyword.fetch!(opts, :encoder_dim)
    projection_dim = Keyword.get(opts, :projection_dim, default_projection_dim())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())

    input = Axon.input("features", shape: {nil, encoder_dim})

    # Encoder (same architecture, different name prefix for separate params)
    encoded =
      input
      |> Axon.dense(hidden_size, name: "target_encoder_fc1")
      |> Axon.activation(:relu, name: "target_encoder_relu1")
      |> Axon.layer_norm(name: "target_encoder_norm")
      |> Axon.dense(hidden_size, name: "target_encoder_fc2")
      |> Axon.activation(:relu, name: "target_encoder_relu2")

    # Projector (no predictor)
    encoded
    |> Axon.dense(projection_dim, name: "target_proj_fc1")
    |> Axon.activation(:relu, name: "target_proj_relu")
    |> Axon.layer_norm(name: "target_proj_norm")
    |> Axon.dense(projection_dim, name: "target_proj_fc2")
  end

  # ============================================================================
  # EMA Update
  # ============================================================================

  @doc """
  Update target network parameters via exponential moving average.

  target_params = momentum * target_params + (1 - momentum) * online_params

  ## Parameters
    - `online_params` - Current online network parameters (map of tensors)
    - `target_params` - Current target network parameters (map of tensors)

  ## Options
    - `:momentum` - EMA momentum coefficient (default: 0.996)

  ## Returns
    Updated target parameters.
  """
  @spec ema_update(map(), map(), keyword()) :: map()
  def ema_update(online_params, target_params, opts \\ []) do
    momentum = Keyword.get(opts, :momentum, default_momentum())

    # Map over matching parameter keys
    # Online keys have "online_" prefix, target keys have "target_" prefix
    # We match encoder and projector params by stripping the prefix
    Map.new(target_params, fn {key, target_val} ->
      online_key = String.replace(key, "target_", "online_", global: false)

      updated =
        case Map.fetch(online_params, online_key) do
          {:ok, online_val} ->
            ema_blend(online_val, target_val, momentum)

          :error ->
            # No matching online param (shouldn't happen for encoder/projector)
            target_val
        end

      {key, updated}
    end)
  end

  defp ema_blend(online, target, momentum)
       when is_map(online) and not is_struct(online) and is_map(target) and not is_struct(target) do
    Map.new(target, fn {k, t_v} ->
      case Map.fetch(online, k) do
        {:ok, o_v} -> {k, ema_blend(o_v, t_v, momentum)}
        :error -> {k, t_v}
      end
    end)
  end

  defp ema_blend(online, target, momentum) do
    Nx.add(Nx.multiply(momentum, target), Nx.multiply(1.0 - momentum, online))
  end

  # ============================================================================
  # Loss
  # ============================================================================

  @doc """
  Compute the BYOL loss (MSE between normalized online predictions and target projections).

  ## Parameters
    - `online_pred` - Online predictor output: [batch, projection_dim]
    - `target_proj` - Target projector output: [batch, projection_dim]

  ## Returns
    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss(online_pred, target_proj) do
    # L2 normalize both
    p = l2_normalize(online_pred)
    z = l2_normalize(target_proj)

    # MSE between normalized representations
    2.0 - 2.0 * Nx.mean(Nx.sum(p * z, axes: [1]))
  end

  defnp l2_normalize(x) do
    norm = Nx.sqrt(Nx.sum(x * x, axes: [1], keep_axes: true) + 1.0e-8)
    x / norm
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of the BYOL model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :projection_dim, default_projection_dim())
  end
end
