defmodule Edifice.Contrastive.SigLIP do
  @moduledoc """
  SigLIP - Sigmoid Loss for Language-Image Pre-training.

  Implements SigLIP, which replaces the softmax-based contrastive loss (used in CLIP)
  with a simpler sigmoid-based binary classification loss. Each image-text pair is
  treated as an independent binary classification problem.

  ## Key Innovation

  Instead of softmax cross-entropy over all pairs:
  ```
  CLIP: -log(exp(sim_pos) / sum(exp(sim_all)))
  ```

  SigLIP uses sigmoid for each pair independently:
  ```
  SigLIP: sum_ij(log(sigmoid(t * sim_ij * y_ij)))
  ```

  Where:
  - `sim_ij` is the cosine similarity between embeddings i and j
  - `y_ij` = +1 for matching pairs, -1 for non-matching pairs
  - `t` is a learnable temperature parameter

  ## Advantages

  - **Simpler**: No need to normalize over all negatives
  - **Scalable**: Each pair is independent, enabling larger batch sizes
  - **Stable**: Sigmoid gradients are bounded, unlike softmax with temperature
  - **Effective**: Matches or exceeds CLIP performance in practice

  ## Architecture

  ```
  Image Encoder          Text Encoder
        |                      |
        v                      v
  +------------+        +------------+
  |  Backbone  |        |  Backbone  |
  +------------+        +------------+
        |                      |
        v                      v
  +------------+        +------------+
  | Projection |        | Projection |
  +------------+        +------------+
        |                      |
        v                      v
       z_img    SigLIP       z_txt
        |       Loss           |
        +-------> <-----------+
  ```

  ## Usage

      {encoder, _} = SigLIP.build(input_dim: 512, projection_dim: 256)

      # Compute SigLIP loss
      loss = SigLIP.loss(z_img, z_txt, temperature: 1.0)

  ## Reference

  - "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
  """

  import Nx.Defn

  @default_projection_dim 256
  @default_hidden_size 512
  @default_temperature 1.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_dim, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:projection_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:temperature_init, float()}

  @doc """
  Build a SigLIP encoder model.

  ## Options

    - `:input_dim` or `:embed_dim` - Input feature dimension (required)
    - `:projection_dim` - Projection head output dimension (default: 256)
    - `:hidden_size` - Hidden dimension (default: 512)
    - `:temperature_init` - Initial temperature value (default: 1.0)

  ## Returns

    Tuple of `{encoder_model, temperature_param}` where:
    - `encoder_model` is an Axon model mapping inputs to normalized embeddings
    - `temperature_param` is an Axon parameter for learnable temperature
  """
  @spec build([build_opt()]) :: {Axon.t(), %Axon.Parameter{}}
  def build(opts \\ []) do
    input_dim = Keyword.get(opts, :input_dim) || Keyword.fetch!(opts, :embed_dim)
    projection_dim = Keyword.get(opts, :projection_dim, @default_projection_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    temp_init = Keyword.get(opts, :temperature_init, @default_temperature)

    input = Axon.input("features", shape: {nil, input_dim})

    # Encoder backbone
    encoded =
      input
      |> Axon.dense(hidden_size, name: "encoder_fc1")
      |> Axon.activation(:gelu, name: "encoder_gelu1")
      |> Axon.layer_norm(name: "encoder_norm1")
      |> Axon.dense(hidden_size, name: "encoder_fc2")
      |> Axon.activation(:gelu, name: "encoder_gelu2")
      |> Axon.layer_norm(name: "encoder_norm2")

    # Projection head
    projected =
      encoded
      |> Axon.dense(hidden_size, name: "proj_fc1")
      |> Axon.activation(:gelu, name: "proj_gelu")
      |> Axon.dense(projection_dim, name: "proj_fc2")

    # L2 normalize output
    encoder =
      Axon.nx(
        projected,
        fn x ->
          norm = Nx.sqrt(Nx.sum(Nx.multiply(x, x), axes: [-1], keep_axes: true))
          Nx.divide(x, Nx.add(norm, 1.0e-8))
        end,
        name: "l2_normalize"
      )

    # Learnable temperature parameter (log scale for stability)
    temperature =
      Axon.param("siglip_log_temperature", {},
        initializer: fn _, _ -> Nx.log(Nx.tensor(temp_init)) end
      )

    {encoder, temperature}
  end

  @doc """
  Compute SigLIP loss between two sets of embeddings.

  ## Parameters

    - `z_a` - Embeddings from modality A (e.g., images): [batch, dim]
    - `z_b` - Embeddings from modality B (e.g., text): [batch, dim]

  ## Options

    - `:temperature` - Temperature scaling (default: 1.0)
    - `:log_temperature` - Log temperature (overrides :temperature if provided)

  ## Returns

    Scalar loss tensor.

  ## Notes

    Assumes diagonal pairs (z_a[i], z_b[i]) are positive matches.
    All other pairs are treated as negatives.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(z_a, z_b, opts \\ []) do
    # Get temperature (handle log_temperature if provided)
    temperature =
      case opts[:log_temperature] do
        nil ->
          case opts[:temperature] do
            nil -> 1.0
            val -> val
          end

        log_temp ->
          Nx.exp(log_temp)
      end

    # L2 normalize (in case not already normalized)
    z_a = l2_normalize(z_a)
    z_b = l2_normalize(z_b)

    batch_size = Nx.axis_size(z_a, 0)

    # Cosine similarity matrix: [batch, batch]
    # sim[i, j] = z_a[i] . z_b[j]
    sim = Nx.dot(z_a, [1], z_b, [1])

    # Scale by temperature
    sim_scaled = Nx.multiply(temperature, sim)

    # Labels: y_ij = +1 for i==j (positive), -1 for i!=j (negative)
    # Create identity matrix for positive pairs
    eye = Nx.eye(batch_size)
    # y = 2 * eye - 1 = +1 on diagonal, -1 off-diagonal
    y = Nx.subtract(Nx.multiply(2.0, eye), 1.0)

    # SigLIP loss: -sum(log(sigmoid(sim * y)))
    # = -sum(log(sigmoid(t * sim_ij * y_ij)))
    # For positive pairs: log(sigmoid(t * sim_ii))
    # For negative pairs: log(sigmoid(-t * sim_ij)) = log(1 - sigmoid(t * sim_ij))

    logits = Nx.multiply(sim_scaled, y)

    # Log sigmoid for numerical stability: log(sigmoid(x)) = x - softplus(x)
    # But we can also use: log(sigmoid(x)) = -softplus(-x)
    log_sigmoid = Nx.negate(Nx.log1p(Nx.exp(Nx.negate(logits))))

    # Average loss over all pairs
    Nx.negate(Nx.mean(log_sigmoid))
  end

  @doc """
  Compute SigLIP loss using log temperature parameter.

  Convenience function that takes the log temperature directly
  (useful when temperature is a learnable parameter in log space).
  """
  @spec loss_with_log_temp(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss_with_log_temp(z_a, z_b, log_temperature) do
    loss(z_a, z_b, log_temperature: log_temperature)
  end

  # L2 normalize along last axis
  defnp l2_normalize(x) do
    norm = Nx.sqrt(Nx.sum(Nx.multiply(x, x), axes: [-1], keep_axes: true))
    Nx.divide(x, Nx.add(norm, 1.0e-8))
  end

  @doc """
  Get the default projection dimension.
  """
  @spec default_projection_dim() :: pos_integer()
  def default_projection_dim, do: @default_projection_dim

  @doc """
  Get the default hidden size.
  """
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: @default_hidden_size

  @doc """
  Get the default temperature.
  """
  @spec default_temperature() :: float()
  def default_temperature, do: @default_temperature
end
