defmodule Edifice.Interpretability.SparseAutoencoder do
  @moduledoc """
  Sparse Autoencoder (SAE) for mechanistic interpretability.

  Learns a sparse, overcomplete dictionary of features from neural network
  activations. Used to decompose model internals into interpretable directions.

  ## Architecture

  ```
  Input [batch, input_size]
        |
  Encoder: dense(dict_size) + ReLU
        |
  Sparsify: top-k or L1 penalty
        |
  [batch, dict_size]  (sparse activations)
        |
  Decoder: dense(input_size)
        |
  Output [batch, input_size]  (reconstruction)
  ```

  ## Sparsity Modes

  - `:top_k` — Zero out all but the top-k activations per sample (hard sparsity)
  - `:l1` — No hard sparsity in the forward pass; use `loss/4` with L1 penalty

  ## Usage

      model = SparseAutoencoder.build(
        input_size: 256,
        dict_size: 4096,
        top_k: 32,
        sparsity: :top_k
      )

      # Training loss includes reconstruction + L1 penalty
      loss = SparseAutoencoder.loss(input, reconstruction, hidden_acts, l1_coeff: 1.0e-3)

  ## References

  - Bricken et al., "Towards Monosemanticity" (Anthropic, 2023)
  - Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features" (2023)
  """

  import Nx.Defn

  @default_dict_size 4096
  @default_top_k 32

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:dict_size, pos_integer()}
          | {:top_k, pos_integer()}
          | {:sparsity, :top_k | :l1}

  @doc """
  Build a sparse autoencoder.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:dict_size` - Number of dictionary features, typically >> input_size (default: 4096)
    - `:top_k` - Number of active features when `sparsity: :top_k` (default: 32)
    - `:sparsity` - Sparsity mode: `:top_k` or `:l1` (default: `:top_k`)

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, input_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    sparsity = Keyword.get(opts, :sparsity, :top_k)

    input = Axon.input("sae_input", shape: {nil, input_size})

    # Encoder
    hidden = Axon.dense(input, dict_size, name: "sae_encoder")
    hidden = Axon.activation(hidden, :relu, name: "sae_encoder_act")

    # Sparsify
    hidden =
      case sparsity do
        :top_k ->
          Axon.layer(
            fn acts, _opts -> top_k_sparsify(acts, top_k) end,
            [hidden],
            name: "sae_top_k",
            op_name: :top_k_sparsify
          )

        :l1 ->
          hidden
      end

    # Decoder
    Axon.dense(hidden, input_size, name: "sae_decoder")
  end

  @doc """
  Build the encoder portion only (for extracting sparse activations).

  Returns the sparse hidden activations `[batch, dict_size]`.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    sparsity = Keyword.get(opts, :sparsity, :top_k)

    input = Axon.input("sae_input", shape: {nil, input_size})

    hidden = Axon.dense(input, dict_size, name: "sae_encoder")
    hidden = Axon.activation(hidden, :relu, name: "sae_encoder_act")

    case sparsity do
      :top_k ->
        Axon.layer(
          fn acts, _opts -> top_k_sparsify(acts, top_k) end,
          [hidden],
          name: "sae_top_k",
          op_name: :top_k_sparsify
        )

      :l1 ->
        hidden
    end
  end

  @doc """
  Compute SAE training loss: reconstruction MSE + L1 sparsity penalty.

  ## Parameters

    - `input` - Original activations `[batch, input_size]`
    - `reconstruction` - SAE output `[batch, input_size]`
    - `hidden_acts` - Sparse hidden activations `[batch, dict_size]`
    - `opts` - Options:
      - `:l1_coeff` - L1 penalty coefficient (default: 1.0e-3)

  ## Returns

    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(input, reconstruction, hidden_acts, opts \\ []) do
    l1_coeff = opts[:l1_coeff]

    # Reconstruction loss (MSE)
    recon_loss = Nx.mean(Nx.pow(input - reconstruction, 2))

    # L1 sparsity penalty on hidden activations
    l1_loss = Nx.mean(Nx.abs(hidden_acts))

    recon_loss + l1_coeff * l1_loss
  end

  @doc """
  Top-k sparsification: keep only the top-k activations, zero out the rest.

  Shared by `SparseAutoencoder` and `Transcoder`.
  """
  @spec top_k_sparsify(Nx.Tensor.t(), non_neg_integer()) :: Nx.Tensor.t()
  defn top_k_sparsify(activations, k) do
    {top_values, _top_indices} = Nx.top_k(activations, k: k)

    # Threshold: the k-th largest value per sample
    threshold = Nx.slice_along_axis(top_values, k - 1, 1, axis: 1)

    # Zero out activations below threshold
    mask = Nx.greater_equal(activations, threshold)
    Nx.select(mask, activations, Nx.tensor(0.0, type: Nx.type(activations)))
  end

  @doc "Get the output size of the SAE (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
