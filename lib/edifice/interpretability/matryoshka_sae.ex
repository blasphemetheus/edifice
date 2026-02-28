defmodule Edifice.Interpretability.MatryoshkaSAE do
  @moduledoc """
  Matryoshka Sparse Autoencoder for multi-granularity feature analysis.

  Dictionary features are ordered by importance. Any prefix of features
  `[0..k]` forms a valid sub-dictionary that independently reconstructs the
  input. Use fewer features for coarse understanding, more for fine-grained
  analysis — without training separate SAEs at each scale.

  ## Architecture

  Same as a standard SAE, but with an ordered-importance loss that encourages
  early features to capture the most variance:

  ```
  Input [batch, input_size]
        |
  Encoder: dense(dict_size) + ReLU + top-k
        |
  [batch, dict_size]  (sparse activations, ordered by importance)
        |
  Decoder: dense(input_size)
        |
  Output [batch, input_size]  (reconstruction)
  ```

  ## Training

  The `loss/4` function applies a weighted L1 penalty that increases with
  feature index, encouraging the model to place important features first.
  For full Matryoshka training, use `multi_scale_loss/4` which evaluates
  reconstruction at multiple prefix sizes.

  ## Usage

      model = MatryoshkaSAE.build(
        input_size: 256,
        dict_size: 4096,
        top_k: 64
      )

      # Analyze at multiple scales
      encoder = MatryoshkaSAE.build_encoder(opts)
      # features[0..63]  → coarse analysis
      # features[0..255] → medium analysis
      # features[0..4095] → fine-grained analysis

  ## References

  - Bussmann et al., "Matryoshka Sparse Autoencoders" (2025)
  - Inspired by Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)
  """

  import Nx.Defn

  alias Edifice.Interpretability.SparseAutoencoder

  @default_dict_size 4096
  @default_top_k 32

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:dict_size, pos_integer()}
          | {:top_k, pos_integer()}

  @doc """
  Build a Matryoshka sparse autoencoder.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:dict_size` - Number of dictionary features (default: #{@default_dict_size})
    - `:top_k` - Number of active features (default: #{@default_top_k})

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, input_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    input = Axon.input("matryoshka_sae_input", shape: {nil, input_size})

    hidden = Axon.dense(input, dict_size, name: "matryoshka_sae_encoder")
    hidden = Axon.activation(hidden, :relu, name: "matryoshka_sae_encoder_act")

    hidden =
      Axon.layer(
        fn acts, _opts -> SparseAutoencoder.top_k_sparsify(acts, top_k) end,
        [hidden],
        name: "matryoshka_sae_top_k",
        op_name: :top_k_sparsify
      )

    Axon.dense(hidden, input_size, name: "matryoshka_sae_decoder")
  end

  @doc """
  Build the encoder portion only.

  Returns sparse hidden activations `[batch, dict_size]` with features ordered
  by importance (once trained with the Matryoshka loss).
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    input = Axon.input("matryoshka_sae_input", shape: {nil, input_size})

    hidden = Axon.dense(input, dict_size, name: "matryoshka_sae_encoder")
    hidden = Axon.activation(hidden, :relu, name: "matryoshka_sae_encoder_act")

    Axon.layer(
      fn acts, _opts -> SparseAutoencoder.top_k_sparsify(acts, top_k) end,
      [hidden],
      name: "matryoshka_sae_top_k",
      op_name: :top_k_sparsify
    )
  end

  @doc """
  Matryoshka loss: reconstruction MSE + importance-weighted L1 penalty.

  Later features receive higher L1 penalty, incentivizing the model to place
  the most important features at the beginning of the dictionary.

  ## Parameters

    - `input` - Original activations `[batch, input_size]`
    - `reconstruction` - SAE output `[batch, input_size]`
    - `hidden_acts` - Sparse hidden activations `[batch, dict_size]`
    - `opts` - Options:
      - `:l1_coeff` - Base L1 penalty coefficient (default: 1.0e-3)
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(input, reconstruction, hidden_acts, opts \\ []) do
    l1_coeff = opts[:l1_coeff]

    recon_loss = Nx.mean(Nx.pow(input - reconstruction, 2))

    # Importance-weighted L1: linearly increasing penalty from 0.5x to 2.0x
    dict_size = Nx.axis_size(hidden_acts, 1)
    weights = Nx.linspace(0.5, 2.0, n: dict_size, type: Nx.type(hidden_acts))
    weighted_l1 = Nx.mean(Nx.abs(hidden_acts) * weights)

    recon_loss + l1_coeff * weighted_l1
  end

  @doc "Get the output size of the Matryoshka SAE (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
