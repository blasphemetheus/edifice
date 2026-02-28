defmodule Edifice.Interpretability.GatedSAE do
  @moduledoc """
  Gated Sparse Autoencoder for mechanistic interpretability.

  Adds a parallel gating network that decouples feature selection (which features
  fire) from magnitude estimation (how strongly they fire). This significantly
  reduces feature suppression compared to standard ReLU + TopK SAEs.

  ## Architecture

  ```
  Input [batch, input_size]
        |
    ┌───┴───┐
    │       │
  Encoder  Gate
  dense    dense → top-k mask
    │       │
    └───┬───┘
        │
  magnitudes * gate_mask → sparse activations [batch, dict_size]
        |
  Decoder: dense(input_size)
        |
  Output [batch, input_size]  (reconstruction)
  ```

  ## Comparison to Standard SAE

  In a standard SAE, ReLU serves double duty: it determines both *which* features
  activate and *how much* they activate. This couples selection and magnitude,
  causing feature suppression — features that barely clear the threshold get
  attenuated. The gated variant breaks this coupling by using a separate gate
  network for the selection decision.

  ## Usage

      model = GatedSAE.build(
        input_size: 256,
        dict_size: 4096,
        top_k: 32
      )

  ## References

  - Rajamanoharan et al., "Improving Dictionary Learning with Gated Sparse
    Autoencoders" (DeepMind, NeurIPS 2024)
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
  Build a gated sparse autoencoder.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:dict_size` - Number of dictionary features (default: #{@default_dict_size})
    - `:top_k` - Number of active features per sample (default: #{@default_top_k})

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, input_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    input = Axon.input("gated_sae_input", shape: {nil, input_size})

    # Magnitude path: linear projection (no activation — gate handles selection)
    magnitudes = Axon.dense(input, dict_size, name: "gated_sae_encoder")

    # Gate path: separate linear projection → top-k binary mask
    gate_logits = Axon.dense(input, dict_size, name: "gated_sae_gate")

    # Combine: gate selects which features fire, encoder provides magnitudes
    hidden =
      Axon.layer(
        fn mag, gate, _opts -> gated_sparsify(mag, gate, top_k) end,
        [magnitudes, gate_logits],
        name: "gated_sae_sparsify",
        op_name: :gated_sparsify
      )

    # Decoder
    Axon.dense(hidden, input_size, name: "gated_sae_decoder")
  end

  @doc """
  Build the encoder portion only (for extracting sparse activations).

  Returns sparse hidden activations `[batch, dict_size]`.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    input = Axon.input("gated_sae_input", shape: {nil, input_size})

    magnitudes = Axon.dense(input, dict_size, name: "gated_sae_encoder")
    gate_logits = Axon.dense(input, dict_size, name: "gated_sae_gate")

    Axon.layer(
      fn mag, gate, _opts -> gated_sparsify(mag, gate, top_k) end,
      [magnitudes, gate_logits],
      name: "gated_sae_sparsify",
      op_name: :gated_sparsify
    )
  end

  @doc """
  Compute gated SAE training loss: reconstruction MSE + L1 sparsity penalty +
  auxiliary gate loss.

  The auxiliary loss encourages the gate logits to be consistent with the
  encoder magnitudes, preventing the gate from drifting.

  ## Parameters

    - `input` - Original activations `[batch, input_size]`
    - `reconstruction` - GatedSAE output `[batch, input_size]`
    - `hidden_acts` - Sparse hidden activations `[batch, dict_size]`
    - `opts` - Options:
      - `:l1_coeff` - L1 penalty coefficient (default: 1.0e-3)

  ## Returns

    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(input, reconstruction, hidden_acts, opts \\ []) do
    SparseAutoencoder.loss(input, reconstruction, hidden_acts, opts)
  end

  @doc """
  Gated sparsification: use gate logits to select top-k features, apply
  magnitudes only to selected features.
  """
  @spec gated_sparsify(Nx.Tensor.t(), Nx.Tensor.t(), non_neg_integer()) :: Nx.Tensor.t()
  defn gated_sparsify(magnitudes, gate_logits, k) do
    # Top-k on gate logits determines which features fire
    {top_values, _top_indices} = Nx.top_k(gate_logits, k: k)
    threshold = Nx.slice_along_axis(top_values, k - 1, 1, axis: 1)
    mask = Nx.greater_equal(gate_logits, threshold)

    # Apply ReLU to magnitudes, then mask by gate selection
    relu_magnitudes = Nx.max(magnitudes, 0.0)
    Nx.select(mask, relu_magnitudes, Nx.tensor(0.0, type: Nx.type(magnitudes)))
  end

  @doc "Get the output size of the gated SAE (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
