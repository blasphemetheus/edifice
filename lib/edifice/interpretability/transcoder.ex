defmodule Edifice.Interpretability.Transcoder do
  @moduledoc """
  Transcoder for cross-layer mechanistic interpretability.

  Like a Sparse Autoencoder but maps between different layers' activation spaces.
  Input and output dimensions can differ, enabling analysis of how representations
  transform across layers.

  ## Architecture

  ```
  Input [batch, input_size]   (layer N activations)
        |
  Encoder: dense(dict_size) + ReLU
        |
  Sparsify: top-k
        |
  [batch, dict_size]  (sparse cross-layer features)
        |
  Decoder: dense(output_size)
        |
  Output [batch, output_size]  (layer M activation prediction)
  ```

  ## Usage

      model = Transcoder.build(
        input_size: 256,
        output_size: 512,
        dict_size: 4096,
        top_k: 32
      )

  ## References

  - Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits" (2024)
  """

  import Nx.Defn

  @default_dict_size 4096
  @default_top_k 32

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:dict_size, pos_integer()}
          | {:top_k, pos_integer()}

  @doc """
  Build a transcoder.

  ## Options

    - `:input_size` - Source layer activation dimension (required)
    - `:output_size` - Target layer activation dimension (required)
    - `:dict_size` - Number of dictionary features (default: 4096)
    - `:top_k` - Number of active features (default: 32)

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, output_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.fetch!(opts, :output_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    input = Axon.input("transcoder_input", shape: {nil, input_size})

    # Encoder
    hidden = Axon.dense(input, dict_size, name: "transcoder_encoder")
    hidden = Axon.activation(hidden, :relu, name: "transcoder_encoder_act")

    # Top-k sparsify
    hidden =
      Axon.layer(
        fn acts, _opts -> top_k_sparsify(acts, top_k) end,
        [hidden],
        name: "transcoder_top_k",
        op_name: :top_k_sparsify
      )

    # Decoder to output space
    Axon.dense(hidden, output_size, name: "transcoder_decoder")
  end

  @doc """
  Compute transcoder training loss: reconstruction MSE + L1 sparsity penalty.

  ## Parameters

    - `target` - Target layer activations `[batch, output_size]`
    - `reconstruction` - Transcoder output `[batch, output_size]`
    - `hidden_acts` - Sparse hidden activations `[batch, dict_size]`
    - `opts` - Options:
      - `:l1_coeff` - L1 penalty coefficient (default: 1.0e-3)

  ## Returns

    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(target, reconstruction, hidden_acts, opts \\ []) do
    l1_coeff = opts[:l1_coeff]

    recon_loss = Nx.mean(Nx.pow(target - reconstruction, 2))
    l1_loss = Nx.mean(Nx.abs(hidden_acts))

    recon_loss + l1_coeff * l1_loss
  end

  defnp top_k_sparsify(activations, k) do
    {top_values, _top_indices} = Nx.top_k(activations, k: k)
    threshold = Nx.slice_along_axis(top_values, k - 1, 1, axis: 1)
    mask = Nx.greater_equal(activations, threshold)
    Nx.select(mask, activations, Nx.tensor(0.0, type: Nx.type(activations)))
  end

  @doc "Get the output size of the transcoder."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :output_size)
  end
end
