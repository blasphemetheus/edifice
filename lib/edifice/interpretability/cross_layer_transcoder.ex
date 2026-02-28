defmodule Edifice.Interpretability.CrossLayerTranscoder do
  @moduledoc """
  Cross-Layer Transcoder for circuit-level sparse analysis.

  <!-- verified: true, date: 2026-02-28 -->

  Extends the Transcoder to process all MLP layers simultaneously with a
  shared dictionary. Each layer gets its own encoder projection but shares
  a single sparse feature dictionary. The decoder maps sparse features back
  to each layer's output space independently.

  This enables full circuit-level analysis: a single feature can activate
  across multiple layers, revealing cross-layer circuits.

  ## Architecture

  ```
  Layer 0 acts [batch, dim]  Layer 1 acts [batch, dim]  ...  Layer L acts [batch, dim]
       |                          |                              |
  Encoder_0(dict_size)       Encoder_1(dict_size)          Encoder_L(dict_size)
       |                          |                              |
       +------------- SUM --------+--------- ... ----------------+
                       |
                  [batch, dict_size]
                       |
                   ReLU + Top-K
                       |
                  [batch, dict_size]  (shared sparse features)
                       |
       +-----------+---+---+-----------+
       |           |       |           |
  Decoder_0    Decoder_1  ...     Decoder_L
       |           |       |           |
  Out_0 [b,dim] Out_1 [b,dim] ... Out_L [b,dim]
  ```

  ## Returns

  An Axon model. Input: concatenated layer activations `[batch, num_layers * dim]`.
  Output: concatenated predictions `[batch, num_layers * dim]`.

  ## References

  - Anthropic, "Cross-Layer Transcoding" (Feb 2025)
  - Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits" (2024)
  """

  alias Edifice.Interpretability.SparseAutoencoder

  @default_dict_size 4096
  @default_top_k 32
  @default_num_layers 6

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dict_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:top_k, pos_integer()}

  @doc """
  Build a cross-layer transcoder.

  ## Options

    - `:hidden_size` - Per-layer activation dimension (required)
    - `:num_layers` - Number of MLP layers to analyze (default: 6)
    - `:dict_size` - Shared dictionary size (default: 4096)
    - `:top_k` - Number of active features (default: 32)

  ## Returns

    An Axon model: `[batch, num_layers * hidden_size]` -> `[batch, num_layers * hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    total_dim = num_layers * hidden_size
    input = Axon.input("activations", shape: {nil, total_dim})

    # Per-layer encoders summed into shared dict space
    encoder_sum =
      Enum.reduce(0..(num_layers - 1), nil, fn i, acc ->
        # Slice this layer's activations
        layer_acts =
          Axon.nx(
            input,
            fn t ->
              Nx.slice_along_axis(t, i * hidden_size, hidden_size, axis: 1)
            end, name: "slice_layer_#{i}")

        encoded = Axon.dense(layer_acts, dict_size, name: "encoder_#{i}")

        case acc do
          nil -> encoded
          prev -> Axon.add(prev, encoded, name: "encoder_sum_#{i}")
        end
      end)

    # Shared ReLU + top-k sparsification
    hidden = Axon.activation(encoder_sum, :relu, name: "shared_relu")

    hidden =
      Axon.layer(
        fn acts, _opts -> SparseAutoencoder.top_k_sparsify(acts, top_k) end,
        [hidden],
        name: "shared_top_k",
        op_name: :top_k_sparsify
      )

    # Per-layer decoders concatenated
    decoded_layers =
      Enum.map(0..(num_layers - 1), fn i ->
        Axon.dense(hidden, hidden_size, name: "decoder_#{i}")
      end)

    # Concatenate all layer outputs
    Axon.concatenate(decoded_layers, axis: -1, name: "concat_output")
  end

  @doc "Get the output size of the cross-layer transcoder."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_layers * hidden_size
  end
end
