defmodule Edifice.Interpretability.Crosscoder do
  @moduledoc """
  Crosscoder for joint sparse dictionary learning across multiple sources.

  Trains a single shared dictionary simultaneously across multiple model
  checkpoints, layers, or components. Each source has its own encoder and
  decoder, but they share a single sparse feature space. Features that appear
  across sources represent shared structure.

  ## Architecture

  ```
  Source 0 [batch, input_size]  Source 1 [batch, input_size]  ...
        |                            |
  Encoder 0: dense(dict_size)  Encoder 1: dense(dict_size)
        |                            |
        └──────────┬─────────────────┘
                   |
            Sum → ReLU → top-k
                   |
        Shared sparse activations [batch, dict_size]
                   |
        ┌──────────┴─────────────────┐
        |                            |
  Decoder 0: dense(output_size) Decoder 1: dense(output_size)
        |                            |
  Output 0 [batch, output_size] Output 1 [batch, output_size]
  ```

  ## Cross-Layer Transcoder Mode

  When `:output_size` differs from `:input_size`, this operates as a
  Cross-Layer Transcoder (CLT): mapping MLP inputs to MLP outputs across layers
  through a shared sparse dictionary. This enables circuit-level analysis by
  replacing all MLPs with a single sparse linear computation.

  ## Usage

      # Symmetric crosscoder (e.g., base model vs fine-tuned model)
      model = Crosscoder.build(
        input_size: 256,
        dict_size: 4096,
        num_sources: 2,
        top_k: 32
      )

      # Cross-layer transcoder (MLP-in → MLP-out across 6 layers)
      model = Crosscoder.build(
        input_size: 256,
        output_size: 256,
        dict_size: 4096,
        num_sources: 6,
        top_k: 32
      )

  ## References

  - Lindsey et al., "Crosscoders: Sparse Dictionary Learning across Model
    Checkpoints" (Anthropic, Dec 2024)
  - Dunefsky et al., "Cross-Layer Transcoders" (Anthropic, Feb 2025)
  """

  import Nx.Defn

  alias Edifice.Interpretability.SparseAutoencoder

  @default_dict_size 4096
  @default_num_sources 2
  @default_top_k 32

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:dict_size, pos_integer()}
          | {:num_sources, pos_integer()}
          | {:top_k, pos_integer()}

  @doc """
  Build a crosscoder.

  ## Options

    - `:input_size` - Dimension of each source's input activations (required)
    - `:output_size` - Dimension of each source's output (default: same as input_size).
      Set differently for cross-layer transcoder mode.
    - `:dict_size` - Number of shared dictionary features (default: #{@default_dict_size})
    - `:num_sources` - Number of sources to jointly train across (default: #{@default_num_sources})
    - `:top_k` - Number of active features (default: #{@default_top_k})

  ## Returns

    An Axon container `%{source_0: ..., source_1: ..., ...}` with one
    reconstruction per source, each of shape `[batch, output_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.get(opts, :output_size, input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    num_sources = Keyword.get(opts, :num_sources, @default_num_sources)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    # Per-source inputs and encoders
    {_inputs, encoded} = build_encoders(input_size, dict_size, num_sources)

    # Sum encoder outputs, activate, sparsify
    hidden = build_shared_hidden(encoded, top_k)

    # Per-source decoders
    outputs = build_decoders(hidden, output_size, num_sources)

    Axon.container(outputs)
  end

  @doc """
  Build the shared encoder portion only.

  Returns sparse shared activations `[batch, dict_size]`.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    num_sources = Keyword.get(opts, :num_sources, @default_num_sources)
    top_k = Keyword.get(opts, :top_k, @default_top_k)

    {_inputs, encoded} = build_encoders(input_size, dict_size, num_sources)
    build_shared_hidden(encoded, top_k)
  end

  @doc """
  Compute crosscoder training loss: mean reconstruction MSE across sources + L1.
  """
  @spec loss([Nx.Tensor.t()], [Nx.Tensor.t()], Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(targets, reconstructions, hidden_acts, opts \\ []) do
    l1_coeff = opts[:l1_coeff]

    # Mean reconstruction loss across all sources
    {recon_sum, count} =
      {targets, reconstructions}
      |> then(fn {ts, rs} ->
        Enum.zip(ts, rs)
        |> Enum.reduce({Nx.tensor(0.0), 0}, fn {t, r}, {acc, n} ->
          {acc + Nx.mean(Nx.pow(t - r, 2)), n + 1}
        end)
      end)

    recon_loss = recon_sum / count
    l1_loss = Nx.mean(Nx.abs(hidden_acts))
    recon_loss + l1_coeff * l1_loss
  end

  @doc "Get the output size of the crosscoder."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :output_size, Keyword.fetch!(opts, :input_size))
  end

  # -- Private builders --

  defp build_encoders(input_size, dict_size, num_sources) do
    inputs_and_encoded =
      for i <- 0..(num_sources - 1) do
        input = Axon.input("crosscoder_source_#{i}", shape: {nil, input_size})
        encoded = Axon.dense(input, dict_size, name: "crosscoder_encoder_#{i}")
        {input, encoded}
      end

    {Enum.map(inputs_and_encoded, &elem(&1, 0)), Enum.map(inputs_and_encoded, &elem(&1, 1))}
  end

  defp build_shared_hidden(encoded, top_k) do
    summed =
      case encoded do
        [single] -> single
        [first | rest] -> Enum.reduce(rest, first, &Axon.add(&2, &1))
      end

    hidden = Axon.activation(summed, :relu, name: "crosscoder_shared_act")

    Axon.layer(
      fn acts, _opts -> SparseAutoencoder.top_k_sparsify(acts, top_k) end,
      [hidden],
      name: "crosscoder_top_k",
      op_name: :top_k_sparsify
    )
  end

  defp build_decoders(hidden, output_size, num_sources) do
    for i <- 0..(num_sources - 1), into: %{} do
      key = String.to_atom("source_#{i}")
      decoded = Axon.dense(hidden, output_size, name: "crosscoder_decoder_#{i}")
      {key, decoded}
    end
  end
end
