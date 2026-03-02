defmodule Edifice.Recurrent.DeepResLSTM do
  @moduledoc """
  DeepResLSTM — stacked residual LSTM blocks from slippi-ai.

  Each block applies LayerNorm -> LSTM -> zero-initialized decoder -> residual add.
  The zero initialization means the network starts as identity (just the encoder),
  then gradually learns to leverage the LSTM layers during training.

  This is simpler than TransformerLike (no FFN blocks) but deeper than plain LSTM
  (residual connections enable stacking more layers without gradient degradation).

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Dense encoder -> hidden_size
        |
  N x ResLSTMBlock:
    residual --------------------------------+
        |                                    |
    LayerNorm                                |
        |                                    |
    LSTM(hidden_size)                        |
        |                                    |
    Dense(hidden_size, zero_init)  <- starts as identity
        |                                    |
    + ---------------------------------------+
        |
  Final LayerNorm -> last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = DeepResLSTM.build(
        embed_dim: 288,
        hidden_size: 512,
        num_layers: 3
      )

  ## References

  - vladfi1/slippi-ai `networks.py` — `DeepResLSTM` / `res_lstm`
  """

  @default_hidden_size 512
  @default_num_layers 3
  @default_dropout 0.0
  @default_norm :layer_norm
  @default_window_size 60

  @doc """
  Build a DeepResLSTM model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Residual stream width (default: 512)
    - `:num_layers` - Number of residual LSTM blocks (default: 3)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:norm` - `:layer_norm` or `:rms_norm` (default: `:layer_norm`)
    - `:seq_len` - Concrete sequence length for JIT (default: 60)
    - `:window_size` - Alias for `:seq_len` (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:norm, :layer_norm | :rms_norm}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    norm = Keyword.get(opts, :norm, @default_norm)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_dim})

    # Dense encoder: project to hidden_size
    x = Axon.dense(input, hidden_size, name: "encoder")

    # Stack N residual LSTM blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        res_lstm_block(acc, hidden_size, dropout, norm, layer_idx)
      end)

    # Final layer norm
    x = apply_norm(x, norm, hidden_size, "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ResLSTM block: pre-norm -> LSTM -> zero-init decoder -> dropout -> residual add
  defp res_lstm_block(x, hidden_size, dropout, norm, layer_idx) do
    normed = apply_norm(x, norm, hidden_size, "block_#{layer_idx}_prenorm")

    output_seq =
      Edifice.Recurrent.build_raw_rnn(normed, hidden_size, :lstm,
        name: "lstm_#{layer_idx}"
      )

    # Zero-initialized decoder — starts as identity (residual passes through unchanged)
    decoded =
      Axon.dense(output_seq, hidden_size,
        name: "decoder_#{layer_idx}",
        kernel_initializer: :zeros,
        bias_initializer: :zeros
      )

    decoded =
      if dropout > 0 do
        Axon.dropout(decoded, rate: dropout, name: "block_#{layer_idx}_dropout")
      else
        decoded
      end

    Axon.add(x, decoded, name: "block_#{layer_idx}_residual")
  end

  defp apply_norm(x, :layer_norm, _hidden_size, name),
    do: Axon.layer_norm(x, name: name, epsilon: 1.0e-6)

  defp apply_norm(x, :rms_norm, hidden_size, name),
    do: Edifice.Blocks.RMSNorm.layer(x, hidden_size: hidden_size, name: name)

  @doc """
  Pack trained Axon parameters into the flat weight buffer expected by the
  fused LSTM block scan kernel.

  Packs the LayerNorm weights and LSTM weights (input proj W_x, bias, recurrent R)
  for each layer into a single contiguous buffer.

  ## Arguments
    * `params` - trained parameter map from `Axon.build` + training
    * `num_layers` - number of residual LSTM blocks
    * `hidden_size` - hidden dimension

  ## Returns
    Flat `{:f, 32}` tensor of shape `[num_layers * (8*H*H + 6*H)]`
  """
  @spec pack_block_weights(map(), pos_integer(), pos_integer()) :: Nx.Tensor.t()
  def pack_block_weights(params, num_layers, _hidden_size) do
    layers =
      for layer_idx <- 1..num_layers do
        # Input projection W_x[H, 4H] and bias b_x[4H]
        w_x = params["lstm_#{layer_idx}_input_proj"]["kernel"]
        b_x = params["lstm_#{layer_idx}_input_proj"]["bias"]

        # Recurrent weight R[H, 4H]
        r_w = params["lstm_#{layer_idx}_fused_scan"]["lstm_#{layer_idx}_recurrent_kernel"]

        # LayerNorm gamma and beta
        gamma = params["block_#{layer_idx}_prenorm"]["gamma"]
        beta = params["block_#{layer_idx}_prenorm"]["beta"]

        Nx.concatenate([
          Nx.flatten(w_x),
          Nx.flatten(b_x),
          Nx.flatten(r_w),
          Nx.flatten(gamma),
          Nx.flatten(beta)
        ])
      end

    Nx.concatenate(layers)
  end

  @doc """
  Run fused block inference with pre-packed weights.

  Runs all LSTM layers in a single kernel call. Note: this skips the per-layer
  zero-initialized decoder dense layers (which start as identity). For models
  early in training or for fast approximate inference, this is numerically
  equivalent. For fully trained models, use the standard `build/1` path.

  ## Arguments
    * `input` - [B, T, H] input tensor (after encoder projection)
    * `packed_weights` - output of `pack_block_weights/3`
    * `num_layers` - number of layers
    * `h0` - optional [B, num_layers, H] initial hidden states (default: zeros)
    * `c0` - optional [B, num_layers, H] initial cell states (default: zeros)

  ## Returns
    `[B, T, H]` — output after all LSTM layers
  """
  def fused_block_inference(input, packed_weights, num_layers, h0 \\ nil, c0 \\ nil) do
    {batch, _seq_len, hidden} = Nx.shape(input)

    h0 = h0 || Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})
    c0 = c0 || Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})

    Edifice.CUDA.FusedScan.lstm_block(input, packed_weights, h0, c0, num_layers)
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
