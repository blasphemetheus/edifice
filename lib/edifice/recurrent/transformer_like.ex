defmodule Edifice.Recurrent.TransformerLike do
  @moduledoc """
  TransformerLike — a hybrid recurrent+FFN architecture from slippi-ai.

  Alternates LSTM (or GRU) layers with feed-forward residual blocks. This is
  the production architecture used by slippi-ai's competitive Melee bots
  (`tx_like` in `networks.py`).

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Dense projection -> hidden_size  (if embed_dim != hidden_size)
        |
  Per layer:
    1. x = x + LSTM(x)                              # bare residual recurrence
    2. x = x + [LayerNorm -> up -> act -> down](x)   # pre-norm FFN residual
        |
  Final LayerNorm -> last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = TransformerLike.build(
        embed_dim: 288,
        hidden_size: 512,
        num_layers: 3,
        ffn_multiplier: 2,
        activation: :gelu
      )

  ## References

  - vladfi1/slippi-ai `networks.py` — `TransformerLike` / `tx_like`
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 512
  @default_num_layers 3
  @default_ffn_multiplier 2
  @default_activation :gelu
  @default_dropout 0.0
  @default_cell_type :lstm
  @default_norm :layer_norm
  @default_window_size 60

  @doc """
  Build a TransformerLike model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Residual stream width (default: 512)
    - `:num_layers` - Number of recurrent+FFN pairs (default: 3)
    - `:cell_type` - `:lstm` or `:gru` (default: `:lstm`)
    - `:ffn_multiplier` - FFN expansion factor (default: 2)
    - `:activation` - FFN activation (default: `:gelu`)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:norm` - `:layer_norm` or `:rms_norm` (default: `:layer_norm`)
    - `:recurrent_norm` - Apply norm before recurrence (default: false)
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
          | {:cell_type, :lstm | :gru}
          | {:ffn_multiplier, pos_integer()}
          | {:activation, atom()}
          | {:dropout, float()}
          | {:norm, :layer_norm | :rms_norm}
          | {:recurrent_norm, boolean()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    ffn_multiplier = Keyword.get(opts, :ffn_multiplier, @default_ffn_multiplier)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    norm = Keyword.get(opts, :norm, @default_norm)
    recurrent_norm = Keyword.get(opts, :recurrent_norm, false)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_dim})

    # Project to hidden_size if dimensions differ
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack num_layers of recurrent + FFN blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        acc
        |> recurrent_block(hidden_size, cell_type, dropout, norm, recurrent_norm, layer_idx)
        |> ffn_block(hidden_size, ffn_multiplier, activation, dropout, norm, layer_idx)
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

  # Block 1: optional_norm -> LSTM/GRU -> dropout -> residual add
  defp recurrent_block(x, hidden_size, cell_type, dropout, norm, recurrent_norm, layer_idx) do
    rnn_input =
      if recurrent_norm do
        apply_norm(x, norm, hidden_size, "rnn_#{layer_idx}_prenorm")
      else
        x
      end

    {output_seq, _hidden} =
      case cell_type do
        :lstm -> Axon.lstm(rnn_input, hidden_size, name: "lstm_#{layer_idx}")
        :gru -> Axon.gru(rnn_input, hidden_size, name: "gru_#{layer_idx}")
      end

    output_seq =
      if dropout > 0 do
        Axon.dropout(output_seq, rate: dropout, name: "rnn_#{layer_idx}_dropout")
      else
        output_seq
      end

    Axon.add(x, output_seq, name: "rnn_#{layer_idx}_residual")
  end

  # Block 2: norm -> FFN(up -> act -> down) -> dropout -> residual add
  defp ffn_block(x, hidden_size, ffn_multiplier, activation, dropout, norm, layer_idx) do
    normed = apply_norm(x, norm, hidden_size, "ffn_#{layer_idx}_prenorm")

    ffn_out =
      FFN.layer(normed,
        hidden_size: hidden_size,
        expansion_factor: ffn_multiplier,
        activation: activation,
        dropout: dropout,
        name: "ffn_#{layer_idx}"
      )

    Axon.add(x, ffn_out, name: "ffn_#{layer_idx}_residual")
  end

  defp apply_norm(x, :layer_norm, _hidden_size, name),
    do: Axon.layer_norm(x, name: name, epsilon: 1.0e-6)

  defp apply_norm(x, :rms_norm, hidden_size, name),
    do: Edifice.Blocks.RMSNorm.layer(x, hidden_size: hidden_size, name: name)

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
