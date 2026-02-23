defmodule Edifice.Recurrent.NativeRecurrence do
  @moduledoc """
  Native Recurrence — unified module for multiple minimal recurrence types.

  Provides three lightweight recurrence variants under a single interface,
  each using a sequential scan over timesteps with different gating mechanics.

  ## Recurrence Types

  - `:elu_gru` — ELU-gated GRU: `z=sigmoid(Wz·x), c=1+elu(Wc·x), h=(1-z)·h+z·c`
  - `:real_gru` — Real-valued MinGRU: `z=sigmoid(Wz·x), c=Wc·x, h=(1-z)·h+z·c`
  - `:diag_linear` — Diagonal linear recurrence: `h=sigmoid(Wa·x)·h + Wb·x`

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection -> hidden_size
        |
  Per layer: pre-norm -> gate+candidate projections -> scan -> residual
        |
  Final norm -> last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = NativeRecurrence.build(
        embed_dim: 256,
        hidden_size: 256,
        num_layers: 4,
        recurrence_type: :elu_gru
      )

  ## References

  - Feng et al., "Were RNNs All We Needed?" (2024) — MinGRU/MinLSTM
  - Orvieto et al., "Resurrecting Recurrent Neural Networks for Long Sequences" (2023)
  """

  @default_hidden_size 256
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Native Recurrence model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of recurrent layers (default: 4)
    - `:recurrence_type` - Recurrence variant: `:elu_gru`, `:real_gru`, or `:diag_linear` (default: `:elu_gru`)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:recurrence_type, :elu_gru | :real_gru | :diag_linear}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    recurrence_type = Keyword.get(opts, :recurrence_type, :elu_gru)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_recurrence_layer(acc, hidden_size, recurrence_type, "nr_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    output = Axon.layer_norm(output, name: "final_norm")

    Axon.nx(
      output,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp build_recurrence_layer(input, hidden_size, recurrence_type, name) do
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    case recurrence_type do
      :elu_gru -> build_elu_gru(normed, input, hidden_size, name)
      :real_gru -> build_real_gru(normed, input, hidden_size, name)
      :diag_linear -> build_diag_linear(normed, input, hidden_size, name)
    end
  end

  # ELU GRU: z=sigmoid(Wz·x), c=1+elu(Wc·x), h=(1-z)·h+z·c
  defp build_elu_gru(normed, residual, hidden_size, name) do
    gate_proj = Axon.dense(normed, hidden_size, name: "#{name}_gate")
    candidate_proj = Axon.dense(normed, hidden_size, name: "#{name}_candidate")

    recurrence_input = Axon.concatenate([gate_proj, candidate_proj], axis: 2, name: "#{name}_cat")

    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined -> elu_gru_scan(combined, hidden_size) end,
        name: "#{name}_recurrence"
      )

    Axon.add(residual, recurrence_output, name: "#{name}_residual")
  end

  defp elu_gru_scan(combined, hidden_size) do
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    gate_pre = Nx.slice_along_axis(combined, 0, hidden_size, axis: 2)
    candidate_pre = Nx.slice_along_axis(combined, hidden_size, hidden_size, axis: 2)

    z = Nx.sigmoid(gate_pre)
    # 1 + elu(x) ensures candidate is non-negative: elu(x) >= -1, so c >= 0
    c = Nx.add(1.0, Nx.max(candidate_pre, 0.0) |> Nx.add(Nx.min(candidate_pre, 0.0) |> Nx.exp() |> Nx.subtract(1.0)))

    h_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Real GRU (MinGRU): z=sigmoid(Wz·x), c=Wc·x, h=(1-z)·h+z·c
  defp build_real_gru(normed, residual, hidden_size, name) do
    gate_proj = Axon.dense(normed, hidden_size, name: "#{name}_gate")
    candidate_proj = Axon.dense(normed, hidden_size, name: "#{name}_candidate")

    recurrence_input = Axon.concatenate([gate_proj, candidate_proj], axis: 2, name: "#{name}_cat")

    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined -> real_gru_scan(combined, hidden_size) end,
        name: "#{name}_recurrence"
      )

    Axon.add(residual, recurrence_output, name: "#{name}_residual")
  end

  defp real_gru_scan(combined, hidden_size) do
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    gate_pre = Nx.slice_along_axis(combined, 0, hidden_size, axis: 2)
    candidate = Nx.slice_along_axis(combined, hidden_size, hidden_size, axis: 2)

    z = Nx.sigmoid(gate_pre)

    h_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidate, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Diagonal linear recurrence: h=sigmoid(Wa·x)·h + Wb·x
  defp build_diag_linear(normed, residual, hidden_size, name) do
    a_proj = Axon.dense(normed, hidden_size, name: "#{name}_a")
    b_proj = Axon.dense(normed, hidden_size, name: "#{name}_b")

    recurrence_input = Axon.concatenate([a_proj, b_proj], axis: 2, name: "#{name}_cat")

    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined -> diag_linear_scan(combined, hidden_size) end,
        name: "#{name}_recurrence"
      )

    Axon.add(residual, recurrence_output, name: "#{name}_residual")
  end

  defp diag_linear_scan(combined, hidden_size) do
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    a_pre = Nx.slice_along_axis(combined, 0, hidden_size, axis: 2)
    b_val = Nx.slice_along_axis(combined, hidden_size, hidden_size, axis: 2)

    a = Nx.sigmoid(a_pre)

    h_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        b_t = Nx.slice_along_axis(b_val, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      recurrence_type: :elu_gru,
      dropout: 0.1,
      window_size: 60
    ]
  end
end
