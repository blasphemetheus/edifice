defmodule Edifice.Feedforward.MLP do
  @moduledoc """
  Multi-Layer Perceptron (feedforward neural network).

  The simplest deep learning architecture - stacked dense layers with
  nonlinear activations. Despite their simplicity, MLPs remain effective
  for many tasks and serve as building blocks in more complex architectures.

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +-------------------+
  | Dense + Act + Drop|  Layer 1
  +-------------------+
        |
        v
  +-------------------+
  | Dense + Act + Drop|  Layer 2
  +-------------------+
        |
        v
  Output [batch, last_hidden_size]
  ```

  ## Options

  - `:input_size` - Input feature dimension (required for standalone build)
  - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.1)
  - `:layer_norm` - Apply layer normalization (default: false)
  - `:residual` - Add residual/skip connections (default: false)

  ## Usage

      # Standalone model
      model = MLP.build(input_size: 256, hidden_sizes: [512, 256])

      # As a backbone from an existing input
      backbone = MLP.build_backbone(input, [512, 256], :relu, 0.1)
  """

  @default_hidden_sizes [512, 512]
  @default_activation :relu
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:hidden_sizes, [pos_integer()]}
          | {:activation, atom()}
          | {:dropout, float()}
          | {:layer_norm, boolean()}
          | {:residual, boolean()}

  @doc """
  Build a standalone MLP model.

  ## Options
    - `:input_size` - Input dimension (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function atom (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:layer_norm` - Apply layer normalization after each dense layer (default: false)
    - `:residual` - Add residual connections between layers (default: false)

  ## Returns
    An Axon model outputting `[batch, last_hidden_size]`.

  ## Examples

      iex> model = Edifice.Feedforward.MLP.build(input_size: 16, hidden_sizes: [32])
      iex> {init_fn, predict_fn} = Axon.build(model)
      iex> params = init_fn.(Nx.template({1, 16}, :f32), Axon.ModelState.empty())
      iex> output = predict_fn.(params, Nx.broadcast(0.5, {1, 16}))
      iex> Nx.shape(output)
      {1, 32}
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    input = Axon.input("input", shape: {nil, input_size})
    build_backbone(input, hidden_sizes, activation, dropout, opts)
  end

  @doc """
  Build a temporal MLP that processes sequences by taking the last frame.

  Useful for single-frame processing in temporal pipelines.

  ## Options
    - `:embed_dim` - Input embedding size per frame (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [512, 512])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
  """
  @spec build_temporal(keyword()) :: Axon.t()
  def build_temporal(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Take last frame: [batch, embed_dim]
    last_frame =
      Axon.nx(
        input,
        fn tensor ->
          seq_len_actual = Nx.axis_size(tensor, 1)

          Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "last_frame"
      )

    # Process with MLP
    build_backbone(last_frame, hidden_sizes, activation, dropout, opts)
  end

  @doc """
  Build an MLP backbone from an existing Axon input layer.

  This is the core builder used by both `build/1` and `build_temporal/1`.

  ## Options
    - `:layer_norm` - Apply layer normalization (default: false)
    - `:residual` - Add residual connections (default: false)
  """
  @spec build_backbone(Axon.t(), list(), atom(), float(), keyword()) :: Axon.t()
  def build_backbone(input, hidden_sizes, activation, dropout, opts \\ []) do
    layer_norm = Keyword.get(opts, :layer_norm, false)
    residual = Keyword.get(opts, :residual, false)

    {final_layer, _} =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce({input, nil}, fn {size, idx}, {acc, prev_size} ->
        layer = Axon.dense(acc, size, name: "mlp_dense_#{idx}")

        layer =
          if layer_norm do
            Axon.layer_norm(layer, name: "mlp_ln_#{idx}")
          else
            layer
          end

        layer =
          layer
          |> Axon.activation(activation)
          |> Axon.dropout(rate: dropout)

        layer =
          if residual do
            add_residual_connection(acc, layer, prev_size, size, idx)
          else
            layer
          end

        {layer, size}
      end)

    final_layer
  end

  @doc """
  Get the output size of an MLP with the given hidden sizes.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    List.last(hidden_sizes)
  end

  # Add a residual connection, projecting if dimensions differ
  defp add_residual_connection(input, layer, prev_size, current_size, idx) do
    if prev_size == current_size do
      Axon.add(input, layer, name: "mlp_residual_#{idx}")
    else
      projected = Axon.dense(input, current_size, name: "mlp_residual_proj_#{idx}")
      Axon.add(projected, layer, name: "mlp_residual_#{idx}")
    end
  end
end
