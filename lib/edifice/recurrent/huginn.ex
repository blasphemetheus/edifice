defmodule Edifice.Recurrent.Huginn do
  @moduledoc """
  Huginn: Depth-Recurrent Transformer with Adaptive Iteration.

  Implements the Huginn architecture from "Scaling up Test-Time Compute with
  Latent Reasoning: A Recurrent Depth Approach" (Geiping et al., 2025).

  ## Key Innovation

  Replaces deep stacking of unique transformer layers with a small set of
  weight-tied recurrent blocks that iterate r times. A small model with
  (2, 4, 2) layer config and r=32 iterations simulates a 132-effective-layer
  model. Reasoning happens in latent space, requiring no specialized
  chain-of-thought training data.

  ## Architecture

  Three-phase design: Prelude -> Core (iterated) -> Coda

  ```
  Input [batch, seq_len, embed_dim]
        |
  +============================+
  | PRELUDE (l_p layers)       |  <- unique transformer layers
  +============================+
        |
    e = prelude output
    s_0 = zeros (initial latent state)
        |
  +============================+
  | CORE (l_r layers x r iter) |  <- weight-tied, iterated
  |                            |
  | for i = 1 to r:           |
  |   input = Adapter([s; e]) |  <- concat + project
  |   s = CoreBlock(input)    |  <- shared-weight layers
  |   s = RMSNorm(s)          |  <- rescaling
  +============================+
        |
  +============================+
  | CODA (l_c layers)         |  <- unique transformer layers
  +============================+
        |
  [batch, hidden_size]  (last timestep)
  ```

  Weight tying is achieved via Axon's name-based parameter sharing: core layers
  use the same names across iterations, so parameters are automatically shared.

  ## Usage

      model = Huginn.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        num_iterations: 4
      )

  ## References
  - Paper: https://arxiv.org/abs/2502.05171
  - Code: https://github.com/seal-rg/recurrent-pretraining
  """

  alias Edifice.Blocks.TransformerBlock
  alias Edifice.Attention.MultiHead

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:mlp_dim, pos_integer()}
          | {:prelude_layers, pos_integer()}
          | {:core_layers, pos_integer()}
          | {:coda_layers, pos_integer()}
          | {:num_iterations, pos_integer()}
          | {:dropout, float()}
          | {:seq_len, pos_integer()}

  @doc """
  Build a Huginn depth-recurrent transformer.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Attention heads (default: 8)
    - `:head_dim` - Per-head dimension (default: hidden_size / num_heads)
    - `:mlp_dim` - SwiGLU inner dimension (default: 4 * hidden_size)
    - `:prelude_layers` - Number of unique prelude layers (default: 2)
    - `:core_layers` - Layers per core iteration (default: 4)
    - `:coda_layers` - Number of unique coda layers (default: 2)
    - `:num_iterations` - Core recurrence depth r (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:seq_len` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.

  ## Examples

      iex> model = Edifice.Recurrent.Huginn.build(embed_dim: 32, hidden_size: 16, num_heads: 4, head_dim: 4, num_iterations: 2)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    prelude_layers = Keyword.get(opts, :prelude_layers, 2)
    core_layers = Keyword.get(opts, :core_layers, 4)
    coda_layers = Keyword.get(opts, :coda_layers, 2)
    num_iterations = Keyword.get(opts, :num_iterations, 4)
    seq_len = Keyword.get(opts, :seq_len, 60)

    input_seq_dim = if seq_len, do: seq_len, else: nil
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden_size if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Phase 1: Prelude -- unique transformer layers
    embedding =
      build_transformer_stack(x, prelude_layers, "prelude", opts)

    # Phase 2: Core -- weight-tied iterations
    # Initialize latent state s_0 as zeros (same shape as embedding)
    state = Axon.nx(embedding, fn t -> Nx.broadcast(0.0, Nx.shape(t)) end, name: "init_state")

    # Iterate core block r times with weight-tied layers
    # Same layer names across iterations -> shared parameters in Axon
    state =
      Enum.reduce(1..num_iterations, state, fn _iter, s ->
        # Adapter: concatenate [s; embedding] -> project to hidden_size
        combined = Axon.concatenate([s, embedding], axis: 2, name: "core_adapter_concat")
        adapted = Axon.dense(combined, hidden_size, name: "core_adapter_proj")

        # Core transformer layers (weight-tied via same names)
        s_new = build_transformer_stack(adapted, core_layers, "core", opts)

        # RMSNorm rescaling after each iteration (weight-tied)
        Axon.layer_norm(s_new, name: "core_rescale_norm")
      end)

    # Phase 3: Coda -- unique transformer layers
    output = build_transformer_stack(state, coda_layers, "coda", opts)

    # Extract last timestep
    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # Build a stack of transformer blocks with causal self-attention + SwiGLU FFN
  defp build_transformer_stack(input, num_layers, prefix, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 8)
    head_dim = Keyword.get(opts, :head_dim, div(hidden_size, num_heads))
    dropout = Keyword.get(opts, :dropout, 0.0)

    attn_fn = fn x, name ->
      attn_dim = num_heads * head_dim

      out =
        MultiHead.sliding_window_attention(x,
          num_heads: num_heads,
          head_dim: head_dim,
          window_size: Keyword.get(opts, :seq_len, 60),
          name: name
        )

      # Project to hidden_size if attention dim differs
      if attn_dim != hidden_size do
        Axon.dense(out, hidden_size, name: "#{name}_proj")
      else
        out
      end
    end

    TransformerBlock.stack(input, num_layers,
      attention_fn: attn_fn,
      hidden_size: hidden_size,
      ffn_type: :gated,
      dropout: dropout,
      name: prefix
    )
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Huginn model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, 256)
  end

  @doc """
  Calculate approximate parameter count for a Huginn model.

  Note: core layers are weight-tied, so their parameters are counted once.
  The effective depth is `prelude + core * num_iterations + coda` but the
  parameter count is `prelude + core + coda` unique layers.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 8)
    head_dim = Keyword.get(opts, :head_dim, div(hidden_size, num_heads))
    mlp_dim = Keyword.get(opts, :mlp_dim, hidden_size * 4)
    prelude_layers = Keyword.get(opts, :prelude_layers, 2)
    core_layers = Keyword.get(opts, :core_layers, 4)
    coda_layers = Keyword.get(opts, :coda_layers, 2)

    attn_dim = num_heads * head_dim

    # Per transformer layer: QKV proj + output proj + FFN (gate + up + down)
    per_layer =
      3 * hidden_size * attn_dim +
        attn_dim * hidden_size +
        3 * hidden_size * mlp_dim

    # Adapter: 2h -> h
    adapter_params = 2 * hidden_size * hidden_size

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    # Core layers counted once (weight-tied), adapter counted once
    unique_layers = prelude_layers + core_layers + coda_layers

    input_proj + unique_layers * per_layer + adapter_params
  end
end
