defmodule Edifice.Recurrent.TTTE2E do
  @moduledoc """
  TTT-E2E: End-to-End Test-Time Training for Long Context.

  Implements the TTT-E2E architecture from "End-to-End Test-Time Training
  for Long Context" (Stanford, NVIDIA, UC Berkeley, Dec 2025). Unlike the
  original TTT layers (which replace attention with self-supervised inner
  model updates), TTT-E2E keeps a standard transformer backbone and mutates
  ~25% of its MLP layers at inference time using end-to-end gradient descent.

  ## Key Differences from TTT-Linear/TTT-MLP

  | Aspect | TTT-Linear/MLP | TTT-E2E |
  |--------|---------------|---------|
  | Where TTT happens | Custom layer replacing attention | Updates existing MLP in last 1/4 blocks |
  | Inner loss | Layer-wise reconstruction | End-to-end next-token prediction |
  | Architecture | Custom TTT layer | Standard transformer + dual MLP |
  | Training | Standard pretraining | Meta-learning (bilevel optimization) |

  ## Architecture: Dual-MLP Blocks

  In the last 1/4 of transformer blocks, each MLP sublayer is split into:

  - **Dynamic MLP**: Updated via SGD at inference (stores document context)
  - **Static MLP**: Frozen at inference (preserves pretrained knowledge)

  Both MLPs receive the same input; their outputs are summed. This prevents
  catastrophic forgetting while allowing the model to adapt to new context.

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +----------------------------------------------+
  |  Frozen Block 1..N*3/4                        |
  |    LayerNorm -> SlidingWindowAttn -> Residual  |
  |    LayerNorm -> MLP -> Residual                |
  +----------------------------------------------+
        |
        v
  +----------------------------------------------+
  |  Mutable Block N*3/4+1..N                     |
  |    LayerNorm -> SlidingWindowAttn -> Residual  |
  |    LayerNorm -> (DynamicMLP + StaticMLP)       |
  |    -> Residual                                 |
  +----------------------------------------------+
        |
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Inference Protocol

  1. Reset dynamic MLP weights to W0 at start of each document
  2. Process tokens in mini-batches of size b (default: 1024)
  3. After each mini-batch: compute next-token loss, backprop to dynamic
     MLP params only, apply SGD step
  4. Dynamic MLPs accumulate context throughout the document

  ## Usage

      model = TTTE2E.build(
        embed_dim: 256,
        hidden_size: 256,
        num_layers: 12,       # Last 3 blocks will have dual MLPs
        num_heads: 4,
        window_size: 60
      )

  ## References
  - Paper: https://arxiv.org/abs/2512.23675
  - Code: https://github.com/test-time-training/e2e
  """

  alias Edifice.Attention.MultiHead, as: Attention

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_hidden_size 256
  @default_num_layers 12
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1
  @default_mlp_ratio 4
  @default_mutable_fraction 0.25

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a TTT-E2E model.

  ## Options

  **Architecture:**
    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Total number of transformer blocks (default: 12)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:mlp_ratio` - MLP expansion ratio (default: 4)

  **TTT-specific:**
    - `:mutable_fraction` - Fraction of blocks with dual MLPs (default: 0.25).
      Mutable blocks are placed at the end of the stack.

  **General:**
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sliding window attention size (default: 60)
    - `:seq_len` - Fixed sequence length for JIT (default: window_size)

  ## Returns
    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:mlp_ratio, pos_integer()}
          | {:mutable_fraction, float()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    mutable_fraction = Keyword.get(opts, :mutable_fraction, @default_mutable_fraction)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Determine which blocks are mutable (last mutable_fraction of layers)
    num_mutable = max(round(num_layers * mutable_fraction), 1)
    first_mutable = num_layers - num_mutable + 1

    attn_hidden_dim = num_heads * head_dim

    # Pre-compute sliding window attention mask
    precomputed_mask =
      if seq_len do
        Attention.window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
      else
        nil
      end

    # Input
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Positional encoding
    x = Attention.add_positional_encoding(x, name: "pos_encoding")

    # Build transformer blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        is_mutable = layer_idx >= first_mutable

        build_ttte2e_block(acc,
          hidden_size: hidden_size,
          attn_hidden_dim: attn_hidden_dim,
          num_heads: num_heads,
          head_dim: head_dim,
          mlp_ratio: mlp_ratio,
          dropout: dropout,
          window_size: window_size,
          precomputed_mask: precomputed_mask,
          mutable: is_mutable,
          name: "block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep
    Axon.nx(
      x,
      fn tensor ->
        seq_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_actual - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # Block Building
  # ============================================================================

  defp build_ttte2e_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    attn_hidden_dim = Keyword.fetch!(opts, :attn_hidden_dim)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    mlp_ratio = Keyword.fetch!(opts, :mlp_ratio)
    dropout = Keyword.fetch!(opts, :dropout)
    window_size = Keyword.fetch!(opts, :window_size)
    precomputed_mask = Keyword.get(opts, :precomputed_mask)
    mutable = Keyword.fetch!(opts, :mutable)
    name = Keyword.fetch!(opts, :name)

    # ---- Attention sub-block ----
    attn_input = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_input =
      if hidden_size != attn_hidden_dim do
        Axon.dense(attn_input, attn_hidden_dim, name: "#{name}_attn_proj_in")
      else
        attn_input
      end

    attended =
      Attention.sliding_window_attention(attn_input,
        window_size: window_size,
        num_heads: num_heads,
        head_dim: head_dim,
        mask: precomputed_mask,
        qk_layernorm: true,
        name: "#{name}_attn"
      )

    attended =
      if hidden_size != attn_hidden_dim do
        Axon.dense(attended, hidden_size, name: "#{name}_attn_proj_out")
      else
        attended
      end

    attended =
      if dropout > 0 do
        Axon.dropout(attended, rate: dropout, name: "#{name}_attn_drop")
      else
        attended
      end

    x = Axon.add(input, attended, name: "#{name}_attn_residual")

    # ---- MLP sub-block ----
    mlp_input = Axon.layer_norm(x, name: "#{name}_mlp_norm")
    ffn_dim = hidden_size * mlp_ratio

    if mutable do
      # Dual-MLP: dynamic (mutable at inference) + static (frozen)
      # Both receive the same input, outputs are summed.
      # During standard training, both are trainable. At inference,
      # only dynamic MLP weights get updated via SGD.

      dynamic_mlp =
        mlp_input
        |> Axon.dense(ffn_dim, name: "#{name}_dynamic_mlp_up")
        |> Axon.activation(:silu, name: "#{name}_dynamic_silu")
        |> Axon.dense(hidden_size, name: "#{name}_dynamic_mlp_down")

      static_mlp =
        mlp_input
        |> Axon.dense(ffn_dim, name: "#{name}_static_mlp_up")
        |> Axon.activation(:silu, name: "#{name}_static_silu")
        |> Axon.dense(hidden_size, name: "#{name}_static_mlp_down")

      # Sum of both MLPs
      mlp_out = Axon.add(dynamic_mlp, static_mlp, name: "#{name}_dual_mlp_sum")

      mlp_out =
        if dropout > 0 do
          Axon.dropout(mlp_out, rate: dropout, name: "#{name}_mlp_drop")
        else
          mlp_out
        end

      Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
    else
      # Standard single MLP (frozen at inference)
      mlp_out =
        mlp_input
        |> Axon.dense(ffn_dim, name: "#{name}_mlp_up")
        |> Axon.activation(:silu, name: "#{name}_mlp_silu")
        |> Axon.dense(hidden_size, name: "#{name}_mlp_down")

      mlp_out =
        if dropout > 0 do
          Axon.dropout(mlp_out, rate: dropout, name: "#{name}_mlp_drop")
        else
          mlp_out
        end

      Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the names of mutable (dynamic MLP) parameters for a built model.

  These are the parameters that should be updated via SGD at inference time.
  Use this to partition parameters into frozen and mutable sets.

  ## Options
    - `:num_layers` - Total layers (default: 12)
    - `:mutable_fraction` - Fraction of mutable blocks (default: 0.25)

  ## Returns
    List of parameter name prefixes for dynamic MLP layers.
  """
  @spec mutable_param_prefixes(keyword()) :: [String.t()]
  def mutable_param_prefixes(opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    mutable_fraction = Keyword.get(opts, :mutable_fraction, @default_mutable_fraction)

    num_mutable = max(round(num_layers * mutable_fraction), 1)
    first_mutable = num_layers - num_mutable + 1

    for layer_idx <- first_mutable..num_layers do
      "block_#{layer_idx}_dynamic"
    end
  end

  @doc """
  Get the layer pattern showing which blocks are mutable.

  ## Example

      iex> TTTE2E.layer_pattern(num_layers: 8, mutable_fraction: 0.25)
      [:frozen, :frozen, :frozen, :frozen, :frozen, :frozen, :mutable, :mutable]
  """
  @spec layer_pattern(keyword()) :: [atom()]
  def layer_pattern(opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    mutable_fraction = Keyword.get(opts, :mutable_fraction, @default_mutable_fraction)

    num_mutable = max(round(num_layers * mutable_fraction), 1)
    first_mutable = num_layers - num_mutable + 1

    Enum.map(1..num_layers, fn idx ->
      if idx >= first_mutable, do: :mutable, else: :frozen
    end)
  end

  @doc """
  Get the output size of a TTT-E2E model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
