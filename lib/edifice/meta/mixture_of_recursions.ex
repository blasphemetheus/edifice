defmodule Edifice.Meta.MixtureOfRecursions do
  @moduledoc """
  Mixture of Recursions (MoR): Dynamic recursive depth with per-token routing.

  <!-- verified: true, date: 2026-02-28 -->

  A weight-tied recursive transformer where a lightweight router decides how
  many recursion steps each token receives. Tokens with high router scores
  get more computation (deeper recursion), while simple tokens exit early.

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        v
  Projection -> [batch, seq, hidden_size]
        |
        v
  Unique First Layer (Phi_0)
        |
        v
  +-------------------------------------------+
  | For recursion r = 1..N_r:                  |
  |   Router[r]: linear -> sigmoid -> alpha    |
  |   Top-k selection (active set shrinks)     |
  |   Shared Block (Phi'): TransformerBlock    |
  |   H = gate * block(H) + H                 |
  +-------------------------------------------+
        |
        v
  Unique Last Layer (Phi_{L-1})
        |
        v
  Final Norm -> Last Timestep
  Output [batch, hidden_size]
  ```

  ## Middle-Cycle Weight Sharing

  The first and last transformer layers have unique parameters. All
  intermediate layers share the same weights (weight-tied), applied
  N_r times. This reduces parameters by ~1/N_r while maintaining
  expressiveness at the boundaries.

  ## Routing Variants

  ### Expert-Choice (default, recommended)

  At each recursion step, the router scores all active tokens and selects
  the top-k to continue. The active set shrinks hierarchically:

      Step 1: process 3/3 tokens
      Step 2: process 2/3 tokens
      Step 3: process 1/3 tokens

  ### Token-Choice

  A single routing decision at step 1 commits each token to a total
  recursion depth via softmax over N_r choices.

  ## References

  - Bae et al., "Mixture-of-Recursions: Learning Dynamic Recursive Depths
    for Adaptive Token-Level Computation" (NeurIPS 2025)
  - https://arxiv.org/abs/2507.10524
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_recursions 3
  @default_num_layers 6
  @default_dropout 0.1
  @default_alpha 0.1
  @default_window_size 60

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:alpha, float()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_recursions, pos_integer()}
          | {:routing, :expert_choice | :token_choice}
          | {:window_size, pos_integer()}

  @doc """
  Build a Mixture of Recursions model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_recursions` - Number of recursion steps N_r (default: 3)
    - `:num_layers` - Total original layers; first+last are unique (default: 6)
    - `:routing` - Routing variant: `:expert_choice` or `:token_choice`
      (default: `:expert_choice`)
    - `:alpha` - Gate scaling factor (default: 0.1)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_recursions = Keyword.get(opts, :num_recursions, @default_num_recursions)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    routing = Keyword.get(opts, :routing, :expert_choice)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    input = Axon.input("input", shape: {nil, window_size, embed_dim})

    # Project to hidden_size if needed
    x =
      if embed_dim == hidden_size do
        input
      else
        Axon.dense(input, hidden_size, name: "input_proj")
      end

    # Unique first layer (Phi_0)
    x = build_unique_layer(x, hidden_size, num_heads, dropout, "first_layer")

    # Recursive middle layers with routing
    # Weight sharing: all recursion steps use the same "shared_block" name
    # so Axon reuses the same parameters
    x =
      case routing do
        :expert_choice ->
          build_expert_choice(x, hidden_size, num_heads, num_recursions,
            num_layers: num_layers,
            alpha: alpha,
            dropout: dropout
          )

        :token_choice ->
          build_token_choice(x, hidden_size, num_heads, num_recursions,
            num_layers: num_layers,
            alpha: alpha,
            dropout: dropout
          )
      end

    # Unique last layer (Phi_{L-1})
    x = build_unique_layer(x, hidden_size, num_heads, dropout, "last_layer")

    # Final norm and last-timestep selection
    x
    |> Axon.layer_norm(name: "final_norm")
    |> Axon.nx(fn t ->
      seq_len = Nx.axis_size(t, 1)
      t[[.., seq_len - 1, ..]]
    end)
  end

  # ==========================================================================
  # Expert-Choice Routing
  # ==========================================================================

  # At each recursion step, route active tokens. Active set shrinks
  # hierarchically: step r processes (N_r - r + 1) / N_r of tokens.
  defp build_expert_choice(x, hidden_size, num_heads, num_recursions, opts) do
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    Enum.reduce(1..num_recursions, x, fn r, acc ->
      # Per-recursion router: linear -> sigmoid -> alpha scaling
      gate =
        acc
        |> Axon.dense(1, name: "router_#{r}")
        |> Axon.sigmoid(name: "router_#{r}_sigmoid")

      # Top-k masking: keep (N_r - r + 1) / N_r fraction
      remaining_frac = (num_recursions - r + 1) / num_recursions

      gate =
        Axon.nx(
          gate,
          fn g ->
            # g: [batch, seq, 1]
            g_sq = Nx.squeeze(g, axes: [2])
            seq_len = Nx.axis_size(g_sq, 1)
            k = max(1, round(remaining_frac * seq_len))
            {top_vals, _} = Nx.top_k(g_sq, k: k)
            threshold = Nx.reduce_min(top_vals, axes: [1], keep_axes: true)
            mask = Nx.greater_equal(g_sq, threshold)
            masked = Nx.select(mask, g_sq, Nx.broadcast(0.0, Nx.shape(g_sq)))
            Nx.multiply(Nx.new_axis(masked, 2), alpha)
          end,
          name: "router_#{r}_topk"
        )

      # Shared transformer block (same name = weight sharing across recursions)
      block_out =
        TransformerBlock.layer(acc,
          attention_fn: fn inp, attn_name ->
            MultiHead.self_attention(inp,
              hidden_size: hidden_size,
              num_heads: num_heads,
              dropout: dropout,
              causal: true,
              name: attn_name
            )
          end,
          hidden_size: hidden_size,
          dropout: dropout,
          name: "shared_block"
        )

      # Gated residual: gate * block_output + input
      Axon.layer(
        fn block, inp, g, _opts ->
          Nx.add(Nx.multiply(g, block), inp)
        end,
        [block_out, acc, gate],
        name: "recursion_#{r}_gate",
        op_name: :mor_gate
      )
    end)
  end

  # ==========================================================================
  # Token-Choice Routing
  # ==========================================================================

  # Single routing decision at step 1 assigns each token a recursion depth.
  # Implemented as soft gating: deeper tokens get more gate weight.
  defp build_token_choice(x, hidden_size, num_heads, num_recursions, opts) do
    alpha = Keyword.get(opts, :alpha, 1.0)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Router produces N_r scores per token, softmax over recursion depths
    depth_scores =
      x
      |> Axon.dense(num_recursions, name: "token_router")
      |> Axon.activation(:softmax, name: "token_router_softmax")

    # Unroll recursion steps with cumulative gating
    Enum.reduce(1..num_recursions, x, fn r, acc ->
      # Gate for this recursion step: sum of probabilities for depths >= r
      # Tokens assigned to depth r or deeper get processed at step r
      gate =
        Axon.nx(
          depth_scores,
          fn scores ->
            # scores: [batch, seq, num_recursions]
            # Sum probabilities for all depths >= r (0-indexed: r-1..end)
            cumulative = Nx.sum(scores[[.., .., (r - 1)..-1//1]], axes: [2], keep_axes: true)
            Nx.multiply(cumulative, alpha)
          end,
          name: "token_gate_#{r}"
        )

      # Shared transformer block
      block_out =
        TransformerBlock.layer(acc,
          attention_fn: fn inp, attn_name ->
            MultiHead.self_attention(inp,
              hidden_size: hidden_size,
              num_heads: num_heads,
              dropout: dropout,
              causal: true,
              name: attn_name
            )
          end,
          hidden_size: hidden_size,
          dropout: dropout,
          name: "shared_block"
        )

      # Gated residual
      Axon.layer(
        fn block, inp, g, _opts ->
          Nx.add(Nx.multiply(g, block), inp)
        end,
        [block_out, acc, gate],
        name: "recursion_#{r}_gate",
        op_name: :mor_gate
      )
    end)
  end

  # ==========================================================================
  # Unique (non-shared) transformer layer
  # ==========================================================================

  defp build_unique_layer(x, hidden_size, num_heads, dropout, name) do
    TransformerBlock.layer(x,
      attention_fn: fn inp, attn_name ->
        MultiHead.self_attention(inp,
          hidden_size: hidden_size,
          num_heads: num_heads,
          dropout: dropout,
          causal: true,
          name: attn_name
        )
      end,
      hidden_size: hidden_size,
      dropout: dropout,
      name: name
    )
  end

  @doc "Get the output size of a MixtureOfRecursions model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
