defmodule Edifice.SSM.S5 do
  @moduledoc """
  S5: Simplified State Space Sequence model.

  S5 uses a single multi-input, multi-output (MIMO) state space model
  instead of the many independent SISO systems used in Mamba. This
  results in a simpler architecture while maintaining strong performance.

  ## Key Innovation: MIMO SSM

  Instead of having many parallel single-input single-output SSMs
  (like Mamba), S5 uses one large MIMO SSM:

  ```
  Mamba: D separate SSMs, each with state size N
  S5: 1 large SSM with D*N combined state
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        │
        ▼
  ┌─────────────────────────────────────┐
  │  S5 Block                            │
  │                                      │
  │  Linear projection → Encoder         │
  │                                      │
  │  ┌─ MIMO SSM ────────────────────┐   │
  │  │                               │   │
  │  │  x'(t) = Ax(t) + Bu(t)        │   │
  │  │  y(t) = Cx(t) + Du(t)         │   │
  │  │                               │   │
  │  │  (Diagonal A for efficiency)  │   │
  │  │                               │   │
  │  └───────────────────────────────┘   │
  │                                      │
  │  Decoder → Linear projection         │
  │                                      │
  └─────────────────────────────────────┘
        │ (repeat for num_layers)
        ▼
  [batch, hidden_size]
  ```

  ## Complexity

  | Aspect | Value |
  |--------|-------|
  | Training | O(L log L) via FFT or O(L) via scan |
  | Inference | O(1) per step |
  | Parameters | Fewer than Mamba |

  ## Key Difference from Mamba

  | Aspect | S5 | Mamba |
  |--------|-----|-------|
  | SSM structure | MIMO | Many SISOs |
  | Input-dependence | Fixed A, B, C | Selective (input-dependent) |
  | Complexity | Simpler | More complex |
  | Gating | Optional | SiLU gating |

  ## Usage

      model = S5.build(
        embed_size: 287,
        hidden_size: 256,
        state_size: 64,
        num_layers: 4
      )

  ## Use Case

  S5 is useful for ablation studies to understand what Mamba's added
  complexity (selective mechanism, gating) contributes.

  ## Reference

  - Paper: "Simplified State Space Layers for Sequence Modeling" (ICLR 2023)
  - arXiv: 2208.04933
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_state_size 64
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an S5 model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:state_size` - SSM state dimension N (default: 64)
    - `:num_layers` - Number of S5 blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack S5 blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_s5_block(
          acc,
          hidden_size: hidden_size,
          state_size: state_size,
          dropout: dropout,
          name: "s5_block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single S5 block.

  Each block has:
  1. MIMO SSM layer
  2. Feed-forward network
  """
  @spec build_s5_block(Axon.t(), keyword()) :: Axon.t()
  def build_s5_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "s5_block")

    # MIMO SSM layer
    x =
      build_mimo_ssm(input,
        hidden_size: hidden_size,
        state_size: state_size,
        dropout: dropout,
        name: "#{name}_ssm"
      )

    # Feed-forward network
    build_ffn(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  @doc """
  Build the MIMO SSM layer.

  Key components:
  1. Encoder projection
  2. State space model (diagonal A for efficiency)
  3. Decoder projection
  """
  @spec build_mimo_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_mimo_ssm(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "mimo_ssm")

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Encoder: project to state space dimension
    encoded = Axon.dense(x, state_size, name: "#{name}_encoder")

    # B matrix projection (input -> state)
    b_proj = Axon.dense(x, state_size, name: "#{name}_b_proj")

    # C matrix projection (state -> output)
    c_proj = Axon.dense(x, state_size, name: "#{name}_c_proj")

    # D matrix (skip connection coefficient)
    d_proj = Axon.dense(x, hidden_size, name: "#{name}_d_proj")

    # Apply MIMO SSM
    ssm_output =
      Axon.layer(
        &mimo_ssm_impl/5,
        [encoded, b_proj, c_proj, d_proj],
        name: "#{name}_ssm",
        hidden_size: hidden_size,
        state_size: state_size,
        op_name: :mimo_ssm
      )

    # Dropout
    ssm_output =
      if dropout > 0 do
        Axon.dropout(ssm_output, rate: dropout, name: "#{name}_dropout")
      else
        ssm_output
      end

    # Residual connection
    Axon.add(input, ssm_output, name: "#{name}_residual")
  end

  @doc """
  Build the Feed-Forward Network layer.
  """
  @spec build_ffn(Axon.t(), keyword()) :: Axon.t()
  def build_ffn(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "ffn")

    # FFN expansion factor
    inner_size = hidden_size * 4

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # GLU-style FFN
    gate_proj = Axon.dense(x, inner_size, name: "#{name}_gate")
    up_proj = Axon.dense(x, inner_size, name: "#{name}_up")

    # SiLU gate * up
    gate = Axon.activation(gate_proj, :silu, name: "#{name}_silu")
    gated = Axon.multiply(gate, up_proj, name: "#{name}_gated")

    # Down projection
    x = Axon.dense(gated, hidden_size, name: "#{name}_down")

    # Dropout
    x =
      if dropout > 0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
      else
        x
      end

    # Residual connection
    Axon.add(input, x, name: "#{name}_residual")
  end

  # MIMO SSM implementation
  # Uses diagonal A matrix for efficient computation via parallel scan
  defp mimo_ssm_impl(encoded, b_proj, c_proj, d_proj, opts) do
    hidden_size = opts[:hidden_size]
    state_size = opts[:state_size]

    # encoded: [batch, seq_len, state_size]
    # b_proj: [batch, seq_len, state_size]
    # c_proj: [batch, seq_len, state_size]
    # d_proj: [batch, seq_len, hidden_size]

    batch = Nx.axis_size(encoded, 0)
    seq_len = Nx.axis_size(encoded, 1)

    # A matrix: diagonal, negative for stability
    # Initialize as -exp(uniform) for HiPPO-like initialization
    # We use fixed values here for simplicity
    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}), 1.0))
    # Normalize
    a_diag = Nx.divide(a_diag, state_size)

    # Discretization step (fixed for simplicity)
    dt = 0.1

    # Discretize A: A_bar = exp(dt * A)
    a_bar = Nx.exp(Nx.multiply(dt, a_diag))
    a_bar = Nx.broadcast(a_bar, {batch, seq_len, state_size})

    # Discretize B: B_bar = dt * B
    b_bar = Nx.multiply(dt, b_proj)

    # Input contribution: B_bar * encoded
    bu = Nx.multiply(b_bar, encoded)

    # Parallel scan for h[t] = A_bar * h[t-1] + B_bar * u[t]
    # Use cumulative operations for parallel computation
    log_a = Nx.log(Nx.add(a_bar, 1.0e-10))
    log_a_cumsum = Nx.cumulative_sum(log_a, axis: 1)
    a_cumprod = Nx.exp(log_a_cumsum)

    # Normalize input by cumulative A product
    eps = 1.0e-10
    bu_normalized = Nx.divide(bu, Nx.add(a_cumprod, eps))
    bu_cumsum = Nx.cumulative_sum(bu_normalized, axis: 1)

    # Hidden state: h = A_cumprod * cumsum(B*u / A_cumprod)
    h = Nx.multiply(a_cumprod, bu_cumsum)

    # Output: y = C * h
    y = Nx.multiply(c_proj, h)

    # Sum over state dimension and project to hidden_size
    # [batch, seq_len]
    y_summed = Nx.sum(y, axes: [2])
    # [batch, seq_len, 1]
    y_expanded = Nx.new_axis(y_summed, 2)
    y_broadcast = Nx.broadcast(y_expanded, {batch, seq_len, hidden_size})

    # Add skip connection (D term)
    Nx.add(y_broadcast, d_proj)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an S5 model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an S5 model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer:
    # SSM layer:
    #   - Encoder: hidden * state
    #   - B, C projections: 2 * hidden * state
    #   - D projection: hidden * hidden
    ssm_params =
      hidden_size * state_size +
        2 * hidden_size * state_size +
        hidden_size * hidden_size

    # FFN (GLU style):
    #   - Gate, Up: 2 * hidden * inner
    #   - Down: inner * hidden
    ffn_params =
      2 * hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = ssm_params + ffn_params

    # Input projection
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults for real-time sequence processing (60fps).
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 64,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end

  @doc """
  Initialize hidden state for O(1) incremental inference.
  """
  @spec init_cache(keyword()) :: map()
  def init_cache(opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 1)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    # Per layer, we cache the SSM hidden state
    layers =
      for layer_idx <- 1..num_layers, into: %{} do
        layer_cache = %{
          h: Nx.broadcast(0.0, {batch_size, state_size})
        }

        {"layer_#{layer_idx}", layer_cache}
      end

    %{
      layers: layers,
      step: 0,
      config: %{
        state_size: state_size,
        num_layers: num_layers
      }
    }
  end
end
