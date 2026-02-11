defmodule Edifice.Attention.HGRN do
  @moduledoc """
  HGRN-2: Hierarchically Gated Linear RNN with State Expansion.

  HGRN-2 is a linear RNN architecture that uses hierarchical gating and
  state expansion to achieve strong performance on sequence modeling tasks
  while maintaining O(L) complexity.

  ## Key Innovation: State Expansion

  HGRN-2 expands the hidden state dimension during recurrence, then
  contracts back. This allows the model to maintain a richer internal
  representation without increasing output complexity:

  ```
  h_expanded = expand(h)  # D -> D*expansion
  h_new = gate * h_expanded + (1 - gate) * input
  output = contract(h_new)  # D*expansion -> D
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |  HGRN-2 Block                        |
  |                                      |
  |  +- State Expansion ---------------+ |
  |  |                               |   |
  |  |  h_expanded = Linear(h, D*E)  |   |
  |  |                               |   |
  |  +-------------------------------+   |
  |                                      |
  |  +- Hierarchical Gating -----------+ |
  |  |                               |   |
  |  |  forget_gate = sigmoid(Wf*x)  |   |
  |  |  input_gate = sigmoid(Wi*x)   |   |
  |  |  h = f*h + i*input            |   |
  |  |                               |   |
  |  +-------------------------------+   |
  |                                      |
  |  +- State Contraction -------------+ |
  |  |                               |   |
  |  |  output = Linear(h, D)        |   |
  |  |                               |   |
  |  +-------------------------------+   |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Aspect | Value |
  |--------|-------|
  | Training Time | O(L) |
  | Training Space | O(L) |
  | Inference Time | O(1) per step |
  | Inference Space | O(1) |

  ## Usage

      model = HGRN.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 6,
        state_expansion: 2
      )

  ## Reference

  - Paper: "HGRN2: Gated Linear RNNs with State Expansion" (arXiv:2404.07904)
  """

  require Axon

  alias Edifice.Blocks.FFN
  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 6
  @default_state_expansion 2
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an HGRN-2 model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:num_layers` - Number of HGRN blocks (default: 6)
    - `:state_expansion` - State expansion factor E (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
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

    # Stack HGRN-2 blocks with hierarchical gating:
    # Lower layers get fine-grained per-element gates; higher layers
    # share gates across groups (coarser gating).
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_hgrn_block(
          acc,
          hidden_size: hidden_size,
          state_expansion: state_expansion,
          dropout: dropout,
          layer_idx: layer_idx,
          num_layers: num_layers,
          name: "hgrn_block_#{layer_idx}"
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
  Build a single HGRN-2 block.

  Each block has:
  1. Hierarchical gated RNN layer with state expansion
  2. Feed-forward network with gating
  """
  @spec build_hgrn_block(Axon.t(), keyword()) :: Axon.t()
  def build_hgrn_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "hgrn_block")

    # Hierarchical gated RNN layer
    x =
      build_hgrn_layer(input,
        hidden_size: hidden_size,
        state_expansion: state_expansion,
        dropout: dropout,
        layer_idx: Keyword.get(opts, :layer_idx, 1),
        num_layers: Keyword.get(opts, :num_layers, @default_num_layers),
        name: "#{name}_rnn"
      )

    # Feed-forward network (norm + FFN + dropout + residual)
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = FFN.layer(ffn_normed, hidden_size: hidden_size, activation: :silu, dropout: dropout, name: "#{name}_ffn")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build the Hierarchical Gated RNN layer with state expansion.

  Key components:
  1. State expansion: D -> D*E
  2. Forget and input gates (hierarchical gating)
  3. Recurrent update with parallel scan
  4. State contraction: D*E -> D
  """
  @spec build_hgrn_layer(Axon.t(), keyword()) :: Axon.t()
  def build_hgrn_layer(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    _num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    name = Keyword.get(opts, :name, "hgrn")

    expanded_size = hidden_size * state_expansion

    # Hierarchical gating: gate granularity decreases with layer depth.
    # Layer 1 (bottom): per-element gates (gate_dim = expanded_size)
    # Higher layers: gates shared across groups (gate_dim halves per layer)
    gate_dim = max(1, Bitwise.bsr(expanded_size, layer_idx - 1))
    group_size = div(expanded_size, gate_dim)

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Forget gate: project to gate_dim (coarser at higher layers)
    forget_proj = Axon.dense(x, gate_dim, name: "#{name}_forget_proj")
    forget_gate = Axon.activation(forget_proj, :sigmoid, name: "#{name}_forget_sigmoid")

    # Input gate: same granularity as forget gate
    input_proj = Axon.dense(x, gate_dim, name: "#{name}_input_proj")
    input_gate = Axon.activation(input_proj, :sigmoid, name: "#{name}_input_sigmoid")

    # Input value (full expanded size)
    value_proj = Axon.dense(x, expanded_size, name: "#{name}_value_proj")
    value_proj = Axon.activation(value_proj, :silu, name: "#{name}_value_silu")

    # Gated recurrence with parallel scan
    output =
      Axon.layer(
        &hgrn_recurrence/4,
        [forget_gate, input_gate, value_proj],
        name: "#{name}_recurrence",
        hidden_size: hidden_size,
        expanded_size: expanded_size,
        gate_dim: gate_dim,
        group_size: group_size,
        op_name: :hgrn_recurrence
      )

    # Contract back to hidden_size
    output = Axon.dense(output, hidden_size, name: "#{name}_contract")

    # Dropout
    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # FFN delegated to Edifice.Blocks.FFN

  # HGRN recurrence with hierarchical gating via parallel scan.
  # Gates may be coarser than values (gate_dim <= expanded_size) — broadcast as needed.
  defp hgrn_recurrence(forget, input_gate, value, opts) do
    # forget: [batch, seq_len, gate_dim]
    # input_gate: [batch, seq_len, gate_dim]
    # value: [batch, seq_len, expanded_size]
    group_size = opts[:group_size]

    # Broadcast coarse gates to full expanded_size by repeating each gate element
    # across its group. If group_size == 1, this is a no-op.
    forget_full =
      if group_size > 1 do
        # [batch, seq, gate_dim] -> [batch, seq, gate_dim, 1] -> [batch, seq, gate_dim, group]
        batch = Nx.axis_size(forget, 0)
        seq_len = Nx.axis_size(forget, 1)
        gate_dim = Nx.axis_size(forget, 2)

        forget
        |> Nx.new_axis(3)
        |> Nx.broadcast({batch, seq_len, gate_dim, group_size})
        |> Nx.reshape({batch, seq_len, gate_dim * group_size})
      else
        forget
      end

    input_gate_full =
      if group_size > 1 do
        batch = Nx.axis_size(input_gate, 0)
        seq_len = Nx.axis_size(input_gate, 1)
        gate_dim = Nx.axis_size(input_gate, 2)

        input_gate
        |> Nx.new_axis(3)
        |> Nx.broadcast({batch, seq_len, gate_dim, group_size})
        |> Nx.reshape({batch, seq_len, gate_dim * group_size})
      else
        input_gate
      end

    # Gated recurrence: h[t] = forget[t] * h[t-1] + input[t] * value[t]
    # Parallel scan via log-cumsum-exp trick:
    gated_value = Nx.multiply(input_gate_full, value)

    # Stable cumulative product of forget gates
    log_forget = Nx.log(Nx.add(forget_full, 1.0e-10))
    log_forget_cumsum = Nx.cumulative_sum(log_forget, axis: 1)
    forget_cumprod = Nx.exp(log_forget_cumsum)

    # h[t] = forget_cumprod[t] * sum_{i<=t} (gated_value[i] / forget_cumprod[i])
    eps = 1.0e-10
    normalized_value = Nx.divide(gated_value, Nx.add(forget_cumprod, eps))
    value_cumsum = Nx.cumulative_sum(normalized_value, axis: 1)
    Nx.multiply(forget_cumprod, value_cumsum)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an HGRN model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an HGRN model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)

    expanded_size = hidden_size * state_expansion
    inner_size = hidden_size * 4

    # Per layer:
    # HGRN layer:
    #   - Forget, Input, Value projections: 3 * hidden * expanded
    #   - Contract projection: expanded * hidden
    hgrn_params =
      3 * hidden_size * expanded_size +
        expanded_size * hidden_size

    # FFN:
    #   - Up projection: hidden * inner
    #   - Down projection: inner * hidden
    ffn_params =
      hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = hgrn_params + ffn_params

    # Input projection
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      state_expansion: 2,
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
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    expanded_size = hidden_size * state_expansion

    # Per layer, we cache the expanded hidden state
    layers =
      for layer_idx <- 1..num_layers, into: %{} do
        layer_cache = %{
          h: Nx.broadcast(0.0, {batch_size, expanded_size})
        }

        {"layer_#{layer_idx}", layer_cache}
      end

    %{
      layers: layers,
      step: 0,
      config: %{
        hidden_size: hidden_size,
        state_expansion: state_expansion,
        num_layers: num_layers
      }
    }
  end
end
