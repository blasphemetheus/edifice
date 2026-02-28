defmodule Edifice.SSM.Longhorn do
  @moduledoc """
  Longhorn: State Space Model from Online Associative Recall.

  Implements the Longhorn architecture from "Longhorn: State Space Models are
  Amortized Online Learners" (Liu et al., ICLR 2025).

  ## Key Innovation

  Derives the SSM recurrence from the closed-form solution of an online
  associative recall problem, rather than discretizing a continuous ODE (S4) or
  adding ad hoc gates (Mamba). The forgetting mechanism emerges naturally from
  the key vector -- no explicit forget gate or A matrix initialization needed.

  ## Architecture

  The outer block structure is identical to Mamba (LayerNorm, input projection,
  depthwise conv + SiLU, gating, output projection). Only the inner SSM scan
  is replaced with Longhorn's recurrence:

  ```
  Projections:
    q_t = W_q * x_t          (query, R^m)
    k_t = W_k * x_t          (key, R^m)
    beta_t = sigmoid(W_b * x_t)  (learning rate, (0,1)^d)

  Epsilon (step size):
    epsilon_t = beta_t / (1 + beta_t * ||k_t||^2)

  State update (online associative recall):
    S_t = (1 - epsilon_t (x) k_t^2) * S_{t-1} + (epsilon_t * x_t) (x) k_t

  Output:
    o_t = S_t * q_t
  ```

  The state S_t is a `{d, m}` matrix per batch element where `d` is the inner
  dimension and `m` is the key/query dimension (analogous to Mamba's state_size).

  ## Usage

      model = Longhorn.build(
        embed_dim: 287,
        hidden_size: 256,
        state_size: 16,
        num_layers: 2
      )

  ## References
  - Paper: https://arxiv.org/abs/2407.14207
  - Code: https://github.com/Cranial-XIX/Longhorn
  """

  alias Edifice.SSM.Common

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:state_size, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:conv_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Longhorn model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:state_size` - Key/query dimension M (default: 16)
    - `:expand_factor` - Expansion factor for inner dim (default: 2)
    - `:conv_size` - 1D convolution kernel size (default: 4)
    - `:num_layers` - Number of Longhorn blocks (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.

  ## Examples

      iex> model = Edifice.SSM.Longhorn.build(embed_dim: 32, hidden_size: 16, state_size: 4)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_longhorn_block/2)
  end

  @doc """
  Build a single Longhorn block with online associative recall SSM.

  Uses the standard Mamba block structure (norm, in_proj, x/z split,
  depthwise conv + SiLU, gating, out_proj) with the Longhorn SSM scan
  replacing Mamba's selective scan.
  """
  @spec build_longhorn_block(Axon.t(), keyword()) :: Axon.t()
  def build_longhorn_block(input, opts \\ []) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "longhorn_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_longhorn_ssm/2)
  end

  @doc """
  Build the Longhorn SSM core.

  Projects input to (q, k, beta) and computes the online associative recall
  state update using parallel scan.

  ## Parameters
    - `input` - Axon node `[batch, seq_len, inner_size]` (after conv + silu)
    - `opts` - Options (hidden_size, state_size, name)
  """
  @spec build_longhorn_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_longhorn_ssm(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    name = Keyword.get(opts, :name, "longhorn_ssm")

    # Key and query projections (combined for efficiency)
    # [batch, seq_len, hidden_size] -> [batch, seq_len, state_size * 2]
    kq_proj = Axon.dense(input, state_size * 2, name: "#{name}_kq_proj")

    k_vec =
      Axon.nx(kq_proj, fn t -> Nx.slice_along_axis(t, 0, state_size, axis: 2) end,
        name: "#{name}_k"
      )

    q_vec =
      Axon.nx(kq_proj, fn t -> Nx.slice_along_axis(t, state_size, state_size, axis: 2) end,
        name: "#{name}_q"
      )

    # Beta (learning rate) projection + sigmoid -> (0, 1)^d
    beta =
      input
      |> Axon.dense(hidden_size, name: "#{name}_beta_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_beta_sigmoid")

    # Longhorn SSM via parallel scan
    Axon.layer(
      &longhorn_scan_impl/5,
      [input, k_vec, q_vec, beta],
      name: "#{name}_scan",
      state_size: state_size,
      op_name: :longhorn_scan
    )
  end

  # Longhorn SSM scan implementation.
  #
  # State update: S_t = (1 - epsilon_t (x) k_t^2) * S_{t-1} + (epsilon_t * x_t) (x) k_t
  # Output: o_t = S_t * q_t
  # Where: epsilon_t = beta_t / (1 + beta_t * ||k_t||^2)
  defp longhorn_scan_impl(x, k, q, beta, opts) do
    _state_size = opts[:state_size]
    seq_len = Nx.axis_size(x, 1)

    # k: [batch, seq, state_size]
    # beta: [batch, seq, hidden_size] (sigmoided, in (0,1))

    # ||k_t||^2: [batch, seq, 1]
    k_sq_sum = Nx.sum(Nx.multiply(k, k), axes: [2], keep_axes: true)

    # epsilon = beta / (1 + beta * ||k||^2): [batch, seq, hidden_size]
    epsilon = Nx.divide(beta, Nx.add(1.0, Nx.multiply(beta, k_sq_sum)))

    # Scan coefficients, both [batch, seq, hidden_size, state_size]:
    # A_t = 1 - outer(epsilon, k^2)
    k_squared = Nx.multiply(k, k)
    eps_expanded = Nx.new_axis(epsilon, 3)
    k_sq_expanded = Nx.new_axis(k_squared, 2)
    a_t = Nx.subtract(1.0, Nx.multiply(eps_expanded, k_sq_expanded))

    # B_t = outer(epsilon * x, k)
    eps_x = Nx.multiply(epsilon, x)
    eps_x_expanded = Nx.new_axis(eps_x, 3)
    k_expanded = Nx.new_axis(k, 2)
    b_t = Nx.multiply(eps_x_expanded, k_expanded)

    # Linear recurrence S_t = A_t * S_{t-1} + B_t via associative scan
    h =
      if seq_len <= 32 do
        Common.sequential_scan(a_t, b_t)
      else
        Common.blelloch_scan(a_t, b_t)
      end

    # Output: o_t = sum(S_t * q_t, axis=state_dim)
    # h: [batch, seq, hidden_size, state_size]
    # q: [batch, seq, state_size] -> [batch, seq, 1, state_size]
    q_expanded = Nx.new_axis(q, 2)
    Nx.sum(Nx.multiply(h, q_expanded), axes: [3])
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Longhorn model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @doc """
  Calculate approximate parameter count for a Longhorn model.

  Similar to Mamba but with q/k/beta projections instead of B/C/dt.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    expand_factor = Keyword.get(opts, :expand_factor, Common.default_expand_factor())
    num_layers = Keyword.get(opts, :num_layers, Common.default_num_layers())
    conv_size = Keyword.get(opts, :conv_size, Common.default_conv_size())

    inner_size = hidden_size * expand_factor

    per_layer =
      hidden_size * (2 * inner_size) +
        conv_size * inner_size +
        inner_size * (2 * state_size) +
        inner_size * inner_size +
        inner_size * hidden_size

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      num_layers: 2,
      dropout: 0.0
    ]
  end
end
