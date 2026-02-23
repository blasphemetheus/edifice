defmodule Edifice.Attention.Hawk do
  @moduledoc """
  Hawk: Pure RG-LRU Recurrent Model (RecurrentGemma).

  Hawk is the recurrence-only variant of Griffin, using Real-Gated Linear
  Recurrent Units (RG-LRU) without any local attention blocks. This makes
  it a purely recurrent model with O(1) per-step inference cost.

  ## Relationship to Griffin

  Griffin alternates RG-LRU and local attention blocks (2:1 ratio).
  Hawk removes the attention blocks entirely:

  ```
  Griffin: [RG-LRU] -> [RG-LRU] -> [LocalAttn] -> [RG-LRU] -> ...
  Hawk:    [RG-LRU] -> [RG-LRU] -> [RG-LRU]    -> [RG-LRU] -> ...
  ```

  ## RG-LRU Equations

  ```
  r_t = sigma(W_a x_t + b_a)           # Recurrence gate
  i_t = sigma(W_x x_t + b_x)           # Input gate
  a_t = a^(c * r_t)                    # Gated decay (a = sigma(Lambda), c = 8)
  h_t = a_t . h_{t-1} + sqrt(1-a_t^2) . (i_t . x_t)
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |       Hawk Block (RG-LRU)           |
  |  RMSNorm -> RG-LRU -> Residual      |
  |  RMSNorm -> Gated MLP -> Residual   |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## When to Use Hawk vs Griffin

  | Aspect | Hawk | Griffin |
  |--------|------|---------|
  | Inference | O(1) per step | O(W) per step (window) |
  | Training | O(L) | O(L*W) |
  | Long-range | Pure recurrence | Hybrid (attention helps) |
  | Simplicity | Simpler | More complex |

  ## Usage

      model = Hawk.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6
      )

  ## References

  - De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention
    for Efficient Language Models" (2024)
  - https://arxiv.org/abs/2402.19427
  """

  alias Edifice.Attention.Griffin

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  # ============================================================================
  # Default Hyperparameters (delegate to Griffin)
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  defdelegate default_hidden_size(), to: Griffin

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  defdelegate default_num_layers(), to: Griffin

  @doc "Default MLP expansion factor"
  @spec default_expand_factor() :: pos_integer()
  defdelegate default_expand_factor(), to: Griffin

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  defdelegate default_dropout(), to: Griffin

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Hawk model (Griffin without local attention).

  ## Options
    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of RG-LRU blocks (default: 6)
    - `:expand_factor` - MLP expansion factor (default: 3)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    opts
    |> Keyword.put(:use_local_attention, false)
    |> Griffin.build()
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a Hawk model."
  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Griffin

  @doc "Get recommended defaults for Hawk."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    Griffin.recommended_defaults()
    |> Keyword.delete(:local_attn_window)
    |> Keyword.delete(:num_heads)
  end
end
