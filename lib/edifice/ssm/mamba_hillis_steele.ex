defmodule Edifice.SSM.MambaHillisSteele do
  @moduledoc """
  Mamba variant using Hillis-Steele parallel scan.

  ## Hillis-Steele vs Blelloch

  **Blelloch** (standard Mamba): O(L) work, O(log L) depth
  - Work-efficient: only half the elements active at each level
  - Fewer total operations

  **Hillis-Steele**: O(L log L) work, O(log L) depth
  - ALL elements active at every level
  - More parallelism per level = better GPU occupancy
  - May be faster despite more total work

  ## Algorithm

  ```
  Level 0: [1] [2] [3] [4] [5] [6] [7] [8]
  Level 1: [1] [1+2] [2+3] [3+4] [4+5] [5+6] [6+7] [7+8]  (stride 1, ALL elements)
  Level 2: [1] [1+2] [1-3] [1-4] [2-5] [3-6] [4-7] [5-8]  (stride 2, ALL elements)
  Level 3: [1] [1+2] [1-3] [1-4] [1-5] [1-6] [1-7] [1-8]  (stride 4, ALL elements)
  ```

  ## Usage

      model = MambaHillisSteele.build(embed_size: 287, hidden_size: 256)
  """

  require Axon

  alias Edifice.SSM.Common

  @doc """
  Build a MambaHillisSteele model for sequence processing.

  Same API as `Mamba.build/1`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba_block/2)
  end

  defp build_mamba_block(input, opts) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "mamba_hs_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_selective_ssm/2)
  end

  defp build_selective_ssm(input, opts) do
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    name = Keyword.get(opts, :name, "ssm")

    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

    Axon.layer(
      &ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :hillis_steele_ssm
    )
  end

  defp ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]

    # Discretize SSM parameters
    {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)

    # Use Hillis-Steele scan instead of Blelloch
    h = hillis_steele_scan(a_bar, bx)

    # Compute output
    Common.compute_ssm_output(h, c)
  end

  # ============================================================================
  # Hillis-Steele Parallel Scan
  # ============================================================================
  #
  # O(L log L) work, O(log L) depth
  # ALL elements active at every level = better GPU occupancy

  defp hillis_steele_scan(a, b) do
    seq_len = Nx.axis_size(a, 1)
    log_len = ceil(:math.log2(seq_len))

    # Hillis-Steele: at each level, combine with element `stride` positions back
    # Unlike Blelloch, ALL elements participate at every level
    {_a_final, b_final} =
      Enum.reduce(0..(log_len - 1), {a, b}, fn level, {a_curr, b_curr} ->
        stride = round(:math.pow(2, level))

        if stride >= seq_len do
          {a_curr, b_curr}
        else
          # Shift by stride: elements 0..(seq_len-stride-1) get combined with elements stride..(seq_len-1)
          # Pad with identity (1.0 for a, 0.0 for b) at the start
          a_shifted = Nx.pad(
            Nx.slice_along_axis(a_curr, 0, seq_len - stride, axis: 1),
            1.0,
            [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
          )
          b_shifted = Nx.pad(
            Nx.slice_along_axis(b_curr, 0, seq_len - stride, axis: 1),
            0.0,
            [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
          )

          # Combine ALL elements using associative operator:
          # (a1, b1) ⊗ (a2, b2) = (a1*a2, a1*b2 + b1)
          a_new = Nx.multiply(a_curr, a_shifted)
          b_new = Nx.add(Nx.multiply(a_curr, b_shifted), b_curr)

          {a_new, b_new}
        end
      end)

    b_final
  end

  # ============================================================================
  # Utilities (delegated to Common)
  # ============================================================================

  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @spec param_count(keyword()) :: non_neg_integer()
  defdelegate param_count(opts), to: Common

  @spec recommended_defaults() :: keyword()
  defdelegate recommended_defaults(), to: Common
end
