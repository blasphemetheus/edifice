defmodule Edifice.SSM.MambaCumsum do
  @moduledoc """
  Mamba variant for experimenting with alternative scan algorithms.

  Currently uses Blelloch scan (same as regular Mamba). This module exists
  to test alternative approaches like:

  - **Hillis-Steele scan**: O(L log L) work but more parallelism per level
  - **SSD algorithm**: Mamba-2's chunked matmul approach for tensor cores
  - **Chunked scan**: Process in chunks with inter-chunk recurrence

  ## Current Status

  The cumsum-based approach (log-space reformulation) doesn't work well in XLA.
  XLA's cumulative_sum kernel is slower than Blelloch's pad/slice/multiply pattern
  for this tensor structure.

  ## Usage

      model = MambaCumsum.build(embed_size: 287, hidden_size: 256)
  """

  require Axon

  alias Edifice.SSM.Common

  @doc """
  Build a MambaCumsum model for sequence processing.

  Same API as `Mamba.build/1`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba_block/2)
  end

  defp build_mamba_block(input, opts) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "mamba_cumsum_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_selective_ssm/2)
  end

  @doc """
  Build the SSM with configurable scan algorithm.

  This is where we can swap in different scan implementations:
  - :blelloch (default) - Work-efficient O(L) work, O(log L) depth
  - :cumsum_transposed - Log-space reformulation with transposed cumsum
  - :cumsum_logspace - Log-space reformulation on original axis ordering
  """
  @spec build_selective_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_selective_ssm(input, opts \\ []) do
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    name = Keyword.get(opts, :name, "ssm")
    scan_algo = Keyword.get(opts, :scan_algo, :blelloch)

    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

    Axon.layer(
      &ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      scan_algo: scan_algo,
      op_name: :cumsum_ssm
    )
  end

  # SSM implementation - selects scan algorithm based on opts
  defp ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    scan_algo = opts[:scan_algo] || :blelloch

    # Discretize SSM parameters
    {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)

    # Apply selected scan algorithm
    h = case scan_algo do
      :cumsum_transposed -> cumsum_transposed_scan(a_bar, bx)
      :cumsum_logspace -> cumsum_logspace_scan(a_bar, bx)
      _ -> select_standard_scan(a_bar, bx)
    end

    # Compute output
    Common.compute_ssm_output(h, c)
  end

  # Select between sequential and parallel scan based on sequence length
  defp select_standard_scan(a_bar, bx) do
    seq_len = Nx.axis_size(a_bar, 1)

    if seq_len <= 32 do
      Common.sequential_scan(a_bar, bx)
    else
      Common.blelloch_scan(a_bar, bx)
    end
  end

  # ============================================================================
  # Experimental: Transposed Cumsum Scan
  # ============================================================================
  #
  # Hypothesis: XLA's cumsum may be faster when operating on trailing dimensions.
  # We transpose [batch, seq, hidden, state] → [batch, hidden, state, seq]
  # then run cumsum on axis 3, then transpose back.
  #
  # The SSM recurrence h_t = a_t * h_{t-1} + b_t can be computed via:
  #   h_t = sum_{i=0}^{t} (prod_{j=i+1}^{t} a_j) * b_i
  #
  # Using log-space: log(prod a) = cumsum(log(a))

  defp cumsum_transposed_scan(a, b) do
    # a, b: [batch, seq, hidden, state]
    # Transpose to put seq last: [batch, hidden, state, seq]
    a_t = Nx.transpose(a, axes: [0, 2, 3, 1])
    b_t = Nx.transpose(b, axes: [0, 2, 3, 1])

    # Work in log-space for numerical stability
    # log_a = log(a) which are negative since 0 < a < 1
    log_a = Nx.log(Nx.clip(a_t, 1.0e-8, 1.0))

    # Cumulative sum of log(a) gives log of cumulative product
    cum_log_a = Nx.cumulative_sum(log_a, axis: 3)

    # For each position t, we need sum_{i=0}^{t} exp(cum_log_a[t] - cum_log_a[i]) * b[i]
    # This is: exp(cum_log_a[t]) * sum_{i=0}^{t} exp(-cum_log_a[i]) * b[i]
    # The inner sum is a cumsum of exp(-cum_log_a) * b

    # exp(-cum_log_a[i]) * b[i]
    scaled_b = Nx.multiply(Nx.exp(Nx.negate(cum_log_a)), b_t)

    # Cumsum of scaled_b
    cum_scaled_b = Nx.cumulative_sum(scaled_b, axis: 3)

    # Multiply by exp(cum_log_a) to get h
    h_t = Nx.multiply(Nx.exp(cum_log_a), cum_scaled_b)

    # Transpose back to [batch, seq, hidden, state]
    Nx.transpose(h_t, axes: [0, 3, 1, 2])
  end

  # ============================================================================
  # Experimental: Log-space Cumsum (original axis ordering)
  # ============================================================================
  #
  # Same algorithm as transposed but without transposing - for comparison

  defp cumsum_logspace_scan(a, b) do
    # a, b: [batch, seq, hidden, state]
    # Work directly on axis 1

    log_a = Nx.log(Nx.clip(a, 1.0e-8, 1.0))
    cum_log_a = Nx.cumulative_sum(log_a, axis: 1)

    scaled_b = Nx.multiply(Nx.exp(Nx.negate(cum_log_a)), b)
    cum_scaled_b = Nx.cumulative_sum(scaled_b, axis: 1)

    Nx.multiply(Nx.exp(cum_log_a), cum_scaled_b)
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
