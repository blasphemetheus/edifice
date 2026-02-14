defmodule Edifice.SSM.MambaSSD do
  @moduledoc """
  Mamba variant using State Space Duality (SSD) algorithm from Mamba-2.

  ## SSD Algorithm

  The key insight: SSM computation can be decomposed into matrix multiplications
  that leverage tensor cores (10-20x faster than scalar operations).

  ### Algorithm Steps

  1. **Split into chunks**: Divide sequence into chunks of size C
  2. **Intra-chunk (matmul)**: Compute outputs within each chunk using dense matmul
     - This uses tensor cores!
     - O(C²) work per chunk, but highly parallel
  3. **Inter-chunk (scan)**: Small sequential scan over chunk boundaries
     - Only L/C elements to scan
  4. **Combine**: Merge chunk outputs with inter-chunk states

  ### Complexity

  - Intra-chunk: O(L/C × C²) = O(L × C) work, but tensor core accelerated
  - Inter-chunk: O(L/C) sequential work (tiny)
  - Total: Much faster in practice due to tensor cores

  ## Training Mode

  When `training_mode: true` is set, the SSD algorithm uses matrix multiplication
  formulation optimized for tensor cores:

      y = (L ⊙ (C @ B^T)) @ x + cumsum(A) @ h_prev

  Where L is a lower-triangular mask. This formulation:
  - Uses dense matmuls for tensor core utilization
  - Computes all positions in parallel within each chunk
  - Is significantly faster for batched training

  For inference, use `training_mode: false` (default) which uses efficient scans
  with O(1) memory per step.

  ## Current Performance

  **Note:** The XLA implementation has limitations compared to fused CUDA kernels.
  For production training, consider using a custom Triton kernel.

  ## Usage

      # Training (matmul formulation)
      model = MambaSSD.build(embed_size: 287, hidden_size: 256, training_mode: true)

      # Inference (scan formulation)
      model = MambaSSD.build(embed_size: 287, hidden_size: 256, training_mode: false)
  """

  require Axon

  alias Edifice.SSM.Common

  @default_chunk_size 16

  @doc """
  Build an SSD Mamba model.

  ## Options

    - `:training_mode` - If true, uses matmul formulation for tensor cores (default: false)
    - `:chunk_size` - Size of chunks for SSD algorithm (default: 16)
    - All common Mamba options (see `Edifice.SSM.Common`)
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba_block/2)
  end

  defp build_mamba_block(input, opts) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "mamba_ssd_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_selective_ssm/2)
  end

  defp build_selective_ssm(input, opts) do
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    training_mode = Keyword.get(opts, :training_mode, false)
    name = Keyword.get(opts, :name, "ssm")

    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

    Axon.layer(
      &ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      chunk_size: chunk_size,
      training_mode: training_mode,
      op_name: :ssd_ssm
    )
  end

  defp ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    chunk_size = opts[:chunk_size] || @default_chunk_size
    training_mode = opts[:training_mode] || false

    # Discretize SSM parameters
    {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)

    # Use SSD algorithm (training or inference mode)
    h =
      if training_mode do
        ssd_matmul_scan(a_bar, bx, c, chunk_size)
      else
        ssd_scan(a_bar, bx, chunk_size)
      end

    # Compute output
    Common.compute_ssm_output(h, c)
  end

  # ============================================================================
  # SSD Training Mode: Matrix Multiplication Formulation
  # ============================================================================
  #
  # The key insight from Mamba-2: SSM can be expressed as structured matrix-vector
  # multiplication within each chunk, enabling tensor core utilization.
  #
  # For a chunk of length C:
  #   y_t = sum_{s=0}^{t} (prod_{r=s+1}^{t} A_r) * B_s * x_s
  #
  # This can be written as: y = L @ x where L is lower-triangular with
  # L[t,s] = prod_{r=s+1}^{t} A_r * B_s for t >= s

  defp ssd_matmul_scan(a, bx, _c, chunk_size) do
    # a: [batch, seq_len, hidden_size, state_size]
    # bx: [batch, seq_len, hidden_size, state_size]
    seq_len = Nx.axis_size(a, 1)

    if seq_len <= chunk_size do
      # Single chunk: use matmul formulation directly
      ssd_matmul_chunk(a, bx)
    else
      # Multiple chunks with inter-chunk propagation
      chunked_ssd_matmul(a, bx, chunk_size, seq_len)
    end
  end

  # Compute SSM output for a single chunk using matmul formulation
  # This is where tensor cores can be utilized in a fused implementation
  defp ssd_matmul_chunk(a_chunk, bx_chunk) do
    # a_chunk: [batch, chunk_len, hidden, state]
    # bx_chunk: [batch, chunk_len, hidden, state]
    batch = Nx.axis_size(a_chunk, 0)
    chunk_len = Nx.axis_size(a_chunk, 1)
    hidden = Nx.axis_size(a_chunk, 2)
    state = Nx.axis_size(a_chunk, 3)

    # Build the lower-triangular transfer matrix L where
    # L[t, s] = prod_{r=s+1}^{t} a[r] for t >= s, else 0
    #
    # For each (t, s) pair, we need cumulative product of A from s+1 to t
    # We can compute this efficiently using log-space operations

    # Compute log(a) for numerical stability
    # Clamp to avoid log(0)
    eps = 1.0e-10
    log_a = Nx.log(Nx.add(Nx.abs(a_chunk), eps))

    # Compute cumulative sum of log_a along sequence axis
    # cumsum[t] = sum_{r=0}^{t} log(a[r])
    log_cumsum = Nx.cumulative_sum(log_a, axis: 1)

    # For L[t, s] = prod_{r=s+1}^{t} a[r] = exp(cumsum[t] - cumsum[s])
    # We need to compute this for all (t, s) pairs where t >= s

    # Reshape for broadcasting: [batch, chunk_len, 1, hidden, state]
    log_cumsum_t = Nx.reshape(log_cumsum, {batch, chunk_len, 1, hidden, state})

    # log_transfer[t, s] = log_cumsum[t] - log_cumsum[s]
    # This gives us log of prod_{r=s+1}^{t} a[r]
    # But we need to shift by 1: we want prod from s+1, not s
    # So we use: log_cumsum[t] - log_cumsum[s] when s < t

    # Create shift: prepend zero and drop last
    zero_slice = Nx.broadcast(0.0, {batch, 1, hidden, state})

    log_cumsum_shifted =
      Nx.concatenate([zero_slice, Nx.slice_along_axis(log_cumsum, 0, chunk_len - 1, axis: 1)],
        axis: 1
      )

    log_cumsum_s_shifted = Nx.reshape(log_cumsum_shifted, {batch, 1, chunk_len, hidden, state})

    # log_transfer[t, s] = log_cumsum[t] - log_cumsum_shifted[s]
    log_transfer = Nx.subtract(log_cumsum_t, log_cumsum_s_shifted)

    # Convert back from log space
    transfer = Nx.exp(log_transfer)

    # Apply lower-triangular mask: L[t, s] = 0 for t < s
    # Create mask: [chunk_len, chunk_len] where mask[t, s] = 1 if t >= s
    t_indices = Nx.iota({chunk_len, 1})
    s_indices = Nx.iota({1, chunk_len})
    mask = Nx.greater_equal(t_indices, s_indices)
    # Broadcast mask to full shape: [batch, chunk_len, chunk_len, hidden, state]
    mask = Nx.broadcast(mask, {batch, chunk_len, chunk_len, hidden, state})

    # Apply mask
    transfer = Nx.select(mask, transfer, Nx.broadcast(0.0, Nx.shape(transfer)))

    # Now compute output: for each position t, sum over s of L[t,s] * bx[s]
    # bx_chunk: [batch, chunk_len, hidden, state] -> [batch, 1, chunk_len, hidden, state]
    bx_expanded = Nx.reshape(bx_chunk, {batch, 1, chunk_len, hidden, state})

    # output[t] = sum_s L[t, s] * bx[s]
    # [batch, chunk_len, chunk_len, hidden, state] * [batch, 1, chunk_len, hidden, state]
    weighted = Nx.multiply(transfer, bx_expanded)

    # Sum over source positions (axis 2)
    Nx.sum(weighted, axes: [2])
  end

  # Multi-chunk SSD with matmul formulation
  defp chunked_ssd_matmul(a, bx, chunk_size, seq_len) do
    batch = Nx.axis_size(a, 0)
    hidden = Nx.axis_size(a, 2)
    state = Nx.axis_size(a, 3)

    num_chunks = div(seq_len, chunk_size)
    remainder = rem(seq_len, chunk_size)

    # Process full chunks using matmul formulation
    chunk_outputs =
      Enum.map(0..(num_chunks - 1), fn chunk_idx ->
        start_idx = chunk_idx * chunk_size

        a_chunk = Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
        bx_chunk = Nx.slice_along_axis(bx, start_idx, chunk_size, axis: 1)

        ssd_matmul_chunk(a_chunk, bx_chunk)
      end)

    # Handle remainder
    chunk_outputs =
      if remainder > 0 do
        start_idx = num_chunks * chunk_size
        a_rem = Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
        bx_rem = Nx.slice_along_axis(bx, start_idx, remainder, axis: 1)
        chunk_outputs ++ [ssd_matmul_chunk(a_rem, bx_rem)]
      else
        chunk_outputs
      end

    # Get final states for inter-chunk propagation
    chunk_final_states =
      Enum.map(chunk_outputs, fn chunk_h ->
        chunk_len = Nx.axis_size(chunk_h, 1)
        Nx.slice_along_axis(chunk_h, chunk_len - 1, 1, axis: 1)
      end)

    # Inter-chunk state propagation (same as inference mode)
    {_, propagated_outputs} =
      Enum.reduce(
        Enum.with_index(chunk_outputs),
        {Nx.broadcast(0.0, {batch, 1, hidden, state}), []},
        fn {chunk_h, idx}, {_running_state, acc} ->
          chunk_len = Nx.axis_size(chunk_h, 1)

          if idx == 0 do
            new_running = Enum.at(chunk_final_states, idx)
            {new_running, acc ++ [chunk_h]}
          else
            a_chunk =
              if idx == length(chunk_outputs) - 1 and remainder > 0 do
                start_idx = num_chunks * chunk_size
                Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
              else
                start_idx = idx * chunk_size
                Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
              end

            a_cumprods = compute_cumulative_products(a_chunk)

            # Get accumulated state from all previous chunks
            prev_states = Enum.take(chunk_final_states, idx)
            prev_a_prods = compute_inter_chunk_products(a, chunk_size, idx, num_chunks, remainder)

            # Compute total state contribution from all previous chunks
            state_contribution =
              prev_states
              |> Enum.with_index()
              |> Enum.map(fn {state, state_idx} ->
                # Product of A from state's chunk end to current chunk start
                a_prod_factor = Enum.at(prev_a_prods, state_idx)
                # Then multiply by cumulative products within current chunk
                Nx.multiply(a_cumprods, Nx.multiply(a_prod_factor, state))
              end)
              |> Enum.reduce(fn x, acc -> Nx.add(x, acc) end)

            adjusted_chunk = Nx.add(chunk_h, state_contribution)
            chunk_final = Nx.slice_along_axis(adjusted_chunk, chunk_len - 1, 1, axis: 1)

            {chunk_final, acc ++ [adjusted_chunk]}
          end
        end
      )

    Nx.concatenate(propagated_outputs, axis: 1)
  end

  # Compute products of A for inter-chunk propagation
  defp compute_inter_chunk_products(a, chunk_size, current_chunk_idx, num_chunks, remainder) do
    Enum.map(0..(current_chunk_idx - 1), fn prev_idx ->
      # Compute product of A from end of prev_idx chunk to start of current chunk
      # This spans all chunks from prev_idx+1 to current_chunk_idx-1, plus partial of prev_idx

      products =
        Enum.map((prev_idx + 1)..(current_chunk_idx - 1)//1, fn between_idx ->
          if between_idx < num_chunks do
            start_idx = between_idx * chunk_size
            a_chunk = Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
            Nx.product(a_chunk, axes: [1])
          else
            # Remainder chunk
            start_idx = num_chunks * chunk_size
            a_chunk = Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
            Nx.product(a_chunk, axes: [1])
          end
        end)

      if products == [] do
        # Adjacent chunks, just need product from prev chunk's end
        batch = Nx.axis_size(a, 0)
        hidden = Nx.axis_size(a, 2)
        state = Nx.axis_size(a, 3)
        Nx.broadcast(1.0, {batch, 1, hidden, state})
      else
        Enum.reduce(products, fn x, acc -> Nx.multiply(x, acc) end)
        |> Nx.new_axis(1)
      end
    end)
  end

  # ============================================================================
  # SSD (State Space Duality) Algorithm
  # ============================================================================
  #
  # From Mamba-2: converts SSM scan into chunked matmul operations

  defp ssd_scan(a, b, chunk_size) do
    # a, b: [batch, seq_len, hidden_size, state_size]
    seq_len = Nx.axis_size(a, 1)

    # If sequence fits in one chunk, use simple scan
    if seq_len <= chunk_size do
      Common.sequential_scan(a, b)
    else
      chunked_ssd_scan(a, b, chunk_size, seq_len)
    end
  end

  defp chunked_ssd_scan(a, b, chunk_size, seq_len) do
    batch = Nx.axis_size(a, 0)
    hidden = Nx.axis_size(a, 2)
    state = Nx.axis_size(a, 3)

    # Number of full chunks
    num_chunks = div(seq_len, chunk_size)
    remainder = rem(seq_len, chunk_size)

    # Process full chunks
    chunk_outputs =
      Enum.map(0..(num_chunks - 1), fn chunk_idx ->
        start_idx = chunk_idx * chunk_size

        # Extract chunk
        a_chunk = Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
        b_chunk = Nx.slice_along_axis(b, start_idx, chunk_size, axis: 1)

        # Intra-chunk computation using matmul-style approach
        # For each position t in chunk, output depends on positions 0..t
        # This is a lower-triangular matmul pattern
        intra_chunk_scan(a_chunk, b_chunk)
      end)

    # Handle remainder if any
    chunk_outputs =
      if remainder > 0 do
        start_idx = num_chunks * chunk_size
        a_rem = Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
        b_rem = Nx.slice_along_axis(b, start_idx, remainder, axis: 1)
        chunk_outputs ++ [intra_chunk_scan(a_rem, b_rem)]
      else
        chunk_outputs
      end

    # Compute final state of each chunk for inter-chunk propagation
    chunk_final_states =
      Enum.map(chunk_outputs, fn chunk_h ->
        # Final state is last element of chunk
        chunk_len = Nx.axis_size(chunk_h, 1)
        Nx.slice_along_axis(chunk_h, chunk_len - 1, 1, axis: 1)
      end)

    # Precompute per-chunk A slices and their cumulative products to avoid
    # redundant slice + cumprod work inside the inter-chunk reduce loop
    chunk_a_cumprods =
      Enum.with_index(chunk_outputs)
      |> Enum.map(fn {_chunk_h, idx} ->
        a_chunk =
          if idx == length(chunk_outputs) - 1 and remainder > 0 do
            start_idx = num_chunks * chunk_size
            Nx.slice_along_axis(a, start_idx, remainder, axis: 1)
          else
            start_idx = idx * chunk_size
            Nx.slice_along_axis(a, start_idx, chunk_size, axis: 1)
          end

        compute_cumulative_products(a_chunk)
      end)

    # Inter-chunk state propagation
    # h_chunk[i] = h_intra[i] + A_prod[i] * h_final[i-1] + ...
    # This is a small scan over chunk boundaries

    {_, propagated_outputs} =
      Enum.reduce(
        Enum.zip([chunk_outputs, chunk_a_cumprods])
        |> Enum.with_index(),
        {Nx.broadcast(0.0, {batch, 1, hidden, state}), []},
        fn {{chunk_h, a_cumprods}, idx}, {running_state, acc} ->
          if idx == 0 do
            # First chunk: no inter-chunk contribution
            new_running = Enum.at(chunk_final_states, idx)
            {new_running, acc ++ [chunk_h]}
          else
            # Propagate running state through chunk using precomputed cumprods
            state_contribution = Nx.multiply(a_cumprods, running_state)

            # Add to intra-chunk output
            adjusted_chunk = Nx.add(chunk_h, state_contribution)

            # Update running state: final state of this chunk
            chunk_len = Nx.axis_size(chunk_h, 1)
            chunk_final = Nx.slice_along_axis(adjusted_chunk, chunk_len - 1, 1, axis: 1)

            {chunk_final, acc ++ [adjusted_chunk]}
          end
        end
      )

    # Concatenate all chunk outputs
    Nx.concatenate(propagated_outputs, axis: 1)
  end

  # Intra-chunk scan using efficient pattern
  # For small chunks, we can compute all pairwise dependencies efficiently
  defp intra_chunk_scan(a_chunk, b_chunk) do
    chunk_len = Nx.axis_size(a_chunk, 1)

    if chunk_len <= 4 do
      # Very small: just do sequential
      Common.sequential_scan(a_chunk, b_chunk)
    else
      # Use Blelloch for intra-chunk (it's efficient for small sequences)
      Common.blelloch_scan(a_chunk, b_chunk)
    end
  end

  # Compute cumulative product along sequence dimension
  # Returns tensor where position t contains prod(a[0..t])
  defp compute_cumulative_products(a) do
    seq_len = Nx.axis_size(a, 1)

    {_, cumprods} =
      Enum.reduce(0..(seq_len - 1), {nil, []}, fn t, {prev_prod, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1)

        new_prod =
          if prev_prod == nil do
            a_t
          else
            Nx.multiply(prev_prod, a_t)
          end

        {new_prod, acc ++ [new_prod]}
      end)

    Nx.concatenate(cumprods, axis: 1)
  end

  # ============================================================================
  # Utilities (delegated to Common)
  # ============================================================================

  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @spec param_count(keyword()) :: non_neg_integer()
  defdelegate param_count(opts), to: Common

  @doc """
  Get recommended defaults for real-time sequence processing (60fps).

  Includes SSD-specific `chunk_size` and `training_mode` options.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    Common.recommended_defaults()
    |> Keyword.put(:chunk_size, 16)
    |> Keyword.put(:training_mode, false)
  end

  @doc """
  Get training-optimized defaults.

  Uses matmul formulation for better tensor core utilization.
  """
  @spec training_defaults() :: keyword()
  def training_defaults do
    recommended_defaults()
    |> Keyword.put(:training_mode, true)
    # Larger chunks for better matmul efficiency
    |> Keyword.put(:chunk_size, 32)
  end
end
