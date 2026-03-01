defmodule Edifice.CUDA.FusedScan do
  @moduledoc false
  # High-level dispatch for fused CUDA scan kernels.
  #
  # Three-tier dispatch (highest priority first):
  # 1. XLA custom call via Nx.Shared.optional — stays inside XLA graph,
  #    no graph breaks. Requires EXLA fork with fused scan kernels.
  # 2. NIF bridge — breaks graph but uses fused CUDA kernel. Requires
  #    Edifice NIF compiled with CUDA support.
  # 3. Elixir sequential scan — pure Nx, works on any backend.
  #
  # The custom call path uses Nx.Shared.optional which:
  # - In defn context (EXLA+CUDA): creates an :optional Expr node that
  #   EXLA pattern-matches into a stablehlo.custom_call
  # - In eager context or non-CUDA: runs the fallback callback
  #
  # Memory management (NIF path only): The NIF returns a gc_ref (NIF
  # resource) alongside the output pointer. This gc_ref calls cudaFree
  # when garbage collected.

  @gc_refs_key :__edifice_cuda_gc_refs__

  @doc false
  def mingru(gates, candidates) do
    cond do
      custom_call_available?() ->
        mingru_custom_call(gates, candidates)

      cuda_available?(gates) ->
        mingru_fused(gates, candidates)

      true ->
        Edifice.Recurrent.MinGRU.min_gru_scan(gates, candidates)
    end
  end

  @doc false
  def minlstm(forget_gates, input_gates, candidates) do
    cond do
      custom_call_available?() ->
        minlstm_custom_call(forget_gates, input_gates, candidates)

      cuda_available?(forget_gates) ->
        minlstm_fused(forget_gates, input_gates, candidates)

      true ->
        Edifice.Recurrent.MinLSTM.min_lstm_scan(forget_gates, input_gates, candidates)
    end
  end

  @doc false
  def elu_gru(gates, candidates) do
    if cuda_available?(gates) do
      elu_gru_fused(gates, candidates)
    else
      Edifice.Recurrent.NativeRecurrence.elu_gru_scan(gates, candidates)
    end
  end

  @doc false
  def real_gru(gates, candidates) do
    if cuda_available?(gates) do
      real_gru_fused(gates, candidates)
    else
      Edifice.Recurrent.NativeRecurrence.real_gru_scan(gates, candidates)
    end
  end

  @doc false
  def diag_linear(a_vals, b_vals) do
    if cuda_available?(a_vals) do
      diag_linear_fused(a_vals, b_vals)
    else
      Edifice.Recurrent.NativeRecurrence.diag_linear_scan(a_vals, b_vals)
    end
  end

  @doc false
  def liquid(tau, activation) do
    if cuda_available?(tau) do
      liquid_fused(tau, activation)
    else
      Edifice.Liquid.liquid_exact_scan(tau, activation)
    end
  end

  @doc """
  Generic linear recurrence scan: h = a * h + b.

  Both `a` and `b` are pre-computed [batch, seq_len, hidden] tensors.
  No nonlinearities are applied — all activations must be computed
  before calling this function.

  Covers: Griffin RG-LRU, MEGA EMA, SSTransformer EMA, HybridBuilder EMA,
  GSS SSM, MambaVision SSM.
  """
  def linear_scan(a_vals, b_vals) do
    if cuda_available?(a_vals) do
      linear_scan_fused(a_vals, b_vals)
    else
      linear_scan_fallback(a_vals, b_vals)
    end
  end

  @doc """
  DeltaNet delta rule scan: matrix-state recurrence.

  All inputs are pre-computed [batch, seq_len, num_heads, head_dim] tensors.
  Keys should be L2-normalized, beta should be post-sigmoid.

  Returns [batch, seq_len, num_heads, head_dim] retrieval outputs.
  """
  def delta_net_scan(q, k, v, beta) do
    if cuda_available?(q) do
      delta_net_fused(q, k, v, beta)
    else
      Edifice.Recurrent.DeltaNet.delta_net_sequential_scan(q, k, v, beta)
    end
  end

  @doc """
  GatedDeltaNet delta rule scan: matrix-state recurrence with scalar decay.

  Inputs q, k, v, beta are [batch, seq_len, num_heads, head_dim].
  Alpha is [batch, seq_len, num_heads] — scalar forget gate per head.

  Returns [batch, seq_len, num_heads, head_dim] retrieval outputs.
  """
  def gated_delta_net_scan(q, k, v, beta, alpha) do
    if cuda_available?(q) do
      gated_delta_net_fused(q, k, v, beta, alpha)
    else
      Edifice.Recurrent.GatedDeltaNet.gated_delta_net_sequential_scan(q, k, v, beta, alpha)
    end
  end

  # ============================================================================
  # XLA custom call paths (graph-preserving)
  # ============================================================================

  defp mingru_custom_call(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)

    # Pre-compute sigmoid — XLA fuses this into the graph
    z = Nx.sigmoid(gates)

    # Zero initial hidden state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

    # Output template for Nx.Shared.optional
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_mingru_scan, [z, candidates, h0], output, fn z, cand, h0 ->
      mingru_scan_fallback(z, cand, h0)
    end)
  end

  defp minlstm_custom_call(forget_gates, input_gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(forget_gates)

    # Pre-compute sigmoid — XLA fuses this into the graph
    f = Nx.sigmoid(forget_gates)
    i = Nx.sigmoid(input_gates)

    # Zero initial cell state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

    # Output template for Nx.Shared.optional
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_minlstm_scan, [f, i, candidates, h0], output, fn f, i, cand, h0 ->
      minlstm_scan_fallback(f, i, cand, h0)
    end)
  end

  # Fallback scans for custom call path — take pre-processed inputs
  # (post-sigmoid gates + h0) since custom call kernels expect that.
  defp mingru_scan_fallback(z, candidates, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(z)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  defp minlstm_scan_fallback(f_gate, i_gate, candidates, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(f_gate)
    norm_eps = 1.0e-6

    # Normalize: f' = f/(f+i+eps), i' = i/(f+i+eps)
    gate_sum = Nx.add(f_gate, Nx.add(i_gate, norm_eps))
    f_norm = Nx.divide(f_gate, gate_sum)
    i_norm = Nx.divide(i_gate, gate_sum)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
        f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        cand_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
        {c_t, [c_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # CUDA NIF fused paths (graph-breaking)
  # ============================================================================

  defp mingru_fused(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)

    # Apply sigmoid on the XLA side (XLA fuses this efficiently)
    z = Nx.sigmoid(gates)

    # Zero initial hidden state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(gates)), {batch, hidden})

    # Extract device pointers
    z_ptr = Nx.to_pointer(z, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_mingru_scan(
           z_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        # Hold gc_ref to prevent cudaFree until we're done with the tensor
        hold_gc_ref(out_addr, gc_ref)

        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused MinGRU scan failed: #{reason}"
    end
  end

  defp minlstm_fused(forget_gates, input_gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(forget_gates)

    # Apply sigmoid on XLA side — kernel expects post-sigmoid values
    f = Nx.sigmoid(forget_gates)
    i = Nx.sigmoid(input_gates)

    # Zero initial cell state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(forget_gates)), {batch, hidden})

    # Extract device pointers
    f_ptr = Nx.to_pointer(f, mode: :local)
    i_ptr = Nx.to_pointer(i, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_minlstm_scan(
           f_ptr.address, i_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)

        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(forget_gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused MinLSTM scan failed: #{reason}"
    end
  end

  # ============================================================================
  # NativeRecurrence fused paths (pre-activation inputs — kernel applies
  # sigmoid/elu internally, unlike MinGRU which gets post-sigmoid)
  # ============================================================================

  defp elu_gru_fused(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(gates)), {batch, hidden})

    g_ptr = Nx.to_pointer(gates, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_elu_gru_scan(
           g_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused ELU-GRU scan failed: #{reason}"
    end
  end

  defp real_gru_fused(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(gates)), {batch, hidden})

    g_ptr = Nx.to_pointer(gates, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_real_gru_scan(
           g_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused Real-GRU scan failed: #{reason}"
    end
  end

  defp diag_linear_fused(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(a_vals)), {batch, hidden})

    a_ptr = Nx.to_pointer(a_vals, mode: :local)
    b_ptr = Nx.to_pointer(b_vals, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_diag_linear_scan(
           a_ptr.address, b_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(a_vals), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused Diag-Linear scan failed: #{reason}"
    end
  end

  defp linear_scan_fused(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(a_vals)), {batch, hidden})

    a_ptr = Nx.to_pointer(a_vals, mode: :local)
    b_ptr = Nx.to_pointer(b_vals, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_linear_scan(
           a_ptr.address, b_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(a_vals), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused linear scan failed: #{reason}"
    end
  end

  defp liquid_fused(tau, activation) do
    {batch, seq_len, hidden} = Nx.shape(tau)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(tau)), {batch, hidden})

    tau_ptr = Nx.to_pointer(tau, mode: :local)
    act_ptr = Nx.to_pointer(activation, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_liquid_scan(
           tau_ptr.address, act_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(tau), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused Liquid scan failed: #{reason}"
    end
  end

  # ============================================================================
  # Delta rule fused paths (matrix-state recurrences)
  # ============================================================================

  defp delta_net_fused(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)

    case Edifice.CUDA.NIF.fused_delta_net_scan(
           q_ptr.address, k_ptr.address, v_ptr.address, beta_ptr.address,
           batch, seq_len, num_heads, head_dim
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * 4

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused DeltaNet scan failed: #{reason}"
    end
  end

  defp gated_delta_net_fused(q, k, v, beta, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)

    case Edifice.CUDA.NIF.fused_gated_delta_net_scan(
           q_ptr.address, k_ptr.address, v_ptr.address,
           beta_ptr.address, alpha_ptr.address,
           batch, seq_len, num_heads, head_dim
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * 4

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused GatedDeltaNet scan failed: #{reason}"
    end
  end

  # ============================================================================
  # Fallback: generic linear recurrence scan (CPU/BinaryBackend)
  # ============================================================================

  @doc false
  def linear_scan_fallback(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)

    h_init = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(a_vals)), {batch, hidden})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        b_t = Nx.slice_along_axis(b_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # GC reference tracking
  # ============================================================================

  # Store a gc_ref in the process dictionary to prevent premature cudaFree.
  # The NIF resource destructor calls cudaFree when the gc_ref is GC'd.
  # We keep only the most recent ref per pointer address (a new allocation
  # at the same address means the old one was already freed).
  defp hold_gc_ref(ptr_addr, gc_ref) do
    refs = Process.get(@gc_refs_key, %{})
    Process.put(@gc_refs_key, Map.put(refs, ptr_addr, gc_ref))
  end

  # ============================================================================
  # Backend detection
  # ============================================================================

  defp cuda_available?(tensor) do
    exla_cuda_backend?(tensor) and nif_loaded?()
  end

  defp exla_cuda_backend?(tensor) do
    exla_backend = Module.concat([EXLA, Backend])
    tensor.data.__struct__ == exla_backend
  rescue
    _ -> false
  end

  defp backend_for(tensor) do
    tensor.data.__struct__
  end

  defp nif_loaded? do
    Code.ensure_loaded?(Edifice.CUDA.NIF) and
      function_exported?(Edifice.CUDA.NIF, :fused_mingru_scan, 6)
  end

  defp custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_mingru_scan, 4)
  rescue
    _ -> false
  end
end
