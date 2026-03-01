defmodule Edifice.CUDA.FusedScan do
  @moduledoc false
  # High-level dispatch for fused CUDA scan kernels.
  #
  # Provides mingru/2 and minlstm/3 that:
  # - On CUDA+EXLA: extract device pointers, call the fused NIF kernel,
  #   and wrap the output back into an Nx tensor (zero-copy GPU path)
  # - Otherwise: fall back to the Elixir sequential scan
  #
  # Memory management: The NIF returns a gc_ref (NIF resource) alongside
  # the output pointer. This gc_ref calls cudaFree when garbage collected.
  # We store it in the process dictionary keyed by pointer address to
  # prevent premature GC while EXLA's PjRt view is alive.

  @gc_refs_key :__edifice_cuda_gc_refs__

  @doc false
  def mingru(gates, candidates) do
    if cuda_available?(gates) do
      mingru_fused(gates, candidates)
    else
      Edifice.Recurrent.MinGRU.min_gru_scan(gates, candidates)
    end
  end

  @doc false
  def minlstm(forget_gates, input_gates, candidates) do
    if cuda_available?(forget_gates) do
      minlstm_fused(forget_gates, input_gates, candidates)
    else
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

  # ============================================================================
  # CUDA fused paths
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
end
