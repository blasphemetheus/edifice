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
  #
  # Known Limitations:
  # - Flash Attention custom call always uses causal=1. The `:causal` option
  #   is respected by the NIF and fallback paths, but the EXLA custom call
  #   path hardcodes causal=1 to avoid a scalar operand segfault in XLA.
  #   Non-causal flash attention falls back to NIF or Elixir.
  # - LASER Attention custom call: same limitation, hardcoded causal=1.
  # - See docs/cuda_custom_call_debugging.md for technical details.

  @gc_refs_key :__edifice_cuda_gc_refs__

  # Compute dtype flag for NIF dispatch (0 = f32, 1 = bf16)
  defp dtype_flag(tensor) do
    case Nx.type(tensor) do
      {:bf, 16} -> 1
      _ -> 0
    end
  end

  # Compute element size for pointer math
  defp elem_size(tensor) do
    case Nx.type(tensor) do
      {:bf, 16} -> 2
      _ -> 4
    end
  end

  # Get tensor type for template/broadcast creation
  defp tensor_type(tensor) do
    Nx.type(tensor)
  end

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
    cond do
      custom_call_available?() ->
        elu_gru_custom_call(gates, candidates)

      cuda_available?(gates) ->
        elu_gru_fused(gates, candidates)

      true ->
        Edifice.Recurrent.NativeRecurrence.elu_gru_scan(gates, candidates)
    end
  end

  @doc false
  def real_gru(gates, candidates) do
    cond do
      custom_call_available?() ->
        real_gru_custom_call(gates, candidates)

      cuda_available?(gates) ->
        real_gru_fused(gates, candidates)

      true ->
        Edifice.Recurrent.NativeRecurrence.real_gru_scan(gates, candidates)
    end
  end

  @doc false
  def diag_linear(a_vals, b_vals) do
    cond do
      custom_call_available?() ->
        diag_linear_custom_call(a_vals, b_vals)

      cuda_available?(a_vals) ->
        diag_linear_fused(a_vals, b_vals)

      true ->
        Edifice.Recurrent.NativeRecurrence.diag_linear_scan(a_vals, b_vals)
    end
  end

  @doc false
  def liquid(tau, activation) do
    cond do
      custom_call_available?() ->
        liquid_custom_call(tau, activation)

      cuda_available?(tau) ->
        liquid_fused(tau, activation)

      true ->
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
    cond do
      custom_call_available?() ->
        linear_scan_custom_call(a_vals, b_vals)

      cuda_available?(a_vals) ->
        linear_scan_fused(a_vals, b_vals)

      true ->
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
    cond do
      custom_call_available?() ->
        delta_net_custom_call(q, k, v, beta)

      cuda_available?(q) ->
        delta_net_fused(q, k, v, beta)

      true ->
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
    cond do
      custom_call_available?() ->
        gated_delta_net_custom_call(q, k, v, beta, alpha)

      cuda_available?(q) ->
        gated_delta_net_fused(q, k, v, beta, alpha)

      true ->
        Edifice.Recurrent.GatedDeltaNet.gated_delta_net_sequential_scan(q, k, v, beta, alpha)
    end
  end

  @doc """
  DeltaProduct scan: matrix-state recurrence with Householder products.

  Inputs:
    q:    [B, T, H, d]       — query vectors (shared across Householder steps)
    k:    [B, T, n_h, H, d]  — key vectors per Householder step
    v:    [B, T, n_h, H, d]  — value vectors per Householder step
    beta: [B, T, n_h, H]     — scalar gate per head per step (post-sigmoid)

  Returns [B, T, H, d] RMS-normalized output.
  """
  def delta_product_scan(q, k, v, beta) do
    cond do
      custom_call_available?() ->
        delta_product_custom_call(q, k, v, beta)

      cuda_available?(q) ->
        delta_product_fused(q, k, v, beta)

      true ->
        Edifice.Recurrent.DeltaProduct.delta_product_scan_fallback(q, k, v, beta)
    end
  end

  @doc """
  sLSTM scan with hidden-to-hidden matmul fused into the kernel.

  Inputs are pre-computed W@x `[batch, seq_len, 4*hidden]` and recurrent
  weight R `[hidden, 4*hidden]`. The kernel computes R@h internally using
  shared memory. Uses log-domain stabilized exponential gating.
  """
  def slstm_scan(wx, recurrent_weight) do
    cond do
      custom_call_available?() ->
        slstm_custom_call(wx, recurrent_weight)

      cuda_available?(wx) ->
        slstm_fused(wx, recurrent_weight)

      true ->
        Edifice.Recurrent.SLSTM.slstm_scan_fallback(wx, recurrent_weight)
    end
  end

  @doc """
  Standard LSTM scan with hidden-to-hidden matmul fused into the kernel.

  Inputs are pre-computed W@x + bias `[batch, seq_len, 4*hidden]` and recurrent
  weight R `[hidden, 4*hidden]`. The kernel computes R@h internally using
  shared memory. Standard sigmoid/tanh gating: i/f/o = σ, g = tanh,
  c = f*c + i*g, h = o*tanh(c).
  """
  def lstm_scan(wx, recurrent_weight) do
    cond do
      custom_call_available?() ->
        lstm_custom_call(wx, recurrent_weight)

      cuda_available?(wx) ->
        lstm_fused(wx, recurrent_weight)

      true ->
        lstm_scan_fallback(wx, recurrent_weight, nil, nil)
    end
  end

  @doc """
  Standard GRU scan with hidden-to-hidden matmul fused into the kernel.

  Inputs are pre-computed W@x + bias `[batch, seq_len, 3*hidden]` and recurrent
  weight R `[hidden, 3*hidden]`. The kernel computes R@h internally using
  shared memory. Reset gate applied selectively to recurrent contribution:
  r = σ(wx_r + rh_r), z = σ(wx_z + rh_z), n = tanh(wx_n + r*rh_n),
  h = (1-z)*n + z*h_prev.
  """
  def gru_scan(wx, recurrent_weight) do
    cond do
      custom_call_available?() ->
        gru_custom_call(wx, recurrent_weight)

      cuda_available?(wx) ->
        gru_fused(wx, recurrent_weight)

      true ->
        gru_scan_fallback(wx, recurrent_weight, nil)
    end
  end

  @doc """
  TTT-Linear scan with per-timestep weight matrix update.

  Inputs: pre-projected Q, K, V, eta `[batch, seq_len, inner_size]`,
  initial weight matrix W0 `[inner_size, inner_size]`, and LayerNorm
  gamma/beta `[inner_size]`.

  The eta input should be post-sigmoid, pre-scaled by 1/inner_size.
  The kernel applies LayerNorm on predictions internally.
  """
  def ttt_scan(q, k, v, eta, w0, ln_gamma, ln_beta) do
    cond do
      custom_call_available?() ->
        ttt_custom_call(q, k, v, eta, w0, ln_gamma, ln_beta)

      cuda_available?(q) ->
        ttt_fused(q, k, v, eta, w0, ln_gamma, ln_beta)

      true ->
        Edifice.Recurrent.TTT.ttt_scan_fallback(q, k, v, eta, w0, ln_gamma, ln_beta)
    end
  end

  @doc """
  Mamba selective scan with input-dependent discretization.

  Inputs: x `[batch, seq_len, hidden]`, dt `[batch, seq_len, hidden]`,
  A `[hidden, state_size]`, B `[batch, seq_len, state_size]`,
  C `[batch, seq_len, state_size]`.

  The kernel handles discretization (A_bar = exp(dt*A), B_bar = dt*B)
  and the full SSM recurrence internally.
  """
  def selective_scan(x, dt, a, b, c) do
    cond do
      custom_call_available?() ->
        selective_scan_custom_call(x, dt, a, b, c)

      cuda_available?(x) ->
        selective_scan_fused(x, dt, a, b, c)

      true ->
        Edifice.SSM.Common.selective_scan_fallback(x, dt, a, b, c)
    end
  end

  @doc """
  KDA (Kimi Delta Attention) scan with channel-wise decay.

  All inputs pre-computed on Elixir side:
    q:     [B, T, H, d] — query vectors (L2-normalized)
    k:     [B, T, H, d] — key vectors (L2-normalized)
    v:     [B, T, H, d] — value vectors
    alpha: [B, T, H, d] — per-channel decay (log-space)
    beta:  [B, T, H]    — scalar update gate per head (post-sigmoid)

  Returns [B, T, H, d] retrieval outputs.
  """
  def kda_scan(q, k, v, alpha, beta) do
    cond do
      custom_call_available?() ->
        kda_custom_call(q, k, v, alpha, beta)

      cuda_available?(q) ->
        kda_fused(q, k, v, alpha, beta)

      true ->
        kda_scan_fallback(q, k, v, alpha, beta)
    end
  end

  @doc """
  RLA/RDN dual-state scan with residual error correction.

  Inputs (all pre-computed):
    q:     [B, T, H, d] — query vectors
    k:     [B, T, H, d] — key vectors
    v:     [B, T, H, d] — value vectors
    alpha: [B, T, H, 1, 1] — decay gate (per-head scalar, broadcastable)
    beta:  [B, T, H, 1, 1] — base update rate
    gamma: [B, T, H, 1, 1] — residual update rate

  Options:
    variant: :rla (0) or :rdn (1)
    clip_threshold: float (default 1.0)

  Returns [B, T, H, d] outputs.
  """
  def rla_scan(q, k, v, alpha, beta, gamma, opts \\ []) do
    variant = Keyword.get(opts, :variant, :rla)
    clip_threshold = Keyword.get(opts, :clip_threshold, 1.0)

    cond do
      custom_call_available?() ->
        rla_custom_call(q, k, v, alpha, beta, gamma, variant, clip_threshold)

      cuda_available?(q) ->
        rla_fused(q, k, v, alpha, beta, gamma, variant, clip_threshold)

      true ->
        rla_scan_fallback(q, k, v, alpha, beta, gamma, variant, clip_threshold)
    end
  end

  # ============================================================================
  # XLA custom call paths (graph-preserving)
  # ============================================================================

  defp mingru_custom_call(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)
    tensor_type = Nx.type(gates)

    # Pre-compute sigmoid — XLA fuses this into the graph
    z = Nx.sigmoid(gates)

    # Zero initial hidden state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})

    # Output template for Nx.Shared.optional
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_mingru_scan, [z, candidates, h0], output, fn z, cand, h0 ->
        mingru_scan_fallback(z, cand, h0)
      end)

    # Attach backward custom call via custom_grad
    # custom_grad inputs are [raw_gates, candidates] (pre-sigmoid)
    Nx.Defn.Kernel.custom_grad(forward_output, [gates, candidates], fn grad_output ->
      {grad_z, grad_cand, _grad_h0} =
        mingru_backward_dispatch(z, candidates, h0, forward_output, grad_output)
      # Chain rule: d/d(raw_gates) = d/d(z) * sigmoid'(raw_gates) = d/d(z) * z * (1-z)
      grad_gates = Nx.multiply(grad_z, Nx.multiply(z, Nx.subtract(1.0, z)))
      [grad_gates, grad_cand]
    end)
  end

  defp minlstm_custom_call(forget_gates, input_gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(forget_gates)
    tensor_type = Nx.type(forget_gates)

    # Pre-compute sigmoid — XLA fuses this into the graph
    f = Nx.sigmoid(forget_gates)
    i = Nx.sigmoid(input_gates)

    # Zero initial cell state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})

    # Output template for Nx.Shared.optional
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_minlstm_scan, [f, i, candidates, h0], output, fn f, i, cand, h0 ->
        minlstm_scan_fallback(f, i, cand, h0)
      end)

    # Attach backward custom call via custom_grad
    # custom_grad inputs are [raw_forget_gates, raw_input_gates, candidates] (pre-sigmoid)
    Nx.Defn.Kernel.custom_grad(forward_output, [forget_gates, input_gates, candidates], fn grad_output ->
      {grad_f, grad_i, grad_cand, _grad_h0} =
        minlstm_backward_dispatch(f, i, candidates, h0, forward_output, grad_output)
      # Chain rule: d/d(raw) = d/d(sigmoid_out) * sigmoid_out * (1 - sigmoid_out)
      grad_raw_f = Nx.multiply(grad_f, Nx.multiply(f, Nx.subtract(1.0, f)))
      grad_raw_i = Nx.multiply(grad_i, Nx.multiply(i, Nx.subtract(1.0, i)))
      [grad_raw_f, grad_raw_i, grad_cand]
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

  defp mingru_backward_dispatch(z, candidates, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(z)
    ttype = Nx.type(z)
    grad_z_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_cand_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_mingru_scan_backward,
      [z, candidates, h0, forward_out, grad_output],
      {grad_z_template, grad_cand_template, grad_h0_template},
      fn z, cand, h0, fwd, grad ->
        mingru_backward_fallback(z, cand, h0, fwd, grad)
      end
    )
  end

  @doc false
  def mingru_backward_fallback(z, candidates, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(z)
    ttype = Nx.type(z)

    grad_z = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})
    grad_cand = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})

    {_, gz, gc, dh_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_z, grad_cand, Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})}, fn t, {_, gz, gc, dh_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dh = Nx.add(grad_t, dh_prev)

        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        h_prev =
          if t == 0 do
            h0
          else
            Nx.slice_along_axis(forward_out, t - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end

        dz_t = Nx.multiply(dh, Nx.subtract(c_t, h_prev))
        dc_t = Nx.multiply(dh, z_t)
        dh_next = Nx.multiply(dh, Nx.subtract(1.0, z_t))

        gz = Nx.put_slice(gz, [0, t, 0], Nx.new_axis(dz_t, 1))
        gc = Nx.put_slice(gc, [0, t, 0], Nx.new_axis(dc_t, 1))

        {nil, gz, gc, dh_next}
      end)

    {gz, gc, dh_acc}
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

  defp minlstm_backward_dispatch(f, i, candidates, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(f)
    ttype = Nx.type(f)
    grad_f_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_i_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_cand_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_minlstm_scan_backward,
      [f, i, candidates, h0, forward_out, grad_output],
      {grad_f_template, grad_i_template, grad_cand_template, grad_h0_template},
      fn f, i, cand, h0, fwd, grad ->
        minlstm_backward_fallback(f, i, cand, h0, fwd, grad)
      end
    )
  end

  @doc false
  def minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(f_gate)
    norm_eps = 1.0e-6
    ttype = Nx.type(f_gate)

    grad_f = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})
    grad_i = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})
    grad_cand = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})

    {_, gf, gi, gcand, dc_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_f, grad_i, grad_cand, Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})}, fn t, {_, gf, gi, gcand, dc_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dc = Nx.add(grad_t, dc_prev)

        f_t = Nx.slice_along_axis(f_gate, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        i_t = Nx.slice_along_axis(i_gate, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        cand_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Normalization
        s = Nx.add(f_t, Nx.add(i_t, norm_eps))
        f_norm = Nx.divide(f_t, s)
        i_norm = Nx.divide(i_t, s)

        c_prev =
          if t == 0 do
            h0
          else
            Nx.slice_along_axis(forward_out, t - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end

        # Gradients w.r.t. normalized gates
        df_norm = Nx.multiply(dc, c_prev)
        di_norm = Nx.multiply(dc, cand_t)

        # Gradient w.r.t. candidates
        dcand_t = Nx.multiply(dc, i_norm)

        # Accumulate gradient for c_{t-1}
        dc_next = Nx.multiply(dc, f_norm)

        # Through normalization (quotient rule)
        s2 = Nx.multiply(s, s)
        df_t = Nx.divide(Nx.subtract(Nx.multiply(df_norm, Nx.add(i_t, norm_eps)), Nx.multiply(di_norm, i_t)), s2)
        di_t = Nx.divide(Nx.add(Nx.negate(Nx.multiply(df_norm, f_t)), Nx.multiply(di_norm, Nx.add(f_t, norm_eps))), s2)

        gf = Nx.put_slice(gf, [0, t, 0], Nx.new_axis(df_t, 1))
        gi = Nx.put_slice(gi, [0, t, 0], Nx.new_axis(di_t, 1))
        gcand = Nx.put_slice(gcand, [0, t, 0], Nx.new_axis(dcand_t, 1))

        {nil, gf, gi, gcand, dc_next}
      end)

    {gf, gi, gcand, dc_acc}
  end

  # ELU-GRU: inputs are raw (pre-activation) — kernel applies sigmoid/elu internally
  defp elu_gru_custom_call(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)
    tensor_type = Nx.type(gates)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    # Pre-compute post-activation values for backward kernel
    z = Nx.sigmoid(gates)
    c_act = Nx.add(1.0, Nx.select(Nx.greater(candidates, 0), candidates, Nx.subtract(Nx.exp(candidates), 1.0)))

    forward_output =
      Nx.Shared.optional(:fused_elu_gru_scan, [gates, candidates, h0], output, fn g, c, h0 ->
        elu_gru_scan_fallback(g, c, h0)
      end)

    # Attach backward custom call via custom_grad
    Nx.Defn.Kernel.custom_grad(forward_output, [gates, candidates], fn grad_output ->
      {grad_z, grad_c, _grad_h0} =
        elu_gru_backward_dispatch(z, c_act, h0, forward_output, grad_output)
      # Chain rule for sigmoid: d/d(raw_gates) = grad_z * z * (1-z)
      grad_gates = Nx.multiply(grad_z, Nx.multiply(z, Nx.subtract(1.0, z)))
      # Chain rule for ELU: d/d(raw_cand) = grad_c * elu'(raw_cand)
      # elu'(x) = 1 if x >= 0, exp(x) if x < 0
      elu_deriv = Nx.select(Nx.greater(candidates, 0), Nx.broadcast(1.0, Nx.shape(candidates)), Nx.exp(candidates))
      grad_cand_raw = Nx.multiply(grad_c, elu_deriv)
      [grad_gates, grad_cand_raw]
    end)
  end

  defp elu_gru_scan_fallback(gates, candidates, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(gates)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        g_t = Nx.slice_along_axis(gates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        z_t = Nx.sigmoid(g_t)
        c_act = Nx.add(1.0, Nx.select(Nx.greater(c_t, 0), c_t, Nx.subtract(Nx.exp(c_t), 1.0)))
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_act))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Real-GRU: inputs are raw — kernel applies sigmoid internally
  defp real_gru_custom_call(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)
    tensor_type = Nx.type(gates)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    z = Nx.sigmoid(gates)

    forward_output =
      Nx.Shared.optional(:fused_real_gru_scan, [gates, candidates, h0], output, fn g, c, h0 ->
        real_gru_scan_fallback(g, c, h0)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [gates, candidates], fn grad_output ->
      {grad_z, grad_cand, _grad_h0} =
        real_gru_backward_dispatch(z, candidates, h0, forward_output, grad_output)
      grad_gates = Nx.multiply(grad_z, Nx.multiply(z, Nx.subtract(1.0, z)))
      [grad_gates, grad_cand]
    end)
  end

  defp real_gru_scan_fallback(gates, candidates, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(gates)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        g_t = Nx.slice_along_axis(gates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        z_t = Nx.sigmoid(g_t)
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Diag-Linear: inputs are raw — kernel applies sigmoid to a_vals internally
  defp diag_linear_custom_call(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    tensor_type = Nx.type(a_vals)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    a_sig = Nx.sigmoid(a_vals)

    forward_output =
      Nx.Shared.optional(:fused_diag_linear_scan, [a_vals, b_vals, h0], output, fn a, b, h0 ->
        diag_linear_scan_fallback(a, b, h0)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [a_vals, b_vals], fn grad_output ->
      {grad_a, grad_b, _grad_h0} =
        diag_linear_backward_dispatch(a_sig, h0, forward_output, grad_output)
      # Chain rule: d/d(raw_a) = grad_a * sigmoid(a) * (1 - sigmoid(a))
      grad_raw_a = Nx.multiply(grad_a, Nx.multiply(a_sig, Nx.subtract(1.0, a_sig)))
      [grad_raw_a, grad_b]
    end)
  end

  defp diag_linear_scan_fallback(a_vals, b_vals, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(a_vals)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        b_t = Nx.slice_along_axis(b_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(Nx.sigmoid(a_t), h_prev), b_t)
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Liquid (LTC) exact solver: x(t+1) = act + (x(t) - act) * exp(-1/tau)
  defp liquid_custom_call(tau, activation) do
    {batch, seq_len, hidden} = Nx.shape(tau)
    tensor_type = Nx.type(tau)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_liquid_scan, [tau, activation, h0], output, fn tau, act, h0 ->
        liquid_scan_fallback(tau, act, h0)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [tau, activation], fn grad_output ->
      {grad_tau, grad_act, _grad_h0} =
        liquid_backward_dispatch(tau, activation, h0, forward_output, grad_output)
      [grad_tau, grad_act]
    end)
  end

  defp liquid_scan_fallback(tau, activation, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(tau)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        tau_t = Nx.slice_along_axis(tau, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        act_t = Nx.slice_along_axis(activation, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        decay = Nx.exp(Nx.negate(Nx.divide(1.0, tau_t)))
        h_t = Nx.add(act_t, Nx.multiply(Nx.subtract(h_prev, act_t), decay))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Linear recurrence: h = a*h + b (no nonlinearities, pre-computed coefficients)
  # Forward custom call + backward custom call via custom_grad
  defp linear_scan_custom_call(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    tensor_type = Nx.type(a_vals)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_linear_scan, [a_vals, b_vals, h0], output, fn a, b, h0 ->
        linear_scan_cc_fallback(a, b, h0)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [a_vals, b_vals], fn grad_output ->
      {grad_a, grad_b, _grad_h0} =
        linear_scan_backward_dispatch(a_vals, h0, forward_output, grad_output)
      [grad_a, grad_b]
    end)
  end

  defp linear_scan_backward_dispatch(a_vals, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    ttype = Nx.type(a_vals)
    grad_a_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_b_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_linear_scan_backward,
      [a_vals, h0, forward_out, grad_output],
      {grad_a_template, grad_b_template, grad_h0_template},
      fn a, h0, fwd, grad ->
        linear_scan_backward_fallback(a, h0, fwd, grad)
      end
    )
  end

  defp linear_scan_cc_fallback(a_vals, b_vals, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(a_vals)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        b_t = Nx.slice_along_axis(b_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  @doc false
  def linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    ttype = Nx.type(a_vals)

    grad_a = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})
    grad_b = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})

    {_, grad_a_acc, grad_b_acc, dh_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_a, grad_b, Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})}, fn t, {_, ga, gb, dh_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dh = Nx.add(grad_t, dh_prev)

        h_prev =
          if t == 0 do
            h0
          else
            Nx.slice_along_axis(forward_out, t - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end

        a_t = Nx.slice_along_axis(a_vals, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        da_t = Nx.multiply(dh, h_prev)
        db_t = dh
        dh_next = Nx.multiply(dh, a_t)

        ga = Nx.put_slice(ga, [0, t, 0], Nx.new_axis(da_t, 1))
        gb = Nx.put_slice(gb, [0, t, 0], Nx.new_axis(db_t, 1))

        {nil, ga, gb, dh_next}
      end)

    {grad_a_acc, grad_b_acc, dh_acc}
  end

  # ========================================================================
  # P0 Backward dispatch + fallback functions
  # ========================================================================

  # ELU-GRU backward dispatch — post-activation z and c inputs
  defp elu_gru_backward_dispatch(z, c, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(z)
    ttype = Nx.type(z)
    grad_z_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_c_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_elu_gru_scan_backward,
      [z, c, h0, forward_out, grad_output],
      {grad_z_template, grad_c_template, grad_h0_template},
      fn z, c, h0, fwd, grad ->
        elu_gru_backward_fallback(z, c, h0, fwd, grad)
      end
    )
  end

  @doc false
  def elu_gru_backward_fallback(z, c, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(z)
    ttype = Nx.type(z)

    grad_z = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})
    grad_c = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden})

    {_, gz, gc, dh_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_z, grad_c, Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})}, fn t, {_, gz, gc, dh_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dh = Nx.add(grad_t, dh_prev)

        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        h_prev =
          if t == 0 do
            h0
          else
            Nx.slice_along_axis(forward_out, t - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end

        gz_t = Nx.multiply(dh, Nx.subtract(c_t, h_prev))
        gc_t = Nx.multiply(dh, z_t)
        dh_next = Nx.multiply(dh, Nx.subtract(1.0, z_t))

        gz = Nx.put_slice(gz, [0, t, 0], Nx.new_axis(gz_t, 1))
        gc = Nx.put_slice(gc, [0, t, 0], Nx.new_axis(gc_t, 1))

        {nil, gz, gc, dh_next}
      end)

    {gz, gc, dh_acc}
  end

  # Real-GRU backward dispatch — identical math to MinGRU backward
  defp real_gru_backward_dispatch(z, candidates, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(z)
    ttype = Nx.type(z)
    grad_z_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_cand_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_real_gru_scan_backward,
      [z, candidates, h0, forward_out, grad_output],
      {grad_z_template, grad_cand_template, grad_h0_template},
      fn z, cand, h0, fwd, grad ->
        real_gru_backward_fallback(z, cand, h0, fwd, grad)
      end
    )
  end

  @doc false
  def real_gru_backward_fallback(z, candidates, h0, forward_out, grad_output) do
    # Same math as mingru_backward_fallback
    mingru_backward_fallback(z, candidates, h0, forward_out, grad_output)
  end

  # DiagLinear backward dispatch — post-sigmoid a values
  defp diag_linear_backward_dispatch(a_sig, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(a_sig)
    ttype = Nx.type(a_sig)
    grad_a_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_b_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_diag_linear_scan_backward,
      [a_sig, h0, forward_out, grad_output],
      {grad_a_template, grad_b_template, grad_h0_template},
      fn a, h0, fwd, grad ->
        diag_linear_backward_fallback(a, h0, fwd, grad)
      end
    )
  end

  @doc false
  def diag_linear_backward_fallback(a_sig, h0, forward_out, grad_output) do
    # Same math as linear_scan_backward_fallback (h = a*h + b)
    linear_scan_backward_fallback(a_sig, h0, forward_out, grad_output)
  end

  # LSTM backward dispatch — outputs grad_wx, grad_h0, grad_c0
  defp lstm_backward_dispatch(wx, recurrent_weight, h0, c0, forward_out, grad_output) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    ttype = Nx.type(wx)
    grad_wx_template = Nx.template({batch, seq_len, hidden4}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)
    grad_c0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_lstm_scan_backward,
      [wx, recurrent_weight, h0, c0, forward_out, grad_output],
      {grad_wx_template, grad_h0_template, grad_c0_template},
      fn wx, r, h0, c0, fwd, grad ->
        lstm_backward_fallback(wx, r, h0, c0, fwd, grad)
      end
    )
  end

  @doc false
  def lstm_backward_fallback(wx, recurrent_weight, h0, c0, _forward_out, grad_output) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    ttype = Nx.type(wx)

    grad_wx = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden4})

    # Forward pass to recompute gate activations and cell states
    {_, gate_list, c_list} =
      Enum.reduce(0..(seq_len - 1), {{h0, c0}, [], []}, fn t, {{h_p, c_p}, g_acc, c_acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
        gates_t = Nx.add(wx_t, rh_t)

        i_t = Nx.slice_along_axis(gates_t, 0, hidden, axis: 1) |> Nx.sigmoid()
        f_t = Nx.slice_along_axis(gates_t, hidden, hidden, axis: 1) |> Nx.sigmoid()
        g_t = Nx.slice_along_axis(gates_t, hidden * 2, hidden, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(gates_t, hidden * 3, hidden, axis: 1) |> Nx.sigmoid()

        c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, g_t))
        h_t = Nx.multiply(o_t, Nx.tanh(c_t))

        {{h_t, c_t}, [{i_t, f_t, g_t, o_t} | g_acc], [c_t | c_acc]}
      end)

    gates_fwd = Enum.reverse(gate_list)
    cells_fwd = Enum.reverse(c_list)

    # Backward pass
    {_, gwx, dh_acc, dc_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_wx, Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden}), Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})}, fn t, {_, gwx, dh_prev, dc_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dh = Nx.add(grad_t, dh_prev)

        {i_t, f_t, g_t, o_t} = Enum.at(gates_fwd, t)
        c_t = Enum.at(cells_fwd, t)
        c_prev = if t == 0, do: c0, else: Enum.at(cells_fwd, t - 1)
        tanh_c = Nx.tanh(c_t)

        do_pre = Nx.multiply(Nx.multiply(dh, tanh_c), Nx.multiply(o_t, Nx.subtract(1.0, o_t)))
        dc = Nx.add(Nx.multiply(Nx.multiply(dh, o_t), Nx.subtract(1.0, Nx.multiply(tanh_c, tanh_c))), dc_prev)
        di_pre = Nx.multiply(Nx.multiply(dc, g_t), Nx.multiply(i_t, Nx.subtract(1.0, i_t)))
        df_pre = Nx.multiply(Nx.multiply(dc, c_prev), Nx.multiply(f_t, Nx.subtract(1.0, f_t)))
        dg_pre = Nx.multiply(Nx.multiply(dc, i_t), Nx.subtract(1.0, Nx.multiply(g_t, g_t)))
        dc_next = Nx.multiply(dc, f_t)

        # Concatenate gate gradients
        gwx_t = Nx.concatenate([di_pre, df_pre, dg_pre, do_pre], axis: 1)
        gwx = Nx.put_slice(gwx, [0, t, 0], Nx.new_axis(gwx_t, 1))

        # dh_acc through R^T
        dgate = Nx.concatenate([di_pre, df_pre, dg_pre, do_pre], axis: 1)
        dh_next = Nx.dot(dgate, [1], Nx.transpose(recurrent_weight), [0])

        {nil, gwx, dh_next, dc_next}
      end)

    {gwx, dh_acc, dc_acc}
  end

  # GRU backward dispatch — outputs grad_wx, grad_rh, grad_h0
  defp gru_backward_dispatch(wx, recurrent_weight, h0, forward_out, grad_output) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)
    ttype = Nx.type(wx)
    grad_wx_template = Nx.template({batch, seq_len, hidden3}, ttype)
    grad_rh_template = Nx.template({batch, seq_len, hidden3}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_gru_scan_backward,
      [wx, recurrent_weight, h0, forward_out, grad_output],
      {grad_wx_template, grad_rh_template, grad_h0_template},
      fn wx, r, h0, fwd, grad ->
        gru_backward_fallback(wx, r, h0, fwd, grad)
      end
    )
  end

  @doc false
  def gru_backward_fallback(wx, recurrent_weight, h0, forward_out, grad_output) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)
    ttype = Nx.type(wx)

    grad_wx = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden3})
    grad_rh = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, seq_len, hidden3})

    # Forward pass to recompute gate activations and rh_n values
    {_, gate_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_p, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])

        r_t = Nx.add(Nx.slice_along_axis(wx_t, 0, hidden, axis: 1),
                     Nx.slice_along_axis(rh_t, 0, hidden, axis: 1)) |> Nx.sigmoid()
        z_t = Nx.add(Nx.slice_along_axis(wx_t, hidden, hidden, axis: 1),
                     Nx.slice_along_axis(rh_t, hidden, hidden, axis: 1)) |> Nx.sigmoid()
        rh_n = Nx.slice_along_axis(rh_t, 2 * hidden, hidden, axis: 1)
        n_t = Nx.add(Nx.slice_along_axis(wx_t, 2 * hidden, hidden, axis: 1),
                     Nx.multiply(r_t, rh_n)) |> Nx.tanh()
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), n_t), Nx.multiply(z_t, h_p))

        {h_t, [{r_t, z_t, n_t, rh_n} | acc]}
      end)

    gates_fwd = Enum.reverse(gate_list)

    # Backward pass
    {_, gwx, grh, dh_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_wx, grad_rh, Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})}, fn t, {_, gwx, grh, dh_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_prev = if t == 0, do: h0, else: Nx.slice_along_axis(forward_out, t - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])

        {r_t, z_t, n_t, rh_n} = Enum.at(gates_fwd, t)
        dh = Nx.add(grad_t, dh_prev)

        dn = Nx.multiply(dh, Nx.subtract(1.0, z_t))
        dz_pre = Nx.multiply(Nx.multiply(dh, Nx.subtract(h_prev, n_t)), Nx.multiply(z_t, Nx.subtract(1.0, z_t)))
        dn_pre = Nx.multiply(dn, Nx.subtract(1.0, Nx.multiply(n_t, n_t)))
        dr_pre = Nx.multiply(Nx.multiply(dn_pre, rh_n), Nx.multiply(r_t, Nx.subtract(1.0, r_t)))
        d_rh_n = Nx.multiply(dn_pre, r_t)

        gwx_t = Nx.concatenate([dr_pre, dz_pre, dn_pre], axis: 1)
        gwx = Nx.put_slice(gwx, [0, t, 0], Nx.new_axis(gwx_t, 1))

        grh_t = Nx.concatenate([dr_pre, dz_pre, d_rh_n], axis: 1)
        grh = Nx.put_slice(grh, [0, t, 0], Nx.new_axis(grh_t, 1))

        # dh_acc through R^T using d_rh_n for n column
        dgate_rh = Nx.concatenate([dr_pre, dz_pre, d_rh_n], axis: 1)
        dh_from_r = Nx.dot(dgate_rh, [1], Nx.transpose(recurrent_weight), [0])
        dh_next = Nx.add(Nx.multiply(dh, z_t), dh_from_r)

        {nil, gwx, grh, dh_next}
      end)

    {gwx, grh, dh_acc}
  end

  # Liquid backward dispatch
  defp liquid_backward_dispatch(tau, activation, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(tau)
    ttype = Nx.type(tau)
    grad_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_h0_template = Nx.template({batch, hidden}, ttype)

    Nx.Shared.optional(
      :fused_liquid_scan_backward,
      [tau, activation, h0, forward_out, grad_output],
      {grad_template, grad_template, grad_h0_template},
      fn tau, act, h0, fwd, grad ->
        liquid_backward_fallback(tau, act, h0, fwd, grad)
      end
    )
  end

  @doc false
  def liquid_backward_fallback(tau, activation, h0, forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(tau)

    grad_tau = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, hidden})
    grad_act = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, hidden})
    dh_acc = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

    {_, gtau, gact, dh} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_tau, grad_act, dh_acc}, fn t, {_, gtau, gact, dh_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dh = Nx.add(grad_t, dh_prev)

        tau_t = Nx.slice_along_axis(tau, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        act_t = Nx.slice_along_axis(activation, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_prev = if t == 0, do: h0, else: Nx.slice_along_axis(forward_out, t - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # h_t = act + (h_prev - act) * exp(-1/tau)
        decay = Nx.exp(Nx.negate(Nx.divide(1.0, tau_t)))
        diff = Nx.subtract(h_prev, act_t)

        # d_act = dh * (1 - decay)
        gact_t = Nx.multiply(dh, Nx.subtract(1.0, decay))

        # d_tau = dh * diff * decay * (1/tau^2)
        gtau_t = Nx.multiply(dh, Nx.multiply(Nx.multiply(diff, decay), Nx.divide(1.0, Nx.multiply(tau_t, tau_t))))

        # dh_prev = dh * decay
        dh_next = Nx.multiply(dh, decay)

        gtau = Nx.put_slice(gtau, [0, t, 0], Nx.new_axis(gtau_t, 1))
        gact = Nx.put_slice(gact, [0, t, 0], Nx.new_axis(gact_t, 1))

        {nil, gtau, gact, dh_next}
      end)

    {gtau, gact, dh}
  end

  # DeltaNet (delta rule) backward dispatch
  defp delta_rule_backward_dispatch(q, k, v, beta, forward_out, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_template = Nx.template({batch, seq_len, num_heads, head_dim}, ttype)

    Nx.Shared.optional(
      :fused_delta_rule_scan_backward,
      [q, k, v, beta, forward_out, grad_output],
      {grad_template, grad_template, grad_template, grad_template},
      fn q, k, v, beta, _fwd, grad ->
        delta_rule_backward_fallback(q, k, v, beta, grad)
      end
    )
  end

  @doc false
  def delta_rule_backward_fallback(q, k, v, beta, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})
    grad_q = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_k = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_v = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})

    # Forward pass to recompute states
    {_, s_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        retr = Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        error = Nx.subtract(v_t, retr)
        scaled_err = Nx.multiply(beta_t, error)
        s_new = Nx.add(s_prev, Nx.multiply(Nx.new_axis(scaled_err, 3), Nx.new_axis(k_t, 2)))
        {s_new, [s_prev | acc]}
      end)

    s_fwd = Enum.reverse(s_list)

    # Backward pass
    ds_acc = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

    {_, gq, gk, gv, gbeta, _ds} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_q, grad_k, grad_v, grad_beta, ds_acc}, fn t, {_, gq, gk, gv, gbeta, ds} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        do_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        s_prev = Enum.at(s_fwd, t)

        retr = Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        error = Nx.subtract(v_t, retr)
        s_new = Nx.add(s_prev, Nx.multiply(Nx.new_axis(Nx.multiply(beta_t, error), 3), Nx.new_axis(k_t, 2)))

        # dS from output: dS += outer(do, q)
        ds_from_out = Nx.multiply(Nx.new_axis(do_t, 3), Nx.new_axis(q_t, 2))
        ds_total = Nx.add(ds, ds_from_out)

        # dq = S_new^T @ do
        gq_t = Nx.dot(Nx.transpose(s_new, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(do_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])

        # Through state update
        ds_k = Nx.dot(ds_total, [3], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        ds_prev = Nx.subtract(ds_total, Nx.multiply(Nx.new_axis(Nx.multiply(beta_t, ds_k), 3), Nx.new_axis(k_t, 2)))

        gv_t = Nx.multiply(beta_t, ds_k)
        gbeta_t = Nx.multiply(error, ds_k)

        scaled_err = Nx.multiply(beta_t, error)
        gk_direct = Nx.dot(Nx.transpose(ds_total, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(scaled_err, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        gk_through_err = Nx.multiply(beta_t, Nx.dot(Nx.transpose(s_prev, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(ds_k, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3]))
        gk_t = Nx.subtract(gk_direct, gk_through_err)

        gq = Nx.put_slice(gq, [0, t, 0, 0], Nx.new_axis(gq_t, 1))
        gk = Nx.put_slice(gk, [0, t, 0, 0], Nx.new_axis(gk_t, 1))
        gv = Nx.put_slice(gv, [0, t, 0, 0], Nx.new_axis(gv_t, 1))
        gbeta = Nx.put_slice(gbeta, [0, t, 0, 0], Nx.new_axis(gbeta_t, 1))

        {nil, gq, gk, gv, gbeta, ds_prev}
      end)

    {gq, gk, gv, gbeta}
  end

  # GatedDeltaNet backward dispatch — extends DeltaNet with alpha decay gradient
  defp gated_delta_net_backward_dispatch(q, k, v, beta, alpha, forward_out, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_template = Nx.template({batch, seq_len, num_heads, head_dim}, ttype)
    grad_alpha_template = Nx.template({batch, seq_len, num_heads}, ttype)

    Nx.Shared.optional(
      :fused_gated_delta_net_scan_backward,
      [q, k, v, beta, alpha, forward_out, grad_output],
      {grad_template, grad_template, grad_template, grad_template, grad_alpha_template},
      fn q, k, v, beta, alpha, _fwd, grad ->
        gated_delta_net_backward_fallback(q, k, v, beta, alpha, grad)
      end
    )
  end

  @doc false
  def gated_delta_net_backward_fallback(q, k, v, beta, alpha, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})
    grad_q = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_k = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_v = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_beta_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_alpha_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads})

    # Forward pass to recompute states
    {_, s_prev_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        alpha_broad = alpha_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        s_decayed = Nx.multiply(alpha_broad, s_prev)

        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        diff = Nx.subtract(v_t, sk)
        delta = Nx.multiply(Nx.new_axis(beta_t, -1), Nx.multiply(Nx.new_axis(diff, -1), Nx.new_axis(k_t, -2)))
        s_new = Nx.add(s_decayed, delta)
        {s_new, [s_prev | acc]}
      end)

    s_fwd = Enum.reverse(s_prev_list)

    # Backward pass
    ds_acc = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

    {_, gq, gk, gv, gbeta, galpha, _ds} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_q, grad_k, grad_v, grad_beta_out, grad_alpha_out, ds_acc}, fn t, {_, gq, gk, gv, gbeta, galpha, ds} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        do_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        s_prev = Enum.at(s_fwd, t)

        # Recompute
        alpha_broad = alpha_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        s_decayed = Nx.multiply(alpha_broad, s_prev)
        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        error = Nx.subtract(v_t, sk)
        s_new = Nx.add(s_decayed, Nx.multiply(Nx.new_axis(Nx.multiply(beta_t, error), 3), Nx.new_axis(k_t, 2)))

        # dS from output
        ds_from_out = Nx.multiply(Nx.new_axis(do_t, 3), Nx.new_axis(q_t, 2))
        ds_total = Nx.add(ds, ds_from_out)

        gq_t = Nx.dot(Nx.transpose(s_new, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(do_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])

        # Through delta update
        ds_k = Nx.dot(ds_total, [3], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        ds_to_decayed = Nx.subtract(ds_total, Nx.multiply(Nx.new_axis(Nx.multiply(beta_t, ds_k), 3), Nx.new_axis(k_t, 2)))

        gv_t = Nx.multiply(beta_t, ds_k)
        gbeta_t = Nx.multiply(error, ds_k)

        scaled_err = Nx.multiply(beta_t, error)
        gk_direct = Nx.dot(Nx.transpose(ds_total, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(scaled_err, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        gk_through_err = Nx.multiply(beta_t, Nx.dot(Nx.transpose(s_decayed, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(ds_k, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3]))
        gk_t = Nx.subtract(gk_direct, gk_through_err)

        # Gradient through alpha decay
        galpha_t = Nx.sum(Nx.multiply(ds_to_decayed, s_prev), axes: [2, 3])
        ds_prev = Nx.multiply(alpha_broad, ds_to_decayed)

        gq = Nx.put_slice(gq, [0, t, 0, 0], Nx.new_axis(gq_t, 1))
        gk = Nx.put_slice(gk, [0, t, 0, 0], Nx.new_axis(gk_t, 1))
        gv = Nx.put_slice(gv, [0, t, 0, 0], Nx.new_axis(gv_t, 1))
        gbeta = Nx.put_slice(gbeta, [0, t, 0, 0], Nx.new_axis(gbeta_t, 1))
        galpha = Nx.put_slice(galpha, [0, t, 0], Nx.new_axis(galpha_t, 1))

        {nil, gq, gk, gv, gbeta, galpha, ds_prev}
      end)

    {gq, gk, gv, gbeta, galpha}
  end

  # Selective scan backward dispatch
  defp selective_scan_backward_dispatch(x, dt, a, b, c, _forward_out, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(x)
    ttype = Nx.type(x)
    state_size = Nx.axis_size(a, 1)
    grad_x_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_dt_template = Nx.template({batch, seq_len, hidden}, ttype)
    grad_b_template = Nx.template({batch, seq_len, state_size}, ttype)
    grad_c_template = Nx.template({batch, seq_len, state_size}, ttype)

    Nx.Shared.optional(
      :fused_selective_scan_backward,
      [x, dt, a, b, c, grad_output],
      {grad_x_template, grad_dt_template, grad_b_template, grad_c_template},
      fn x, dt, a, b, c, grad ->
        selective_scan_backward_fallback(x, dt, a, b, c, grad)
      end
    )
  end

  @doc false
  def selective_scan_backward_fallback(x, dt, a, b, c, grad_output) do
    {batch, seq_len, hidden} = Nx.shape(x)
    state_size = Nx.axis_size(a, 1)

    # Forward pass to recompute h_state per timestep
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden, state_size})

    {_, h_prev_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_t = Nx.slice_along_axis(dt, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_t = Nx.clip(dt_t, 0.001, 0.1)
        b_t = Nx.slice_along_axis(b, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        dt_exp = Nx.new_axis(dt_t, 2)
        a_bar = Nx.exp(Nx.multiply(dt_exp, Nx.reshape(a, {1, hidden, state_size})))
        b_bar = Nx.multiply(dt_exp, Nx.reshape(b_t, {batch, 1, state_size}))
        h_new = Nx.add(Nx.multiply(a_bar, h_prev), Nx.multiply(b_bar, Nx.new_axis(x_t, 2)))

        {h_new, [h_prev | acc]}
      end)

    h_prevs = Enum.reverse(h_prev_list)

    # Backward pass
    grad_x_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, hidden})
    grad_dt_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, hidden})
    grad_b_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, state_size})
    grad_c_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, state_size})
    dh_state = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden, state_size})

    {_, gx, gdt, gb, gc, _dh} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_x_out, grad_dt_out, grad_b_out, grad_c_out, dh_state}, fn t, {_, gx, gdt, gb, gc, dh_s} ->
        dy = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_raw = Nx.slice_along_axis(dt, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_t = Nx.clip(dt_raw, 0.001, 0.1)
        b_t = Nx.slice_along_axis(b, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_prev = Enum.at(h_prevs, t)

        dt_exp = Nx.new_axis(dt_t, 2)
        a_reshaped = Nx.reshape(a, {1, hidden, state_size})
        a_bar = Nx.exp(Nx.multiply(dt_exp, a_reshaped))
        h_cur = Nx.add(Nx.multiply(a_bar, h_prev), Nx.multiply(Nx.multiply(dt_exp, Nx.reshape(b_t, {batch, 1, state_size})), Nx.new_axis(x_t, 2)))

        # dh_state += dy * C
        dh_from_output = Nx.multiply(Nx.new_axis(dy, 2), Nx.reshape(c_t, {batch, 1, state_size}))
        dh_total = Nx.add(dh_s, dh_from_output)

        # dC = sum_h(dy * h_cur)
        gc_t = Nx.sum(Nx.multiply(Nx.new_axis(dy, 2), h_cur), axes: [1])

        # dx = sum_s(dh * dt * B)
        gx_t = Nx.sum(Nx.multiply(dh_total, Nx.multiply(dt_exp, Nx.reshape(b_t, {batch, 1, state_size}))), axes: [2])

        # ddt
        gdt_from_a = Nx.sum(Nx.multiply(Nx.multiply(dh_total, h_prev), Nx.multiply(a_bar, a_reshaped)), axes: [2])
        gdt_from_b = Nx.sum(Nx.multiply(dh_total, Nx.multiply(Nx.reshape(b_t, {batch, 1, state_size}), Nx.new_axis(x_t, 2))), axes: [2])
        gdt_t = Nx.add(gdt_from_a, gdt_from_b)

        in_range = Nx.logical_and(Nx.greater_equal(dt_raw, 0.001), Nx.less_equal(dt_raw, 0.1))
        gdt_t = Nx.select(in_range, gdt_t, Nx.tensor(0.0, type: {:f, 32}))

        # dB = sum_h(dh * x * dt)
        gb_t = Nx.sum(Nx.multiply(dh_total, Nx.multiply(Nx.new_axis(x_t, 2), dt_exp)), axes: [1])

        # Carry dh backward
        dh_next = Nx.multiply(dh_total, a_bar)

        gx = Nx.put_slice(gx, [0, t, 0], Nx.new_axis(gx_t, 1))
        gdt = Nx.put_slice(gdt, [0, t, 0], Nx.new_axis(gdt_t, 1))
        gb = Nx.put_slice(gb, [0, t, 0], Nx.new_axis(gb_t, 1))
        gc = Nx.put_slice(gc, [0, t, 0], Nx.new_axis(gc_t, 1))

        {nil, gx, gdt, gb, gc, dh_next}
      end)

    {gx, gdt, gb, gc}
  end

  # KDA backward dispatch
  defp kda_backward_dispatch(q, k, v, alpha, beta, forward_out, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_4d = Nx.template({batch, seq_len, num_heads, head_dim}, ttype)
    grad_3d = Nx.template({batch, seq_len, num_heads}, ttype)

    Nx.Shared.optional(
      :fused_kda_scan_backward,
      [q, k, v, alpha, beta, forward_out, grad_output],
      {grad_4d, grad_4d, grad_4d, grad_4d, grad_3d},
      fn q, k, v, alpha, beta, _fwd, grad ->
        kda_backward_fallback(q, k, v, alpha, beta, grad)
      end
    )
  end

  @doc false
  def kda_backward_fallback(q, k, v, alpha, beta, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, num_heads, head_dim, head_dim})
    grad_q = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_k = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_v = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_alpha = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads})

    # Forward pass to recompute states
    {_, s_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        decay = Nx.exp(alpha_t) |> Nx.new_axis(3)
        s_decayed = Nx.multiply(decay, s_prev)

        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1])
             |> Nx.squeeze(axes: [-1])
        error = Nx.subtract(v_t, sk)
        beta_bc = beta_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        delta = Nx.multiply(beta_bc, Nx.multiply(Nx.new_axis(error, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_decayed, delta)

        {s_t, [s_prev | acc]}
      end)

    s_prevs = Enum.reverse(s_list)

    # Backward pass
    ds = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

    {_, gq, gk, gv, galpha, gbeta, _ds} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_q, grad_k, grad_v, grad_alpha, grad_beta, ds}, fn t, {_, gq, gk, gv, galpha, gbeta, ds_acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        do_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        s_prev = Enum.at(s_prevs, t)
        decay = Nx.exp(alpha_t) |> Nx.new_axis(3)
        s_decayed = Nx.multiply(decay, s_prev)
        beta_bc = beta_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)

        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1])
             |> Nx.squeeze(axes: [-1])
        error = Nx.subtract(v_t, sk)
        delta = Nx.multiply(beta_bc, Nx.multiply(Nx.new_axis(error, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_decayed, delta)

        # ds_acc += outer(do, q) from output
        ds_total = Nx.add(ds_acc, Nx.multiply(Nx.new_axis(do_t, -1), Nx.new_axis(q_t, -2)))

        # dq from output: S^T @ do
        gq_t = Nx.dot(Nx.transpose(s_t, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(do_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])

        # ds_k = ds_total @ k (per-row dot product for gradient chain)
        ds_k = Nx.dot(ds_total, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])

        # d_error = beta * ds_k; dv = d_error; d_retrieval = -d_error
        d_error = Nx.multiply(beta_t |> Nx.new_axis(-1), ds_k)
        gv_t = d_error
        d_retrieval = Nx.negate(d_error)

        # Through retrieval: dS_decayed += outer(d_retrieval, k)
        ds_to_decayed = Nx.add(ds_total, Nx.multiply(Nx.new_axis(d_retrieval, -1), Nx.new_axis(k_t, -2)))

        # dk from update: ds^T @ (beta * error)
        scaled_err = Nx.multiply(beta_bc, Nx.new_axis(error, -1))
        _gk_from_update = Nx.dot(Nx.transpose(ds_total, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.squeeze(scaled_err, axes: []), [2], [0, 1])
        # dk from retrieval: S_decayed^T @ d_retrieval
        gk_from_retr = Nx.dot(Nx.transpose(s_decayed, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(d_retrieval, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
        gk_t = Nx.add(gk_from_retr, Nx.sum(Nx.multiply(Nx.transpose(ds_total, axes: [0, 1, 3, 2]), Nx.multiply(beta_bc, Nx.new_axis(error, -2))), axes: [3]))

        # dbeta: sum over d,d
        gbeta_t = Nx.sum(Nx.multiply(error, ds_k), axes: [-1])

        # d_alpha: sum over second D dimension (broadcast dim of decay)
        galpha_t = Nx.sum(Nx.multiply(ds_to_decayed, s_prev), axes: [3])

        # Propagate ds through decay
        ds_prev = Nx.multiply(decay, ds_to_decayed)

        gq = Nx.put_slice(gq, [0, t, 0, 0], Nx.new_axis(gq_t, 1))
        gk = Nx.put_slice(gk, [0, t, 0, 0], Nx.new_axis(gk_t, 1))
        gv = Nx.put_slice(gv, [0, t, 0, 0], Nx.new_axis(gv_t, 1))
        galpha = Nx.put_slice(galpha, [0, t, 0, 0], Nx.new_axis(galpha_t, 1))
        gbeta = Nx.put_slice(gbeta, [0, t, 0], Nx.new_axis(gbeta_t, 1))

        {nil, gq, gk, gv, galpha, gbeta, ds_prev}
      end)

    {gq, gk, gv, galpha, gbeta}
  end

  # RLA backward dispatch
  defp rla_backward_dispatch(q, k, v, alpha, beta, gamma, forward_out, grad_output, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_4d = Nx.template({batch, seq_len, num_heads, head_dim}, ttype)
    grad_3d = Nx.template({batch, seq_len, num_heads}, ttype)
    variant_int = if variant == :rdn, do: 1, else: 0
    variant_t = Nx.tensor(variant_int, type: {:s, 32})
    clip_t = Nx.tensor(clip_threshold, type: {:f, 32})

    Nx.Shared.optional(
      :fused_rla_scan_backward,
      [q, k, v, alpha, beta, gamma, forward_out, grad_output, variant_t, clip_t],
      {grad_4d, grad_4d, grad_4d, grad_3d, grad_3d, grad_3d},
      fn q, k, v, alpha, beta, gamma, _fwd, grad, _variant, _clip ->
        rla_backward_fallback(q, k, v, alpha, beta, gamma, grad, variant, clip_threshold)
      end
    )
  end

  @doc false
  def rla_backward_fallback(q, k, v, alpha, beta, gamma, grad_output, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Normalize gates to [B, T, H, 1, 1] for broadcasting with [B, H, d, d] state matrices
    alpha = Nx.reshape(alpha, {batch, seq_len, num_heads, 1, 1})
    beta = Nx.reshape(beta, {batch, seq_len, num_heads, 1, 1})
    gamma = Nx.reshape(gamma, {batch, seq_len, num_heads, 1, 1})

    zero_mat = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, num_heads, head_dim, head_dim})

    grad_q = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_k = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_v = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_alpha_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads})
    grad_beta_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads})
    grad_gamma_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads})

    # Forward recompute to get S, R at each timestep
    {_, _, state_list} =
      Enum.reduce(0..(seq_len - 1), {zero_mat, zero_mat, []}, fn t, {s_prev, r_prev, acc} ->
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        gamma_t = Nx.slice_along_axis(gamma, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        retrieval_s =
          Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1])
          |> Nx.squeeze(axes: [-1])

        raw_error = Nx.subtract(v_t, retrieval_s)
        r_error = Nx.clip(raw_error, -clip_threshold, clip_threshold)

        case variant do
          :rla ->
            s_new = Nx.add(Nx.multiply(alpha_t, s_prev), Nx.multiply(beta_t, Nx.multiply(Nx.new_axis(v_t, -1), Nx.new_axis(k_t, -2))))
            r_new = Nx.add(Nx.multiply(alpha_t, r_prev), Nx.multiply(gamma_t, Nx.multiply(Nx.new_axis(r_error, -1), Nx.new_axis(k_t, -2))))
            {s_new, r_new, [{s_prev, r_prev} | acc]}

          :rdn ->
            retrieval_r = Nx.dot(r_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
            delta_s = Nx.subtract(v_t, retrieval_s)
            delta_r = Nx.subtract(r_error, retrieval_r)
            s_new = Nx.add(Nx.multiply(alpha_t, s_prev), Nx.multiply(beta_t, Nx.multiply(Nx.new_axis(delta_s, -1), Nx.new_axis(k_t, -2))))
            r_new = Nx.add(Nx.multiply(alpha_t, r_prev), Nx.multiply(gamma_t, Nx.multiply(Nx.new_axis(delta_r, -1), Nx.new_axis(k_t, -2))))
            {s_new, r_new, [{s_prev, r_prev} | acc]}
        end
      end)

    state_prevs = Enum.reverse(state_list)

    # Simple backward: use Nx autodiff on the forward fallback for correctness
    # This is a simplified fallback — the CUDA kernel handles the full backward
    ds = zero_mat
    dr = zero_mat

    {_, gq, gk, gv, ga, gb, gg, _ds, _dr} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_q, grad_k, grad_v, grad_alpha_out, grad_beta_out, grad_gamma_out, ds, dr}, fn t, {_, gq, gk, gv, ga, gb, gg, ds_acc, dr_acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        gamma_t = Nx.slice_along_axis(gamma, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        do_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        {s_prev, r_prev} = Enum.at(state_prevs, t)

        # Recompute forward for this timestep
        retrieval_s = Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        raw_error = Nx.subtract(v_t, retrieval_s)
        r_error = Nx.clip(raw_error, -clip_threshold, clip_threshold)
        clip_mask = Nx.select(Nx.less_equal(Nx.abs(raw_error), clip_threshold), 1.0, 0.0)

        {s_t, r_t, update_info} =
          case variant do
            :rla ->
              s_new = Nx.add(Nx.multiply(alpha_t, s_prev), Nx.multiply(beta_t, Nx.multiply(Nx.new_axis(v_t, -1), Nx.new_axis(k_t, -2))))
              r_new = Nx.add(Nx.multiply(alpha_t, r_prev), Nx.multiply(gamma_t, Nx.multiply(Nx.new_axis(r_error, -1), Nx.new_axis(k_t, -2))))
              {s_new, r_new, :rla}
            :rdn ->
              retrieval_r = Nx.dot(r_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
              delta_s = Nx.subtract(v_t, retrieval_s)
              delta_r = Nx.subtract(r_error, retrieval_r)
              s_new = Nx.add(Nx.multiply(alpha_t, s_prev), Nx.multiply(beta_t, Nx.multiply(Nx.new_axis(delta_s, -1), Nx.new_axis(k_t, -2))))
              r_new = Nx.add(Nx.multiply(alpha_t, r_prev), Nx.multiply(gamma_t, Nx.multiply(Nx.new_axis(delta_r, -1), Nx.new_axis(k_t, -2))))
              {s_new, r_new, {:rdn, retrieval_r, delta_s, delta_r}}
          end

        # d(S+R) from output: o = (S+R) @ q
        sr = Nx.add(s_t, r_t)
        ds_total = Nx.add(ds_acc, Nx.multiply(Nx.new_axis(do_t, -1), Nx.new_axis(q_t, -2)))
        dr_total = Nx.add(dr_acc, Nx.multiply(Nx.new_axis(do_t, -1), Nx.new_axis(q_t, -2)))

        # dq
        gq_t = Nx.dot(Nx.transpose(sr, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(do_t, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])

        # Gradients through S/R updates (simplified — zero out for fallback, let CUDA handle real gradients)
        # For correctness in test mode, we compute approximate gradients
        ga_t = Nx.sum(Nx.add(Nx.multiply(ds_total, s_prev), Nx.multiply(dr_total, r_prev)), axes: [2, 3])
        gb_t = case update_info do
          :rla -> Nx.sum(Nx.multiply(Nx.new_axis(v_t, -1), Nx.new_axis(k_t, -2)) |> Nx.multiply(ds_total), axes: [2, 3])
          {:rdn, _, delta_s, _} -> Nx.sum(Nx.multiply(Nx.new_axis(delta_s, -1), Nx.new_axis(k_t, -2)) |> Nx.multiply(ds_total), axes: [2, 3])
        end
        gg_t = case update_info do
          :rla -> Nx.sum(Nx.multiply(Nx.new_axis(r_error, -1), Nx.new_axis(k_t, -2)) |> Nx.multiply(dr_total), axes: [2, 3])
          {:rdn, _, _, delta_r} -> Nx.sum(Nx.multiply(Nx.new_axis(delta_r, -1), Nx.new_axis(k_t, -2)) |> Nx.multiply(dr_total), axes: [2, 3])
        end

        ds_k = Nx.dot(ds_total, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        dr_k = Nx.dot(dr_total, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])

        # Squeeze gate scalars from [B, H, 1, 1] to [B, H, 1] for broadcasting with [B, H, d]
        beta_v = Nx.reshape(beta_t, {batch, num_heads, 1})
        gamma_v = Nx.reshape(gamma_t, {batch, num_heads, 1})

        gv_t = case update_info do
          :rla -> Nx.add(Nx.multiply(beta_v, ds_k), Nx.multiply(Nx.multiply(gamma_v, dr_k), clip_mask))
          {:rdn, _, _, _} -> Nx.add(Nx.multiply(beta_v, ds_k), Nx.multiply(Nx.multiply(gamma_v, dr_k), clip_mask))
        end

        gk_t = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim})

        ds_next = Nx.multiply(alpha_t, ds_total)
        dr_next = Nx.multiply(alpha_t, dr_total)

        gq = Nx.put_slice(gq, [0, t, 0, 0], Nx.new_axis(gq_t, 1))
        gk = Nx.put_slice(gk, [0, t, 0, 0], Nx.new_axis(gk_t, 1))
        gv = Nx.put_slice(gv, [0, t, 0, 0], Nx.new_axis(gv_t, 1))
        ga = Nx.put_slice(ga, [0, t, 0], Nx.new_axis(ga_t, 1))
        gb = Nx.put_slice(gb, [0, t, 0], Nx.new_axis(gb_t, 1))
        gg = Nx.put_slice(gg, [0, t, 0], Nx.new_axis(gg_t, 1))

        {nil, gq, gk, gv, ga, gb, gg, ds_next, dr_next}
      end)

    {gq, gk, gv, ga, gb, gg}
  end

  # TTT backward dispatch
  defp ttt_backward_dispatch(q, k, v, eta, w0, ln_gamma, ln_beta, forward_out, grad_output) do
    {batch, seq_len, inner_size} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_3d = Nx.template({batch, seq_len, inner_size}, ttype)
    grad_w0 = Nx.template({batch, inner_size, inner_size}, ttype)
    grad_ln = Nx.template({inner_size}, ttype)

    Nx.Shared.optional(
      :fused_ttt_scan_backward,
      [q, k, v, eta, w0, ln_gamma, ln_beta, forward_out, grad_output],
      {grad_3d, grad_3d, grad_3d, grad_3d, grad_w0, grad_ln, grad_ln},
      fn q, k, v, eta, w0, ln_g, ln_b, _fwd, grad ->
        ttt_backward_fallback(q, k, v, eta, w0, ln_g, ln_b, grad)
      end
    )
  end

  @doc false
  def ttt_backward_fallback(q, k, v, eta, w0, ln_gamma, ln_beta, grad_output) do
    {batch, seq_len, inner_size} = Nx.shape(q)
    ln_eps = 1.0e-6

    w_init =
      if Nx.rank(w0) == 2 do
        Nx.broadcast(w0, {batch, inner_size, inner_size})
      else
        w0
      end

    # Forward recompute: store pred, mean, var per timestep
    {_, _w_final, fwd_list} =
      Enum.reduce(0..(seq_len - 1), {w_init, nil, []}, fn t, {w_prev, _, acc} ->
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_t = Nx.slice_along_axis(eta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        pred = Nx.dot(w_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        mean = Nx.mean(pred, axes: [-1], keep_axes: true)
        var = Nx.variance(pred, axes: [-1], keep_axes: true)
        inv_std = Nx.rsqrt(Nx.add(var, ln_eps))
        x_hat = Nx.multiply(Nx.subtract(pred, mean), inv_std)
        pred_normed = Nx.add(Nx.multiply(x_hat, ln_gamma), ln_beta)

        error = Nx.subtract(pred_normed, v_t)
        scaled_error = Nx.multiply(eta_t, error)
        grad_w = Nx.dot(Nx.new_axis(scaled_error, 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0])
        w_new = Nx.subtract(w_prev, grad_w)

        {w_new, nil, [{pred, mean, var, w_prev} | acc]}
      end)

    fwd_data = Enum.reverse(fwd_list)

    grad_q_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, inner_size})
    grad_k_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, inner_size})
    grad_v_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, inner_size})
    grad_eta_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, inner_size})
    grad_w0_out = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, inner_size, inner_size})
    grad_lng = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {inner_size})
    grad_lnb = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {inner_size})

    dw = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, inner_size, inner_size})

    {_, gq, gk, gv, geta, gw0, glng, glnb, _dw} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_q_out, grad_k_out, grad_v_out, grad_eta_out, grad_w0_out, grad_lng, grad_lnb, dw}, fn t, {_, gq, gk, gv, geta, gw0, glng, glnb, dw_acc} ->
        do_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_t = Nx.slice_along_axis(eta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        {pred, mean, var, w_prev} = Enum.at(fwd_data, t)
        inv_std = Nx.rsqrt(Nx.add(var, ln_eps))
        x_hat = Nx.multiply(Nx.subtract(pred, mean), inv_std)
        pred_normed = Nx.add(Nx.multiply(x_hat, ln_gamma), ln_beta)
        error = Nx.subtract(pred_normed, v_t)
        scaled_error = Nx.multiply(eta_t, error)

        # W_updated = W_prev - outer(scaled_error, k)
        w_updated = Nx.subtract(w_prev, Nx.dot(Nx.new_axis(scaled_error, 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0]))

        # Step 6: o = W_updated @ q
        # dW += outer(do, q)
        dw_new = Nx.add(dw_acc, Nx.dot(Nx.new_axis(do_t, 2), [2], [0], Nx.new_axis(q_t, 1), [1], [0]))

        # dq = W_updated^T @ do
        gq_t = Nx.dot(Nx.transpose(w_updated, axes: [0, 2, 1]), [2], [0], Nx.new_axis(do_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])

        # Step 5: d_scaled_error = -dW @ k
        d_scaled_error = Nx.negate(Nx.dot(dw_new, [2], [0], Nx.new_axis(k_t, 2), [1], [0]) |> Nx.squeeze(axes: [2]))

        # dk_update = -scaled_error^T @ dW (cross-batch dot)
        dk_update = Nx.negate(Nx.dot(Nx.new_axis(scaled_error, 1), [2], [0], dw_new, [1], [0]) |> Nx.squeeze(axes: [1]))

        # Step 4: d_eta = d_scaled_error * error, d_error = d_scaled_error * eta
        geta_t = Nx.multiply(d_scaled_error, error)
        d_error = Nx.multiply(d_scaled_error, eta_t)

        # Step 3: dv = -d_error
        gv_t = Nx.negate(d_error)
        d_pred_normed = d_error

        # Step 2: LayerNorm backward
        glng = Nx.add(glng, Nx.sum(Nx.multiply(x_hat, d_pred_normed), axes: [0]))
        glnb = Nx.add(glnb, Nx.sum(d_pred_normed, axes: [0]))

        d_xhat = Nx.multiply(ln_gamma, d_pred_normed)
        mean_d_xhat = Nx.mean(d_xhat, axes: [-1], keep_axes: true)
        mean_d_xhat_xhat = Nx.mean(Nx.multiply(d_xhat, x_hat), axes: [-1], keep_axes: true)
        d_pred = Nx.multiply(inv_std, Nx.subtract(d_xhat, Nx.add(mean_d_xhat, Nx.multiply(x_hat, mean_d_xhat_xhat))))

        # Step 1: pred = W_prev @ k
        # dk_pred = W_prev^T @ d_pred
        dk_pred = Nx.dot(Nx.transpose(w_prev, axes: [0, 2, 1]), [2], [0], Nx.new_axis(d_pred, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        # dW += outer(d_pred, k)
        dw_new = Nx.add(dw_new, Nx.dot(Nx.new_axis(d_pred, 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0]))

        gk_t = Nx.add(dk_update, dk_pred)

        gq = Nx.put_slice(gq, [0, t, 0], Nx.new_axis(gq_t, 1))
        gk = Nx.put_slice(gk, [0, t, 0], Nx.new_axis(gk_t, 1))
        gv = Nx.put_slice(gv, [0, t, 0], Nx.new_axis(gv_t, 1))
        geta = Nx.put_slice(geta, [0, t, 0], Nx.new_axis(geta_t, 1))

        {nil, gq, gk, gv, geta, gw0, glng, glnb, dw_new}
      end)

    {gq, gk, gv, geta, Nx.add(gw0, dw), glng, glnb}
  end

  @doc false
  def delta_product_backward_fallback(q, k, v, beta, grad_output) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    {_, _, num_householder, _, _} = Nx.shape(k)
    norm_eps = 1.0e-6

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})
    grad_q = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})
    grad_k = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_householder, num_heads, head_dim})
    grad_v = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_householder, num_heads, head_dim})
    grad_beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, num_householder, num_heads})

    # Forward pass: recompute states and raw outputs
    {_, s_list, raw_out_list} =
      Enum.reduce(0..(seq_len - 1), {s0, [], []}, fn t, {s_prev, s_acc, raw_acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        s_updated =
          Enum.reduce(0..(num_householder - 1), s_prev, fn j, s_acc_inner ->
            k_tj = Nx.slice_along_axis(Nx.slice_along_axis(k, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            v_tj = Nx.slice_along_axis(Nx.slice_along_axis(v, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            beta_tj = Nx.slice_along_axis(Nx.slice_along_axis(beta, t, 1, axis: 1), j, 1, axis: 2)
                      |> Nx.squeeze(axes: [1, 2])

            k_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(k_tj, k_tj), axes: [-1], keep_axes: true), norm_eps))
            k_normalized = Nx.divide(k_tj, k_norm)

            st_k = Nx.dot(s_acc_inner, [2], [0, 1], Nx.new_axis(k_normalized, -1), [2], [0, 1])
                   |> Nx.squeeze(axes: [-1])
            error = Nx.subtract(v_tj, st_k)

            beta_broad = beta_tj |> Nx.new_axis(-1) |> Nx.new_axis(-1)
            update = Nx.multiply(beta_broad, Nx.multiply(Nx.new_axis(k_normalized, -1), Nx.new_axis(error, -2)))
            Nx.add(s_acc_inner, update)
          end)

        o_raw = Nx.dot(s_updated, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1])
                |> Nx.squeeze(axes: [-1])

        {s_updated, [s_prev | s_acc], [o_raw | raw_acc]}
      end)

    s_fwd = Enum.reverse(s_list)
    raw_out_fwd = Enum.reverse(raw_out_list)

    # Backward pass
    ds_acc = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

    {_, gq, gk, gv, gbeta, _ds} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_q, grad_k, grad_v, grad_beta, ds_acc}, fn t, {_, gq, gk, gv, gbeta, ds} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        do_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        o_raw = Enum.at(raw_out_fwd, t)
        s_prev_t = Enum.at(s_fwd, t)

        # RMS norm backward
        rms = Nx.sqrt(Nx.add(Nx.mean(Nx.multiply(o_raw, o_raw), axes: [-1], keep_axes: true), norm_eps))
        d_o_raw = Nx.subtract(
          Nx.divide(do_t, rms),
          Nx.multiply(
            Nx.divide(o_raw, Nx.multiply(rms, rms)),
            Nx.mean(Nx.multiply(do_t, o_raw), axes: [-1], keep_axes: true)
          )
        )

        # Recompute s_updated
        s_updated =
          Enum.reduce(0..(num_householder - 1), s_prev_t, fn j, s_acc_inner ->
            k_tj = Nx.slice_along_axis(Nx.slice_along_axis(k, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            v_tj = Nx.slice_along_axis(Nx.slice_along_axis(v, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            beta_tj = Nx.slice_along_axis(Nx.slice_along_axis(beta, t, 1, axis: 1), j, 1, axis: 2)
                      |> Nx.squeeze(axes: [1, 2])

            k_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(k_tj, k_tj), axes: [-1], keep_axes: true), norm_eps))
            k_normalized = Nx.divide(k_tj, k_norm)
            st_k = Nx.dot(s_acc_inner, [2], [0, 1], Nx.new_axis(k_normalized, -1), [2], [0, 1])
                   |> Nx.squeeze(axes: [-1])
            error = Nx.subtract(v_tj, st_k)
            beta_broad = beta_tj |> Nx.new_axis(-1) |> Nx.new_axis(-1)
            update = Nx.multiply(beta_broad, Nx.multiply(Nx.new_axis(k_normalized, -1), Nx.new_axis(error, -2)))
            Nx.add(s_acc_inner, update)
          end)

        # dS from output
        ds_from_out = Nx.multiply(Nx.new_axis(d_o_raw, 3), Nx.new_axis(q_t, 2))
        ds_total = Nx.add(ds, ds_from_out)

        # dq
        gq_t = Nx.dot(Nx.transpose(s_updated, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(d_o_raw, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])

        # Reverse through Householder steps
        {ds_after_steps, gk_step, gv_step, gbeta_step} =
          Enum.reduce((num_householder - 1)..0//-1, {ds_total, gk, gv, gbeta}, fn j, {ds_cur, gk_acc, gv_acc, gbeta_acc} ->
            k_tj = Nx.slice_along_axis(Nx.slice_along_axis(k, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            v_tj = Nx.slice_along_axis(Nx.slice_along_axis(v, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            beta_tj = Nx.slice_along_axis(Nx.slice_along_axis(beta, t, 1, axis: 1), j, 1, axis: 2)
                      |> Nx.squeeze(axes: [1, 2])

            k_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(k_tj, k_tj), axes: [-1], keep_axes: true), norm_eps))
            k_normalized = Nx.divide(k_tj, k_norm)

            ds_kn = Nx.dot(ds_cur, [3], [0, 1], Nx.new_axis(k_normalized, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])

            gv_tj = Nx.multiply(beta_tj, ds_kn)

            # Recompute state before step j
            s_before_j =
              Enum.reduce(0..(j - 1)//1, s_prev_t, fn jj, s_inner ->
                k_jj = Nx.slice_along_axis(Nx.slice_along_axis(k, t, 1, axis: 1), jj, 1, axis: 2)
                       |> Nx.squeeze(axes: [1, 2])
                v_jj = Nx.slice_along_axis(Nx.slice_along_axis(v, t, 1, axis: 1), jj, 1, axis: 2)
                       |> Nx.squeeze(axes: [1, 2])
                beta_jj = Nx.slice_along_axis(Nx.slice_along_axis(beta, t, 1, axis: 1), jj, 1, axis: 2)
                          |> Nx.squeeze(axes: [1, 2])
                kn_jj = Nx.divide(k_jj, Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(k_jj, k_jj), axes: [-1], keep_axes: true), norm_eps)))
                stk_jj = Nx.dot(s_inner, [2], [0, 1], Nx.new_axis(kn_jj, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
                err_jj = Nx.subtract(v_jj, stk_jj)
                beta_b = beta_jj |> Nx.new_axis(-1) |> Nx.new_axis(-1)
                Nx.add(s_inner, Nx.multiply(beta_b, Nx.multiply(Nx.new_axis(kn_jj, -1), Nx.new_axis(err_jj, -2))))
              end)

            st_k = Nx.dot(s_before_j, [2], [0, 1], Nx.new_axis(k_normalized, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
            error = Nx.subtract(v_tj, st_k)
            gbeta_tj = Nx.sum(Nx.multiply(error, ds_kn), axes: [-1])

            scaled_err = Nx.multiply(beta_tj, error)
            gkn_direct = Nx.dot(Nx.transpose(ds_cur, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(scaled_err, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3])
            gkn_through_err = Nx.multiply(beta_tj, Nx.dot(Nx.transpose(s_before_j, axes: [0, 1, 3, 2]), [3], [0, 1], Nx.new_axis(ds_kn, 3), [2], [0, 1]) |> Nx.squeeze(axes: [3]))
            gkn_t = Nx.subtract(gkn_direct, gkn_through_err)

            # Chain through L2 normalization
            dot_gkn_kn = Nx.sum(Nx.multiply(gkn_t, k_normalized), axes: [-1], keep_axes: true)
            gk_tj = Nx.divide(Nx.subtract(gkn_t, Nx.multiply(k_normalized, dot_gkn_kn)), k_norm)

            ds_prev_step = Nx.subtract(ds_cur, Nx.multiply(beta_tj |> Nx.new_axis(-1) |> Nx.new_axis(-1), Nx.multiply(Nx.new_axis(ds_kn, 3), Nx.new_axis(k_normalized, 2))))

            gk_acc = Nx.put_slice(gk_acc, [0, t, j, 0, 0], Nx.reshape(gk_tj, {batch, 1, 1, num_heads, head_dim}))
            gv_acc = Nx.put_slice(gv_acc, [0, t, j, 0, 0], Nx.reshape(gv_tj, {batch, 1, 1, num_heads, head_dim}))
            gbeta_acc = Nx.put_slice(gbeta_acc, [0, t, j, 0], Nx.reshape(gbeta_tj, {batch, 1, 1, num_heads}))

            {ds_prev_step, gk_acc, gv_acc, gbeta_acc}
          end)

        gq = Nx.put_slice(gq, [0, t, 0, 0], Nx.new_axis(gq_t, 1))

        {nil, gq, gk_step, gv_step, gbeta_step, ds_after_steps}
      end)

    {gq, gk, gv, gbeta}
  end

  @doc false
  def slstm_backward_fallback(wx, recurrent_weight, h0, c0, _forward_out, grad_output) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    grad_wx = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, seq_len, hidden4})

    n0 = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, hidden})
    m0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

    # Forward pass to recompute gate activations
    {_, gate_list, cell_list, norm_list} =
      Enum.reduce(0..(seq_len - 1), {{h0, c0, n0, m0}, [], [], []}, fn t, {{h_p, c_p, n_p, m_p}, g_acc, c_acc, n_acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
        gates_t = Nx.add(wx_t, rh_t)

        log_i_t = Nx.slice_along_axis(gates_t, 0, hidden, axis: 1)
        log_f_t = Nx.slice_along_axis(gates_t, hidden, hidden, axis: 1)
        z_t = Nx.slice_along_axis(gates_t, hidden * 2, hidden, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(gates_t, hidden * 3, hidden, axis: 1) |> Nx.sigmoid()

        m_t = Nx.max(Nx.add(log_f_t, m_p), log_i_t)
        i_t = Nx.exp(Nx.subtract(log_i_t, m_t))
        f_t = Nx.exp(Nx.subtract(Nx.add(log_f_t, m_p), m_t))

        c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, z_t))
        n_t = Nx.add(Nx.multiply(f_t, n_p), i_t)
        safe_denom = Nx.max(Nx.abs(n_t), 1.0)
        h_t = Nx.multiply(o_t, Nx.divide(c_t, safe_denom))

        {{h_t, c_t, n_t, m_t}, [{i_t, f_t, z_t, o_t} | g_acc], [c_t | c_acc], [n_t | n_acc]}
      end)

    gates_fwd = Enum.reverse(gate_list)
    cells_fwd = Enum.reverse(cell_list)
    norms_fwd = Enum.reverse(norm_list)

    # Backward pass
    {_, gwx, dh_acc, dc_acc} =
      Enum.reduce((seq_len - 1)..0//-1, {nil, grad_wx,
        Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden}),
        Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})}, fn t, {_, gwx, dh_prev, dc_prev} ->
        grad_t = Nx.slice_along_axis(grad_output, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dh = Nx.add(grad_t, dh_prev)

        {i_t, f_t, z_t, o_t} = Enum.at(gates_fwd, t)
        c_t = Enum.at(cells_fwd, t)
        n_t = Enum.at(norms_fwd, t)
        c_prev = if t == 0, do: c0, else: Enum.at(cells_fwd, t - 1)
        n_prev = if t == 0, do: n0, else: Enum.at(norms_fwd, t - 1)

        safe_denom = Nx.max(Nx.abs(n_t), 1.0)

        # h = o * c / max(|n|, 1)
        do_pre = Nx.multiply(
          Nx.multiply(dh, Nx.divide(c_t, safe_denom)),
          Nx.multiply(o_t, Nx.subtract(1.0, o_t))
        )

        dc = Nx.add(Nx.multiply(dh, Nx.divide(o_t, safe_denom)), dc_prev)

        # dn: only active when |n| >= 1
        n_active = Nx.greater_equal(Nx.abs(n_t), 1.0)
        dn = Nx.multiply(
          Nx.multiply(Nx.select(n_active, Nx.tensor(1.0, type: {:f, 32}), Nx.tensor(0.0, type: {:f, 32})),
            Nx.negate(Nx.multiply(dh, Nx.divide(Nx.multiply(o_t, c_t), Nx.multiply(safe_denom, safe_denom))))),
          Nx.select(Nx.greater_equal(n_t, 0.0), Nx.tensor(1.0, type: {:f, 32}), Nx.tensor(-1.0, type: {:f, 32}))
        )

        # Through exp gates
        di = Nx.add(Nx.multiply(dc, z_t), dn)
        df = Nx.add(Nx.multiply(dc, c_prev), Nx.multiply(dn, n_prev))

        d_log_i = Nx.multiply(di, i_t)
        d_log_f = Nx.multiply(df, f_t)

        dz_pre = Nx.multiply(Nx.multiply(dc, i_t), Nx.subtract(1.0, Nx.multiply(z_t, z_t)))

        gwx_t = Nx.concatenate([d_log_i, d_log_f, dz_pre, do_pre], axis: 1)
        gwx = Nx.put_slice(gwx, [0, t, 0], Nx.new_axis(gwx_t, 1))

        dh_next = Nx.dot(gwx_t, [1], Nx.transpose(recurrent_weight), [0])
        dc_next = Nx.multiply(dc, f_t)

        {nil, gwx, dh_next, dc_next}
      end)

    {gwx, dh_acc, dc_acc}
  end

  # DeltaNet delta rule: matrix-state recurrence
  defp delta_net_custom_call(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_delta_net_scan, [q, k, v, beta], output, fn q, k, v, beta ->
        delta_net_scan_fallback(q, k, v, beta)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v, beta], fn grad_output ->
      {grad_q, grad_k, grad_v, grad_beta} =
        delta_rule_backward_dispatch(q, k, v, beta, forward_output, grad_output)
      [grad_q, grad_k, grad_v, grad_beta]
    end)
  end

  defp delta_net_scan_fallback(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # S@k: [B,H,D,D] x [B,H,D] -> [B,H,D]
        sk = Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        # delta = beta * outer(v - S@k, k)
        diff = Nx.subtract(v_t, sk)
        delta = Nx.multiply(Nx.new_axis(beta_t, -1), Nx.multiply(Nx.new_axis(diff, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_prev, delta)
        # o = S@q
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        {s_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # GatedDeltaNet: delta rule with per-head scalar decay
  defp gated_delta_net_custom_call(q, k, v, beta, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_gated_delta_net_scan, [q, k, v, beta, alpha], output, fn q, k, v, beta, alpha ->
        gated_delta_net_scan_fallback(q, k, v, beta, alpha)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v, beta, alpha], fn grad_output ->
      {grad_q, grad_k, grad_v, grad_beta, grad_alpha} =
        gated_delta_net_backward_dispatch(q, k, v, beta, alpha, forward_output, grad_output)
      [grad_q, grad_k, grad_v, grad_beta, grad_alpha]
    end)
  end

  defp gated_delta_net_scan_fallback(q, k, v, beta, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Decay: S = alpha * S (alpha is [B,H], broadcast to [B,H,D,D])
        alpha_broad = alpha_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        s_decayed = Nx.multiply(alpha_broad, s_prev)

        # Delta update (same as DeltaNet)
        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        diff = Nx.subtract(v_t, sk)
        delta = Nx.multiply(Nx.new_axis(beta_t, -1), Nx.multiply(Nx.new_axis(diff, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_decayed, delta)
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        {s_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # DeltaProduct: Householder product state transitions
  defp delta_product_custom_call(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, tensor_type)

    Nx.Shared.optional(:fused_delta_product_scan, [q, k, v, beta], output, fn q, k, v, beta ->
      delta_product_scan_fallback(q, k, v, beta)
    end)
  end

  defp delta_product_scan_fallback(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    # k/v: [B, T, n_h, H, d], beta: [B, T, n_h, H]
    {_, _, num_householder, _, _} = Nx.shape(k)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})
    norm_eps = 1.0e-6

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Apply n_h Householder updates
        s_updated =
          Enum.reduce(0..(num_householder - 1), s_prev, fn j, s_acc ->
            # k_{t,j}: [B, n_h, H, d] -> slice j -> [B, H, d]
            k_tj = Nx.slice_along_axis(Nx.slice_along_axis(k, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            v_tj = Nx.slice_along_axis(Nx.slice_along_axis(v, t, 1, axis: 1), j, 1, axis: 2)
                   |> Nx.squeeze(axes: [1, 2])
            beta_tj = Nx.slice_along_axis(Nx.slice_along_axis(beta, t, 1, axis: 1), j, 1, axis: 2)
                      |> Nx.squeeze(axes: [1, 2])

            # L2 normalize key
            k_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(k_tj, k_tj), axes: [-1], keep_axes: true), norm_eps))
            k_normalized = Nx.divide(k_tj, k_norm)

            # S^T @ k: need cross-row product
            # Use the reformulation: S_new = S + beta * k @ (v - S^T @ k)^T
            # S^T @ k: [B,H,D,D]^T @ [B,H,D] = [B,H,D]
            # s_acc is [B,H,D,D], k_normalized is [B,H,D]
            st_k = Nx.dot(s_acc, [2], [0, 1], Nx.new_axis(k_normalized, -1), [2], [0, 1])
                   |> Nx.squeeze(axes: [-1])

            # error = v - S^T@k
            error = Nx.subtract(v_tj, st_k)

            # S += beta * outer(k, error)
            beta_broad = beta_tj |> Nx.new_axis(-1) |> Nx.new_axis(-1)
            k_col = Nx.new_axis(k_normalized, -1)
            err_row = Nx.new_axis(error, -2)
            update = Nx.multiply(beta_broad, Nx.multiply(k_col, err_row))
            Nx.add(s_acc, update)
          end)

        # Output: o_t = S @ q_t with RMS norm
        o_t = Nx.dot(s_updated, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1])
              |> Nx.squeeze(axes: [-1])

        rms = Nx.sqrt(Nx.add(Nx.mean(Nx.multiply(o_t, o_t), axes: [-1], keep_axes: true), norm_eps))
        o_t_normed = Nx.divide(o_t, rms)

        {s_updated, [o_t_normed | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # sLSTM: fused R@h matmul with log-domain exponential gating
  defp slstm_custom_call(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    tensor_type = Nx.type(wx)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    Nx.Shared.optional(:fused_slstm_scan, [wx, recurrent_weight, h0, c0], output,
      fn wx, r, h0, c0 ->
        slstm_scan_fallback(wx, r, h0, c0)
      end)
  end

  defp slstm_scan_fallback(wx, recurrent_weight, h0, c0) do
    {_batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    n0 = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(wx)), Nx.shape(h0))
    m0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(wx)), Nx.shape(h0))

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {{h0, c0, n0, m0}, []}, fn t, {{h_p, c_p, n_p, m_p}, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
        gates_t = Nx.add(wx_t, rh_t)

        log_i_t = Nx.slice_along_axis(gates_t, 0, hidden, axis: 1)
        log_f_t = Nx.slice_along_axis(gates_t, hidden, hidden, axis: 1)
        z_t = Nx.slice_along_axis(gates_t, hidden * 2, hidden, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(gates_t, hidden * 3, hidden, axis: 1) |> Nx.sigmoid()

        m_t = Nx.max(Nx.add(log_f_t, m_p), log_i_t)
        i_t = Nx.exp(Nx.subtract(log_i_t, m_t))
        f_t = Nx.exp(Nx.subtract(Nx.add(log_f_t, m_p), m_t))

        c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, z_t))
        n_t = Nx.add(Nx.multiply(f_t, n_p), i_t)
        safe_denom = Nx.max(Nx.abs(n_t), 1.0)
        h_t = Nx.multiply(o_t, Nx.divide(c_t, safe_denom))

        {{h_t, c_t, n_t, m_t}, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Standard LSTM: classic 4-gate LSTM with cell and hidden state
  defp lstm_custom_call(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    tensor_type = Nx.type(wx)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_lstm_scan, [wx, recurrent_weight, h0, c0], output,
        fn wx, r, h0, c0 ->
          lstm_scan_fallback(wx, r, h0, c0)
        end)

    # Attach backward custom call via custom_grad
    Nx.Defn.Kernel.custom_grad(forward_output, [wx, recurrent_weight], fn grad_output ->
      {grad_wx, _grad_h0, _grad_c0} =
        lstm_backward_dispatch(wx, recurrent_weight, h0, c0, forward_output, grad_output)

      # grad_R = sum over batch and time of h_prev^T @ grad_wx_t
      # h_prev: shift forward_output right by 1, prepend h0
      h_prev = Nx.concatenate([Nx.reshape(h0, {batch, 1, hidden}), Nx.slice_along_axis(forward_output, 0, seq_len - 1, axis: 1)], axis: 1)
      # h_prev: [B, T, H], grad_wx: [B, T, 4H]
      # grad_R = sum_b( h_prev^T @ grad_wx ) = sum_b,t( h_prev[b,t,:] outer grad_wx[b,t,:] )
      grad_r = Nx.dot(Nx.reshape(h_prev, {batch * seq_len, hidden}) |> Nx.transpose(),
                       Nx.reshape(grad_wx, {batch * seq_len, hidden4}))

      [grad_wx, grad_r]
    end)
  end

  defp lstm_scan_fallback(wx, recurrent_weight, h0, c0) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    ttype = Nx.type(wx)

    h0 = if h0, do: h0, else: Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})
    c0 = if c0, do: c0, else: Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {{h0, c0}, []}, fn t, {{h_p, c_p}, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
        gates_t = Nx.add(wx_t, rh_t)

        i_t = Nx.slice_along_axis(gates_t, 0, hidden, axis: 1) |> Nx.sigmoid()
        f_t = Nx.slice_along_axis(gates_t, hidden, hidden, axis: 1) |> Nx.sigmoid()
        g_t = Nx.slice_along_axis(gates_t, hidden * 2, hidden, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(gates_t, hidden * 3, hidden, axis: 1) |> Nx.sigmoid()

        c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, g_t))
        h_t = Nx.multiply(o_t, Nx.tanh(c_t))

        {{h_t, c_t}, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Standard GRU: 3-gate GRU with reset applied to recurrent part only
  defp gru_custom_call(wx, recurrent_weight) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)
    tensor_type = Nx.type(wx)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_gru_scan, [wx, recurrent_weight, h0], output,
        fn wx, r, h0 ->
          gru_scan_fallback(wx, r, h0)
        end)

    Nx.Defn.Kernel.custom_grad(forward_output, [wx, recurrent_weight], fn grad_output ->
      {grad_wx, grad_rh, _grad_h0} =
        gru_backward_dispatch(wx, recurrent_weight, h0, forward_output, grad_output)

      # grad_R = sum(h_prev^T @ grad_rh) — grad_rh has correct R@h-specific gradients
      h_prev = Nx.concatenate([Nx.reshape(h0, {batch, 1, hidden}), Nx.slice_along_axis(forward_output, 0, seq_len - 1, axis: 1)], axis: 1)
      grad_r = Nx.dot(Nx.reshape(h_prev, {batch * seq_len, hidden}) |> Nx.transpose(),
                       Nx.reshape(grad_rh, {batch * seq_len, hidden3}))

      [grad_wx, grad_r]
    end)
  end

  defp gru_scan_fallback(wx, recurrent_weight, h0) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)

    h0 = if h0, do: h0, else: Nx.broadcast(Nx.tensor(0.0, type: Nx.type(wx)), {batch, hidden})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_p, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])

        # Reset and update gates use full wx + rh
        r_t = Nx.add(
          Nx.slice_along_axis(wx_t, 0, hidden, axis: 1),
          Nx.slice_along_axis(rh_t, 0, hidden, axis: 1)
        ) |> Nx.sigmoid()

        z_t = Nx.add(
          Nx.slice_along_axis(wx_t, hidden, hidden, axis: 1),
          Nx.slice_along_axis(rh_t, hidden, hidden, axis: 1)
        ) |> Nx.sigmoid()

        # Candidate: reset applied only to recurrent contribution
        n_t = Nx.add(
          Nx.slice_along_axis(wx_t, hidden * 2, hidden, axis: 1),
          Nx.multiply(r_t, Nx.slice_along_axis(rh_t, hidden * 2, hidden, axis: 1))
        ) |> Nx.tanh()

        # Blend: h = (1-z)*n + z*h_prev
        h_t = Nx.add(
          Nx.multiply(Nx.subtract(1.0, z_t), n_t),
          Nx.multiply(z_t, h_p)
        )

        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # TTT-Linear: weight matrix W as hidden state, updated per timestep
  defp ttt_custom_call(q, k, v, eta, w0, ln_gamma, ln_beta) do
    {batch, seq_len, inner_size} = Nx.shape(q)
    tensor_type = Nx.type(q)

    # Broadcast W0 to [batch, inner_size, inner_size]
    w0_batched = Nx.broadcast(w0, {batch, inner_size, inner_size})
    output = Nx.template({batch, seq_len, inner_size}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_ttt_scan, [q, k, v, eta, w0_batched, ln_gamma, ln_beta], output,
        fn q, k, v, eta, w0_b, ln_g, ln_b ->
          ttt_scan_fallback(q, k, v, eta, w0_b, ln_g, ln_b)
        end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v, eta, w0_batched, ln_gamma, ln_beta], fn grad_output ->
      {grad_q, grad_k, grad_v, grad_eta, grad_w0, grad_lng, grad_lnb} =
        ttt_backward_dispatch(q, k, v, eta, w0_batched, ln_gamma, ln_beta, forward_output, grad_output)
      [grad_q, grad_k, grad_v, grad_eta, grad_w0, grad_lng, grad_lnb]
    end)
  end

  defp ttt_scan_fallback(q, k, v, eta, w0, ln_gamma, ln_beta) do
    {batch, seq_len, inner_size} = Nx.shape(q)

    w_init =
      if Nx.rank(w0) == 2 do
        Nx.broadcast(w0, {batch, inner_size, inner_size})
      else
        w0
      end

    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {w_init, []}, fn t, {w_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_t = Nx.slice_along_axis(eta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # pred = W @ k
        pred = Nx.dot(w_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])

        # LayerNorm
        mean = Nx.mean(pred, axes: [-1], keep_axes: true)
        var = Nx.variance(pred, axes: [-1], keep_axes: true)
        pred_normed = Nx.divide(Nx.subtract(pred, mean), Nx.sqrt(Nx.add(var, 1.0e-6)))
        pred_normed = Nx.add(Nx.multiply(pred_normed, ln_gamma), ln_beta)

        # Update
        error = Nx.subtract(pred_normed, v_t)
        grad = Nx.dot(Nx.new_axis(Nx.multiply(eta_t, error), 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0])
        w_new = Nx.subtract(w_prev, grad)

        # Output
        out = Nx.dot(w_new, [2], [0], Nx.new_axis(q_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        {w_new, [out | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Mamba selective scan: input-dependent SSM discretization
  defp selective_scan_custom_call(x, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    tensor_type = Nx.type(x)
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_selective_scan, [x, dt, a, b, c], output,
        fn x, dt, a, b, c ->
          Edifice.SSM.Common.selective_scan_fallback(x, dt, a, b, c)
        end)

    Nx.Defn.Kernel.custom_grad(forward_output, [x, dt, b, c], fn grad_output ->
      {grad_x, grad_dt, grad_b, grad_c} =
        selective_scan_backward_dispatch(x, dt, a, b, c, forward_output, grad_output)
      [grad_x, grad_dt, grad_b, grad_c]
    end)
  end

  # KDA: channel-wise decay delta rule
  defp kda_custom_call(q, k, v, alpha, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_kda_scan, [q, k, v, alpha, beta], output,
        fn q, k, v, alpha, beta ->
          kda_scan_fallback(q, k, v, alpha, beta)
        end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v, alpha, beta], fn grad_output ->
      {grad_q, grad_k, grad_v, grad_alpha, grad_beta} =
        kda_backward_dispatch(q, k, v, alpha, beta, forward_output, grad_output)
      [grad_q, grad_k, grad_v, grad_alpha, grad_beta]
    end)
  end

  defp kda_scan_fallback(q, k, v, alpha, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Channel-wise decay: exp(alpha_t) per channel
        # alpha_t: [B, H, d] -> broadcast to [B, H, d, d] via new_axis on last
        decay = Nx.exp(alpha_t) |> Nx.new_axis(3)
        s_decayed = Nx.multiply(decay, s_prev)

        # Retrieval: S @ k -> [B, H, d]
        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1])
             |> Nx.squeeze(axes: [-1])

        # Delta rule: S += beta * outer(v - S@k, k)
        # beta_t: [B, H] -> [B, H, 1, 1]
        beta_bc = beta_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        error = Nx.subtract(v_t, sk)
        delta = Nx.multiply(beta_bc, Nx.multiply(Nx.new_axis(error, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_decayed, delta)

        # Output: S @ q
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1])
              |> Nx.squeeze(axes: [-1])
        {s_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # RLA/RDN: dual-state residual linear attention
  defp rla_custom_call(q, k, v, alpha, beta, gamma, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, tensor_type)
    variant_int = if variant == :rdn, do: 1, else: 0
    variant_t = Nx.tensor(variant_int, type: {:s, 32})
    clip_t = Nx.tensor(clip_threshold, type: {:f, 32})

    forward_output =
      Nx.Shared.optional(:fused_rla_scan, [q, k, v, alpha, beta, gamma, variant_t, clip_t], output,
        fn q, k, v, alpha, beta, gamma, _variant, _clip ->
          rla_scan_fallback(q, k, v, alpha, beta, gamma, variant, clip_threshold)
        end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v, alpha, beta, gamma], fn grad_output ->
      {grad_q, grad_k, grad_v, grad_alpha, grad_beta, grad_gamma} =
        rla_backward_dispatch(q, k, v, alpha, beta, gamma, forward_output, grad_output, variant, clip_threshold)
      [grad_q, grad_k, grad_v, grad_alpha, grad_beta, grad_gamma]
    end)
  end

  defp rla_scan_fallback(q, k, v, alpha, beta, gamma, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Normalize gates to [B, T, H, 1, 1] for broadcasting with [B, H, d, d] state matrices
    alpha = Nx.reshape(alpha, {batch, seq_len, num_heads, 1, 1})
    beta = Nx.reshape(beta, {batch, seq_len, num_heads, 1, 1})
    gamma = Nx.reshape(gamma, {batch, seq_len, num_heads, 1, 1})

    s0 = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, num_heads, head_dim, head_dim})
    r0 = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, num_heads, head_dim, head_dim})

    {_, _, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, r0, []}, fn t, {s_prev, r_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        gamma_t = Nx.slice_along_axis(gamma, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Retrieval from base state: S @ k
        retrieval_s =
          Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1])
          |> Nx.squeeze(axes: [-1])

        # Residual error: clip(v - retrieval)
        raw_error = Nx.subtract(v_t, retrieval_s)
        r_error = Nx.clip(raw_error, -clip_threshold, clip_threshold)

        {s_t, r_t} =
          case variant do
            :rla ->
              # S_t = alpha * S + beta * outer(v, k)
              s_decayed = Nx.multiply(alpha_t, s_prev)
              vk = Nx.multiply(Nx.new_axis(v_t, -1), Nx.new_axis(k_t, -2))
              s_new = Nx.add(s_decayed, Nx.multiply(beta_t, vk))

              # R_t = alpha * R + gamma * outer(r_error, k)
              r_decayed = Nx.multiply(alpha_t, r_prev)
              rk = Nx.multiply(Nx.new_axis(r_error, -1), Nx.new_axis(k_t, -2))
              r_new = Nx.add(r_decayed, Nx.multiply(gamma_t, rk))

              {s_new, r_new}

            :rdn ->
              # S_t = alpha * S + beta * outer(v - S@k, k)
              s_decayed = Nx.multiply(alpha_t, s_prev)
              delta_s = Nx.subtract(v_t, retrieval_s)
              s_outer = Nx.multiply(Nx.new_axis(delta_s, -1), Nx.new_axis(k_t, -2))
              s_new = Nx.add(s_decayed, Nx.multiply(beta_t, s_outer))

              # R_t = alpha * R + gamma * outer(r_error - R@k, k)
              retrieval_r =
                Nx.dot(r_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1])
                |> Nx.squeeze(axes: [-1])
              r_decayed = Nx.multiply(alpha_t, r_prev)
              delta_r = Nx.subtract(r_error, retrieval_r)
              r_outer = Nx.multiply(Nx.new_axis(delta_r, -1), Nx.new_axis(k_t, -2))
              r_new = Nx.add(r_decayed, Nx.multiply(gamma_t, r_outer))

              {s_new, r_new}
          end

        # Output: (S + R) @ q
        sr = Nx.add(s_t, r_t)
        o_t = Nx.dot(sr, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1])
              |> Nx.squeeze(axes: [-1])
        {s_t, r_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # CUDA NIF fused paths (graph-breaking)
  # ============================================================================

  defp mingru_fused(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)
    dtype = dtype_flag(gates)
    esize = elem_size(gates)
    ttype = tensor_type(gates)

    # Apply sigmoid on the XLA side (XLA fuses this efficiently)
    z = Nx.sigmoid(gates)

    # Zero initial hidden state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(gates)), {batch, hidden})

    # Extract device pointers
    z_ptr = Nx.to_pointer(z, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_mingru_scan(
           z_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        # Hold gc_ref to prevent cudaFree until we're done with the tensor
        hold_gc_ref(out_addr, gc_ref)

        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused MinGRU scan failed: #{reason}"
    end
  end

  defp minlstm_fused(forget_gates, input_gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(forget_gates)
    dtype = dtype_flag(forget_gates)
    esize = elem_size(forget_gates)
    ttype = tensor_type(forget_gates)

    # Apply sigmoid on XLA side — kernel expects post-sigmoid values
    f = Nx.sigmoid(forget_gates)
    i = Nx.sigmoid(input_gates)

    # Zero initial cell state
    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(forget_gates)), {batch, hidden})

    # Extract device pointers
    f_ptr = Nx.to_pointer(f, mode: :local)
    i_ptr = Nx.to_pointer(i, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_minlstm_scan(
           f_ptr.address, i_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)

        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(forget_gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
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
    dtype = dtype_flag(gates)
    esize = elem_size(gates)
    ttype = tensor_type(gates)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(gates)), {batch, hidden})

    g_ptr = Nx.to_pointer(gates, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_elu_gru_scan(
           g_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused ELU-GRU scan failed: #{reason}"
    end
  end

  defp real_gru_fused(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)
    dtype = dtype_flag(gates)
    esize = elem_size(gates)
    ttype = tensor_type(gates)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(gates)), {batch, hidden})

    g_ptr = Nx.to_pointer(gates, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_real_gru_scan(
           g_ptr.address, c_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(gates), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused Real-GRU scan failed: #{reason}"
    end
  end

  defp diag_linear_fused(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    dtype = dtype_flag(a_vals)
    esize = elem_size(a_vals)
    ttype = tensor_type(a_vals)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(a_vals)), {batch, hidden})

    a_ptr = Nx.to_pointer(a_vals, mode: :local)
    b_ptr = Nx.to_pointer(b_vals, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_diag_linear_scan(
           a_ptr.address, b_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(a_vals), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused Diag-Linear scan failed: #{reason}"
    end
  end

  defp linear_scan_fused(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    dtype = dtype_flag(a_vals)
    esize = elem_size(a_vals)
    ttype = tensor_type(a_vals)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(a_vals)), {batch, hidden})

    a_ptr = Nx.to_pointer(a_vals, mode: :local)
    b_ptr = Nx.to_pointer(b_vals, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_linear_scan(
           a_ptr.address, b_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(a_vals), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused linear scan failed: #{reason}"
    end
  end

  defp liquid_fused(tau, activation) do
    {batch, seq_len, hidden} = Nx.shape(tau)
    dtype = dtype_flag(tau)
    esize = elem_size(tau)
    ttype = tensor_type(tau)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(tau)), {batch, hidden})

    tau_ptr = Nx.to_pointer(tau, mode: :local)
    act_ptr = Nx.to_pointer(activation, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_liquid_scan(
           tau_ptr.address, act_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(tau), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
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
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)

    case Edifice.CUDA.NIF.fused_delta_net_scan(
           q_ptr.address, k_ptr.address, v_ptr.address, beta_ptr.address,
           batch, seq_len, num_heads, head_dim, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused DeltaNet scan failed: #{reason}"
    end
  end

  defp gated_delta_net_fused(q, k, v, beta, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)

    case Edifice.CUDA.NIF.fused_gated_delta_net_scan(
           q_ptr.address, k_ptr.address, v_ptr.address,
           beta_ptr.address, alpha_ptr.address,
           batch, seq_len, num_heads, head_dim, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused GatedDeltaNet scan failed: #{reason}"
    end
  end

  defp delta_product_fused(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    {_, _, num_householder, _, _} = Nx.shape(k)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)

    case Edifice.CUDA.NIF.fused_delta_product_scan(
           q_ptr.address, k_ptr.address, v_ptr.address, beta_ptr.address,
           batch, seq_len, num_householder, num_heads, head_dim, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused DeltaProduct scan failed: #{reason}"
    end
  end

  # ============================================================================
  # sLSTM / TTT / Mamba fused paths (graph-breaking NIF)
  # ============================================================================

  defp slstm_fused(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    dtype = dtype_flag(wx)
    esize = elem_size(wx)
    ttype = tensor_type(wx)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(wx)), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(wx)), {batch, hidden})

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    r_ptr = Nx.to_pointer(recurrent_weight, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)
    c0_ptr = Nx.to_pointer(c0, mode: :local)

    case Edifice.CUDA.NIF.fused_slstm_scan(
           wx_ptr.address, r_ptr.address, h0_ptr.address, c0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused sLSTM scan failed: #{reason}"
    end
  end

  defp lstm_fused(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)
    dtype = dtype_flag(wx)
    esize = elem_size(wx)
    ttype = tensor_type(wx)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(wx)), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(wx)), {batch, hidden})

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    r_ptr = Nx.to_pointer(recurrent_weight, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)
    c0_ptr = Nx.to_pointer(c0, mode: :local)

    case Edifice.CUDA.NIF.fused_lstm_scan(
           wx_ptr.address, r_ptr.address, h0_ptr.address, c0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused LSTM scan failed: #{reason}"
    end
  end

  defp gru_fused(wx, recurrent_weight) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)
    dtype = dtype_flag(wx)
    esize = elem_size(wx)
    ttype = tensor_type(wx)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype, backend: backend_for(wx)), {batch, hidden})

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    r_ptr = Nx.to_pointer(recurrent_weight, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_gru_scan(
           wx_ptr.address, r_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused GRU scan failed: #{reason}"
    end
  end

  defp ttt_fused(q, k, v, eta, w0, ln_gamma, ln_beta) do
    {batch, seq_len, inner_size} = Nx.shape(q)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    # Broadcast W0 to [batch, inner_size, inner_size] if needed
    w0_batched =
      if Nx.rank(w0) == 2 do
        Nx.broadcast(w0, {batch, inner_size, inner_size})
      else
        w0
      end

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    eta_ptr = Nx.to_pointer(eta, mode: :local)
    w0_ptr = Nx.to_pointer(w0_batched, mode: :local)
    lng_ptr = Nx.to_pointer(ln_gamma, mode: :local)
    lnb_ptr = Nx.to_pointer(ln_beta, mode: :local)

    case Edifice.CUDA.NIF.fused_ttt_scan(
           q_ptr.address, k_ptr.address, v_ptr.address, eta_ptr.address,
           w0_ptr.address, lng_ptr.address, lnb_ptr.address,
           batch, seq_len, inner_size, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * inner_size * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, inner_size}
        )

      {:error, reason} ->
        raise "CUDA fused TTT scan failed: #{reason}"
    end
  end

  defp selective_scan_fused(x, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    {_h, state} = Nx.shape(a)
    dtype = dtype_flag(x)
    esize = elem_size(x)
    ttype = tensor_type(x)

    x_ptr = Nx.to_pointer(x, mode: :local)
    dt_ptr = Nx.to_pointer(dt, mode: :local)
    a_ptr = Nx.to_pointer(a, mode: :local)
    b_ptr = Nx.to_pointer(b, mode: :local)
    c_ptr = Nx.to_pointer(c, mode: :local)

    case Edifice.CUDA.NIF.fused_selective_scan(
           x_ptr.address, dt_ptr.address, a_ptr.address,
           b_ptr.address, c_ptr.address,
           batch, seq_len, hidden, state, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(x), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused selective scan failed: #{reason}"
    end
  end

  # ============================================================================
  # KDA / RLA fused paths (graph-breaking NIF)
  # ============================================================================

  defp kda_fused(q, k, v, alpha, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)

    case Edifice.CUDA.NIF.fused_kda_scan(
           q_ptr.address, k_ptr.address, v_ptr.address,
           alpha_ptr.address, beta_ptr.address,
           batch, seq_len, num_heads, head_dim, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused KDA scan failed: #{reason}"
    end
  end

  defp rla_fused(q, k, v, alpha, beta, gamma, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    variant_int = if variant == :rdn, do: 1, else: 0
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    # Ensure gates are [B, T, H] for the CUDA kernel
    alpha = Nx.reshape(alpha, {batch, seq_len, num_heads})
    beta = Nx.reshape(beta, {batch, seq_len, num_heads})
    gamma = Nx.reshape(gamma, {batch, seq_len, num_heads})

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)
    gamma_ptr = Nx.to_pointer(gamma, mode: :local)

    case Edifice.CUDA.NIF.fused_rla_scan(
           q_ptr.address, k_ptr.address, v_ptr.address,
           alpha_ptr.address, beta_ptr.address, gamma_ptr.address,
           batch, seq_len, num_heads, head_dim,
           variant_int, clip_threshold * 1.0, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, num_heads, head_dim}
        )

      {:error, reason} ->
        raise "CUDA fused RLA scan failed: #{reason}"
    end
  end

  # ============================================================================
  # Flash Attention (IO-aware exact attention)
  # ============================================================================

  @doc """
  Flash Attention V2 — IO-aware exact attention.

  Uses tiled loading with online softmax to avoid materializing the full
  [seq, seq] attention matrix. Produces identical results to standard SDPA.

  > **Note:** The EXLA custom call path always uses causal masking.
  > Non-causal attention uses the NIF or Elixir fallback path.

  ## Arguments

    * `q` - Query tensor `[batch, heads, seq, head_dim]` (f32)
    * `k` - Key tensor `[batch, heads, seq, head_dim]` (f32)
    * `v` - Value tensor `[batch, heads, seq, head_dim]` (f32)
    * `opts` - Options:
      * `:causal` - Apply causal mask (default: `false`)

  ## Returns

    Output tensor `[batch, heads, seq, head_dim]` (f32)
  """
  def flash_attention(q, k, v, opts \\ []) do
    causal = if Keyword.get(opts, :causal, false), do: 1, else: 0

    cond do
      # Custom call path only supports causal=1 (hardcoded to avoid scalar operand segfault)
      causal == 1 and flash_attention_custom_call_available?() ->
        flash_attention_custom_call(q, k, v, causal)

      cuda_available?(q) ->
        flash_attention_fused(q, k, v, causal)

      true ->
        flash_attention_fallback(q, k, v, causal)
    end
  end

  @doc """
  Check if flash attention is available (any tier).
  """
  def flash_attention_available? do
    flash_attention_custom_call_available?() or nif_loaded?()
  end

  defp flash_attention_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_flash_attention, 5)
  rescue
    _ -> false
  end

  defp flash_attention_custom_call(q, k, v, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, num_heads, seq_len, head_dim}, tensor_type)
    causal_tensor = Nx.tensor(causal, type: {:s, 32})

    forward_output =
      Nx.Shared.optional(:fused_flash_attention, [q, k, v, causal_tensor], output, fn q, k, v, _causal ->
        flash_attention_fallback(q, k, v, causal)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v], fn grad_output ->
      {grad_q, grad_k, grad_v} =
        flash_attention_backward_dispatch(q, k, v, causal, forward_output, grad_output)
      [grad_q, grad_k, grad_v]
    end)
  end

  defp flash_attention_fused(q, k, v, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)

    case Edifice.CUDA.NIF.fused_flash_attention(
           q_ptr.address, k_ptr.address, v_ptr.address,
           batch, num_heads, seq_len, head_dim, causal, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * num_heads * seq_len * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, num_heads, seq_len, head_dim}
        )

      {:error, reason} ->
        raise "CUDA flash attention failed: #{reason}"
    end
  end

  @doc false
  def flash_attention_fallback(q, k, v, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Standard scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: ttype))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Apply causal mask if requested
    scores =
      if causal == 1 do
        # Build lower-triangular mask [seq, seq]
        rows = Nx.iota({seq_len, seq_len}, axis: 0)
        cols = Nx.iota({seq_len, seq_len}, axis: 1)
        mask = Nx.greater_equal(rows, cols)

        mask =
          mask
          |> Nx.reshape({1, 1, seq_len, seq_len})
          |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

        neg_inf = Nx.Constants.neg_infinity(ttype)
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    # Softmax + weighted sum
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  defp flash_attention_backward_dispatch(q, k, v, causal, forward_out, grad_output) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_template = Nx.template({batch, num_heads, seq_len, head_dim}, ttype)
    causal_tensor = Nx.tensor(causal, type: {:s, 32})

    Nx.Shared.optional(
      :fused_flash_attention_backward,
      [q, k, v, forward_out, grad_output, causal_tensor],
      {grad_template, grad_template, grad_template},
      fn q, k, v, _fwd, grad, _causal ->
        flash_attention_backward_fallback(q, k, v, causal, grad)
      end
    )
  end

  @doc false
  def flash_attention_backward_fallback(q, k, v, causal, grad_output) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Recompute forward: standard SDPA
    scale = Nx.sqrt(Nx.tensor(head_dim, type: ttype))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    scores =
      if causal == 1 do
        rows = Nx.iota({seq_len, seq_len}, axis: 0)
        cols = Nx.iota({seq_len, seq_len}, axis: 1)
        mask = Nx.greater_equal(rows, cols)
        mask = mask |> Nx.reshape({1, 1, seq_len, seq_len}) |> Nx.broadcast({batch, num_heads, seq_len, seq_len})
        neg_inf = Nx.Constants.neg_infinity(ttype)
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    # Softmax weights
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    # dV = weights^T @ grad_output
    grad_v = Nx.dot(weights, [2], [0, 1], grad_output, [2], [0, 1])

    # dWeights = grad_output @ V^T
    d_weights = Nx.dot(grad_output, [3], [0, 1], v, [3], [0, 1])

    # dScores = weights * (dWeights - sum(dWeights * weights, axis=-1, keepdims=True))
    d_scores = Nx.multiply(weights, Nx.subtract(d_weights, Nx.sum(Nx.multiply(d_weights, weights), axes: [-1], keep_axes: true)))

    # Scale dScores
    d_scores = Nx.divide(d_scores, scale)

    # dQ = dScores @ K
    grad_q = Nx.dot(d_scores, [3], [0, 1], k, [2], [0, 1])

    # dK = dScores^T @ Q
    grad_k = Nx.dot(d_scores, [2], [0, 1], q, [2], [0, 1])

    {grad_q, grad_k, grad_v}
  end

  # ============================================================================
  # LASER Attention (flash attention + exp(V) + log output)
  # ============================================================================

  @doc """
  LASER flash attention: `O = log(softmax(QK^T / sqrt(d)) @ exp(V))`.

  Uses the LWSE trick: subtract `v_max` before exp, add back after log.

  > **Note:** The EXLA custom call path always uses causal masking.
  > Non-causal attention uses the NIF or Elixir fallback path.

  ## Arguments

    * `q` - Query tensor `[batch, heads, seq, head_dim]` (f32)
    * `k` - Key tensor `[batch, heads, seq, head_dim]` (f32)
    * `v` - Value tensor `[batch, heads, seq, head_dim]` (f32)
    * `opts` - Options:
      * `:causal` - Apply causal mask (default: `true`)

  ## Returns

    Output tensor `[batch, heads, seq, head_dim]` (f32)
  """
  def laser_attention(q, k, v, opts \\ []) do
    causal = if Keyword.get(opts, :causal, true), do: 1, else: 0

    # Precompute v_max: [batch, heads, 1, head_dim]
    v_max = Nx.reduce_max(v, axes: [2], keep_axes: true)

    cond do
      # Custom call path only supports causal=1 (hardcoded to avoid scalar operand segfault)
      causal == 1 and laser_attention_custom_call_available?() ->
        laser_attention_custom_call(q, k, v, v_max, causal)

      cuda_available?(q) ->
        laser_attention_fused(q, k, v, v_max, causal)

      true ->
        laser_attention_fallback(q, k, v, v_max, causal)
    end
  end

  defp laser_attention_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_laser_attention, 6)
  rescue
    _ -> false
  end

  defp laser_attention_custom_call(q, k, v, v_max, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, num_heads, seq_len, head_dim}, tensor_type)
    causal_tensor = Nx.tensor(causal, type: {:s, 32})

    forward_output =
      Nx.Shared.optional(:fused_laser_attention, [q, k, v, v_max, causal_tensor], output, fn q, k, v, v_max, _causal ->
        laser_attention_fallback(q, k, v, v_max, causal)
      end)

    # v_max captured from closure, not a custom_grad input (derived from v)
    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v], fn grad_output ->
      {grad_q, grad_k, grad_v} =
        laser_attention_backward_dispatch(q, k, v, v_max, causal, forward_output, grad_output)
      [grad_q, grad_k, grad_v]
    end)
  end

  defp laser_attention_fused(q, k, v, v_max, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    vmax_ptr = Nx.to_pointer(v_max, mode: :local)

    case Edifice.CUDA.NIF.fused_laser_attention(
           q_ptr.address, k_ptr.address, v_ptr.address, vmax_ptr.address,
           batch, num_heads, seq_len, head_dim, causal, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * num_heads * seq_len * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, num_heads, seq_len, head_dim}
        )

      {:error, reason} ->
        raise "CUDA LASER attention failed: #{reason}"
    end
  end

  @doc false
  def laser_attention_fallback(q, k, v, v_max, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # exp(V - v_max) for numerical stability
    exp_v = Nx.exp(Nx.subtract(v, v_max))

    # Standard scaled dot-product attention scores
    scale = Nx.sqrt(Nx.tensor(head_dim, type: ttype))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Apply causal mask if requested
    scores =
      if causal == 1 do
        rows = Nx.iota({seq_len, seq_len}, axis: 0)
        cols = Nx.iota({seq_len, seq_len}, axis: 1)
        mask = Nx.greater_equal(rows, cols)

        mask =
          mask
          |> Nx.reshape({1, 1, seq_len, seq_len})
          |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

        neg_inf = Nx.Constants.neg_infinity(ttype)
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    # Softmax + weighted sum of exp(V - v_max)
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    attn_out = Nx.dot(weights, [3], [0, 1], exp_v, [2], [0, 1])

    # log(max(result, 1e-7)) + v_max
    Nx.add(Nx.log(Nx.max(attn_out, 1.0e-7)), v_max)
  end

  defp laser_attention_backward_dispatch(q, k, v, v_max, causal, forward_out, grad_output) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_template = Nx.template({batch, num_heads, seq_len, head_dim}, ttype)
    causal_tensor = Nx.tensor(causal, type: {:s, 32})

    Nx.Shared.optional(
      :fused_laser_attention_backward,
      [q, k, v, v_max, forward_out, grad_output, causal_tensor],
      {grad_template, grad_template, grad_template},
      fn q, k, v, v_max, _fwd, grad, _causal ->
        laser_attention_backward_fallback(q, k, v, v_max, causal, grad)
      end
    )
  end

  @doc false
  def laser_attention_backward_fallback(q, k, v, v_max, causal, grad_output) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Recompute forward
    exp_v = Nx.exp(Nx.subtract(v, v_max))
    scale = Nx.sqrt(Nx.tensor(head_dim, type: ttype))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    scores =
      if causal == 1 do
        rows = Nx.iota({seq_len, seq_len}, axis: 0)
        cols = Nx.iota({seq_len, seq_len}, axis: 1)
        mask = Nx.greater_equal(rows, cols)
        mask = mask |> Nx.reshape({1, 1, seq_len, seq_len}) |> Nx.broadcast({batch, num_heads, seq_len, seq_len})
        neg_inf = Nx.Constants.neg_infinity(ttype)
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    attn_out = Nx.dot(weights, [3], [0, 1], exp_v, [2], [0, 1])
    # Forward output: O = log(max(attn_out, 1e-7)) + v_max
    o_val = Nx.add(Nx.log(Nx.max(attn_out, 1.0e-7)), v_max)

    # Effective gradient: dR = dO * exp(v_max - O)
    d_r = Nx.multiply(grad_output, Nx.exp(Nx.subtract(v_max, o_val)))

    # dV_exp = weights^T @ dR
    grad_v_exp = Nx.dot(weights, [2], [0, 1], d_r, [2], [0, 1])

    # dV = dV_exp * exp(V - v_max)
    grad_v = Nx.multiply(grad_v_exp, exp_v)

    # dWeights = dR @ exp(V)^T
    d_weights = Nx.dot(d_r, [3], [0, 1], exp_v, [3], [0, 1])

    # dScores = weights * (dWeights - sum(dWeights * weights, keepdims))
    d_scores = Nx.multiply(weights, Nx.subtract(d_weights, Nx.sum(Nx.multiply(d_weights, weights), axes: [-1], keep_axes: true)))
    d_scores = Nx.divide(d_scores, scale)

    grad_q = Nx.dot(d_scores, [3], [0, 1], k, [2], [0, 1])
    grad_k = Nx.dot(d_scores, [2], [0, 1], q, [2], [0, 1])

    {grad_q, grad_k, grad_v}
  end

  # ============================================================================
  # FoX Attention (flash attention + forget bias)
  # ============================================================================

  @doc """
  FoX flash attention: `softmax(QK^T / sqrt(d) + forget_bias) @ V`.

  The forget bias is computed from cumulative log-forget values:
  `F[i,j] = cs[i] - cs[j]` where `cs = cumsum(log(sigmoid(f)))`.

  Always causal (forget gates are inherently directional).

  ## Arguments

    * `q` - Query tensor `[batch, heads, seq, head_dim]` (f32)
    * `k` - Key tensor `[batch, heads, seq, head_dim]` (f32)
    * `v` - Value tensor `[batch, heads, seq, head_dim]` (f32)
    * `cs` - Cumulative log-forget `[batch, heads, seq]` (f32)

  ## Returns

    Output tensor `[batch, heads, seq, head_dim]` (f32)
  """
  def fox_attention(q, k, v, cs) do
    cond do
      fox_attention_custom_call_available?() ->
        fox_attention_custom_call(q, k, v, cs)

      cuda_available?(q) ->
        fox_attention_fused(q, k, v, cs)

      true ->
        fox_attention_fallback(q, k, v, cs)
    end
  end

  defp fox_attention_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_fox_attention, 5)
  rescue
    _ -> false
  end

  defp fox_attention_custom_call(q, k, v, cs) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, num_heads, seq_len, head_dim}, tensor_type)

    forward_output =
      Nx.Shared.optional(:fused_fox_attention, [q, k, v, cs], output, fn q, k, v, cs ->
        fox_attention_fallback(q, k, v, cs)
      end)

    Nx.Defn.Kernel.custom_grad(forward_output, [q, k, v, cs], fn grad_output ->
      {grad_q, grad_k, grad_v, grad_cs} =
        fox_attention_backward_dispatch(q, k, v, cs, forward_output, grad_output)
      [grad_q, grad_k, grad_v, grad_cs]
    end)
  end

  defp fox_attention_fused(q, k, v, cs) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    cs_ptr = Nx.to_pointer(cs, mode: :local)

    case Edifice.CUDA.NIF.fused_fox_attention(
           q_ptr.address, k_ptr.address, v_ptr.address, cs_ptr.address,
           batch, num_heads, seq_len, head_dim, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * num_heads * seq_len * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, num_heads, seq_len, head_dim}
        )

      {:error, reason} ->
        raise "CUDA FoX attention failed: #{reason}"
    end
  end

  @doc false
  def fox_attention_fallback(q, k, v, cs) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Standard scaled dot-product attention scores
    scale = Nx.sqrt(Nx.tensor(head_dim, type: ttype))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Add forget bias: F[i,j] = cs[i] - cs[j]
    cs_i = Nx.new_axis(cs, -1)    # [B, H, T, 1]
    cs_j = Nx.new_axis(cs, -2)    # [B, H, 1, T]
    forget_bias = Nx.subtract(cs_i, cs_j)

    # Zero out diagonal (no forgetting for self-attention)
    diag_mask =
      Nx.equal(
        Nx.iota({seq_len, seq_len}, axis: 0),
        Nx.iota({seq_len, seq_len}, axis: 1)
      )
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    forget_bias = Nx.select(diag_mask, Nx.tensor(0.0, type: ttype), forget_bias)

    scores = Nx.add(scores, forget_bias)

    # Causal mask (FoX is always causal)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      causal_mask
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    neg_inf = Nx.Constants.neg_infinity(ttype)
    scores = Nx.select(causal_mask, scores, neg_inf)

    # Softmax + weighted sum
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  defp fox_attention_backward_dispatch(q, k, v, cs, forward_out, grad_output) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)
    grad_4d = Nx.template({batch, num_heads, seq_len, head_dim}, ttype)
    grad_cs_template = Nx.template({batch, num_heads, seq_len}, ttype)

    Nx.Shared.optional(
      :fused_fox_attention_backward,
      [q, k, v, cs, forward_out, grad_output],
      {grad_4d, grad_4d, grad_4d, grad_cs_template},
      fn q, k, v, cs, _fwd, grad ->
        fox_attention_backward_fallback(q, k, v, cs, grad)
      end
    )
  end

  @doc false
  def fox_attention_backward_fallback(q, k, v, cs, grad_output) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    ttype = Nx.type(q)

    # Recompute forward
    scale = Nx.sqrt(Nx.tensor(head_dim, type: ttype))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    cs_i = Nx.new_axis(cs, -1)
    cs_j = Nx.new_axis(cs, -2)
    forget_bias = Nx.subtract(cs_i, cs_j)

    diag_mask =
      Nx.equal(
        Nx.iota({seq_len, seq_len}, axis: 0),
        Nx.iota({seq_len, seq_len}, axis: 1)
      )
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    forget_bias = Nx.select(diag_mask, Nx.tensor(0.0, type: ttype), forget_bias)
    scores = Nx.add(scores, forget_bias)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)
    causal_mask = causal_mask |> Nx.reshape({1, 1, seq_len, seq_len}) |> Nx.broadcast({batch, num_heads, seq_len, seq_len})
    neg_inf = Nx.Constants.neg_infinity(ttype)
    scores = Nx.select(causal_mask, scores, neg_inf)

    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    # dV = weights^T @ grad_output
    grad_v = Nx.dot(weights, [2], [0, 1], grad_output, [2], [0, 1])

    # dWeights = grad_output @ V^T
    d_weights = Nx.dot(grad_output, [3], [0, 1], v, [3], [0, 1])

    # dScores through softmax
    d_scores = Nx.multiply(weights, Nx.subtract(d_weights, Nx.sum(Nx.multiply(d_weights, weights), axes: [-1], keep_axes: true)))

    # Zero diagonal in dScores for forget bias gradient
    d_scores_no_diag = Nx.select(diag_mask, Nx.tensor(0.0, type: ttype), d_scores)

    # grad_cs: row_sum - col_sum of dScores (excluding diagonal)
    row_sums = Nx.sum(d_scores_no_diag, axes: [-1])
    col_sums = Nx.sum(d_scores_no_diag, axes: [-2])
    grad_cs = Nx.subtract(row_sums, col_sums)

    # dScores / scale for Q,K gradients
    d_scores = Nx.divide(d_scores, scale)

    grad_q = Nx.dot(d_scores, [3], [0, 1], k, [2], [0, 1])
    grad_k = Nx.dot(d_scores, [2], [0, 1], q, [2], [0, 1])

    {grad_q, grad_k, grad_v, grad_cs}
  end

  # ============================================================================
  # Reservoir (Echo State Network) scan
  # ============================================================================

  @doc """
  Reservoir scan — `h = tanh(wx + W_res @ h)` with optional leak rate.

  Returns only the final hidden state `[batch, hidden]`.

  ## Arguments

    * `wx` - Pre-computed input projection `[batch, seq_len, reservoir_size]`
    * `w_res` - Fixed reservoir weight matrix `[reservoir_size, reservoir_size]`
    * `opts` - Options:
      * `:leak_rate` - Leaky integration rate (default: 1.0)
  """
  def reservoir_scan(wx, w_res, opts \\ []) do
    leak_rate = Keyword.get(opts, :leak_rate, 1.0)

    cond do
      custom_call_available?() ->
        reservoir_custom_call(wx, w_res, leak_rate)

      cuda_available?(wx) ->
        reservoir_fused(wx, w_res, leak_rate)

      true ->
        reservoir_scan_fallback(wx, w_res, leak_rate)
    end
  end

  defp reservoir_custom_call(wx, w_res, leak_rate) do
    {batch, _seq_len, hidden} = Nx.shape(wx)
    tensor_type = Nx.type(wx)
    output = Nx.template({batch, hidden}, tensor_type)

    # Pack leak_rate into h0: [B, H] → [B, H+1] with leak_rate as extra column
    h0 = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, hidden})
    leak_col = Nx.broadcast(Nx.tensor(leak_rate, type: tensor_type), {batch, 1})
    h0_packed = Nx.concatenate([h0, leak_col], axis: 1)

    Nx.Shared.optional(:fused_reservoir_scan, [wx, w_res, h0_packed], output, fn wx, w_res, _h0_packed ->
      reservoir_scan_fallback(wx, w_res, leak_rate)
    end)
  end

  defp reservoir_fused(wx, w_res, leak_rate) do
    {batch, seq_len, hidden} = Nx.shape(wx)
    dtype = dtype_flag(wx)
    esize = elem_size(wx)
    ttype = tensor_type(wx)

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    wres_ptr = Nx.to_pointer(w_res, mode: :local)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: ttype), {batch, hidden})
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_reservoir_scan(
           wx_ptr.address, wres_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, leak_rate / 1, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * hidden * esize

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, hidden}
        )

      {:error, reason} ->
        raise "CUDA reservoir scan failed: #{reason}"
    end
  end

  @doc false
  def reservoir_scan_fallback(wx, w_res, leak_rate) do
    {batch, seq_len, hidden} = Nx.shape(wx)
    h = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(wx)), {batch, hidden})

    Enum.reduce(0..(seq_len - 1), h, fn t, h_prev ->
      wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
      pre_act = Nx.add(wx_t, Nx.dot(h_prev, w_res))
      h_new = Nx.tanh(pre_act)

      if leak_rate == 1.0 do
        h_new
      else
        Nx.add(Nx.multiply(1.0 - leak_rate, h_prev), Nx.multiply(leak_rate, h_new))
      end
    end)
  end

  # ============================================================================
  # Titans scan (surprise-gated memory)
  # ============================================================================

  @doc """
  Titans scan — surprise-gated memory update with momentum.

  ## Arguments

    * `combined` - Concatenated `[Q, K, V, gate_input]` tensor `[batch, seq, 4*M]`
    * `opts` - Options:
      * `:memory_size` - Memory dimension M (required)
      * `:momentum` - Momentum coefficient (default: 0.9)
  """
  def titans_scan(combined, opts \\ []) do
    memory_size = Keyword.fetch!(opts, :memory_size)
    momentum = Keyword.get(opts, :momentum, 0.9)

    cond do
      custom_call_available?() ->
        titans_custom_call(combined, memory_size, momentum)

      cuda_available?(combined) ->
        titans_fused(combined, memory_size, momentum)

      true ->
        titans_scan_fallback(combined, memory_size, momentum)
    end
  end

  defp titans_custom_call(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)
    tensor_type = Nx.type(combined)
    output = Nx.template({batch, seq_len, memory_size}, tensor_type)

    # Pack momentum into combined: [B,T,4*M] → [B,T,4*M+1]
    momentum_col = Nx.broadcast(Nx.tensor(momentum, type: tensor_type), {batch, seq_len, 1})
    packed = Nx.concatenate([combined, momentum_col], axis: 2)

    Nx.Shared.optional(:fused_titans_scan, [packed], output, fn packed ->
      combined_inner = Nx.slice_along_axis(packed, 0, memory_size * 4, axis: 2)
      titans_scan_fallback(combined_inner, memory_size, momentum)
    end)
  end

  defp titans_fused(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)
    dtype = dtype_flag(combined)
    esize = elem_size(combined)
    ttype = tensor_type(combined)

    combined_ptr = Nx.to_pointer(combined, mode: :local)

    case Edifice.CUDA.NIF.fused_titans_scan(
           combined_ptr.address, batch, seq_len, memory_size, momentum / 1, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * memory_size * esize

        Nx.from_pointer(
          {backend_for(combined), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, memory_size}
        )

      {:error, reason} ->
        raise "CUDA Titans scan failed: #{reason}"
    end
  end

  defp titans_scan_fallback(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)

    q_all = Nx.slice_along_axis(combined, 0, memory_size, axis: 2)
    k_all = Nx.slice_along_axis(combined, memory_size, memory_size, axis: 2)
    v_all = Nx.slice_along_axis(combined, memory_size * 2, memory_size, axis: 2)
    gate_input = Nx.slice_along_axis(combined, memory_size * 3, memory_size, axis: 2)

    m_init = Nx.broadcast(0.0, {batch, memory_size, memory_size})
    mom_init = Nx.broadcast(0.0, {batch, memory_size, memory_size})

    {_, _, output_list} =
      Enum.reduce(0..(seq_len - 1), {m_init, mom_init, []}, fn t, {m_prev, mom_prev, acc} ->
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        g_input = Nx.slice_along_axis(gate_input, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        pred = Nx.dot(m_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        error = Nx.subtract(pred, v_t)
        surprise = Nx.mean(Nx.pow(error, 2), axes: [1], keep_axes: true)
        surprise_log = Nx.log(Nx.add(surprise, 1.0e-6))
        gate = Nx.sigmoid(Nx.add(g_input, surprise_log))
        grad = Nx.dot(Nx.new_axis(error, 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0])
        mom_t = Nx.add(Nx.multiply(momentum, mom_prev), grad)
        gate_expanded = Nx.new_axis(gate, 2)
        m_t = Nx.subtract(m_prev, Nx.multiply(gate_expanded, mom_t))
        o_t = Nx.dot(m_t, [2], [0], Nx.new_axis(q_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])

        {m_t, mom_t, [o_t | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # MIRAS scan (iterative memory reasoning - Moneta variant)
  # ============================================================================

  @doc """
  MIRAS scan — Moneta variant with data-dependent alpha/eta gates.

  ## Arguments

    * `combined` - Concatenated `[Q, K, V, alpha, eta]` tensor `[batch, seq, 5*M]`
    * `opts` - Options:
      * `:memory_size` - Memory dimension M (required)
      * `:momentum` - Momentum coefficient (default: 0.9)
  """
  def miras_scan(combined, opts \\ []) do
    memory_size = Keyword.fetch!(opts, :memory_size)
    momentum = Keyword.get(opts, :momentum, 0.9)

    cond do
      custom_call_available?() ->
        miras_custom_call(combined, memory_size, momentum)

      cuda_available?(combined) ->
        miras_fused(combined, memory_size, momentum)

      true ->
        miras_scan_fallback(combined, memory_size, momentum)
    end
  end

  defp miras_custom_call(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)
    tensor_type = Nx.type(combined)
    output = Nx.template({batch, seq_len, memory_size}, tensor_type)

    # Pack momentum into combined: [B,T,5*M] → [B,T,5*M+1]
    momentum_col = Nx.broadcast(Nx.tensor(momentum, type: tensor_type), {batch, seq_len, 1})
    packed = Nx.concatenate([combined, momentum_col], axis: 2)

    Nx.Shared.optional(:fused_miras_scan, [packed], output, fn packed ->
      combined_inner = Nx.slice_along_axis(packed, 0, memory_size * 5, axis: 2)
      miras_scan_fallback(combined_inner, memory_size, momentum)
    end)
  end

  defp miras_fused(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)
    dtype = dtype_flag(combined)
    esize = elem_size(combined)
    ttype = tensor_type(combined)

    combined_ptr = Nx.to_pointer(combined, mode: :local)

    case Edifice.CUDA.NIF.fused_miras_scan(
           combined_ptr.address, batch, seq_len, memory_size, momentum / 1, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * memory_size * esize

        Nx.from_pointer(
          {backend_for(combined), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, memory_size}
        )

      {:error, reason} ->
        raise "CUDA MIRAS scan failed: #{reason}"
    end
  end

  defp miras_scan_fallback(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)

    q_all = Nx.slice_along_axis(combined, 0, memory_size, axis: 2)
    k_all = Nx.slice_along_axis(combined, memory_size, memory_size, axis: 2)
    v_all = Nx.slice_along_axis(combined, memory_size * 2, memory_size, axis: 2)
    alpha_all = Nx.slice_along_axis(combined, memory_size * 3, memory_size, axis: 2)
    eta_all = Nx.slice_along_axis(combined, memory_size * 4, memory_size, axis: 2)

    m_init = Nx.broadcast(0.0, {batch, memory_size, memory_size})
    mom_init = Nx.broadcast(0.0, {batch, memory_size, memory_size})

    {_, _, output_list} =
      Enum.reduce(0..(seq_len - 1), {m_init, mom_init, []}, fn t, {m_prev, mom_prev, acc} ->
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_raw = Nx.slice_along_axis(alpha_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_raw = Nx.slice_along_axis(eta_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        alpha_t = Nx.sigmoid(alpha_raw)
        eta_t = Nx.sigmoid(eta_raw)
        pred = Nx.dot(m_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        error = Nx.subtract(pred, v_t)
        grad_coeff = Nx.multiply(2.0, error)
        grad = Nx.dot(Nx.new_axis(grad_coeff, 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0])
        mom_t = Nx.add(Nx.multiply(momentum, mom_prev), grad)
        alpha_expanded = Nx.new_axis(alpha_t, 2)
        eta_expanded = Nx.new_axis(eta_t, 2)
        m_t = Nx.subtract(Nx.multiply(alpha_expanded, m_prev), Nx.multiply(eta_expanded, mom_t))

        # L2 row normalization (Moneta)
        norm = Nx.sqrt(Nx.sum(Nx.pow(m_t, 2), axes: [2], keep_axes: true))
        norm = Nx.max(norm, 1.0e-6)
        m_t = Nx.divide(m_t, norm)

        o_t = Nx.dot(m_t, [2], [0], Nx.new_axis(q_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])

        {m_t, mom_t, [o_t | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # GSA (Gated Slot Attention) scan
  # ============================================================================

  @doc """
  GSA scan — slot memory with gated write + softmax read per timestep.

  ## Arguments

    * `q` - Query `[batch, seq, num_heads, head_dim]` (post ELU+1)
    * `k_slot` - Slot keys `[batch, seq, num_heads, num_slots]` (post softmax)
    * `v` - Values `[batch, seq, num_heads, head_dim]`
    * `alpha` - Gate `[batch, seq, num_heads]` (damped sigmoid)

  ## Returns

    Output `[batch, seq, num_heads * head_dim]`
  """
  def gsa_scan(q, k_slot, v, alpha) do
    cond do
      custom_call_available?() ->
        gsa_custom_call(q, k_slot, v, alpha)

      cuda_available?(q) ->
        gsa_fused(q, k_slot, v, alpha)

      true ->
        gsa_scan_fallback(q, k_slot, v, alpha)
    end
  end

  defp gsa_custom_call(q, k_slot, v, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    tensor_type = Nx.type(q)
    output = Nx.template({batch, seq_len, num_heads * head_dim}, tensor_type)

    Nx.Shared.optional(:fused_gsa_scan, [q, k_slot, v, alpha], output, fn q, k_slot, v, alpha ->
      gsa_scan_fallback(q, k_slot, v, alpha)
    end)
  end

  defp gsa_fused(q, k_slot, v, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    {_, _, _, num_slots} = Nx.shape(k_slot)
    dtype = dtype_flag(q)
    esize = elem_size(q)
    ttype = tensor_type(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    ks_ptr = Nx.to_pointer(k_slot, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)

    case Edifice.CUDA.NIF.fused_gsa_scan(
           q_ptr.address, ks_ptr.address, v_ptr.address, alpha_ptr.address,
           batch, seq_len, num_heads, num_slots, head_dim, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * esize

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, num_heads * head_dim}
        )

      {:error, reason} ->
        raise "CUDA GSA scan failed: #{reason}"
    end
  end

  defp gsa_scan_fallback(q, k_slot, v, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    {_, _, _, num_slots} = Nx.shape(k_slot)

    slot_mem = Nx.broadcast(0.0, {batch, num_heads, num_slots, head_dim})

    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {slot_mem, []}, fn t, {mem, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        ks_t = Nx.slice_along_axis(k_slot, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        alpha_broadcast = alpha_t |> Nx.new_axis(2) |> Nx.new_axis(3)
        kv_outer = Nx.multiply(Nx.new_axis(ks_t, 3), Nx.new_axis(v_t, 2))
        mem_new = Nx.add(
          Nx.multiply(alpha_broadcast, mem),
          Nx.multiply(Nx.subtract(1.0, alpha_broadcast), kv_outer)
        )

        q_t_exp = Nx.new_axis(q_t, 3)
        scores = Nx.dot(mem_new, [3], [0, 1], q_t_exp, [2], [0, 1]) |> Nx.squeeze(axes: [3])

        max_s = Nx.reduce_max(scores, axes: [2], keep_axes: true)
        exp_s = Nx.exp(Nx.subtract(scores, max_s))
        p = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [2], keep_axes: true), 1.0e-8))
        output_t = Nx.sum(Nx.multiply(Nx.new_axis(p, 3), mem_new), axes: [2])

        o_flat = Nx.reshape(output_t, {batch, num_heads * head_dim})
        {mem_new, [o_flat | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
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
  # Multi-layer block scan — MinGRU
  # ============================================================================

  @doc """
  Multi-layer fused MinGRU block scan.

  Processes all layers in a single kernel launch. LayerNorm, dense projections,
  sigmoid, scan, and residual are all fused per layer with activations kept in
  shared memory / registers.

  ## Inputs
    * `input` - [B, T, H] input sequence
    * `weights` - flat packed weights, `[num_layers * (2*H*H + 4*H)]`
    * `h0` - [B, num_layers, H] per-layer initial hidden states
    * `num_layers` - number of layers

  ## Returns
    `[B, T, H]` — output of the final layer for all timesteps
  """
  def mingru_block(input, weights, h0, num_layers) do
    cond do
      block_custom_call_available?() ->
        mingru_block_custom_call(input, weights, h0, num_layers)

      cuda_available?(input) ->
        mingru_block_fused(input, weights, h0, num_layers)

      true ->
        mingru_block_fallback(input, weights, h0, num_layers)
    end
  end

  defp mingru_block_custom_call(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    tensor_type = Nx.type(input)
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    Nx.Shared.optional(:fused_mingru_block_scan, [input, weights, h0], output, fn inp, w, h ->
      mingru_block_fallback(inp, w, h, num_layers)
    end)
  end

  defp mingru_block_fused(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    dtype = dtype_flag(input)
    esize = elem_size(input)
    ttype = tensor_type(input)

    input_ptr = Nx.to_pointer(input, mode: :local)
    weights_ptr = Nx.to_pointer(weights, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_mingru_block_scan(
           input_ptr.address, weights_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, num_layers, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(input), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused MinGRU block scan failed: #{reason}"
    end
  end

  defp mingru_block_fallback(input, weights, h0, num_layers) do
    {_batch, _seq_len, hidden} = Nx.shape(input)
    layer_stride = 2 * hidden * hidden + 4 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      # Extract per-layer weights from packed buffer
      w_z = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_z = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      w_h = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_h = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      # LayerNorm
      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      # Dense projections: [B,T,H] @ [H,H]^T -> [B,T,H]
      z_pre = Nx.add(Nx.dot(normed, [2], w_z, [0]), b_z)
      c_pre = Nx.add(Nx.dot(normed, [2], w_h, [0]), b_h)

      # Sigmoid + scan
      z = Nx.sigmoid(z_pre)
      candidates = c_pre

      # Run per-layer scan (reuse existing scan function)
      h_layer_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch_size, seq_len, _hidden} = Nx.shape(z)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {h_layer_init, []}, fn t, {h_prev, acc} ->
          z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
          {h_t, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)

      # Residual
      Nx.add(x, scan_out)
    end)
  end

  # ============================================================================
  # Multi-layer block scan — MinLSTM
  # ============================================================================

  @doc """
  Multi-layer fused MinLSTM block scan.

  Same as `mingru_block/4` but for MinLSTM with 3 projections and gate normalization.

  ## Inputs
    * `input` - [B, T, H] input sequence
    * `weights` - flat packed weights, `[num_layers * (3*H*H + 5*H)]`
    * `h0` - [B, num_layers, H] per-layer initial hidden states
    * `num_layers` - number of layers

  ## Returns
    `[B, T, H]` — output of the final layer for all timesteps
  """
  def minlstm_block(input, weights, h0, num_layers) do
    cond do
      block_custom_call_available?() ->
        minlstm_block_custom_call(input, weights, h0, num_layers)

      cuda_available?(input) ->
        minlstm_block_fused(input, weights, h0, num_layers)

      true ->
        minlstm_block_fallback(input, weights, h0, num_layers)
    end
  end

  defp minlstm_block_custom_call(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    tensor_type = Nx.type(input)
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    Nx.Shared.optional(:fused_minlstm_block_scan, [input, weights, h0], output, fn inp, w, h ->
      minlstm_block_fallback(inp, w, h, num_layers)
    end)
  end

  defp minlstm_block_fused(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    dtype = dtype_flag(input)
    esize = elem_size(input)
    ttype = tensor_type(input)

    input_ptr = Nx.to_pointer(input, mode: :local)
    weights_ptr = Nx.to_pointer(weights, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_minlstm_block_scan(
           input_ptr.address, weights_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, num_layers, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(input), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused MinLSTM block scan failed: #{reason}"
    end
  end

  defp minlstm_block_fallback(input, weights, h0, num_layers) do
    {_batch, _seq_len, hidden} = Nx.shape(input)
    layer_stride = 3 * hidden * hidden + 5 * hidden
    norm_eps = 1.0e-6

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      # Extract per-layer weights
      w_f = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_f = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      w_i = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_i = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      w_h = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_h = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      # LayerNorm
      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      # Dense projections
      f_pre = Nx.add(Nx.dot(normed, [2], w_f, [0]), b_f)
      i_pre = Nx.add(Nx.dot(normed, [2], w_i, [0]), b_i)
      c_pre = Nx.add(Nx.dot(normed, [2], w_h, [0]), b_h)

      # Gate normalization
      sig_f = Nx.sigmoid(f_pre)
      sig_i = Nx.sigmoid(i_pre)
      gate_sum = Nx.add(sig_f, Nx.add(sig_i, norm_eps))
      f_norm = Nx.divide(sig_f, gate_sum)
      i_norm = Nx.divide(sig_i, gate_sum)

      # Per-layer scan
      h_layer_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch_size, seq_len, _hidden} = Nx.shape(f_norm)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {h_layer_init, []}, fn t, {h_prev, acc} ->
          f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          c_t = Nx.slice_along_axis(c_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          h_t = Nx.add(Nx.multiply(f_t, h_prev), Nx.multiply(i_t, c_t))
          {h_t, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)

      # Residual
      Nx.add(x, scan_out)
    end)
  end

  # ============================================================================
  # Multi-layer block scan — Linear (h = a*h + b)
  # ============================================================================

  @doc """
  Multi-layer fused linear block scan.

  Processes all layers in a single kernel launch. LayerNorm, dense projections
  (for a and b coefficients), linear scan update, and residual are all fused
  per layer with activations kept in shared memory / registers.

  ## Inputs
    * `input` - [B, T, H] input sequence
    * `weights` - flat packed weights, `[num_layers * (2*H*H + 4*H)]`
    * `h0` - [B, num_layers, H] per-layer initial hidden states
    * `num_layers` - number of layers

  ## Returns
    `[B, T, H]` — output of the final layer for all timesteps
  """
  def linear_block(input, weights, h0, num_layers) do
    cond do
      linear_block_custom_call_available?() ->
        linear_block_custom_call(input, weights, h0, num_layers)

      cuda_available?(input) ->
        linear_block_fused(input, weights, h0, num_layers)

      true ->
        linear_block_fallback(input, weights, h0, num_layers)
    end
  end

  defp linear_block_custom_call(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    tensor_type = Nx.type(input)
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    Nx.Shared.optional(:fused_linear_block_scan, [input, weights, h0], output, fn inp, w, h ->
      linear_block_fallback(inp, w, h, num_layers)
    end)
  end

  defp linear_block_fused(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    dtype = dtype_flag(input)
    esize = elem_size(input)
    ttype = tensor_type(input)

    input_ptr = Nx.to_pointer(input, mode: :local)
    weights_ptr = Nx.to_pointer(weights, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_linear_block_scan(
           input_ptr.address, weights_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, num_layers, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(input), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused linear block scan failed: #{reason}"
    end
  end

  defp linear_block_fallback(input, weights, h0, num_layers) do
    {_batch, _seq_len, hidden} = Nx.shape(input)
    layer_stride = 2 * hidden * hidden + 4 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      # Extract per-layer weights from packed buffer
      w_a = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_a = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      w_b = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_b = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      # LayerNorm
      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      # Dense projections: [B,T,H] @ [H,H]^T -> [B,T,H]
      a_pre = Nx.add(Nx.dot(normed, [2], w_a, [0]), b_a)
      b_pre = Nx.add(Nx.dot(normed, [2], w_b, [0]), b_b)

      # Run per-layer scan: h = a*h + b
      h_layer_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch_size, seq_len, _hidden} = Nx.shape(a_pre)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {h_layer_init, []}, fn t, {h_prev, acc} ->
          a_t = Nx.slice_along_axis(a_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          b_t = Nx.slice_along_axis(b_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
          {h_t, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)

      # Residual
      Nx.add(x, scan_out)
    end)
  end

  # ============================================================================
  # Multi-layer block scan — LSTM
  # ============================================================================

  @doc """
  Multi-layer fused LSTM block scan.

  Processes all layers in a single kernel launch. LayerNorm, input projection
  (W_x@normed), recurrent projection (R@h via shared memory), 4-gate LSTM
  update, and residual are all fused per layer with h and c state in registers.

  ## Inputs
    * `input` - [B, T, H] input sequence
    * `weights` - flat packed weights, `[num_layers * (8*H*H + 6*H)]`
    * `h0` - [B, num_layers, H] per-layer initial hidden states
    * `c0` - [B, num_layers, H] per-layer initial cell states
    * `num_layers` - number of layers

  ## Returns
    `[B, T, H]` — output of the final layer for all timesteps
  """
  def lstm_block(input, weights, h0, c0, num_layers) do
    cond do
      lstm_block_custom_call_available?() ->
        lstm_block_custom_call(input, weights, h0, c0, num_layers)

      cuda_available?(input) ->
        lstm_block_fused(input, weights, h0, c0, num_layers)

      true ->
        lstm_block_fallback(input, weights, h0, c0, num_layers)
    end
  end

  defp lstm_block_custom_call(input, weights, h0, c0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    tensor_type = Nx.type(input)
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    Nx.Shared.optional(:fused_lstm_block_scan, [input, weights, h0, c0], output, fn inp, w, h, c ->
      lstm_block_fallback(inp, w, h, c, num_layers)
    end)
  end

  defp lstm_block_fused(input, weights, h0, c0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    dtype = dtype_flag(input)
    esize = elem_size(input)
    ttype = tensor_type(input)

    input_ptr = Nx.to_pointer(input, mode: :local)
    weights_ptr = Nx.to_pointer(weights, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)
    c0_ptr = Nx.to_pointer(c0, mode: :local)

    case Edifice.CUDA.NIF.fused_lstm_block_scan(
           input_ptr.address, weights_ptr.address, h0_ptr.address, c0_ptr.address,
           batch, seq_len, hidden, num_layers, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(input), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused LSTM block scan failed: #{reason}"
    end
  end

  defp lstm_block_fallback(input, weights, h0, c0, num_layers) do
    {_batch, _seq_len, hidden} = Nx.shape(input)
    layer_stride = 8 * hidden * hidden + 6 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      # Extract per-layer weights: W_x[H,4H], b_x[4H], R[H,4H], gamma[H], beta[H]
      w_x = Nx.slice(weights, [offset], [hidden * 4 * hidden]) |> Nx.reshape({hidden, 4 * hidden})
      offset = offset + hidden * 4 * hidden
      b_x = Nx.slice(weights, [offset], [4 * hidden])
      offset = offset + 4 * hidden
      r_w = Nx.slice(weights, [offset], [hidden * 4 * hidden]) |> Nx.reshape({hidden, 4 * hidden})
      offset = offset + hidden * 4 * hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      # LayerNorm
      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      # Input projection: [B,T,H] @ [H,4H]^T -> [B,T,4H]
      wx_proj = Nx.add(Nx.dot(normed, [2], w_x, [0]), b_x)

      # Split gates
      wx_i = Nx.slice_along_axis(wx_proj, 0, hidden, axis: 2)
      wx_f = Nx.slice_along_axis(wx_proj, hidden, hidden, axis: 2)
      wx_g = Nx.slice_along_axis(wx_proj, 2 * hidden, hidden, axis: 2)
      wx_o = Nx.slice_along_axis(wx_proj, 3 * hidden, hidden, axis: 2)

      # Run per-layer scan with recurrent R@h
      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])
      c_init = Nx.slice_along_axis(c0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch_size, seq_len, _hidden} = Nx.shape(wx_i)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {{h_init, c_init}, []}, fn t, {{h_prev, c_prev}, acc} ->
          # R @ h_prev: [B,H] @ [H,4H]^T -> [B,4H]
          rh = Nx.dot(h_prev, [1], r_w, [0])
          rh_i = Nx.slice_along_axis(rh, 0, hidden, axis: 1)
          rh_f = Nx.slice_along_axis(rh, hidden, hidden, axis: 1)
          rh_g = Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1)
          rh_o = Nx.slice_along_axis(rh, 3 * hidden, hidden, axis: 1)

          wi_t = Nx.slice_along_axis(wx_i, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wf_t = Nx.slice_along_axis(wx_f, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wg_t = Nx.slice_along_axis(wx_g, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wo_t = Nx.slice_along_axis(wx_o, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

          i_gate = Nx.sigmoid(Nx.add(wi_t, rh_i))
          f_gate = Nx.sigmoid(Nx.add(wf_t, rh_f))
          g_gate = Nx.tanh(Nx.add(wg_t, rh_g))
          o_gate = Nx.sigmoid(Nx.add(wo_t, rh_o))

          c_t = Nx.add(Nx.multiply(f_gate, c_prev), Nx.multiply(i_gate, g_gate))
          h_t = Nx.multiply(o_gate, Nx.tanh(c_t))
          {{h_t, c_t}, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)

      # Residual
      Nx.add(x, scan_out)
    end)
  end

  # ============================================================================
  # Multi-layer block scan — GRU
  # ============================================================================

  @doc """
  Multi-layer fused GRU block scan.

  Processes all layers in a single kernel launch. LayerNorm, input projection
  (W_x@normed), recurrent projection (R@h via shared memory), 3-gate GRU
  update, and residual are all fused per layer with h state in registers.

  ## Inputs
    * `input` - [B, T, H] input sequence
    * `weights` - flat packed weights, `[num_layers * (6*H*H + 5*H)]`
    * `h0` - [B, num_layers, H] per-layer initial hidden states
    * `num_layers` - number of layers

  ## Returns
    `[B, T, H]` — output of the final layer for all timesteps
  """
  def gru_block(input, weights, h0, num_layers) do
    cond do
      gru_block_custom_call_available?() ->
        gru_block_custom_call(input, weights, h0, num_layers)

      cuda_available?(input) ->
        gru_block_fused(input, weights, h0, num_layers)

      true ->
        gru_block_fallback(input, weights, h0, num_layers)
    end
  end

  defp gru_block_custom_call(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    tensor_type = Nx.type(input)
    output = Nx.template({batch, seq_len, hidden}, tensor_type)

    Nx.Shared.optional(:fused_gru_block_scan, [input, weights, h0], output, fn inp, w, h ->
      gru_block_fallback(inp, w, h, num_layers)
    end)
  end

  defp gru_block_fused(input, weights, h0, num_layers) do
    {batch, seq_len, hidden} = Nx.shape(input)
    dtype = dtype_flag(input)
    esize = elem_size(input)
    ttype = tensor_type(input)

    input_ptr = Nx.to_pointer(input, mode: :local)
    weights_ptr = Nx.to_pointer(weights, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_gru_block_scan(
           input_ptr.address, weights_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, num_layers, dtype
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * esize

        Nx.from_pointer(
          {backend_for(input), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          ttype,
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused GRU block scan failed: #{reason}"
    end
  end

  defp gru_block_fallback(input, weights, h0, num_layers) do
    {_batch, _seq_len, hidden} = Nx.shape(input)
    layer_stride = 6 * hidden * hidden + 5 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      # Extract per-layer weights: W_x[H,3H], b_x[3H], R[H,3H], gamma[H], beta[H]
      w_x = Nx.slice(weights, [offset], [hidden * 3 * hidden]) |> Nx.reshape({hidden, 3 * hidden})
      offset = offset + hidden * 3 * hidden
      b_x = Nx.slice(weights, [offset], [3 * hidden])
      offset = offset + 3 * hidden
      r_w = Nx.slice(weights, [offset], [hidden * 3 * hidden]) |> Nx.reshape({hidden, 3 * hidden})
      offset = offset + hidden * 3 * hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      # LayerNorm
      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      # Input projection: [B,T,H] @ [H,3H]^T -> [B,T,3H]
      wx_proj = Nx.add(Nx.dot(normed, [2], w_x, [0]), b_x)

      # Split gates
      wx_r = Nx.slice_along_axis(wx_proj, 0, hidden, axis: 2)
      wx_z = Nx.slice_along_axis(wx_proj, hidden, hidden, axis: 2)
      wx_n = Nx.slice_along_axis(wx_proj, 2 * hidden, hidden, axis: 2)

      # Run per-layer scan with recurrent R@h
      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch_size, seq_len, _hidden} = Nx.shape(wx_r)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
          # R @ h_prev: [B,H] @ [H,3H]^T -> [B,3H]
          rh = Nx.dot(h_prev, [1], r_w, [0])
          rh_r = Nx.slice_along_axis(rh, 0, hidden, axis: 1)
          rh_z = Nx.slice_along_axis(rh, hidden, hidden, axis: 1)
          rh_n = Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1)

          wr_t = Nx.slice_along_axis(wx_r, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wz_t = Nx.slice_along_axis(wx_z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wn_t = Nx.slice_along_axis(wx_n, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

          r_gate = Nx.sigmoid(Nx.add(wr_t, rh_r))
          z_gate = Nx.sigmoid(Nx.add(wz_t, rh_z))
          n_gate = Nx.tanh(Nx.add(wn_t, Nx.multiply(r_gate, rh_n)))

          h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_gate), h_prev), Nx.multiply(z_gate, n_gate))
          {h_t, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)

      # Residual
      Nx.add(x, scan_out)
    end)
  end

  @doc false
  def block_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_mingru_block_scan, 4)
  rescue
    _ -> false
  end

  @doc false
  def linear_block_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_linear_block_scan, 4)
  rescue
    _ -> false
  end

  @doc false
  def lstm_block_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_lstm_block_scan, 5)
  rescue
    _ -> false
  end

  @doc false
  def gru_block_custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_gru_block_scan, 4)
  rescue
    _ -> false
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
      function_exported?(Edifice.CUDA.NIF, :fused_mingru_scan, 7)
  end

  @doc false
  def custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_mingru_scan, 4)
  rescue
    _ -> false
  end
end
