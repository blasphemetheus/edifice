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
  sLSTM scan with hidden-to-hidden matmul fused into the kernel.

  Inputs are pre-computed W@x `[batch, seq_len, 4*hidden]` and recurrent
  weight R `[hidden, 4*hidden]`. The kernel computes R@h internally using
  shared memory. Uses log-domain stabilized exponential gating.
  """
  def slstm_scan(wx, recurrent_weight) do
    cond do
      custom_call_available?() ->
        slstm_custom_call(wx, recurrent_weight)

      true ->
        Edifice.Recurrent.SLSTM.slstm_scan_fallback(wx, recurrent_weight)
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

      true ->
        Edifice.SSM.Common.selective_scan_fallback(x, dt, a, b, c)
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

  # ELU-GRU: inputs are raw (pre-activation) — kernel applies sigmoid/elu internally
  defp elu_gru_custom_call(gates, candidates) do
    {batch, seq_len, hidden} = Nx.shape(gates)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_elu_gru_scan, [gates, candidates, h0], output, fn g, c, h0 ->
      elu_gru_scan_fallback(g, c, h0)
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
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_real_gru_scan, [gates, candidates, h0], output, fn g, c, h0 ->
      real_gru_scan_fallback(g, c, h0)
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
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_diag_linear_scan, [a_vals, b_vals, h0], output, fn a, b, h0 ->
      diag_linear_scan_fallback(a, b, h0)
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
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_liquid_scan, [tau, activation, h0], output, fn tau, act, h0 ->
      liquid_scan_fallback(tau, act, h0)
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
  defp linear_scan_custom_call(a_vals, b_vals) do
    {batch, seq_len, hidden} = Nx.shape(a_vals)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_linear_scan, [a_vals, b_vals, h0], output, fn a, b, h0 ->
      linear_scan_cc_fallback(a, b, h0)
    end)
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

  # DeltaNet delta rule: matrix-state recurrence
  defp delta_net_custom_call(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, {:f, 32})

    Nx.Shared.optional(:fused_delta_net_scan, [q, k, v, beta], output, fn q, k, v, beta ->
      delta_net_scan_fallback(q, k, v, beta)
    end)
  end

  defp delta_net_scan_fallback(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

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
    output = Nx.template({batch, seq_len, num_heads, head_dim}, {:f, 32})

    Nx.Shared.optional(:fused_gated_delta_net_scan, [q, k, v, beta, alpha], output, fn q, k, v, beta, alpha ->
      gated_delta_net_scan_fallback(q, k, v, beta, alpha)
    end)
  end

  defp gated_delta_net_scan_fallback(q, k, v, beta, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

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

  # sLSTM: fused R@h matmul with log-domain exponential gating
  defp slstm_custom_call(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_slstm_scan, [wx, recurrent_weight, h0, c0], output,
      fn wx, r, h0, c0 ->
        slstm_scan_fallback(wx, r, h0, c0)
      end)
  end

  defp slstm_scan_fallback(wx, recurrent_weight, h0, c0) do
    {_batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    n0 = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), Nx.shape(h0))
    m0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), Nx.shape(h0))

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

  # TTT-Linear: weight matrix W as hidden state, updated per timestep
  defp ttt_custom_call(q, k, v, eta, w0, ln_gamma, ln_beta) do
    {batch, seq_len, inner_size} = Nx.shape(q)

    # Broadcast W0 to [batch, inner_size, inner_size]
    w0_batched = Nx.broadcast(w0, {batch, inner_size, inner_size})
    output = Nx.template({batch, seq_len, inner_size}, {:f, 32})

    Nx.Shared.optional(:fused_ttt_scan, [q, k, v, eta, w0_batched, ln_gamma, ln_beta], output,
      fn q, k, v, eta, w0_b, ln_g, ln_b ->
        ttt_scan_fallback(q, k, v, eta, w0_b, ln_g, ln_b)
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
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_selective_scan, [x, dt, a, b, c], output,
      fn x, dt, a, b, c ->
        selective_scan_fallback(x, dt, a, b, c)
      end)
  end

  defp selective_scan_fallback(x, dt, a, b_proj, c_proj) do
    {batch, seq_len, hidden} = Nx.shape(x)
    {_h, state} = Nx.shape(a)

    results =
      for bi <- 0..(batch - 1) do
        per_hidden =
          for hi <- 0..(hidden - 1) do
            h_state = List.duplicate(0.0, state)

            {_, outputs} =
              Enum.reduce(0..(seq_len - 1), {h_state, []}, fn t, {hs, acc} ->
                x_t = Nx.to_number(x[bi][t][hi])
                dt_t = Nx.to_number(dt[bi][t][hi])
                dt_t = min(max(dt_t, 0.001), 0.1)

                {new_hs, y_t} =
                  Enum.reduce(0..(state - 1), {hs, 0.0}, fn s, {hs_acc, y_acc} ->
                    a_s = Nx.to_number(a[hi][s])
                    b_s = Nx.to_number(b_proj[bi][t][s])
                    c_s = Nx.to_number(c_proj[bi][t][s])

                    a_bar = :math.exp(dt_t * a_s)
                    b_bar = dt_t * b_s

                    new_h = a_bar * Enum.at(hs_acc, s) + b_bar * x_t
                    {List.replace_at(hs_acc, s, new_h), y_acc + c_s * new_h}
                  end)

                {new_hs, [y_t | acc]}
              end)

            Enum.reverse(outputs)
          end

        for t <- 0..(seq_len - 1) do
          for hi <- 0..(hidden - 1) do
            Enum.at(Enum.at(per_hidden, hi), t)
          end
        end
      end

    results |> List.flatten() |> Nx.tensor(type: :f32) |> Nx.reshape({batch, seq_len, hidden})
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

  @doc false
  def custom_call_available? do
    exla_value = Module.concat([EXLA, MLIR, Value])

    Code.ensure_loaded?(exla_value) and
      function_exported?(exla_value, :fused_mingru_scan, 4)
  rescue
    _ -> false
  end
end
