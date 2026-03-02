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

  # DeltaProduct: Householder product state transitions
  defp delta_product_custom_call(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, {:f, 32})

    Nx.Shared.optional(:fused_delta_product_scan, [q, k, v, beta], output, fn q, k, v, beta ->
      delta_product_scan_fallback(q, k, v, beta)
    end)
  end

  defp delta_product_scan_fallback(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    # k/v: [B, T, n_h, H, d], beta: [B, T, n_h, H]
    {_, _, num_householder, _, _} = Nx.shape(k)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})
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

  # Standard LSTM: classic 4-gate LSTM with cell and hidden state
  defp lstm_custom_call(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_lstm_scan, [wx, recurrent_weight, h0, c0], output,
      fn wx, r, h0, c0 ->
        lstm_scan_fallback(wx, r, h0, c0)
      end)
  end

  defp lstm_scan_fallback(wx, recurrent_weight, h0, c0) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    h0 = if h0, do: h0, else: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    c0 = if c0, do: c0, else: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

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

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    output = Nx.template({batch, seq_len, hidden}, {:f, 32})

    Nx.Shared.optional(:fused_gru_scan, [wx, recurrent_weight, h0], output,
      fn wx, r, h0 ->
        gru_scan_fallback(wx, r, h0)
      end)
  end

  defp gru_scan_fallback(wx, recurrent_weight, h0) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)

    h0 = if h0, do: h0, else: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

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
        # Must be defn-compatible (no Nx.to_number) since EXLA traces the fallback
        Edifice.SSM.Common.selective_scan_fallback(x, dt, a, b, c)
      end)
  end

  # KDA: channel-wise decay delta rule
  defp kda_custom_call(q, k, v, alpha, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    output = Nx.template({batch, seq_len, num_heads, head_dim}, {:f, 32})

    Nx.Shared.optional(:fused_kda_scan, [q, k, v, alpha, beta], output,
      fn q, k, v, alpha, beta ->
        kda_scan_fallback(q, k, v, alpha, beta)
      end)
  end

  defp kda_scan_fallback(q, k, v, alpha, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

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
    output = Nx.template({batch, seq_len, num_heads, head_dim}, {:f, 32})

    Nx.Shared.optional(:fused_rla_scan, [q, k, v, alpha, beta, gamma], output,
      fn q, k, v, alpha, beta, gamma ->
        rla_scan_fallback(q, k, v, alpha, beta, gamma, variant, clip_threshold)
      end)
  end

  defp rla_scan_fallback(q, k, v, alpha, beta, gamma, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)

    s0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})
    r0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_heads, head_dim, head_dim})

    {_, _, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, r0, []}, fn t, {s_prev, r_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        gamma_t = Nx.slice_along_axis(gamma, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Gates are [B, H, 1, 1] (per-head scalar, broadcastable to [B, H, d, d])
        # alpha/beta/gamma already shaped for broadcasting

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

  defp delta_product_fused(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    {_, _, num_householder, _, _} = Nx.shape(k)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)

    case Edifice.CUDA.NIF.fused_delta_product_scan(
           q_ptr.address, k_ptr.address, v_ptr.address, beta_ptr.address,
           batch, seq_len, num_householder, num_heads, head_dim
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
        raise "CUDA fused DeltaProduct scan failed: #{reason}"
    end
  end

  # ============================================================================
  # sLSTM / TTT / Mamba fused paths (graph-breaking NIF)
  # ============================================================================

  defp slstm_fused(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(wx)), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(wx)), {batch, hidden})

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    r_ptr = Nx.to_pointer(recurrent_weight, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)
    c0_ptr = Nx.to_pointer(c0, mode: :local)

    case Edifice.CUDA.NIF.fused_slstm_scan(
           wx_ptr.address, r_ptr.address, h0_ptr.address, c0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused sLSTM scan failed: #{reason}"
    end
  end

  defp lstm_fused(wx, recurrent_weight) do
    {batch, seq_len, hidden4} = Nx.shape(wx)
    hidden = div(hidden4, 4)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(wx)), {batch, hidden})
    c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(wx)), {batch, hidden})

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    r_ptr = Nx.to_pointer(recurrent_weight, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)
    c0_ptr = Nx.to_pointer(c0, mode: :local)

    case Edifice.CUDA.NIF.fused_lstm_scan(
           wx_ptr.address, r_ptr.address, h0_ptr.address, c0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused LSTM scan failed: #{reason}"
    end
  end

  defp gru_fused(wx, recurrent_weight) do
    {batch, seq_len, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)

    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: backend_for(wx)), {batch, hidden})

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    r_ptr = Nx.to_pointer(recurrent_weight, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_gru_scan(
           wx_ptr.address, r_ptr.address, h0_ptr.address,
           batch, seq_len, hidden
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, hidden}
        )

      {:error, reason} ->
        raise "CUDA fused GRU scan failed: #{reason}"
    end
  end

  defp ttt_fused(q, k, v, eta, w0, ln_gamma, ln_beta) do
    {batch, seq_len, inner_size} = Nx.shape(q)

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
           batch, seq_len, inner_size
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * inner_size * 4

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, seq_len, inner_size}
        )

      {:error, reason} ->
        raise "CUDA fused TTT scan failed: #{reason}"
    end
  end

  defp selective_scan_fused(x, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    {_h, state} = Nx.shape(a)

    x_ptr = Nx.to_pointer(x, mode: :local)
    dt_ptr = Nx.to_pointer(dt, mode: :local)
    a_ptr = Nx.to_pointer(a, mode: :local)
    b_ptr = Nx.to_pointer(b, mode: :local)
    c_ptr = Nx.to_pointer(c, mode: :local)

    case Edifice.CUDA.NIF.fused_selective_scan(
           x_ptr.address, dt_ptr.address, a_ptr.address,
           b_ptr.address, c_ptr.address,
           batch, seq_len, hidden, state
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * hidden * 4

        Nx.from_pointer(
          {backend_for(x), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
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

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)
    beta_ptr = Nx.to_pointer(beta, mode: :local)

    case Edifice.CUDA.NIF.fused_kda_scan(
           q_ptr.address, k_ptr.address, v_ptr.address,
           alpha_ptr.address, beta_ptr.address,
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
        raise "CUDA fused KDA scan failed: #{reason}"
    end
  end

  defp rla_fused(q, k, v, alpha, beta, gamma, variant, clip_threshold) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    variant_int = if variant == :rdn, do: 1, else: 0

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
           variant_int, clip_threshold * 1.0
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
      flash_attention_custom_call_available?() ->
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
    output = Nx.template({batch, num_heads, seq_len, head_dim}, {:f, 32})
    causal_tensor = Nx.tensor(causal, type: {:s, 32})

    Nx.Shared.optional(:fused_flash_attention, [q, k, v, causal_tensor], output, fn q, k, v, _causal ->
      flash_attention_fallback(q, k, v, causal)
    end)
  end

  defp flash_attention_fused(q, k, v, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)

    q_ptr = Nx.to_pointer(q, mode: :local)
    k_ptr = Nx.to_pointer(k, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)

    case Edifice.CUDA.NIF.fused_flash_attention(
           q_ptr.address, k_ptr.address, v_ptr.address,
           batch, num_heads, seq_len, head_dim, causal
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * num_heads * seq_len * head_dim * 4

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, num_heads, seq_len, head_dim}
        )

      {:error, reason} ->
        raise "CUDA flash attention failed: #{reason}"
    end
  end

  defp flash_attention_fallback(q, k, v, causal) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)

    # Standard scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: {:f, 32}))
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

        neg_inf = Nx.Constants.neg_infinity({:f, 32})
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    # Softmax + weighted sum
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
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
    output = Nx.template({batch, hidden}, {:f, 32})
    leak_tensor = Nx.tensor(leak_rate, type: {:f, 32})

    Nx.Shared.optional(:fused_reservoir_scan, [wx, w_res, Nx.broadcast(0.0, {batch, hidden}), leak_tensor], output, fn wx, w_res, _h0, _leak ->
      reservoir_scan_fallback(wx, w_res, leak_rate)
    end)
  end

  defp reservoir_fused(wx, w_res, leak_rate) do
    {batch, seq_len, hidden} = Nx.shape(wx)

    wx_ptr = Nx.to_pointer(wx, mode: :local)
    wres_ptr = Nx.to_pointer(w_res, mode: :local)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    case Edifice.CUDA.NIF.fused_reservoir_scan(
           wx_ptr.address, wres_ptr.address, h0_ptr.address,
           batch, seq_len, hidden, leak_rate / 1
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * hidden * 4

        Nx.from_pointer(
          {backend_for(wx), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
          {batch, hidden}
        )

      {:error, reason} ->
        raise "CUDA reservoir scan failed: #{reason}"
    end
  end

  defp reservoir_scan_fallback(wx, w_res, leak_rate) do
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
    output = Nx.template({batch, seq_len, memory_size}, {:f, 32})
    momentum_tensor = Nx.tensor(momentum, type: {:f, 32})

    Nx.Shared.optional(:fused_titans_scan, [combined, momentum_tensor], output, fn combined, _mom ->
      titans_scan_fallback(combined, memory_size, momentum)
    end)
  end

  defp titans_fused(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)

    combined_ptr = Nx.to_pointer(combined, mode: :local)

    case Edifice.CUDA.NIF.fused_titans_scan(
           combined_ptr.address, batch, seq_len, memory_size, momentum / 1
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * memory_size * 4

        Nx.from_pointer(
          {backend_for(combined), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
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
    output = Nx.template({batch, seq_len, memory_size}, {:f, 32})
    momentum_tensor = Nx.tensor(momentum, type: {:f, 32})

    Nx.Shared.optional(:fused_miras_scan, [combined, momentum_tensor], output, fn combined, _mom ->
      miras_scan_fallback(combined, memory_size, momentum)
    end)
  end

  defp miras_fused(combined, memory_size, momentum) do
    {batch, seq_len, _} = Nx.shape(combined)

    combined_ptr = Nx.to_pointer(combined, mode: :local)

    case Edifice.CUDA.NIF.fused_miras_scan(
           combined_ptr.address, batch, seq_len, memory_size, momentum / 1
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * memory_size * 4

        Nx.from_pointer(
          {backend_for(combined), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
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
    output = Nx.template({batch, seq_len, num_heads * head_dim}, {:f, 32})

    Nx.Shared.optional(:fused_gsa_scan, [q, k_slot, v, alpha], output, fn q, k_slot, v, alpha ->
      gsa_scan_fallback(q, k_slot, v, alpha)
    end)
  end

  defp gsa_fused(q, k_slot, v, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    {_, _, _, num_slots} = Nx.shape(k_slot)

    q_ptr = Nx.to_pointer(q, mode: :local)
    ks_ptr = Nx.to_pointer(k_slot, mode: :local)
    v_ptr = Nx.to_pointer(v, mode: :local)
    alpha_ptr = Nx.to_pointer(alpha, mode: :local)

    case Edifice.CUDA.NIF.fused_gsa_scan(
           q_ptr.address, ks_ptr.address, v_ptr.address, alpha_ptr.address,
           batch, seq_len, num_heads, num_slots, head_dim
         ) do
      {:ok, out_addr, gc_ref} ->
        hold_gc_ref(out_addr, gc_ref)
        out_bytes = batch * seq_len * num_heads * head_dim * 4

        Nx.from_pointer(
          {backend_for(q), client: :cuda, device_id: 0},
          %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
          {:f, 32},
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
