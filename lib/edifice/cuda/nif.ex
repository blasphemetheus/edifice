defmodule Edifice.CUDA.NIF do
  @moduledoc false
  # Low-level NIF bindings for fused CUDA scan kernels.
  #
  # All functions take raw uint64 device pointer addresses and integer dimensions.
  # Use Edifice.CUDA.FusedScan for the high-level API that handles Nx tensors.
  #
  # Return format: {:ok, output_ptr, gc_ref} | {:error, reason}
  #
  # gc_ref is a NIF resource that calls cudaFree when garbage collected.
  # The caller MUST hold gc_ref for the lifetime of any tensor wrapping
  # output_ptr. When gc_ref is GC'd, the GPU memory is freed.
  #
  # dtype argument: 0 = f32, 1 = bf16. Controls which kernel library
  # (f32 or bf16) is dispatched and the element size for buffer allocation.

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:edifice), ~c"libedifice_cuda_nif")

    case :erlang.load_nif(path, 0) do
      :ok -> :ok
      {:error, {:load_failed, _}} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc false
  def fused_mingru_scan(_gates_ptr, _cand_ptr, _h0_ptr, _batch, _seq_len, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_minlstm_scan(_forget_ptr, _input_ptr, _cand_ptr, _h0_ptr, _batch, _seq_len, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_elu_gru_scan(_gates, _cand, _h0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_real_gru_scan(_gates, _cand, _h0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_diag_linear_scan(_a, _b, _h0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_liquid_scan(_tau, _act, _h0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_linear_scan(_a, _b, _h0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_delta_net_scan(_q, _k, _v, _beta, _batch, _seq, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_gated_delta_net_scan(_q, _k, _v, _beta, _alpha, _batch, _seq, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_delta_product_scan(_q, _k, _v, _beta, _batch, _seq, _n_h, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_slstm_scan(_wx, _r, _h0, _c0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_lstm_scan(_wx, _r, _h0, _c0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_gru_scan(_wx, _r, _h0, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_ttt_scan(_q, _k, _v, _eta, _w0, _lng, _lnb, _batch, _seq, _inner, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_selective_scan(_x, _dt, _a, _b, _c, _batch, _seq, _hidden, _state, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_kda_scan(_q, _k, _v, _alpha, _beta, _batch, _seq, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_rla_scan(_q, _k, _v, _alpha, _beta, _gamma, _batch, _seq, _heads, _head_dim, _variant, _clip, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_flash_attention(_q, _k, _v, _batch, _heads, _seq, _head_dim, _causal, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_laser_attention(_q, _k, _v, _v_max, _batch, _heads, _seq, _head_dim, _causal, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_fox_attention(_q, _k, _v, _cs, _batch, _heads, _seq, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_reservoir_scan(_wx, _w_res, _h0, _batch, _seq, _hidden, _leak_rate, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_titans_scan(_combined, _batch, _seq, _mem_size, _momentum, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_miras_scan(_combined, _batch, _seq, _mem_size, _momentum, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_gsa_scan(_q, _k_slot, _v, _alpha, _batch, _seq, _heads, _slots, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  # Multi-layer block scan kernels
  @doc false
  def fused_mingru_block_scan(_input_ptr, _weights_ptr, _h0_ptr, _batch, _seq_len, _hidden, _num_layers, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_minlstm_block_scan(_input_ptr, _weights_ptr, _h0_ptr, _batch, _seq_len, _hidden, _num_layers, _dtype),
    do: :erlang.nif_error(:not_loaded)

  # Backward kernels — multi-output via concatenated buffer
  @doc false
  def fused_linear_scan_backward(_a_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_mingru_scan_backward(_z_ptr, _cand_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_minlstm_scan_backward(_f_ptr, _i_ptr, _cand_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_elu_gru_scan_backward(_z_ptr, _c_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_real_gru_scan_backward(_z_ptr, _cand_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_diag_linear_scan_backward(_a_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_lstm_scan_backward(_wx_ptr, _r_ptr, _h0_ptr, _c0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_gru_scan_backward(_wx_ptr, _r_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  # Phase 3 backward kernels
  @doc false
  def fused_liquid_scan_backward(_tau_ptr, _act_ptr, _h0_ptr, _fwd_ptr, _grad_ptr, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_delta_rule_scan_backward(_q, _k, _v, _beta, _fwd, _grad, _batch, _seq, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_gated_delta_net_scan_backward(_q, _k, _v, _beta, _alpha, _fwd, _grad, _batch, _seq, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_delta_product_scan_backward(_q, _k, _v, _beta, _fwd, _grad, _batch, _seq, _n_h, _heads, _head_dim, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_slstm_scan_backward(_wx, _r, _h0, _c0, _fwd, _grad, _batch, _seq, _hidden, _dtype),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_selective_scan_backward(_x, _dt, _a, _b, _c, _grad, _batch, _seq, _hidden, _state, _dtype),
    do: :erlang.nif_error(:not_loaded)
end
