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
  def fused_mingru_scan(_gates_ptr, _cand_ptr, _h0_ptr, _batch, _seq_len, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_minlstm_scan(_forget_ptr, _input_ptr, _cand_ptr, _h0_ptr, _batch, _seq_len, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_elu_gru_scan(_gates, _cand, _h0, _batch, _seq, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_real_gru_scan(_gates, _cand, _h0, _batch, _seq, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_diag_linear_scan(_a, _b, _h0, _batch, _seq, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_liquid_scan(_tau, _act, _h0, _batch, _seq, _hidden),
    do: :erlang.nif_error(:not_loaded)
end
