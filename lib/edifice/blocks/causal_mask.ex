defmodule Edifice.Blocks.CausalMask do
  @moduledoc """
  Causal and window attention mask utilities.

  Provides unified mask creation functions used across attention modules
  (MultiHead, GQA, InfiniAttention, RingAttention, etc.).

  ## Mask Types

  - **Causal**: Lower-triangular — each position attends only to itself and earlier positions
  - **Window**: Causal + limited lookback window
  - **Block diagonal**: For chunked/ring attention patterns

  All masks are boolean tensors where `true` = attend, `false` = mask out.

  ## Usage

      mask = CausalMask.causal(64)
      window = CausalMask.window(64, 16)

      # Copy to BinaryBackend for use inside Axon.nx closures
      mask = CausalMask.to_binary_backend(mask)
  """

  @doc """
  Create a causal (autoregressive) attention mask.

  Returns a boolean tensor of shape `[seq_len, seq_len]` where position `i`
  can attend to positions `0..i`.

  ## Examples

      iex> mask = Edifice.Blocks.CausalMask.causal(4)
      iex> Nx.shape(mask)
      {4, 4}
  """
  @spec causal(non_neg_integer()) :: Nx.Tensor.t()
  def causal(seq_len) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    Nx.greater_equal(rows, cols)
  end

  @doc """
  Create a sliding window attention mask.

  Each position attends to at most `window_size` preceding positions
  (including itself). Combines causal constraint with window constraint.

  ## Examples

      iex> mask = Edifice.Blocks.CausalMask.window(8, 3)
      iex> Nx.shape(mask)
      {8, 8}
  """
  @spec window(non_neg_integer(), non_neg_integer()) :: Nx.Tensor.t()
  def window(seq_len, window_size) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    causal_cond = Nx.greater_equal(rows, cols)
    window_cond = Nx.greater_equal(cols, Nx.subtract(rows, window_size - 1))

    Nx.logical_and(causal_cond, window_cond)
  end

  @doc """
  Copy a mask tensor to `Nx.BinaryBackend`.

  Required when capturing masks in `Axon.nx` closures to avoid
  EXLA/Defn.Expr backend mismatch during JIT compilation.
  """
  @spec to_binary_backend(Nx.Tensor.t()) :: Nx.Tensor.t()
  def to_binary_backend(mask) do
    Nx.backend_copy(mask, Nx.BinaryBackend)
  end
end
