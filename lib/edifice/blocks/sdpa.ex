defmodule Edifice.Blocks.SDPA do
  @moduledoc """
  Scaled Dot-Product Attention (SDPA) computation.

  Shared multi-head attention math used across detection (DETR, RT-DETR, SAM 2),
  robotics (ACT), audio (Whisper), and cross-attention modules. Handles:

  1. Reshape Q/K/V from `[batch, seq, hidden]` to `[batch, heads, seq, head_dim]`
  2. Scaled dot-product: `softmax(QK^T / sqrt(d_k)) V`
  3. Optional boolean mask for causal or padding attention
  4. Reshape output back to `[batch, seq, hidden]`

  Uses `FusedOps.fused_softmax/1` for numerically stable softmax (FP32 internal).

  ## Usage

      # Without mask
      output = SDPA.compute(q, k, v, num_heads, head_dim)

      # With causal mask
      mask = CausalMask.causal(seq_len)
      output = SDPA.compute(q, k, v, num_heads, head_dim, mask)

  ## Examples

      iex> q = Nx.broadcast(0.5, {1, 4, 8})
      iex> k = Nx.broadcast(0.5, {1, 4, 8})
      iex> v = Nx.broadcast(0.5, {1, 4, 8})
      iex> output = Edifice.Blocks.SDPA.compute(q, k, v, 2, 4)
      iex> Nx.shape(output)
      {1, 4, 8}
  """

  alias Edifice.Utils.FusedOps

  @doc """
  Compute scaled dot-product attention without masking.

  ## Parameters

    - `q` - Query tensor `[batch, q_len, hidden]`
    - `k` - Key tensor `[batch, kv_len, hidden]`
    - `v` - Value tensor `[batch, kv_len, hidden]`
    - `num_heads` - Number of attention heads
    - `head_dim` - Dimension per head (`hidden / num_heads`)
  """
  @spec compute(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), pos_integer(), pos_integer()) ::
          Nx.Tensor.t()
  def compute(q, k, v, num_heads, head_dim) do
    compute(q, k, v, num_heads, head_dim, nil)
  end

  @doc """
  Compute scaled dot-product attention with optional masking.

  ## Parameters

    - `q` - Query tensor `[batch, q_len, hidden]`
    - `k` - Key tensor `[batch, kv_len, hidden]`
    - `v` - Value tensor `[batch, kv_len, hidden]`
    - `num_heads` - Number of attention heads
    - `head_dim` - Dimension per head
    - `mask` - Boolean mask `[q_len, kv_len]` where `true` = attend,
      `false` = mask out. Pass `nil` for no masking.
  """
  @spec compute(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          pos_integer(),
          pos_integer(),
          Nx.Tensor.t() | nil
        ) :: Nx.Tensor.t()
  def compute(q, k, v, num_heads, head_dim, mask) do
    {batch, q_len, _} = Nx.shape(q)
    {_, kv_len, _} = Nx.shape(k)

    # Reshape to [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, q_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Apply mask if provided
    scores =
      if mask do
        mask =
          mask
          |> Nx.reshape({1, 1, q_len, kv_len})
          |> Nx.broadcast({batch, num_heads, q_len, kv_len})

        neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    weights = FusedOps.fused_softmax(scores)

    # Apply to values: [batch, heads, q_len, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, q_len, hidden_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, q_len, num_heads * head_dim})
  end
end
