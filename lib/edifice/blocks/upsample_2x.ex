defmodule Edifice.Blocks.Upsample2x do
  @moduledoc """
  Nearest-neighbor 2x spatial upsample for channels-last tensors.

  Doubles both height and width by reshaping to insert singleton dimensions,
  broadcasting to 2x, then reshaping back. Used in feature pyramid networks
  (RT-DETR CCFM) and mask decoders (SAM 2).

  ## Usage

      upsampled = Upsample2x.layer(input, "upsample_stage1")
      # [batch, H, W, C] → [batch, 2*H, 2*W, C]
  """

  @doc """
  Build a 2x nearest-neighbor upsample Axon layer.

  ## Parameters

    - `input` - Axon node with shape `[batch, H, W, C]` (channels-last)
    - `name` - Layer name
  """
  @spec layer(Axon.t(), String.t()) :: Axon.t()
  def layer(input, name) do
    Axon.nx(
      input,
      fn t ->
        {b, h, w, c} = Nx.shape(t)

        t
        |> Nx.reshape({b, h, 1, w, 1, c})
        |> Nx.broadcast({b, h, 2, w, 2, c})
        |> Nx.reshape({b, h * 2, w * 2, c})
      end,
      name: name
    )
  end
end
