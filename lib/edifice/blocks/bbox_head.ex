defmodule Edifice.Blocks.BBoxHead do
  @moduledoc """
  Bounding box regression head.

  A 3-layer MLP that predicts normalized bounding box coordinates
  `(cx, cy, w, h)` in `[0, 1]`. Used by DETR and RT-DETR for object
  detection output and iterative box refinement.

  ## Architecture

  ```
  Input [batch, *, hidden_dim]
        |
  Dense(hidden_dim) → ReLU
        |
  Dense(hidden_dim) → ReLU
        |
  Dense(4) → Sigmoid
        |
  [batch, *, 4]  (cx, cy, w, h in [0, 1])
  ```

  ## Usage

      bbox_pred = BBoxHead.layer(decoded_features, 256, "bbox")
  """

  @doc """
  Build a bounding box regression MLP layer.

  ## Parameters

    - `input` - Axon node with last dim >= `hidden_dim`
    - `hidden_dim` - MLP hidden dimension
    - `name` - Layer name prefix
  """
  @spec layer(Axon.t(), pos_integer(), String.t()) :: Axon.t()
  def layer(input, hidden_dim, name) do
    input
    |> Axon.dense(hidden_dim, name: "#{name}_mlp1")
    |> Axon.activation(:relu, name: "#{name}_act1")
    |> Axon.dense(hidden_dim, name: "#{name}_mlp2")
    |> Axon.activation(:relu, name: "#{name}_act2")
    |> Axon.dense(4, name: "#{name}_mlp3")
    |> Axon.activation(:sigmoid, name: "#{name}_sig")
  end
end
