defmodule Edifice.Pretrained.CompatibilityRegressionTest do
  use ExUnit.Case, async: true
  import Nx.Testing

  test "DETR query weights reach the parameter used by the model without transposition" do
    key_map = Edifice.Pretrained.KeyMaps.DETR
    name = key_map.map_key("model.query_position_embeddings.weight")
    assert name == "broadcast_queries.object_queries"
    queries = Nx.iota({3, 8}, type: :f32)

    assert_equal(
      Edifice.Pretrained.Transform.apply_transform(name, key_map.tensor_transforms(), queries),
      queries
    )
  end

  test "ViT adds a distinct learned vector at every position" do
    graph =
      Edifice.Vision.ViT.build(image_size: 4, patch_size: 2, embed_dim: 4, depth: 1, num_heads: 2)

    {id, _} = Enum.find(graph.nodes, fn {_, node} -> node.op_name == :add_pos_embed end)
    {init, predict} = Axon.build(%{graph | output: id})
    x = Nx.broadcast(0.0, {2, 3, 4, 4})
    state = init.(%{"image" => x}, Axon.ModelState.empty())

    data =
      Map.new(state.data, fn {layer, params} ->
        {layer, Map.new(params, fn {key, value} -> {key, Nx.broadcast(0.0, Nx.shape(value))} end)}
      end)

    positions = Nx.iota({5, 4}, type: :f32)
    data = put_in(data, ["pos_embed_proj", "kernel"], positions)

    assert_equal(
      predict.(%{state | data: data}, %{"image" => x}),
      Nx.broadcast(positions, {2, 5, 4})
    )
  end

  test "ViT patch projection preserves convolution channel and pixel order" do
    image = Nx.iota({1, 3, 4, 4}, type: :f32) |> Nx.divide(48)
    weight = Nx.iota({5, 3, 2, 2}, type: :f32) |> Nx.divide(60)
    input = Axon.input("image", shape: {nil, 3, 4, 4})
    model = Edifice.Blocks.PatchEmbed.layer(input, patch_size: 2, in_channels: 3, embed_dim: 5)

    kernel =
      Edifice.Pretrained.Transform.apply_transform(
        "patch_embed_proj.kernel",
        Edifice.Pretrained.KeyMaps.ViT.tensor_transforms(),
        weight
      )

    params =
      Axon.ModelState.new(%{
        "patch_embed_proj" => %{"kernel" => kernel, "bias" => Nx.broadcast(0.0, {5})}
      })

    {_, predict} = Axon.build(model)
    actual = predict.(params, %{"image" => image})

    expected =
      Nx.conv(image, weight, strides: [2, 2])
      |> Nx.transpose(axes: [0, 2, 3, 1])
      |> Nx.reshape({1, 4, 5})

    assert_all_close(actual, expected)
  end

  test "ViT uses separate attention heads" do
    input = Axon.input("x", shape: {nil, 2, 4})
    model = Edifice.Vision.ViT.transformer_block(input, 4, 2, 8, 0.0, name: "block")
    x = Nx.tensor([[[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.5, 3.0]]])
    {init, predict} = Axon.build(model)
    state = init.(%{"x" => x}, Axon.ModelState.empty())

    data =
      Map.new(state.data, fn {layer, params} ->
        {layer,
         Map.new(params, fn {key, value} ->
           {key,
            if(key == "gamma",
              do: Nx.broadcast(1.0, Nx.shape(value)),
              else: Nx.broadcast(0.0, Nx.shape(value))
            )}
         end)}
      end)

    data =
      put_in(
        data,
        ["block_attn_qkv", "kernel"],
        Nx.concatenate([Nx.eye(4), Nx.eye(4), Nx.eye(4)], axis: 1)
      )

    data = put_in(data, ["block_attn_proj", "kernel"], Nx.eye(4))

    normalized =
      Nx.divide(
        Nx.subtract(x, Nx.mean(x, axes: [2], keep_axes: true)),
        Nx.sqrt(Nx.add(Nx.variance(x, axes: [2], keep_axes: true), 1.0e-5))
      )

    heads =
      for start <- [0, 2] do
        h = normalized |> Nx.slice_along_axis(start, 2, axis: 2) |> Nx.squeeze(axes: [0])
        scores = Nx.dot(h, Nx.transpose(h)) |> Nx.divide(:math.sqrt(2))
        weights = Axon.Activations.softmax(scores)
        Nx.dot(weights, h)
      end

    expected = Nx.add(x, Nx.new_axis(Nx.concatenate(heads, axis: 1), 0))
    assert_all_close(predict.(%{state | data: data}, %{"x" => x}), expected)
  end

  test "DETR positions use the actual rectangular grid and alternate sine and cosine" do
    table = Edifice.Blocks.SinusoidalPE2D.detr_table(2, 3, 8)

    expected =
      for y <- 1..2, x <- 1..3 do
        for {position, size} <- [{y, 2}, {x, 3}],
            frequency <- [1.0, 0.01],
            fun <- [&:math.sin/1, &:math.cos/1] do
          fun.(position / (size + 1.0e-6) * 2 * :math.pi() * frequency)
        end
      end

    assert_all_close(table, Nx.tensor([expected]), atol: 1.0e-6)
  end
end
