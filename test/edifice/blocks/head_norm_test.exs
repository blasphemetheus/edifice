defmodule Edifice.Blocks.HeadNormTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  test "normalizes per token and head, with channel-wise affine parameters" do
    model =
      Axon.input("x", shape: {nil, nil, 4}) |> Edifice.Blocks.HeadNorm.layer(2, name: "norm")

    {init, predict} = Axon.build(model)
    x = Nx.tensor([[[1.0, 3, 10, 14], [100, 104, -5, 1]]])
    state = init.(x, Axon.ModelState.empty())

    state = %{
      state
      | data: %{
          "norm" => %{"gamma" => Nx.tensor([1.0, 2, 3, 4]), "beta" => Nx.tensor([0.0, 1, 2, 3])}
        }
    }

    expected = Nx.tensor([[[-1.0, 3, -1, 7], [-1.0, 3, -1, 7]]])
    assert Nx.to_number(Nx.all_close(predict.(state, x), expected, atol: 2.0e-5)) == 1

    # Future tokens and unrelated heads cannot change the first head at t=0.
    changed = Nx.tensor([[[1.0, 3, -100, 100], [0, 0, 0, 0]]])
    actual = predict.(state, changed)
    assert Nx.to_flat_list(actual[0][0][0..1]) == Nx.to_flat_list(predict.(state, x)[0][0][0..1])

    assert Nx.to_flat_list(predict.(state, Nx.slice_along_axis(x, 0, 1, axis: 1))) ==
             Nx.to_flat_list(predict.(state, x)[[0..0, 0..0]])
  end

  test "input gradients match finite differences" do
    model = Axon.input("x", shape: {nil, 2, 6}) |> Edifice.Blocks.HeadNorm.layer(2, name: "norm")
    {init, predict} = Axon.build(model)
    x = Nx.tensor([[[0.2, 1.3, -0.7, 2.0, 0.1, -1.0], [1.0, 2.0, 4.0, -2.0, 0.3, 1.0]]])
    state = init.(x, Axon.ModelState.empty())
    weights = Nx.iota({1, 2, 6}, type: :f32)
    loss = fn input -> Nx.sum(Nx.multiply(predict.(state, input), weights)) end
    gradient = grad(x, loss) |> Nx.to_flat_list()

    for i <- 0..11 do
      delta = Nx.equal(Nx.iota({1, 2, 6}), i) |> Nx.multiply(1.0e-3)

      numerical =
        (Nx.to_number(loss.(Nx.add(x, delta))) - Nx.to_number(loss.(Nx.subtract(x, delta)))) /
          0.002

      assert_in_delta Enum.at(gradient, i), numerical, 0.004
    end
  end

  for module <- [Edifice.Attention.RetNet, Edifice.Attention.RetNetV2] do
    test "#{module} keeps prefixes causal with shared weights" do
      module = unquote(module)
      opts = [embed_dim: 8, hidden_size: 8, num_heads: 2, num_layers: 1]
      {init, full} = Axon.build(module.build(Keyword.put(opts, :seq_len, 4)))
      {_, prefix} = Axon.build(module.build(Keyword.put(opts, :seq_len, 2)))
      {x, _} = Nx.Random.normal(Nx.Random.key(14), shape: {1, 4, 8})
      state = init.(x, Axon.ModelState.empty())
      # Remove only last-token selection to inspect all causal outputs.
      graph = module.build(Keyword.put(opts, :seq_len, 4))
      [parent] = graph.nodes[graph.output].parent
      {_, sequence} = Axon.build(%{graph | output: parent})
      short = Nx.slice_along_axis(x, 0, 2, axis: 1)
      expected = sequence.(state, x)[[0..0, 1, 0..7]]
      assert Nx.to_number(Nx.all_close(prefix.(state, short), expected, atol: 1.0e-5)) == 1
      assert Nx.to_number(Nx.all(Nx.is_nan(full.(state, x)))) == 0
      assert Enum.any?(module.build(opts).nodes, fn {_, node} -> node.op_name == :head_norm end)
    end
  end
end
