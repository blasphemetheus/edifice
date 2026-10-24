defmodule Edifice.Serving.GenerateFusedContextTest do
  use ExUnit.Case, async: true
  alias Edifice.Serving.GenerateFused

  defp generate(prompt, opts) do
    predict = fn _, %{"state_sequence" => x} ->
      send(self(), {:context, Nx.to_list(Nx.as_type(Nx.squeeze(x, axes: [2]), :s32))})

      target =
        x
        |> Nx.squeeze(axes: [2])
        |> Nx.as_type(:s32)
        |> Nx.cumulative_sum(axis: 1)
        |> Nx.remainder(7)
        |> Nx.new_axis(-1)

      Nx.select(Nx.equal(target, Nx.iota({7})), 1000.0, -1000.0)
    end

    GenerateFused.generate(
      predict,
      %{},
      Keyword.merge(
        [
          prompt: prompt,
          embed_fn: &Nx.new_axis(Nx.as_type(&1, :f32), -1),
          seq_len: 3,
          max_tokens: 4,
          temperature: 0.01,
          seed: 42
        ],
        opts
      )
    )
  end

  test "every prediction sees prompt plus previous outputs, with sliding overflow" do
    output = generate(Nx.tensor([[1, 2]]), [])
    assert Nx.to_list(output) == [[3, 6, 4, 6]]
    assert_receive {:context, [[1, 2, 0]]}
    assert_receive {:context, [[1, 2, 3]]}
    assert_receive {:context, [[2, 3, 6]]}
    assert_receive {:context, [[3, 6, 4]]}
    refute_receive {:context, _}
  end

  test "long prompts use the last retained position and one token means one prediction" do
    assert Nx.to_list(generate(Nx.tensor([[6, 1, 2, 3]]), max_tokens: 1)) == [[6]]
    assert_receive {:context, [[1, 2, 3]]}
    refute_receive {:context, _}
  end

  test "stopping one batch row does not lose the other row's context" do
    output = generate(Nx.tensor([[0], [1]]), max_tokens: 3, stop_token: 0)
    assert Nx.to_list(output) == [[0, 0, 0], [1, 2, 4]]
  end

  test "invalid lengths and temperature fail before predicting" do
    for opts <- [[max_tokens: 0], [max_tokens: -1], [seq_len: 0], [temperature: 0]] do
      assert_raise ArgumentError, fn -> generate(Nx.tensor([[1]]), opts) end
    end

    refute_receive {:context, _}
  end
end
