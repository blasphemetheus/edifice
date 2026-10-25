defmodule Edifice.Stateful.ReplayTest do
  use ExUnit.Case, async: true
  @moduletag :exla_only

  for arch <- [:gru, :lstm, :min_gru, :mamba, :mamba_ssd, :gated_delta_net] do
    test "#{arch}: compiled replay preserves nonzero carry and chunk boundaries" do
      arch = unquote(arch)
      Process.put(:__edifice_force_fallback__, true)

      opts = [
        embed_dim: 6,
        hidden_size: 8,
        num_layers: 2,
        state_size: 3,
        num_heads: 2,
        conv_size: 3,
        window_size: 9,
        dropout: 0.0
      ]

      {init, _} = Axon.build(Edifice.build(arch, opts))
      {x, _} = Nx.Random.normal(Nx.Random.key(91), shape: {2, 9, 6})
      params = init.(x, Axon.ModelState.empty())
      module = Edifice.module_for(arch)
      step = Edifice.Stateful.jit_step(module, EXLA, client: :host, precision: :highest)
      replay = Edifice.Stateful.jit_replay(module, EXLA, client: :host, precision: :highest)

      assert replay ==
               Edifice.Stateful.jit_replay(module, EXLA, client: :host, precision: :highest)

      initial = Edifice.init_state(arch, params, opts ++ [batch_size: 2])
      {_, carry} = step.(params, initial, Nx.take(x, 0, axis: 1))

      expected =
        Enum.reduce(0..8, carry, fn i, s -> elem(step.(params, s, Nx.take(x, i, axis: 1)), 1) end)

      actual = replay.(params, carry, x)
      split = replay.(params, carry, Nx.slice_along_axis(x, 0, 4, axis: 1))
      split = replay.(params, split, Nx.slice_along_axis(x, 4, 5, axis: 1))
      assert_state_close(actual, expected)
      assert_state_close(split, expected)
      {a, _} = step.(params, actual, Nx.take(x, 0, axis: 1))
      {b, _} = step.(params, expected, Nx.take(x, 0, axis: 1))
      Nx.Testing.assert_all_close(a, b, atol: 1.0e-5, rtol: 1.0e-5)
    end
  end

  defp assert_state_close(%Nx.Tensor{} = a, b),
    do: Nx.Testing.assert_all_close(a, b, atol: 1.0e-5, rtol: 1.0e-5)

  defp assert_state_close(a, b) when is_map(a) do
    assert Map.keys(a) == Map.keys(b)
    Enum.each(a, fn {k, v} -> assert_state_close(v, b[k]) end)
  end
end

defmodule Edifice.Stateful.SelectedReplayTest do
  use ExUnit.Case, async: true
  @moduletag :exla_only

  defp leaves(%Nx.Tensor{} = value), do: [value]

  defp leaves(state),
    do: state |> Enum.sort_by(&elem(&1, 0)) |> Enum.flat_map(fn {_, value} -> leaves(value) end)

  defmodule NestedStep do
    def step(_params, %{layer: %{h: h}} = state, frame),
      do: {frame, %{state | layer: %{h: Nx.add(h, frame)}}}
  end

  test "nested carry, padded row selection and zero count preserve untouched bits" do
    replay = Edifice.Stateful.jit_replay_rows(NestedStep, EXLA, client: :host)
    current = %{layer: %{h: Nx.tensor([[10.0], [20.0], [30.0]])}}
    initial = %{layer: %{h: Nx.tensor([[1.0], [2.0], [3.0]])}}
    frames = {Nx.tensor([[1.0], [2.0], [3.0]]), Nx.tensor([[4.0], [5.0], [6.0]])}
    rows = Nx.tensor([2, 0], type: :s32)
    actual = replay.(%{}, current, initial, frames, rows, Nx.tensor(1, type: :s32))
    assert Nx.to_flat_list(actual.layer.h) == [10.0, 20.0, 12.0]
    actual = replay.(%{}, current, initial, frames, rows, Nx.tensor(0, type: :s32))
    assert Nx.to_binary(actual.layer.h) == Nx.to_binary(current.layer.h)
  end

  for {arch, conv_size} <-
        Enum.map([:gru, :lstm, :min_gru, :mamba, :mamba_ssd, :gated_delta_net], &{&1, 3}) ++
          [{:mamba, 1}, {:mamba_ssd, 1}] do
    test "#{arch}, conv #{conv_size}: tuple history and selected rows match sequential replay" do
      arch = unquote(arch)
      Process.put(:__edifice_force_fallback__, true)

      opts = [
        embed_dim: 6,
        hidden_size: 8,
        num_layers: 2,
        state_size: 3,
        num_heads: 2,
        conv_size: unquote(conv_size),
        window_size: 7,
        dropout: 0.0
      ]

      {init, _} = Axon.build(Edifice.build(arch, opts))
      {frames, _} = Nx.Random.normal(Nx.Random.key(101), shape: {4, 7, 6})
      params = init.(frames, Axon.ModelState.empty())
      initial = Edifice.init_state(arch, params, opts ++ [batch_size: 4])
      module = Edifice.module_for(arch)
      step = Edifice.Stateful.jit_step(module, EXLA, client: :host, precision: :highest)
      replay = Edifice.Stateful.jit_replay(module, EXLA, client: :host, precision: :highest)

      selected =
        Edifice.Stateful.jit_replay_rows(module, EXLA, client: :host, precision: :highest)

      assert selected ==
               Edifice.Stateful.jit_replay_rows(module, EXLA, client: :host, precision: :highest)

      {_, current} = step.(params, initial, Nx.take(frames, 0, axis: 1))
      history = for i <- 0..6, do: Nx.take(frames, i, axis: 1)

      expected =
        Enum.reduce(history, initial, fn frame, carry -> elem(step.(params, carry, frame), 1) end)

      tuple = List.to_tuple(history)
      all = replay.(params, initial, tuple)

      for indices <- [[3, 1], [0, 2]] do
        # Second index is padding: retain it, even though its replay is computed.
        actual =
          selected.(
            params,
            current,
            initial,
            tuple,
            Nx.tensor(indices, type: :s32),
            Nx.tensor(1, type: :s32)
          )

        tensor_actual =
          selected.(
            params,
            current,
            initial,
            frames,
            Nx.tensor(indices, type: :s32),
            Nx.tensor(1, type: :s32)
          )

        for {value, tensor_value, all_value, expected_value, current_value} <-
              Enum.zip(Enum.map([actual, tensor_actual, all, expected, current], &leaves/1)) do
          Nx.Testing.assert_all_close(all_value, expected_value, atol: 1.0e-5, rtol: 1.0e-5)
          Nx.Testing.assert_all_close(value, tensor_value, atol: 1.0e-5, rtol: 1.0e-5)

          for row <- 0..3 do
            if row == hd(indices) do
              Nx.Testing.assert_all_close(Nx.take(value, row), Nx.take(expected_value, row),
                atol: 1.0e-5,
                rtol: 1.0e-5
              )
            else
              assert Nx.to_binary(Nx.take(value, row)) ==
                       Nx.to_binary(Nx.take(current_value, row))
            end
          end
        end
      end
    end
  end
end
