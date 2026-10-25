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
