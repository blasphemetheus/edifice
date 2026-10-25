defmodule Edifice.Stateful.NewRecurrencesTest do
  use ExUnit.Case, async: true
  import Edifice.StatefulCase

  setup do
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  for conv <- [false, true], size <- [1, 3] do
    @opts [
      embed_dim: 6,
      hidden_size: 8,
      num_heads: 2,
      num_layers: 2,
      use_short_conv: conv,
      conv_size: size,
      dropout: 0.0
    ]
    test "GatedDeltaNet prefix and rollback, conv=#{conv}, size=#{size}" do
      assert_step_matches_forward(:gated_delta_net, @opts, seq_len: 7, batch: 2)
      assert_rollback_deterministic(:gated_delta_net, @opts, seq_len: 8, rollback_at: 3, batch: 2)
    end
  end

  for training <- [false, true], structured <- [false, true] do
    @opts [
      embed_dim: 6,
      hidden_size: 8,
      state_size: 4,
      num_layers: 2,
      conv_size: 3,
      chunk_size: 3,
      dropout: 0.0,
      training_mode: training,
      structured_mask: structured
    ]
    test "Mamba SSD prefix/chunk boundary/rollback, training=#{training}, structured=#{structured}" do
      # Prefixes 1..8 exercise single-token, matmul, full chunks and remainder.
      assert_step_matches_forward(:mamba_ssd, @opts, seq_len: 8, batch: 2)
      assert_rollback_deterministic(:mamba_ssd, @opts, seq_len: 8, rollback_at: 3, batch: 2)
    end
  end

  for arch <- [:gated_delta_net, :mamba_ssd] do
    @arch arch
    @opts [
      embed_dim: 6,
      hidden_size: 8,
      state_size: 4,
      num_heads: 2,
      num_layers: 2,
      conv_size: 3,
      chunk_size: 3,
      dropout: 0.0
    ]
    @tag :exla_only
    test "#{arch} compiles as one host step and replays exactly" do
      Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)

      assert_step_matches_forward(@arch, @opts,
        seq_len: 8,
        step_opts: [compiler: EXLA],
        atol: 1.0e-4
      )

      assert_rollback_deterministic(@arch, @opts,
        seq_len: 8,
        rollback_at: 3,
        step_opts: [compiler: EXLA]
      )
    end
  end

  test "Mamba SSD pointwise convolution needs no history" do
    assert_step_matches_forward(
      :mamba_ssd,
      [
        embed_dim: 8,
        hidden_size: 8,
        state_size: 4,
        num_layers: 1,
        conv_size: 1,
        chunk_size: 3,
        dropout: 0.0
      ],
      seq_len: 6
    )
  end

  for arch <- [:gated_delta_net, :mamba_ssd] do
    @arch arch
    test "#{arch} stepped parameter gradients match the full model" do
      opts = [
        embed_dim: 3,
        hidden_size: 4,
        state_size: 2,
        num_heads: 2,
        num_layers: 1,
        conv_size: 2,
        chunk_size: 3,
        training_mode: true,
        dropout: 0.0
      ]

      {params, predict, x} = build_forward(@arch, opts, 5, 1, 42)
      state = Edifice.init_state(@arch, params, opts)
      # Weighted output avoids the zero-sum invariant of final LayerNorm.
      loss = fn out -> Nx.sum(Nx.multiply(out, Nx.add(Nx.iota(Nx.shape(out)), 1.0))) end
      full_grad = Nx.Defn.grad(params, fn p -> loss.(predict.(p, x)) end)

      differentiate_steps =
        Nx.Defn.jit(fn p, frames, initial ->
          Nx.Defn.grad(p, fn weights ->
            {out, _} =
              Enum.reduce(0..4, {nil, initial}, fn t, {_, carry} ->
                Edifice.step(@arch, weights, carry, Nx.take(frames, t, axis: 1))
              end)

            loss.(out)
          end)
        end)

      step_grad = differentiate_steps.(params, x, state)

      for {layer, tensors} <- full_grad.data, {name, expected} <- tensors do
        Nx.Testing.assert_all_close(step_grad.data[layer][name], expected,
          atol: 1.0e-4,
          rtol: 1.0e-4
        )
      end
    end
  end

  test "Mamba SSD crosses the default 32-frame chunk boundary" do
    assert_step_matches_forward(
      :mamba_ssd,
      [embed_dim: 4, hidden_size: 4, state_size: 2, num_layers: 1, conv_size: 2, dropout: 0.0],
      seq_len: 35
    )
  end
end
