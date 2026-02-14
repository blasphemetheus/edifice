# Scaling Profile: How do architectures scale with key dimensions?
#
# Compares Attention vs SSM scaling with sequence length,
# and GCN scaling with graph size.
#
# Usage:
#   mix run bench/scaling_profile.exs
#
# Requires EXLA compiled (EXLA_TARGET=host for CPU, or CUDA).

Nx.default_backend(EXLA.Backend)

defmodule ScalingProfile do
  @batch 4
  @hidden 64

  defp rand(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  def run do
    IO.puts("=" |> String.duplicate(70))
    IO.puts("Edifice Scaling Profile — EXLA Backend")
    IO.puts("=" |> String.duplicate(70))
    IO.puts("")

    attention_vs_ssm()
    IO.puts("")
    gcn_scaling()
  end

  # ── Attention vs Mamba vs S4 across sequence lengths ──

  defp attention_vs_ssm do
    IO.puts("## Attention vs Mamba vs S4 — Sequence Length Scaling")
    IO.puts("batch=#{@batch}, hidden=#{@hidden}")
    IO.puts("-" |> String.duplicate(50))

    seq_lengths = [16, 32, 64, 128, 256]

    header =
      "  #{String.pad_trailing("seq_len", 10)}" <>
        "#{String.pad_trailing("MultiHead", 15)}" <>
        "#{String.pad_trailing("LinearTfm", 15)}" <>
        "#{String.pad_trailing("Mamba", 15)}" <>
        "S4"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 65))

    for seq_len <- seq_lengths do
      input = rand({@batch, seq_len, @hidden})
      template = Nx.template({@batch, seq_len, @hidden}, :f32)

      results =
        for {name, builder} <- [
              {"MultiHead", fn ->
                 Edifice.Attention.MultiHead.build(
                   embed_size: @hidden,
                   hidden_size: @hidden,
                   num_heads: 4,
                   head_dim: 16,
                   num_layers: 2,
                   window_size: seq_len
                 )
               end},
              {"LinearTfm", fn ->
                 Edifice.Attention.LinearTransformer.build(
                   embed_size: @hidden,
                   hidden_size: @hidden,
                   num_heads: 4,
                   num_layers: 2,
                   window_size: seq_len
                 )
               end},
              {"Mamba", fn ->
                 Edifice.SSM.Mamba.build(
                   embed_size: @hidden,
                   hidden_size: @hidden,
                   state_size: 16,
                   num_layers: 2,
                   window_size: seq_len
                 )
               end},
              {"S4", fn ->
                 Edifice.SSM.S4.build(
                   embed_size: @hidden,
                   hidden_size: @hidden,
                   state_size: 16,
                   num_layers: 2,
                   window_size: seq_len
                 )
               end}
            ] do
          model = builder.()
          {init_fn, predict_fn} = Axon.build(model)
          params = init_fn.(template, Axon.ModelState.empty())

          # Warm up (3 iters to ensure EXLA compilation is done)
          for _ <- 1..3, do: predict_fn.(params, input)

          # Time 10 iterations
          {total_us, _} =
            :timer.tc(fn ->
              for _ <- 1..10, do: predict_fn.(params, input)
            end)

          avg_ms = total_us / 10 / 1_000
          {name, avg_ms}
        end

      formatted =
        "  #{String.pad_trailing(Integer.to_string(seq_len), 10)}" <>
          Enum.map_join(results, "", fn {_name, ms} ->
            String.pad_trailing("#{Float.round(ms, 2)} ms", 15)
          end)

      IO.puts(formatted)
    end
  end

  # ── GCN scaling with number of nodes ──

  defp gcn_scaling do
    IO.puts("## GCN — Graph Size Scaling")
    IO.puts("batch=#{@batch}, input_dim=8, hidden=[32,32]")
    IO.puts("-" |> String.duplicate(50))

    IO.puts(
      "  #{String.pad_trailing("nodes", 10)}#{String.pad_trailing("inference", 15)}compile"
    )

    IO.puts("  " <> String.duplicate("-", 40))

    node_counts = [16, 32, 64, 128, 256]

    for num_nodes <- node_counts do
      model =
        Edifice.Graph.GCN.build(
          input_dim: 8,
          hidden_dims: [32, 32],
          num_classes: 10
        )

      nodes = rand({@batch, num_nodes, 8})
      adj = Nx.eye(num_nodes) |> Nx.broadcast({@batch, num_nodes, num_nodes})
      input = %{"nodes" => nodes, "adjacency" => adj}

      template = %{
        "nodes" => Nx.template({@batch, num_nodes, 8}, :f32),
        "adjacency" => Nx.template({@batch, num_nodes, num_nodes}, :f32)
      }

      {compile_us, {init_fn, predict_fn}} =
        :timer.tc(fn -> Axon.build(model) end)

      params = init_fn.(template, Axon.ModelState.empty())

      # Warm up
      predict_fn.(params, input)

      # Time 5 iterations
      {total_us, _} =
        :timer.tc(fn ->
          for _ <- 1..5, do: predict_fn.(params, input)
        end)

      avg_ms = total_us / 5 / 1_000
      compile_ms = compile_us / 1_000

      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(num_nodes), 10)}" <>
          "#{String.pad_trailing("#{Float.round(avg_ms, 2)} ms", 15)}" <>
          "#{Float.round(compile_ms, 1)} ms"
      )
    end
  end
end

ScalingProfile.run()
