defmodule Edifice.PropertyTest do
  @moduledoc """
  Property-based tests using StreamData to verify architectural invariants
  hold across randomly generated inputs. Tests properties that should be
  true for ALL valid inputs, not just specific examples.

  Strategy C: Catches edge cases that hand-crafted tests miss by exploring
  the input space more broadly.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Edifice.TestHelpers

  # ── Generators ──────────────────────────────────────────────────

  # Generate a bounded float tensor of given shape
  defp bounded_tensor_gen(shape) do
    size = Tuple.product(shape)

    StreamData.list_of(
      StreamData.float(min: -10.0, max: 10.0),
      length: size
    )
    |> StreamData.map(fn vals ->
      Nx.tensor(vals, type: :f32) |> Nx.reshape(shape)
    end)
  end

  # Generate a valid symmetric adjacency matrix
  defp adjacency_gen(batch, nodes) do
    bounded_tensor_gen({batch, nodes, nodes})
    |> StreamData.map(fn adj ->
      # Make symmetric and non-negative
      Nx.add(adj, Nx.transpose(adj, axes: [0, 2, 1]))
      |> Nx.abs()
    end)
  end

  # ── Property: Output Finiteness ────────────────────────────────
  # For bounded inputs, output should always be finite (no NaN/Inf)

  @tag timeout: 120_000
  property "mlp produces finite output for bounded inputs" do
    model = Edifice.build(:mlp, input_size: 8, hidden_sizes: [8])

    check all(input <- bounded_tensor_gen({2, 8}), max_runs: 20) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})
      assert_finite!(output, "mlp property")
    end
  end

  @tag timeout: 120_000
  property "lora produces finite output for bounded inputs" do
    model = Edifice.build(:lora, input_size: 8, output_size: 4, rank: 2)

    check all(input <- bounded_tensor_gen({2, 8}), max_runs: 20) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})
      assert_finite!(output, "lora property")
    end
  end

  @tag timeout: 120_000
  property "deep_sets produces finite output for bounded inputs" do
    model = Edifice.build(:deep_sets, input_dim: 4, output_dim: 3)

    check all(input <- bounded_tensor_gen({2, 6, 4}), max_runs: 20) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})
      assert_finite!(output, "deep_sets property")
    end
  end

  # ── Property: Output Shape Invariance ──────────────────────────
  # Output shape should not depend on input values, only on input shape

  @tag timeout: 120_000
  property "mlp output shape is always {batch, last_hidden}" do
    model = Edifice.build(:mlp, input_size: 8, hidden_sizes: [16, 4])

    check all(input <- bounded_tensor_gen({2, 8}), max_runs: 20) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})
      assert {2, 4} = Nx.shape(output)
    end
  end

  @tag timeout: 120_000
  property "adapter preserves input shape for any bounded input" do
    model = Edifice.build(:adapter, hidden_size: 8)

    check all(input <- bounded_tensor_gen({2, 8}), max_runs: 20) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})
      assert Nx.shape(output) == Nx.shape(input)
    end
  end

  # ── Property: Permutation Invariance ───────────────────────────
  # For set models, shuffling elements should not change output

  @tag timeout: 120_000
  property "deep_sets is permutation invariant for random inputs" do
    model = Edifice.build(:deep_sets, input_dim: 4, output_dim: 3)

    check all(input <- bounded_tensor_gen({1, 6, 4}), max_runs: 15) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      out_orig = predict_fn.(params, %{"input" => input})

      # Reverse the set elements
      perm = Nx.tensor([5, 4, 3, 2, 1, 0])
      input_perm = Nx.take(input, perm, axis: 1)
      out_perm = predict_fn.(params, %{"input" => input_perm})

      diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()
      assert diff < 0.01, "Permutation invariance violated: diff = #{diff}"
    end
  end

  # ── Property: Graph Equivariance ───────────────────────────────
  # Swapping two nodes (in features AND adjacency) should swap outputs

  @tag timeout: 120_000
  property "gcn is permutation equivariant for random graphs" do
    model =
      Edifice.build(:gcn,
        input_dim: 4,
        hidden_size: 4,
        num_classes: 4,
        num_layers: 1,
        dropout: 0.0
      )

    check all(
            nodes <- bounded_tensor_gen({1, 4, 4}),
            adj <- adjacency_gen(1, 4),
            max_runs: 10
          ) do
      input_map = %{"nodes" => nodes, "adjacency" => adj}
      {predict_fn, params} = build_and_init(model, input_map)
      out_orig = predict_fn.(params, input_map)

      # Swap nodes 0 and 1
      perm = Nx.tensor([1, 0, 2, 3])
      nodes_p = Nx.take(nodes, perm, axis: 1)
      adj_p = Nx.take(Nx.take(adj, perm, axis: 1), perm, axis: 2)
      out_perm = predict_fn.(params, %{"nodes" => nodes_p, "adjacency" => adj_p})

      # Permute original output to compare
      out_orig_p = Nx.take(out_orig, perm, axis: 1)

      diff = Nx.mean(Nx.abs(Nx.subtract(out_perm, out_orig_p))) |> Nx.to_number()
      assert diff < 0.05, "GCN equivariance violated: diff = #{diff}"
    end
  end

  # ── Property: Non-Collapse ─────────────────────────────────────
  # Different inputs should produce different outputs (model not degenerate)

  @tag timeout: 120_000
  property "mlp does not collapse distinct inputs to same output" do
    model = Edifice.build(:mlp, input_size: 8, hidden_sizes: [8])

    check all(
            input1 <- bounded_tensor_gen({1, 8}),
            input2 <- bounded_tensor_gen({1, 8}),
            max_runs: 20
          ) do
      {predict_fn, params} = build_and_init(model, %{"input" => input1})
      out1 = predict_fn.(params, %{"input" => input1})
      out2 = predict_fn.(params, %{"input" => input2})

      input_diff = Nx.mean(Nx.abs(Nx.subtract(input1, input2))) |> Nx.to_number()

      # Only check non-collapse when inputs are meaningfully different
      if input_diff > 1.0 do
        output_diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()

        assert output_diff > 1.0e-8,
               "MLP collapsed: output_diff=#{output_diff} for input_diff=#{input_diff}"
      end
    end
  end

  # ── Property: Evidential Positivity ────────────────────────────
  # Dirichlet alpha parameters must be positive for all valid inputs

  @tag timeout: 120_000
  property "evidential always produces positive alpha parameters" do
    model = Edifice.build(:evidential, input_size: 8, num_classes: 4)

    check all(input <- bounded_tensor_gen({2, 8}), max_runs: 20) do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})

      min_val = Nx.reduce_min(output) |> Nx.to_number()
      assert min_val > 0, "Evidential produced non-positive alpha: #{min_val}"
    end
  end

  # ── Property: GCN Finite on Random Graphs ──────────────────────
  # GCN should handle arbitrary graph structures without NaN/Inf

  @tag timeout: 120_000
  property "gcn produces finite output for random graphs" do
    model =
      Edifice.build(:gcn,
        input_dim: 4,
        hidden_size: 4,
        num_classes: 4,
        num_layers: 1,
        dropout: 0.0
      )

    check all(
            nodes <- bounded_tensor_gen({1, 4, 4}),
            adj <- adjacency_gen(1, 4),
            max_runs: 15
          ) do
      input_map = %{"nodes" => nodes, "adjacency" => adj}
      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "gcn random graph")
    end
  end

  # ── Property: Sequence Model Finiteness ────────────────────────
  # Sequence models should handle bounded random sequences

  @tag timeout: 120_000
  property "mamba produces finite output for random sequences" do
    model =
      Edifice.build(:mamba,
        embed_dim: 8,
        hidden_size: 8,
        state_size: 4,
        num_layers: 1,
        seq_len: 4,
        window_size: 4
      )

    check all(input <- bounded_tensor_gen({1, 4, 8}), max_runs: 15) do
      {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert_finite!(output, "mamba random seq")
    end
  end

  @tag timeout: 120_000
  property "lstm produces finite output for random sequences" do
    model =
      Edifice.build(:lstm,
        embed_dim: 8,
        hidden_size: 8,
        num_layers: 1,
        seq_len: 4,
        window_size: 4
      )

    check all(input <- bounded_tensor_gen({1, 4, 8}), max_runs: 15) do
      {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert_finite!(output, "lstm random seq")
    end
  end

  # ── Property: Memory Network Finiteness ────────────────────────

  @tag timeout: 120_000
  property "memory_network produces finite output for random queries and memories" do
    model = Edifice.build(:memory_network, input_dim: 8, output_dim: 4, num_hops: 2)

    check all(
            query <- bounded_tensor_gen({1, 8}),
            memories <- bounded_tensor_gen({1, 4, 8}),
            max_runs: 15
          ) do
      input_map = %{"query" => query, "memories" => memories}
      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "memory_network random")
    end
  end

  # ── Property: Contrastive Non-Collapse ─────────────────────────

  @tag timeout: 120_000
  @tag :slow
  property "simclr produces different outputs for different inputs" do
    model = Edifice.build(:simclr, encoder_dim: 8, projection_dim: 4)

    check all(
            input1 <- bounded_tensor_gen({1, 8}),
            input2 <- bounded_tensor_gen({1, 8}),
            max_runs: 15
          ) do
      {predict_fn, params} = build_and_init(model, %{"features" => input1})
      out1 = predict_fn.(params, %{"features" => input1})
      out2 = predict_fn.(params, %{"features" => input2})

      assert_finite!(out1, "simclr")
      assert_finite!(out2, "simclr")

      input_diff = Nx.mean(Nx.abs(Nx.subtract(input1, input2))) |> Nx.to_number()

      if input_diff > 1.0 do
        output_diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
        assert output_diff > 1.0e-8, "SimCLR collapsed"
      end
    end
  end
end
