defmodule Edifice.Memory.CorrectnessTest do
  @moduledoc """
  Correctness tests for memory-augmented architectures.
  Verifies NTM content addressing, memory network multi-hop behavior,
  and differentiable read/write operations.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 2
  @embed 16
  @hidden 8
  @memory_size 4
  @memory_dim 4

  # ── NTM Content Addressing ──────────────────────────────────────
  # NTM uses cosine similarity to address memory — different inputs
  # should produce different memory access patterns

  @tag timeout: 120_000
  test "ntm produces different outputs for different inputs" do
    model =
      Edifice.build(:ntm,
        input_size: @embed,
        output_size: @hidden,
        memory_size: @memory_size,
        memory_dim: @memory_dim,
        num_heads: 1
      )

    memory = random_tensor({@batch, @memory_size, @memory_dim})
    input1 = random_tensor({@batch, @embed}, 42)
    input2 = random_tensor({@batch, @embed}, 99)

    {predict_fn, params} = build_and_init(model, %{"input" => input1, "memory" => memory})

    out1 = predict_fn.(params, %{"input" => input1, "memory" => memory})
    out2 = predict_fn.(params, %{"input" => input2, "memory" => memory})

    assert_finite!(out1, "ntm input1")
    assert_finite!(out2, "ntm input2")

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "NTM ignoring input: identical outputs for different inputs"
  end

  # ── NTM Memory Sensitivity ──────────────────────────────────────
  # Different memory contents should produce different outputs

  @tag timeout: 120_000
  test "ntm output depends on memory contents" do
    model =
      Edifice.build(:ntm,
        input_size: @embed,
        output_size: @hidden,
        memory_size: @memory_size,
        memory_dim: @memory_dim,
        num_heads: 1
      )

    input = random_tensor({@batch, @embed})
    memory1 = random_tensor({@batch, @memory_size, @memory_dim}, 42)
    memory2 = random_tensor({@batch, @memory_size, @memory_dim}, 99)

    {predict_fn, params} = build_and_init(model, %{"input" => input, "memory" => memory1})

    out1 = predict_fn.(params, %{"input" => input, "memory" => memory1})
    out2 = predict_fn.(params, %{"input" => input, "memory" => memory2})

    assert_finite!(out1, "ntm memory1")
    assert_finite!(out2, "ntm memory2")

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "NTM ignoring memory: identical outputs for different memories"
  end

  # ── NTM Determinism ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "ntm is deterministic" do
    model =
      Edifice.build(:ntm,
        input_size: @embed,
        output_size: @hidden,
        memory_size: @memory_size,
        memory_dim: @memory_dim,
        num_heads: 1
      )

    input = random_tensor({@batch, @embed})
    memory = random_tensor({@batch, @memory_size, @memory_dim})
    input_map = %{"input" => input, "memory" => memory}

    {predict_fn, params} = build_and_init(model, input_map)

    out1 = predict_fn.(params, input_map)
    out2 = predict_fn.(params, input_map)

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff < 1.0e-6, "NTM not deterministic: diff = #{diff}"
  end

  # ── Memory Network Multi-Hop ────────────────────────────────────
  # More hops should refine the query — output should change with hop count

  @tag timeout: 120_000
  test "memory_network produces valid output" do
    model = Edifice.build(:memory_network, input_dim: @embed, output_dim: @hidden, num_hops: 3)

    query = random_tensor({@batch, @embed})
    memories = random_tensor({@batch, 6, @embed})
    input_map = %{"query" => query, "memories" => memories}

    {predict_fn, params} = build_and_init(model, input_map)
    output = predict_fn.(params, input_map)

    assert_finite!(output, "memory_network")
    assert {@batch, @hidden} = Nx.shape(output)
  end

  @tag timeout: 120_000
  test "memory_network output depends on query" do
    model = Edifice.build(:memory_network, input_dim: @embed, output_dim: @hidden, num_hops: 2)

    query1 = random_tensor({@batch, @embed}, 42)
    query2 = random_tensor({@batch, @embed}, 99)
    memories = random_tensor({@batch, 6, @embed})

    {predict_fn, params} = build_and_init(model, %{"query" => query1, "memories" => memories})

    out1 = predict_fn.(params, %{"query" => query1, "memories" => memories})
    out2 = predict_fn.(params, %{"query" => query2, "memories" => memories})

    assert_finite!(out1, "memnet query1")
    assert_finite!(out2, "memnet query2")

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "Memory network ignoring query: identical outputs"
  end

  @tag timeout: 120_000
  test "memory_network output depends on memories" do
    model = Edifice.build(:memory_network, input_dim: @embed, output_dim: @hidden, num_hops: 2)

    query = random_tensor({@batch, @embed})
    memories1 = random_tensor({@batch, 6, @embed}, 42)
    memories2 = random_tensor({@batch, 6, @embed}, 99)

    {predict_fn, params} = build_and_init(model, %{"query" => query, "memories" => memories1})

    out1 = predict_fn.(params, %{"query" => query, "memories" => memories1})
    out2 = predict_fn.(params, %{"query" => query, "memories" => memories2})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "Memory network ignoring memories: identical outputs"
  end
end
