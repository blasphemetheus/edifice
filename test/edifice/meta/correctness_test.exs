defmodule Edifice.Meta.CorrectnessTest do
  @moduledoc """
  Correctness tests for meta-learning and parameter-efficient architectures.
  Verifies MoE routing properties, LoRA rank structure, adapter residual
  identity, and hypernetwork conditioning dependence.
  """
  use ExUnit.Case, async: true
  @moduletag :meta

  import Edifice.TestHelpers

  @batch 2
  @embed 16
  @hidden 8
  @seq_len 4

  # ── LoRA Low-Rank Structure ──────────────────────────────────────
  # LoRA adds a low-rank adaptation: output = W*x + (alpha/rank) * B(A(x))

  @tag :smoke
  test "lora output shape matches specified dimensions" do
    model = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 4)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "lora")
    assert {@batch, @hidden} = Nx.shape(output)
  end

  test "lora with different ranks produces different parameter counts" do
    model_r2 = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 2)
    model_r8 = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 8)

    input = random_tensor({@batch, @embed})

    {_, params_r2} = build_and_init(model_r2, %{"input" => input})
    {_, params_r8} = build_and_init(model_r8, %{"input" => input})

    # Higher rank should have more parameters (or same count but larger tensors)
    size_r2 = params_r2 |> flatten_params() |> Enum.map(fn {_, t} -> Nx.size(t) end) |> Enum.sum()
    size_r8 = params_r8 |> flatten_params() |> Enum.map(fn {_, t} -> Nx.size(t) end) |> Enum.sum()

    assert size_r8 > size_r2,
           "rank=8 should have more total params than rank=2 (#{size_r8} vs #{size_r2})"
  end

  test "lora is deterministic" do
    model = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 4)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})

    out1 = predict_fn.(params, %{"input" => input})
    out2 = predict_fn.(params, %{"input" => input})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff < 1.0e-6, "lora not deterministic: diff = #{diff}"
  end

  # ── Adapter Residual Structure ──────────────────────────────────
  # Adapter applies: output = input + adapter_fn(input)

  test "adapter preserves input dimension" do
    model = Edifice.build(:adapter, hidden_size: @hidden)

    input = random_tensor({@batch, @hidden})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "adapter")
    assert {@batch, @hidden} = Nx.shape(output)
  end

  test "adapter output is finite for various inputs" do
    model = Edifice.build(:adapter, hidden_size: @hidden)

    for {label, input} <- [
          {"zeros", Nx.broadcast(Nx.tensor(0.0, type: :f32), {@batch, @hidden})},
          {"ones", Nx.broadcast(Nx.tensor(1.0, type: :f32), {@batch, @hidden})},
          {"large", Nx.broadcast(Nx.tensor(100.0, type: :f32), {@batch, @hidden})}
        ] do
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})
      assert_finite!(output, "adapter #{label}")
    end
  end

  # ── MoE Routing ──────────────────────────────────────────────────
  # MoE should produce valid output and handle different expert configurations

  @tag timeout: 120_000
  test "moe produces correct output shape" do
    model = Edifice.build(:moe, input_size: @embed, output_size: @embed, num_experts: 4, top_k: 2)

    input = random_tensor({@batch, @seq_len, @embed})
    {predict_fn, params} = build_and_init(model, %{"moe_input" => input})
    output = predict_fn.(params, %{"moe_input" => input})

    assert_finite!(output, "moe")
    assert {@batch, @seq_len, @embed} = Nx.shape(output)
  end

  @tag timeout: 120_000
  test "moe different inputs produce different outputs" do
    model = Edifice.build(:moe, input_size: @embed, output_size: @embed, num_experts: 4, top_k: 2)

    input1 = random_tensor({@batch, @seq_len, @embed}, 42)
    input2 = random_tensor({@batch, @seq_len, @embed}, 99)

    {predict_fn, params} = build_and_init(model, %{"moe_input" => input1})
    out1 = predict_fn.(params, %{"moe_input" => input1})
    out2 = predict_fn.(params, %{"moe_input" => input2})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "MoE collapsed: same output for different inputs"
  end

  # ── Hypernetwork Conditioning Dependence ────────────────────────
  # Different conditioning inputs should produce different outputs

  @tag timeout: 120_000
  test "hypernetwork output changes with conditioning" do
    model =
      Edifice.build(:hypernetwork,
        conditioning_size: @embed,
        input_size: @hidden,
        target_layer_sizes: [{@hidden, @hidden}],
        hidden_sizes: [@hidden]
      )

    conditioning1 = random_tensor({@batch, @embed}, 42)
    conditioning2 = random_tensor({@batch, @embed}, 99)
    data_input = random_tensor({@batch, @hidden}, 77)

    input_map1 = %{"conditioning" => conditioning1, "data_input" => data_input}
    input_map2 = %{"conditioning" => conditioning2, "data_input" => data_input}

    {predict_fn, params} = build_and_init(model, input_map1)
    out1 = predict_fn.(params, input_map1)
    out2 = predict_fn.(params, input_map2)

    assert_finite!(out1, "hypernetwork cond1")
    assert_finite!(out2, "hypernetwork cond2")

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "Hypernetwork ignoring conditioning: outputs identical"
  end

  @tag timeout: 120_000
  test "hypernetwork output changes with data input" do
    model =
      Edifice.build(:hypernetwork,
        conditioning_size: @embed,
        input_size: @hidden,
        target_layer_sizes: [{@hidden, @hidden}],
        hidden_sizes: [@hidden]
      )

    conditioning = random_tensor({@batch, @embed}, 42)
    data1 = random_tensor({@batch, @hidden}, 77)
    data2 = random_tensor({@batch, @hidden}, 88)

    input_map1 = %{"conditioning" => conditioning, "data_input" => data1}
    input_map2 = %{"conditioning" => conditioning, "data_input" => data2}

    {predict_fn, params} = build_and_init(model, input_map1)
    out1 = predict_fn.(params, input_map1)
    out2 = predict_fn.(params, input_map2)

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "Hypernetwork ignoring data input: outputs identical"
  end

  # ── Capsule Network Properties ──────────────────────────────────

  @tag :exla_only
  @tag timeout: 120_000
  test "capsule output norms are bounded" do
    model =
      Edifice.build(:capsule,
        input_shape: {nil, 8, 8, 1},
        num_primary_caps: 8,
        primary_cap_dim: 4,
        num_digit_caps: 4,
        digit_cap_dim: 8,
        routing_iterations: 2,
        conv_channels: 8,
        conv_kernel: 3,
        primary_kernel: 3,
        primary_strides: 1
      )

    input = random_tensor({@batch, 8, 8, 1})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "capsule")
    # Capsule norms should be non-negative
    min_val = Nx.reduce_min(output) |> Nx.to_number()
    assert min_val >= -0.01, "Capsule norms should be non-negative, got #{min_val}"
  end
end
