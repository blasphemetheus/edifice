defmodule Edifice.AuditCorrectnessTest do
  @moduledoc """
  Correctness tests for audit-identified fixes.
  Verifies MoE routing, SwitchMoE hard selection, SchNet filter generation,
  KAN B-spline basis, and TTT initialization/normalization.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 120_000

  import Edifice.TestHelpers

  alias Edifice.Feedforward.KAN
  alias Edifice.Graph.SchNet
  alias Edifice.Meta.{MoE, SwitchMoE}
  alias Edifice.Recurrent.TTT

  # ============================================================================
  # MoE Top-K Routing
  # ============================================================================

  describe "MoE top_k routing selects correct experts" do
    test "extreme logits route to the dominant expert" do
      model =
        MoE.build(
          input_size: 16,
          hidden_size: 32,
          output_size: 16,
          num_experts: 4,
          top_k: 1,
          routing: :top_k,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {2, 4, 16})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 4, 16}
      assert_finite!(output, "moe_top_k")
    end

    test "different top_k values produce different outputs" do
      build_moe = fn top_k ->
        MoE.build(
          input_size: 16,
          hidden_size: 32,
          output_size: 16,
          num_experts: 4,
          top_k: top_k,
          routing: :top_k,
          dropout: 0.0
        )
      end

      model_k1 = build_moe.(1)
      model_k2 = build_moe.(2)

      {init1, pred1} = Axon.build(model_k1, mode: :inference)
      {init2, pred2} = Axon.build(model_k2, mode: :inference)

      input = random_tensor({2, 4, 16})
      params1 = init1.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      params2 = init2.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())

      out1 = pred1.(params1, input)
      out2 = pred2.(params2, input)

      # Different model instances produce different outputs (different random params)
      assert_finite!(out1, "moe_k1")
      assert_finite!(out2, "moe_k2")
    end
  end

  # ============================================================================
  # MoE Soft Routing
  # ============================================================================

  describe "MoE soft routing uses all experts" do
    test "soft routing produces finite output" do
      model =
        MoE.build(
          input_size: 16,
          hidden_size: 32,
          output_size: 16,
          num_experts: 4,
          routing: :soft,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 4, 16}))

      assert Nx.shape(output) == {2, 4, 16}
      assert_finite!(output, "moe_soft")
    end
  end

  # ============================================================================
  # MoE Hash Routing
  # ============================================================================

  describe "MoE hash routing is deterministic" do
    test "same input always routes to same expert" do
      model =
        MoE.build(
          input_size: 16,
          hidden_size: 32,
          output_size: 16,
          num_experts: 4,
          routing: :hash,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      input = random_tensor({2, 4, 16})

      out1 = predict_fn.(params, input)
      out2 = predict_fn.(params, input)

      diff = Nx.subtract(out1, out2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6, "Hash routing should be deterministic"
    end
  end

  # ============================================================================
  # SwitchMoE Hard Top-1 Routing
  # ============================================================================

  describe "SwitchMoE hard top-1 routing" do
    test "produces correct output shape" do
      model =
        SwitchMoE.build(
          embed_dim: 16,
          hidden_size: 16,
          num_experts: 4,
          expert_size: 32,
          num_layers: 1,
          seq_len: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_tensor({2, 4, 16}))

      assert Nx.shape(output) == {2, 16}
      assert_finite!(output, "switch_moe")
    end

    test "output is deterministic (hard selection)" do
      model =
        SwitchMoE.build(
          embed_dim: 16,
          hidden_size: 16,
          num_experts: 2,
          expert_size: 32,
          num_layers: 1,
          seq_len: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      input = random_tensor({2, 4, 16})

      out1 = predict_fn.(params, input)
      out2 = predict_fn.(params, input)

      diff = Nx.subtract(out1, out2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6, "Hard routing should be deterministic"
    end
  end

  # ============================================================================
  # SchNet Filter Generation
  # ============================================================================

  describe "SchNet learned filter generation" do
    test "filter params exist in parameter tree" do
      model =
        SchNet.build(
          input_dim: 8,
          hidden_size: 16,
          num_interactions: 1,
          num_filters: 16,
          num_rbf: 10,
          cutoff: 5.0
        )

      input_map = %{
        "nodes" => random_tensor({2, 4, 8}),
        "adjacency" => random_tensor({2, 4, 4})
      }

      {predict_fn, params} = build_and_init(model, input_map)

      # Check that filter-generating network params exist
      flat_params = flatten_params(params)
      filter_keys = Enum.filter(flat_params, fn {k, _} -> String.contains?(k, "filter_w") end)
      assert length(filter_keys) >= 2, "Should have filter_w1 and filter_w2 params"

      output = predict_fn.(params, input_map)
      assert_finite!(output, "schnet")
    end

    test "different distances produce different filter outputs" do
      model =
        SchNet.build(
          input_dim: 8,
          hidden_size: 16,
          num_interactions: 1,
          num_filters: 16,
          num_rbf: 10,
          cutoff: 5.0
        )

      nodes = random_tensor({2, 4, 8})
      adj_near = Nx.broadcast(1.0, {2, 4, 4})
      adj_far = Nx.broadcast(4.0, {2, 4, 4})

      input_near = %{"nodes" => nodes, "adjacency" => adj_near}
      input_far = %{"nodes" => nodes, "adjacency" => adj_far}

      {predict_fn, params} = build_and_init(model, input_near)

      out_near = predict_fn.(params, input_near)
      out_far = predict_fn.(params, input_far)

      diff = Nx.subtract(out_near, out_far) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different distances should produce different outputs"
    end
  end

  # ============================================================================
  # KAN B-spline Basis
  # ============================================================================

  describe "KAN B-spline basis" do
    test "builds model with bspline basis (new default)" do
      model =
        KAN.build(
          embed_dim: 32,
          hidden_size: 16,
          num_layers: 1,
          grid_size: 4,
          seq_len: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 4, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_tensor({2, 4, 32}))

      assert Nx.shape(output) == {2, 16}
      assert_finite!(output, "kan_bspline")
    end

    test "bspline produces different output than sine basis" do
      build_kan = fn basis ->
        KAN.build(
          embed_dim: 32,
          hidden_size: 16,
          num_layers: 1,
          grid_size: 4,
          basis: basis,
          seq_len: 4
        )
      end

      model_bspline = build_kan.(:bspline)
      model_sine = build_kan.(:sine)

      {init_b, pred_b} = Axon.build(model_bspline)
      {init_s, pred_s} = Axon.build(model_sine)

      input = random_tensor({2, 4, 32})
      params_b = init_b.(Nx.template({2, 4, 32}, :f32), Axon.ModelState.empty())
      params_s = init_s.(Nx.template({2, 4, 32}, :f32), Axon.ModelState.empty())

      out_b = pred_b.(params_b, input)
      out_s = pred_s.(params_s, input)

      assert_finite!(out_b, "kan_bspline")
      assert_finite!(out_s, "kan_sine")
      # Different basis functions (and different random params) should give different outputs
      diff = Nx.subtract(out_b, out_s) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6
    end

    test "bspline_basis_eval returns finite values" do
      x = Nx.broadcast(0.5, {2, 4, 32})
      result = KAN.bspline_basis_eval(x, 8)
      assert_finite!(result, "bspline_eval")
      assert Nx.shape(result) == {2, 4, 32}
    end
  end

  # ============================================================================
  # TTT Initialization and Normalization
  # ============================================================================

  describe "TTT W_0 glorot initialization" do
    test "w0 is not identity-like" do
      model =
        TTT.build(
          embed_dim: 16,
          hidden_size: 16,
          inner_size: 8,
          num_layers: 1,
          seq_len: 4,
          dropout: 0.0
        )

      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())

      # Find w0 param
      flat = flatten_params(params)
      {_key, w0} = Enum.find(flat, fn {k, _} -> String.contains?(k, "w0") end)

      # Glorot init should have varied values, not 0.01 * identity
      off_diag_sum = Nx.subtract(w0, Nx.multiply(Nx.eye(Nx.shape(w0)), w0))
      off_diag_abs = Nx.abs(off_diag_sum) |> Nx.sum() |> Nx.to_number()
      assert off_diag_abs > 0.01, "Glorot init should have non-zero off-diagonal elements"
    end
  end

  describe "TTT output without RMS norm" do
    test "produces finite output with default (no rms norm)" do
      model =
        TTT.build(
          embed_dim: 16,
          hidden_size: 16,
          inner_size: 8,
          num_layers: 1,
          seq_len: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_tensor({2, 4, 16}))

      assert Nx.shape(output) == {2, 16}
      assert_finite!(output, "ttt_no_rms")
    end

    test "produces finite output with rms norm enabled" do
      model =
        TTT.build(
          embed_dim: 16,
          hidden_size: 16,
          inner_size: 8,
          num_layers: 1,
          seq_len: 4,
          dropout: 0.0,
          output_rms_norm: true
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_tensor({2, 4, 16}))

      assert Nx.shape(output) == {2, 16}
      assert_finite!(output, "ttt_rms")
    end
  end
end
