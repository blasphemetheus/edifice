defmodule Edifice.Meta.MoECoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Meta.MoE.
  Targets uncovered code paths: build_block, build_moe_backbone,
  expert types (glu, mamba), num_experts variants (2, 8, generic),
  embed_dim == hidden_size branch, dropout=0 branches.
  """
  use ExUnit.Case, async: true

  alias Edifice.Meta.MoE

  @input_size 16
  @batch_size 2
  @seq_len 4

  # ============================================================================
  # Expert Types
  # ============================================================================

  describe "GLU expert type" do
    test "build with expert_type :glu produces correct output shape" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          top_k: 1,
          routing: :top_k,
          expert_type: :glu,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "Mamba expert type" do
    @tag :slow
    test "build with expert_type :mamba produces correct output shape" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          top_k: 1,
          routing: :top_k,
          expert_type: :mamba,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  # ============================================================================
  # Expert Count Variants (stack_fn branches)
  # ============================================================================

  describe "num_experts = 2 (2-arity stack_fn)" do
    test "builds and produces correct output" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          top_k: 1,
          routing: :top_k,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  describe "num_experts = 8 (8-arity stack_fn)" do
    @tag :slow
    test "builds and produces correct output" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 8,
          top_k: 2,
          routing: :top_k,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  describe "num_experts = 4 with soft routing" do
    test "builds and produces correct output with 4-expert soft routing" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 4,
          routing: :soft,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  # ============================================================================
  # build_block/2 (pre-norm + MoE + residual)
  # ============================================================================

  describe "build_block/2" do
    test "produces correct output shape with residual" do
      input = Axon.input("block_input", shape: {nil, nil, @input_size})

      block =
        MoE.build_block(input,
          hidden_size: @input_size,
          num_experts: 2,
          top_k: 1,
          dropout: 0.0,
          name: "test_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"block_input" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, %{"block_input" => input_data})

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "with dropout > 0" do
      input = Axon.input("block_input", shape: {nil, nil, @input_size})

      block =
        MoE.build_block(input,
          hidden_size: @input_size,
          num_experts: 2,
          top_k: 1,
          dropout: 0.1,
          name: "test_block_drop"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"block_input" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, %{"block_input" => input_data})

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "with dropout = 0 skips dropout layer" do
      input = Axon.input("block_input", shape: {nil, nil, @input_size})

      block =
        MoE.build_block(input,
          hidden_size: @input_size,
          num_experts: 4,
          top_k: 2,
          dropout: 0.0,
          name: "test_block_no_drop"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"block_input" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, %{"block_input" => input_data})

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  # ============================================================================
  # build_moe_backbone/1
  # ============================================================================

  describe "build_moe_backbone/1" do
    @tag :slow
    test "with mamba backbone" do
      model =
        MoE.build_moe_backbone(
          embed_dim: @input_size,
          hidden_size: @input_size,
          num_layers: 2,
          moe_every: 2,
          num_experts: 2,
          top_k: 1,
          backbone: :mamba,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      # Last timestep extraction: [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, @input_size}
    end

    @tag :slow
    @tag timeout: 120_000
    test "with attention backbone" do
      # build_attention_sublayer hardcodes num_heads=4, head_dim=64
      # so hidden_size must be >= 256 for the residual to match
      attn_hidden = 256

      model =
        MoE.build_moe_backbone(
          embed_dim: @input_size,
          hidden_size: attn_hidden,
          num_layers: 2,
          moe_every: 2,
          num_experts: 2,
          top_k: 1,
          backbone: :attention,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, attn_hidden}
    end

    @tag :slow
    test "embed_dim != hidden_size triggers input projection" do
      model =
        MoE.build_moe_backbone(
          embed_dim: 8,
          hidden_size: @input_size,
          num_layers: 2,
          moe_every: 2,
          num_experts: 2,
          top_k: 1,
          backbone: :mamba,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, 8}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 8})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @input_size}
    end

    @tag :slow
    test "with dropout > 0" do
      model =
        MoE.build_moe_backbone(
          embed_dim: @input_size,
          hidden_size: @input_size,
          num_layers: 2,
          moe_every: 2,
          num_experts: 2,
          top_k: 1,
          backbone: :mamba,
          dropout: 0.1,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @input_size}
    end

    @tag :slow
    test "unknown backbone type defaults to mamba" do
      model =
        MoE.build_moe_backbone(
          embed_dim: @input_size,
          hidden_size: @input_size,
          num_layers: 2,
          moe_every: 2,
          num_experts: 2,
          top_k: 1,
          backbone: :unknown_type,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @input_size}
    end
  end

  # ============================================================================
  # Dropout=0 branches in expert builders
  # ============================================================================

  describe "expert dropout=0" do
    test "FFN expert with dropout=0 skips dropout layer" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          top_k: 1,
          routing: :top_k,
          expert_type: :ffn,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "GLU expert with dropout=0 skips dropout layer" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          top_k: 1,
          routing: :top_k,
          expert_type: :glu,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  # ============================================================================
  # Soft routing and hash routing
  # ============================================================================

  describe "soft routing with different expert counts" do
    test "soft routing with 2 experts" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          routing: :soft,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  describe "hash routing with different expert counts" do
    test "hash routing with 2 experts" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 32,
          output_size: @input_size,
          num_experts: 2,
          routing: :hash,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  # ============================================================================
  # compute_aux_loss edge cases
  # ============================================================================

  describe "compute_aux_loss edge cases" do
    test "with all-zero expert_mask" do
      num_experts = 4
      router_probs = Nx.broadcast(0.25, {@batch_size, @seq_len, num_experts})
      expert_mask = Nx.broadcast(0.0, {@batch_size, @seq_len, num_experts})

      loss = MoE.compute_aux_loss(router_probs, expert_mask)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0.0
    end

    test "with single-expert routing mask" do
      num_experts = 4
      router_probs = Nx.broadcast(0.25, {@batch_size, @seq_len, num_experts})

      # Only expert 0 is selected
      expert_mask =
        Nx.concatenate(
          [
            Nx.broadcast(1.0, {@batch_size, @seq_len, 1}),
            Nx.broadcast(0.0, {@batch_size, @seq_len, 3})
          ],
          axis: 2
        )

      loss = MoE.compute_aux_loss(router_probs, expert_mask)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0.0
    end
  end

  # ============================================================================
  # estimate_speedup edge cases
  # ============================================================================

  describe "estimate_speedup edge cases" do
    test "with default expert_fraction" do
      speedup = MoE.estimate_speedup(8, 2)
      assert speedup > 1.0
    end

    test "with top_k = num_experts (no speedup on expert layers)" do
      speedup = MoE.estimate_speedup(4, 4, 0.5)
      assert_in_delta speedup, 1.0, 0.01
    end

    test "with expert_fraction = 1.0 (all expert layers)" do
      speedup = MoE.estimate_speedup(8, 2, 1.0)
      assert_in_delta speedup, 4.0, 0.01
    end
  end

  # ============================================================================
  # recommended_defaults completeness
  # ============================================================================

  describe "recommended_defaults/0 completeness" do
    test "returns all expected keys" do
      defaults = MoE.recommended_defaults()
      assert Keyword.has_key?(defaults, :num_experts)
      assert Keyword.has_key?(defaults, :top_k)
      assert Keyword.has_key?(defaults, :routing)
      assert Keyword.has_key?(defaults, :expert_type)
      assert Keyword.has_key?(defaults, :capacity_factor)
      assert Keyword.has_key?(defaults, :load_balance_weight)
      assert Keyword.has_key?(defaults, :moe_every)
    end
  end
end
