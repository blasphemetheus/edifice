defmodule Edifice.CoverageBatchATest do
  @moduledoc "Coverage tests for lowest-coverage modules: MambaSSD, MoE, HybridBuilder, MambaCumsum, PointNet, LoRA"
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  @batch 2
  @seq_len 8
  @embed 16

  # ==========================================================================
  # MambaSSD (16%) - Need training_mode forward pass
  # ==========================================================================
  describe "MambaSSD" do
    alias Edifice.SSM.MambaSSD

    test "build with training_mode: true" do
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: true,
          chunk_size: 4
        )

      assert %Axon{} = model
    end

    test "forward pass with training_mode: true" do
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: true,
          chunk_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "forward pass with training_mode: false (inference)" do
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: false,
          chunk_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # MoE (39%) - Need backbone, soft/hash routing, compute_aux_loss
  # ==========================================================================
  describe "MoE" do
    alias Edifice.Meta.MoE

    test "build with :soft routing" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          routing: :soft
        )

      assert %Axon{} = model
    end

    test "build with :hash routing" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          routing: :hash
        )

      assert %Axon{} = model
    end

    test "build with :switch routing" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          routing: :switch
        )

      assert %Axon{} = model
    end

    test "build with 4 experts" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 4,
          top_k: 2
        )

      assert %Axon{} = model
    end

    test "build_block wraps MoE in transformer block pattern" do
      input = Axon.input("moe_input", shape: {nil, nil, @embed})

      model =
        MoE.build_block(input,
          hidden_size: @embed,
          num_experts: 2,
          top_k: 1,
          dropout: 0.0,
          name: "moe_block"
        )

      assert %Axon{} = model
    end

    test "build_moe_backbone builds complete model" do
      model =
        MoE.build_moe_backbone(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          moe_every: 2,
          num_experts: 2,
          top_k: 1,
          seq_len: @seq_len,
          dropout: 0.0
        )

      assert %Axon{} = model
    end

    test "compute_aux_loss returns scalar" do
      router_probs = Nx.broadcast(0.25, {@batch, @seq_len, 4})
      expert_mask = Nx.broadcast(1.0, {@batch, @seq_len, 4})
      loss = MoE.compute_aux_loss(router_probs, expert_mask)
      assert Nx.shape(loss) == {}
    end

    test "estimate_speedup returns a number" do
      result = MoE.estimate_speedup(8, 2, 256)
      assert is_number(result)
    end
  end

  # ==========================================================================
  # HybridBuilder (42%) - Need patterns, param_count, visualize, layer types
  # ==========================================================================
  describe "HybridBuilder" do
    alias Edifice.SSM.HybridBuilder

    test "pattern/2 returns :rwkv_attention pattern" do
      p = HybridBuilder.pattern(:rwkv_attention, 4)
      assert is_list(p)
      assert :rwkv in p
    end

    test "pattern/2 returns :full_hybrid pattern" do
      p = HybridBuilder.pattern(:full_hybrid, 6)
      assert is_list(p)
      assert length(p) == 6
      assert :mamba in p
      assert :attention in p
    end

    test "param_count returns positive integer" do
      pattern = [:mamba, :attention, :ffn]
      count = HybridBuilder.param_count(pattern, embed_dim: @embed, hidden_size: @embed)
      assert is_integer(count)
      assert count > 0
    end

    test "visualize returns string diagram" do
      pattern = [:mamba, :attention, :gla, :ffn]
      viz = HybridBuilder.visualize(pattern)
      assert is_binary(viz)
      assert viz =~ "M"
      assert viz =~ "A"
      assert viz =~ "Legend"
    end

    test "build_pattern convenience function" do
      model =
        HybridBuilder.build_pattern(:jamba_like, 2,
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "build with :ffn layer type" do
      model =
        HybridBuilder.build([:ffn, :ffn],
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "build with :attention layer type" do
      model =
        HybridBuilder.build([:attention],
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len,
          attention_num_heads: 2
        )

      assert %Axon{} = model
    end

    test "build with :gla layer type" do
      model =
        HybridBuilder.build([:gla],
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len,
          gla_num_heads: 2
        )

      assert %Axon{} = model
    end

    test "build with :rwkv layer type" do
      model =
        HybridBuilder.build([:rwkv],
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len,
          rwkv_head_size: 8
        )

      assert %Axon{} = model
    end

    test "build with :kan layer type" do
      model =
        HybridBuilder.build([:kan],
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end
  end

  # ==========================================================================
  # MambaCumsum (51%) - Need alternative scan algorithms
  # ==========================================================================
  describe "MambaCumsum" do
    alias Edifice.SSM.MambaCumsum

    test "build with scan_algo: :cumsum_transposed" do
      model =
        MambaCumsum.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          scan_algo: :cumsum_transposed
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with scan_algo: :cumsum_logspace" do
      model =
        MambaCumsum.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          scan_algo: :cumsum_logspace
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # PointNet (53%) - Need T-Net options
  # ==========================================================================
  describe "PointNet" do
    alias Edifice.Sets.PointNet

    test "build with use_t_net: true" do
      model =
        PointNet.build(
          input_dim: 3,
          num_classes: 4,
          hidden_dims: [8, 16],
          global_dims: [16, 8],
          use_t_net: true
        )

      assert %Axon{} = model
    end

    test "build with use_feature_t_net: true" do
      model =
        PointNet.build(
          input_dim: 3,
          num_classes: 4,
          hidden_dims: [8, 16],
          global_dims: [16, 8],
          use_feature_t_net: true
        )

      assert %Axon{} = model
    end

    test "build with both T-Nets" do
      model =
        PointNet.build(
          input_dim: 3,
          num_classes: 4,
          hidden_dims: [8, 16],
          global_dims: [16, 8],
          use_t_net: true,
          use_feature_t_net: true
        )

      assert %Axon{} = model
    end
  end

  # ==========================================================================
  # LoRA (53%) - Need wrap/3
  # ==========================================================================
  describe "LoRA" do
    alias Edifice.Meta.LoRA

    test "wrap wraps an existing layer" do
      input = Axon.input("input", shape: {nil, @embed})
      base_layer = Axon.dense(input, @embed, name: "base")

      wrapped = LoRA.wrap(input, base_layer, rank: 4, output_size: @embed, name: "lora_wrap")
      assert %Axon{} = wrapped
    end

    test "lora_delta builds low-rank delta" do
      input = Axon.input("input", shape: {nil, @embed})
      delta = LoRA.lora_delta(input, @embed, rank: 4, name: "lora_test")
      assert %Axon{} = delta
    end
  end
end
