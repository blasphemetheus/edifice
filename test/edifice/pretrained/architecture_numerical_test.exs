defmodule Edifice.Pretrained.ArchitectureNumericalTest do
  @moduledoc """
  Random-weight cross-framework numerical validation for 7 architectures.

  For each architecture:
  1. PyTorch builds an identical model, inits with seed(42), saves weights + forward output
  2. Edifice builds the same architecture, loads PyTorch weights, runs forward pass
  3. Outputs compared at atol=1e-4

  Prerequisites:
    python scripts/generate_random_weight_fixtures.py

  Tagged :external since they require fixture files.
  """
  use ExUnit.Case, async: false
  import Edifice.NumericalFixtureHelper

  @moduletag :external

  # ============================================================================
  # MinGRU
  # ============================================================================

  describe "MinGRU forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("min_gru_random.safetensors")
      input = fixture["input.state_sequence"]

      model =
        Edifice.Recurrent.MinGRU.build(
          embed_dim: 32,
          hidden_size: 64,
          num_layers: 2,
          dropout: 0.0,
          seq_len: 8
        )

      template = %{"state_sequence" => Nx.template({2, 8, 32}, :f32)}

      key_mapping = [
        # Input projection
        {"input_projection.weight", ["input_projection", "kernel"], :transpose_2d},
        {"input_projection.bias", ["input_projection", "bias"], :identity},
        # Layer 1
        {"layers.0.norm.weight", ["min_gru_1_norm", "gamma"], :identity},
        {"layers.0.norm.bias", ["min_gru_1_norm", "beta"], :identity},
        {"layers.0.gate.weight", ["min_gru_1_gate", "kernel"], :transpose_2d},
        {"layers.0.gate.bias", ["min_gru_1_gate", "bias"], :identity},
        {"layers.0.candidate.weight", ["min_gru_1_candidate", "kernel"], :transpose_2d},
        {"layers.0.candidate.bias", ["min_gru_1_candidate", "bias"], :identity},
        # Layer 2
        {"layers.1.norm.weight", ["min_gru_2_norm", "gamma"], :identity},
        {"layers.1.norm.bias", ["min_gru_2_norm", "beta"], :identity},
        {"layers.1.gate.weight", ["min_gru_2_gate", "kernel"], :transpose_2d},
        {"layers.1.gate.bias", ["min_gru_2_gate", "bias"], :identity},
        {"layers.1.candidate.weight", ["min_gru_2_candidate", "kernel"], :transpose_2d},
        {"layers.1.candidate.bias", ["min_gru_2_candidate", "bias"], :identity},
        # Final norm
        {"final_norm.weight", ["final_norm", "gamma"], :identity},
        {"final_norm.bias", ["final_norm", "beta"], :identity}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"state_sequence" => input})

      expected = fixture["expected_output"]
      assert_all_close(output, expected, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # DeepResLSTM
  # ============================================================================

  describe "DeepResLSTM forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("lstm_random.safetensors")
      input = fixture["input.state_sequence"]

      model =
        Edifice.Recurrent.DeepResLSTM.build(
          embed_dim: 32,
          hidden_size: 64,
          num_layers: 2,
          dropout: 0.0,
          seq_len: 8
        )

      template = %{"state_sequence" => Nx.template({2, 8, 32}, :f32)}

      # Axon.lstm stores per-gate params:
      #   input_kernel.{wii,wif,wig,wio} [H, H], hidden_kernel.{whi,whf,whg,who} [H, H]
      #   bias.{bi,bf,bg,bo} [H]
      # PyTorch ManualLSTM matches this structure with ParameterDict.
      key_mapping =
        [
          # Encoder
          {"encoder.weight", ["encoder", "kernel"], :transpose_2d},
          {"encoder.bias", ["encoder", "bias"], :identity},
          # Final norm
          {"final_norm.weight", ["final_norm", "gamma"], :identity},
          {"final_norm.bias", ["final_norm", "beta"], :identity}
        ] ++
          Enum.flat_map(0..1, fn i ->
            block_idx = i + 1
            [
              {"blocks.#{i}.prenorm.weight", ["block_#{block_idx}_prenorm", "gamma"], :identity},
              {"blocks.#{i}.prenorm.bias", ["block_#{block_idx}_prenorm", "beta"], :identity},
              {"blocks.#{i}.decoder.weight", ["decoder_#{block_idx}", "kernel"], :transpose_2d},
              {"blocks.#{i}.decoder.bias", ["decoder_#{block_idx}", "bias"], :identity}
            ] ++
              Enum.flat_map(
                [{"wii", "wii"}, {"wif", "wif"}, {"wig", "wig"}, {"wio", "wio"}],
                fn {pt_gate, ax_gate} ->
                  [{"blocks.#{i}.lstm.input_kernel.#{pt_gate}", ["lstm_#{block_idx}", "input_kernel", ax_gate], :transpose_2d}]
                end
              ) ++
              Enum.flat_map(
                [{"whi", "whi"}, {"whf", "whf"}, {"whg", "whg"}, {"who", "who"}],
                fn {pt_gate, ax_gate} ->
                  [{"blocks.#{i}.lstm.hidden_kernel.#{pt_gate}", ["lstm_#{block_idx}", "hidden_kernel", ax_gate], :transpose_2d}]
                end
              ) ++
              Enum.map(
                [{"bi", "bi"}, {"bf", "bf"}, {"bg", "bg"}, {"bo", "bo"}],
                fn {pt_gate, ax_gate} ->
                  {"blocks.#{i}.lstm.bias.#{pt_gate}", ["lstm_#{block_idx}", "bias", ax_gate], :identity}
                end
              )
          end)

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"state_sequence" => input})

      expected = fixture["expected_output"]
      assert_all_close(output, expected, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # GAT
  # ============================================================================

  describe "GAT forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("gat_random.safetensors")
      nodes = fixture["input.nodes"]
      adjacency = fixture["input.adjacency"]

      model =
        Edifice.Graph.GAT.build(
          input_dim: 16,
          hidden_size: 8,
          num_heads: 4,
          num_classes: 7,
          num_layers: 2,
          dropout: 0.0
        )

      template = %{
        "nodes" => Nx.template({2, 10, 16}, :f32),
        "adjacency" => Nx.template({2, 10, 10}, :f32)
      }

      key_mapping = [
        # Hidden layer 0
        {"hidden_layers.0.proj.weight", ["gat_0_proj", "kernel"], :transpose_2d},
        {"hidden_layers.0.proj.bias", ["gat_0_proj", "bias"], :identity},
        {"hidden_layers.0.attn_src.weight", ["gat_0_attn_src", "kernel"], :transpose_2d},
        {"hidden_layers.0.attn_tgt.weight", ["gat_0_attn_tgt", "kernel"], :transpose_2d},
        # Output layer
        {"output_layer.proj.weight", ["gat_output_proj", "kernel"], :transpose_2d},
        {"output_layer.proj.bias", ["gat_output_proj", "bias"], :identity},
        {"output_layer.attn_src.weight", ["gat_output_attn_src", "kernel"], :transpose_2d},
        {"output_layer.attn_tgt.weight", ["gat_output_attn_tgt", "kernel"], :transpose_2d}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})

      expected = fixture["expected_output"]
      assert_all_close(output, expected, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # GQA
  # ============================================================================

  describe "GQA forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("gqa_random.safetensors")
      input = fixture["input.state_sequence"]

      # embed_dim == hidden_size to avoid input_projection (matching Python)
      model =
        Edifice.Attention.GQA.build(
          embed_dim: 32,
          hidden_size: 32,
          num_heads: 4,
          num_kv_heads: 2,
          num_layers: 2,
          dropout: 0.0,
          rope: false,
          seq_len: 8
        )

      template = %{"state_sequence" => Nx.template({2, 8, 32}, :f32)}

      # GQA uses ModelBuilder -> TransformerBlock -> GQA attention
      # TransformerBlock names: gqa_block_{idx}_attn_norm, gqa_block_{idx}_ffn_norm
      # GQA attention names: gqa_block_{idx}_attn_q_proj, etc.
      key_mapping = [
        # Block 1
        {"blocks.0.attn_norm.weight", ["gqa_block_1_attn_norm", "gamma"], :identity},
        {"blocks.0.attn_norm.bias", ["gqa_block_1_attn_norm", "beta"], :identity},
        {"blocks.0.attn.q_proj.weight", ["gqa_block_1_attn_q_proj", "kernel"], :transpose_2d},
        {"blocks.0.attn.q_proj.bias", ["gqa_block_1_attn_q_proj", "bias"], :identity},
        {"blocks.0.attn.k_proj.weight", ["gqa_block_1_attn_k_proj", "kernel"], :transpose_2d},
        {"blocks.0.attn.k_proj.bias", ["gqa_block_1_attn_k_proj", "bias"], :identity},
        {"blocks.0.attn.v_proj.weight", ["gqa_block_1_attn_v_proj", "kernel"], :transpose_2d},
        {"blocks.0.attn.v_proj.bias", ["gqa_block_1_attn_v_proj", "bias"], :identity},
        {"blocks.0.attn.out_proj.weight", ["gqa_block_1_attn_out_proj", "kernel"], :transpose_2d},
        {"blocks.0.attn.out_proj.bias", ["gqa_block_1_attn_out_proj", "bias"], :identity},
        {"blocks.0.ffn_norm.weight", ["gqa_block_1_ffn_norm", "gamma"], :identity},
        {"blocks.0.ffn_norm.bias", ["gqa_block_1_ffn_norm", "beta"], :identity},
        {"blocks.0.ffn_up.weight", ["gqa_block_1_ffn_up", "kernel"], :transpose_2d},
        {"blocks.0.ffn_up.bias", ["gqa_block_1_ffn_up", "bias"], :identity},
        {"blocks.0.ffn_down.weight", ["gqa_block_1_ffn_down", "kernel"], :transpose_2d},
        {"blocks.0.ffn_down.bias", ["gqa_block_1_ffn_down", "bias"], :identity},
        # Block 2
        {"blocks.1.attn_norm.weight", ["gqa_block_2_attn_norm", "gamma"], :identity},
        {"blocks.1.attn_norm.bias", ["gqa_block_2_attn_norm", "beta"], :identity},
        {"blocks.1.attn.q_proj.weight", ["gqa_block_2_attn_q_proj", "kernel"], :transpose_2d},
        {"blocks.1.attn.q_proj.bias", ["gqa_block_2_attn_q_proj", "bias"], :identity},
        {"blocks.1.attn.k_proj.weight", ["gqa_block_2_attn_k_proj", "kernel"], :transpose_2d},
        {"blocks.1.attn.k_proj.bias", ["gqa_block_2_attn_k_proj", "bias"], :identity},
        {"blocks.1.attn.v_proj.weight", ["gqa_block_2_attn_v_proj", "kernel"], :transpose_2d},
        {"blocks.1.attn.v_proj.bias", ["gqa_block_2_attn_v_proj", "bias"], :identity},
        {"blocks.1.attn.out_proj.weight", ["gqa_block_2_attn_out_proj", "kernel"], :transpose_2d},
        {"blocks.1.attn.out_proj.bias", ["gqa_block_2_attn_out_proj", "bias"], :identity},
        {"blocks.1.ffn_norm.weight", ["gqa_block_2_ffn_norm", "gamma"], :identity},
        {"blocks.1.ffn_norm.bias", ["gqa_block_2_ffn_norm", "beta"], :identity},
        {"blocks.1.ffn_up.weight", ["gqa_block_2_ffn_up", "kernel"], :transpose_2d},
        {"blocks.1.ffn_up.bias", ["gqa_block_2_ffn_up", "bias"], :identity},
        {"blocks.1.ffn_down.weight", ["gqa_block_2_ffn_down", "kernel"], :transpose_2d},
        {"blocks.1.ffn_down.bias", ["gqa_block_2_ffn_down", "bias"], :identity},
        # Final norm
        {"final_norm.weight", ["final_norm", "gamma"], :identity},
        {"final_norm.bias", ["final_norm", "beta"], :identity}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"state_sequence" => input})

      expected = fixture["expected_output"]
      # Attention models may have slightly larger numerical differences
      assert_all_close(output, expected, atol: 5.0e-4)
    end
  end

  # ============================================================================
  # DeltaNet
  # ============================================================================

  describe "DeltaNet forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("delta_net_random.safetensors")
      input = fixture["input.state_sequence"]

      model =
        Edifice.Recurrent.DeltaNet.build(
          embed_dim: 32,
          hidden_size: 32,
          num_heads: 4,
          num_layers: 2,
          dropout: 0.0,
          seq_len: 8
        )

      template = %{"state_sequence" => Nx.template({2, 8, 32}, :f32)}

      # embed_dim == hidden_size, so no input_projection
      key_mapping = [
        # Layer 1
        {"layers.0.norm.weight", ["delta_net_1_norm", "gamma"], :identity},
        {"layers.0.norm.bias", ["delta_net_1_norm", "beta"], :identity},
        {"layers.0.qkvb_proj.weight", ["delta_net_1_qkvb_proj", "kernel"], :transpose_2d},
        {"layers.0.qkvb_proj.bias", ["delta_net_1_qkvb_proj", "bias"], :identity},
        {"layers.0.out_proj.weight", ["delta_net_1_out_proj", "kernel"], :transpose_2d},
        {"layers.0.out_proj.bias", ["delta_net_1_out_proj", "bias"], :identity},
        # Layer 2
        {"layers.1.norm.weight", ["delta_net_2_norm", "gamma"], :identity},
        {"layers.1.norm.bias", ["delta_net_2_norm", "beta"], :identity},
        {"layers.1.qkvb_proj.weight", ["delta_net_2_qkvb_proj", "kernel"], :transpose_2d},
        {"layers.1.qkvb_proj.bias", ["delta_net_2_qkvb_proj", "bias"], :identity},
        {"layers.1.out_proj.weight", ["delta_net_2_out_proj", "kernel"], :transpose_2d},
        {"layers.1.out_proj.bias", ["delta_net_2_out_proj", "bias"], :identity},
        # Final norm
        {"final_norm.weight", ["final_norm", "gamma"], :identity},
        {"final_norm.bias", ["final_norm", "beta"], :identity}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"state_sequence" => input})

      expected = fixture["expected_output"]
      assert_all_close(output, expected, atol: 1.0e-4)
    end

    test "gradients match PyTorch reference" do
      fixture = load_fixture("delta_net_random.safetensors")
      input = fixture["input.state_sequence"]

      model =
        Edifice.Recurrent.DeltaNet.build(
          embed_dim: 32,
          hidden_size: 32,
          num_heads: 4,
          num_layers: 2,
          dropout: 0.0,
          seq_len: 8
        )

      template = %{"state_sequence" => Nx.template({2, 8, 32}, :f32)}

      key_mapping = [
        {"layers.0.norm.weight", ["delta_net_1_norm", "gamma"], :identity},
        {"layers.0.norm.bias", ["delta_net_1_norm", "beta"], :identity},
        {"layers.0.qkvb_proj.weight", ["delta_net_1_qkvb_proj", "kernel"], :transpose_2d},
        {"layers.0.qkvb_proj.bias", ["delta_net_1_qkvb_proj", "bias"], :identity},
        {"layers.0.out_proj.weight", ["delta_net_1_out_proj", "kernel"], :transpose_2d},
        {"layers.0.out_proj.bias", ["delta_net_1_out_proj", "bias"], :identity},
        {"layers.1.norm.weight", ["delta_net_2_norm", "gamma"], :identity},
        {"layers.1.norm.bias", ["delta_net_2_norm", "beta"], :identity},
        {"layers.1.qkvb_proj.weight", ["delta_net_2_qkvb_proj", "kernel"], :transpose_2d},
        {"layers.1.qkvb_proj.bias", ["delta_net_2_qkvb_proj", "bias"], :identity},
        {"layers.1.out_proj.weight", ["delta_net_2_out_proj", "kernel"], :transpose_2d},
        {"layers.1.out_proj.bias", ["delta_net_2_out_proj", "bias"], :identity},
        {"final_norm.weight", ["final_norm", "gamma"], :identity},
        {"final_norm.bias", ["final_norm", "beta"], :identity}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)

      # Compute gradient w.r.t. input using value_and_grad
      grad_fn = fn input_tensor ->
        output = predict_fn.(params, %{"state_sequence" => input_tensor})
        Nx.sum(output)
      end

      {_value, input_grad} = Nx.Defn.value_and_grad(grad_fn).(input)

      expected_grad = fixture["gradient.input"]
      assert_all_close(input_grad, expected_grad, atol: 1.0e-3)
    end
  end

  # ============================================================================
  # Mamba
  # ============================================================================

  describe "Mamba forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("mamba_random.safetensors")
      input = fixture["input.state_sequence"]

      model =
        Edifice.SSM.Mamba.build(
          embed_dim: 32,
          hidden_size: 32,
          state_size: 8,
          num_layers: 2,
          expand_factor: 2,
          conv_size: 4,
          dropout: 0.0,
          seq_len: 8
        )

      template = %{"state_sequence" => Nx.template({2, 8, 32}, :f32)}

      # Mamba uses Common.build_model + build_block
      # embed_dim == hidden_size, so no input_projection
      # inner_size = 32 * 2 = 64, dt_rank = max(32 / 16, 1) = 2
      key_mapping = [
        # Block 1
        {"blocks.0.norm.weight", ["mamba_block_1_norm", "gamma"], :identity},
        {"blocks.0.norm.bias", ["mamba_block_1_norm", "beta"], :identity},
        {"blocks.0.in_proj.weight", ["mamba_block_1_in_proj", "kernel"], :transpose_2d},
        {"blocks.0.in_proj.bias", ["mamba_block_1_in_proj", "bias"], :identity},
        {"blocks.0.dw_conv.weight", ["mamba_block_1_conv_dw_conv", "kernel"], :transpose_conv},
        {"blocks.0.dw_conv.bias", ["mamba_block_1_conv_dw_conv", "bias"], :identity},
        {"blocks.0.bc_proj.weight", ["mamba_block_1_ssm_bc_proj", "kernel"], :transpose_2d},
        {"blocks.0.bc_proj.bias", ["mamba_block_1_ssm_bc_proj", "bias"], :identity},
        {"blocks.0.dt_rank_proj.weight", ["mamba_block_1_ssm_dt_rank", "kernel"], :transpose_2d},
        {"blocks.0.dt_rank_proj.bias", ["mamba_block_1_ssm_dt_rank", "bias"], :identity},
        {"blocks.0.dt_proj.weight", ["mamba_block_1_ssm_dt_proj", "kernel"], :transpose_2d},
        {"blocks.0.dt_proj.bias", ["mamba_block_1_ssm_dt_proj", "bias"], :identity},
        {"blocks.0.out_proj.weight", ["mamba_block_1_out_proj", "kernel"], :transpose_2d},
        {"blocks.0.out_proj.bias", ["mamba_block_1_out_proj", "bias"], :identity},
        # Block 2
        {"blocks.1.norm.weight", ["mamba_block_2_norm", "gamma"], :identity},
        {"blocks.1.norm.bias", ["mamba_block_2_norm", "beta"], :identity},
        {"blocks.1.in_proj.weight", ["mamba_block_2_in_proj", "kernel"], :transpose_2d},
        {"blocks.1.in_proj.bias", ["mamba_block_2_in_proj", "bias"], :identity},
        {"blocks.1.dw_conv.weight", ["mamba_block_2_conv_dw_conv", "kernel"], :transpose_conv},
        {"blocks.1.dw_conv.bias", ["mamba_block_2_conv_dw_conv", "bias"], :identity},
        {"blocks.1.bc_proj.weight", ["mamba_block_2_ssm_bc_proj", "kernel"], :transpose_2d},
        {"blocks.1.bc_proj.bias", ["mamba_block_2_ssm_bc_proj", "bias"], :identity},
        {"blocks.1.dt_rank_proj.weight", ["mamba_block_2_ssm_dt_rank", "kernel"], :transpose_2d},
        {"blocks.1.dt_rank_proj.bias", ["mamba_block_2_ssm_dt_rank", "bias"], :identity},
        {"blocks.1.dt_proj.weight", ["mamba_block_2_ssm_dt_proj", "kernel"], :transpose_2d},
        {"blocks.1.dt_proj.bias", ["mamba_block_2_ssm_dt_proj", "bias"], :identity},
        {"blocks.1.out_proj.weight", ["mamba_block_2_out_proj", "kernel"], :transpose_2d},
        {"blocks.1.out_proj.bias", ["mamba_block_2_out_proj", "bias"], :identity}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"state_sequence" => input})

      expected = fixture["expected_output"]
      assert_all_close(output, expected, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # DiT
  # ============================================================================

  describe "DiT forward pass" do
    test "matches PyTorch reference output" do
      fixture = load_fixture("dit_random.safetensors")
      noisy_input = fixture["input.noisy_input"]
      timestep = fixture["input.timestep"]

      model =
        Edifice.Generative.DiT.build(
          input_dim: 64,
          hidden_size: 128,
          depth: 2,
          num_heads: 4,
          mlp_ratio: 4.0,
          num_steps: 1000
        )

      template = %{
        "noisy_input" => Nx.template({2, 64}, :f32),
        "timestep" => Nx.template({2}, :f32)
      }

      key_mapping = [
        # Time MLP
        {"time_mlp_1.weight", ["time_mlp_1", "kernel"], :transpose_2d},
        {"time_mlp_1.bias", ["time_mlp_1", "bias"], :identity},
        {"time_mlp_2.weight", ["time_mlp_2", "kernel"], :transpose_2d},
        {"time_mlp_2.bias", ["time_mlp_2", "bias"], :identity},
        # Input projection + pos embed
        {"input_embed.weight", ["input_embed", "kernel"], :transpose_2d},
        {"input_embed.bias", ["input_embed", "bias"], :identity},
        {"pos_embed_bias", ["pos_embed", "bias"], :identity},
        # Block 1
        {"blocks.0.adaln_attn.weight", ["dit_block_1_adaln_attn", "kernel"], :transpose_2d},
        {"blocks.0.adaln_attn.bias", ["dit_block_1_adaln_attn", "bias"], :identity},
        {"blocks.0.attn_norm.weight", ["dit_block_1_attn_norm", "gamma"], :identity},
        {"blocks.0.attn_norm.bias", ["dit_block_1_attn_norm", "beta"], :identity},
        {"blocks.0.attn_q.weight", ["dit_block_1_attn_q", "kernel"], :transpose_2d},
        {"blocks.0.attn_q.bias", ["dit_block_1_attn_q", "bias"], :identity},
        {"blocks.0.attn_k.weight", ["dit_block_1_attn_k", "kernel"], :transpose_2d},
        {"blocks.0.attn_k.bias", ["dit_block_1_attn_k", "bias"], :identity},
        {"blocks.0.attn_v.weight", ["dit_block_1_attn_v", "kernel"], :transpose_2d},
        {"blocks.0.attn_v.bias", ["dit_block_1_attn_v", "bias"], :identity},
        {"blocks.0.attn_out_proj.weight", ["dit_block_1_attn_out_proj", "kernel"], :transpose_2d},
        {"blocks.0.attn_out_proj.bias", ["dit_block_1_attn_out_proj", "bias"], :identity},
        {"blocks.0.adaln_mlp.weight", ["dit_block_1_adaln_mlp", "kernel"], :transpose_2d},
        {"blocks.0.adaln_mlp.bias", ["dit_block_1_adaln_mlp", "bias"], :identity},
        {"blocks.0.mlp_norm.weight", ["dit_block_1_mlp_norm", "gamma"], :identity},
        {"blocks.0.mlp_norm.bias", ["dit_block_1_mlp_norm", "beta"], :identity},
        {"blocks.0.mlp_up.weight", ["dit_block_1_mlp_up", "kernel"], :transpose_2d},
        {"blocks.0.mlp_up.bias", ["dit_block_1_mlp_up", "bias"], :identity},
        {"blocks.0.mlp_down.weight", ["dit_block_1_mlp_down", "kernel"], :transpose_2d},
        {"blocks.0.mlp_down.bias", ["dit_block_1_mlp_down", "bias"], :identity},
        # Block 2
        {"blocks.1.adaln_attn.weight", ["dit_block_2_adaln_attn", "kernel"], :transpose_2d},
        {"blocks.1.adaln_attn.bias", ["dit_block_2_adaln_attn", "bias"], :identity},
        {"blocks.1.attn_norm.weight", ["dit_block_2_attn_norm", "gamma"], :identity},
        {"blocks.1.attn_norm.bias", ["dit_block_2_attn_norm", "beta"], :identity},
        {"blocks.1.attn_q.weight", ["dit_block_2_attn_q", "kernel"], :transpose_2d},
        {"blocks.1.attn_q.bias", ["dit_block_2_attn_q", "bias"], :identity},
        {"blocks.1.attn_k.weight", ["dit_block_2_attn_k", "kernel"], :transpose_2d},
        {"blocks.1.attn_k.bias", ["dit_block_2_attn_k", "bias"], :identity},
        {"blocks.1.attn_v.weight", ["dit_block_2_attn_v", "kernel"], :transpose_2d},
        {"blocks.1.attn_v.bias", ["dit_block_2_attn_v", "bias"], :identity},
        {"blocks.1.attn_out_proj.weight", ["dit_block_2_attn_out_proj", "kernel"], :transpose_2d},
        {"blocks.1.attn_out_proj.bias", ["dit_block_2_attn_out_proj", "bias"], :identity},
        {"blocks.1.adaln_mlp.weight", ["dit_block_2_adaln_mlp", "kernel"], :transpose_2d},
        {"blocks.1.adaln_mlp.bias", ["dit_block_2_adaln_mlp", "bias"], :identity},
        {"blocks.1.mlp_norm.weight", ["dit_block_2_mlp_norm", "gamma"], :identity},
        {"blocks.1.mlp_norm.bias", ["dit_block_2_mlp_norm", "beta"], :identity},
        {"blocks.1.mlp_up.weight", ["dit_block_2_mlp_up", "kernel"], :transpose_2d},
        {"blocks.1.mlp_up.bias", ["dit_block_2_mlp_up", "bias"], :identity},
        {"blocks.1.mlp_down.weight", ["dit_block_2_mlp_down", "kernel"], :transpose_2d},
        {"blocks.1.mlp_down.bias", ["dit_block_2_mlp_down", "bias"], :identity},
        # Final layers
        {"final_norm.weight", ["final_norm", "gamma"], :identity},
        {"final_norm.bias", ["final_norm", "beta"], :identity},
        {"output_proj.weight", ["output_proj", "kernel"], :transpose_2d},
        {"output_proj.bias", ["output_proj", "bias"], :identity}
      ]

      {params, predict_fn} = build_params_from_fixture(fixture, key_mapping, model, template)
      output = predict_fn.(params, %{"noisy_input" => noisy_input, "timestep" => timestep})

      expected = fixture["expected_output"]
      assert_all_close(output, expected, atol: 1.0e-4)
    end
  end
end
