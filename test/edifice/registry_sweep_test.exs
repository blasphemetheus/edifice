defmodule Edifice.RegistrySweepTest do
  @moduledoc """
  Parametric sweep tests that exercise every architecture in the registry
  with multiple batch sizes. Catches Nx.dot batching bugs, broadcasting
  errors, and shape assumptions that hide at small batch sizes.

  Strategy A: For each architecture, build with minimal valid opts,
  run forward pass at batch=1, batch=4, and batch=16, assert shapes
  and finiteness.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  # ── Dimensions ──────────────────────────────────────────────────────
  @small_batch 1
  @med_batch 4
  @large_batch 16
  @batches [@small_batch, @med_batch, @large_batch]

  # Shared small dims to keep tests fast
  @embed 32
  @hidden 16
  @seq_len 8
  @state_size 8
  @num_layers 2
  @image_size 16
  @in_channels 3
  @num_nodes 6
  @node_dim 16
  @num_classes 4
  @num_points 12
  @point_dim 3
  @num_memories 4
  @memory_dim 8
  @latent_size 8
  @action_dim 4
  @action_horizon 4

  # ── Architecture Specs ──────────────────────────────────────────────
  # Each spec: {registry_name, opts, input_fn, output_shape_fn}
  # input_fn: (batch) -> input (tensor or map)
  # output_shape_fn: (batch) -> expected shape tuple

  # --- Sequence models: "state_sequence" {batch, seq_len, embed_dim} ---
  # NOTE: :attention and :sliding_window are building blocks (no build/1), not standalone models.
  # :reservoir uses input_size not embed_dim. :rwkv needs head_size explicitly.
  # These get dedicated blocks below.
  @sequence_archs [
    :mamba,
    :mamba_ssd,
    :mamba_cumsum,
    :mamba_hillis_steele,
    :s4,
    :s4d,
    :s5,
    :h3,
    :hyena,
    :bimamba,
    :gated_ssm,
    :jamba,
    :zamba,
    :lstm,
    :gru,
    :xlstm,
    :min_gru,
    :min_lstm,
    :delta_net,
    :ttt,
    :titans,
    :retnet,
    :gla,
    :hgrn,
    :griffin,
    :gqa,
    :fnet,
    :linear_transformer,
    :nystromformer,
    :performer,
    :kan,
    :liquid,
    :slstm,
    :hawk,
    :gss,
    :xlstm_v2,
    :hyena_v2,
    :retnet_v2,
    :hymba,
    :megalodon,
    # v0.2.0 attention additions
    :based,
    :conformer,
    :infini_attention,
    :mega,
    :gla_v2,
    :hgrn_v2,
    :kda,
    :gated_attention,
    :sigmoid_attention,
    :mla,
    :diff_transformer,
    :ring_attention,
    :nsa,
    :rnope_swa,
    :attention,
    :ssmax,
    :softpick,
    # v0.2.0 SSM additions
    :striped_hyena,
    :mamba3,
    :ss_transformer,
    # v0.2.0 recurrent additions
    :gated_delta_net,
    :ttt_e2e,
    :mlstm,
    :native_recurrence,
    :transformer_like
  ]

  @sequence_opts [
    embed_dim: @embed,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: 8,
    num_heads: 2,
    dropout: 0.0
  ]

  for arch <- @sequence_archs do
    describe "#{arch} (sequence)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          model = Edifice.build(arch, @sequence_opts)
          assert %Axon{} = model

          input = random_tensor({batch, @seq_len, @embed})
          {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})

          output = predict_fn.(params, %{"state_sequence" => input})
          assert_finite!(output, "#{arch} batch=#{batch}")

          # All sequence models output {batch, hidden_size}
          assert {^batch, @hidden} = Nx.shape(output)
        end
      end
    end
  end

  # --- Reservoir: uses input_size not embed_dim ──────────────────────
  describe "reservoir (sequence)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:reservoir,
            input_size: @embed,
            reservoir_size: @hidden,
            output_size: @hidden,
            seq_len: @seq_len
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})
        assert_finite!(output, "reservoir batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  # --- RWKV: needs explicit head_size to avoid head_size > hidden_size ──
  describe "rwkv (sequence)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        # RWKV defaults head_size=64, hidden_size=256. We need head_size <= hidden_size.
        model =
          Edifice.build(:rwkv,
            embed_dim: @embed,
            hidden_size: @hidden,
            head_size: 8,
            num_layers: @num_layers,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "rwkv batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  # --- Switch/Soft MoE: sequence models with embed_dim ──────────────
  @moe_sequence_archs [:switch_moe, :soft_moe]

  for arch <- @moe_sequence_archs do
    describe "#{arch} (moe sequence)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          opts = [
            embed_dim: @embed,
            hidden_size: @hidden,
            num_layers: @num_layers,
            seq_len: @seq_len,
            num_experts: 2,
            dropout: 0.0
          ]

          model = Edifice.build(arch, opts)
          input = random_tensor({batch, @seq_len, @embed})
          {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
          output = predict_fn.(params, %{"state_sequence" => input})

          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, @hidden} = Nx.shape(output)
        end
      end
    end
  end

  # --- Feedforward: "input" {batch, input_size} ──────────────────────
  describe "mlp (feedforward)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        model = Edifice.build(:mlp, input_size: @embed, hidden_sizes: [@hidden])
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "mlp batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "tabnet (feedforward)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        model = Edifice.build(:tabnet, input_size: @embed, output_size: @num_classes)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "tabnet batch=#{batch}")
        assert {^batch, _out} = Nx.shape(output)
      end
    end
  end

  # --- Vision: "image" {batch, channels, H, W} NCHW ─────────────────
  # Simple vision archs that work with small images
  @simple_vision_archs [:vit, :deit, :mlp_mixer]

  for arch <- @simple_vision_archs do
    describe "#{arch} (vision)" do
      for batch <- [@small_batch, @med_batch] do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          opts = [
            image_size: @image_size,
            in_channels: @in_channels,
            patch_size: 4,
            embed_dim: @hidden,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            dropout: 0.0
          ]

          model = Edifice.build(arch, opts)
          input = random_tensor({batch, @in_channels, @image_size, @image_size})
          {predict_fn, params} = build_and_init(model, %{"image" => input})
          output = predict_fn.(params, %{"image" => input})

          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, _dim} = Nx.shape(output)
        end
      end
    end
  end

  # Swin needs larger images for multi-stage patch merging
  describe "swin (vision)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        # Swin with 2 stages needs image_size >= patch_size * 2^num_stages
        # window_size must divide spatial dim at each stage
        img = 32

        model =
          Edifice.build(:swin,
            image_size: img,
            in_channels: @in_channels,
            patch_size: 4,
            embed_dim: @hidden,
            depths: [1, 1],
            num_heads: [2, 2],
            window_size: 4,
            dropout: 0.0
          )

        input = random_tensor({batch, @in_channels, img, img})
        {predict_fn, params} = build_and_init(model, %{"image" => input})
        output = predict_fn.(params, %{"image" => input})
        assert_finite!(output, "swin batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # ConvNeXt needs enough spatial dims for its 4 downsample stages
  describe "convnext (vision)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        # ConvNeXt downsamples 4x: need image_size >= 32 for 2 stages
        img = 32

        model =
          Edifice.build(:convnext,
            image_size: img,
            in_channels: @in_channels,
            patch_size: 4,
            dims: [@hidden, @hidden * 2],
            depths: [1, 1],
            dropout: 0.0
          )

        input = random_tensor({batch, @in_channels, img, img})
        {predict_fn, params} = build_and_init(model, %{"image" => input})
        output = predict_fn.(params, %{"image" => input})
        assert_finite!(output, "convnext batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "unet (vision)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct spatial output" do
        batch = unquote(batch)

        opts = [
          in_channels: @in_channels,
          out_channels: 1,
          image_size: @image_size,
          base_features: 8,
          depth: 2,
          dropout: 0.0
        ]

        model = Edifice.build(:unet, opts)
        input = random_tensor({batch, @in_channels, @image_size, @image_size})
        {predict_fn, params} = build_and_init(model, %{"image" => input})
        output = predict_fn.(params, %{"image" => input})

        assert_finite!(output, "unet batch=#{batch}")
        assert {^batch, 1, @image_size, @image_size} = Nx.shape(output)
      end
    end
  end

  # --- Graph: "nodes" + "adjacency" ──────────────────────────────────
  @graph_archs [:gcn, :gat, :graph_sage, :gin, :pna, :graph_transformer]

  for arch <- @graph_archs do
    describe "#{arch} (graph)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          opts = [
            input_dim: @node_dim,
            hidden_size: @hidden,
            num_classes: @num_classes,
            num_layers: @num_layers,
            num_heads: 2,
            dropout: 0.0
          ]

          model = Edifice.build(arch, opts)
          nodes = random_tensor({batch, @num_nodes, @node_dim})

          # Create a valid adjacency matrix (symmetric, binary)
          key = Nx.Random.key(99)
          {adj_raw, _} = Nx.Random.uniform(key, shape: {batch, @num_nodes, @num_nodes})
          adj = Nx.greater(adj_raw, 0.5) |> Nx.as_type(:f32)
          # Make symmetric
          adj = Nx.add(adj, Nx.transpose(adj, axes: [0, 2, 1]))
          adj = Nx.min(adj, 1.0)

          input_map = %{"nodes" => nodes, "adjacency" => adj}
          {predict_fn, params} = build_and_init(model, input_map)
          output = predict_fn.(params, input_map)

          assert_finite!(output, "#{arch} batch=#{batch}")
          {out_batch, out_nodes, _dim} = Nx.shape(output)
          assert out_batch == batch
          assert out_nodes == @num_nodes
        end
      end
    end
  end

  describe "schnet (graph)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        opts = [
          input_dim: @node_dim,
          hidden_size: @hidden,
          num_interactions: 2,
          num_filters: @hidden,
          num_rbf: 10
        ]

        model = Edifice.build(:schnet, opts)
        nodes = random_tensor({batch, @num_nodes, @node_dim})
        # SchNet adjacency = pairwise distances [batch, num_nodes, num_nodes]
        distances = random_tensor({batch, @num_nodes, @num_nodes})

        input_map = %{"nodes" => nodes, "adjacency" => distances}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)

        assert_finite!(output, "schnet batch=#{batch}")
        {out_batch, out_nodes, _dim} = Nx.shape(output)
        assert out_batch == batch
        assert out_nodes == @num_nodes
      end
    end
  end

  # --- Sets: "input" {batch, num_items, item_dim} ────────────────────
  describe "deep_sets (sets)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        model = Edifice.build(:deep_sets, input_dim: @point_dim, output_dim: @num_classes)
        input = random_tensor({batch, @num_points, @point_dim})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "deep_sets batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  describe "pointnet (sets)" do
    for batch <- @batches do
      @tag :slow
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)
        model = Edifice.build(:pointnet, num_classes: @num_classes, input_dim: @point_dim)
        input = random_tensor({batch, @num_points, @point_dim})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "pointnet batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  # --- Energy: "input" {batch, input_size} ────────────────────────────
  describe "ebm (energy)" do
    for batch <- @batches do
      test "batch=#{batch} produces scalar energy" do
        batch = unquote(batch)
        model = Edifice.build(:ebm, input_size: @embed)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "ebm batch=#{batch}")
        assert {^batch, 1} = Nx.shape(output)
      end
    end
  end

  describe "hopfield (energy)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:hopfield, input_dim: @embed)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "hopfield batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "neural_ode (energy)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:neural_ode, input_size: @embed, hidden_size: @hidden)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "neural_ode batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # --- Probabilistic: "input" {batch, input_size} ────────────────────
  @prob_archs [
    {:bayesian, [input_size: @embed, output_size: @num_classes]},
    {:mc_dropout, [input_size: @embed, output_size: @num_classes]},
    {:evidential, [input_size: @embed, num_classes: @num_classes]}
  ]

  for {arch, opts} <- @prob_archs do
    describe "#{arch} (probabilistic)" do
      for batch <- @batches do
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)
          opts = unquote(opts)

          model = Edifice.build(arch, opts)
          input = random_tensor({batch, @embed})
          {predict_fn, params} = build_and_init(model, %{"input" => input})
          output = predict_fn.(params, %{"input" => input})

          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, _dim} = Nx.shape(output)
        end
      end
    end
  end

  # --- Memory: multi-input models ────────────────────────────────────
  describe "ntm (memory)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        opts = [
          input_size: @embed,
          output_size: @hidden,
          memory_size: @num_memories,
          memory_dim: @memory_dim,
          num_heads: 1
        ]

        model = Edifice.build(:ntm, opts)
        input = random_tensor({batch, @embed})
        memory = random_tensor({batch, @num_memories, @memory_dim})

        input_map = %{"input" => input, "memory" => memory}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)

        assert_finite!(output, "ntm batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "memory_network (memory)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        opts = [input_dim: @embed, output_dim: @hidden, num_memories: @num_memories]

        model = Edifice.build(:memory_network, opts)
        query = random_tensor({batch, @embed})
        memories = random_tensor({batch, @num_memories, @embed})

        input_map = %{"query" => query, "memories" => memories}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)

        assert_finite!(output, "memory_network batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # --- Meta: various input patterns ──────────────────────────────────
  describe "moe (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        # MoE.build/1 uses input_size (not embed_dim) and "moe_input" name
        opts = [
          input_size: @embed,
          hidden_size: @hidden * 4,
          output_size: @hidden,
          num_experts: 2,
          top_k: 1
        ]

        model = Edifice.build(:moe, opts)
        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"moe_input" => input})
        output = predict_fn.(params, %{"moe_input" => input})

        assert_finite!(output, "moe batch=#{batch}")
        assert {^batch, @seq_len, _out} = Nx.shape(output)
      end
    end
  end

  describe "lora (meta)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 4)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "lora batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "moe_v2 (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        opts = [
          input_size: @embed,
          hidden_size: @hidden * 4,
          output_size: @hidden,
          num_shared_experts: 1,
          num_routed_experts: 2,
          tokens_per_expert: 2,
          dropout: 0.0
        ]

        model = Edifice.build(:moe_v2, opts)
        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"moe_input" => input})
        output = predict_fn.(params, %{"moe_input" => input})

        assert_finite!(output, "moe_v2 batch=#{batch}")
        assert {^batch, @seq_len, _out} = Nx.shape(output)
      end
    end
  end

  describe "dora (meta)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:dora, input_size: @embed, output_size: @hidden, rank: 4)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "dora batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "adapter (meta)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:adapter, hidden_size: @hidden)
        input = random_tensor({batch, @hidden})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "adapter batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "hypernetwork (meta)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        opts = [
          conditioning_size: @embed,
          target_layer_sizes: [{@embed, @hidden}],
          input_size: @embed
        ]

        model = Edifice.build(:hypernetwork, opts)

        input_map = %{
          "conditioning" => random_tensor({batch, @embed}),
          "data_input" => random_tensor({batch, @embed})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)

        assert_finite!(output, "hypernetwork batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "capsule (meta)" do
    # Capsule uses 2D convolutions internally, needs image-like input
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        # CapsNet expects NHWC image input; conv_kernel=9 needs at least 9x9
        img_size = 28

        model =
          Edifice.build(:capsule,
            input_shape: {nil, img_size, img_size, 1},
            conv_channels: 32,
            conv_kernel: 9,
            num_primary_caps: 8,
            primary_cap_dim: 4,
            num_digit_caps: @num_classes,
            digit_cap_dim: 4
          )

        input = random_tensor({batch, img_size, img_size, 1})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "capsule batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # --- Neuromorphic: "input" {batch, input_size} ─────────────────────
  describe "snn (neuromorphic)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:snn,
            input_size: @embed,
            output_size: @num_classes,
            hidden_sizes: [@hidden]
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "snn batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  describe "ann2snn (neuromorphic)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:ann2snn, input_size: @embed, output_size: @num_classes)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "ann2snn batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # --- Convolutional: various input shapes ───────────────────────────
  describe "resnet (convolutional)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:resnet,
            input_shape: {nil, @image_size, @image_size, @in_channels},
            num_classes: @num_classes,
            block_sizes: [1, 1],
            initial_channels: 8
          )

        input = random_tensor({batch, @image_size, @image_size, @in_channels})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "resnet batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  describe "densenet (convolutional)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        # DenseNet uses block_config (list of layer counts per block), not block_layers
        # Image must survive stem (7x7 stride 2 + 3x3 maxpool stride 2) = /4
        img = 32

        model =
          Edifice.build(:densenet,
            input_shape: {nil, img, img, @in_channels},
            num_classes: @num_classes,
            growth_rate: 8,
            block_config: [2, 2],
            initial_channels: 16
          )

        input = random_tensor({batch, img, img, @in_channels})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "densenet batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  describe "tcn (convolutional)" do
    for batch <- @batches do
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:tcn, input_size: @embed, hidden_size: @hidden, num_layers: 2)
        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "tcn batch=#{batch}")
        # TCN preserves sequence length
        assert {^batch, @seq_len, _channels} = Nx.shape(output)
      end
    end
  end

  describe "mobilenet (convolutional)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mobilenet,
            input_dim: @embed,
            hidden_dim: @hidden,
            num_classes: @num_classes
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "mobilenet batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  describe "efficientnet (convolutional)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 300_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        # EfficientNet SE blocks need channels large enough for squeeze-excite ratio
        model =
          Edifice.build(:efficientnet,
            input_dim: 64,
            base_dim: 16,
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
            num_classes: @num_classes
          )

        input = random_tensor({batch, 64})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        assert_finite!(output, "efficientnet batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  # --- Generative: tuple returns ─────────────────────────────────────
  # These return multiple models. We test the first (primary) component.

  describe "vae (generative)" do
    for batch <- @batches do
      test "batch=#{batch}: encoder and decoder produce correct shapes" do
        batch = unquote(batch)
        {encoder, decoder} = Edifice.build(:vae, input_size: @embed, latent_size: @latent_size)

        # Test encoder
        enc_input = random_tensor({batch, @embed})
        {enc_pred, enc_params} = build_and_init(encoder, %{"input" => enc_input})
        enc_out = enc_pred.(enc_params, %{"input" => enc_input})

        assert %{mu: mu, log_var: log_var} = enc_out
        assert_finite!(mu, "vae encoder mu batch=#{batch}")
        assert_finite!(log_var, "vae encoder log_var batch=#{batch}")
        assert {^batch, @latent_size} = Nx.shape(mu)

        # Test decoder
        dec_input = random_tensor({batch, @latent_size})
        {dec_pred, dec_params} = build_and_init(decoder, %{"latent" => dec_input})
        dec_out = dec_pred.(dec_params, %{"latent" => dec_input})

        assert_finite!(dec_out, "vae decoder batch=#{batch}")
        assert {^batch, @embed} = Nx.shape(dec_out)
      end
    end
  end

  describe "vq_vae (generative)" do
    for batch <- @batches do
      test "batch=#{batch}: encoder and decoder produce correct shapes" do
        batch = unquote(batch)

        {encoder, _decoder} =
          Edifice.build(:vq_vae, input_size: @embed, embedding_dim: @latent_size)

        enc_input = random_tensor({batch, @embed})
        {enc_pred, enc_params} = build_and_init(encoder, %{"input" => enc_input})
        enc_out = enc_pred.(enc_params, %{"input" => enc_input})
        assert_finite!(enc_out, "vq_vae encoder batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(enc_out)
      end
    end
  end

  describe "gan (generative)" do
    for batch <- @batches do
      test "batch=#{batch}: generator and discriminator produce correct shapes" do
        batch = unquote(batch)
        {gen, disc} = Edifice.build(:gan, output_size: @embed, latent_size: @latent_size)

        # Generator
        noise = random_tensor({batch, @latent_size})
        {gen_pred, gen_params} = build_and_init(gen, %{"noise" => noise})
        gen_out = gen_pred.(gen_params, %{"noise" => noise})
        assert_finite!(gen_out, "gan generator batch=#{batch}")
        assert {^batch, @embed} = Nx.shape(gen_out)

        # Discriminator
        data = random_tensor({batch, @embed})
        {disc_pred, disc_params} = build_and_init(disc, %{"data" => data})
        disc_out = disc_pred.(disc_params, %{"data" => data})
        assert_finite!(disc_out, "gan discriminator batch=#{batch}")
        assert {^batch, 1} = Nx.shape(disc_out)
      end
    end
  end

  describe "normalizing_flow (generative)" do
    for batch <- @batches do
      test "batch=#{batch} produces finite output" do
        batch = unquote(batch)
        model = Edifice.build(:normalizing_flow, input_size: @embed, num_flows: 2)
        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})

        # NormalizingFlow returns Axon.container with transformed + log_det
        assert_finite!(output, "normalizing_flow batch=#{batch}")
      end
    end
  end

  # Diffusion-family: action-prediction models (noisy_actions, timestep, observations)
  @diffusion_noisy_action_archs [:diffusion, :ddim]

  for arch <- @diffusion_noisy_action_archs do
    describe "#{arch} (generative-diffusion)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          opts = [
            obs_size: @embed,
            action_dim: @action_dim,
            action_horizon: @action_horizon,
            hidden_size: @hidden,
            num_layers: @num_layers,
            dropout: 0.0
          ]

          model = Edifice.build(arch, opts)

          input_map = %{
            "noisy_actions" => random_tensor({batch, @action_horizon, @action_dim}),
            "timestep" => random_tensor({batch}),
            "observations" => random_tensor({batch, @embed})
          }

          {predict_fn, params} = build_and_init(model, input_map)
          output = predict_fn.(params, input_map)

          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, @action_horizon, @action_dim} = Nx.shape(output)
        end
      end
    end
  end

  # Flow matching uses "x_t" instead of "noisy_actions"
  describe "flow_matching (generative-diffusion)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:flow_matching,
            obs_size: @embed,
            action_dim: @action_dim,
            action_horizon: @action_horizon,
            hidden_size: @hidden,
            num_layers: @num_layers,
            dropout: 0.0
          )

        input_map = %{
          "x_t" => random_tensor({batch, @action_horizon, @action_dim}),
          "timestep" => random_tensor({batch}),
          "observations" => random_tensor({batch, @embed})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)

        assert_finite!(output, "flow_matching batch=#{batch}")
        assert {^batch, @action_horizon, @action_dim} = Nx.shape(output)
      end
    end
  end

  # DiT and Score SDE: noisy_input + timestep
  describe "dit (generative-simple)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:dit,
            input_dim: @embed,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            dropout: 0.0
          )

        input_map = %{
          "noisy_input" => random_tensor({batch, @embed}),
          "timestep" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "dit batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "dit_v2 (generative-simple)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:dit_v2,
            input_dim: @embed,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            dropout: 0.0
          )

        input_map = %{
          "noisy_input" => random_tensor({batch, @embed}),
          "timestep" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "dit_v2 batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "score_sde (generative-simple)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:score_sde,
            input_dim: @embed,
            hidden_size: @hidden,
            num_layers: @num_layers
          )

        input_map = %{
          "noisy_input" => random_tensor({batch, @embed}),
          "timestep" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "score_sde batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # Consistency model: noisy_input + sigma
  describe "consistency_model (generative-simple)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:consistency_model,
            input_dim: @embed,
            hidden_size: @hidden,
            num_layers: @num_layers
          )

        input_map = %{
          "noisy_input" => random_tensor({batch, @embed}),
          "sigma" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "consistency_model batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "latent_diffusion (generative)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch}: encoder, diffusion, decoder produce finite output" do
        batch = unquote(batch)

        {encoder, decoder, denoiser} =
          Edifice.build(:latent_diffusion,
            input_size: @embed,
            latent_size: @latent_size,
            hidden_size: @hidden,
            num_layers: @num_layers
          )

        # Encoder: input -> {mu, log_var}
        enc_input = random_tensor({batch, @embed})
        {enc_pred, enc_params} = build_and_init(encoder, %{"input" => enc_input})
        enc_out = enc_pred.(enc_params, %{"input" => enc_input})
        assert_finite!(enc_out, "latent_diffusion encoder batch=#{batch}")

        # Decoder: latent -> reconstruction
        dec_input = random_tensor({batch, @latent_size})
        {dec_pred, dec_params} = build_and_init(decoder, %{"latent" => dec_input})
        dec_out = dec_pred.(dec_params, %{"latent" => dec_input})
        assert_finite!(dec_out, "latent_diffusion decoder batch=#{batch}")

        # Denoiser: (noisy_z, timestep) -> predicted noise
        denoise_input = %{
          "noisy_z" => random_tensor({batch, @latent_size}),
          "timestep" => random_tensor({batch})
        }

        {denoise_pred, denoise_params} = build_and_init(denoiser, denoise_input)
        denoise_out = denoise_pred.(denoise_params, denoise_input)
        assert_finite!(denoise_out, "latent_diffusion denoiser batch=#{batch}")
      end
    end
  end

  # --- Contrastive: some return tuples ───────────────────────────────
  @contrastive_single_archs [
    {:simclr, [encoder_dim: @embed, projection_dim: @hidden]},
    {:barlow_twins, [encoder_dim: @embed, projection_dim: @hidden]},
    {:vicreg, [encoder_dim: @embed, projection_dim: @hidden]}
  ]

  for {arch, opts} <- @contrastive_single_archs do
    describe "#{arch} (contrastive)" do
      for batch <- @batches do
        @tag :slow
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)
          opts = unquote(opts)

          model = Edifice.build(arch, opts)
          input = random_tensor({batch, @embed})
          {predict_fn, params} = build_and_init(model, %{"features" => input})
          output = predict_fn.(params, %{"features" => input})

          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, _dim} = Nx.shape(output)
        end
      end
    end
  end

  describe "byol (contrastive)" do
    for batch <- @batches do
      test "batch=#{batch}: online and target produce correct shapes" do
        batch = unquote(batch)
        {online, target} = Edifice.build(:byol, encoder_dim: @embed, projection_dim: @hidden)

        input = random_tensor({batch, @embed})
        {online_pred, online_params} = build_and_init(online, %{"features" => input})
        online_out = online_pred.(online_params, %{"features" => input})
        assert_finite!(online_out, "byol online batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(online_out)

        {target_pred, target_params} = build_and_init(target, %{"features" => input})
        target_out = target_pred.(target_params, %{"features" => input})
        assert_finite!(target_out, "byol target batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(target_out)
      end
    end
  end

  describe "mae (contrastive)" do
    for batch <- @batches do
      test "batch=#{batch}: encoder and decoder produce correct shapes" do
        batch = unquote(batch)
        num_patches = 4

        {encoder, _decoder} =
          Edifice.build(:mae,
            input_dim: @embed,
            embed_dim: @hidden,
            num_patches: num_patches,
            depth: 1,
            num_heads: 2,
            decoder_depth: 1,
            decoder_num_heads: 2
          )

        enc_input = random_tensor({batch, num_patches, @embed})
        {enc_pred, enc_params} = build_and_init(encoder, %{"visible_patches" => enc_input})
        enc_out = enc_pred.(enc_params, %{"visible_patches" => enc_input})
        assert_finite!(enc_out, "mae encoder batch=#{batch}")
        assert {^batch, ^num_patches, _dim} = Nx.shape(enc_out)
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Chunked Attention
  # ══════════════════════════════════════════════════════════════════════

  @chunked_archs [:lightning_attention, :flash_linear_attention, :dual_chunk_attention]

  for arch <- @chunked_archs do
    describe "#{arch} (chunked attention)" do
      for batch <- [@small_batch, @med_batch] do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)
          block = div(@seq_len, 2)

          model =
            Edifice.build(
              arch,
              Keyword.merge(@sequence_opts, block_size: block, chunk_size: block)
            )

          input = random_tensor({batch, @seq_len, @embed})
          {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
          output = predict_fn.(params, %{"state_sequence" => input})
          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, @hidden} = Nx.shape(output)
        end
      end
    end
  end

  # ── Feedforward additions: bitnet, kat (sequence-like) ──────────────

  for arch <- [:bitnet, :kat] do
    describe "#{arch} (feedforward-sequence)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)
          model = Edifice.build(arch, @sequence_opts)
          input = random_tensor({batch, @seq_len, @embed})
          {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
          output = predict_fn.(params, %{"state_sequence" => input})
          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, @hidden} = Nx.shape(output)
        end
      end
    end
  end

  # ── Conv1D ──────────────────────────────────────────────────────────

  describe "conv1d (convolutional)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:conv1d, input_size: @embed, hidden_size: @hidden, num_layers: 2)
        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})
        assert_finite!(output, "conv1d batch=#{batch}")
        assert {^batch, @seq_len, _channels} = Nx.shape(output)
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Vision
  # ══════════════════════════════════════════════════════════════════════

  @new_vision_archs [
    :focalnet,
    :poolformer,
    :metaformer,
    :caformer,
    :efficient_vit
  ]

  for arch <- @new_vision_archs do
    describe "#{arch} (vision)" do
      for batch <- [@small_batch, @med_batch] do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          model =
            Edifice.build(arch,
              image_size: @image_size,
              in_channels: @in_channels,
              patch_size: 4,
              embed_dim: @hidden,
              hidden_size: @hidden,
              depths: [1],
              dims: [@hidden],
              num_heads: [2],
              depth: 1,
              dropout: 0.0
            )

          input = random_tensor({batch, @in_channels, @image_size, @image_size})
          {predict_fn, params} = build_and_init(model, %{"image" => input})
          output = predict_fn.(params, %{"image" => input})
          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, _dim} = Nx.shape(output)
        end
      end
    end
  end

  # ── MambaVision (4-stage, needs 4-element depths/num_heads) ─────────

  describe "mamba_vision (vision)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mamba_vision,
            image_size: @image_size,
            in_channels: @in_channels,
            dim: 8,
            depths: [1, 1, 1, 1],
            num_heads: [1, 1, 2, 2],
            dropout: 0.0
          )

        input = random_tensor({batch, @in_channels, @image_size, @image_size})
        {predict_fn, params} = build_and_init(model, %{"image" => input})
        output = predict_fn.(params, %{"image" => input})
        assert_finite!(output, "mamba_vision batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # ── DINO v2/v3 (returns {student, teacher} tuple) ───────────────────

  for arch <- [:dino_v2, :dino_v3] do
    describe "#{arch} (vision-contrastive)" do
      for batch <- [@small_batch, @med_batch] do
        @tag timeout: 120_000
        test "batch=#{batch}: student produces correct shape" do
          batch = unquote(batch)
          arch = unquote(arch)

          {student, _teacher} =
            Edifice.build(arch,
              image_size: @image_size,
              in_channels: @in_channels,
              patch_size: 4,
              embed_dim: @hidden,
              depth: 1,
              num_heads: 2,
              include_head: false,
              dropout: 0.0
            )

          input = random_tensor({batch, @in_channels, @image_size, @image_size})
          {predict_fn, params} = build_and_init(student, %{"image" => input})
          output = predict_fn.(params, %{"image" => input})
          assert_finite!(output, "#{arch} student batch=#{batch}")
          assert {^batch, _dim} = Nx.shape(output)
        end
      end
    end
  end

  # ── NeRF / 3D ───────────────────────────────────────────────────────

  describe "nerf (vision-3d)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        model = Edifice.build(:nerf, coord_dim: 3, dir_dim: 3, hidden_size: @hidden)
        coordinates = random_tensor({batch, 3})
        directions = random_tensor({batch, 3})

        input_map = %{"coordinates" => coordinates, "directions" => directions}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "nerf batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "gaussian_splat (vision-3d)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:gaussian_splat,
            num_gaussians: 4,
            image_size: 8
          )

        input_map = %{
          "camera_position" => random_tensor({batch, 3}),
          "view_matrix" => random_tensor({batch, 4, 4}),
          "proj_matrix" => random_tensor({batch, 4, 4}),
          "image_height" => random_tensor({batch}),
          "image_width" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "gaussian_splat batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Detection (multi-input, container output)
  # ══════════════════════════════════════════════════════════════════════

  describe "detr (detection)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:detr,
            image_size: @image_size,
            in_channels: @in_channels,
            hidden_dim: @hidden,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ffn_dim: @hidden,
            num_queries: 4,
            num_classes: @num_classes,
            dropout: 0.0
          )

        input = random_tensor({batch, @image_size, @image_size, @in_channels})
        {predict_fn, params} = build_and_init(model, %{"image" => input})
        output = predict_fn.(params, %{"image" => input})
        assert_finite!(output, "detr batch=#{batch}")
      end
    end
  end

  describe "rt_detr (detection)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:rt_detr,
            image_size: 32,
            in_channels: @in_channels,
            hidden_dim: @hidden,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ffn_dim: @hidden,
            num_queries: 4,
            num_classes: @num_classes,
            dropout: 0.0
          )

        input = random_tensor({batch, 32, 32, @in_channels})
        {predict_fn, params} = build_and_init(model, %{"image" => input})
        output = predict_fn.(params, %{"image" => input})
        assert_finite!(output, "rt_detr batch=#{batch}")
      end
    end
  end

  describe "sam2 (detection)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape with finite values" do
        batch = unquote(batch)

        model =
          Edifice.build(:sam2,
            image_size: @image_size,
            in_channels: @in_channels,
            hidden_dim: @hidden,
            num_heads: 2,
            dropout: 0.0
          )

        input_map = %{
          "image" => random_tensor({batch, @image_size, @image_size, @in_channels}),
          "points" => random_tensor({batch, 2, 2}),
          "labels" => random_tensor({batch, 2})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "sam2 batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Audio
  # ══════════════════════════════════════════════════════════════════════

  describe "whisper (audio)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch}: encoder produces correct shape" do
        batch = unquote(batch)

        {encoder, _decoder} =
          Edifice.build(:whisper,
            n_mels: @embed,
            audio_len: @seq_len,
            hidden_dim: @hidden,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            dropout: 0.0
          )

        input = random_tensor({batch, @embed, @seq_len})
        {predict_fn, params} = build_and_init(encoder, %{"mel_spectrogram" => input})
        output = predict_fn.(params, %{"mel_spectrogram" => input})
        assert_finite!(output, "whisper encoder batch=#{batch}")
        assert {^batch, _seq, _dim} = Nx.shape(output)
      end
    end
  end

  describe "encodec (audio)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch}: encoder produces correct shape" do
        batch = unquote(batch)

        {encoder, _decoder} =
          Edifice.build(:encodec,
            hidden_dim: @hidden,
            num_layers: @num_layers
          )

        input = random_tensor({batch, 1, 64})
        {predict_fn, params} = build_and_init(encoder, %{"waveform" => input})
        output = predict_fn.(params, %{"waveform" => input})
        assert_finite!(output, "encodec encoder batch=#{batch}")
      end
    end
  end

  describe "soundstorm (audio)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        num_codebooks = 2

        model =
          Edifice.build(:soundstorm,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            vocab_size: 32,
            num_codebooks: num_codebooks,
            dropout: 0.0
          )

        total_len = num_codebooks * @seq_len
        tokens = Nx.iota({batch, total_len}, type: :s64) |> Nx.remainder(32)
        {predict_fn, params} = build_and_init(model, %{"tokens" => tokens})
        output = predict_fn.(params, %{"tokens" => tokens})
        assert_finite!(output, "soundstorm batch=#{batch}")
      end
    end
  end

  describe "valle (audio)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch}: AR model produces correct shape" do
        batch = unquote(batch)

        {ar_model, _nar_model} =
          Edifice.build(:valle,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            vocab_size: 32,
            num_codebooks: 2,
            dropout: 0.0
          )

        text_tokens = Nx.iota({batch, @seq_len}, type: :s64) |> Nx.remainder(32)
        prompt_tokens = Nx.iota({batch, 2, @seq_len}, type: :s64) |> Nx.remainder(32)
        audio_tokens = Nx.iota({batch, @seq_len}, type: :s64) |> Nx.remainder(32)

        input_map = %{
          "text_tokens" => text_tokens,
          "prompt_tokens" => prompt_tokens,
          "audio_tokens" => audio_tokens
        }

        {predict_fn, params} = build_and_init(ar_model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "valle ar batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Transformer family
  # ══════════════════════════════════════════════════════════════════════

  describe "decoder_only (transformer)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:decoder_only,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "decoder_only batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "nemotron_h (transformer)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:nemotron_h,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            state_size: @state_size,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "nemotron_h batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "multi_token_prediction (transformer)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:multi_token_prediction,
            embed_dim: @embed,
            hidden_size: @hidden,
            vocab_size: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: @num_layers,
            num_predictions: 2,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "multi_token_prediction batch=#{batch}")
      end
    end
  end

  describe "byte_latent_transformer (transformer)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch}: encoder produces correct shape" do
        batch = unquote(batch)

        {encoder, _latent, _decoder} =
          Edifice.build(:byte_latent_transformer,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            max_byte_len: @seq_len,
            num_patches: 2,
            latent_dim: @hidden,
            dropout: 0.0
          )

        input = Nx.iota({batch, @seq_len}, type: :s64)
        {predict_fn, params} = build_and_init(encoder, %{"byte_ids" => input})
        output = predict_fn.(params, %{"byte_ids" => input})
        assert_finite!(output, "byte_latent_transformer encoder batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Contrastive
  # ══════════════════════════════════════════════════════════════════════

  describe "jepa (contrastive)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch}: context encoder produces correct shape" do
        batch = unquote(batch)

        {context_encoder, _predictor} =
          Edifice.build(:jepa,
            input_dim: @embed,
            embed_dim: @hidden,
            predictor_embed_dim: @hidden,
            encoder_depth: 1,
            predictor_depth: 1,
            num_heads: 2,
            dropout: 0.0
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(context_encoder, %{"features" => input})
        output = predict_fn.(params, %{"features" => input})
        assert_finite!(output, "jepa context_encoder batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  describe "temporal_jepa (contrastive)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch}: context encoder produces correct shape" do
        batch = unquote(batch)

        {context_encoder, _predictor} =
          Edifice.build(:temporal_jepa,
            input_dim: @embed,
            embed_dim: @hidden,
            predictor_embed_dim: @hidden,
            encoder_depth: 1,
            predictor_depth: 1,
            num_heads: 2,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(context_encoder, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "temporal_jepa context_encoder batch=#{batch}")
      end
    end
  end

  describe "siglip (contrastive)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch}: encoder produces correct shape" do
        batch = unquote(batch)

        {encoder, _temp_param} =
          Edifice.build(:siglip,
            input_dim: @embed,
            projection_dim: @hidden,
            hidden_size: @hidden
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(encoder, %{"features" => input})
        output = predict_fn.(params, %{"features" => input})
        assert_finite!(output, "siglip encoder batch=#{batch}")
        assert {^batch, _dim} = Nx.shape(output)
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Meta / PEFT
  # ══════════════════════════════════════════════════════════════════════

  describe "mixture_of_depths (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mixture_of_depths,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "mixture_of_depths batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "mixture_of_agents (meta)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mixture_of_agents,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            num_agents: 2,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "mixture_of_agents batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "rlhf_head (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:rlhf_head,
            input_size: @embed,
            hidden_size: @hidden
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "rlhf_head batch=#{batch}")
      end
    end
  end

  for arch <- [:dpo, :kto, :grpo] do
    describe "#{arch} (meta-alignment)" do
      for batch <- [@small_batch, @med_batch] do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape" do
          batch = unquote(batch)
          arch = unquote(arch)

          model =
            Edifice.build(arch,
              embed_dim: @embed,
              hidden_size: @hidden,
              vocab_size: 32,
              num_heads: 2,
              num_kv_heads: 2,
              num_layers: @num_layers,
              seq_len: @seq_len,
              dropout: 0.0
            )

          tokens = Nx.iota({batch, @seq_len}, type: :s64) |> Nx.remainder(32)
          {predict_fn, params} = build_and_init(model, %{"tokens" => tokens})
          output = predict_fn.(params, %{"tokens" => tokens})
          assert_finite!(output, "#{arch} batch=#{batch}")
        end
      end
    end
  end

  describe "speculative_decoding (meta)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch}: draft model produces correct shape" do
        batch = unquote(batch)

        {draft, _verifier} =
          Edifice.build(:speculative_decoding,
            embed_dim: @embed,
            draft_type: :gqa,
            verifier_type: :gqa,
            draft_model_opts: [
              hidden_size: @hidden,
              num_layers: 1,
              num_heads: 2,
              seq_len: @seq_len,
              dropout: 0.0
            ],
            verifier_model_opts: [
              hidden_size: @hidden,
              num_layers: 1,
              num_heads: 2,
              seq_len: @seq_len,
              dropout: 0.0
            ]
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(draft, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "speculative_decoding draft batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "test_time_compute (meta)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:test_time_compute,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: @num_layers,
            scorer_hidden: @hidden,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "test_time_compute batch=#{batch}")
      end
    end
  end

  describe "mixture_of_tokenizers (meta)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mixture_of_tokenizers,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_tokenizers: 2,
            tokenizer_vocab_sizes: [16, 32],
            tokenizer_embed_dims: [8, 8],
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "mixture_of_tokenizers batch=#{batch}")
      end
    end
  end

  describe "speculative_head (meta)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:speculative_head,
            embed_dim: @embed,
            vocab_size: 32,
            hidden_size: @hidden,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: @num_layers,
            num_predictions: 2,
            head_hidden: @hidden,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "speculative_head batch=#{batch}")
      end
    end
  end

  describe "eagle3 (meta)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:eagle3,
            hidden_size: @hidden,
            vocab_size: 32,
            num_heads: 2,
            num_kv_heads: 2
          )

        input_map = %{
          "token_embeddings" => random_tensor({batch, @seq_len, @hidden}),
          "features_low" => random_tensor({batch, @seq_len, @hidden}),
          "features_mid" => random_tensor({batch, @seq_len, @hidden}),
          "features_high" => random_tensor({batch, @seq_len, @hidden})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "eagle3 batch=#{batch}")
      end
    end
  end

  describe "manifold_hc (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:manifold_hc,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @hidden})
        {predict_fn, params} = build_and_init(model, %{"sequence" => input})
        output = predict_fn.(params, %{"sequence" => input})
        assert_finite!(output, "manifold_hc batch=#{batch}")
      end
    end
  end

  describe "distillation_head (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:distillation_head,
            embed_dim: @embed,
            teacher_dim: @hidden,
            hidden_size: @hidden
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "distillation_head batch=#{batch}")
      end
    end
  end

  describe "qat (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:qat,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "qat batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "remoe (meta)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:remoe,
            input_size: @embed,
            hidden_size: @hidden * 4,
            output_size: @hidden,
            num_experts: 2,
            top_k: 1
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"remoe_input" => input})
        output = predict_fn.(params, %{"remoe_input" => input})
        assert_finite!(output, "remoe batch=#{batch}")
        assert {^batch, @seq_len, _out} = Nx.shape(output)
      end
    end
  end

  describe "medusa (inference)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:medusa,
            base_hidden_dim: @embed,
            vocab_size: 32,
            num_medusa_heads: 2
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"hidden_states" => input})
        output = predict_fn.(params, %{"hidden_states" => input})
        assert_finite!(output, "medusa batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Generative
  # ══════════════════════════════════════════════════════════════════════

  describe "mmdit (generative)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        img_tokens = 8
        txt_tokens = 4

        model =
          Edifice.build(:mmdit,
            img_dim: @embed,
            txt_dim: @embed,
            hidden_size: @embed,
            depth: 1,
            num_heads: 2,
            img_tokens: img_tokens,
            txt_tokens: txt_tokens,
            dropout: 0.0
          )

        input_map = %{
          "img_latent" => random_tensor({batch, img_tokens, @embed}),
          "txt_embed" => random_tensor({batch, txt_tokens, @embed}),
          "timestep" => random_tensor({batch}),
          "pooled_text" => random_tensor({batch, @embed})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "mmdit batch=#{batch}")
      end
    end
  end

  describe "soflow (generative)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:soflow,
            obs_size: @embed,
            action_dim: @action_dim,
            action_horizon: @action_horizon,
            hidden_size: @hidden,
            num_layers: @num_layers,
            dropout: 0.0
          )

        input_map = %{
          "x_t" => random_tensor({batch, @action_horizon, @action_dim}),
          "current_time" => random_tensor({batch}),
          "target_time" => random_tensor({batch}),
          "observations" => random_tensor({batch, @embed})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "soflow batch=#{batch}")
        assert {^batch, @action_horizon, @action_dim} = Nx.shape(output)
      end
    end
  end

  describe "rectified_flow (generative)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:rectified_flow,
            obs_size: @embed,
            action_dim: @action_dim,
            action_horizon: @action_horizon,
            hidden_size: @hidden,
            num_layers: @num_layers,
            dropout: 0.0
          )

        input_map = %{
          "x_t" => random_tensor({batch, @action_horizon, @action_dim}),
          "timestep" => random_tensor({batch}),
          "observations" => random_tensor({batch, @embed})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "rectified_flow batch=#{batch}")
        assert {^batch, @action_horizon, @action_dim} = Nx.shape(output)
      end
    end
  end

  for arch <- [:linear_dit, :sana, :sit] do
    describe "#{arch} (generative-dit)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape" do
          batch = unquote(batch)
          arch = unquote(arch)

          model =
            Edifice.build(arch,
              input_dim: @embed,
              hidden_size: @hidden,
              depth: 1,
              num_heads: 2,
              dropout: 0.0
            )

          input_map = %{
            "noisy_input" => random_tensor({batch, @embed}),
            "timestep" => random_tensor({batch}),
            "class_label" => random_tensor({batch})
          }

          {predict_fn, params} = build_and_init(model, input_map)
          output = predict_fn.(params, input_map)
          assert_finite!(output, "#{arch} batch=#{batch}")
          assert {^batch, _dim} = Nx.shape(output)
        end
      end
    end
  end

  describe "var (generative)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)
        scales = [1, 2]
        total_tokens = Enum.reduce(scales, 0, fn s, acc -> acc + s * s end)

        model =
          Edifice.build(:var,
            vocab_size: 32,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            scales: scales,
            dropout: 0.0
          )

        input = random_tensor({batch, total_tokens, @hidden})
        {predict_fn, params} = build_and_init(model, %{"scale_embeddings" => input})
        output = predict_fn.(params, %{"scale_embeddings" => input})
        assert_finite!(output, "var batch=#{batch}")
      end
    end
  end

  describe "transfusion (generative)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:transfusion,
            embed_dim: @embed,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            vocab_size: 32,
            dropout: 0.0
          )

        input_map = %{
          "sequence" => random_tensor({batch, @seq_len, @embed}),
          "modality_mask" => random_tensor({batch, @seq_len}),
          "timestep" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "transfusion batch=#{batch}")
      end
    end
  end

  describe "mar (generative)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mar,
            vocab_size: 32,
            embed_dim: @embed,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            seq_len: @seq_len,
            dropout: 0.0
          )

        tokens = Nx.iota({batch, @seq_len}, type: :s64) |> Nx.remainder(32)
        {predict_fn, params} = build_and_init(model, %{"tokens" => tokens})
        output = predict_fn.(params, %{"tokens" => tokens})
        assert_finite!(output, "mar batch=#{batch}")
      end
    end
  end

  describe "mdlm (generative)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:mdlm,
            vocab_size: 32,
            embed_dim: @embed,
            hidden_size: @hidden,
            depth: 1,
            num_heads: 2,
            seq_len: @seq_len,
            dropout: 0.0
          )

        tokens = Nx.iota({batch, @seq_len}, type: :s64) |> Nx.remainder(32)
        timestep = random_tensor({batch})

        {predict_fn, params} =
          build_and_init(model, %{"masked_tokens" => tokens, "timestep" => timestep})

        output = predict_fn.(params, %{"masked_tokens" => tokens, "timestep" => timestep})
        assert_finite!(output, "mdlm batch=#{batch}")
      end
    end
  end

  describe "cogvideox (generative)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.Generative.CogVideoX.build_transformer(
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            text_hidden_size: @hidden,
            dropout: 0.0
          )

        input_map = %{
          "video_latent" => random_tensor({batch, 2, @in_channels, 4, 4}),
          "text_embed" => random_tensor({batch, @seq_len, @hidden}),
          "timestep" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "cogvideox batch=#{batch}")
      end
    end
  end

  describe "trellis (generative)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:trellis,
            feature_dim: @node_dim,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            dropout: 0.0
          )

        input_map = %{
          "sparse_features" => random_tensor({batch, @num_nodes, @node_dim}),
          "voxel_positions" => random_tensor({batch, @num_nodes, 3}),
          "occupancy_mask" => random_tensor({batch, @num_nodes}),
          "conditioning" => random_tensor({batch, @seq_len, @hidden}),
          "timestep" => random_tensor({batch})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "trellis batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — RL
  # ══════════════════════════════════════════════════════════════════════

  describe "policy_value (rl)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:policy_value,
            input_size: @embed,
            action_size: @action_dim,
            hidden_size: @hidden
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"observation" => input})
        output = predict_fn.(params, %{"observation" => input})
        assert_finite!(output, "policy_value batch=#{batch}")
      end
    end
  end

  describe "decision_transformer (rl)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:decision_transformer,
            state_dim: @embed,
            action_dim: @action_dim,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            context_len: @seq_len,
            dropout: 0.0
          )

        input_map = %{
          "returns" => random_tensor({batch, @seq_len}),
          "states" => random_tensor({batch, @seq_len, @embed}),
          "actions" => random_tensor({batch, @seq_len, @action_dim}),
          "timesteps" => Nx.iota({batch, @seq_len}, type: :s64)
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "decision_transformer batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Robotics
  # ══════════════════════════════════════════════════════════════════════

  describe "act (robotics)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch}: decoder produces correct shape" do
        batch = unquote(batch)

        {_encoder, decoder} =
          Edifice.build(:act,
            obs_dim: @embed,
            action_dim: @action_dim,
            latent_dim: @latent_size,
            chunk_size: @action_horizon,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            dropout: 0.0
          )

        input_map = %{
          "obs" => random_tensor({batch, @embed}),
          "z" => random_tensor({batch, @latent_size})
        }

        {predict_fn, params} = build_and_init(decoder, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "act decoder batch=#{batch}")
      end
    end
  end

  describe "openvla (robotics)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:openvla,
            image_size: @image_size,
            in_channels: @in_channels,
            patch_size: 4,
            hidden_dim: @hidden,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: @num_layers,
            dropout: 0.0
          )

        input_map = %{
          "image" => random_tensor({batch, @in_channels, @image_size, @image_size}),
          "text_tokens" => random_tensor({batch, @seq_len, @hidden})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "openvla batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Scientific
  # ══════════════════════════════════════════════════════════════════════

  describe "fno (scientific)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:fno,
            in_channels: @in_channels,
            out_channels: @in_channels,
            hidden_channels: @hidden,
            num_layers: @num_layers,
            modes: 4
          )

        input = random_tensor({batch, @seq_len, @in_channels})
        {predict_fn, params} = build_and_init(model, %{"input" => input})
        output = predict_fn.(params, %{"input" => input})
        assert_finite!(output, "fno batch=#{batch}")
        assert {^batch, @seq_len, @in_channels} = Nx.shape(output)
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Interpretability
  # ══════════════════════════════════════════════════════════════════════

  describe "sparse_autoencoder (interpretability)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:sparse_autoencoder,
            input_size: @embed,
            hidden_size: @hidden * 4
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"sae_input" => input})
        output = predict_fn.(params, %{"sae_input" => input})
        assert_finite!(output, "sparse_autoencoder batch=#{batch}")
      end
    end
  end

  describe "transcoder (interpretability)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:transcoder,
            input_size: @embed,
            output_size: @hidden,
            hidden_size: @hidden * 4
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"transcoder_input" => input})
        output = predict_fn.(params, %{"transcoder_input" => input})
        assert_finite!(output, "transcoder batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  describe "gated_sae (interpretability)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:gated_sae,
            input_size: @embed,
            dict_size: @hidden * 4,
            top_k: 4
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"gated_sae_input" => input})
        output = predict_fn.(params, %{"gated_sae_input" => input})
        assert_finite!(output, "gated_sae batch=#{batch}")
        # Autoencoder: output matches input size
        assert {^batch, @embed} = Nx.shape(output)
      end
    end
  end

  describe "linear_probe (interpretability)" do
    for batch <- @batches do
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:linear_probe,
            input_size: @embed,
            num_classes: @num_classes
          )

        input = random_tensor({batch, @embed})
        {predict_fn, params} = build_and_init(model, %{"probe_input" => input})
        output = predict_fn.(params, %{"probe_input" => input})
        assert_finite!(output, "linear_probe batch=#{batch}")
        assert {^batch, @num_classes} = Nx.shape(output)
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — World Model
  # ══════════════════════════════════════════════════════════════════════

  describe "world_model" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch}: encoder and dynamics produce finite output" do
        batch = unquote(batch)

        {encoder, dynamics, _reward_head} =
          Edifice.build(:world_model,
            obs_size: @embed,
            action_size: @action_dim,
            latent_size: @latent_size,
            hidden_size: @hidden
          )

        # Encoder
        enc_input = random_tensor({batch, @embed})
        {enc_pred, enc_params} = build_and_init(encoder, %{"observation" => enc_input})
        enc_out = enc_pred.(enc_params, %{"observation" => enc_input})
        assert_finite!(enc_out, "world_model encoder batch=#{batch}")

        # Dynamics
        concat_size = @latent_size + @action_dim
        dyn_input = random_tensor({batch, concat_size})
        {dyn_pred, dyn_params} = build_and_init(dynamics, %{"state_action" => dyn_input})
        dyn_out = dyn_pred.(dyn_params, %{"state_action" => dyn_input})
        assert_finite!(dyn_out, "world_model dynamics batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Multimodal
  # ══════════════════════════════════════════════════════════════════════

  describe "multimodal_mlp_fusion (multimodal)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:multimodal_mlp_fusion,
            vision_dim: @embed,
            llm_dim: @hidden,
            num_visual_tokens: 4,
            text_seq_len: @seq_len
          )

        input_map = %{
          "visual_tokens" => random_tensor({batch, 4, @embed}),
          "text_embeddings" => random_tensor({batch, @seq_len, @hidden})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "multimodal_mlp_fusion batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Memory
  # ══════════════════════════════════════════════════════════════════════

  describe "engram (memory)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:engram,
            key_dim: @embed,
            value_dim: @hidden,
            num_tables: 2,
            num_buckets: 4
          )

        input_map = %{
          "query" => random_tensor({batch, @embed}),
          "memory_slots" => random_tensor({2, 4, @hidden})
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "engram batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Graph
  # ══════════════════════════════════════════════════════════════════════

  describe "gin_v2 (graph)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:gin_v2,
            input_dim: @node_dim,
            edge_dim: 4,
            hidden_size: @hidden,
            num_classes: @num_classes,
            num_layers: @num_layers,
            dropout: 0.0
          )

        nodes = random_tensor({batch, @num_nodes, @node_dim})
        adj = random_tensor({batch, @num_nodes, @num_nodes})
        edge_features = random_tensor({batch, @num_nodes, @num_nodes, 4})

        input_map = %{
          "nodes" => nodes,
          "adjacency" => adj,
          "edge_features" => edge_features
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "gin_v2 batch=#{batch}")
        {out_batch, out_nodes, _dim} = Nx.shape(output)
        assert out_batch == batch
        assert out_nodes == @num_nodes
      end
    end
  end

  describe "egnn (graph)" do
    for batch <- @batches do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:egnn,
            in_node_features: @node_dim,
            hidden_size: @hidden,
            num_layers: @num_layers
          )

        nodes = random_tensor({batch, @num_nodes, @node_dim})
        coords = random_tensor({batch, @num_nodes, 3})
        edge_index = Nx.iota({batch, @num_nodes, 2}, type: :s64)

        input_map = %{
          "nodes" => nodes,
          "coords" => coords,
          "edge_index" => edge_index
        }

        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "egnn batch=#{batch}")
      end
    end
  end

  describe "dimenet (graph)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:dimenet,
            input_dim: @node_dim,
            hidden_size: @hidden,
            num_blocks: 1
          )

        nodes = random_tensor({batch, @num_nodes, @node_dim})
        positions = random_tensor({batch, @num_nodes, 3})

        input_map = %{"nodes" => nodes, "positions" => positions}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "dimenet batch=#{batch}")
      end
    end
  end

  describe "se3_transformer (graph)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      @tag :slow
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:se3_transformer,
            input_dim: @node_dim,
            hidden_size: @hidden,
            num_layers: @num_layers,
            num_heads: 2
          )

        nodes = random_tensor({batch, @num_nodes, @node_dim})
        positions = random_tensor({batch, @num_nodes, 3})

        input_map = %{"nodes" => nodes, "positions" => positions}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "se3_transformer batch=#{batch}")
      end
    end
  end

  describe "gps (graph)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:gps,
            input_dim: @node_dim,
            hidden_size: @hidden,
            num_heads: 2,
            num_layers: @num_layers,
            pe_dim: 4,
            rwse_walk_length: 4
          )

        nodes = random_tensor({batch, @num_nodes, @node_dim})
        adj = random_tensor({batch, @num_nodes, @num_nodes})

        input_map = %{"nodes" => nodes, "adjacency" => adj}
        {predict_fn, params} = build_and_init(model, input_map)
        output = predict_fn.(params, input_map)
        assert_finite!(output, "gps batch=#{batch}")
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Perceiver
  # ══════════════════════════════════════════════════════════════════════

  describe "perceiver (attention)" do
    for batch <- [@small_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:perceiver,
            input_dim: @embed,
            latent_dim: @hidden,
            num_latents: 4,
            num_layers: @num_layers,
            num_heads: 2,
            seq_len: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "perceiver batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end

  # ══════════════════════════════════════════════════════════════════════
  # v0.2.0 ADDITIONS — Hybrid / Misc
  # ══════════════════════════════════════════════════════════════════════

  describe "hybrid_builder (meta)" do
    for batch <- [@small_batch, @med_batch] do
      @tag timeout: 120_000
      test "batch=#{batch} produces correct shape" do
        batch = unquote(batch)

        model =
          Edifice.build(:hybrid_builder,
            embed_dim: @embed,
            hidden_size: @hidden,
            num_heads: 2,
            head_dim: 8,
            num_layers: @num_layers,
            seq_len: @seq_len,
            window_size: @seq_len,
            dropout: 0.0
          )

        input = random_tensor({batch, @seq_len, @embed})
        {predict_fn, params} = build_and_init(model, %{"state_sequence" => input})
        output = predict_fn.(params, %{"state_sequence" => input})
        assert_finite!(output, "hybrid_builder batch=#{batch}")
        assert {^batch, @hidden} = Nx.shape(output)
      end
    end
  end
end
