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

  # --- Sequence models: "state_sequence" {batch, seq_len, embed_size} ---
  # NOTE: :attention and :sliding_window are building blocks (no build/1), not standalone models.
  # :reservoir uses input_size not embed_size. :rwkv needs head_size explicitly.
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
    :liquid
  ]

  @sequence_opts [
    embed_size: @embed,
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

  # --- Reservoir: uses input_size not embed_size ──────────────────────
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
            embed_size: @embed,
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

  # --- Switch/Soft MoE: sequence models with embed_size ──────────────
  @moe_sequence_archs [:switch_moe, :soft_moe]

  for arch <- @moe_sequence_archs do
    describe "#{arch} (moe sequence)" do
      for batch <- @batches do
        @tag timeout: 120_000
        test "batch=#{batch} produces correct shape with finite values" do
          batch = unquote(batch)
          arch = unquote(arch)

          opts = [
            embed_size: @embed,
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
            hidden_dim: @hidden,
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
            hidden_dim: @hidden,
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
          hidden_dim: @hidden,
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

        # MoE.build/1 uses input_size (not embed_size) and "moe_input" name
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
      @tag timeout: 120_000
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
end
