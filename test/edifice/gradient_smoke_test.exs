defmodule Edifice.GradientSmokeTest do
  @moduledoc """
  Gradient smoke tests for every architecture in the registry.
  Verifies that gradients flow through each model by:

  1. Building in :inference mode (predict_fn returns plain tensor)
  2. Computing scalar loss via Nx.mean of forward pass output
  3. Taking gradients w.r.t. all parameters via value_and_grad
  4. Asserting gradients are finite (no NaN/Inf)
  5. Asserting at least some gradients are non-zero (no dead layers)

  Strategy B: Catches graph disconnections, dead layers, and broken backprop.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  # Use batch=2 (>1 to catch batch dim issues, small for speed)
  @batch 2

  # Shared small dims — keep even smaller than sweep for speed
  @embed 16
  @hidden 8
  @seq_len 4
  @state_size 4
  @num_layers 1
  @image_size 16
  @in_channels 1
  @num_nodes 4
  @node_dim 8
  @num_classes 4
  @num_points 8
  @point_dim 3
  @num_memories 4
  @memory_dim 4
  @latent_size 4
  @action_dim 4
  @action_horizon 4

  # ── Gradient checker ────────────────────────────────────────────

  defp check_gradients(model, input_map, opts \\ []) do
    compiler = Keyword.get(opts, :compiler)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    template =
      Map.new(input_map, fn {name, tensor} ->
        {name, Nx.template(Nx.shape(tensor), Nx.type(tensor))}
      end)

    model_state = init_fn.(template, Axon.ModelState.empty())
    params_data = model_state.data

    loss_fn = fn params ->
      state = %{model_state | data: params}
      output = predict_fn.(state, input_map)
      Nx.mean(output)
    end

    prev_opts = if compiler, do: Nx.Defn.default_options(compiler: compiler)
    {loss, grads} = Nx.Defn.value_and_grad(params_data, loss_fn)
    if compiler, do: Nx.Defn.default_options(prev_opts)

    # Assert loss is finite
    assert_finite!(loss, "loss")

    # Assert gradients exist and are meaningful
    flat_grads = flatten_params(grads)
    assert flat_grads != [], "model has no trainable parameters"

    # Assert all gradients are finite
    Enum.each(flat_grads, fn {path, grad_tensor} ->
      has_nan = Nx.any(Nx.is_nan(grad_tensor)) |> Nx.to_number()
      has_inf = Nx.any(Nx.is_infinity(grad_tensor)) |> Nx.to_number()
      assert has_nan == 0, "gradient NaN at #{path}"
      assert has_inf == 0, "gradient Inf at #{path}"
    end)

    # Assert at least some gradients are non-zero (no dead layers)
    any_nonzero =
      Enum.any?(flat_grads, fn {_path, grad_tensor} ->
        Nx.any(Nx.not_equal(grad_tensor, 0)) |> Nx.to_number() == 1
      end)

    assert any_nonzero, "all gradients are zero — model may have dead/disconnected layers"
  end

  # ── Sequence Models ──────────────────────────────────────────────

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
    embed_dim: @embed,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: 4,
    num_heads: 2,
    dropout: 0.0
  ]

  for arch <- @sequence_archs do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model = Edifice.build(unquote(arch), @sequence_opts)
      input = random_tensor({@batch, @seq_len, @embed})
      check_gradients(model, %{"state_sequence" => input})
    end
  end

  # Reservoir: fixed random weights — no trainable params, skip gradient test

  @tag timeout: 120_000
  test "gradient flows through rwkv" do
    model =
      Edifice.build(:rwkv,
        embed_dim: @embed,
        hidden_size: @hidden,
        head_size: 4,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── MoE variants ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through moe" do
    model =
      Edifice.build(:moe,
        input_size: @embed,
        hidden_size: @hidden * 4,
        output_size: @hidden,
        num_experts: 2,
        top_k: 1
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"moe_input" => input})
  end

  for arch <- [:switch_moe, :soft_moe] do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
          embed_dim: @embed,
          hidden_size: @hidden,
          num_layers: @num_layers,
          seq_len: @seq_len,
          num_experts: 2,
          dropout: 0.0
        )

      input = random_tensor({@batch, @seq_len, @embed})
      check_gradients(model, %{"state_sequence" => input})
    end
  end

  # ── Feedforward Models ──────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through mlp" do
    model = Edifice.build(:mlp, input_size: @embed, hidden_sizes: [@hidden])
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through tabnet" do
    model = Edifice.build(:tabnet, input_size: @embed, output_size: @num_classes)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # ── Vision Models ──────────────────────────────────────────────

  for arch <- [:vit, :deit, :mlp_mixer] do
    @tag timeout: 120_000
    @tag :slow
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
          image_size: @image_size,
          in_channels: @in_channels,
          patch_size: 4,
          embed_dim: @hidden,
          hidden_size: @hidden,
          depth: 1,
          num_heads: 2,
          dropout: 0.0
        )

      input = random_tensor({@batch, @in_channels, @image_size, @image_size})
      check_gradients(model, %{"image" => input})
    end
  end

  @tag timeout: 120_000
  test "gradient flows through swin" do
    model =
      Edifice.build(:swin,
        image_size: 16,
        in_channels: @in_channels,
        patch_size: 4,
        embed_dim: @hidden,
        depths: [1],
        num_heads: [2],
        window_size: 4,
        dropout: 0.0
      )

    input = random_tensor({@batch, @in_channels, 16, 16})
    check_gradients(model, %{"image" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through unet" do
    model =
      Edifice.build(:unet,
        in_channels: @in_channels,
        out_channels: 1,
        image_size: @image_size,
        base_features: 4,
        depth: 2,
        dropout: 0.0
      )

    input = random_tensor({@batch, @in_channels, @image_size, @image_size})
    check_gradients(model, %{"image" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through convnext" do
    model =
      Edifice.build(:convnext,
        image_size: 16,
        in_channels: @in_channels,
        patch_size: 4,
        dims: [@hidden, @hidden * 2],
        depths: [1, 1],
        dropout: 0.0
      )

    input = random_tensor({@batch, @in_channels, 16, 16})
    check_gradients(model, %{"image" => input})
  end

  # ── Convolutional Models ──────────────────────────────────────

  # Conv-based models: BinaryBackend can't differentiate through Axon.Layers.conv
  # (tries to materialize traced Defn.Expr tensors). Works with EXLA.
  @tag timeout: 120_000
  @tag :exla_only
  test "gradient flows through resnet" do
    model =
      Edifice.build(:resnet,
        input_shape: {nil, @image_size, @image_size, @in_channels},
        num_classes: @num_classes,
        block_sizes: [1, 1],
        initial_channels: 4
      )

    input = random_tensor({@batch, @image_size, @image_size, @in_channels})
    check_gradients(model, %{"input" => input}, compiler: EXLA)
  end

  @tag timeout: 120_000
  @tag :exla_only
  test "gradient flows through densenet" do
    model =
      Edifice.build(:densenet,
        input_shape: {nil, 32, 32, @in_channels},
        num_classes: @num_classes,
        growth_rate: 4,
        block_config: [2],
        initial_channels: 8
      )

    input = random_tensor({@batch, 32, 32, @in_channels})
    check_gradients(model, %{"input" => input}, compiler: EXLA)
  end

  @tag timeout: 120_000
  @tag :exla_only
  test "gradient flows through tcn" do
    model = Edifice.build(:tcn, input_size: @embed, hidden_size: @hidden, num_layers: 2)
    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"input" => input}, compiler: EXLA)
  end

  @tag timeout: 120_000
  test "gradient flows through mobilenet" do
    model =
      Edifice.build(:mobilenet, input_dim: @embed, hidden_dim: @hidden, num_classes: @num_classes)

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # EfficientNet: too slow for BinaryBackend, skip

  @tag timeout: 120_000
  @tag :exla_only
  test "gradient flows through capsule" do
    model =
      Edifice.build(:capsule,
        input_shape: {nil, 28, 28, 1},
        conv_channels: 16,
        conv_kernel: 9,
        num_primary_caps: 4,
        primary_cap_dim: 4,
        num_digit_caps: @num_classes,
        digit_cap_dim: 4
      )

    input = random_tensor({@batch, 28, 28, 1})
    check_gradients(model, %{"input" => input}, compiler: EXLA)
  end

  # ── Graph Models ──────────────────────────────────────────────

  @graph_archs [:gcn, :gat, :graph_sage, :gin, :pna, :graph_transformer]

  for arch <- @graph_archs do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
          input_dim: @node_dim,
          hidden_size: @hidden,
          num_classes: @num_classes,
          num_layers: @num_layers,
          num_heads: 2,
          dropout: 0.0
        )

      nodes = random_tensor({@batch, @num_nodes, @node_dim})
      adj = random_tensor({@batch, @num_nodes, @num_nodes})
      check_gradients(model, %{"nodes" => nodes, "adjacency" => adj})
    end
  end

  @tag timeout: 120_000
  test "gradient flows through schnet" do
    model =
      Edifice.build(:schnet,
        input_dim: @node_dim,
        hidden_size: @hidden,
        num_interactions: 1,
        num_filters: @hidden,
        num_rbf: 8
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    distances = random_tensor({@batch, @num_nodes, @num_nodes})
    check_gradients(model, %{"nodes" => nodes, "adjacency" => distances})
  end

  # message_passing is not registered as a standalone architecture

  # ── Set Models ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through deep_sets" do
    model = Edifice.build(:deep_sets, input_dim: @point_dim, output_dim: @num_classes)
    input = random_tensor({@batch, @num_points, @point_dim})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through pointnet" do
    model = Edifice.build(:pointnet, num_classes: @num_classes, input_dim: @point_dim)
    input = random_tensor({@batch, @num_points, @point_dim})
    check_gradients(model, %{"input" => input})
  end

  # ── Energy / Dynamic ──────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through ebm" do
    model = Edifice.build(:ebm, input_size: @embed)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through hopfield" do
    model = Edifice.build(:hopfield, input_dim: @embed)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through neural_ode" do
    model = Edifice.build(:neural_ode, input_size: @embed, hidden_size: @hidden)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # ── Probabilistic ──────────────────────────────────────────

  @prob_archs [
    {:bayesian, [input_size: @embed, output_size: @num_classes]},
    {:mc_dropout, [input_size: @embed, output_size: @num_classes]},
    {:evidential, [input_size: @embed, num_classes: @num_classes]}
  ]

  for {arch, opts} <- @prob_archs do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model = Edifice.build(unquote(arch), unquote(opts))
      input = random_tensor({@batch, @embed})
      check_gradients(model, %{"input" => input})
    end
  end

  # ── Memory ──────────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through ntm" do
    model =
      Edifice.build(:ntm,
        input_size: @embed,
        output_size: @hidden,
        memory_size: @num_memories,
        memory_dim: @memory_dim,
        num_heads: 1
      )

    input = random_tensor({@batch, @embed})
    memory = random_tensor({@batch, @num_memories, @memory_dim})
    check_gradients(model, %{"input" => input, "memory" => memory})
  end

  @tag timeout: 120_000
  test "gradient flows through memory_network" do
    model =
      Edifice.build(:memory_network,
        input_dim: @embed,
        output_dim: @hidden,
        num_memories: @num_memories
      )

    query = random_tensor({@batch, @embed})
    memories = random_tensor({@batch, @num_memories, @embed})
    check_gradients(model, %{"query" => query, "memories" => memories})
  end

  # ── Neuromorphic ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through snn" do
    model =
      Edifice.build(:snn, input_size: @embed, output_size: @num_classes, hidden_sizes: [@hidden])

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through ann2snn" do
    model = Edifice.build(:ann2snn, input_size: @embed, output_size: @num_classes)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # ── Meta / PEFT ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through lora" do
    model = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 4)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through adapter" do
    model = Edifice.build(:adapter, hidden_size: @hidden)
    input = random_tensor({@batch, @hidden})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through hypernetwork" do
    model =
      Edifice.build(:hypernetwork,
        conditioning_size: @embed,
        target_layer_sizes: [{@embed, @hidden}],
        input_size: @embed
      )

    input_map = %{
      "conditioning" => random_tensor({@batch, @embed}),
      "data_input" => random_tensor({@batch, @embed})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  @tag :slow
  test "gradient flows through perceiver" do
    model =
      Edifice.build(:perceiver,
        input_dim: @embed,
        hidden_dim: @hidden,
        output_dim: @hidden,
        num_latents: 4,
        num_layers: @num_layers,
        num_heads: 2,
        seq_len: @seq_len
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Generative (test non-container components) ────────────────

  # VAE: test decoder (returns plain tensor; encoder returns container)
  @tag timeout: 120_000
  test "gradient flows through vae decoder" do
    {_encoder, decoder} = Edifice.build(:vae, input_size: @embed, latent_size: @latent_size)
    input = random_tensor({@batch, @latent_size})
    check_gradients(decoder, %{"latent" => input})
  end

  # VQ-VAE: encoder returns plain tensor
  @tag timeout: 120_000
  test "gradient flows through vq_vae encoder" do
    {encoder, _decoder} = Edifice.build(:vq_vae, input_size: @embed, embedding_dim: @latent_size)
    input = random_tensor({@batch, @embed})
    check_gradients(encoder, %{"input" => input})
  end

  # GAN
  @tag timeout: 120_000
  test "gradient flows through gan generator" do
    {generator, _disc} = Edifice.build(:gan, output_size: @embed, latent_size: @latent_size)
    noise = random_tensor({@batch, @latent_size})
    check_gradients(generator, %{"noise" => noise})
  end

  @tag timeout: 120_000
  test "gradient flows through gan discriminator" do
    {_gen, discriminator} = Edifice.build(:gan, output_size: @embed, latent_size: @latent_size)
    input = random_tensor({@batch, @embed})
    check_gradients(discriminator, %{"data" => input})
  end

  # Normalizing flow: single model
  @tag timeout: 120_000
  test "gradient flows through normalizing_flow" do
    model = Edifice.build(:normalizing_flow, input_size: @embed, num_flows: 2)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # Latent diffusion: test decoder (plain tensor) and denoiser
  @tag timeout: 120_000
  test "gradient flows through latent_diffusion decoder" do
    {_encoder, decoder, _denoiser} =
      Edifice.build(:latent_diffusion,
        input_size: @embed,
        latent_size: @latent_size,
        hidden_size: @hidden,
        num_layers: @num_layers
      )

    input = random_tensor({@batch, @latent_size})
    check_gradients(decoder, %{"latent" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through latent_diffusion denoiser" do
    {_encoder, _decoder, denoiser} =
      Edifice.build(:latent_diffusion,
        input_size: @embed,
        latent_size: @latent_size,
        hidden_size: @hidden,
        num_layers: @num_layers
      )

    input_map = %{
      "noisy_z" => random_tensor({@batch, @latent_size}),
      "timestep" => random_tensor({@batch})
    }

    check_gradients(denoiser, input_map)
  end

  # BYOL: test online encoder
  @tag timeout: 120_000
  test "gradient flows through byol online_encoder" do
    {online, _target} = Edifice.build(:byol, encoder_dim: @embed, projection_dim: @hidden)
    input = random_tensor({@batch, @embed})
    check_gradients(online, %{"features" => input})
  end

  # MAE: test encoder
  @tag timeout: 120_000
  test "gradient flows through mae encoder" do
    {encoder, _decoder} =
      Edifice.build(:mae,
        input_dim: @embed,
        embed_dim: @hidden,
        num_patches: 4,
        depth: 1,
        num_heads: 2,
        decoder_depth: 1,
        decoder_num_heads: 2
      )

    input = random_tensor({@batch, 4, @embed})
    check_gradients(encoder, %{"visible_patches" => input})
  end

  # ── Contrastive (single model) ──────────────────────────────

  for arch <- [:simclr, :barlow_twins, :vicreg] do
    @tag timeout: 120_000
    @tag :slow
    test "gradient flows through #{arch}" do
      model = Edifice.build(unquote(arch), encoder_dim: @embed, projection_dim: @hidden)
      input = random_tensor({@batch, @embed})
      check_gradients(model, %{"features" => input})
    end
  end

  # ── Diffusion family ────────────────────────────────────────

  for arch <- [:diffusion, :ddim] do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
          obs_size: @embed,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: @hidden,
          num_layers: @num_layers,
          dropout: 0.0
        )

      input_map = %{
        "noisy_actions" => random_tensor({@batch, @action_horizon, @action_dim}),
        "timestep" => random_tensor({@batch}),
        "observations" => random_tensor({@batch, @embed})
      }

      check_gradients(model, input_map)
    end
  end

  @tag timeout: 120_000
  test "gradient flows through flow_matching" do
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
      "x_t" => random_tensor({@batch, @action_horizon, @action_dim}),
      "timestep" => random_tensor({@batch}),
      "observations" => random_tensor({@batch, @embed})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through dit" do
    model =
      Edifice.build(:dit,
        input_dim: @embed,
        hidden_size: @hidden,
        depth: 1,
        num_heads: 2,
        dropout: 0.0
      )

    input_map = %{
      "noisy_input" => random_tensor({@batch, @embed}),
      "timestep" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through score_sde" do
    model =
      Edifice.build(:score_sde, input_dim: @embed, hidden_size: @hidden, num_layers: @num_layers)

    input_map = %{
      "noisy_input" => random_tensor({@batch, @embed}),
      "timestep" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through consistency_model" do
    model =
      Edifice.build(:consistency_model,
        input_dim: @embed,
        hidden_size: @hidden,
        num_layers: @num_layers
      )

    input_map = %{
      "noisy_input" => random_tensor({@batch, @embed}),
      "sigma" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end
end
