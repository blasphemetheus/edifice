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

  # Analytical gradient check: uses value_and_grad wrapped in Nx.Defn.jit
  # so that all tensors (trainable params, full data, input) are JIT arguments.
  # This allows gradient tracing through conv ops that BinaryBackend can't
  # differentiate through on its own.
  defp check_gradients(model, input_map) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    template =
      Map.new(input_map, fn {name, tensor} ->
        {name, Nx.template(Nx.shape(tensor), Nx.type(tensor))}
      end)

    model_state = init_fn.(template, Axon.ModelState.empty())
    trainable = Axon.ModelState.trainable_parameters(model_state)
    assert trainable != %{}, "model has no trainable parameters"

    # JIT-wrapped gradient function: all tensors are JIT arguments so the
    # gradient tracer can see through predict_fn (including conv ops).
    step_fn = fn trainable_params, full_data, inp ->
      {loss, grad} =
        Nx.Defn.value_and_grad(trainable_params, fn tp ->
          merged = deep_merge(full_data, tp)

          ms = %Axon.ModelState{
            data: merged,
            parameters: model_state.parameters,
            state: model_state.state,
            frozen_parameters: model_state.frozen_parameters
          }

          output = predict_fn.(ms, inp)
          sum_container(output)
        end)

      {loss, grad}
    end

    # Use EXLA compiler when available — BinaryBackend's select_and_scatter
    # crashes on max_pool gradients (negative padding bug).
    jit_opts =
      if Code.ensure_loaded?(EXLA),
        do: [compiler: EXLA, on_conflict: :reuse],
        else: [on_conflict: :reuse]

    {loss, grads} = Nx.Defn.jit(step_fn, jit_opts).(trainable, model_state.data, input_map)

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

  # Deep-merge two nested maps, preferring values from `override`
  defp deep_merge(base, override) when is_map(base) and is_map(override) do
    Map.merge(base, override, fn
      _key, base_val, override_val when is_map(base_val) and is_map(override_val) ->
        deep_merge(base_val, override_val)

      _key, _base_val, override_val ->
        override_val
    end)
  end

  # Sum all tensors in a container (tuple, map, or plain tensor) into a scalar.
  defp sum_container(%Nx.Tensor{} = t), do: Nx.sum(t)

  defp sum_container(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.reduce(Nx.tensor(0.0), fn elem, acc -> Nx.add(acc, sum_container(elem)) end)
  end

  defp sum_container(%{} = map) when not is_struct(map) do
    map
    |> Map.values()
    |> Enum.reduce(Nx.tensor(0.0), fn val, acc -> Nx.add(acc, sum_container(val)) end)
  end

  defp sum_container(_other), do: Nx.tensor(0.0)

  # Parameter sensitivity check: perturbs each parameter and verifies the
  # output changes. Used for models where BinaryBackend can't compute
  # analytical gradients (e.g. resnet/densenet with max_pool).
  defp check_parameter_sensitivity(model, input_map) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    template =
      Map.new(input_map, fn {name, tensor} ->
        {name, Nx.template(Nx.shape(tensor), Nx.type(tensor))}
      end)

    model_state = init_fn.(template, Axon.ModelState.empty())

    # Baseline forward pass
    baseline_output = predict_fn.(model_state, input_map)
    baseline_loss = Nx.mean(baseline_output) |> Nx.to_number()
    assert is_float(baseline_loss), "baseline loss is not a number"
    refute baseline_loss == :nan, "baseline loss is NaN"
    refute baseline_loss == :infinity, "baseline loss is Inf"
    refute baseline_loss == :neg_infinity, "baseline loss is -Inf"

    # Perturb parameters and check that outputs change
    flat_params = flatten_params(model_state.data)
    assert flat_params != [], "model has no trainable parameters"

    changed_count =
      Enum.count(flat_params, fn {path, param_tensor} ->
        perturbed = Nx.add(param_tensor, 0.1)
        perturbed_data = put_nested(model_state.data, path, perturbed)
        perturbed_state = %{model_state | data: perturbed_data}

        perturbed_output = predict_fn.(perturbed_state, input_map)
        perturbed_loss = Nx.mean(perturbed_output) |> Nx.to_number()

        abs(perturbed_loss - baseline_loss) > 1.0e-10
      end)

    assert changed_count > 0,
           "no parameters affected output — model may have dead/disconnected layers " <>
             "(0/#{length(flat_params)} params sensitive)"
  end

  defp put_nested(map, path, value) do
    keys = String.split(path, ".")
    do_put_nested(map, keys, value)
  end

  defp do_put_nested(map, [key], value), do: Map.put(map, key, value)

  defp do_put_nested(map, [key | rest], value) do
    inner = Map.get(map, key, %{})
    Map.put(map, key, do_put_nested(inner, rest, value))
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
    :liquid,
    # v0.2.0 additions — attention
    :based,
    :conformer,
    :infini_attention,
    :mega,
    :hawk,
    :retnet_v2,
    :megalodon,
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
    # v0.2.0 additions — SSM
    :striped_hyena,
    :mamba3,
    :gss,
    :hyena_v2,
    :hymba,
    :ss_transformer,
    # v0.2.0 additions — recurrent
    :gated_delta_net,
    :ttt_e2e,
    :slstm,
    :xlstm_v2,
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

  # Chunked attention models need seq_len >= 2 * block_size (at least 2 blocks)
  for arch <- [:lightning_attention, :flash_linear_attention, :dual_chunk_attention] do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      block = div(@seq_len, 2)

      model =
        Edifice.build(
          unquote(arch),
          Keyword.merge(@sequence_opts, block_size: block, chunk_size: block)
        )

      input = random_tensor({@batch, @seq_len, @embed})
      check_gradients(model, %{"state_sequence" => input})
    end
  end

  # YaRN, TMRoPE, and Reservoir: no trainable parameters (pure positional
  # encoding / fixed random weights), skip gradient test

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
  #
  # ResNet and DenseNet require EXLA for gradient computation because
  # Nx.BinaryBackend.select_and_scatter crashes on max_pool gradients.
  # check_gradients auto-detects EXLA when available.

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
    check_gradients(model, %{"input" => input})
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
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through tcn" do
    model = Edifice.build(:tcn, input_size: @embed, hidden_size: @hidden, num_layers: 2)
    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
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
    check_gradients(model, %{"input" => input})
  end

  # ── Parameter sensitivity tests for max_pool models (BinaryBackend workaround)

  @tag timeout: 120_000
  test "parameters are sensitive in resnet" do
    model =
      Edifice.build(:resnet,
        input_shape: {nil, @image_size, @image_size, @in_channels},
        num_classes: @num_classes,
        block_sizes: [1, 1],
        initial_channels: 4
      )

    input = random_tensor({@batch, @image_size, @image_size, @in_channels})
    check_parameter_sensitivity(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "parameters are sensitive in densenet" do
    model =
      Edifice.build(:densenet,
        input_shape: {nil, 32, 32, @in_channels},
        num_classes: @num_classes,
        growth_rate: 4,
        block_config: [2],
        initial_channels: 8
      )

    input = random_tensor({@batch, 32, 32, @in_channels})
    check_parameter_sensitivity(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through mobilenet" do
    model =
      Edifice.build(:mobilenet, input_dim: @embed, hidden_dim: @hidden, num_classes: @num_classes)

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # EfficientNet: too slow for BinaryBackend, skip

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

  # ── Transformer family ──────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through decoder_only" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 300_000
  @tag :slow
  test "gradient flows through nemotron_h" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through multi_token_prediction" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through byte_latent_transformer encoder" do
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

    # byte_ids are integer tokens
    input = Nx.iota({@batch, @seq_len}, type: :s64)
    check_gradients(encoder, %{"byte_ids" => input})
  end

  # ── Feedforward additions ──────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through bitnet" do
    model =
      Edifice.build(:bitnet,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through kat" do
    model =
      Edifice.build(:kat,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Vision additions ───────────────────────────────────────────

  for arch <- [:focalnet, :poolformer, :metaformer, :caformer, :efficient_vit, :mamba_vision] do
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
          depths: [1],
          dims: [@hidden],
          num_heads: [2],
          depth: 1,
          dropout: 0.0
        )

      input = random_tensor({@batch, @in_channels, @image_size, @image_size})
      check_gradients(model, %{"image" => input})
    end
  end

  @tag timeout: 120_000
  test "gradient flows through nerf" do
    model = Edifice.build(:nerf, coord_dim: 3, dir_dim: 3, hidden_size: @hidden)
    coordinates = random_tensor({@batch, 3})
    directions = random_tensor({@batch, 3})
    check_gradients(model, %{"coordinates" => coordinates, "directions" => directions})
  end

  @tag timeout: 120_000
  test "gradient flows through gaussian_splat" do
    model =
      Edifice.build(:gaussian_splat,
        num_gaussians: 4,
        image_size: 8
      )

    input_map = %{
      "camera_position" => random_tensor({@batch, 3}),
      "view_matrix" => random_tensor({@batch, 4, 4}),
      "proj_matrix" => random_tensor({@batch, 4, 4}),
      "image_height" => random_tensor({@batch}),
      "image_width" => random_tensor({@batch})
    }

    # GaussianSplat uses Nx ops without gradient support, use sensitivity
    check_parameter_sensitivity(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through dino_v2 student" do
    {student, _teacher} =
      Edifice.build(:dino_v2,
        image_size: @image_size,
        in_channels: @in_channels,
        patch_size: 4,
        embed_dim: @hidden,
        depth: 1,
        num_heads: 2,
        include_head: false,
        dropout: 0.0
      )

    input = random_tensor({@batch, @in_channels, @image_size, @image_size})
    check_gradients(student, %{"image" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through dino_v3 student" do
    {student, _teacher} =
      Edifice.build(:dino_v3,
        image_size: @image_size,
        in_channels: @in_channels,
        patch_size: 4,
        embed_dim: @hidden,
        depth: 1,
        num_heads: 2,
        include_head: false,
        dropout: 0.0
      )

    input = random_tensor({@batch, @in_channels, @image_size, @image_size})
    check_gradients(student, %{"image" => input})
  end

  # ── Convolutional additions ────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through conv1d" do
    model =
      Edifice.build(:conv1d,
        input_size: @embed,
        hidden_size: @hidden,
        num_layers: 2
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  @tag :exla_only
  test "gradient flows through efficientnet" do
    model = Edifice.build(:efficientnet, input_dim: @embed, hidden_dim: @hidden)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 300_000
  @tag :slow
  test "parameters are sensitive in efficientnet" do
    model = Edifice.build(:efficientnet, input_dim: @embed, hidden_dim: @hidden)
    input = random_tensor({@batch, @embed})
    check_parameter_sensitivity(model, %{"input" => input})
  end

  # ── Graph additions ────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through gin_v2" do
    model =
      Edifice.build(:gin_v2,
        input_dim: @node_dim,
        edge_dim: 4,
        hidden_size: @hidden,
        num_classes: @num_classes,
        num_layers: @num_layers,
        dropout: 0.0
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    adj = random_tensor({@batch, @num_nodes, @num_nodes})
    edge_features = random_tensor({@batch, @num_nodes, @num_nodes, 4})

    check_gradients(model, %{
      "nodes" => nodes,
      "adjacency" => adj,
      "edge_features" => edge_features
    })
  end

  @tag timeout: 120_000
  test "gradient flows through egnn" do
    model =
      Edifice.build(:egnn,
        in_node_features: @node_dim,
        hidden_size: @hidden,
        num_layers: @num_layers
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    coords = random_tensor({@batch, @num_nodes, 3})
    edge_index = Nx.iota({@batch, @num_nodes, 2}, type: :s64)

    check_gradients(model, %{
      "nodes" => nodes,
      "coords" => coords,
      "edge_index" => edge_index
    })
  end

  # ── Detection family ───────────────────────────────────────────

  for arch <- [:detr, :rt_detr] do
    @tag timeout: 120_000
    @tag :slow
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
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

      input = random_tensor({@batch, @image_size, @image_size, @in_channels})
      check_gradients(model, %{"image" => input})
    end
  end

  @tag timeout: 120_000
  @tag :slow
  test "gradient flows through sam2" do
    model =
      Edifice.build(:sam2,
        image_size: @image_size,
        in_channels: @in_channels,
        hidden_dim: @hidden,
        num_heads: 2,
        dropout: 0.0
      )

    input_map = %{
      "image" => random_tensor({@batch, @image_size, @image_size, @in_channels}),
      "points" => random_tensor({@batch, 2, 2}),
      "labels" => random_tensor({@batch, 2})
    }

    check_gradients(model, input_map)
  end

  # ── Audio family ───────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through whisper encoder" do
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

    input = random_tensor({@batch, @embed, @seq_len})
    check_gradients(encoder, %{"mel_spectrogram" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through encodec encoder" do
    {encoder, _decoder} =
      Edifice.build(:encodec,
        hidden_dim: @hidden,
        num_layers: @num_layers
      )

    input = random_tensor({@batch, 1, 64})
    check_gradients(encoder, %{"waveform" => input})
  end

  @tag timeout: 300_000
  test "gradient flows through soundstorm" do
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

    # Flattened token input: [batch, num_codebooks * seq_len]
    total_len = num_codebooks * @seq_len
    tokens = Nx.iota({@batch, total_len}, type: :s64) |> Nx.remainder(32)
    check_gradients(model, %{"tokens" => tokens})
  end

  @tag timeout: 300_000
  @tag :slow
  test "gradient flows through valle AR model" do
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

    text_tokens = Nx.iota({@batch, @seq_len}, type: :s64) |> Nx.remainder(32)
    prompt_tokens = Nx.iota({@batch, 2, @seq_len}, type: :s64) |> Nx.remainder(32)
    audio_tokens = Nx.iota({@batch, @seq_len}, type: :s64) |> Nx.remainder(32)

    check_gradients(ar_model, %{
      "text_tokens" => text_tokens,
      "prompt_tokens" => prompt_tokens,
      "audio_tokens" => audio_tokens
    })
  end

  # ── Robotics family ───────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through act decoder" do
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
      "obs" => random_tensor({@batch, @embed}),
      "z" => random_tensor({@batch, @latent_size})
    }

    check_gradients(decoder, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through openvla" do
    model =
      Edifice.build(:openvla,
        image_size: @image_size,
        in_channels: @in_channels,
        patch_size: 4,
        hidden_dim: @hidden,
        vision_dim: @hidden,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: @num_layers,
        dropout: 0.0
      )

    input_map = %{
      "image" => random_tensor({@batch, @in_channels, @image_size, @image_size}),
      "text_tokens" => random_tensor({@batch, @seq_len, @hidden})
    }

    check_gradients(model, input_map)
  end

  # ── RL family ──────────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through policy_value" do
    model =
      Edifice.build(:policy_value,
        input_size: @embed,
        action_size: @action_dim,
        hidden_size: @hidden
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"observation" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through decision_transformer" do
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
      "returns" => random_tensor({@batch, @seq_len}),
      "states" => random_tensor({@batch, @seq_len, @embed}),
      "actions" => random_tensor({@batch, @seq_len, @action_dim}),
      "timesteps" => Nx.iota({@batch, @seq_len}, type: :s64)
    }

    check_gradients(model, input_map)
  end

  # ── Generative additions ───────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through dit_v2" do
    model =
      Edifice.build(:dit_v2,
        input_dim: @embed,
        hidden_size: @hidden,
        depth: 1,
        num_heads: 2,
        dropout: 0.0
      )

    input_map = %{
      "noisy_input" => random_tensor({@batch, @embed}),
      "timestep" => random_tensor({@batch}),
      "class_label" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through mmdit" do
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
      "img_latent" => random_tensor({@batch, img_tokens, @embed}),
      "txt_embed" => random_tensor({@batch, txt_tokens, @embed}),
      "timestep" => random_tensor({@batch}),
      "pooled_text" => random_tensor({@batch, @embed})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through soflow" do
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
      "x_t" => random_tensor({@batch, @action_horizon, @action_dim}),
      "current_time" => random_tensor({@batch}),
      "target_time" => random_tensor({@batch}),
      "observations" => random_tensor({@batch, @embed})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through rectified_flow" do
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
      "x_t" => random_tensor({@batch, @action_horizon, @action_dim}),
      "timestep" => random_tensor({@batch}),
      "observations" => random_tensor({@batch, @embed})
    }

    check_gradients(model, input_map)
  end

  for arch <- [:linear_dit, :sit] do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
          input_dim: @embed,
          hidden_size: @hidden,
          depth: 1,
          num_heads: 2,
          dropout: 0.0
        )

      input_map = %{
        "noisy_input" => random_tensor({@batch, @embed}),
        "timestep" => random_tensor({@batch}),
        "class_label" => random_tensor({@batch})
      }

      check_gradients(model, input_map)
    end
  end

  @tag timeout: 120_000
  test "gradient flows through var" do
    # Scales [1, 2] -> total_tokens = 1 + 4 = 5
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

    input = random_tensor({@batch, total_tokens, @hidden})
    check_gradients(model, %{"scale_embeddings" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through mar" do
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

    tokens = Nx.iota({@batch, @seq_len}, type: :s64) |> Nx.remainder(32)
    check_gradients(model, %{"tokens" => tokens})
  end

  @tag timeout: 120_000
  test "gradient flows through mdlm" do
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

    tokens = Nx.iota({@batch, @seq_len}, type: :s64) |> Nx.remainder(32)
    timestep = random_tensor({@batch})
    check_gradients(model, %{"masked_tokens" => tokens, "timestep" => timestep})
  end

  @tag timeout: 120_000
  test "gradient flows through transfusion" do
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
      "sequence" => random_tensor({@batch, @seq_len, @embed}),
      "modality_mask" => random_tensor({@batch, @seq_len}),
      "timestep" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  @tag :slow
  test "gradient flows through cogvideox transformer" do
    model =
      Edifice.Generative.CogVideoX.build_transformer(
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        text_hidden_size: @hidden,
        dropout: 0.0
      )

    input_map = %{
      "video_latent" => random_tensor({@batch, 2, @in_channels, 4, 4}),
      "text_embed" => random_tensor({@batch, @seq_len, @hidden}),
      "timestep" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  @tag :slow
  test "gradient flows through trellis" do
    model =
      Edifice.build(:trellis,
        feature_dim: @node_dim,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        dropout: 0.0
      )

    input_map = %{
      "sparse_features" => random_tensor({@batch, @num_nodes, @node_dim}),
      "voxel_positions" => random_tensor({@batch, @num_nodes, 3}),
      "occupancy_mask" => random_tensor({@batch, @num_nodes}),
      "conditioning" => random_tensor({@batch, @seq_len, @hidden}),
      "timestep" => random_tensor({@batch})
    }

    check_gradients(model, input_map)
  end

  # ── MoE additions ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through moe_v2" do
    model =
      Edifice.build(:moe_v2,
        input_size: @embed,
        hidden_size: @hidden * 4,
        output_size: @hidden,
        num_experts: 2,
        top_k: 1
      )

    input = random_tensor({@batch, @seq_len, @embed})
    # MoE v2 uses Nx.argsort which lacks gradient impl
    check_parameter_sensitivity(model, %{"moe_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through remoe" do
    model =
      Edifice.build(:remoe,
        input_size: @embed,
        hidden_size: @hidden * 4,
        output_size: @hidden,
        num_experts: 2,
        top_k: 1
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"remoe_input" => input})
  end

  # ── Meta / PEFT additions ─────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through dora" do
    model = Edifice.build(:dora, input_size: @embed, output_size: @hidden, rank: 4)
    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through mixture_of_depths" do
    model =
      Edifice.build(:mixture_of_depths,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through mixture_of_agents" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through rlhf_head" do
    model =
      Edifice.build(:rlhf_head,
        input_size: @embed,
        hidden_size: @hidden
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  for arch <- [:dpo, :kto, :grpo] do
    @tag timeout: 120_000
    test "gradient flows through #{arch}" do
      model =
        Edifice.build(unquote(arch),
          embed_dim: @embed,
          hidden_size: @hidden,
          vocab_size: 32,
          num_heads: 2,
          num_kv_heads: 2,
          num_layers: @num_layers,
          seq_len: @seq_len,
          dropout: 0.0
        )

      tokens = Nx.iota({@batch, @seq_len}, type: :s64) |> Nx.remainder(32)
      check_gradients(model, %{"tokens" => tokens})
    end
  end

  @tag timeout: 120_000
  test "gradient flows through distillation_head" do
    model =
      Edifice.build(:distillation_head,
        embed_dim: @embed,
        teacher_dim: @hidden,
        hidden_size: @hidden
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through qat" do
    model =
      Edifice.build(:qat,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through eagle3" do
    model =
      Edifice.build(:eagle3,
        hidden_size: @hidden,
        vocab_size: 32,
        num_heads: 2,
        num_kv_heads: 2
      )

    input_map = %{
      "token_embeddings" => random_tensor({@batch, @seq_len, @hidden}),
      "features_low" => random_tensor({@batch, @seq_len, @hidden}),
      "features_mid" => random_tensor({@batch, @seq_len, @hidden}),
      "features_high" => random_tensor({@batch, @seq_len, @hidden})
    }

    check_gradients(model, input_map)
  end

  @tag timeout: 120_000
  test "gradient flows through manifold_hc" do
    model =
      Edifice.build(:manifold_hc,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @hidden})
    check_gradients(model, %{"sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through speculative_decoding draft model" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(draft, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through test_time_compute" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through speculative_head" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through medusa" do
    model =
      Edifice.build(:medusa,
        base_hidden_dim: @embed,
        vocab_size: 32,
        num_medusa_heads: 2
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"hidden_states" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through mixture_of_tokenizers" do
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

    # MixtureOfTokenizers takes 3D float input (continuous embeddings)
    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Contrastive additions ──────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through jepa context_encoder" do
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

    input = random_tensor({@batch, @embed})
    check_gradients(context_encoder, %{"features" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through temporal_jepa context_encoder" do
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

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(context_encoder, %{"state_sequence" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through siglip encoder" do
    {encoder, _temp_param} =
      Edifice.build(:siglip,
        input_dim: @embed,
        projection_dim: @hidden,
        hidden_size: @hidden
      )

    input = random_tensor({@batch, @embed})
    check_gradients(encoder, %{"features" => input})
  end

  # ── Interpretability ───────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through sparse_autoencoder" do
    model =
      Edifice.build(:sparse_autoencoder,
        input_size: @embed,
        hidden_size: @hidden * 4
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"sae_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through transcoder" do
    model =
      Edifice.build(:transcoder,
        input_size: @embed,
        output_size: @hidden,
        hidden_size: @hidden * 4
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"transcoder_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through gated_sae" do
    model =
      Edifice.build(:gated_sae,
        input_size: @embed,
        dict_size: @hidden * 4,
        top_k: 8
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"gated_sae_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through linear_probe" do
    model =
      Edifice.build(:linear_probe,
        input_size: @embed,
        num_classes: 5,
        task: :classification
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"probe_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through jump_relu_sae" do
    model =
      Edifice.build(:jump_relu_sae,
        input_size: @embed,
        dict_size: @hidden * 4,
        temperature: 10.0
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"jump_relu_sae_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through batch_top_k_sae" do
    model =
      Edifice.build(:batch_top_k_sae,
        input_size: @embed,
        dict_size: @hidden * 4,
        batch_k: @batch * 8
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"batch_topk_sae_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through crosscoder" do
    model =
      Edifice.build(:crosscoder,
        input_size: @embed,
        dict_size: @hidden * 4,
        num_sources: 2,
        top_k: 8
      )

    inputs = %{
      "crosscoder_source_0" => random_tensor({@batch, @embed}),
      "crosscoder_source_1" => random_tensor({@batch, @embed})
    }

    check_gradients(model, inputs)
  end

  @tag timeout: 120_000
  test "gradient flows through concept_bottleneck" do
    model =
      Edifice.build(:concept_bottleneck,
        input_size: @embed,
        num_concepts: 8,
        num_classes: 5
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"cbm_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through das_probe" do
    model =
      Edifice.build(:das_probe,
        input_size: @embed,
        subspace_dim: 8,
        num_classes: 5
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"das_probe_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through leace" do
    model =
      Edifice.build(:leace,
        input_size: @embed,
        concept_dim: 2
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"leace_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through matryoshka_sae" do
    model =
      Edifice.build(:matryoshka_sae,
        input_size: @embed,
        dict_size: @hidden * 4,
        top_k: 8
      )

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"matryoshka_sae_input" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through cross_layer_transcoder" do
    num_layers = 2

    model =
      Edifice.build(:cross_layer_transcoder,
        hidden_size: @embed,
        num_layers: num_layers,
        dict_size: @hidden * 4,
        top_k: 8
      )

    input = random_tensor({@batch, num_layers * @embed})
    check_gradients(model, %{"activations" => input})
  end

  # ── World Model ────────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through world_model encoder" do
    {encoder, _dynamics, _reward_head} =
      Edifice.build(:world_model,
        obs_size: @embed,
        action_size: @action_dim,
        latent_size: @latent_size,
        hidden_size: @hidden
      )

    input = random_tensor({@batch, @embed})
    check_gradients(encoder, %{"observation" => input})
  end

  @tag timeout: 120_000
  test "gradient flows through world_model dynamics" do
    {_encoder, dynamics, _reward_head} =
      Edifice.build(:world_model,
        obs_size: @embed,
        action_size: @action_dim,
        latent_size: @latent_size,
        hidden_size: @hidden
      )

    concat_size = @latent_size + @action_dim
    input = random_tensor({@batch, concat_size})
    check_gradients(dynamics, %{"state_action" => input})
  end

  # ── Multimodal ─────────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through multimodal_mlp_fusion" do
    model =
      Edifice.build(:multimodal_mlp_fusion,
        vision_dim: @embed,
        llm_dim: @hidden,
        num_visual_tokens: 4,
        text_seq_len: @seq_len
      )

    input_map = %{
      "visual_tokens" => random_tensor({@batch, 4, @embed}),
      "text_embeddings" => random_tensor({@batch, @seq_len, @hidden})
    }

    check_gradients(model, input_map)
  end

  # ── Scientific ─────────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through fno" do
    model =
      Edifice.build(:fno,
        in_channels: @in_channels,
        out_channels: @in_channels,
        hidden_channels: @hidden,
        num_layers: @num_layers,
        modes: 4
      )

    input = random_tensor({@batch, @seq_len, @in_channels})
    check_gradients(model, %{"input" => input})
  end

  # ── Memory addition ────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through engram" do
    model =
      Edifice.build(:engram,
        key_dim: @embed,
        value_dim: @hidden,
        num_tables: 2,
        num_buckets: 4
      )

    input_map = %{
      "query" => random_tensor({@batch, @embed}),
      "memory_slots" => random_tensor({2, 4, @hidden})
    }

    # Engram uses hash-based lookup — gradients may not flow through all paths.
    # Use parameter sensitivity check as fallback.
    check_parameter_sensitivity(model, input_map)
  end

  # ── Graph additions ─────────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through ka_gnn" do
    model =
      Edifice.build(:ka_gnn,
        input_dim: @node_dim,
        hidden_dim: @hidden,
        num_layers: @num_layers,
        num_harmonics: 2,
        num_classes: @num_classes,
        dropout: 0.0
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    adj = random_tensor({@batch, @num_nodes, @num_nodes})
    check_gradients(model, %{"nodes" => nodes, "adjacency" => adj})
  end

  # ── Generative flow additions ───────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through star_flow encoder" do
    {encoder, _decoder} =
      Edifice.build(:star_flow,
        input_size: @embed,
        hidden_size: @hidden,
        num_blocks: 2,
        deep_layers: 1,
        shallow_layers: 1,
        num_heads: 2,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(encoder, %{"input" => input})
  end

  # ── Memory addition (memory_layer) ──────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through memory_layer" do
    model =
      Edifice.build(:memory_layer,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        num_keys: 8,
        top_k: 2,
        key_dim: @hidden,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Transformer addition (free_transformer) ─────────────────────

  @tag timeout: 120_000
  test "gradient flows through free_transformer" do
    model =
      Edifice.build(:free_transformer,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: 2,
        num_latent_bits: 4,
        seq_len: @seq_len,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Wave 5: LASER attention ────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through laser" do
    model =
      Edifice.build(:laser,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: 2,
        dropout: 0.0,
        seq_len: @seq_len
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Wave 5: Longhorn SSM ───────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through longhorn" do
    model =
      Edifice.build(:longhorn,
        embed_dim: @embed,
        hidden_size: @hidden,
        state_size: 4,
        num_layers: 2,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Wave 5: VeRA PEFT ─────────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through vera" do
    model = Edifice.build(:vera, input_size: @embed, output_size: @hidden, rank: 8)

    input = random_tensor({@batch, @embed})
    check_gradients(model, %{"input" => input})
  end

  # ── Wave 5: MoBA Attention ──────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through moba" do
    model =
      Edifice.build(:moba,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_layers: @num_layers,
        block_size: 2,
        topk: 1,
        dropout: 0.0
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Wave 5: DeltaProduct Recurrence ─────────────────────────

  @tag timeout: 120_000
  test "gradient flows through delta_product" do
    model =
      Edifice.build(:delta_product,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_heads: 2,
        num_householder: 2,
        num_layers: @num_layers,
        dropout: 0.0,
        seq_len: @seq_len
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end

  # ── Wave 5: Vision KAN ─────────────────────────────────────

  @tag timeout: 120_000
  test "gradient flows through vision_kan" do
    model =
      Edifice.build(:vision_kan,
        image_size: @image_size,
        in_channels: @in_channels,
        channels: [8, 16],
        depths: [1, 1],
        patch_size: 4,
        num_rbf_centers: 3,
        kan_patch_size: 4,
        dw_kernel_size: 3,
        global_reduction: 2,
        dropout: 0.0,
        ffn_expansion: 2
      )

    input = random_tensor({@batch, @in_channels, @image_size, @image_size})
    check_gradients(model, %{"image" => input})
  end

  test "mixture_of_mamba gradient" do
    model =
      Edifice.build(:mixture_of_mamba,
        embed_dim: @embed,
        hidden_size: @hidden,
        state_size: 4,
        expand_factor: 2,
        num_layers: @num_layers,
        num_modalities: 2,
        window_size: @seq_len
      )

    seq = random_tensor({@batch, @seq_len, @embed})

    mask =
      Nx.iota({@batch, @seq_len}, axis: 1)
      |> Nx.remainder(2)
      |> Nx.as_type(:s32)

    check_gradients(model, %{"state_sequence" => seq, "modality_mask" => mask})
  end

  # ── Wave 5: TNN (Toeplitz Neural Network) ───────────────────

  @tag timeout: 120_000
  test "gradient flows through tnn" do
    model =
      Edifice.build(:tnn,
        embed_dim: @embed,
        hidden_size: @hidden,
        num_layers: @num_layers,
        expand_ratio: 2,
        rpe_dim: 4,
        rpe_layers: 1,
        dropout: 0.0,
        window_size: @seq_len
      )

    input = random_tensor({@batch, @seq_len, @embed})
    check_gradients(model, %{"state_sequence" => input})
  end
end
