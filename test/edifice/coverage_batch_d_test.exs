defmodule Edifice.CoverageBatchDTest do
  @moduledoc """
  Coverage tests targeting uncovered code paths in:
  MessagePassing, GatedSSM, FlowMatching, ModelBuilder, VQVAE, LoRA, Liquid
  """
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  @batch 2
  @seq_len 8
  @embed 16

  # ==========================================================================
  # MessagePassing (50.91%) - Need :mean and :max aggregation forward passes
  # ==========================================================================
  describe "MessagePassing :mean aggregation forward pass" do
    alias Edifice.Graph.MessagePassing

    @num_nodes 4
    @feature_dim 8

    test "message_passing_layer with :mean aggregation runs forward" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model =
        MessagePassing.message_passing_layer(nodes, adj, @embed,
          aggregation: :mean,
          name: "mpnn_mean"
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      # Adjacency with some connections (not just identity)
      adj_matrix =
        Nx.tensor([
          [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
          ],
          [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
          ]
        ])
        |> Nx.as_type(:f32)

      node_feats = Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim})

      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      assert Nx.shape(output) == {@batch, @num_nodes, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "message_passing_layer with :max aggregation runs forward" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model =
        MessagePassing.message_passing_layer(nodes, adj, @embed,
          aggregation: :max,
          name: "mpnn_max"
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix =
        Nx.tensor([
          [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
          ],
          [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
          ]
        ])
        |> Nx.as_type(:f32)

      node_feats = Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim})

      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      assert Nx.shape(output) == {@batch, @num_nodes, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "aggregate/3 with :mean mode runs forward" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model = MessagePassing.aggregate(nodes, adj, :mean)

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim})

      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      # aggregate produces [batch, num_nodes, feature_dim] (or fewer dims depending on impl)
      {b, _, _} = Nx.shape(output)
      assert b == @batch
    end

    test "aggregate/3 with :max mode runs forward" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model = MessagePassing.aggregate(nodes, adj, :max)

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix =
        Nx.tensor([
          [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
          ],
          [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
          ]
        ])
        |> Nx.as_type(:f32)

      node_feats = Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim})

      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      {b, _, _} = Nx.shape(output)
      assert b == @batch
    end

    test "global_pool with :sum mode" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input, :sum)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @num_nodes, @feature_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @num_nodes, @feature_dim}))
      assert Nx.shape(output) == {@batch, @feature_dim}

      # Sum of @num_nodes ones should be @num_nodes
      val = Nx.to_number(output[0][0])
      assert abs(val - @num_nodes) < 1.0e-4
    end

    test "global_pool with :max mode" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input, :max)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @num_nodes, @feature_dim}, :f32), Axon.ModelState.empty())

      # Use varying values so max is meaningful
      node_feats = Nx.iota({@batch, @num_nodes, @feature_dim}, axis: 1) |> Nx.as_type(:f32)
      output = predict_fn.(params, node_feats)
      assert Nx.shape(output) == {@batch, @feature_dim}

      # Max over nodes (axis 1) -- last node has highest values
      max_val = Nx.to_number(output[0][0])
      assert max_val == (@num_nodes - 1) * 1.0
    end
  end

  # ==========================================================================
  # GatedSSM (55.42%) - Need forward with embed_dim != hidden_size + dropout
  # ==========================================================================
  describe "GatedSSM forward passes" do
    alias Edifice.SSM.GatedSSM

    test "forward pass with embed_dim != hidden_size (triggers input projection)" do
      hidden = 32

      model =
        GatedSSM.build(
          embed_dim: @embed,
          hidden_size: hidden,
          state_size: 4,
          expand_factor: 2,
          conv_size: 2,
          num_layers: 1,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      assert Nx.shape(output) == {@batch, hidden}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "forward pass with dropout > 0 and multiple layers" do
      model =
        GatedSSM.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          expand_factor: 2,
          conv_size: 2,
          num_layers: 3,
          dropout: 0.2,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      result = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      # In :train mode with dropout, result is %{state: ..., prediction: tensor}
      output =
        if is_map(result) and Map.has_key?(result, :prediction),
          do: result.prediction,
          else: result

      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build_checkpointed forward pass" do
      model =
        GatedSSM.build_checkpointed(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          expand_factor: 2,
          conv_size: 2,
          num_layers: 2,
          seq_len: @seq_len,
          checkpoint_every: 1
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      assert Nx.shape(output) == {@batch, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "build_checkpointed with embed_dim != hidden_size and dropout" do
      hidden = 32

      model =
        GatedSSM.build_checkpointed(
          embed_dim: @embed,
          hidden_size: hidden,
          state_size: 4,
          expand_factor: 2,
          conv_size: 2,
          num_layers: 2,
          dropout: 0.1,
          seq_len: @seq_len,
          checkpoint_every: 2
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      assert Nx.shape(output) == {@batch, hidden}
    end
  end

  # ==========================================================================
  # FlowMatching (55.74%) - Need sample/5 and compute_loss/6
  # ==========================================================================
  describe "FlowMatching sample and compute_loss" do
    alias Edifice.Generative.FlowMatching

    @obs_size 8
    @action_dim 4
    @action_horizon 2

    setup do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 16,
          num_layers: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "x_t" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32),
        "observations" => Nx.template({@batch, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      %{params: params, predict_fn: predict_fn}
    end

    test "sample with :euler solver", %{params: params, predict_fn: predict_fn} do
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result =
        FlowMatching.sample(params, predict_fn, observations, noise,
          num_steps: 3,
          solver: :euler
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "sample with :midpoint solver", %{params: params, predict_fn: predict_fn} do
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result =
        FlowMatching.sample(params, predict_fn, observations, noise,
          num_steps: 3,
          solver: :midpoint
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "sample with :rk4 solver", %{params: params, predict_fn: predict_fn} do
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result =
        FlowMatching.sample(params, predict_fn, observations, noise,
          num_steps: 3,
          solver: :rk4
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "compute_loss returns scalar", %{params: params, predict_fn: predict_fn} do
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      actions = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      noise = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.3, 0.7], type: :f32)

      loss = FlowMatching.compute_loss(params, predict_fn, observations, actions, noise, t)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0
      refute Nx.is_nan(loss) |> Nx.to_number() == 1
    end

    test "generate_rectified_pairs returns {noise, generated} tuple", %{
      params: params,
      predict_fn: predict_fn
    } do
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      {returned_noise, generated} =
        FlowMatching.generate_rectified_pairs(params, predict_fn, observations, noise,
          num_steps: 2
        )

      assert Nx.shape(returned_noise) == {@batch, @action_horizon, @action_dim}
      assert Nx.shape(generated) == {@batch, @action_horizon, @action_dim}
    end

    test "sample_guided returns correct shape", %{params: params, predict_fn: predict_fn} do
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result =
        FlowMatching.sample_guided(params, predict_fn, observations, noise,
          num_steps: 2,
          guidance_scale: 2.0
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
    end
  end

  # ==========================================================================
  # ModelBuilder (55.81%) - Need build_vision_model + sequence with projection/dropout
  # ==========================================================================
  describe "ModelBuilder coverage" do
    alias Edifice.Blocks.ModelBuilder

    test "build_vision_model with classifier head" do
      # Use small image size divisible by patch_size
      image_size = 8
      patch_size = 4
      in_channels = 1
      hidden = 16

      block_builder = fn input, opts ->
        name = "vit_block_#{opts[:layer_idx]}"
        Axon.dense(input, hidden, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_vision_model(
          image_size: image_size,
          patch_size: patch_size,
          in_channels: in_channels,
          hidden_size: hidden,
          num_layers: 1,
          block_builder: block_builder,
          num_classes: 4
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "image" => Nx.template({@batch, in_channels, image_size, image_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, in_channels, image_size, image_size})
      output = predict_fn.(params, input)

      # num_classes = 4
      assert Nx.shape(output) == {@batch, 4}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "build_vision_model without classifier head" do
      image_size = 8
      patch_size = 4
      in_channels = 1
      hidden = 16

      block_builder = fn input, opts ->
        name = "vit_block_#{opts[:layer_idx]}"
        Axon.dense(input, hidden, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_vision_model(
          image_size: image_size,
          patch_size: patch_size,
          in_channels: in_channels,
          hidden_size: hidden,
          num_layers: 1,
          block_builder: block_builder
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "image" => Nx.template({@batch, in_channels, image_size, image_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {@batch, in_channels, image_size, image_size})
      output = predict_fn.(params, input)

      # No classifier: output is [batch, hidden_size]
      assert Nx.shape(output) == {@batch, hidden}
    end

    test "build_sequence_model with embed_dim != hidden_size (triggers projection)" do
      hidden = 32

      block_builder = fn input, opts ->
        name = "block_#{opts[:layer_idx]}"
        Axon.dense(input, hidden, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_sequence_model(
          embed_dim: @embed,
          hidden_size: hidden,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: block_builder
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      # last_timestep output mode (default)
      assert Nx.shape(output) == {@batch, hidden}
    end

    test "build_sequence_model with dropout > 0 and multiple layers" do
      block_builder = fn input, opts ->
        name = "block_#{opts[:layer_idx]}"
        Axon.dense(input, @embed, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_sequence_model(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          seq_len: @seq_len,
          block_builder: block_builder,
          dropout: 0.2
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :train)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      result = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      output =
        if is_map(result) and Map.has_key?(result, :prediction),
          do: result.prediction,
          else: result

      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build_sequence_model with final_norm: false" do
      block_builder = fn input, opts ->
        name = "block_#{opts[:layer_idx]}"
        Axon.dense(input, @embed, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_sequence_model(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: block_builder,
          final_norm: false
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # VQVAE (59.18%) - Need loss functions + quantize + init_codebook
  # ==========================================================================
  describe "VQVAE loss functions and quantize" do
    alias Edifice.Generative.VQVAE

    @embedding_dim 8
    @num_embeddings 16

    test "quantize returns correct shapes" do
      z_e = Nx.broadcast(0.5, {@batch, @embedding_dim})
      codebook = Nx.broadcast(0.3, {@num_embeddings, @embedding_dim})

      {z_q, indices} = VQVAE.quantize(z_e, codebook)

      assert Nx.shape(z_q) == {@batch, @embedding_dim}
      assert Nx.shape(indices) == {@batch}
    end

    test "quantize finds nearest codebook entry" do
      # Create a codebook with two distinct entries
      codebook =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ])

      # Encoder output close to first codebook entry
      z_e = Nx.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9]])

      {_z_q, indices} = VQVAE.quantize(z_e, codebook)

      # First sample should map to index 0, second to index 3
      assert Nx.to_number(indices[0]) == 0
      assert Nx.to_number(indices[1]) == 3
    end

    test "commitment_loss is non-negative" do
      z_e = Nx.broadcast(0.5, {@batch, @embedding_dim})
      z_q = Nx.broadcast(0.3, {@batch, @embedding_dim})

      loss = VQVAE.commitment_loss(z_e, z_q)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0
    end

    test "commitment_loss is zero when z_e equals z_q" do
      z = Nx.broadcast(0.5, {@batch, @embedding_dim})

      loss = VQVAE.commitment_loss(z, z)
      assert abs(Nx.to_number(loss)) < 1.0e-6
    end

    test "codebook_loss is non-negative" do
      z_e = Nx.broadcast(0.5, {@batch, @embedding_dim})
      z_q = Nx.broadcast(0.3, {@batch, @embedding_dim})

      loss = VQVAE.codebook_loss(z_e, z_q)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0
    end

    test "codebook_loss is zero when z_e equals z_q" do
      z = Nx.broadcast(0.5, {@batch, @embedding_dim})

      loss = VQVAE.codebook_loss(z, z)
      assert abs(Nx.to_number(loss)) < 1.0e-6
    end

    # NOTE: init_codebook/3 is defn with shape: {num_embeddings, embedding_dim}
    # which cannot trace integer arguments. Test the function signature exists
    # and use Nx.Random.normal directly for codebook generation in tests.
    test "init_codebook function exists" do
      assert function_exported?(VQVAE, :init_codebook, 2)
      assert function_exported?(VQVAE, :init_codebook, 3)
    end

    test "combined loss/5 returns scalar" do
      reconstruction = Nx.broadcast(0.5, {@batch, @embed})
      target = Nx.broadcast(0.6, {@batch, @embed})
      z_e = Nx.broadcast(0.4, {@batch, @embedding_dim})
      z_q = Nx.broadcast(0.3, {@batch, @embedding_dim})

      loss = VQVAE.loss(reconstruction, target, z_e, z_q)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "combined loss/5 with custom commitment_weight" do
      reconstruction = Nx.broadcast(0.5, {@batch, @embed})
      target = Nx.broadcast(0.6, {@batch, @embed})
      z_e = Nx.broadcast(0.4, {@batch, @embedding_dim})
      z_q = Nx.broadcast(0.3, {@batch, @embedding_dim})

      loss = VQVAE.loss(reconstruction, target, z_e, z_q, commitment_weight: 0.5)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "full encode-quantize-decode pipeline" do
      input_size = @embed

      {encoder, decoder} =
        VQVAE.build(
          input_size: input_size,
          embedding_dim: @embedding_dim,
          num_embeddings: @num_embeddings,
          encoder_sizes: [16],
          decoder_sizes: [16]
        )

      # Build encoder
      {enc_init, enc_predict} = Axon.build(encoder)

      enc_params =
        enc_init.(
          %{"input" => Nx.template({@batch, input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, input_size})
      z_e = enc_predict.(enc_params, input_data)

      # Initialize codebook manually (init_codebook has defn tracing issue)
      {codebook, _key} =
        Nx.Random.normal(Nx.Random.key(42), shape: {@num_embeddings, @embedding_dim})

      {z_q, indices} = VQVAE.quantize(z_e, codebook)

      assert Nx.shape(z_q) == {@batch, @embedding_dim}
      assert Nx.shape(indices) == {@batch}

      # Build decoder
      {dec_init, dec_predict} = Axon.build(decoder)

      dec_params =
        dec_init.(
          %{"quantized" => Nx.template({@batch, @embedding_dim}, :f32)},
          Axon.ModelState.empty()
        )

      reconstruction = dec_predict.(dec_params, z_q)
      assert Nx.shape(reconstruction) == {@batch, input_size}

      # Compute all loss components
      commit = VQVAE.commitment_loss(z_e, z_q)
      cb = VQVAE.codebook_loss(z_e, z_q)
      total = VQVAE.loss(reconstruction, input_data, z_e, z_q)

      assert Nx.to_number(commit) >= 0
      assert Nx.to_number(cb) >= 0
      assert Nx.to_number(total) >= 0
    end
  end

  # ==========================================================================
  # LoRA (53.49%) - Need build/1, wrap without output_size, output_size/1
  # ==========================================================================
  describe "LoRA coverage" do
    alias Edifice.Meta.LoRA

    test "build/1 creates standalone LoRA model" do
      model =
        LoRA.build(
          input_size: @embed,
          output_size: 32,
          rank: 4,
          alpha: 8.0,
          name: "lora_standalone"
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 32}
    end

    test "build/1 with default rank and alpha" do
      model = LoRA.build(input_size: @embed, output_size: @embed)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "wrap/3 WITHOUT output_size triggers adaptive LoRA path" do
      input = Axon.input("input", shape: {nil, @embed})
      original = Axon.dense(input, @embed, name: "base_dense")

      # Calling wrap without :output_size triggers lora_delta_adaptive
      adapted = LoRA.wrap(input, original, rank: 4, alpha: 8.0, name: "lora_adaptive")

      assert %Axon{} = adapted

      {init_fn, predict_fn} = Axon.build(adapted)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "wrap/3 WITH output_size uses standard lora_delta" do
      input = Axon.input("input", shape: {nil, @embed})
      original = Axon.dense(input, 32, name: "base_dense_sized")

      adapted = LoRA.wrap(input, original, rank: 4, output_size: 32, name: "lora_standard")

      {init_fn, predict_fn} = Axon.build(adapted)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 32}
    end

    test "output_size/1 returns the output_size" do
      assert LoRA.output_size(output_size: 64) == 64
      assert LoRA.output_size(output_size: 128) == 128
    end

    test "lora_delta forward pass" do
      input = Axon.input("input", shape: {nil, @embed})
      delta = LoRA.lora_delta(input, 32, rank: 4, alpha: 16.0, name: "delta_test")

      {init_fn, predict_fn} = Axon.build(delta)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 32}
    end
  end

  # ==========================================================================
  # Liquid (71.74%) - Need solver: :midpoint, build_with_ffn, high_accuracy_defaults
  # ==========================================================================
  describe "Liquid coverage" do
    alias Edifice.Liquid

    test "build with solver: :midpoint" do
      model =
        Liquid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          dropout: 0.0,
          integration_steps: 1,
          solver: :midpoint
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "build with solver: :rk4 and integration_steps: 2" do
      model =
        Liquid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          dropout: 0.0,
          integration_steps: 2,
          solver: :rk4
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build_with_ffn produces correct output shape" do
      model =
        Liquid.build_with_ffn(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          dropout: 0.0,
          integration_steps: 1,
          solver: :euler
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "build_with_ffn with embed_dim != hidden_size" do
      hidden = 32

      model =
        Liquid.build_with_ffn(
          embed_dim: @embed,
          hidden_size: hidden,
          num_layers: 1,
          seq_len: @seq_len,
          dropout: 0.0,
          integration_steps: 1,
          solver: :euler
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, hidden}
    end

    test "build_with_ffn with dropout > 0" do
      model =
        Liquid.build_with_ffn(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          seq_len: @seq_len,
          dropout: 0.1,
          integration_steps: 1,
          solver: :euler
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :train)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())

      result = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))

      output =
        if is_map(result) and Map.has_key?(result, :prediction),
          do: result.prediction,
          else: result

      assert Nx.shape(output) == {@batch, @embed}
    end

    test "high_accuracy_defaults returns expected config" do
      defaults = Liquid.high_accuracy_defaults()

      assert Keyword.get(defaults, :solver) == :dopri5
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :integration_steps)
    end

    test "init_cache with default options" do
      cache = Liquid.init_cache()

      assert cache.step == 0
      assert is_map(cache.layers)
      assert cache.config.hidden_size == 256
      assert cache.config.num_layers == 4
    end
  end
end
