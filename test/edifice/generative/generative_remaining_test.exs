defmodule Edifice.Generative.GenerativeRemainingTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.Diffusion
  alias Edifice.Generative.FlowMatching
  alias Edifice.Generative.NormalizingFlow
  alias Edifice.Generative.VQVAE

  @batch_size 2

  # ============================================================================
  # Diffusion Tests
  # ============================================================================

  describe "Diffusion.build/1" do
    @obs_size 16
    @action_dim 4
    @action_horizon 4

    @diffusion_opts [
      obs_size: @obs_size,
      action_dim: @action_dim,
      action_horizon: @action_horizon,
      hidden_size: 32,
      num_layers: 2,
      num_steps: 10
    ]

    test "builds an Axon model" do
      model = Diffusion.build(@diffusion_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Diffusion.build(@diffusion_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_actions" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch_size}, :f32),
        "observations" => Nx.template({@batch_size, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim}),
        "timestep" => Nx.broadcast(5.0, {@batch_size}),
        "observations" => Nx.broadcast(0.1, {@batch_size, @obs_size})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @action_horizon, @action_dim}
    end

    test "output contains finite values" do
      model = Diffusion.build(@diffusion_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_actions" => Nx.template({@batch_size, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch_size}, :f32),
        "observations" => Nx.template({@batch_size, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch_size, @action_horizon, @action_dim}),
        "timestep" => Nx.broadcast(5.0, {@batch_size}),
        "observations" => Nx.broadcast(0.1, {@batch_size, @obs_size})
      }

      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "Diffusion.make_schedule/1" do
    test "returns schedule map with expected keys" do
      schedule = Diffusion.make_schedule(num_steps: 10)
      assert is_map(schedule)
      assert Map.has_key?(schedule, :betas)
      assert Map.has_key?(schedule, :alphas)
    end
  end

  describe "Diffusion.output_size/1" do
    test "returns action_horizon * action_dim" do
      assert Diffusion.output_size(action_horizon: 8, action_dim: 4) == 32
    end
  end

  # ============================================================================
  # FlowMatching Tests
  # ============================================================================

  describe "FlowMatching.build/1" do
    @fm_obs_size 16
    @fm_action_dim 4
    @fm_action_horizon 4

    @fm_opts [
      obs_size: @fm_obs_size,
      action_dim: @fm_action_dim,
      action_horizon: @fm_action_horizon,
      hidden_size: 32,
      num_layers: 2,
      num_steps: 5
    ]

    test "builds an Axon model" do
      model = FlowMatching.build(@fm_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = FlowMatching.build(@fm_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "x_t" => Nx.template({@batch_size, @fm_action_horizon, @fm_action_dim}, :f32),
        "timestep" => Nx.template({@batch_size}, :f32),
        "observations" => Nx.template({@batch_size, @fm_obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "x_t" => Nx.broadcast(0.5, {@batch_size, @fm_action_horizon, @fm_action_dim}),
        "timestep" => Nx.broadcast(0.5, {@batch_size}),
        "observations" => Nx.broadcast(0.1, {@batch_size, @fm_obs_size})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @fm_action_horizon, @fm_action_dim}
    end

    test "output contains finite values" do
      model = FlowMatching.build(@fm_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "x_t" => Nx.template({@batch_size, @fm_action_horizon, @fm_action_dim}, :f32),
        "timestep" => Nx.template({@batch_size}, :f32),
        "observations" => Nx.template({@batch_size, @fm_obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "x_t" => Nx.broadcast(0.5, {@batch_size, @fm_action_horizon, @fm_action_dim}),
        "timestep" => Nx.broadcast(0.5, {@batch_size}),
        "observations" => Nx.broadcast(0.1, {@batch_size, @fm_obs_size})
      }

      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "FlowMatching.interpolate/3" do
    test "produces tensor of correct shape" do
      x0 = Nx.broadcast(0.0, {2, 4, 3})
      x1 = Nx.broadcast(1.0, {2, 4, 3})
      t = Nx.broadcast(0.5, {2})
      result = FlowMatching.interpolate(x0, x1, t)
      assert Nx.shape(result) == {2, 4, 3}
    end
  end

  describe "FlowMatching.output_size/1" do
    test "returns action_horizon * action_dim" do
      assert FlowMatching.output_size(action_horizon: 8, action_dim: 4) == 32
    end
  end

  # ============================================================================
  # NormalizingFlow Tests
  # ============================================================================

  describe "NormalizingFlow.build/1" do
    @nf_input_size 8

    @nf_opts [
      input_size: @nf_input_size,
      num_flows: 2,
      hidden_sizes: [16]
    ]

    test "builds an Axon model" do
      model = NormalizingFlow.build(@nf_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = NormalizingFlow.build(@nf_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @nf_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @nf_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @nf_input_size}
    end

    test "output contains finite values" do
      model = NormalizingFlow.build(@nf_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @nf_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @nf_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # VQVAE Tests
  # ============================================================================

  describe "VQVAE.build/1" do
    @vq_input_size 16
    @vq_embedding_dim 8

    @vq_opts [
      input_size: @vq_input_size,
      embedding_dim: @vq_embedding_dim,
      num_embeddings: 32,
      encoder_sizes: [16],
      decoder_sizes: [16]
    ]

    test "returns {encoder, decoder} tuple" do
      {encoder, decoder} = VQVAE.build(@vq_opts)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end

    test "encoder produces correct output shape" do
      {encoder, _decoder} = VQVAE.build(@vq_opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @vq_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @vq_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @vq_embedding_dim}
    end

    test "decoder reconstructs to input_size" do
      {_encoder, decoder} = VQVAE.build(@vq_opts)
      {init_fn, predict_fn} = Axon.build(decoder)

      params =
        init_fn.(
          %{"quantized" => Nx.template({@batch_size, @vq_embedding_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"quantized" => Nx.broadcast(0.5, {@batch_size, @vq_embedding_dim})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @vq_input_size}
    end

    test "encoder output contains finite values" do
      {encoder, _decoder} = VQVAE.build(@vq_opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @vq_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @vq_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "VQVAE.quantize/2" do
    test "returns quantized and indices" do
      z = Nx.broadcast(0.5, {2, 8})
      codebook = Nx.broadcast(0.3, {32, 8})
      {quantized, indices} = VQVAE.quantize(z, codebook)
      assert Nx.shape(quantized) == {2, 8}
      assert Nx.shape(indices) == {2}
    end
  end
end
