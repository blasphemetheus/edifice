defmodule Edifice.CoverageBatchBTest do
  @moduledoc "Coverage tests for generative and misc modules: DDIM, Diffusion, FlowMatching, ModelBuilder, MCDropout"
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  @batch 2
  @seq_len 8
  @embed 16

  # ==========================================================================
  # DDIM (52%) - Need build forward pass and schedule
  # ==========================================================================
  describe "DDIM" do
    alias Edifice.Generative.DDIM

    test "build produces Axon model" do
      model =
        DDIM.build(
          obs_size: @embed,
          action_dim: 4,
          action_horizon: 2,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model =
        DDIM.build(
          obs_size: @embed,
          action_dim: 4,
          action_horizon: 2,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "noisy_actions" => Nx.template({@batch, 2, 4}, :f32),
            "timestep" => Nx.template({@batch}, :f32),
            "observations" => Nx.template({@batch, @embed}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noisy_actions" => Nx.broadcast(0.5, {@batch, 2, 4}),
          "timestep" => Nx.tensor([5, 3], type: :f32),
          "observations" => Nx.broadcast(0.5, {@batch, @embed})
        })

      assert Nx.shape(output) == {@batch, 2, 4}
    end

    test "make_schedule returns correct keys" do
      schedule = DDIM.make_schedule(num_steps: 10)
      assert is_map(schedule)
      assert Map.has_key?(schedule, :alphas_cumprod)
      assert Map.has_key?(schedule, :betas)
      assert Map.has_key?(schedule, :sqrt_alphas_cumprod)
    end

    test "ddim_step computes reverse step (6-arity with eta)" do
      schedule = DDIM.make_schedule(num_steps: 10)
      noisy = Nx.broadcast(0.5, {@batch, 2, 4})
      pred_noise = Nx.broadcast(0.1, {@batch, 2, 4})
      # ddim_step uses scalar timestep indexing into alphas_cumprod
      timestep = 5
      prev_timestep = 3

      result = DDIM.ddim_step(noisy, pred_noise, timestep, prev_timestep, schedule, 0.0)
      assert Nx.shape(result) == {@batch, 2, 4}
    end

    test "output_size returns correct value" do
      assert DDIM.output_size(action_dim: 4, action_horizon: 2) == 8
    end

    test "param_count returns positive" do
      count = DDIM.param_count(obs_size: @embed, action_dim: 4, action_horizon: 2)
      assert count > 0
    end
  end

  # ==========================================================================
  # Diffusion (55%) - Need obs_encoder, q_sample, p_sample, make_schedule
  # ==========================================================================
  describe "Diffusion" do
    alias Edifice.Generative.Diffusion

    test "make_schedule returns map with all keys" do
      schedule = Diffusion.make_schedule(num_steps: 10)
      assert Map.has_key?(schedule, :betas)
      assert Map.has_key?(schedule, :alphas)
      assert Map.has_key?(schedule, :alphas_cumprod)
      assert Map.has_key?(schedule, :sqrt_alphas_cumprod)
      assert Map.has_key?(schedule, :sqrt_one_minus_alphas_cumprod)
      assert Map.has_key?(schedule, :posterior_variance)
    end

    test "build_obs_encoder produces Axon model" do
      model =
        Diffusion.build_obs_encoder(embed_dim: @embed, hidden_size: 16, window_size: @seq_len)

      assert %Axon{} = model
    end

    test "build forward pass" do
      model =
        Diffusion.build(
          obs_size: @embed,
          action_dim: 4,
          action_horizon: 2,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "noisy_actions" => Nx.template({@batch, 2, 4}, :f32),
            "timestep" => Nx.template({@batch}, :f32),
            "observations" => Nx.template({@batch, @embed}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noisy_actions" => Nx.broadcast(0.5, {@batch, 2, 4}),
          "timestep" => Nx.tensor([5, 3], type: :f32),
          "observations" => Nx.broadcast(0.5, {@batch, @embed})
        })

      assert Nx.shape(output) == {@batch, 2, 4}
    end

    test "q_sample adds noise to actions" do
      schedule = Diffusion.make_schedule(num_steps: 10)
      actions = Nx.broadcast(0.5, {@batch, 2, 4})
      timestep = Nx.tensor([3, 5])
      noise = Nx.broadcast(0.1, {@batch, 2, 4})

      noisy = Diffusion.q_sample(actions, timestep, noise, schedule)
      assert Nx.shape(noisy) == {@batch, 2, 4}
    end

    test "p_sample denoises" do
      schedule = Diffusion.make_schedule(num_steps: 10)
      noisy = Nx.broadcast(0.5, {@batch, 2, 4})
      pred_noise = Nx.broadcast(0.1, {@batch, 2, 4})
      timestep = Nx.tensor([5, 5])
      random_noise = Nx.broadcast(0.01, {@batch, 2, 4})

      result = Diffusion.p_sample(noisy, pred_noise, timestep, random_noise, schedule)
      assert Nx.shape(result) == {@batch, 2, 4}
    end

    test "compute_loss returns scalar" do
      true_noise = Nx.broadcast(0.1, {@batch, 2, 4})
      pred_noise = Nx.broadcast(0.2, {@batch, 2, 4})
      loss = Diffusion.compute_loss(true_noise, pred_noise)
      assert Nx.shape(loss) == {}
    end

    test "fast_inference_defaults returns keyword list" do
      defaults = Diffusion.fast_inference_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :num_steps)
    end
  end

  # ==========================================================================
  # FlowMatching (56%) - Need forward pass
  # ==========================================================================
  describe "FlowMatching forward" do
    alias Edifice.Generative.FlowMatching

    test "build and forward pass" do
      model =
        FlowMatching.build(
          obs_size: @embed,
          action_dim: 4,
          action_horizon: 2,
          hidden_size: 16,
          num_layers: 1
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch, 2, 4}, :f32),
            "timestep" => Nx.template({@batch}, :f32),
            "observations" => Nx.template({@batch, @embed}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "x_t" => Nx.broadcast(0.5, {@batch, 2, 4}),
          "timestep" => Nx.tensor([0.3, 0.7], type: :f32),
          "observations" => Nx.broadcast(0.5, {@batch, @embed})
        })

      assert Nx.shape(output) == {@batch, 2, 4}
    end

    test "velocity_loss computes MSE" do
      target = Nx.broadcast(1.0, {@batch, @embed})
      pred = Nx.broadcast(0.5, {@batch, @embed})

      loss = FlowMatching.velocity_loss(target, pred)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  # ==========================================================================
  # ModelBuilder (56%) - Need build_sequence_model with block_builder
  # ==========================================================================
  describe "ModelBuilder" do
    alias Edifice.Blocks.ModelBuilder

    test "build_sequence_model with FFN block builder" do
      ffn_builder = fn input, opts ->
        hidden_size = Keyword.get(opts, :hidden_size, 16)
        name = Keyword.get(opts, :name, "block")

        input
        |> Axon.dense(hidden_size, name: "#{name}_dense")
        |> Axon.activation(:relu, name: "#{name}_act")
      end

      model =
        ModelBuilder.build_sequence_model(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          seq_len: @seq_len,
          block_builder: ffn_builder
        )

      assert %Axon{} = model
    end

    test "build_sequence_model with output_mode: :all" do
      ffn_builder = fn input, opts ->
        name = Keyword.get(opts, :name, "block")
        Axon.dense(input, @embed, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_sequence_model(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: ffn_builder,
          output_mode: :all
        )

      assert %Axon{} = model
    end

    test "build_sequence_model with output_mode: :mean_pool" do
      ffn_builder = fn input, opts ->
        name = Keyword.get(opts, :name, "block")
        Axon.dense(input, @embed, name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_sequence_model(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: ffn_builder,
          output_mode: :mean_pool
        )

      assert %Axon{} = model
    end
  end

  # ==========================================================================
  # MCDropout (57%) - Need build and forward
  # ==========================================================================
  describe "MCDropout" do
    alias Edifice.Probabilistic.MCDropout

    test "build creates model" do
      model =
        MCDropout.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          dropout_rate: 0.1
        )

      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model =
        MCDropout.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          dropout_rate: 0.1
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 4}
    end

    test "predict_with_uncertainty returns mean and variance" do
      model =
        MCDropout.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          dropout_rate: 0.1
        )

      # Build in :train mode so dropout keys are generated in state
      template = %{"input" => Nx.template({@batch, @embed}, :f32)}
      {init_fn, _predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(template, Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {@batch, @embed})

      # predict_with_uncertainty takes the Axon model struct, not compiled predict_fn
      {mean, variance} = MCDropout.predict_with_uncertainty(model, params, input, num_samples: 3)
      assert Nx.shape(mean) == {@batch, 4}
      assert Nx.shape(variance) == {@batch, 4}
    end

    test "predictive_entropy returns scalar per sample" do
      probs = Nx.tensor([[0.25, 0.25, 0.25, 0.25], [0.9, 0.05, 0.025, 0.025]])
      entropy = MCDropout.predictive_entropy(probs)
      assert Nx.shape(entropy) == {2}
    end
  end
end
