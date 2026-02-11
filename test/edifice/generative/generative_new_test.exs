defmodule Edifice.Generative.GenerativeNewTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.DiT
  alias Edifice.Generative.DDIM
  alias Edifice.Generative.LatentDiffusion
  alias Edifice.Generative.ConsistencyModel
  alias Edifice.Generative.ScoreSDE

  @batch 2
  @input_dim 32
  @hidden_size 32

  # ============================================================================
  # DiT Tests
  # ============================================================================

  describe "DiT.build/1" do
    @dit_opts [
      input_dim: @input_dim,
      hidden_size: @hidden_size,
      depth: 2,
      num_heads: 2,
      num_steps: 100
    ]

    test "builds an Axon model" do
      model = DiT.build(@dit_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = DiT.build(@dit_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
    end

    test "output contains finite values" do
      model = DiT.build(@dit_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "DiT.output_size/1" do
    test "returns input_dim" do
      assert DiT.output_size(input_dim: @input_dim) == @input_dim
    end
  end

  # ============================================================================
  # DDIM Tests
  # ============================================================================

  describe "DDIM.build/1" do
    @action_dim 16
    @action_horizon 4
    @obs_size 32

    @ddim_opts [
      obs_size: @obs_size,
      action_dim: @action_dim,
      action_horizon: @action_horizon,
      hidden_size: @hidden_size,
      num_layers: 2,
      num_steps: 100
    ]

    test "builds an Axon model" do
      model = DDIM.build(@ddim_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = DDIM.build(@ddim_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim}),
        "timestep" => Nx.tensor([10, 20]),
        "observations" => Nx.broadcast(0.5, {@batch, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @action_horizon, @action_dim}
    end

    test "output contains finite values" do
      model = DDIM.build(@ddim_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim}),
        "timestep" => Nx.tensor([10, 20]),
        "observations" => Nx.broadcast(0.5, {@batch, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "DDIM.make_schedule/1" do
    test "returns schedule map with expected keys" do
      schedule = DDIM.make_schedule(num_steps: 100)

      assert is_map(schedule)
      assert schedule.num_steps == 100
      assert %Nx.Tensor{} = schedule.alphas_cumprod
      assert %Nx.Tensor{} = schedule.sqrt_alphas_cumprod
      assert %Nx.Tensor{} = schedule.sqrt_one_minus_alphas_cumprod
    end
  end

  # ============================================================================
  # LatentDiffusion Tests
  # ============================================================================

  describe "LatentDiffusion.build/1" do
    @input_size 32
    @latent_size 8

    @ld_opts [
      input_size: @input_size,
      latent_size: @latent_size,
      hidden_size: @hidden_size,
      num_layers: 2,
      num_steps: 100
    ]

    test "returns {encoder, decoder, denoiser} tuple" do
      {encoder, decoder, denoiser} = LatentDiffusion.build(@ld_opts)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
      assert %Axon{} = denoiser
    end

    test "encoder outputs mu and log_var with correct shapes" do
      {encoder, _decoder, _denoiser} = LatentDiffusion.build(@ld_opts)

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @input_size}))

      assert %{mu: mu, log_var: log_var} = output
      assert Nx.shape(mu) == {@batch, @latent_size}
      assert Nx.shape(log_var) == {@batch, @latent_size}
    end

    test "decoder reconstructs to input_size" do
      {_encoder, decoder, _denoiser} = LatentDiffusion.build(@ld_opts)

      {init_fn, predict_fn} = Axon.build(decoder)
      params = init_fn.(Nx.template({@batch, @latent_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @latent_size}))

      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "denoiser predicts noise in latent space" do
      {_encoder, _decoder, denoiser} = LatentDiffusion.build(@ld_opts)

      {init_fn, predict_fn} = Axon.build(denoiser)

      input = %{
        "noisy_z" => Nx.broadcast(0.5, {@batch, @latent_size}),
        "timestep" => Nx.tensor([10, 20])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_size}
    end

    test "all outputs contain finite values" do
      {encoder, decoder, denoiser} = LatentDiffusion.build(@ld_opts)

      # Test encoder
      {enc_init, enc_pred} = Axon.build(encoder)
      enc_params = enc_init.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())
      enc_out = enc_pred.(enc_params, Nx.broadcast(0.5, {@batch, @input_size}))
      assert Nx.all(Nx.is_nan(enc_out.mu) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(enc_out.log_var) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Test decoder
      {dec_init, dec_pred} = Axon.build(decoder)
      dec_params = dec_init.(Nx.template({@batch, @latent_size}, :f32), Axon.ModelState.empty())
      dec_out = dec_pred.(dec_params, Nx.broadcast(0.5, {@batch, @latent_size}))
      assert Nx.all(Nx.is_nan(dec_out) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Test denoiser
      {den_init, den_pred} = Axon.build(denoiser)
      den_input = %{
        "noisy_z" => Nx.broadcast(0.5, {@batch, @latent_size}),
        "timestep" => Nx.tensor([10, 20])
      }
      den_params = den_init.(den_input, Axon.ModelState.empty())
      den_out = den_pred.(den_params, den_input)
      assert Nx.all(Nx.is_nan(den_out) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "LatentDiffusion.make_schedule/1" do
    test "returns schedule map with expected keys" do
      schedule = LatentDiffusion.make_schedule(num_steps: 100)

      assert is_map(schedule)
      assert schedule.num_steps == 100
      assert %Nx.Tensor{} = schedule.alphas_cumprod
    end
  end

  # ============================================================================
  # ConsistencyModel Tests
  # ============================================================================

  describe "ConsistencyModel.build/1" do
    @cm_opts [
      input_dim: @input_dim,
      hidden_size: @hidden_size,
      num_layers: 2
    ]

    test "builds an Axon model" do
      model = ConsistencyModel.build(@cm_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = ConsistencyModel.build(@cm_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "sigma" => Nx.tensor([1.0, 5.0])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
    end

    test "output contains finite values" do
      model = ConsistencyModel.build(@cm_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "sigma" => Nx.tensor([1.0, 5.0])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "ConsistencyModel.consistency_loss/2" do
    test "returns a scalar loss" do
      pred_current = Nx.broadcast(0.5, {4, @input_dim})
      pred_target = Nx.broadcast(0.3, {4, @input_dim})

      loss = ConsistencyModel.consistency_loss(pred_current, pred_target)

      assert Nx.shape(loss) == {}
    end

    test "loss is zero when predictions match" do
      pred = Nx.broadcast(0.5, {4, @input_dim})

      loss = ConsistencyModel.consistency_loss(pred, pred)

      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-5
    end
  end

  describe "ConsistencyModel.output_size/1" do
    test "returns input_dim" do
      assert ConsistencyModel.output_size(input_dim: @input_dim) == @input_dim
    end
  end

  # ============================================================================
  # ScoreSDE Tests
  # ============================================================================

  describe "ScoreSDE.build/1" do
    @score_opts [
      input_dim: @input_dim,
      hidden_size: @hidden_size,
      num_layers: 2,
      sde_type: :vp
    ]

    test "builds an Axon model" do
      model = ScoreSDE.build(@score_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = ScoreSDE.build(@score_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([0.3, 0.7])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
    end

    test "output contains finite values" do
      model = ScoreSDE.build(@score_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([0.3, 0.7])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "ScoreSDE.dsm_loss/3" do
    test "returns a scalar loss" do
      score_pred = Nx.broadcast(0.1, {4, @input_dim})
      noise = Nx.broadcast(0.5, {4, @input_dim})
      sigma = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      loss = ScoreSDE.dsm_loss(score_pred, noise, sigma)

      assert Nx.shape(loss) == {}
    end

    test "loss is non-negative" do
      score_pred = Nx.broadcast(0.1, {4, @input_dim})
      noise = Nx.broadcast(0.5, {4, @input_dim})
      sigma = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      loss = ScoreSDE.dsm_loss(score_pred, noise, sigma)

      assert Nx.to_number(loss) >= 0.0
    end
  end

  describe "ScoreSDE schedules" do
    test "vp_schedule returns map with expected keys" do
      schedule = ScoreSDE.vp_schedule()

      assert schedule.type == :vp
      assert is_number(schedule.beta_min)
      assert is_number(schedule.beta_max)
    end

    test "ve_schedule returns map with expected keys" do
      schedule = ScoreSDE.ve_schedule()

      assert schedule.type == :ve
      assert is_number(schedule.sigma_min)
      assert is_number(schedule.sigma_max)
    end
  end

  describe "ScoreSDE.output_size/1" do
    test "returns input_dim" do
      assert ScoreSDE.output_size(input_dim: @input_dim) == @input_dim
    end
  end
end
