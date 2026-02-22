defmodule Edifice.CoverageBatchFTest do
  @moduledoc """
  Coverage tests targeting uncovered code paths in lowest-coverage modules.
  Focuses on utility functions, loss functions, schedule functions, and
  alternative code branches NOT already tested in batches A-E.
  """
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  @batch 2
  @seq_len 8
  @embed 16

  # ==========================================================================
  # MambaSSD (41.94%) - Different chunk_size values, utility functions,
  # chunked_ssd_matmul path (seq_len > chunk_size in training_mode)
  # ==========================================================================
  describe "MambaSSD utility functions and chunk branches" do
    alias Edifice.SSM.MambaSSD

    test "recommended_defaults returns keyword list with chunk_size and training_mode" do
      defaults = MambaSSD.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :chunk_size) == 16
      assert Keyword.get(defaults, :training_mode) == false
    end

    test "training_defaults returns keyword list with training_mode: true" do
      defaults = MambaSSD.training_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :training_mode) == true
      assert Keyword.get(defaults, :chunk_size) == 32
    end

    test "output_size delegates to Common" do
      size = MambaSSD.output_size(hidden_size: 128)
      assert size == 128
    end

    test "output_size with default" do
      size = MambaSSD.output_size()
      assert is_integer(size)
      assert size > 0
    end

    test "param_count returns positive integer" do
      count =
        MambaSSD.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1
        )

      assert is_integer(count)
      assert count > 0
    end

    test "training_mode with chunk_size larger than seq_len (single chunk matmul path)" do
      # seq_len=8, chunk_size=16 => single chunk matmul path
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: true,
          chunk_size: 16
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "training_mode with small chunk_size triggers multi-chunk matmul path" do
      # seq_len=8, chunk_size=2 => 4 chunks, hits chunked_ssd_matmul
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: true,
          chunk_size: 2
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "inference mode with chunk_size larger than seq_len (simple scan path)" do
      # seq_len=8, chunk_size=16 => seq fits in one chunk, uses simple scan
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: false,
          chunk_size: 16
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "inference mode with small chunk_size triggers chunked scan with remainder" do
      # seq_len=8, chunk_size=3 => 2 full chunks + 2 remainder
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          seq_len: @seq_len,
          training_mode: false,
          chunk_size: 3
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # GatedSSM (57.92%) - Utility functions: param_count, output_size,
  # recommended_defaults, init_cache, step function
  # ==========================================================================
  describe "GatedSSM utility functions" do
    alias Edifice.SSM.GatedSSM

    test "output_size returns hidden_size" do
      assert GatedSSM.output_size(hidden_size: 128) == 128
    end

    test "output_size with default" do
      assert GatedSSM.output_size() == 256
    end

    test "param_count returns positive integer" do
      count =
        GatedSSM.param_count(
          embed_dim: @embed,
          hidden_size: 32,
          state_size: 4,
          expand_factor: 2,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count with embed_dim == hidden_size (no input projection)" do
      count1 =
        GatedSSM.param_count(
          embed_dim: 256,
          hidden_size: 256,
          state_size: 16,
          num_layers: 2
        )

      count2 =
        GatedSSM.param_count(
          embed_dim: 128,
          hidden_size: 256,
          state_size: 16,
          num_layers: 2
        )

      # Different embed_dim => different total (count2 includes input_proj)
      assert count2 > count1
    end

    test "recommended_defaults returns keyword list with expected keys" do
      defaults = GatedSSM.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :state_size) == 16
      assert Keyword.get(defaults, :dropout) == 0.1
      assert Keyword.get(defaults, :window_size) == 60
    end

    test "init_cache returns proper cache structure" do
      cache =
        GatedSSM.init_cache(
          batch_size: 2,
          hidden_size: 32,
          state_size: 4,
          expand_factor: 2,
          conv_size: 4,
          num_layers: 2
        )

      assert cache.step == 0
      assert is_map(cache.layers)
      assert Map.has_key?(cache.layers, "layer_1")
      assert Map.has_key?(cache.layers, "layer_2")

      layer1 = cache.layers["layer_1"]
      assert Map.has_key?(layer1, :h)
      assert Map.has_key?(layer1, :conv_buffer)

      # hidden_size * expand_factor = 32 * 2 = 64
      assert Nx.shape(layer1.h) == {2, 64, 4}
      # conv_size - 1 = 3
      assert Nx.shape(layer1.conv_buffer) == {2, 3, 64}
    end

    test "init_cache with default options" do
      cache = GatedSSM.init_cache()
      assert cache.step == 0
      assert cache.config.hidden_size == 256
      assert cache.config.num_layers == 2
    end

    test "build_mamba_block standalone" do
      input = Axon.input("input", shape: {nil, @seq_len, @embed})

      block =
        GatedSSM.build_mamba_block(input,
          hidden_size: @embed,
          state_size: 4,
          expand_factor: 2,
          conv_size: 2,
          name: "test_block"
        )

      assert %Axon{} = block

      {init_fn, predict_fn} = Axon.build(block)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert {_b, _s, _h} = Nx.shape(output)
    end
  end

  # ==========================================================================
  # NormalizingFlow (60.34%) - Loss functions: log_det_jacobian,
  # total_log_det_jacobian, log_probability, nll_loss, inverse_coupling_layer
  # ==========================================================================
  describe "NormalizingFlow loss and density functions" do
    alias Edifice.Generative.NormalizingFlow

    test "log_det_jacobian computes sum of scale" do
      scale = Nx.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
      result = NormalizingFlow.log_det_jacobian(scale)
      assert Nx.shape(result) == {2}
      # First batch: 1 + 2 + 3 = 6
      assert abs(Nx.to_number(result[0]) - 6.0) < 1.0e-5
      # Second batch: 0.5 + 0.5 + 0.5 = 1.5
      assert abs(Nx.to_number(result[1]) - 1.5) < 1.0e-5
    end

    test "total_log_det_jacobian sums across all layers" do
      scales = [
        Nx.tensor([[1.0, 2.0], [0.5, 0.5]]),
        Nx.tensor([[0.3, 0.4], [0.1, 0.2]])
      ]

      result = NormalizingFlow.total_log_det_jacobian(scales)
      assert Nx.shape(result) == {2}
      # First batch: (1+2) + (0.3+0.4) = 3.7
      assert abs(Nx.to_number(result[0]) - 3.7) < 1.0e-5
    end

    test "log_probability returns correct shape" do
      z = Nx.broadcast(0.0, {@batch, @embed})
      total_log_det = Nx.tensor([0.5, 0.3])

      result = NormalizingFlow.log_probability(z, total_log_det)
      assert Nx.shape(result) == {@batch}
      # z=0 => log_pz = -0.5 * d * log(2*pi) + total_log_det
      # Should be finite
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "nll_loss returns scalar" do
      z = Nx.broadcast(0.1, {@batch, @embed})
      total_log_det = Nx.tensor([0.5, 0.3])

      loss = NormalizingFlow.nll_loss(z, total_log_det)
      assert Nx.shape(loss) == {}
      refute Nx.is_nan(loss) |> Nx.to_number() == 1
    end

    test "inverse_coupling_layer inverts the affine transform" do
      # inverse_coupling_layer is defn: half_size must be a compile-time integer,
      # not a traced tensor. We wrap the call inside a defn where we hardcode half_size.
      half_size = div(@embed, 2)

      hidden_weights = Nx.broadcast(0.1, {half_size, 32})
      hidden_biases = Nx.broadcast(0.0, {32})
      scale_w = Nx.broadcast(0.01, {32, half_size})
      scale_b = Nx.broadcast(0.0, {half_size})
      trans_w = Nx.broadcast(0.01, {32, half_size})
      trans_b = Nx.broadcast(0.0, {half_size})

      y = Nx.broadcast(0.5, {@batch, @embed})

      # Use Nx.Defn.jit with a closure that captures the integer half_size
      # so it stays a compile-time constant inside the defn trace
      inverse_fn =
        Nx.Defn.jit(fn y_in, hw, hb, sw, sb, tw, tb ->
          NormalizingFlow.inverse_coupling_layer(
            y_in,
            hw,
            hb,
            {sw, sb},
            {tw, tb},
            8
          )
        end)

      result = inverse_fn.(y, hidden_weights, hidden_biases, scale_w, scale_b, trans_w, trans_b)

      assert Nx.shape(result) == {@batch, @embed}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "build with different num_flows values" do
      for num_flows <- [1, 3, 6] do
        model = NormalizingFlow.build(input_size: @embed, num_flows: num_flows)
        assert %Axon{} = model

        {init_fn, predict_fn} = Axon.build(model)
        params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
        output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
        assert Nx.shape(output) == {@batch, @embed}
      end
    end
  end

  # ==========================================================================
  # FNet (63.33%) - Utility functions: param_count, output_size,
  # recommended_defaults, fourier_mixing, multi-layer build
  # ==========================================================================
  describe "FNet utility functions and branches" do
    alias Edifice.Attention.FNet

    test "output_size returns hidden_size" do
      assert FNet.output_size(hidden_size: 128) == 128
    end

    test "output_size with default" do
      assert FNet.output_size() == 256
    end

    test "param_count returns positive integer" do
      count =
        FNet.param_count(
          embed_dim: @embed,
          hidden_size: 32,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count with embed_dim == hidden_size (no input proj)" do
      count1 = FNet.param_count(embed_dim: 256, hidden_size: 256, num_layers: 2)
      count2 = FNet.param_count(embed_dim: 128, hidden_size: 256, num_layers: 2)
      # Different embed_dim adds input projection params
      assert count2 > count1
    end

    test "recommended_defaults returns expected config" do
      defaults = FNet.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 4
      assert Keyword.get(defaults, :dropout) == 0.1
    end

    test "fourier_mixing_real produces correct shape" do
      tensor = Nx.broadcast(1.0, {@batch, @seq_len, @embed})
      dft_seq = FNet.dft_real_matrix(@seq_len)
      dft_hidden = FNet.dft_real_matrix(@embed)
      result = FNet.fourier_mixing_real(tensor, dft_seq, dft_hidden)
      assert Nx.shape(result) == {@batch, @seq_len, @embed}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "build with multiple layers" do
      model =
        FNet.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # Zamba (65.18%) - Utility functions: param_count, output_size,
  # recommended_defaults, layer_pattern, compare_to_jamba
  # ==========================================================================
  describe "Zamba utility functions" do
    alias Edifice.SSM.Zamba

    test "output_size returns hidden_size" do
      assert Zamba.output_size(hidden_size: 128) == 128
    end

    test "output_size with default" do
      assert Zamba.output_size() == 256
    end

    test "param_count returns positive integer" do
      count =
        Zamba.param_count(
          embed_dim: @embed,
          hidden_size: 32,
          num_layers: 4,
          num_heads: 2,
          head_dim: 16
        )

      assert is_integer(count)
      assert count > 0
    end

    test "recommended_defaults returns expected config" do
      defaults = Zamba.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :attention_every) == 3
      assert Keyword.get(defaults, :num_heads) == 4
    end

    test "layer_pattern with default config" do
      pattern = Zamba.layer_pattern()
      assert length(pattern) == 6
      # Every 3rd layer should be :mamba_attention
      assert Enum.at(pattern, 2) == :mamba_attention
      assert Enum.at(pattern, 5) == :mamba_attention
      assert Enum.at(pattern, 0) == :mamba
    end

    test "layer_pattern with custom config" do
      pattern = Zamba.layer_pattern(num_layers: 4, attention_every: 2)
      assert length(pattern) == 4
      assert pattern == [:mamba, :mamba_attention, :mamba, :mamba_attention]
    end

    test "compare_to_jamba returns savings info" do
      result =
        Zamba.compare_to_jamba(
          embed_dim: @embed,
          hidden_size: 32,
          num_layers: 6,
          attention_every: 3,
          num_heads: 2,
          head_dim: 16
        )

      assert is_map(result)
      assert Map.has_key?(result, :zamba_params)
      assert Map.has_key?(result, :jamba_params)
      assert Map.has_key?(result, :savings)
      assert Map.has_key?(result, :savings_percent)
      # Jamba should have more params than Zamba
      assert result.jamba_params >= result.zamba_params
      assert result.savings >= 0
    end

    test "build with different attention_every values" do
      # attention_every: 1 means shared attention after every Mamba layer
      model =
        Zamba.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          attention_every: 1,
          seq_len: @seq_len,
          state_size: 4,
          num_heads: 2,
          head_dim: 8
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # Bayesian (65.91%) - kl_cost, elbo_loss, different activation options
  # ==========================================================================
  describe "Bayesian loss functions and options" do
    alias Edifice.Probabilistic.Bayesian

    test "kl_cost with default prior_sigma" do
      mu = Nx.broadcast(0.5, {4, 8})
      rho = Nx.broadcast(-2.0, {4, 8})

      kl = Bayesian.kl_cost(mu, rho)
      assert Nx.shape(kl) == {}
      assert Nx.to_number(kl) > 0
      refute Nx.is_nan(kl) |> Nx.to_number() == 1
    end

    test "kl_cost with custom prior_sigma" do
      mu = Nx.broadcast(0.5, {4, 8})
      rho = Nx.broadcast(-2.0, {4, 8})

      kl_default = Bayesian.kl_cost(mu, rho)
      kl_custom = Bayesian.kl_cost(mu, rho, prior_sigma: 2.0)

      # Both should be positive and finite
      assert Nx.to_number(kl_default) > 0
      assert Nx.to_number(kl_custom) > 0
      # Custom prior_sigma changes the result
      refute Nx.to_number(kl_default) == Nx.to_number(kl_custom)
    end

    test "kl_cost is zero when posterior matches prior" do
      # mu=0, sigma=1 (rho s.t. softplus(rho) = 1 => rho ~ 0.5413)
      rho_val = :math.log(:math.exp(1.0) - 1.0)
      mu = Nx.broadcast(0.0, {4, 8})
      rho = Nx.broadcast(rho_val, {4, 8})

      kl = Bayesian.kl_cost(mu, rho, prior_sigma: 1.0)
      assert abs(Nx.to_number(kl)) < 0.1
    end

    test "elbo_loss with default beta" do
      predictions = Nx.broadcast(0.5, {@batch, 4})
      targets = Nx.broadcast(0.7, {@batch, 4})
      kl = Nx.tensor(0.1)

      loss = Bayesian.elbo_loss(predictions, targets, kl)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "elbo_loss with custom beta" do
      predictions = Nx.broadcast(0.5, {@batch, 4})
      targets = Nx.broadcast(0.7, {@batch, 4})
      kl = Nx.tensor(1.0)

      loss_default = Bayesian.elbo_loss(predictions, targets, kl)
      loss_low_beta = Bayesian.elbo_loss(predictions, targets, kl, beta: 0.01)

      # Lower beta should give smaller total loss (less KL penalty)
      assert Nx.to_number(loss_low_beta) < Nx.to_number(loss_default)
    end

    test "build with :silu activation" do
      model =
        Bayesian.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          activation: :silu
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 4}
    end

    test "build with multiple hidden layers" do
      model =
        Bayesian.build(
          input_size: @embed,
          hidden_sizes: [32, 16, 8],
          output_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 4}
    end
  end

  # ==========================================================================
  # ScoreSDE (66.13%) - Schedule functions: vp_schedule, ve_schedule,
  # vp_marginal, ve_sigma, dsm_loss
  # ==========================================================================
  describe "ScoreSDE schedules and loss" do
    alias Edifice.Generative.ScoreSDE

    test "vp_schedule returns map with expected keys" do
      schedule = ScoreSDE.vp_schedule()
      assert schedule.type == :vp
      assert schedule.beta_min == 0.1
      assert schedule.beta_max == 20.0
    end

    test "vp_schedule with custom parameters" do
      schedule = ScoreSDE.vp_schedule(beta_min: 0.01, beta_max: 10.0)
      assert schedule.beta_min == 0.01
      assert schedule.beta_max == 10.0
    end

    test "ve_schedule returns map with expected keys" do
      schedule = ScoreSDE.ve_schedule()
      assert schedule.type == :ve
      assert schedule.sigma_min == 0.01
      assert schedule.sigma_max == 50.0
    end

    test "ve_schedule with custom parameters" do
      schedule = ScoreSDE.ve_schedule(sigma_min: 0.001, sigma_max: 100.0)
      assert schedule.sigma_min == 0.001
      assert schedule.sigma_max == 100.0
    end

    test "vp_marginal returns mean_coeff and std" do
      schedule = ScoreSDE.vp_schedule()
      # Strip :type atom — defn can't trace atoms through Nx.LazyContainer
      numeric_schedule = Map.drop(schedule, [:type])
      t = Nx.tensor(0.5)

      {mean_coeff, std} = ScoreSDE.vp_marginal(t, numeric_schedule)
      assert Nx.shape(mean_coeff) == {}
      assert Nx.shape(std) == {}
      # At t=0, mean_coeff should be ~1 and std ~0
      # At t=0.5, both should be between 0 and 1
      assert Nx.to_number(mean_coeff) > 0
      assert Nx.to_number(mean_coeff) < 1
      assert Nx.to_number(std) > 0
    end

    test "vp_marginal at t=0 is near identity" do
      schedule = ScoreSDE.vp_schedule()
      numeric_schedule = Map.drop(schedule, [:type])
      t = Nx.tensor(0.001)

      {mean_coeff, std} = ScoreSDE.vp_marginal(t, numeric_schedule)
      # Near t=0: mean_coeff ~ 1, std ~ 0
      assert Nx.to_number(mean_coeff) > 0.99
      assert Nx.to_number(std) < 0.1
    end

    test "ve_sigma returns noise level" do
      schedule = ScoreSDE.ve_schedule()
      numeric_schedule = Map.drop(schedule, [:type])
      t = Nx.tensor(0.5)

      sigma = ScoreSDE.ve_sigma(t, numeric_schedule)
      assert Nx.shape(sigma) == {}
      # sigma should be between sigma_min and sigma_max
      val = Nx.to_number(sigma)
      assert val > 0.01
      assert val < 50.0
    end

    test "ve_sigma at t=0 equals sigma_min" do
      schedule = ScoreSDE.ve_schedule()
      numeric_schedule = Map.drop(schedule, [:type])
      t = Nx.tensor(0.0)

      sigma = ScoreSDE.ve_sigma(t, numeric_schedule)
      assert abs(Nx.to_number(sigma) - 0.01) < 1.0e-5
    end

    test "ve_sigma at t=1 equals sigma_max" do
      schedule = ScoreSDE.ve_schedule()
      numeric_schedule = Map.drop(schedule, [:type])
      t = Nx.tensor(1.0)

      sigma = ScoreSDE.ve_sigma(t, numeric_schedule)
      assert abs(Nx.to_number(sigma) - 50.0) < 0.1
    end

    test "dsm_loss returns non-negative scalar" do
      score_pred = Nx.broadcast(0.1, {@batch, @embed})
      noise = Nx.broadcast(0.5, {@batch, @embed})
      sigma = Nx.tensor([0.3, 0.7])

      loss = ScoreSDE.dsm_loss(score_pred, noise, sigma)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0
      refute Nx.is_nan(loss) |> Nx.to_number() == 1
    end

    test "output_size returns input_dim" do
      assert ScoreSDE.output_size(input_dim: 32) == 32
    end

    test "param_count returns positive integer" do
      count = ScoreSDE.param_count(input_dim: @embed, hidden_size: 32, num_layers: 2)
      assert is_integer(count)
      assert count > 0
    end

    test "recommended_defaults returns expected config" do
      defaults = ScoreSDE.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :sde_type) == :vp
      assert Keyword.get(defaults, :num_layers) == 4
    end
  end

  # ==========================================================================
  # LatentDiffusion (69.12%) - reparameterize, kl_divergence, make_schedule,
  # utility functions
  # ==========================================================================
  describe "LatentDiffusion VAE ops and schedule" do
    alias Edifice.Generative.LatentDiffusion

    test "reparameterize produces sample of correct shape" do
      mu = Nx.broadcast(0.0, {@batch, 8})
      log_var = Nx.broadcast(0.0, {@batch, 8})
      key = Nx.Random.key(42)

      {z, _new_key} = LatentDiffusion.reparameterize(mu, log_var, key)
      assert Nx.shape(z) == {@batch, 8}
      refute Nx.any(Nx.is_nan(z)) |> Nx.to_number() == 1
    end

    test "reparameterize with zero log_var samples near mu" do
      mu = Nx.broadcast(5.0, {1, 8})
      # Very negative log_var => very small std => samples near mu
      log_var = Nx.broadcast(-20.0, {1, 8})
      key = Nx.Random.key(42)

      {z, _} = LatentDiffusion.reparameterize(mu, log_var, key)
      # Should be very close to mu=5.0
      mean_val = Nx.to_number(Nx.mean(z))
      assert abs(mean_val - 5.0) < 0.1
    end

    test "kl_divergence returns non-negative scalar" do
      mu = Nx.broadcast(0.5, {@batch, 8})
      log_var = Nx.broadcast(-1.0, {@batch, 8})

      kl = LatentDiffusion.kl_divergence(mu, log_var)
      assert Nx.shape(kl) == {}
      assert Nx.to_number(kl) >= 0
    end

    test "kl_divergence is zero for standard normal" do
      mu = Nx.broadcast(0.0, {@batch, 8})
      log_var = Nx.broadcast(0.0, {@batch, 8})

      kl = LatentDiffusion.kl_divergence(mu, log_var)
      assert abs(Nx.to_number(kl)) < 1.0e-5
    end

    test "make_schedule returns proper schedule map" do
      schedule = LatentDiffusion.make_schedule(num_steps: 100)

      assert Map.has_key?(schedule, :num_steps)
      assert schedule.num_steps == 100
      assert Map.has_key?(schedule, :alphas_cumprod)
      assert Map.has_key?(schedule, :sqrt_alphas_cumprod)
      assert Map.has_key?(schedule, :sqrt_one_minus_alphas_cumprod)

      # alphas_cumprod should be monotonically decreasing
      ac = schedule.alphas_cumprod
      first = Nx.to_number(ac[0])
      last = Nx.to_number(ac[99])
      assert first > last
    end

    test "make_schedule with custom beta values" do
      schedule = LatentDiffusion.make_schedule(num_steps: 50, beta_start: 0.001, beta_end: 0.05)

      assert schedule.num_steps == 50
      ac = schedule.alphas_cumprod
      assert Nx.shape(ac) == {50}
      # First alpha should be near 1
      assert Nx.to_number(ac[0]) > 0.99
    end

    test "make_schedule default produces 1000 steps" do
      schedule = LatentDiffusion.make_schedule()
      assert schedule.num_steps == 1000
      assert Nx.shape(schedule.alphas_cumprod) == {1000}
    end

    test "output_size returns latent_size" do
      assert LatentDiffusion.output_size(latent_size: 64) == 64
    end

    test "param_count returns positive integer" do
      count =
        LatentDiffusion.param_count(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 32,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "recommended_defaults returns expected config" do
      defaults = LatentDiffusion.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :latent_size) == 32
      assert Keyword.get(defaults, :num_steps) == 1000
    end
  end

  # ==========================================================================
  # GAT (70.59%) - attention_coefficients, num_layers=1 branch,
  # negative_slope option, dropout option
  # ==========================================================================
  describe "GAT attention coefficients and branches" do
    alias Edifice.Graph.GAT

    @num_nodes 4

    test "attention_coefficients returns normalized attention matrix" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, 8})
      adjacency = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model = GAT.attention_coefficients(nodes, adjacency, 4, name: "test_attn")

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, 8}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix =
        Nx.tensor([
          [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
          [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]]
        ])
        |> Nx.as_type(:f32)

      output =
        predict_fn.(params, %{
          "nodes" => Nx.broadcast(0.5, {@batch, @num_nodes, 8}),
          "adjacency" => adj_matrix
        })

      # Output should be [batch, num_nodes, num_nodes]
      assert Nx.shape(output) == {@batch, @num_nodes, @num_nodes}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "build with num_layers: 1 (skip hidden layers, only output)" do
      model =
        GAT.build(
          input_dim: 8,
          hidden_size: 4,
          num_heads: 2,
          num_classes: 3,
          num_layers: 1
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, 8}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})

      output =
        predict_fn.(params, %{
          "nodes" => Nx.broadcast(0.5, {@batch, @num_nodes, 8}),
          "adjacency" => adj
        })

      assert {_b, _n, _c} = Nx.shape(output)
    end

    test "build with dropout > 0" do
      model =
        GAT.build(
          input_dim: 8,
          hidden_size: 4,
          num_heads: 2,
          num_classes: 3,
          num_layers: 2,
          dropout: 0.3
        )

      assert %Axon{} = model
    end

    test "gat_layer with negative_slope option" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, 8})
      adjacency = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model =
        GAT.gat_layer(nodes, adjacency, 4,
          num_heads: 2,
          negative_slope: 0.1,
          activation: :relu,
          name: "gat_neg_slope"
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, 8}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})

      output =
        predict_fn.(params, %{
          "nodes" => Nx.broadcast(0.5, {@batch, @num_nodes, 8}),
          "adjacency" => adj
        })

      # concat_heads: true (default) => num_heads * output_dim = 2 * 4 = 8
      assert Nx.shape(output) == {@batch, @num_nodes, 8}
    end

    test "gat_layer with concat_heads: false (average heads)" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, 8})
      adjacency = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model =
        GAT.gat_layer(nodes, adjacency, 4,
          num_heads: 2,
          concat_heads: false,
          name: "gat_avg_heads"
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, 8}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})

      output =
        predict_fn.(params, %{
          "nodes" => Nx.broadcast(0.5, {@batch, @num_nodes, 8}),
          "adjacency" => adj
        })

      # concat_heads: false => output_dim = 4 (averaged over heads)
      assert Nx.shape(output) == {@batch, @num_nodes, 4}
    end
  end

  # ==========================================================================
  # MessagePassing (70.91%) - aggregate with :sum, global_pool :mean,
  # message_passing_layer with dropout, :sum aggregation forward
  # ==========================================================================
  describe "MessagePassing remaining branches" do
    alias Edifice.Graph.MessagePassing

    @num_nodes 4
    @feature_dim 8

    test "aggregate/3 with :sum mode" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model = MessagePassing.aggregate(nodes, adj, :sum)

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(1.0, {@batch, @num_nodes, @feature_dim})

      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      assert {_b, _n, _f} = Nx.shape(output)
    end

    test "global_pool with default :mean mode" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @num_nodes, @feature_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @num_nodes, @feature_dim}))
      assert Nx.shape(output) == {@batch, @feature_dim}
    end

    test "message_passing_layer with :sum aggregation forward pass" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model =
        MessagePassing.message_passing_layer(nodes, adj, @embed,
          aggregation: :sum,
          name: "mpnn_sum"
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix =
        Nx.tensor([
          [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
          [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
        ])
        |> Nx.as_type(:f32)

      node_feats = Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim})
      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      assert Nx.shape(output) == {@batch, @num_nodes, @embed}
    end

    test "message_passing_layer with dropout > 0" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      model =
        MessagePassing.message_passing_layer(nodes, adj, @embed,
          aggregation: :sum,
          dropout: 0.2,
          name: "mpnn_dropout"
        )

      assert %Axon{} = model
    end
  end

  # ==========================================================================
  # KAN (72.90%) - Utility functions: output_size, param_count,
  # recommended_defaults, basis functions, grid_size variants
  # ==========================================================================
  describe "KAN utility functions and basis variants" do
    alias Edifice.Feedforward.KAN

    test "output_size returns hidden_size" do
      assert KAN.output_size(hidden_size: 128) == 128
    end

    test "output_size with default" do
      assert KAN.output_size() == 256
    end

    test "param_count returns positive integer" do
      count =
        KAN.param_count(
          embed_dim: @embed,
          hidden_size: 32,
          num_layers: 2,
          grid_size: 4
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count increases with grid_size" do
      count_small =
        KAN.param_count(embed_dim: @embed, hidden_size: 32, num_layers: 1, grid_size: 4)

      count_large =
        KAN.param_count(embed_dim: @embed, hidden_size: 32, num_layers: 1, grid_size: 16)

      assert count_large > count_small
    end

    test "recommended_defaults returns expected config" do
      defaults = KAN.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :grid_size) == 8
      assert Keyword.get(defaults, :basis) == :bspline
      assert Keyword.get(defaults, :hidden_size) == 256
    end

    test "sine_basis function produces correct output" do
      x = Nx.broadcast(0.5, {@batch, @seq_len, @embed})
      frequencies = Nx.broadcast(1.0, {4})
      phases = Nx.broadcast(0.0, {4})

      result = KAN.sine_basis(x, frequencies, phases)
      # Should have extra grid dimension
      assert {_b, _s, _e, 4} = Nx.shape(result)
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "chebyshev_basis function produces correct output" do
      x = Nx.broadcast(0.5, {@batch, @seq_len, @embed})

      result = KAN.chebyshev_basis(x, 5)
      # Stacks [T0, T1, T2, T3, T4] along last axis
      assert {_b, _s, _e, 5} = Nx.shape(result)
      # T0(x) = 1 for all x
      t0_slice = result[[.., .., .., 0]]
      assert Nx.to_number(Nx.mean(t0_slice)) |> abs() |> Kernel.-(1.0) |> abs() < 0.01
    end

    test "rbf_basis function produces correct output" do
      x = Nx.broadcast(0.5, {@batch, @seq_len, @embed})
      centers = Nx.tensor([-1.0, 0.0, 1.0, 2.0])
      sigma = 1.0

      result = KAN.rbf_basis(x, centers, sigma)
      assert {_b, _s, _e, 4} = Nx.shape(result)
      # All values should be positive (Gaussian)
      assert Nx.to_number(Nx.reduce_min(result)) >= 0
    end

    test "build with grid_size: 4 (non-default)" do
      model =
        KAN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          grid_size: 4,
          basis: :sine
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with dropout > 0 and multiple layers" do
      model =
        KAN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          seq_len: @seq_len,
          dropout: 0.1,
          basis: :sine
        )

      assert %Axon{} = model
    end

    test "default_* accessor functions" do
      assert KAN.default_hidden_size() == 256
      assert KAN.default_num_layers() == 4
      assert KAN.default_grid_size() == 8
      assert KAN.default_basis() == :bspline
      assert KAN.default_dropout() == 0.0
      assert KAN.eps() == 1.0e-6
    end
  end

  # ==========================================================================
  # MoE (39.46%) - compute_aux_loss variations, estimate_speedup,
  # recommended_defaults, GLU expert, forward passes with different routing
  # ==========================================================================
  describe "MoE utility functions and routing variants" do
    alias Edifice.Meta.MoE

    test "compute_aux_loss with custom load_balance_weight" do
      router_probs = Nx.broadcast(0.25, {@batch, @seq_len, 4})
      expert_mask = Nx.broadcast(1.0, {@batch, @seq_len, 4})

      loss_default = MoE.compute_aux_loss(router_probs, expert_mask)
      loss_high = MoE.compute_aux_loss(router_probs, expert_mask, load_balance_weight: 0.1)

      # Higher weight => larger aux loss
      assert Nx.to_number(loss_high) > Nx.to_number(loss_default)
    end

    test "compute_aux_loss with unbalanced routing" do
      router_probs = Nx.broadcast(0.25, {@batch, @seq_len, 4})

      # Only expert 0 gets all tokens
      expert_mask =
        Nx.concatenate(
          [
            Nx.broadcast(1.0, {@batch, @seq_len, 1}),
            Nx.broadcast(0.0, {@batch, @seq_len, 3})
          ],
          axis: -1
        )

      loss = MoE.compute_aux_loss(router_probs, expert_mask)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "estimate_speedup with default expert_fraction" do
      speedup = MoE.estimate_speedup(8, 2)
      assert is_float(speedup)
      assert speedup > 1.0
    end

    test "estimate_speedup with various configurations" do
      # More experts with same top_k => more speedup
      speedup_8_2 = MoE.estimate_speedup(8, 2, 0.5)
      speedup_16_2 = MoE.estimate_speedup(16, 2, 0.5)
      assert speedup_16_2 > speedup_8_2

      # top_k == num_experts => no speedup from experts
      speedup_same = MoE.estimate_speedup(4, 4, 0.5)
      assert abs(speedup_same - 1.0) < 0.01
    end

    test "recommended_defaults returns expected config" do
      defaults = MoE.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :num_experts) == 8
      assert Keyword.get(defaults, :top_k) == 2
      assert Keyword.get(defaults, :routing) == :top_k
      assert Keyword.get(defaults, :moe_every) == 2
    end

    test "build with :glu expert_type" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          top_k: 1,
          expert_type: :glu
        )

      assert %Axon{} = model
    end

    test "forward pass with 2 experts top_k routing" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          top_k: 1,
          routing: :top_k
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"moe_input" => Nx.template({@batch, @seq_len, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "moe_input" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})
        })

      assert {_b, _s, _o} = Nx.shape(output)
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "forward pass with :hash routing" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          routing: :hash
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"moe_input" => Nx.template({@batch, @seq_len, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "moe_input" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})
        })

      assert {_b, _s, _o} = Nx.shape(output)
    end

    test "forward pass with :soft routing" do
      model =
        MoE.build(
          input_size: @embed,
          hidden_size: 32,
          num_experts: 2,
          routing: :soft
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"moe_input" => Nx.template({@batch, @seq_len, @embed}, :f32)},
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "moe_input" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})
        })

      assert {_b, _s, _o} = Nx.shape(output)
    end
  end
end
