defmodule Edifice.Robotics.ACTTest do
  use ExUnit.Case, async: true

  alias Edifice.Robotics.ACT

  @batch_size 2
  @obs_dim 32
  @action_dim 7
  @chunk_size 10
  @latent_dim 8

  describe "ACT.build/1" do
    test "returns encoder and decoder tuple" do
      {encoder, decoder} = ACT.build(
        obs_dim: @obs_dim,
        action_dim: @action_dim,
        chunk_size: @chunk_size,
        hidden_dim: 32,
        num_layers: 2,
        latent_dim: @latent_dim
      )

      assert is_struct(encoder, Axon)
      assert is_struct(decoder, Axon)
    end
  end

  describe "ACT.build_encoder/1" do
    test "produces mu and log_var outputs" do
      encoder = ACT.build_encoder(
        obs_dim: @obs_dim,
        action_dim: @action_dim,
        chunk_size: @chunk_size,
        hidden_dim: 32,
        latent_dim: @latent_dim
      )

      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{
            "obs" => Nx.template({@batch_size, @obs_dim}, :f32),
            "actions" => Nx.template({@batch_size, @chunk_size, @action_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      obs = Nx.broadcast(0.5, {@batch_size, @obs_dim})
      actions = Nx.broadcast(0.1, {@batch_size, @chunk_size, @action_dim})

      output = predict_fn.(params, %{"obs" => obs, "actions" => actions})

      assert is_map(output)
      assert Map.has_key?(output, :mu)
      assert Map.has_key?(output, :log_var)
      assert Nx.shape(output.mu) == {@batch_size, @latent_dim}
      assert Nx.shape(output.log_var) == {@batch_size, @latent_dim}
    end
  end

  describe "ACT.build_decoder/1" do
    test "produces action chunk output" do
      decoder = ACT.build_decoder(
        obs_dim: @obs_dim,
        action_dim: @action_dim,
        chunk_size: @chunk_size,
        hidden_dim: 32,
        num_layers: 2,
        latent_dim: @latent_dim
      )

      {init_fn, predict_fn} = Axon.build(decoder, mode: :inference)

      params =
        init_fn.(
          %{
            "obs" => Nx.template({@batch_size, @obs_dim}, :f32),
            "z" => Nx.template({@batch_size, @latent_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      obs = Nx.broadcast(0.5, {@batch_size, @obs_dim})
      z = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      output = predict_fn.(params, %{"obs" => obs, "z" => z})

      # Output should be [batch, chunk_size, action_dim]
      assert Nx.shape(output) == {@batch_size, @chunk_size, @action_dim}
    end
  end

  describe "ACT.reparameterize/3" do
    test "produces z with correct shape" do
      mu = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      key = Nx.Random.key(42)

      {z, _new_key} = ACT.reparameterize(mu, log_var, key)

      assert Nx.shape(z) == {@batch_size, @latent_dim}
    end

    test "respects mu and log_var distributions" do
      # Large positive mu should shift samples
      mu = Nx.broadcast(10.0, {100, @latent_dim})
      log_var = Nx.broadcast(-10.0, {100, @latent_dim})  # Very small variance
      key = Nx.Random.key(42)

      {z, _} = ACT.reparameterize(mu, log_var, key)

      # Mean of z should be close to mu when variance is small
      mean_z = z |> Nx.mean() |> Nx.to_number()
      assert_in_delta mean_z, 10.0, 0.5
    end
  end

  describe "ACT.act_loss/5" do
    test "combines MSE and KL loss" do
      pred = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})
      target = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})
      mu = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      loss = ACT.act_loss(pred, target, mu, log_var)

      assert Nx.shape(loss) == {}
      # With perfect reconstruction and standard normal posterior, loss is just KL
      # KL(N(0,1) || N(0,1)) = 0, so loss should be close to 0
      loss_val = Nx.to_number(loss)
      assert loss_val >= 0
    end

    test "MSE increases with reconstruction error" do
      pred = Nx.broadcast(1.0, {@batch_size, @chunk_size, @action_dim})
      target = Nx.broadcast(0.0, {@batch_size, @chunk_size, @action_dim})
      mu = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      loss = ACT.act_loss(pred, target, mu, log_var)
      loss_val = Nx.to_number(loss)

      # MSE of 1.0 = (1-0)^2 = 1.0
      assert loss_val >= 1.0
    end

    test "beta parameter scales KL term" do
      pred = Nx.broadcast(0.0, {@batch_size, @chunk_size, @action_dim})
      target = Nx.broadcast(0.0, {@batch_size, @chunk_size, @action_dim})
      # Large mu gives large KL
      mu = Nx.broadcast(5.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      loss_beta_1 = ACT.act_loss(pred, target, mu, log_var, beta: 1.0) |> Nx.to_number()
      loss_beta_0 = ACT.act_loss(pred, target, mu, log_var, beta: 0.0) |> Nx.to_number()

      # With beta=0, only MSE (which is 0), so loss should be ~0
      assert loss_beta_0 < 0.1
      # With beta=1, KL is included
      assert loss_beta_1 > loss_beta_0
    end
  end

  describe "ACT.output_size/1" do
    test "returns action_dim" do
      assert ACT.output_size(action_dim: 7) == 7
      assert ACT.output_size(action_dim: 14) == 14
    end
  end
end
