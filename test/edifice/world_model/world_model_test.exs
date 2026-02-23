defmodule Edifice.WorldModel.WorldModelTest do
  use ExUnit.Case, async: true

  alias Edifice.WorldModel.WorldModel

  @batch 4
  @obs_size 32
  @action_size 4
  @latent_size 16
  @hidden_size 32

  @base_opts [
    obs_size: @obs_size,
    action_size: @action_size,
    latent_size: @latent_size,
    hidden_size: @hidden_size
  ]

  defp random_obs do
    key = Nx.Random.key(42)
    {obs, _key} = Nx.Random.uniform(key, shape: {@batch, @obs_size})
    obs
  end

  defp random_state_action do
    key = Nx.Random.key(99)
    {sa, _key} = Nx.Random.uniform(key, shape: {@batch, @latent_size + @action_size})
    sa
  end

  defp random_latent do
    key = Nx.Random.key(77)
    {z, _key} = Nx.Random.uniform(key, shape: {@batch, @latent_size})
    z
  end

  describe "build/1 with MLP dynamics" do
    test "returns {encoder, dynamics, reward_head} tuple" do
      {encoder, dynamics, reward_head} = WorldModel.build(@base_opts ++ [dynamics: :mlp])
      assert %Axon{} = encoder
      assert %Axon{} = dynamics
      assert %Axon{} = reward_head
    end

    test "encoder maps obs to latent" do
      {encoder, _dyn, _rh} = WorldModel.build(@base_opts ++ [dynamics: :mlp])
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)
      params = init_fn.(%{"observation" => Nx.template({@batch, @obs_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"observation" => random_obs()})
      assert Nx.shape(output) == {@batch, @latent_size}
    end

    test "dynamics maps (z, action) to next_z" do
      {_enc, dynamics, _rh} = WorldModel.build(@base_opts ++ [dynamics: :mlp])
      {init_fn, predict_fn} = Axon.build(dynamics, mode: :inference)
      params = init_fn.(%{"state_action" => Nx.template({@batch, @latent_size + @action_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_action" => random_state_action()})
      assert Nx.shape(output) == {@batch, @latent_size}
    end

    test "reward head maps z to scalar" do
      {_enc, _dyn, reward_head} = WorldModel.build(@base_opts ++ [dynamics: :mlp])
      {init_fn, predict_fn} = Axon.build(reward_head, mode: :inference)
      params = init_fn.(%{"latent_state" => Nx.template({@batch, @latent_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"latent_state" => random_latent()})
      assert Nx.shape(output) == {@batch}
    end
  end

  describe "build/1 with Neural ODE dynamics" do
    test "dynamics produces correct shape" do
      {_enc, dynamics, _rh} = WorldModel.build(@base_opts ++ [dynamics: :neural_ode])
      {init_fn, predict_fn} = Axon.build(dynamics, mode: :inference)
      params = init_fn.(%{"state_action" => Nx.template({@batch, @latent_size + @action_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_action" => random_state_action()})
      assert Nx.shape(output) == {@batch, @latent_size}
    end

    test "output contains finite values" do
      {_enc, dynamics, _rh} = WorldModel.build(@base_opts ++ [dynamics: :neural_ode])
      {init_fn, predict_fn} = Axon.build(dynamics, mode: :inference)
      params = init_fn.(%{"state_action" => Nx.template({@batch, @latent_size + @action_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_action" => random_state_action()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with GRU dynamics" do
    test "dynamics produces correct shape" do
      {_enc, dynamics, _rh} = WorldModel.build(@base_opts ++ [dynamics: :gru])
      {init_fn, predict_fn} = Axon.build(dynamics, mode: :inference)
      params = init_fn.(%{"state_action" => Nx.template({@batch, @latent_size + @action_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_action" => random_state_action()})
      assert Nx.shape(output) == {@batch, @latent_size}
    end

    test "output contains finite values" do
      {_enc, dynamics, _rh} = WorldModel.build(@base_opts ++ [dynamics: :gru])
      {init_fn, predict_fn} = Axon.build(dynamics, mode: :inference)
      params = init_fn.(%{"state_action" => Nx.template({@batch, @latent_size + @action_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_action" => random_state_action()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with decoder" do
    test "returns 4-tuple when use_decoder: true" do
      {encoder, dynamics, reward_head, decoder} =
        WorldModel.build(@base_opts ++ [dynamics: :mlp, use_decoder: true])

      assert %Axon{} = encoder
      assert %Axon{} = dynamics
      assert %Axon{} = reward_head
      assert %Axon{} = decoder
    end

    test "decoder maps latent to obs" do
      {_enc, _dyn, _rh, decoder} =
        WorldModel.build(@base_opts ++ [dynamics: :mlp, use_decoder: true])

      {init_fn, predict_fn} = Axon.build(decoder, mode: :inference)
      params = init_fn.(%{"latent_state" => Nx.template({@batch, @latent_size}, :f32)}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"latent_state" => random_latent()})
      assert Nx.shape(output) == {@batch, @obs_size}
    end
  end

  describe "output_size/1" do
    test "returns latent_size" do
      assert WorldModel.output_size(@base_opts) == @latent_size
    end
  end
end
