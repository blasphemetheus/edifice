defmodule Edifice.RL.DecisionTransformerTest do
  use ExUnit.Case, async: true

  alias Edifice.RL.DecisionTransformer

  @batch 2
  @state_dim 16
  @action_dim 8
  @hidden_size 32
  @num_heads 2
  @num_layers 2
  @context_len 4

  @base_opts [
    state_dim: @state_dim,
    action_dim: @action_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    context_len: @context_len,
    max_timestep: 100,
    dropout: 0.0
  ]

  defp random_inputs do
    key = Nx.Random.key(42)
    {returns, key} = Nx.Random.uniform(key, shape: {@batch, @context_len})
    {states, key} = Nx.Random.uniform(key, shape: {@batch, @context_len, @state_dim})
    {actions, key} = Nx.Random.uniform(key, shape: {@batch, @context_len, @action_dim})
    {timesteps_f, _key} = Nx.Random.uniform(key, 0, 100, shape: {@batch, @context_len})
    timesteps = Nx.as_type(timesteps_f, :s64)

    %{
      "returns" => returns,
      "states" => states,
      "actions" => actions,
      "timesteps" => timesteps
    }
  end

  defp template do
    %{
      "returns" => Nx.template({@batch, @context_len}, :f32),
      "states" => Nx.template({@batch, @context_len, @state_dim}, :f32),
      "actions" => Nx.template({@batch, @context_len, @action_dim}, :f32),
      "timesteps" => Nx.template({@batch, @context_len}, :s64)
    }
  end

  describe "build/1" do
    test "returns an Axon model" do
      model = DecisionTransformer.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = DecisionTransformer.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, random_inputs())

      assert Nx.shape(output) == {@batch, @context_len, @action_dim}
    end

    test "output is finite" do
      model = DecisionTransformer.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, random_inputs())

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with different num_layers" do
      model = DecisionTransformer.build(Keyword.put(@base_opts, :num_layers, 1))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, random_inputs())

      assert Nx.shape(output) == {@batch, @context_len, @action_dim}
    end

    test "works with different num_heads" do
      opts = @base_opts |> Keyword.put(:num_heads, 4) |> Keyword.put(:hidden_size, 32)
      model = DecisionTransformer.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, random_inputs())

      assert Nx.shape(output) == {@batch, @context_len, @action_dim}
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      model = Edifice.build(:decision_transformer, @base_opts)
      assert %Axon{} = model
    end
  end

  describe "output_size/1" do
    test "returns action_dim" do
      assert DecisionTransformer.output_size(@base_opts) == @action_dim
    end
  end
end
