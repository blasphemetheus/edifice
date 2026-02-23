defmodule Edifice.RL.PolicyValueTest do
  use ExUnit.Case, async: true

  alias Edifice.RL.PolicyValue

  @batch 4
  @input_size 32
  @action_size 4
  @hidden_size 64

  @base_opts [
    input_size: @input_size,
    action_size: @action_size,
    hidden_size: @hidden_size
  ]

  defp random_obs do
    key = Nx.Random.key(42)
    {obs, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    obs
  end

  defp template,
    do: %{"observation" => Nx.template({@batch, @input_size}, :f32)}

  describe "build/1 with discrete actions" do
    test "builds an Axon model" do
      model = PolicyValue.build(@base_opts ++ [action_type: :discrete])
      assert %Axon{} = model
    end

    test "produces policy and value outputs" do
      model = PolicyValue.build(@base_opts ++ [action_type: :discrete])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"observation" => random_obs()})

      assert %{policy: policy, value: value} = output
      assert Nx.shape(policy) == {@batch, @action_size}
      assert Nx.shape(value) == {@batch}
    end

    test "discrete policy sums to approximately 1" do
      model = PolicyValue.build(@base_opts ++ [action_type: :discrete])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      %{policy: policy} = predict_fn.(params, %{"observation" => random_obs()})

      # Each row should sum to ~1 (softmax)
      row_sums = Nx.sum(policy, axes: [1])
      sums = Nx.to_flat_list(row_sums)

      for sum <- sums do
        assert_in_delta sum, 1.0, 0.01
      end
    end

    test "all policy values are non-negative" do
      model = PolicyValue.build(@base_opts ++ [action_type: :discrete])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      %{policy: policy} = predict_fn.(params, %{"observation" => random_obs()})

      assert Nx.all(Nx.greater_equal(policy, 0.0)) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with continuous actions" do
    test "produces policy in [-1, 1] range" do
      model = PolicyValue.build(@base_opts ++ [action_type: :continuous])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      %{policy: policy} = predict_fn.(params, %{"observation" => random_obs()})

      assert Nx.shape(policy) == {@batch, @action_size}
      assert Nx.all(Nx.greater_equal(policy, -1.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(policy, 1.0)) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns action_size" do
      assert PolicyValue.output_size(@base_opts) == @action_size
    end
  end
end
