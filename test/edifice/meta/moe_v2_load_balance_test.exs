defmodule Edifice.Meta.MoEv2LoadBalanceTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.MoEv2

  @input_size 32
  @batch_size 2
  @seq_len 4

  @base_opts [
    input_size: @input_size,
    hidden_size: 64,
    output_size: @input_size,
    num_shared_experts: 1,
    num_routed_experts: 4,
    tokens_per_expert: 4,
    dropout: 0.0
  ]

  defp template,
    do: %{"moe_input" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)}

  defp constant_input,
    do: %{"moe_input" => Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})}

  describe "build/1 with load_balance: :bias" do
    test "builds an Axon model" do
      model = MoEv2.build(@base_opts ++ [load_balance: :bias])
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = MoEv2.build(@base_opts ++ [load_balance: :bias])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, constant_input())
      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "output contains finite values" do
      model = MoEv2.build(@base_opts ++ [load_balance: :bias])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, constant_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with load_balance: :none" do
    test "builds and runs without bias" do
      model = MoEv2.build(@base_opts ++ [load_balance: :none])
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, constant_input())
      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  describe "build/1 with non-standard expert count" do
    test "num_routed: 3 builds and runs correctly" do
      model =
        MoEv2.build(
          @base_opts
          |> Keyword.put(:num_routed_experts, 3)
          |> Keyword.put(:load_balance, :none)
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, constant_input())
      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "num_routed: 6 builds and runs correctly" do
      model =
        MoEv2.build(
          @base_opts
          |> Keyword.put(:num_routed_experts, 6)
          |> Keyword.put(:load_balance, :none)
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, constant_input())
      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "compute_utilization/2" do
    test "returns tensor of shape [num_experts]" do
      router_logits = Nx.broadcast(0.5, {@batch_size, @seq_len, 4})
      utilization = MoEv2.compute_utilization(router_logits, 4)
      assert Nx.shape(utilization) == {4}
    end

    test "returns values in valid range" do
      key = Nx.Random.key(42)
      {router_logits, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, 4})
      utilization = MoEv2.compute_utilization(router_logits, 4)

      # All values should be non-negative
      assert Nx.all(Nx.greater_equal(utilization, 0.0)) |> Nx.to_number() == 1
    end
  end

  describe "update_load_balance_bias/3" do
    test "adjusts bias toward uniform utilization" do
      # Create mock params with bias
      params = %{
        "moe_v2_load_balance_bias" => %{
          "bias" => Nx.broadcast(0.0, {4})
        }
      }

      # Imbalanced utilization: expert 0 overused, expert 3 underused
      utilization = Nx.tensor([0.5, 0.25, 0.25, 0.0])

      updated = MoEv2.update_load_balance_bias(params, utilization, lr: 0.1)
      updated_bias = updated["moe_v2_load_balance_bias"]["bias"]

      # Expert 0 bias should decrease (overused), expert 3 should increase (underused)
      values = Nx.to_flat_list(updated_bias)
      assert Enum.at(values, 0) < 0
      assert Enum.at(values, 3) > 0
    end
  end
end
