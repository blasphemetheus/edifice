defmodule Edifice.Meta.ReMoETest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.ReMoE

  @input_size 32
  @batch 2
  @seq_len 4
  @num_experts 4
  @hidden_size 64

  @base_opts [
    input_size: @input_size,
    hidden_size: @hidden_size,
    num_experts: @num_experts,
    target_active: 2,
    dropout: 0.0
  ]

  defp build_and_run(opts \\ []) do
    merged = Keyword.merge(@base_opts, opts)
    model = ReMoE.build(merged)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    params =
      init_fn.(
        Nx.template({@batch, @seq_len, @input_size}, :f32),
        Axon.ModelState.empty()
      )

    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_size})
    output = predict_fn.(params, input)
    {output, params, predict_fn}
  end

  describe "build/1" do
    test "returns an Axon model" do
      model = ReMoE.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      {output, _params, _predict_fn} = build_and_run()
      assert Nx.shape(output) == {@batch, @seq_len, @input_size}
    end

    test "output is finite" do
      {output, _params, _predict_fn} = build_and_run()
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with custom output_size" do
      {output, _params, _predict_fn} = build_and_run(output_size: 16)
      assert Nx.shape(output) == {@batch, @seq_len, 16}
    end

    test "works with 2 experts" do
      {output, _params, _predict_fn} = build_and_run(num_experts: 2)
      assert Nx.shape(output) == {@batch, @seq_len, @input_size}
    end

    test "works with 8 experts" do
      {output, _params, _predict_fn} = build_and_run(num_experts: 8)
      assert Nx.shape(output) == {@batch, @seq_len, @input_size}
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      model = Edifice.build(:remoe, @base_opts)
      assert %Axon{} = model
    end
  end

  describe "sparsity_loss/1" do
    test "returns scalar" do
      router_weights = Nx.tensor([[[0.5, 0.0, 0.3, 0.0], [0.0, 0.8, 0.0, 0.1]]])
      loss = ReMoE.sparsity_loss(router_weights)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0.0
    end

    test "zero for all-zero routing" do
      router_weights = Nx.broadcast(0.0, {2, 3, 4})
      loss = ReMoE.sparsity_loss(router_weights)
      assert abs(Nx.to_number(loss)) < 1.0e-6
    end
  end

  describe "balanced_sparsity_loss/2" do
    test "returns scalar" do
      router_weights = Nx.tensor([[[0.5, 0.0, 0.3, 0.0], [0.0, 0.8, 0.0, 0.1]]])
      loss = ReMoE.balanced_sparsity_loss(router_weights, 2)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0.0
    end
  end

  describe "update_lambda/3" do
    test "increases lambda when too dense" do
      # All experts active (no sparsity) -> should increase lambda
      router_weights = Nx.broadcast(1.0, {2, 4, 4})
      new_lambda = ReMoE.update_lambda(1.0e-4, router_weights, target_active: 2, alpha: 1.5)
      assert new_lambda > 1.0e-4
    end

    test "decreases lambda when too sparse" do
      # Almost all zeros (too sparse) -> should decrease lambda
      router_weights = Nx.broadcast(0.0, {2, 4, 4})
      new_lambda = ReMoE.update_lambda(1.0e-4, router_weights, target_active: 2, alpha: 1.5)
      assert new_lambda < 1.0e-4
    end
  end

  describe "output_size/1" do
    test "returns input_size by default" do
      assert ReMoE.output_size(input_size: 128) == 128
    end

    test "returns custom output_size" do
      assert ReMoE.output_size(input_size: 128, output_size: 64) == 64
    end
  end
end
