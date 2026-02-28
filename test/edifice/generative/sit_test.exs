defmodule Edifice.Generative.SiTTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.SiT

  @batch 2
  @input_dim 32
  @hidden_size 64
  @num_heads 4
  @num_layers 2

  @base_opts [
    input_dim: @input_dim,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: @num_heads
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, key} = Nx.Random.normal(key, shape: {@batch, @input_dim})
    {timestep, _key} = Nx.Random.uniform(key, shape: {@batch})
    %{"noisy_input" => input, "timestep" => timestep}
  end

  defp build_and_predict(opts) do
    model = SiT.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    params =
      init_fn.(
        %{
          "noisy_input" => Nx.template({@batch, @input_dim}, :f32),
          "timestep" => Nx.template({@batch}, :f32)
        },
        Axon.ModelState.empty()
      )

    predict_fn.(params, random_input())
  end

  describe "build/1" do
    test "returns an Axon model" do
      model = SiT.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      output = build_and_predict(@base_opts)
      assert Nx.shape(output) == {@batch, @input_dim}
    end

    test "output contains finite values" do
      output = build_and_predict(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with class conditioning" do
      opts = Keyword.put(@base_opts, :num_classes, 5)
      model = SiT.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "noisy_input" => Nx.template({@batch, @input_dim}, :f32),
            "timestep" => Nx.template({@batch}, :f32),
            "class_label" => Nx.template({@batch}, :f32)
          },
          Axon.ModelState.empty()
        )

      input = Map.put(random_input(), "class_label", Nx.tensor([1, 3]))
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @input_dim}
    end
  end

  describe "sit_loss/2" do
    test "computes MSE loss between velocities" do
      key = Nx.Random.key(123)
      {pred, key} = Nx.Random.normal(key, shape: {@batch, @input_dim})
      {target, _key} = Nx.Random.normal(key, shape: {@batch, @input_dim})

      loss = SiT.sit_loss(pred, target)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "loss is zero when predictions match targets" do
      pred = Nx.broadcast(1.0, {@batch, @input_dim})
      loss = SiT.sit_loss(pred, pred)
      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-6
    end
  end

  describe "sample_interpolant_time/2" do
    test "returns values in [0, 1]" do
      {t, _key} = SiT.sample_interpolant_time(8, key: Nx.Random.key(42))
      assert Nx.shape(t) == {8}
      assert Nx.to_number(Nx.reduce_min(t)) >= 0.0
      assert Nx.to_number(Nx.reduce_max(t)) <= 1.0
    end
  end

  describe "linear_interpolant/3" do
    test "returns data at t=0 and noise at t=1" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      noise = Nx.tensor([[4.0, 5.0, 6.0]])

      at_zero = SiT.linear_interpolant(x, noise, Nx.tensor([0.0]))
      assert_all_close(at_zero, x)

      at_one = SiT.linear_interpolant(x, noise, Nx.tensor([1.0]))
      assert_all_close(at_one, noise)
    end

    test "midpoint is average" do
      x = Nx.tensor([[0.0, 0.0]])
      noise = Nx.tensor([[2.0, 4.0]])

      mid = SiT.linear_interpolant(x, noise, Nx.tensor([0.5]))
      assert_all_close(mid, Nx.tensor([[1.0, 2.0]]))
    end
  end

  describe "linear_velocity/2" do
    test "computes noise - data" do
      x = Nx.tensor([[1.0, 2.0]])
      noise = Nx.tensor([[3.0, 5.0]])

      vel = SiT.linear_velocity(x, noise)
      assert_all_close(vel, Nx.tensor([[2.0, 3.0]]))
    end
  end

  describe "cosine_interpolant/3" do
    test "returns data at t=0 and noise at t=1" do
      x = Nx.tensor([[1.0, 2.0]])
      noise = Nx.tensor([[3.0, 4.0]])

      at_zero = SiT.cosine_interpolant(x, noise, Nx.tensor([0.0]))
      assert_all_close(at_zero, x, atol: 1.0e-4)

      at_one = SiT.cosine_interpolant(x, noise, Nx.tensor([1.0]))
      assert_all_close(at_one, noise, atol: 1.0e-4)
    end
  end

  describe "output_size/1" do
    test "returns input_dim" do
      assert SiT.output_size(@base_opts) == @input_dim
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all(Nx.less_equal(Nx.abs(Nx.subtract(a, b)), atol)) |> Nx.to_number() == 1
  end
end
