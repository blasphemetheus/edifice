defmodule Edifice.ShardingTest do
  @moduledoc """
  Tests for multi-device sharding.

  Run with:
    XLA_FLAGS="--xla_force_host_platform_device_count=4" EDIFICE_LOCAL_NX=1 \\
      mix test test/edifice/sharding_test.exs

  The XLA_FLAGS env var creates virtual host devices for testing
  without requiring multiple GPUs.
  """
  use ExUnit.Case, async: false

  alias Edifice.Sharding

  @num_devices 2

  # Skip all tests if host doesn't have enough devices
  setup_all do
    try do
      count = Sharding.device_count(:host)

      if count < @num_devices do
        IO.puts(
          "[ShardingTest] Skipping: host has #{count} devices, need #{@num_devices}. " <>
            "Set XLA_FLAGS=\"--xla_force_host_platform_device_count=#{@num_devices}\""
        )

        :skip
      else
        :ok
      end
    rescue
      _ -> :skip
    end
  end

  # ============================================================================
  # Mesh and device count
  # ============================================================================

  describe "mesh/1" do
    test "creates mesh with explicit num_devices" do
      mesh = Sharding.mesh(num_devices: 4, name: "test")
      assert mesh.name == "test"
      assert mesh.shape == {4}
    end

    test "creates mesh with explicit shape" do
      mesh = Sharding.mesh(shape: {2, 2}, name: "grid")
      assert mesh.shape == {2, 2}
    end
  end

  describe "device_count/1" do
    test "returns positive count for host client" do
      count = Sharding.device_count(:host)
      assert count >= 1
    end
  end

  # ============================================================================
  # Data parallel: tensor input
  # ============================================================================

  describe "data_parallel with tensor input" do
    test "splits batch and gathers output" do
      # Simple function: double the input
      predict_fn = fn _params, input -> Nx.multiply(input, 2) end
      params = %{}

      dp = Sharding.data_parallel(predict_fn, params, num_devices: @num_devices, client: :host)

      input = Nx.iota({4, 3})
      result = dp.(input)

      expected = Nx.multiply(input, 2)
      assert Nx.shape(result) == Nx.shape(expected)

      assert Nx.all_close(
               Nx.backend_transfer(result, Nx.BinaryBackend),
               expected
             )
             |> Nx.to_number() == 1
    end

    test "preserves output type" do
      predict_fn = fn _params, input -> Nx.as_type(input, :f32) end
      params = %{}

      dp = Sharding.data_parallel(predict_fn, params, num_devices: @num_devices, client: :host)
      result = dp.(Nx.iota({4, 2}))

      assert Nx.type(result) == {:f, 32}
    end

    test "works with linear model" do
      # Simple linear: input @ kernel + bias
      kernel = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      bias = Nx.tensor([0.1, 0.2])

      predict_fn = fn params, input ->
        Nx.add(Nx.dot(input, params["kernel"]), params["bias"])
      end

      params = %{"kernel" => kernel, "bias" => bias}

      # Non-sharded reference
      input = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
      expected = predict_fn.(params, input)

      # Sharded
      dp = Sharding.data_parallel(predict_fn, params, num_devices: @num_devices, client: :host)
      result = dp.(input)

      assert Nx.shape(result) == Nx.shape(expected)

      assert Nx.all_close(
               Nx.backend_transfer(result, Nx.BinaryBackend),
               expected,
               atol: 1.0e-5
             )
             |> Nx.to_number() == 1
    end

    test "raises on indivisible batch size" do
      predict_fn = fn _params, input -> input end
      dp = Sharding.data_parallel(predict_fn, %{}, num_devices: @num_devices, client: :host)

      assert_raise ArgumentError, ~r/not divisible/, fn ->
        dp.(Nx.iota({3, 2}))
      end
    end
  end

  # ============================================================================
  # Data parallel: map input
  # ============================================================================

  describe "data_parallel with map input" do
    test "handles single-key map" do
      predict_fn = fn _params, input ->
        Nx.multiply(input["x"], 3)
      end

      dp = Sharding.data_parallel(predict_fn, %{}, num_devices: @num_devices, client: :host)

      input = %{"x" => Nx.iota({4, 2})}
      result = dp.(input)

      expected = Nx.multiply(Nx.iota({4, 2}), 3)
      assert Nx.shape(result) == Nx.shape(expected)

      assert Nx.all_close(
               Nx.backend_transfer(result, Nx.BinaryBackend),
               expected
             )
             |> Nx.to_number() == 1
    end

    test "handles multi-key map" do
      predict_fn = fn _params, input ->
        Nx.add(input["a"], input["b"])
      end

      dp = Sharding.data_parallel(predict_fn, %{}, num_devices: @num_devices, client: :host)

      input = %{
        "a" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        "b" => Nx.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
      }

      result = dp.(input)

      expected = Nx.add(input["a"], input["b"])
      assert Nx.shape(result) == Nx.shape(expected)

      assert Nx.all_close(
               Nx.backend_transfer(result, Nx.BinaryBackend),
               expected
             )
             |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Report
  # ============================================================================

  describe "report/2" do
    test "prints sharding plan" do
      params = %{
        "layer" => %{
          "kernel" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
          "bias" => Nx.tensor([0.0, 0.0])
        }
      }

      output =
        ExUnit.CaptureIO.capture_io(fn ->
          result = Sharding.report(params, num_devices: @num_devices)
          assert result.total_bytes == 24
          assert result.tensor_count == 2
          assert result.num_devices == @num_devices
        end)

      assert output =~ "[Sharding] Data Parallel Plan"
      assert output =~ "#{@num_devices} devices"
    end
  end
end
