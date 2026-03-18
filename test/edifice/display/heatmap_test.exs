defmodule Edifice.Display.HeatmapTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO
  import Nx.Defn

  alias Edifice.Display.Heatmap

  setup do
    previous_level = Logger.level()
    Logger.configure(level: :debug)
    on_exit(fn -> Logger.configure(level: previous_level) end)
  end

  @params %{
    "dense_0" => %{
      "kernel" => Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
      "bias" => Nx.tensor([0.1, 0.2, 0.3])
    },
    "dense_1" => %{
      "kernel" => Nx.tensor([[0.5, -0.5], [1.0, -1.0], [0.0, 0.0]]),
      "bias" => Nx.tensor([0.0, 0.0])
    }
  }

  describe "weights/3" do
    test "displays heatmap for a specific parameter" do
      output =
        capture_io(fn ->
          Heatmap.weights(@params, "dense_0.kernel")
        end)

      assert output =~ "dense_0.kernel"
      assert output =~ "{2, 3}"
      assert output =~ "f32"
    end

    test "shows error for missing parameter" do
      output =
        capture_io(fn ->
          Heatmap.weights(@params, "nonexistent.param")
        end)

      assert output =~ "not found"
      assert output =~ "Available:"
    end

    test "accepts custom label" do
      output =
        capture_io(fn ->
          Heatmap.weights(@params, "dense_0.kernel", label: "First Layer")
        end)

      assert output =~ "First Layer"
    end
  end

  describe "all_weights/2" do
    test "displays all 2D+ weight tensors" do
      output =
        capture_io(fn ->
          Heatmap.all_weights(@params)
        end)

      # Should show both kernels (2D) but not biases (1D, min_rank=2)
      assert output =~ "dense_0.kernel"
      assert output =~ "dense_1.kernel"
      refute output =~ "dense_0.bias"
    end

    test "filters by pattern" do
      output =
        capture_io(fn ->
          Heatmap.all_weights(@params, pattern: "dense_0")
        end)

      assert output =~ "dense_0.kernel"
      refute output =~ "dense_1.kernel"
    end

    test "respects min_rank option" do
      output =
        capture_io(fn ->
          Heatmap.all_weights(@params, min_rank: 1)
        end)

      # With min_rank=1, biases should also appear
      assert output =~ "dense_0.bias"
    end
  end

  describe "gradients/2" do
    test "displays gradient magnitudes" do
      grads = %{
        "dense_0" => %{
          "kernel" => Nx.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]]),
          "bias" => Nx.tensor([0.01, 0.02, 0.03])
        }
      }

      output =
        capture_io(fn ->
          Heatmap.gradients(grads)
        end)

      assert output =~ "Gradient magnitudes"
      assert output =~ "dense_0.kernel"
      assert output =~ "norm="
    end
  end

  describe "capture_grad/2 in defn" do
    defn capture_test(x) do
      Heatmap.capture_grad(x, label: "test_grad")
    end

    test "captures gradient heatmap and returns tensor unchanged" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

      output =
        capture_io(fn ->
          result = capture_test(x)
          assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert output =~ "test_grad"
    end
  end

  describe "norm_summary/2" do
    test "displays per-tensor norm summary" do
      output =
        capture_io(fn ->
          Heatmap.norm_summary(@params, label: "weight norms")
        end)

      assert output =~ "weight norms"
      assert output =~ "dense_0.kernel"
      assert output =~ "dense_1.kernel"
    end
  end

  describe "helpers" do
    test "handles Axon.ModelState wrapper" do
      model_state = Axon.ModelState.new(@params)

      output =
        capture_io(fn ->
          Heatmap.weights(model_state, "dense_0.kernel")
        end)

      assert output =~ "dense_0.kernel"
    end
  end
end
