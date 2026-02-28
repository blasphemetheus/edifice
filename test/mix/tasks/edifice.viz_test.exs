defmodule Mix.Tasks.Edifice.VizTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Edifice.Viz

  import ExUnit.CaptureIO

  # MLP requires input_size, so we always pass it
  @mlp_args ["mlp", "--input_size", "16"]

  describe "run/1" do
    test "renders table for mlp (default format)" do
      output = capture_io(fn -> Viz.run(@mlp_args) end)

      assert output =~ "Building :mlp"
      assert output =~ "Layer"
      assert output =~ "Total Parameters:"
    end

    test "renders tree format" do
      output = capture_io(fn -> Viz.run(@mlp_args ++ ["--format", "tree"]) end)

      assert output =~ "(dense)"
      assert output =~ "(input)"
    end

    test "renders mermaid format" do
      output =
        capture_io(fn -> Viz.run(@mlp_args ++ ["--format", "mermaid"]) end)

      assert output =~ "graph TD;"
      assert output =~ "-->"
    end

    test "accepts -f alias for format" do
      output = capture_io(fn -> Viz.run(@mlp_args ++ ["-f", "tree"]) end)

      assert output =~ "(dense)"
    end

    test "passes build options to architecture" do
      output =
        capture_io(fn ->
          Viz.run(["mlp", "-f", "tree", "--input_size", "32"])
        end)

      assert output =~ "(dense)"
    end

    test "raises for unknown architecture" do
      assert_raise Mix.Error, ~r/Unknown architecture/, fn ->
        Viz.run(["nonexistent_arch_xyz"])
      end
    end

    test "raises for invalid format" do
      assert_raise Mix.Error, ~r/Invalid format/, fn ->
        Viz.run(@mlp_args ++ ["--format", "png"])
      end
    end

    test "raises with no arguments" do
      assert_raise Mix.Error, ~r/Usage/, fn ->
        Viz.run([])
      end
    end

    test "handles --component for tuple-returning model" do
      # VAE returns {encoder, decoder}
      output =
        capture_io(fn ->
          Viz.run(["vae", "-f", "tree", "-c", "encoder", "--input_size", "16"])
        end)

      assert output =~ "(input)"
      # Should only show one component, not both
      refute output =~ "=== Decoder ==="
    end

    test "handles --component all for tuple-returning model" do
      output =
        capture_io(fn ->
          Viz.run(["vae", "-f", "tree", "-c", "all", "--input_size", "16"])
        end)

      assert output =~ "=== Encoder ==="
      assert output =~ "=== Decoder ==="
    end
  end
end
