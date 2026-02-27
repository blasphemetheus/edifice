defmodule Edifice.DisplayTest do
  use ExUnit.Case, async: true

  alias Edifice.Display

  describe "as_tree/2" do
    test "renders ASCII tree for a simple model" do
      model = Axon.input("input", shape: {nil, 16}) |> Axon.dense(32)
      output = Display.as_tree(model)

      assert output =~ "Model"
      assert output =~ "└──"
      assert output =~ "(dense)"
      assert output =~ "(input)"
    end

    test "renders tree for MLP" do
      model = Edifice.build(:mlp, input_size: 16, hidden_sizes: [32], output_size: 8)
      output = Display.as_tree(model, name: "MLP")

      assert output =~ "MLP"
      assert output =~ "(dense)"
      assert output =~ "(input)"
    end

    test "accepts custom name option" do
      model = Axon.input("input", shape: {nil, 8}) |> Axon.dense(4)
      output = Display.as_tree(model, name: "CustomName")

      assert String.starts_with?(output, "CustomName")
    end
  end

  describe "as_mermaid/2" do
    test "generates valid Mermaid flowchart text" do
      model = Axon.input("input", shape: {nil, 16}) |> Axon.dense(32)
      output = Display.as_mermaid(model)

      assert output =~ "graph TD;"
      assert output =~ "-->"
      assert output =~ ":input"
      assert output =~ ":dense"
    end

    test "generates Mermaid for MLP" do
      model = Edifice.build(:mlp, input_size: 16, hidden_sizes: [32], output_size: 8)
      output = Display.as_mermaid(model)

      assert output =~ "graph TD;"
      assert output =~ "-->"
    end

    test "supports left_right direction" do
      model = Axon.input("input", shape: {nil, 8}) |> Axon.dense(4)
      output = Display.as_mermaid(model, direction: :left_right)

      assert output =~ "graph LR;"
    end
  end

  describe "as_table/2" do
    test "renders table with header columns" do
      model = Axon.input("input", shape: {nil, 16}) |> Axon.dense(32)
      output = Display.as_table(model)

      assert output =~ "Layer"
      assert output =~ "Output Shape"
      assert output =~ "Parameters"
      assert output =~ "Total Parameters:"
    end

    test "shows parameter counts for dense layers" do
      model = Axon.input("input", shape: {nil, 16}) |> Axon.dense(32)
      output = Display.as_table(model)

      # dense(16, 32) has 16*32 kernel + 32 bias = 544 params
      assert output =~ "544"
      assert output =~ "Total Parameters: 544"
    end

    test "renders table for MLP" do
      model = Edifice.build(:mlp, input_size: 16, hidden_sizes: [32], output_size: 8)
      output = Display.as_table(model, name: "MLP")

      assert output =~ "MLP"
      assert output =~ "Total Parameters:"
    end
  end

  describe "format_build_result/3" do
    test "handles single Axon model" do
      model = Axon.input("input", shape: {nil, 8}) |> Axon.dense(4)
      output = Display.format_build_result(model, :tree)

      assert output =~ "(dense)"
      assert output =~ "(input)"
    end

    test "handles 2-tuple" do
      enc = Axon.input("enc_input", shape: {nil, 8}) |> Axon.dense(4)
      dec = Axon.input("dec_input", shape: {nil, 4}) |> Axon.dense(8)
      output = Display.format_build_result({enc, dec}, :tree)

      assert output =~ "=== Component 1 ==="
      assert output =~ "=== Component 2 ==="
    end

    test "handles 3-tuple" do
      a = Axon.input("a", shape: {nil, 8}) |> Axon.dense(4)
      b = Axon.input("b", shape: {nil, 4}) |> Axon.dense(4)
      c = Axon.input("c", shape: {nil, 4}) |> Axon.dense(2)
      output = Display.format_build_result({a, b, c}, :tree)

      assert output =~ "=== Component 1 ==="
      assert output =~ "=== Component 2 ==="
      assert output =~ "=== Component 3 ==="
    end

    test "works with all three formats" do
      model = Axon.input("input", shape: {nil, 8}) |> Axon.dense(4)

      for fmt <- [:tree, :mermaid, :table] do
        output = Display.format_build_result(model, fmt)
        assert is_binary(output), "Expected string for format #{fmt}"
        assert byte_size(output) > 0, "Expected non-empty string for format #{fmt}"
      end
    end
  end

  describe "build_input_templates/1" do
    test "extracts input shapes" do
      model = Axon.input("features", shape: {nil, 32}) |> Axon.dense(16)
      templates = Display.build_input_templates(model)

      assert is_map(templates)
      assert Map.has_key?(templates, "features")
      assert %Nx.Tensor{} = templates["features"]
      assert Nx.shape(templates["features"]) == {1, 32}
    end

    test "handles multiple inputs" do
      a = Axon.input("a", shape: {nil, 8})
      b = Axon.input("b", shape: {nil, 4})
      model = Axon.concatenate(a, b) |> Axon.dense(16)
      templates = Display.build_input_templates(model)

      assert Map.has_key?(templates, "a")
      assert Map.has_key?(templates, "b")
    end
  end

  describe "integration with Edifice architectures" do
    test "tree format works for mamba" do
      model = Edifice.build(:mamba, embed_dim: 16, hidden_size: 32, num_layers: 1)
      output = Display.as_tree(model)

      assert is_binary(output)
      assert byte_size(output) > 0
    end

    test "all formats work for attention" do
      model = Edifice.build(:attention, embed_dim: 16, num_heads: 2, num_layers: 1)

      for fmt <- [:tree, :table, :mermaid] do
        output = Display.format_build_result(model, fmt)
        assert is_binary(output), "format #{fmt} failed"
      end
    end
  end
end
