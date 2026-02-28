defmodule Edifice.Pretrained.TransformTest do
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.Transform

  describe "transpose_linear/1" do
    test "transposes rank-2 tensor from [out, in] to [in, out]" do
      tensor = Nx.iota({4, 3}, type: :f32)
      result = Transform.transpose_linear(tensor)

      assert Nx.shape(result) == {3, 4}
      # First element stays same, but layout changes
      assert Nx.to_number(result[0][0]) == Nx.to_number(tensor[0][0])
    end

    test "returns rank-1 tensor unchanged" do
      tensor = Nx.iota({5}, type: :f32)
      result = Transform.transpose_linear(tensor)
      assert result == tensor
    end

    test "returns rank-3 tensor unchanged" do
      tensor = Nx.iota({2, 3, 4}, type: :f32)
      result = Transform.transpose_linear(tensor)
      assert result == tensor
    end
  end

  describe "permute_conv2d/1" do
    test "returns tensor unchanged (identity for now)" do
      tensor = Nx.iota({8, 3, 3, 3}, type: :f32)
      assert Transform.permute_conv2d(tensor) == tensor
    end
  end

  describe "cast/2" do
    test "casts f32 to bf16" do
      tensor = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      result = Transform.cast(tensor, :bf16)
      assert Nx.type(result) == {:bf, 16}
    end

    test "casts bf16 to f32" do
      tensor = Nx.tensor([1.0, 2.0], type: :bf16)
      result = Transform.cast(tensor, :f32)
      assert Nx.type(result) == {:f, 32}
    end
  end

  describe "nest_params/1" do
    test "converts flat dot-separated keys to nested map" do
      flat = %{
        "block.norm.scale" => 1,
        "block.norm.bias" => 2,
        "block.dense.kernel" => 3,
        "embed" => 4
      }

      result = Transform.nest_params(flat)

      assert result == %{
               "block" => %{
                 "norm" => %{"scale" => 1, "bias" => 2},
                 "dense" => %{"kernel" => 3}
               },
               "embed" => 4
             }
    end

    test "handles single-level keys" do
      flat = %{"alpha" => :a, "beta" => :b}
      assert Transform.nest_params(flat) == %{"alpha" => :a, "beta" => :b}
    end

    test "handles empty map" do
      assert Transform.nest_params(%{}) == %{}
    end
  end

  describe "flatten_params/1" do
    test "converts nested map to flat dot-separated keys" do
      nested = %{
        "block" => %{
          "norm" => %{"scale" => Nx.tensor(1.0), "bias" => Nx.tensor(2.0)}
        }
      }

      result = Transform.flatten_params(nested)
      assert Map.has_key?(result, "block.norm.scale")
      assert Map.has_key?(result, "block.norm.bias")
      assert map_size(result) == 2
    end

    test "handles Axon.ModelState input" do
      data = %{"layer" => %{"kernel" => Nx.tensor([1.0, 2.0])}}
      model_state = Axon.ModelState.new(data)

      result = Transform.flatten_params(model_state)
      assert Map.has_key?(result, "layer.kernel")
    end
  end

  describe "nest_params/1 and flatten_params/1 round-trip" do
    test "round-trips correctly with tensor values" do
      original = %{
        "encoder.block_0.attention.kernel" => Nx.tensor([1.0]),
        "encoder.block_0.attention.bias" => Nx.tensor([2.0]),
        "decoder.output.kernel" => Nx.tensor([3.0])
      }

      round_tripped =
        original
        |> Transform.nest_params()
        |> Transform.flatten_params()

      assert Map.keys(round_tripped) |> Enum.sort() == Map.keys(original) |> Enum.sort()

      for {key, tensor} <- original do
        assert Nx.equal(round_tripped[key], tensor) |> Nx.all() |> Nx.to_number() == 1
      end
    end
  end

  describe "apply_transform/3" do
    test "applies first matching pattern" do
      patterns = [
        {~r/\.kernel$/, &Nx.negate/1},
        {~r/\.scale$/, fn t -> Nx.multiply(t, 2) end}
      ]

      tensor = Nx.tensor([1.0, 2.0, 3.0])

      result = Transform.apply_transform("dense.kernel", patterns, tensor)
      expected = Nx.tensor([-1.0, -2.0, -3.0])
      assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "returns tensor unchanged when no pattern matches" do
      patterns = [{~r/\.kernel$/, &Nx.negate/1}]
      tensor = Nx.tensor([1.0, 2.0])

      result = Transform.apply_transform("dense.bias", patterns, tensor)
      assert result == tensor
    end

    test "uses only the first match when multiple patterns could match" do
      patterns = [
        {~r/kernel/, &Nx.negate/1},
        {~r/dense/, fn t -> Nx.multiply(t, 10) end}
      ]

      tensor = Nx.tensor([1.0])

      # "dense.kernel" matches both, but first match wins
      result = Transform.apply_transform("dense.kernel", patterns, tensor)
      expected = Nx.tensor([-1.0])
      assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles empty pattern list" do
      tensor = Nx.tensor([1.0])
      result = Transform.apply_transform("anything", [], tensor)
      assert result == tensor
    end
  end
end
