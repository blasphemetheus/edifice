defmodule Edifice.SSM.MambaSSDCoverageTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  @moduletag timeout: 180_000

  alias Edifice.SSM.MambaSSD

  @embed 16

  describe "build/1" do
    test "builds default (inference) model" do
      model = MambaSSD.build(embed_dim: @embed, hidden_size: @embed, seq_len: 8, num_layers: 1)
      assert %Axon{} = model
    end

    test "builds training mode model" do
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: 8,
          num_layers: 1,
          training_mode: true,
          chunk_size: 4
        )

      assert %Axon{} = model
    end

    test "builds with custom chunk_size" do
      model =
        MambaSSD.build(
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: 16,
          num_layers: 1,
          chunk_size: 8
        )

      assert %Axon{} = model
    end
  end

  describe "output_size/1" do
    test "delegates to Common" do
      size = MambaSSD.output_size(hidden_size: 64)
      assert is_integer(size)
      assert size > 0
    end
  end

  describe "recommended_defaults/0" do
    test "includes SSD-specific options" do
      defaults = MambaSSD.recommended_defaults()
      assert Keyword.has_key?(defaults, :chunk_size)
      assert defaults[:training_mode] == false
    end
  end

  describe "training_defaults/0" do
    test "uses training mode and larger chunks" do
      defaults = MambaSSD.training_defaults()
      assert defaults[:training_mode] == true
      assert defaults[:chunk_size] > 16
    end
  end
end
