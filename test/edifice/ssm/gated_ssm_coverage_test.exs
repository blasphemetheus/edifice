defmodule Edifice.SSM.GatedSSMCoverageTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  @moduletag timeout: 180_000

  alias Edifice.SSM.GatedSSM

  @batch 2
  @embed 16
  @seq_len 8

  describe "build/1" do
    test "builds with matching embed and hidden sizes" do
      model =
        GatedSSM.build(embed_dim: @embed, hidden_size: @embed, seq_len: @seq_len, num_layers: 1)

      assert %Axon{} = model
    end

    test "builds with different embed and hidden sizes (input projection)" do
      model =
        GatedSSM.build(embed_dim: @embed, hidden_size: 32, seq_len: @seq_len, num_layers: 1)

      assert %Axon{} = model
    end

    test "supports dropout" do
      model =
        GatedSSM.build(
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len,
          num_layers: 2,
          dropout: 0.2
        )

      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model =
        GatedSSM.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          expand_factor: 2,
          conv_size: 2,
          num_layers: 1,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @seq_len, @embed}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  describe "build_checkpointed/1" do
    test "builds a checkpointed model" do
      model =
        GatedSSM.build_checkpointed(
          embed_dim: @embed,
          hidden_size: @embed,
          seq_len: @seq_len,
          num_layers: 2,
          checkpoint_every: 1
        )

      assert %Axon{} = model
    end
  end

  describe "build_mamba_block/2" do
    test "builds a single block" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})
      block = GatedSSM.build_mamba_block(input, hidden_size: @embed, state_size: 4)
      assert %Axon{} = block
    end
  end

  describe "build_causal_conv1d/4" do
    test "builds causal conv layer" do
      input = Axon.input("input", shape: {nil, @seq_len, @embed})
      conv = GatedSSM.build_causal_conv1d(input, @embed, 4, "test_conv")
      assert %Axon{} = conv
    end
  end

  describe "build_selective_ssm/2" do
    test "builds selective SSM" do
      input = Axon.input("input", shape: {nil, @seq_len, @embed})
      ssm = GatedSSM.build_selective_ssm(input, hidden_size: @embed, state_size: 4)
      assert %Axon{} = ssm
    end
  end

  describe "init_cache/1" do
    test "initializes cache with correct structure" do
      cache = GatedSSM.init_cache(batch_size: 1, hidden_size: 8, state_size: 4, num_layers: 2)
      assert is_map(cache)
      assert cache.step == 0
      assert Map.has_key?(cache.layers, "layer_1")
      assert Map.has_key?(cache.layers, "layer_2")
      assert Nx.shape(cache.layers["layer_1"].h) == {1, 16, 4}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert GatedSSM.output_size(hidden_size: 128) == 128
      assert GatedSSM.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = GatedSSM.param_count(embed_dim: 64, hidden_size: 32, num_layers: 2)
      assert is_integer(count)
      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = GatedSSM.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
    end
  end
end
