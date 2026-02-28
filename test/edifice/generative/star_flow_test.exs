defmodule Edifice.Generative.STARFlowTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.STARFlow

  import Edifice.TestHelpers

  @moduletag :generative

  @batch 2
  @seq_len 4
  @input_size 16
  @hidden_size 32

  describe "build/1" do
    test "returns {encoder, decoder} tuple" do
      {encoder, decoder} =
        STARFlow.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_flows: 2,
          deep_layers: 2,
          shallow_layers: 1
        )

      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end

    test "encoder forward pass produces output and log_det" do
      {encoder, _} =
        STARFlow.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_flows: 2,
          deep_layers: 2,
          shallow_layers: 1,
          num_heads: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      input = random_tensor({@batch, @seq_len, @input_size}, 42)
      params = init_fn.(input, %{})
      result = predict_fn.(params, input)

      assert %{output: output, log_det: _log_det} = result
      assert Nx.shape(output) == {@batch, @seq_len, @input_size}
      assert_finite!(output)
    end

    test "decoder forward pass produces correct shape" do
      {_, decoder} =
        STARFlow.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_flows: 2,
          deep_layers: 2,
          shallow_layers: 1,
          num_heads: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(decoder)
      input = random_tensor({@batch, @seq_len, @input_size}, 99)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @seq_len, @input_size}
    end

    test "batch=1 forward pass" do
      {encoder, _} =
        STARFlow.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_flows: 2,
          deep_layers: 1,
          shallow_layers: 1,
          num_heads: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      input = random_tensor({1, @seq_len, @input_size}, 55)
      params = init_fn.(input, %{})
      result = predict_fn.(params, input)

      assert %{output: output} = result
      assert Nx.shape(output) == {1, @seq_len, @input_size}
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert STARFlow.output_size(input_size: 128) == 128
    end

    test "returns default" do
      assert STARFlow.output_size() == 64
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = STARFlow.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:deep_layers] == 8
      assert defaults[:shallow_layers] == 2
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      {encoder, decoder} =
        Edifice.build(:star_flow,
          input_size: 8,
          hidden_size: 16,
          num_flows: 2,
          deep_layers: 1,
          shallow_layers: 1,
          num_heads: 2,
          dropout: 0.0
        )

      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end
end
