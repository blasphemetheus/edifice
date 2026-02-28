defmodule Edifice.Generative.TarFlowTest do
  use ExUnit.Case, async: true

  @moduletag :generative

  alias Edifice.Generative.TarFlow

  import Edifice.TestHelpers

  @batch 2
  @seq_len 4
  @input_size 16

  defp base_opts do
    [
      input_size: @input_size,
      num_flows: 2,
      hidden_size: 32,
      num_heads: 4,
      dropout: 0.0
    ]
  end

  describe "build/1" do
    test "returns a tuple of encoder and decoder" do
      {encoder, decoder} = TarFlow.build(base_opts())
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end

  describe "encoder" do
    test "produces container output with output and log_det" do
      {encoder, _decoder} = TarFlow.build(base_opts())
      {init_fn, predict_fn} = Axon.build(encoder)

      template = %{"input" => Nx.template({@batch, @seq_len, @input_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = random_tensor({@batch, @seq_len, @input_size}, 42)
      result = predict_fn.(params, %{"input" => input})

      assert is_map(result)
      assert Map.has_key?(result, :output)
      assert Map.has_key?(result, :log_det)
    end

    test "output has correct shape" do
      {encoder, _decoder} = TarFlow.build(base_opts())
      {init_fn, predict_fn} = Axon.build(encoder)

      template = %{"input" => Nx.template({@batch, @seq_len, @input_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = random_tensor({@batch, @seq_len, @input_size}, 42)
      result = predict_fn.(params, %{"input" => input})

      assert Nx.shape(result.output) == {@batch, @seq_len, @input_size}
    end

    test "log_det has correct shape" do
      {encoder, _decoder} = TarFlow.build(base_opts())
      {init_fn, predict_fn} = Axon.build(encoder)

      template = %{"input" => Nx.template({@batch, @seq_len, @input_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = random_tensor({@batch, @seq_len, @input_size}, 42)
      result = predict_fn.(params, %{"input" => input})

      assert Nx.shape(result.log_det) == {@batch, @seq_len}
    end

    test "output is finite" do
      {encoder, _decoder} = TarFlow.build(base_opts())
      {init_fn, predict_fn} = Axon.build(encoder)

      template = %{"input" => Nx.template({@batch, @seq_len, @input_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = random_tensor({@batch, @seq_len, @input_size}, 42)
      result = predict_fn.(params, %{"input" => input})

      assert_finite!(result.output, "tar_flow_enc_output")
      assert_finite!(result.log_det, "tar_flow_enc_logdet")
    end
  end

  describe "decoder" do
    test "produces correct output shape" do
      {_encoder, decoder} = TarFlow.build(base_opts())
      {init_fn, predict_fn} = Axon.build(decoder)

      template = %{"latent" => Nx.template({@batch, @seq_len, @input_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      latent = random_tensor({@batch, @seq_len, @input_size}, 99)
      output = predict_fn.(params, %{"latent" => latent})

      assert Nx.shape(output) == {@batch, @seq_len, @input_size}
    end

    test "output is finite" do
      {_encoder, decoder} = TarFlow.build(base_opts())
      {init_fn, predict_fn} = Axon.build(decoder)

      template = %{"latent" => Nx.template({@batch, @seq_len, @input_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      latent = random_tensor({@batch, @seq_len, @input_size}, 99)
      output = predict_fn.(params, %{"latent" => latent})

      assert_finite!(output, "tar_flow_dec_output")
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert TarFlow.output_size(input_size: 32) == 32
    end

    test "returns default when no opts" do
      assert TarFlow.output_size() == 64
    end
  end

  describe "registry integration" do
    test "Edifice.build(:tar_flow, ...) works" do
      result =
        Edifice.build(:tar_flow,
          input_size: @input_size,
          num_flows: 2,
          hidden_size: 32,
          num_heads: 4
        )

      assert {%Axon{}, %Axon{}} = result
    end
  end
end
