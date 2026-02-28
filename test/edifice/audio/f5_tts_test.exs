defmodule Edifice.Audio.F5TTSTest do
  use ExUnit.Case, async: true
  @moduletag :audio

  alias Edifice.Audio.F5TTS

  import Edifice.TestHelpers

  @batch 2
  @seq 8
  @mel_dim 4
  @text_dim 16

  # Tiny config for BinaryBackend testing
  @opts [
    mel_dim: @mel_dim,
    dim: 32,
    depth: 2,
    heads: 4,
    ff_mult: 2,
    dropout: 0.0,
    text_dim: @text_dim,
    text_num_embeds: 26,
    conv_layers: 2,
    conv_mult: 2,
    conv_pos_kernel: 3,
    conv_pos_groups: 4
  ]

  defp build_and_predict(opts \\ @opts) do
    model = F5TTS.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    mel_dim = Keyword.get(opts, :mel_dim, @mel_dim)
    batch = Keyword.get(opts, :batch, @batch)
    seq = Keyword.get(opts, :seq, @seq)

    input = %{
      "noisy_mel" => random_tensor({batch, seq, mel_dim}),
      "cond_mel" => random_tensor({batch, seq, mel_dim}),
      "text" =>
        Nx.tensor(for(_ <- 1..batch, do: Enum.map(1..seq, fn _ -> :rand.uniform(26) end))),
      "timestep" => Nx.tensor(for(_ <- 1..batch, do: :rand.uniform()))
    }

    params = init_fn.(input, %{})
    output = predict_fn.(params, input)
    {output, {batch, seq, mel_dim}}
  end

  describe "build/1" do
    test "produces correct output shape" do
      {output, {batch, seq, mel_dim}} = build_and_predict()
      assert Nx.shape(output) == {batch, seq, mel_dim}
    end

    test "output values are finite" do
      {output, _shape} = build_and_predict()
      assert_finite!(output)
    end

    test "handles batch_size=1" do
      {output, _} = build_and_predict(Keyword.put(@opts, :batch, 1))
      assert {1, @seq, @mel_dim} == Nx.shape(output)
      assert_finite!(output)
    end

    test "works with different mel dimensions" do
      opts = Keyword.put(@opts, :mel_dim, 8)
      {output, _} = build_and_predict(opts)
      assert {2, @seq, 8} == Nx.shape(output)
    end

    test "works with different sequence length" do
      {output, _} = build_and_predict(Keyword.put(@opts, :seq, 16))
      assert {2, 16, @mel_dim} == Nx.shape(output)
    end

    test "works with single DiT block" do
      opts = Keyword.put(@opts, :depth, 1)
      {output, _} = build_and_predict(opts)
      assert {2, @seq, @mel_dim} == Nx.shape(output)
      assert_finite!(output)
    end

    test "works with single ConvNeXt layer" do
      opts = Keyword.put(@opts, :conv_layers, 1)
      {output, _} = build_and_predict(opts)
      assert {2, @seq, @mel_dim} == Nx.shape(output)
    end
  end

  describe "output_size/1" do
    test "returns mel_dim" do
      assert F5TTS.output_size(mel_dim: 100) == 100
    end

    test "uses default mel_dim" do
      assert F5TTS.output_size([]) == 100
    end
  end
end
