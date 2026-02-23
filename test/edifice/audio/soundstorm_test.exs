defmodule Edifice.Audio.SoundStormTest do
  use ExUnit.Case, async: true

  alias Edifice.Audio.SoundStorm

  @batch_size 2
  @num_codebooks 4
  @codebook_size 64
  @seq_len 8

  describe "SoundStorm.build/1" do
    test "produces correct output shape" do
      model = SoundStorm.build(
        num_codebooks: @num_codebooks,
        codebook_size: @codebook_size,
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 2,
        conv_kernel_size: 3
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      total_len = @num_codebooks * @seq_len

      params =
        init_fn.(
          %{"tokens" => Nx.template({@batch_size, total_len}, :s64)},
          Axon.ModelState.empty()
        )

      tokens = Nx.broadcast(0, {@batch_size, total_len}) |> Nx.as_type(:s64)
      output = predict_fn.(params, %{"tokens" => tokens})

      # Output: [batch, total_len, codebook_size]
      assert Nx.shape(output) == {@batch_size, total_len, @codebook_size}
    end

    test "with different configurations" do
      model = SoundStorm.build(
        num_codebooks: 2,
        codebook_size: 32,
        hidden_dim: 16,
        num_layers: 1,
        num_heads: 2,
        conv_kernel_size: 3
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      total_len = 2 * 4  # 2 codebooks, 4 seq_len

      params =
        init_fn.(
          %{"tokens" => Nx.template({1, total_len}, :s64)},
          Axon.ModelState.empty()
        )

      tokens = Nx.broadcast(0, {1, total_len}) |> Nx.as_type(:s64)
      output = predict_fn.(params, %{"tokens" => tokens})

      assert Nx.shape(output) == {1, total_len, 32}
    end
  end

  describe "SoundStorm.soundstorm_step/6" do
    test "performs one refinement step" do
      model = SoundStorm.build(
        num_codebooks: @num_codebooks,
        codebook_size: @codebook_size,
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        conv_kernel_size: 3
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      total_len = @num_codebooks * @seq_len

      params =
        init_fn.(
          %{"tokens" => Nx.template({@batch_size, total_len}, :s64)},
          Axon.ModelState.empty()
        )

      tokens = Nx.broadcast(0, {@batch_size, total_len}) |> Nx.as_type(:s64)

      # Mask: all positions masked
      mask = Nx.broadcast(1, {@batch_size, total_len}) |> Nx.as_type(:u8) |> Nx.equal(1)

      new_tokens = SoundStorm.soundstorm_step(predict_fn, params, tokens, mask, 1, 4)

      assert Nx.shape(new_tokens) == {@batch_size, total_len}
    end
  end

  describe "SoundStorm.generate/4" do
    @tag :slow
    test "generates full sequence from conditioning" do
      model = SoundStorm.build(
        num_codebooks: 2,
        codebook_size: 16,
        hidden_dim: 16,
        num_layers: 1,
        num_heads: 2,
        conv_kernel_size: 3
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      seq_len = 4
      total_len = 2 * seq_len

      params =
        init_fn.(
          %{"tokens" => Nx.template({1, total_len}, :s64)},
          Axon.ModelState.empty()
        )

      # Conditioning tokens for codebook 0
      conditioning = Nx.tensor([[1, 2, 3, 4]]) |> Nx.as_type(:s64)

      generated = SoundStorm.generate(predict_fn, params, conditioning,
        num_steps: 2,
        num_codebooks: 2,
        mask_token: 0
      )

      # Output: [batch, num_codebooks, seq_len]
      assert Nx.shape(generated) == {1, 2, seq_len}

      # First codebook should match conditioning
      first_codebook = Nx.slice(generated, [0, 0, 0], [1, 1, seq_len]) |> Nx.squeeze()
      conditioning_squeezed = Nx.squeeze(conditioning)
      assert Nx.to_list(first_codebook) == Nx.to_list(conditioning_squeezed)
    end
  end

  describe "SoundStorm.output_size/1" do
    test "returns codebook_size" do
      assert SoundStorm.output_size(codebook_size: 1024) == 1024
      assert SoundStorm.output_size(codebook_size: 2048) == 2048
      assert SoundStorm.output_size() == 1024  # default
    end
  end
end
