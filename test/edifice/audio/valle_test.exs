defmodule Edifice.Audio.VALLETest do
  use ExUnit.Case, async: true

  alias Edifice.Audio.VALLE

  @batch_size 2
  @text_len 8
  @prompt_len 4
  @audio_len 6
  @hidden_dim 64
  @text_vocab_size 32
  @audio_vocab_size 64
  @num_codebooks 4
  @num_layers 2
  @num_heads 4

  describe "VALLE.build/1" do
    test "produces AR and NAR model tuple" do
      {ar_model, nar_model} =
        VALLE.build(
          text_vocab_size: @text_vocab_size,
          audio_vocab_size: @audio_vocab_size,
          hidden_dim: @hidden_dim,
          num_layers: @num_layers,
          num_heads: @num_heads,
          num_codebooks: @num_codebooks
        )

      assert %Axon{} = ar_model
      assert %Axon{} = nar_model
    end
  end

  describe "VALLE.build_ar/1" do
    test "produces correct output shape" do
      ar_model =
        VALLE.build_ar(
          text_vocab_size: @text_vocab_size,
          audio_vocab_size: @audio_vocab_size,
          hidden_dim: @hidden_dim,
          num_layers: @num_layers,
          num_heads: @num_heads,
          num_codebooks: @num_codebooks
        )

      {init_fn, predict_fn} = Axon.build(ar_model, mode: :inference)

      params =
        init_fn.(
          %{
            "text_tokens" => Nx.template({@batch_size, @text_len}, :s64),
            "prompt_tokens" => Nx.template({@batch_size, @num_codebooks, @prompt_len}, :s64),
            "audio_tokens" => Nx.template({@batch_size, @audio_len}, :s64)
          },
          Axon.ModelState.empty()
        )

      text_tokens = Nx.broadcast(0, {@batch_size, @text_len}) |> Nx.as_type(:s64)

      prompt_tokens =
        Nx.broadcast(0, {@batch_size, @num_codebooks, @prompt_len}) |> Nx.as_type(:s64)

      audio_tokens = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)

      output =
        predict_fn.(params, %{
          "text_tokens" => text_tokens,
          "prompt_tokens" => prompt_tokens,
          "audio_tokens" => audio_tokens
        })

      # Total len = text_len + prompt_len + audio_len
      total_len = @text_len + @prompt_len + @audio_len

      # Output: [batch, total_len, audio_vocab_size]
      assert Nx.shape(output) == {@batch_size, total_len, @audio_vocab_size}
    end

    test "with minimal configuration" do
      ar_model =
        VALLE.build_ar(
          text_vocab_size: 16,
          audio_vocab_size: 32,
          hidden_dim: 32,
          num_layers: 1,
          num_heads: 2,
          num_codebooks: 2
        )

      {init_fn, predict_fn} = Axon.build(ar_model, mode: :inference)

      params =
        init_fn.(
          %{
            "text_tokens" => Nx.template({1, 4}, :s64),
            "prompt_tokens" => Nx.template({1, 2, 3}, :s64),
            "audio_tokens" => Nx.template({1, 2}, :s64)
          },
          Axon.ModelState.empty()
        )

      text = Nx.broadcast(0, {1, 4}) |> Nx.as_type(:s64)
      prompt = Nx.broadcast(0, {1, 2, 3}) |> Nx.as_type(:s64)
      audio = Nx.broadcast(0, {1, 2}) |> Nx.as_type(:s64)

      output =
        predict_fn.(params, %{
          "text_tokens" => text,
          "prompt_tokens" => prompt,
          "audio_tokens" => audio
        })

      # total_len = 4 + 3 + 2 = 9
      assert Nx.shape(output) == {1, 9, 32}
    end
  end

  describe "VALLE.build_nar/1" do
    test "produces correct output shape" do
      nar_model =
        VALLE.build_nar(
          text_vocab_size: @text_vocab_size,
          audio_vocab_size: @audio_vocab_size,
          hidden_dim: @hidden_dim,
          num_layers: @num_layers,
          num_heads: @num_heads,
          num_codebooks: @num_codebooks
        )

      {init_fn, predict_fn} = Axon.build(nar_model, mode: :inference)

      params =
        init_fn.(
          %{
            "text_tokens" => Nx.template({@batch_size, @text_len}, :s64),
            "coarse_tokens" => Nx.template({@batch_size, @audio_len}, :s64),
            "prev_codebook_tokens" => Nx.template({@batch_size, @audio_len}, :s64),
            "codebook_idx" => Nx.template({}, :s64)
          },
          Axon.ModelState.empty()
        )

      text_tokens = Nx.broadcast(0, {@batch_size, @text_len}) |> Nx.as_type(:s64)
      coarse_tokens = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)
      prev_tokens = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)
      codebook_idx = Nx.tensor(1, type: :s64)

      output =
        predict_fn.(params, %{
          "text_tokens" => text_tokens,
          "coarse_tokens" => coarse_tokens,
          "prev_codebook_tokens" => prev_tokens,
          "codebook_idx" => codebook_idx
        })

      # Total len = text_len + audio_len
      total_len = @text_len + @audio_len

      # Output: [batch, total_len, audio_vocab_size]
      assert Nx.shape(output) == {@batch_size, total_len, @audio_vocab_size}
    end
  end

  describe "VALLE.cross_entropy_loss/2" do
    test "computes loss correctly" do
      logits = Nx.broadcast(0.0, {@batch_size, @audio_len, @audio_vocab_size})
      targets = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)

      loss = VALLE.cross_entropy_loss(logits, targets)

      assert Nx.shape(loss) == {}
      # Uniform logits with target 0 should give log(@audio_vocab_size)
      expected_loss = :math.log(@audio_vocab_size)
      assert_in_delta Nx.to_number(loss), expected_loss, 0.01
    end
  end

  describe "VALLE.valle_loss/5" do
    test "computes combined loss" do
      ar_logits = Nx.broadcast(0.0, {@batch_size, @audio_len, @audio_vocab_size})
      ar_targets = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)
      nar_logits = Nx.broadcast(0.0, {@batch_size, @audio_len, @audio_vocab_size})
      nar_targets = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)

      loss = VALLE.valle_loss(ar_logits, ar_targets, nar_logits, nar_targets)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "respects weights" do
      ar_logits = Nx.broadcast(0.0, {@batch_size, @audio_len, @audio_vocab_size})
      ar_targets = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)
      nar_logits = Nx.broadcast(0.0, {@batch_size, @audio_len, @audio_vocab_size})
      nar_targets = Nx.broadcast(0, {@batch_size, @audio_len}) |> Nx.as_type(:s64)

      # AR only
      loss_ar_only =
        VALLE.valle_loss(ar_logits, ar_targets, nar_logits, nar_targets,
          ar_weight: 1.0,
          nar_weight: 0.0
        )

      # NAR only
      loss_nar_only =
        VALLE.valle_loss(ar_logits, ar_targets, nar_logits, nar_targets,
          ar_weight: 0.0,
          nar_weight: 1.0
        )

      # Both losses should be similar for uniform logits
      assert_in_delta Nx.to_number(loss_ar_only), Nx.to_number(loss_nar_only), 0.01
    end
  end

  describe "VALLE.output_size/1" do
    test "returns audio vocabulary size" do
      assert VALLE.output_size(audio_vocab_size: 1024) == 1024
      assert VALLE.output_size(audio_vocab_size: 2048) == 2048
      # default
      assert VALLE.output_size() == 1024
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build VALL-E via registry" do
      {ar_model, nar_model} =
        Edifice.build(:valle,
          text_vocab_size: @text_vocab_size,
          audio_vocab_size: @audio_vocab_size,
          hidden_dim: @hidden_dim,
          num_layers: @num_layers,
          num_heads: @num_heads
        )

      assert %Axon{} = ar_model
      assert %Axon{} = nar_model
    end
  end
end
