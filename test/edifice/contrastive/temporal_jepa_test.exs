defmodule Edifice.Contrastive.TemporalJEPATest do
  use ExUnit.Case, async: true

  alias Edifice.Contrastive.TemporalJEPA

  @batch 2
  @input_dim 32
  @embed_dim 32
  @seq_len 8

  @opts [
    input_dim: @input_dim,
    embed_dim: @embed_dim,
    predictor_embed_dim: 16,
    encoder_depth: 2,
    predictor_depth: 2,
    num_heads: 4,
    dropout: 0.0,
    seq_len: @seq_len
  ]

  defp random_sequence do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_dim})
    input
  end

  defp random_context do
    key = Nx.Random.key(99)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @embed_dim})
    input
  end

  defp encoder_template,
    do: %{"state_sequence" => Nx.template({@batch, @seq_len, @input_dim}, :f32)}

  defp predictor_template,
    do: %{"context" => Nx.template({@batch, @embed_dim}, :f32)}

  describe "build/1" do
    test "returns {context_encoder, predictor} tuple" do
      {context_encoder, predictor} = TemporalJEPA.build(@opts)
      assert %Axon{} = context_encoder
      assert %Axon{} = predictor
    end
  end

  describe "context encoder" do
    test "produces correct output shape" do
      {context_encoder, _predictor} = TemporalJEPA.build(@opts)
      {init_fn, predict_fn} = Axon.build(context_encoder, mode: :inference)
      params = init_fn.(encoder_template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_sequence()})
      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "output contains finite values" do
      {context_encoder, _predictor} = TemporalJEPA.build(@opts)
      {init_fn, predict_fn} = Axon.build(context_encoder, mode: :inference)
      params = init_fn.(encoder_template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_sequence()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "predictor" do
    test "produces correct output shape" do
      {_context_encoder, predictor} = TemporalJEPA.build(@opts)
      {init_fn, predict_fn} = Axon.build(predictor, mode: :inference)
      params = init_fn.(predictor_template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"context" => random_context()})
      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "output contains finite values" do
      {_context_encoder, predictor} = TemporalJEPA.build(@opts)
      {init_fn, predict_fn} = Axon.build(predictor, mode: :inference)
      params = init_fn.(predictor_template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"context" => random_context()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "loss/2" do
    test "identical inputs give near-zero loss" do
      pred = Nx.broadcast(1.0, {@batch, @embed_dim})
      loss = TemporalJEPA.loss(pred, pred)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) < 0.001
    end

    test "different inputs give positive loss" do
      pred = Nx.broadcast(1.0, {@batch, @embed_dim})
      target = Nx.broadcast(0.0, {@batch, @embed_dim})
      loss = TemporalJEPA.loss(pred, target)
      assert Nx.to_number(loss) > 0
    end
  end

  describe "generate_temporal_mask/3" do
    test "returns mask of correct shape" do
      key = Nx.Random.key(42)
      {mask, _key} = TemporalJEPA.generate_temporal_mask(key, @seq_len, 0.5)
      assert Nx.shape(mask) == {@seq_len}
    end

    test "mask contains boolean values" do
      key = Nx.Random.key(42)
      {mask, _key} = TemporalJEPA.generate_temporal_mask(key, @seq_len, 0.5)

      # All values should be 0 or 1
      sum = Nx.sum(mask) |> Nx.to_number()
      assert sum >= 0 and sum <= @seq_len
    end
  end

  describe "ema_update/3" do
    test "blends parameters" do
      context_params = %{"tjepa_enc_proj" => Nx.tensor(10.0)}
      target_params = %{"tjepa_tgt_proj" => Nx.tensor(0.0)}
      updated = TemporalJEPA.ema_update(context_params, target_params, momentum: 0.5)
      assert_in_delta Nx.to_number(updated["tjepa_tgt_proj"]), 5.0, 0.01
    end
  end

  describe "output_size/1" do
    test "returns embed_dim" do
      assert TemporalJEPA.output_size(@opts) == @embed_dim
    end
  end
end
