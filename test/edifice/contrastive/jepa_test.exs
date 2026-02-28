defmodule Edifice.Contrastive.JEPATest do
  use ExUnit.Case, async: true
  @moduletag :contrastive

  alias Edifice.Contrastive.JEPA

  @batch 4
  @input_dim 32
  @embed_dim 16
  @predictor_embed_dim 8
  @encoder_depth 2
  @predictor_depth 2
  @num_heads 4

  @opts [
    input_dim: @input_dim,
    embed_dim: @embed_dim,
    predictor_embed_dim: @predictor_embed_dim,
    encoder_depth: @encoder_depth,
    predictor_depth: @predictor_depth,
    num_heads: @num_heads,
    dropout: 0.0
  ]

  defp random_features do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_dim})
    input
  end

  defp random_context do
    key = Nx.Random.key(99)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @embed_dim})
    input
  end

  # ============================================================================
  # build/1
  # ============================================================================

  describe "build/1" do
    test "returns {context_encoder, predictor} tuple" do
      {context_encoder, predictor} = JEPA.build(@opts)
      assert %Axon{} = context_encoder
      assert %Axon{} = predictor
    end

    test "context encoder produces correct output shape" do
      {context_encoder, _predictor} = JEPA.build(@opts)
      {init_fn, predict_fn} = Axon.build(context_encoder)

      params =
        init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_features())
      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "predictor produces correct output shape" do
      {_context_encoder, predictor} = JEPA.build(@opts)
      {init_fn, predict_fn} = Axon.build(predictor)

      params =
        init_fn.(Nx.template({@batch, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_context())
      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "outputs are finite" do
      {context_encoder, predictor} = JEPA.build(@opts)

      # Context encoder
      {init_fn, predict_fn} = Axon.build(context_encoder)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_features())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Predictor
      {init_fn_p, predict_fn_p} = Axon.build(predictor)
      params_p = init_fn_p.(Nx.template({@batch, @embed_dim}, :f32), Axon.ModelState.empty())
      output_p = predict_fn_p.(params_p, random_context())
      assert Nx.all(Nx.is_nan(output_p) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output_p) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # build_context_encoder/1 standalone
  # ============================================================================

  describe "build_context_encoder/1" do
    test "builds standalone context encoder" do
      encoder = JEPA.build_context_encoder(@opts)
      assert %Axon{} = encoder

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_features())

      assert Nx.shape(output) == {@batch, @embed_dim}
    end
  end

  # ============================================================================
  # build_predictor/1 standalone
  # ============================================================================

  describe "build_predictor/1" do
    test "builds standalone predictor" do
      predictor = JEPA.build_predictor(@opts)
      assert %Axon{} = predictor

      {init_fn, predict_fn} = Axon.build(predictor)
      params = init_fn.(Nx.template({@batch, @embed_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_context())

      assert Nx.shape(output) == {@batch, @embed_dim}
    end
  end

  # ============================================================================
  # EMA update
  # ============================================================================

  describe "ema_update/3" do
    test "blends context params into target params" do
      context_params = %{
        "ctx_enc_proj" => 10.0
      }

      target_params = %{
        "tgt_enc_proj" => 0.0
      }

      updated = JEPA.ema_update(context_params, target_params, momentum: 0.5)

      # 0.5 * 0 + 0.5 * 10 = 5
      assert_in_delta Nx.to_number(updated["tgt_enc_proj"]), 5.0, 0.01
    end

    test "momentum=0.0 copies context params entirely" do
      context_params = %{"ctx_enc_proj" => 10.0}
      target_params = %{"tgt_enc_proj" => 0.0}

      updated = JEPA.ema_update(context_params, target_params, momentum: 0.0)
      assert_in_delta Nx.to_number(updated["tgt_enc_proj"]), 10.0, 0.01
    end

    test "momentum=1.0 keeps target unchanged" do
      context_params = %{"ctx_enc_proj" => 10.0}
      target_params = %{"tgt_enc_proj" => 5.0}

      updated = JEPA.ema_update(context_params, target_params, momentum: 1.0)
      assert_in_delta Nx.to_number(updated["tgt_enc_proj"]), 5.0, 0.01
    end

    test "handles nested map params" do
      context_params = %{
        "ctx_enc_proj" => %{"kernel" => 10.0, "bias" => 2.0}
      }

      target_params = %{
        "tgt_enc_proj" => %{"kernel" => 0.0, "bias" => 0.0}
      }

      updated = JEPA.ema_update(context_params, target_params, momentum: 0.5)

      inner = updated["tgt_enc_proj"]
      assert_in_delta Nx.to_number(inner["kernel"]), 5.0, 0.01
      assert_in_delta Nx.to_number(inner["bias"]), 1.0, 0.01
    end

    test "unmatched keys remain unchanged" do
      context_params = %{"ctx_enc_proj" => 10.0}
      target_params = %{"tgt_enc_proj" => 5.0, "tgt_enc_extra" => 99.0}

      updated = JEPA.ema_update(context_params, target_params, momentum: 0.5)

      assert_in_delta Nx.to_number(updated["tgt_enc_proj"]), 7.5, 0.01
      assert_in_delta Nx.to_number(updated["tgt_enc_extra"]), 99.0, 0.01
    end
  end

  # ============================================================================
  # Loss
  # ============================================================================

  describe "loss/2" do
    test "loss of identical inputs is zero" do
      key = Nx.Random.key(300)
      {pred, _} = Nx.Random.uniform(key, shape: {@batch, @embed_dim})

      loss = JEPA.loss(pred, pred)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) < 0.001
    end

    test "loss is non-negative" do
      key = Nx.Random.key(301)
      {pred, key} = Nx.Random.normal(key, shape: {@batch, @embed_dim})
      {target, _} = Nx.Random.normal(key, shape: {@batch, @embed_dim})

      loss = JEPA.loss(pred, target)
      assert Nx.to_number(loss) >= 0.0
    end

    test "loss increases with distance" do
      base = Nx.broadcast(1.0, {@batch, @embed_dim})

      close = Nx.broadcast(1.1, {@batch, @embed_dim})
      far = Nx.broadcast(5.0, {@batch, @embed_dim})

      loss_close = JEPA.loss(base, close) |> Nx.to_number()
      loss_far = JEPA.loss(base, far) |> Nx.to_number()

      assert loss_far > loss_close
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  describe "output_size/1" do
    test "returns embed_dim" do
      assert JEPA.output_size(@opts) == @embed_dim
    end

    test "returns default when no opts" do
      assert JEPA.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = JEPA.recommended_defaults()
      assert Keyword.has_key?(defaults, :embed_dim)
      assert Keyword.has_key?(defaults, :predictor_embed_dim)
      assert Keyword.has_key?(defaults, :encoder_depth)
      assert Keyword.has_key?(defaults, :predictor_depth)
    end
  end

  describe "default_momentum/0" do
    test "returns 0.996" do
      assert JEPA.default_momentum() == 0.996
    end
  end
end
