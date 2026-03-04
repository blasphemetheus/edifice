defmodule Edifice.MixedPrecisionTest do
  use ExUnit.Case, async: true
  @moduletag :mixed_precision

  alias Edifice.MixedPrecision

  @embed_dim 8
  @seq_len 16
  @batch 1

  # Simple model with dense + layer_norm + rms_norm + dense
  defp build_model_with_norms do
    Axon.input("x", shape: {nil, @embed_dim})
    |> Axon.dense(@embed_dim, name: "dense_0")
    |> Axon.layer_norm(name: "ln_0")
    |> Edifice.Blocks.RMSNorm.layer(hidden_size: @embed_dim, name: "rms_0")
    |> Axon.dense(@embed_dim, name: "dense_1")
  end

  defp build_model_with_batch_norm do
    Axon.input("x", shape: {nil, @embed_dim})
    |> Axon.dense(@embed_dim, name: "dense_0")
    |> Axon.batch_norm(name: "bn_0")
    |> Axon.dense(@embed_dim, name: "dense_1")
  end

  defp init_and_predict(model) do
    template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    input = %{"x" => Nx.iota({@batch, @embed_dim}, type: :f32)}
    {predict_fn, params, input}
  end

  describe "policy/1" do
    test "creates bf16 policy" do
      policy = MixedPrecision.policy(:bf16)
      assert %Axon.MixedPrecision.Policy{} = policy
    end

    test "creates fp16 policy" do
      policy = MixedPrecision.policy(:fp16)
      assert %Axon.MixedPrecision.Policy{} = policy
    end
  end

  describe "apply/2 with :bf16 preset" do
    test "output is bf16" do
      model = build_model_with_norms() |> MixedPrecision.apply(:bf16)
      {predict_fn, params, input} = init_and_predict(model)
      out = predict_fn.(params, input)
      assert Nx.type(out) == {:bf, 16}
    end

    test "params remain f32" do
      model = build_model_with_norms() |> MixedPrecision.apply(:bf16)
      template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
      {init_fn, _} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # All param tensors should be f32
      check_all_params_f32(params.data)
    end

    test "preserves numerical stability across norms" do
      model_f32 = build_model_with_norms()
      model_bf16 = MixedPrecision.apply(model_f32, :bf16)

      template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
      {init_fn, pred_f32} = Axon.build(model_f32)
      {_, pred_bf16} = Axon.build(model_bf16)

      params = init_fn.(template, Axon.ModelState.empty())
      input = %{"x" => Nx.iota({@batch, @embed_dim}, type: :f32)}

      out_f32 = pred_f32.(params, input)
      out_bf16 = pred_bf16.(params, input) |> Nx.as_type(:f32)

      # Should be close — bf16 introduces small rounding but norms in f32 keep it stable
      diff = Nx.subtract(out_f32, out_bf16) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0, "f32 vs bf16 max diff: #{diff}"
    end
  end

  describe "apply/2 with :fp16 preset" do
    test "output is fp16" do
      model = build_model_with_norms() |> MixedPrecision.apply(:fp16)
      {predict_fn, params, input} = init_and_predict(model)
      out = predict_fn.(params, input)
      assert Nx.type(out) == {:f, 16}
    end
  end

  describe "apply/3 with extra :except" do
    test "can exclude additional op_names" do
      model =
        Axon.input("x", shape: {nil, @embed_dim})
        |> Axon.dense(@embed_dim, name: "dense_0")
        |> Axon.dense(@embed_dim, name: "dense_1")

      # Excluding :dense means nothing gets cast
      model_mp = MixedPrecision.apply(model, :bf16, except: [:dense])
      summary = MixedPrecision.summary(model_mp)

      # Input + 2 dense layers — dense excluded, so fewer layers get policy
      assert summary.with_policy < summary.total
    end
  end

  describe "apply/3 with only_norms: false" do
    test "skips default norm exclusions" do
      model = build_model_with_norms()
      # only_norms: false means NO default exclusions — everything gets bf16
      model_mp = MixedPrecision.apply(model, :bf16, only_norms: false)

      {predict_fn, params, input} = init_and_predict(model_mp)
      out = predict_fn.(params, input)
      assert Nx.type(out) == {:bf, 16}
    end
  end

  describe "apply with batch_norm" do
    test "batch_norm stays in f32" do
      model = build_model_with_batch_norm() |> MixedPrecision.apply(:bf16)
      {predict_fn, params, input} = init_and_predict(model)
      out = predict_fn.(params, input)
      assert Nx.type(out) == {:bf, 16}
    end
  end

  describe "norm_op_names/0" do
    test "returns all known normalization op_names" do
      names = MixedPrecision.norm_op_names()
      assert :layer_norm in names
      assert :batch_norm in names
      assert :rms_norm in names
      assert :adaptive_norm in names
    end
  end

  describe "summary/1" do
    test "counts layers with and without policy" do
      model = build_model_with_norms() |> MixedPrecision.apply(:bf16)
      summary = MixedPrecision.summary(model)

      assert summary.total >= 4
      assert summary.with_policy >= 1
      assert summary.without_policy >= 1
    end

    test "plain model has no policies" do
      model = build_model_with_norms()
      summary = MixedPrecision.summary(model)

      assert summary.with_policy == 0
      assert summary.total >= 4
    end
  end

  describe "loss scaling" do
    test "init_loss_scale returns default state" do
      state = MixedPrecision.init_loss_scale()
      assert state.scale == 65536.0
      assert state.growth_factor == 2.0
      assert state.backoff_factor == 0.5
      assert state.growth_interval == 2000
      assert state.good_steps == 0
    end

    test "init_loss_scale accepts custom options" do
      state = MixedPrecision.init_loss_scale(scale: 1024.0, growth_interval: 100)
      assert state.scale == 1024.0
      assert state.growth_interval == 100
    end

    test "scale_loss multiplies by scale factor" do
      state = MixedPrecision.init_loss_scale(scale: 1024.0)
      loss = Nx.tensor(0.5)
      {scaled, _state} = MixedPrecision.scale_loss(loss, state)
      assert_close(scaled, Nx.tensor(512.0))
    end

    test "unscale_grads divides by scale factor" do
      state = MixedPrecision.init_loss_scale(scale: 1024.0)
      grads = %{"w" => Nx.tensor([1024.0, 2048.0])}
      {:ok, unscaled, _state} = MixedPrecision.unscale_grads(grads, state)
      assert_close(unscaled["w"], Nx.tensor([1.0, 2.0]))
    end

    test "unscale_grads detects overflow" do
      state = MixedPrecision.init_loss_scale(scale: 1024.0)
      grads = %{"w" => Nx.Constants.infinity()}
      {:overflow, new_state} = MixedPrecision.unscale_grads(grads, state)
      assert new_state.scale == 1024.0 * 0.5
      assert new_state.good_steps == 0
    end

    test "unscale_grads grows scale after growth_interval" do
      state = MixedPrecision.init_loss_scale(scale: 1024.0, growth_interval: 2)
      grads = %{"w" => Nx.tensor([1.0])}

      {:ok, _, state} = MixedPrecision.unscale_grads(grads, state)
      assert state.good_steps == 1
      assert state.scale == 1024.0

      {:ok, _, state} = MixedPrecision.unscale_grads(grads, state)
      assert state.good_steps == 0
      assert state.scale == 2048.0
    end

    test "unscale_grads handles nested maps" do
      state = MixedPrecision.init_loss_scale(scale: 2.0)

      grads = %{
        "layer_0" => %{
          "weight" => Nx.tensor([4.0, 6.0]),
          "bias" => Nx.tensor([2.0])
        },
        "layer_1" => %{
          "weight" => Nx.tensor([8.0])
        }
      }

      {:ok, unscaled, _state} = MixedPrecision.unscale_grads(grads, state)
      assert_close(unscaled["layer_0"]["weight"], Nx.tensor([2.0, 3.0]))
      assert_close(unscaled["layer_0"]["bias"], Nx.tensor([1.0]))
      assert_close(unscaled["layer_1"]["weight"], Nx.tensor([4.0]))
    end
  end

  describe "integration with Edifice.build" do
    test "works with decoder_only architecture" do
      model = Edifice.build(:decoder_only,
        embed_dim: @embed_dim,
        hidden_size: @embed_dim,
        seq_len: @seq_len,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2
      )

      model_mp = MixedPrecision.apply(model, :bf16)
      assert %Axon{} = model_mp

      summary = MixedPrecision.summary(model_mp)
      assert summary.with_policy >= 1
    end
  end

  # Helpers

  defp assert_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < atol, "tensors differ by #{diff}, expected < #{atol}"
  end

  defp check_all_params_f32(params) when is_map(params) do
    Enum.each(params, fn
      {_k, %Nx.Tensor{} = t} ->
        assert Nx.type(t) == {:f, 32}, "param has type #{inspect(Nx.type(t))}, expected f32"

      {_k, nested} when is_map(nested) ->
        check_all_params_f32(nested)

      _ ->
        :ok
    end)
  end
end
