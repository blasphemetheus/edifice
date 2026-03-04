defmodule Edifice.TrainingTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias Edifice.Training

  @embed_dim 8
  @batch 1

  defp build_model do
    Axon.input("x", shape: {nil, @embed_dim})
    |> Axon.dense(@embed_dim, name: "d1", activation: :relu)
    |> Axon.dense(@embed_dim, name: "d2", activation: :relu)
    |> Axon.dense(4, name: "out")
  end

  defp build_deep_model do
    Enum.reduce(1..6, Axon.input("x", shape: {nil, @embed_dim}), fn i, acc ->
      acc |> Axon.dense(@embed_dim, name: "layer_#{i}", activation: :relu)
    end)
    |> Axon.dense(4, name: "out")
  end

  defp init_model(model) do
    template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    {predict_fn, params}
  end

  defp make_input do
    %{"x" => Nx.iota({@batch, @embed_dim}, type: :f32)}
  end

  describe "remat/2" do
    test "produces same forward output" do
      model = build_model()
      {predict_fn, params} = init_model(model)
      input = make_input()

      ckpt_fn = Training.remat(predict_fn)

      out_normal = predict_fn.(params, input)
      out_ckpt = ckpt_fn.(params, input)

      assert_close(out_normal, out_ckpt)
    end

    test "policy: :none is passthrough" do
      model = build_model()
      {predict_fn, params} = init_model(model)
      input = make_input()

      passthrough = Training.remat(predict_fn, policy: :none)

      out_normal = predict_fn.(params, input)
      out_pass = passthrough.(params, input)

      assert_close(out_normal, out_pass)
    end
  end

  describe "checkpointed_grad/4" do
    test "produces same loss as normal forward" do
      model = build_model()
      {predict_fn, params} = init_model(model)
      input = make_input()

      # Normal
      loss_fn = fn p -> predict_fn.(p, input) |> Nx.mean() end
      {loss_normal, _grads_normal} = Nx.Defn.value_and_grad(params, loss_fn)

      # Checkpointed
      {loss_ckpt, _grads_ckpt} = Training.checkpointed_grad(params, predict_fn, input)

      assert_close(loss_normal, loss_ckpt)
    end

    test "produces correct gradients" do
      model = build_model()
      {predict_fn, params} = init_model(model)
      input = make_input()

      # Normal gradients
      loss_fn = fn p -> predict_fn.(p, input) |> Nx.mean() end
      {_loss_normal, grads_normal} = Nx.Defn.value_and_grad(params, loss_fn)

      # Checkpointed gradients
      {_loss_ckpt, grads_ckpt} = Training.checkpointed_grad(params, predict_fn, input)

      assert_grads_close(grads_normal.data, grads_ckpt.data)
    end

    test "works with deeper model" do
      model = build_deep_model()
      {predict_fn, params} = init_model(model)
      input = make_input()

      loss_fn = fn p -> predict_fn.(p, input) |> Nx.mean() end
      {loss_normal, grads_normal} = Nx.Defn.value_and_grad(params, loss_fn)
      {loss_ckpt, grads_ckpt} = Training.checkpointed_grad(params, predict_fn, input)

      assert_close(loss_normal, loss_ckpt)
      assert_grads_close(grads_normal.data, grads_ckpt.data)
    end

    test "works with custom loss function" do
      model = build_model()
      {predict_fn, params} = init_model(model)
      input = make_input()

      custom_loss = fn output -> Nx.sum(Nx.pow(output, 2)) end

      loss_fn = fn p -> predict_fn.(p, input) |> custom_loss.() end
      {loss_normal, grads_normal} = Nx.Defn.value_and_grad(params, loss_fn)

      {loss_ckpt, grads_ckpt} =
        Training.checkpointed_grad(params, predict_fn, input, loss_fn: custom_loss)

      assert_close(loss_normal, loss_ckpt)
      assert_grads_close(grads_normal.data, grads_ckpt.data)
    end

    test "works with targets" do
      model = build_model()
      {predict_fn, params} = init_model(model)
      input = make_input()
      targets = Nx.iota({@batch, 4}, type: :f32)

      supervised_loss = fn output, tgt ->
        Nx.subtract(output, tgt) |> Nx.pow(2) |> Nx.mean()
      end

      loss_fn = fn p ->
        out = predict_fn.(p, input)
        supervised_loss.(out, targets)
      end

      {loss_normal, grads_normal} = Nx.Defn.value_and_grad(params, loss_fn)

      {loss_ckpt, grads_ckpt} =
        Training.checkpointed_grad(params, predict_fn, input,
          loss_fn: supervised_loss,
          targets: targets
        )

      assert_close(loss_normal, loss_ckpt)
      assert_grads_close(grads_normal.data, grads_ckpt.data)
    end
  end

  describe "checkpoint/1" do
    test "executes function and returns result" do
      result = Training.checkpoint(fn -> Nx.tensor([1, 2, 3]) end)
      assert Nx.to_flat_list(result) == [1, 2, 3]
    end
  end

  describe "estimate_memory/3" do
    test "returns memory estimates" do
      model = build_deep_model()
      est = Training.estimate_memory(model, {@batch, @embed_dim})

      assert est.layer_count >= 7
      assert est.normal_bytes > est.checkpointed_bytes
      assert est.savings_ratio > 1.0
      assert est.segment_bytes > 0
      assert est.segment_savings_ratio >= 1.0
    end

    test "deeper models have higher savings ratio" do
      shallow = build_model()
      deep = build_deep_model()

      est_shallow = Training.estimate_memory(shallow, {@batch, @embed_dim})
      est_deep = Training.estimate_memory(deep, {@batch, @embed_dim})

      assert est_deep.savings_ratio > est_shallow.savings_ratio
    end

    test "respects type option" do
      model = build_model()
      est_f32 = Training.estimate_memory(model, {@batch, @embed_dim}, type: {:f, 32})
      est_bf16 = Training.estimate_memory(model, {@batch, @embed_dim}, type: {:bf, 16})

      assert est_f32.normal_bytes == est_bf16.normal_bytes * 2
    end
  end

  describe "format_memory/1" do
    test "formats as human-readable string" do
      est = %{
        layer_count: 10,
        normal_bytes: 40_960,
        checkpointed_bytes: 8_192,
        segment_bytes: 16_384,
        savings_ratio: 5.0,
        segment_savings_ratio: 2.5
      }

      formatted = Training.format_memory(est)
      assert formatted =~ "Layers: 10"
      assert formatted =~ "Normal:"
      assert formatted =~ "Checkpointed:"
      assert formatted =~ "5.0x"
    end
  end

  describe "integration with Edifice architectures" do
    test "checkpointed_grad works with decoder_only" do
      model = Edifice.build(:decoder_only,
        embed_dim: @embed_dim,
        hidden_size: @embed_dim,
        seq_len: 16,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2
      )

      template = %{"state_sequence" => Nx.template({@batch, 16, @embed_dim}, :f32)}
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())
      input = %{"state_sequence" => Nx.iota({@batch, 16, @embed_dim}, type: :f32) |> Nx.divide(128)}

      # Normal
      loss_fn = fn p -> predict_fn.(p, input) |> Nx.mean() end
      {loss_normal, _} = Nx.Defn.value_and_grad(params, loss_fn)

      # Checkpointed
      {loss_ckpt, grads_ckpt} = Training.checkpointed_grad(params, predict_fn, input)

      assert_close(loss_normal, loss_ckpt)
      # Grads should have values (not all zero)
      has_nonzero =
        Enum.any?(flatten_grads(grads_ckpt.data), fn t ->
          Nx.any(Nx.not_equal(t, 0)) |> Nx.to_number() == 1
        end)

      assert has_nonzero, "checkpointed gradients should have non-zero values"
    end

    test "estimate_memory works with edifice model" do
      model = Edifice.build(:decoder_only,
        embed_dim: @embed_dim,
        hidden_size: @embed_dim,
        seq_len: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2
      )

      est = Training.estimate_memory(model, {@batch, 16, @embed_dim})
      formatted = Training.format_memory(est)

      assert est.layer_count > 5
      assert est.savings_ratio > 1.0
      assert is_binary(formatted)
    end
  end

  # Helpers

  defp assert_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < atol, "tensors differ by #{diff}, expected < #{atol}"
  end

  defp assert_grads_close(grads_a, grads_b) when is_map(grads_a) and is_map(grads_b) do
    for key <- Map.keys(grads_a) do
      a = Map.fetch!(grads_a, key)
      b = Map.fetch!(grads_b, key)

      case {a, b} do
        {%Nx.Tensor{}, %Nx.Tensor{}} ->
          assert_close(a, b, atol: 1.0e-4)

        {map_a, map_b} when is_map(map_a) and is_map(map_b) ->
          assert_grads_close(map_a, map_b)
      end
    end
  end

  defp flatten_grads(map) when is_map(map) do
    Enum.flat_map(map, fn
      {_k, %Nx.Tensor{} = t} -> [t]
      {_k, nested} when is_map(nested) -> flatten_grads(nested)
      _ -> []
    end)
  end
end
