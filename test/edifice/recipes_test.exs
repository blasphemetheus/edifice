defmodule Edifice.RecipesTest do
  use ExUnit.Case, async: true
  @moduletag :recipes

  alias Edifice.Recipes

  @embed_dim 8
  @num_classes 4
  @batch 2
  @seq_len 16

  defp build_classifier do
    Axon.input("x", shape: {nil, @embed_dim})
    |> Axon.dense(@embed_dim, name: "hidden", activation: :relu)
    |> Axon.dense(@num_classes, name: "out", activation: :softmax)
  end

  defp build_lm do
    Axon.input("x", shape: {nil, @embed_dim})
    |> Axon.dense(@embed_dim, name: "hidden", activation: :relu)
    |> Axon.dense(@num_classes, name: "out", activation: :softmax)
  end

  defp build_encoder do
    Axon.input("x", shape: {nil, @embed_dim})
    |> Axon.dense(@embed_dim, name: "encoder", activation: :relu)
    |> Axon.dense(@embed_dim, name: "projector")
  end

  defp make_classification_data(n \\ 10) do
    Stream.repeatedly(fn ->
      {input, _key} = Nx.Random.uniform(Nx.Random.key(:rand.uniform(10000)), shape: {@batch, @embed_dim})
      # One-hot labels
      labels = Nx.iota({@batch}, type: :s64) |> Nx.remainder(@num_classes)
      target = Nx.equal(Nx.new_axis(labels, 1), Nx.iota({1, @num_classes})) |> Nx.as_type(:f32)
      {%{"x" => input}, target}
    end)
    |> Enum.take(n)
  end

  defp make_contrastive_data(n \\ 10) do
    Stream.repeatedly(fn ->
      # Concatenated positive pairs: first half = view1, second half = view2
      {input, _key} = Nx.Random.uniform(Nx.Random.key(:rand.uniform(10000)), shape: {2 * @batch, @embed_dim})
      # Dummy labels (not used by infonce loss)
      labels = Nx.broadcast(0.0, {2 * @batch, @embed_dim})
      {%{"x" => input}, labels}
    end)
    |> Enum.take(n)
  end

  describe "classify/2" do
    test "returns an Axon.Loop" do
      model = build_classifier()
      loop = Recipes.classify(model, num_classes: @num_classes, log: false)
      assert %Axon.Loop{} = loop
    end

    test "trains on data" do
      model = build_classifier()
      data = make_classification_data(5)
      loop = Recipes.classify(model, num_classes: @num_classes, log: false, patience: 100)
      state = Axon.Loop.run(loop, data, %{}, epochs: 1, iterations: 5)
      assert %Axon.ModelState{} = state
    end

    test "accepts label_smoothing" do
      model = build_classifier()
      loop = Recipes.classify(model, num_classes: @num_classes, label_smoothing: 0.1, log: false)
      assert %Axon.Loop{} = loop
    end

    test "accepts precision option" do
      model = build_classifier()
      loop = Recipes.classify(model, num_classes: @num_classes, precision: :bf16, log: false)
      assert %Axon.Loop{} = loop
    end
  end

  describe "language_model/2" do
    test "returns an Axon.Loop" do
      model = build_lm()
      loop = Recipes.language_model(model, vocab_size: @num_classes, log: false)
      assert %Axon.Loop{} = loop
    end

    test "trains on data" do
      model = build_lm()
      data = make_classification_data(5)
      loop = Recipes.language_model(model, vocab_size: @num_classes, log: false)
      state = Axon.Loop.run(loop, data, %{}, epochs: 1, iterations: 5)
      assert %Axon.ModelState{} = state
    end

    test "respects gradient clipping config" do
      model = build_lm()
      loop = Recipes.language_model(model,
        vocab_size: @num_classes,
        max_grad_norm: 0.5,
        log: false
      )
      assert %Axon.Loop{} = loop
    end
  end

  describe "contrastive/2" do
    test "returns an Axon.Loop" do
      model = build_encoder()
      loop = Recipes.contrastive(model, temperature: 0.1, log: false)
      assert %Axon.Loop{} = loop
    end

    test "trains on paired data" do
      model = build_encoder()
      data = make_contrastive_data(5)
      loop = Recipes.contrastive(model, temperature: 0.1, log: false)
      state = Axon.Loop.run(loop, data, %{}, epochs: 1, iterations: 5)
      assert %Axon.ModelState{} = state
    end
  end

  describe "fine_tune/3" do
    test "returns an Axon.Loop with head_only strategy" do
      model = build_classifier()
      template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
      {init_fn, _} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      loop = Recipes.fine_tune(model, params, strategy: :head_only, log: false)
      assert %Axon.Loop{} = loop
    end

    test "returns an Axon.Loop with full strategy" do
      model = build_classifier()
      template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
      {init_fn, _} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      loop = Recipes.fine_tune(model, params, strategy: :full, log: false)
      assert %Axon.Loop{} = loop
    end

    test "accepts ModelState as base_params" do
      model = build_classifier()
      template = %{"x" => Nx.template({@batch, @embed_dim}, :f32)}
      {init_fn, _} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      loop = Recipes.fine_tune(model, params, strategy: :head_only, log: false)
      assert %Axon.Loop{} = loop
    end
  end

  describe "describe/2" do
    test "returns classify config" do
      desc = Recipes.describe(:classify, num_classes: 10)
      assert desc.loss == :categorical_cross_entropy
      assert desc.optimizer == :adamw
      assert desc.schedule == :cosine_decay
      assert :accuracy in desc.metrics
      assert :early_stop in desc.callbacks
    end

    test "returns language_model config" do
      desc = Recipes.describe(:language_model, vocab_size: 32000)
      assert desc.loss == :categorical_cross_entropy
      assert desc.schedule == :warmup_cosine
      assert desc.max_grad_norm == 1.0
      assert :perplexity in desc.metrics
    end

    test "returns contrastive config" do
      desc = Recipes.describe(:contrastive)
      assert desc.loss == :infonce
      assert desc.temperature == 0.07
    end

    test "returns fine_tune config" do
      desc = Recipes.describe(:fine_tune)
      assert desc.strategy == :head_only
      assert desc.learning_rate == 2.0e-5
      assert desc.warmup_ratio == 0.1
    end

    test "respects custom options" do
      desc = Recipes.describe(:classify, label_smoothing: 0.1)
      assert desc.loss == :categorical_cross_entropy_smoothed

      desc = Recipes.describe(:language_model, max_grad_norm: 0.5)
      assert desc.max_grad_norm == 0.5
    end
  end

  describe "infonce_loss/2" do
    test "returns scalar loss" do
      embeddings = Nx.iota({4, @embed_dim}, type: :f32) |> Nx.divide(32)
      loss = Recipes.infonce_loss(embeddings, 0.07)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0.0
    end

    test "perfect similarity gives low loss" do
      # Two identical pairs
      z = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      embeddings = Nx.concatenate([z, z], axis: 0)
      loss = Recipes.infonce_loss(embeddings, 0.07)
      assert Nx.to_number(loss) < 1.0
    end

    test "orthogonal pairs give higher loss" do
      z1 = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      z2 = Nx.tensor([[0.0, 1.0], [1.0, 0.0]])
      embeddings = Nx.concatenate([z1, z2], axis: 0)
      loss = Recipes.infonce_loss(embeddings, 0.07)
      # Should be higher than perfect similarity
      assert Nx.to_number(loss) > 0.5
    end
  end
end
