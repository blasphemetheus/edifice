defmodule Edifice.Detection.RTDETRTest do
  use ExUnit.Case, async: true
  @moduletag :detection

  alias Edifice.Detection.RTDETR

  # Minimal config for BinaryBackend: 32x32 image, 1 backbone stage per level,
  # hidden_dim=8, 2 heads, 1 decoder layer, 4 queries
  @batch 2
  @image_size 32
  @hidden_dim 8
  @num_heads 2
  @num_queries 4
  @num_classes 3

  @small_opts [
    image_size: @image_size,
    hidden_dim: @hidden_dim,
    num_heads: @num_heads,
    num_decoder_layers: 1,
    ffn_dim: 16,
    num_queries: @num_queries,
    num_classes: @num_classes,
    backbone_stages: 1,
    dropout: 0.0
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @image_size, @image_size, 3})
    input
  end

  describe "RTDETR.build/1" do
    test "returns an Axon container model" do
      model = RTDETR.build(@small_opts)
      assert is_struct(model, Axon)
    end

    test "forward pass produces correct output shapes" do
      model = RTDETR.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"image" => random_image()})

      assert is_map(output)
      assert Map.has_key?(output, :class_logits)
      assert Map.has_key?(output, :bbox_pred)

      # class_logits: [batch, num_queries, num_classes] (no background class)
      assert Nx.shape(output.class_logits) == {@batch, @num_queries, @num_classes}

      # bbox_pred: [batch, num_queries, 4]
      assert Nx.shape(output.bbox_pred) == {@batch, @num_queries, 4}
    end

    test "outputs contain finite values" do
      model = RTDETR.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"image" => random_image()})

      refute Nx.any(Nx.is_nan(output.class_logits)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_nan(output.bbox_pred)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output.bbox_pred)) |> Nx.to_number() == 1
    end

    test "bbox predictions are in [0, 1] range (sigmoid output)" do
      model = RTDETR.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"image" => random_image()})

      min_val = Nx.reduce_min(output.bbox_pred) |> Nx.to_number()
      max_val = Nx.reduce_max(output.bbox_pred) |> Nx.to_number()

      assert min_val >= 0.0
      assert max_val <= 1.0
    end
  end

  describe "RTDETR.output_size/1" do
    test "returns num_classes + 4 (no background class)" do
      assert RTDETR.output_size(num_classes: 80) == 84
      assert RTDETR.output_size(num_classes: 3) == 7
      assert RTDETR.output_size() == 84
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build RT-DETR via registry" do
      model = Edifice.build(:rt_detr, @small_opts)
      assert is_struct(model, Axon)
    end
  end

  describe "configuration variants" do
    test "works with different num_queries" do
      model = RTDETR.build(Keyword.put(@small_opts, :num_queries, 2))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"image" => random_image()})
      assert Nx.shape(output.class_logits) == {@batch, 2, @num_classes}
      assert Nx.shape(output.bbox_pred) == {@batch, 2, 4}
    end

    test "works with different num_classes" do
      model = RTDETR.build(Keyword.put(@small_opts, :num_classes, 10))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"image" => random_image()})
      assert Nx.shape(output.class_logits) == {@batch, @num_queries, 10}
    end
  end
end
