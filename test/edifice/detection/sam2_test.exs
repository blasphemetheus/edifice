defmodule Edifice.Detection.SAM2Test do
  use ExUnit.Case, async: true

  alias Edifice.Detection.SAM2

  # Minimal config for BinaryBackend: 16x16 image, backbone_stages=2 (stride 4),
  # hidden_dim=8, 2 heads, 1 decoder layer, 2 point prompts
  @batch 2
  @image_size 16
  @hidden_dim 8
  @num_heads 2
  @max_points 2
  @num_multimask 3

  @small_opts [
    image_size: @image_size,
    hidden_dim: @hidden_dim,
    num_heads: @num_heads,
    num_decoder_layers: 1,
    ffn_dim: 16,
    num_multimask_outputs: @num_multimask,
    max_points: @max_points,
    backbone_stages: 2,
    pe_dim: 4,
    dropout: 0.0
  ]

  defp random_inputs do
    key = Nx.Random.key(42)
    {image, key} = Nx.Random.uniform(key, shape: {@batch, @image_size, @image_size, 3})
    {points, key} = Nx.Random.uniform(key, shape: {@batch, @max_points, 2})
    {labels, _key} = Nx.Random.uniform(key, shape: {@batch, @max_points})
    # Binarize labels to 0.0 or 1.0
    labels = Nx.round(labels)
    %{"image" => image, "points" => points, "labels" => labels}
  end

  defp build_and_run(opts) do
    model = SAM2.build(opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    template = %{
      "image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32),
      "points" => Nx.template({@batch, @max_points, 2}, :f32),
      "labels" => Nx.template({@batch, @max_points}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_inputs())
    {model, output}
  end

  describe "SAM2.build/1" do
    test "returns an Axon container model" do
      model = SAM2.build(@small_opts)
      assert is_struct(model, Axon)
    end

    test "forward pass produces correct output shapes" do
      {_model, output} = build_and_run(@small_opts)

      assert is_map(output)
      assert Map.has_key?(output, :masks)
      assert Map.has_key?(output, :iou_scores)

      # masks: [batch, num_masks, mask_h, mask_w]
      # With backbone_stages=2 (stride 4), 4x upscale: mask = image_size
      num_masks = @num_multimask + 1
      assert Nx.shape(output.masks) == {@batch, num_masks, @image_size, @image_size}

      # iou_scores: [batch, num_masks]
      assert Nx.shape(output.iou_scores) == {@batch, num_masks}
    end

    test "outputs contain finite values" do
      {_model, output} = build_and_run(@small_opts)

      refute Nx.any(Nx.is_nan(output.masks)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_nan(output.iou_scores)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output.masks)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output.iou_scores)) |> Nx.to_number() == 1
    end
  end

  describe "SAM2.output_size/1" do
    test "returns num_multimask_outputs + 1" do
      assert SAM2.output_size(num_multimask_outputs: 3) == 4
      assert SAM2.output_size(num_multimask_outputs: 1) == 2
      assert SAM2.output_size() == 4
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build SAM 2 via registry" do
      model = Edifice.build(:sam2, @small_opts)
      assert is_struct(model, Axon)
    end
  end

  describe "configuration variants" do
    test "works with different num_multimask_outputs" do
      opts = Keyword.put(@small_opts, :num_multimask_outputs, 1)
      {_model, output} = build_and_run(opts)

      # 1 multimask + 1 single = 2 masks
      assert Nx.shape(output.masks) == {@batch, 2, @image_size, @image_size}
      assert Nx.shape(output.iou_scores) == {@batch, 2}
    end

    test "works with different max_points" do
      opts = Keyword.put(@small_opts, :max_points, 4)
      model = SAM2.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "image" => Nx.template({@batch, @image_size, @image_size, 3}, :f32),
        "points" => Nx.template({@batch, 4, 2}, :f32),
        "labels" => Nx.template({@batch, 4}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {image, key} = Nx.Random.uniform(key, shape: {@batch, @image_size, @image_size, 3})
      {points, key} = Nx.Random.uniform(key, shape: {@batch, 4, 2})
      {labels, _key} = Nx.Random.uniform(key, shape: {@batch, 4})

      output = predict_fn.(params, %{"image" => image, "points" => points, "labels" => labels})
      num_masks = @num_multimask + 1
      assert Nx.shape(output.masks) == {@batch, num_masks, @image_size, @image_size}
    end
  end
end
