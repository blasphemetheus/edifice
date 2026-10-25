defmodule Edifice.Pretrained.NumericalValidationTest do
  @moduledoc """
  Forward pass numerical validation against PyTorch reference outputs.

  These tests load pre-generated fixtures (from scripts/generate_numerical_fixtures.py)
  containing deterministic inputs and their expected outputs from PyTorch models,
  then verify that Edifice produces matching outputs within tolerance.

  Prerequisites:
    1. Run `python scripts/generate_numerical_fixtures.py` to generate fixtures
    2. Tests download real HuggingFace weights (~350MB for ViT, ~145MB for Whisper)

  Tagged :external since they require fixture files + HuggingFace downloads.
  Full-size models (ViT-base 224x224, ResNet-50, Whisper) are too slow for BinaryBackend;
  setup explicitly selects EXLA's host backend and compiler. Run with:
  `mix test test/edifice/pretrained/numerical_validation_test.exs --include external`
  """
  use ExUnit.Case, async: false

  @moduletag :external

  @fixtures_dir Path.join([__DIR__, "..", "..", "fixtures", "numerical"])

  setup do
    Nx.default_backend({EXLA.Backend, client: :host})
    Nx.Defn.default_options(compiler: EXLA, client: :host)
    :ok
  end

  defp assert_all_close(actual, expected, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-4)

    assert Nx.shape(actual) == Nx.shape(expected),
           "Shape mismatch: #{inspect(Nx.shape(actual))} vs #{inspect(Nx.shape(expected))}"

    diff = actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff < atol,
           "Max absolute difference #{diff} exceeds tolerance #{atol}. " <>
             "Mean diff: #{actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.mean() |> Nx.to_number()}"
  end

  defp fixture_available?(name) do
    File.exists?(Path.join(@fixtures_dir, name))
  end

  # Build predict_fn and merge HF weights with zero-init defaults for any missing params.
  # Needed for models where HF checkpoint is missing params that Axon expects (e.g. conv bias).
  defp build_with_pretrained(model, hf_state, template) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    full_state = init_fn.(template, Axon.ModelState.empty())
    merged = deep_merge(full_state.data, hf_state.data)
    {merged, predict_fn}
  end

  defp deep_merge(base, overlay) when is_map(base) and is_map(overlay) do
    Map.merge(base, overlay, fn
      _k, base_v, overlay_v when is_map(base_v) and is_map(overlay_v) ->
        deep_merge(base_v, overlay_v)

      _k, _base_v, overlay_v ->
        overlay_v
    end)
  end

  describe "ViT forward pass" do
    @tag timeout: 600_000
    test "matches PyTorch reference output" do
      fixture_file = "vit_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      input = fixture["input"]
      expected_logits = fixture["expected_logits"]

      assert Nx.shape(input) == {1, 3, 224, 224}
      assert Nx.shape(expected_logits) == {1, 1000}

      {model, hf_params} =
        Edifice.Pretrained.from_hub("google/vit-base-patch16-224",
          build_opts: [num_classes: 1000]
        )

      # Pass .data (plain map) to avoid Axon.ModelState Access protocol issue
      {_init_fn, predict_fn} = Axon.build(model, mode: :inference)
      output = predict_fn.(hf_params.data, %{"image" => input})

      assert_all_close(output, expected_logits, atol: 1.0e-4)
    end
  end

  describe "ConvNeXt forward pass" do
    @tag timeout: 600_000
    @tag :skip
    # Skipped: facebook/convnext-tiny-224 only has pytorch_model.bin (no safetensors).
    # ConvNeXt is validated via random-weight tests in architecture_numerical_test.exs.
    test "matches PyTorch reference output" do
      fixture_file = "convnext_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      input = fixture["input"]
      expected_logits = fixture["expected_logits"]

      assert Nx.shape(input) == {1, 3, 224, 224}
      assert Nx.shape(expected_logits) == {1, 1000}

      {model, hf_params} =
        Edifice.Pretrained.from_hub("facebook/convnext-tiny-224",
          build_opts: [num_classes: 1000]
        )

      {_init_fn, predict_fn} = Axon.build(model, mode: :inference)
      output = predict_fn.(hf_params.data, %{"image" => input})

      assert_all_close(output, expected_logits, atol: 1.0e-4)
    end
  end

  describe "ResNet forward pass" do
    @tag timeout: 600_000
    test "matches PyTorch reference output" do
      fixture_file = "resnet_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      input_nchw = fixture["input"]
      expected_logits = fixture["expected_logits"]

      assert Nx.shape(input_nchw) == {1, 3, 224, 224}
      assert Nx.shape(expected_logits) == {1, 1000}

      # Transpose NCHW → NHWC for Edifice ResNet
      input = Nx.transpose(input_nchw, axes: [0, 2, 3, 1])

      {model, hf_params} =
        Edifice.Pretrained.from_hub("microsoft/resnet-50",
          build_opts: [num_classes: 1000]
        )

      # HF ResNet has no conv bias — init fills zero defaults for missing params
      template = %{"input" => Nx.template({1, 224, 224, 3}, :f32)}
      {params, predict_fn} = build_with_pretrained(model, hf_params, template)
      output = predict_fn.(params, %{"input" => input})

      assert_all_close(output, expected_logits, atol: 1.0e-4)
    end
  end

  describe "DETR forward pass" do
    @tag timeout: 600_000
    test "matches PyTorch reference output" do
      fixture_file = "detr_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      input_nchw = fixture["input"]
      expected_class_logits = fixture["expected_class_logits"]
      expected_bbox_pred = fixture["expected_bbox_pred"]

      {1, 3, h, _w} = Nx.shape(input_nchw)

      # Transpose NCHW → NHWC for Edifice
      input = Nx.transpose(input_nchw, axes: [0, 2, 3, 1])

      {model, hf_params} =
        Edifice.Pretrained.from_hub("facebook/detr-resnet-50",
          build_opts: [
            image_size: h,
            backbone: :resnet50,
            norm_position: :post
          ]
        )

      # DETR backbone has no conv bias — init fills zero defaults for missing params
      template = %{"image" => Nx.template({1, h, h, 3}, :f32)}
      {params, predict_fn} = build_with_pretrained(model, hf_params, template)
      output = predict_fn.(params, %{"image" => input})

      assert_all_close(output.class_logits, expected_class_logits, atol: 1.0e-4)
      assert_all_close(output.bbox_pred, expected_bbox_pred, atol: 1.0e-4)
    end
  end

  describe "Whisper encoder forward pass" do
    @tag timeout: 600_000
    test "matches PyTorch reference output" do
      fixture_file = "whisper_encoder_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      mel_input = fixture["mel_input"]
      expected_output = fixture["expected_encoder_output"]

      assert Nx.shape(mel_input) == {1, 80, 3000}

      {encoder, _decoder, hf_params} =
        Edifice.Pretrained.from_hub("openai/whisper-base")

      {_init_fn, predict_fn} = Axon.build(encoder, mode: :inference)
      output = predict_fn.(hf_params.data, %{"mel_spectrogram" => mel_input})

      assert_all_close(output, expected_output, atol: 1.0e-4)
    end
  end
end
