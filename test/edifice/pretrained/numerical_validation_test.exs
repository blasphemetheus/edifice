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
  """
  use ExUnit.Case, async: false

  @moduletag :external

  @fixtures_dir Path.join([__DIR__, "..", "..", "fixtures", "numerical"])

  defp assert_all_close(actual, expected, opts \\ []) do
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

  describe "ViT forward pass" do
    @tag timeout: 120_000
    test "matches PyTorch reference output" do
      fixture_file = "vit_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      # Load fixture
      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      input = fixture["input"]
      expected_logits = fixture["expected_logits"]

      assert Nx.shape(input) == {1, 3, 224, 224}
      assert Nx.shape(expected_logits) == {1, 1000}

      # Load real pretrained weights
      {model, params} =
        Edifice.Pretrained.from_hub("google/vit-base-patch16-224",
          build_opts: [num_classes: 1000]
        )

      # Build and run forward pass
      {predict_fn, _} = Axon.build(model, mode: :inference)
      output = predict_fn.(params, %{"image" => input})

      # Compare against PyTorch reference
      assert_all_close(output, expected_logits, atol: 1.0e-4)
    end
  end

  describe "Whisper encoder forward pass" do
    @tag timeout: 120_000
    test "matches PyTorch reference output" do
      fixture_file = "whisper_encoder_reference.safetensors"

      unless fixture_available?(fixture_file) do
        flunk(
          "Fixture #{fixture_file} not found. " <>
            "Run: python scripts/generate_numerical_fixtures.py"
        )
      end

      # Load fixture
      fixture_path = Path.join(@fixtures_dir, fixture_file)
      fixture = Safetensors.read!(fixture_path)

      mel_input = fixture["mel_input"]
      expected_output = fixture["expected_encoder_output"]

      assert Nx.shape(mel_input) == {1, 80, 100}

      # Load real pretrained weights
      {encoder, _decoder, params} =
        Edifice.Pretrained.from_hub("openai/whisper-base")

      # Build encoder and run forward pass
      {predict_fn, _} = Axon.build(encoder, mode: :inference)
      output = predict_fn.(params, %{"mel" => mel_input})

      # Compare against PyTorch reference
      assert_all_close(output, expected_output, atol: 1.0e-4)
    end
  end
end
