defmodule Edifice.Sets.CorrectnessTest do
  @moduledoc """
  Correctness tests for set-processing architectures.
  Verifies the fundamental property: permutation invariance —
  shuffling set elements should not change the output.
  """
  use ExUnit.Case, async: true
  @moduletag :sets

  import Edifice.TestHelpers

  @batch 2
  @set_size 8
  @input_dim 6
  @output_dim 4

  # Helper to get build opts per arch
  defp build_opts(:deep_sets), do: [input_dim: @input_dim, output_dim: @output_dim]
  defp build_opts(:pointnet), do: [input_dim: @input_dim, num_classes: @output_dim]

  # ── DeepSets Permutation Invariance (fast) ─────────────────────

  @tag timeout: 120_000
  test "deep_sets is permutation invariant" do
    model = Edifice.build(:deep_sets, build_opts(:deep_sets))
    input = random_tensor({@batch, @set_size, @input_dim})

    {predict_fn, params} = build_and_init(model, %{"input" => input})
    out_orig = predict_fn.(params, %{"input" => input})

    perm = Enum.to_list((@set_size - 1)..0//-1)
    input_perm = Nx.take(input, Nx.tensor(perm), axis: 1)
    out_perm = predict_fn.(params, %{"input" => input_perm})

    diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()
    assert diff < 0.01, "deep_sets not permutation invariant: mean diff = #{diff}"
  end

  @tag timeout: 120_000
  test "deep_sets invariance holds for arbitrary permutation" do
    model = Edifice.build(:deep_sets, build_opts(:deep_sets))
    input = random_tensor({@batch, @set_size, @input_dim}, 123)

    {predict_fn, params} = build_and_init(model, %{"input" => input})
    out_orig = predict_fn.(params, %{"input" => input})

    perm = [3, 7, 1, 5, 0, 4, 6, 2]
    input_perm = Nx.take(input, Nx.tensor(perm), axis: 1)
    out_perm = predict_fn.(params, %{"input" => input_perm})

    diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()
    assert diff < 0.01, "deep_sets not invariant for arbitrary perm: mean diff = #{diff}"
  end

  @tag timeout: 120_000
  test "deep_sets is deterministic in inference mode" do
    model = Edifice.build(:deep_sets, build_opts(:deep_sets))
    input = random_tensor({@batch, @set_size, @input_dim})
    {predict_fn, params} = build_and_init(model, %{"input" => input})

    out1 = predict_fn.(params, %{"input" => input})
    out2 = predict_fn.(params, %{"input" => input})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff < 1.0e-6, "deep_sets not deterministic: diff = #{diff}"
  end

  test "deep_sets output shape is {batch, output_dim}" do
    model = Edifice.build(:deep_sets, build_opts(:deep_sets))
    input = random_tensor({@batch, @set_size, @input_dim})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "deep_sets")
    assert {@batch, @output_dim} = Nx.shape(output)
  end

  @tag timeout: 120_000
  test "deep_sets handles different set sizes" do
    model = Edifice.build(:deep_sets, build_opts(:deep_sets))

    for set_size <- [4, 8, 16] do
      input = random_tensor({@batch, set_size, @input_dim})
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})

      assert_finite!(output, "deep_sets set_size=#{set_size}")
      assert {@batch, @output_dim} = Nx.shape(output)
    end
  end

  # ── PointNet Tests (slow — large hidden dims on BinaryBackend) ──

  @tag :slow
  @tag timeout: 120_000
  test "pointnet is permutation invariant" do
    model = Edifice.build(:pointnet, build_opts(:pointnet))
    input = random_tensor({@batch, @set_size, @input_dim})

    {predict_fn, params} = build_and_init(model, %{"input" => input})
    out_orig = predict_fn.(params, %{"input" => input})

    perm = Enum.to_list((@set_size - 1)..0//-1)
    input_perm = Nx.take(input, Nx.tensor(perm), axis: 1)
    out_perm = predict_fn.(params, %{"input" => input_perm})

    diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()
    assert diff < 0.01, "pointnet not permutation invariant: mean diff = #{diff}"
  end

  @tag :slow
  @tag timeout: 120_000
  test "pointnet invariance holds for arbitrary permutation" do
    model = Edifice.build(:pointnet, build_opts(:pointnet))
    input = random_tensor({@batch, @set_size, @input_dim}, 123)

    {predict_fn, params} = build_and_init(model, %{"input" => input})
    out_orig = predict_fn.(params, %{"input" => input})

    perm = [3, 7, 1, 5, 0, 4, 6, 2]
    input_perm = Nx.take(input, Nx.tensor(perm), axis: 1)
    out_perm = predict_fn.(params, %{"input" => input_perm})

    diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()
    assert diff < 0.01, "pointnet not invariant for arbitrary perm: mean diff = #{diff}"
  end

  @tag :slow
  @tag timeout: 120_000
  test "pointnet is deterministic in inference mode" do
    model = Edifice.build(:pointnet, build_opts(:pointnet))
    input = random_tensor({@batch, @set_size, @input_dim})
    {predict_fn, params} = build_and_init(model, %{"input" => input})

    out1 = predict_fn.(params, %{"input" => input})
    out2 = predict_fn.(params, %{"input" => input})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff < 1.0e-6, "pointnet not deterministic: diff = #{diff}"
  end

  @tag :slow
  @tag timeout: 120_000
  test "pointnet output shape is {batch, num_classes}" do
    model = Edifice.build(:pointnet, build_opts(:pointnet))
    input = random_tensor({@batch, @set_size, @input_dim})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "pointnet")
    assert {@batch, @output_dim} = Nx.shape(output)
  end
end
