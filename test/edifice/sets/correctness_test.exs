defmodule Edifice.Sets.CorrectnessTest do
  @moduledoc """
  Correctness tests for set-processing architectures.
  Verifies the fundamental property: permutation invariance —
  shuffling set elements should not change the output.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 2
  @set_size 8
  @input_dim 6
  @output_dim 4

  # ── Permutation Invariance ──────────────────────────────────────
  # f(π(X)) = f(X) for any permutation π of set elements

  for arch <- [:deep_sets, :pointnet] do
    @tag timeout: 120_000
    test "#{arch} is permutation invariant" do
      opts =
        case unquote(arch) do
          :deep_sets -> [input_dim: @input_dim, output_dim: @output_dim]
          :pointnet -> [input_dim: @input_dim, num_classes: @output_dim]
        end

      model = Edifice.build(unquote(arch), opts)

      input = random_tensor({@batch, @set_size, @input_dim})

      {predict_fn, params} = build_and_init(model, %{"input" => input})
      out_orig = predict_fn.(params, %{"input" => input})

      # Permute set elements (axis 1): reverse the order
      perm = Enum.to_list((@set_size - 1)..0//-1)
      input_perm = Nx.take(input, Nx.tensor(perm), axis: 1)

      out_perm = predict_fn.(params, %{"input" => input_perm})

      # Outputs should be identical (within floating point tolerance)
      diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()

      assert diff < 0.01,
             "#{unquote(arch)} not permutation invariant: mean diff = #{diff}"
    end
  end

  # ── Permutation Invariance with Random Shuffle ──────────────────
  # Use a non-trivial permutation (not just reversal)

  for arch <- [:deep_sets, :pointnet] do
    @tag timeout: 120_000
    test "#{arch} invariance holds for arbitrary permutation" do
      opts =
        case unquote(arch) do
          :deep_sets -> [input_dim: @input_dim, output_dim: @output_dim]
          :pointnet -> [input_dim: @input_dim, num_classes: @output_dim]
        end

      model = Edifice.build(unquote(arch), opts)

      input = random_tensor({@batch, @set_size, @input_dim}, 123)

      {predict_fn, params} = build_and_init(model, %{"input" => input})
      out_orig = predict_fn.(params, %{"input" => input})

      # Arbitrary permutation: [3, 7, 1, 5, 0, 4, 6, 2]
      perm = [3, 7, 1, 5, 0, 4, 6, 2]
      input_perm = Nx.take(input, Nx.tensor(perm), axis: 1)

      out_perm = predict_fn.(params, %{"input" => input_perm})

      diff = Nx.mean(Nx.abs(Nx.subtract(out_orig, out_perm))) |> Nx.to_number()

      assert diff < 0.01,
             "#{unquote(arch)} not invariant for arbitrary perm: mean diff = #{diff}"
    end
  end

  # ── Determinism ──────────────────────────────────────────────────

  for arch <- [:deep_sets, :pointnet] do
    @tag timeout: 120_000
    test "#{arch} is deterministic in inference mode" do
      opts =
        case unquote(arch) do
          :deep_sets -> [input_dim: @input_dim, output_dim: @output_dim]
          :pointnet -> [input_dim: @input_dim, num_classes: @output_dim]
        end

      model = Edifice.build(unquote(arch), opts)

      input = random_tensor({@batch, @set_size, @input_dim})
      {predict_fn, params} = build_and_init(model, %{"input" => input})

      out1 = predict_fn.(params, %{"input" => input})
      out2 = predict_fn.(params, %{"input" => input})

      diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
      assert diff < 1.0e-6, "#{unquote(arch)} not deterministic: diff = #{diff}"
    end
  end

  # ── Output Shape ────────────────────────────────────────────────

  test "deep_sets output shape is {batch, output_dim}" do
    model = Edifice.build(:deep_sets, input_dim: @input_dim, output_dim: @output_dim)

    input = random_tensor({@batch, @set_size, @input_dim})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "deep_sets")
    assert {@batch, @output_dim} = Nx.shape(output)
  end

  @tag timeout: 120_000
  test "pointnet output shape is {batch, num_classes}" do
    model = Edifice.build(:pointnet, input_dim: @input_dim, num_classes: @output_dim)

    input = random_tensor({@batch, @set_size, @input_dim})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "pointnet")
    assert {@batch, @output_dim} = Nx.shape(output)
  end

  # ── Set Size Independence ──────────────────────────────────────
  # Models should handle different set sizes (aggregation is size-agnostic)

  @tag timeout: 120_000
  test "deep_sets handles different set sizes" do
    model = Edifice.build(:deep_sets, input_dim: @input_dim, output_dim: @output_dim)

    for set_size <- [4, 8, 16] do
      input = random_tensor({@batch, set_size, @input_dim})
      {predict_fn, params} = build_and_init(model, %{"input" => input})
      output = predict_fn.(params, %{"input" => input})

      assert_finite!(output, "deep_sets set_size=#{set_size}")
      assert {@batch, @output_dim} = Nx.shape(output)
    end
  end
end
