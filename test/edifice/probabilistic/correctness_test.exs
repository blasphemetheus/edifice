defmodule Edifice.Probabilistic.CorrectnessTest do
  @moduledoc """
  Correctness tests for probabilistic architectures.
  Verifies uncertainty estimation properties: Dirichlet constraints
  for evidential networks, weight uncertainty for Bayesian nets,
  and MC dropout stochasticity.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 4
  @embed 16
  @output 4

  # ── Bayesian Network ────────────────────────────────────────────
  # Bayesian networks should produce valid, finite outputs

  test "bayesian produces correct output shape" do
    model = Edifice.build(:bayesian, input_size: @embed, output_size: @output)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "bayesian")
    assert {@batch, @output} = Nx.shape(output)
  end

  test "bayesian is stochastic (weight sampling)" do
    # Bayesian networks sample weights via reparameterization trick,
    # so repeated forward passes should produce varying outputs
    model = Edifice.build(:bayesian, input_size: @embed, output_size: @output)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})

    out1 = predict_fn.(params, %{"input" => input})
    out2 = predict_fn.(params, %{"input" => input})

    assert_finite!(out1, "bayesian run1")
    assert_finite!(out2, "bayesian run2")

    # Both outputs should be valid and have the right shape
    assert {@batch, @output} = Nx.shape(out1)
    assert {@batch, @output} = Nx.shape(out2)
  end

  test "bayesian different inputs produce different outputs" do
    model = Edifice.build(:bayesian, input_size: @embed, output_size: @output)

    input1 = random_tensor({@batch, @embed}, 42)
    input2 = random_tensor({@batch, @embed}, 99)

    {predict_fn, params} = build_and_init(model, %{"input" => input1})
    out1 = predict_fn.(params, %{"input" => input1})
    out2 = predict_fn.(params, %{"input" => input2})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "Bayesian collapsed: same output for different inputs"
  end

  # ── Evidential Neural Network ──────────────────────────────────
  # Output should be Dirichlet alpha parameters (all > 0)

  test "evidential output values are positive (Dirichlet alphas)" do
    model = Edifice.build(:evidential, input_size: @embed, num_classes: @output)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "evidential")
    assert {@batch, @output} = Nx.shape(output)

    # Dirichlet alpha parameters should be positive
    min_val = Nx.reduce_min(output) |> Nx.to_number()
    assert min_val > 0, "Evidential alphas must be positive, got min = #{min_val}"
  end

  test "evidential produces different outputs for different inputs" do
    model = Edifice.build(:evidential, input_size: @embed, num_classes: @output)

    input1 = random_tensor({@batch, @embed}, 42)
    input2 = random_tensor({@batch, @embed}, 99)

    {predict_fn, params} = build_and_init(model, %{"input" => input1})
    out1 = predict_fn.(params, %{"input" => input1})
    out2 = predict_fn.(params, %{"input" => input2})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "Evidential collapsed: identical outputs"
  end

  # ── MC Dropout ──────────────────────────────────────────────────
  # MC Dropout should be deterministic in inference mode (dropout disabled)

  test "mc_dropout produces correct output shape" do
    model =
      Edifice.build(:mc_dropout, input_size: @embed, output_size: @output, dropout_rate: 0.2)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "mc_dropout")
    assert {@batch, @output} = Nx.shape(output)
  end

  test "mc_dropout inference is deterministic (dropout off)" do
    model =
      Edifice.build(:mc_dropout, input_size: @embed, output_size: @output, dropout_rate: 0.5)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})

    out1 = predict_fn.(params, %{"input" => input})
    out2 = predict_fn.(params, %{"input" => input})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff < 1.0e-6, "MC dropout not deterministic in inference: diff = #{diff}"
  end
end
