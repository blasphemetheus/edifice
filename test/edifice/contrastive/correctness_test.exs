defmodule Edifice.Contrastive.CorrectnessTest do
  @moduledoc """
  Correctness tests for contrastive learning architectures.
  Verifies collapse detection, representation properties, and
  encoder consistency.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 4
  @embed 16
  @hidden 8

  # ── Non-Collapse: Different inputs should produce different outputs ──

  # SimCLR is fast enough for the default suite
  @tag timeout: 120_000
  test "simclr does not collapse — different inputs produce different outputs" do
    model = Edifice.build(:simclr, encoder_dim: @embed, projection_dim: @hidden)

    input1 = random_tensor({@batch, @embed}, 42)
    input2 = random_tensor({@batch, @embed}, 99)

    {predict_fn, params} = build_and_init(model, %{"features" => input1})
    out1 = predict_fn.(params, %{"features" => input1})
    out2 = predict_fn.(params, %{"features" => input2})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff > 1.0e-6, "simclr collapsed: outputs identical for different inputs"
  end

  # Barlow Twins and VICReg use cross-correlation matrices — slow on BinaryBackend
  for arch <- [:barlow_twins, :vicreg] do
    @tag timeout: 120_000
    @tag :slow
    test "#{arch} does not collapse — different inputs produce different outputs" do
      model = Edifice.build(unquote(arch), encoder_dim: @embed, projection_dim: @hidden)

      input1 = random_tensor({@batch, @embed}, 42)
      input2 = random_tensor({@batch, @embed}, 99)

      {predict_fn, params} = build_and_init(model, %{"features" => input1})
      out1 = predict_fn.(params, %{"features" => input1})
      out2 = predict_fn.(params, %{"features" => input2})

      diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
      assert diff > 1.0e-6, "#{unquote(arch)} collapsed: outputs identical for different inputs"
    end
  end

  # ── Determinism: Same input -> same output ──────────────────

  @tag timeout: 120_000
  test "simclr is deterministic in inference mode" do
    model = Edifice.build(:simclr, encoder_dim: @embed, projection_dim: @hidden)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"features" => input})

    out1 = predict_fn.(params, %{"features" => input})
    out2 = predict_fn.(params, %{"features" => input})

    diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
    assert diff < 1.0e-6, "simclr not deterministic: diff = #{diff}"
  end

  for arch <- [:barlow_twins, :vicreg] do
    @tag timeout: 120_000
    @tag :slow
    test "#{arch} is deterministic in inference mode" do
      model = Edifice.build(unquote(arch), encoder_dim: @embed, projection_dim: @hidden)

      input = random_tensor({@batch, @embed})
      {predict_fn, params} = build_and_init(model, %{"features" => input})

      out1 = predict_fn.(params, %{"features" => input})
      out2 = predict_fn.(params, %{"features" => input})

      diff = Nx.mean(Nx.abs(Nx.subtract(out1, out2))) |> Nx.to_number()
      assert diff < 1.0e-6, "#{unquote(arch)} not deterministic: diff = #{diff}"
    end
  end

  # ── BYOL Online/Target Asymmetry ──────────────────────────────
  # Online and target encoders should produce different outputs (asymmetric design)

  test "byol online and target produce different representations" do
    {online, target} = Edifice.build(:byol, encoder_dim: @embed, projection_dim: @hidden)

    input = random_tensor({@batch, @embed})

    {online_pred, online_params} = build_and_init(online, %{"features" => input})
    online_out = online_pred.(online_params, %{"features" => input})

    {target_pred, target_params} = build_and_init(target, %{"features" => input})
    target_out = target_pred.(target_params, %{"features" => input})

    assert_finite!(online_out, "byol online")
    assert_finite!(target_out, "byol target")

    # Different init means different params, so outputs should differ
    # (unless architecturally identical AND same random seed, which they're not)
    diff = Nx.mean(Nx.abs(Nx.subtract(online_out, target_out))) |> Nx.to_number()
    assert diff > 1.0e-6, "online and target producing identical output"
  end

  # ── MAE Encoder Output Properties ──────────────────────────────

  test "mae encoder preserves patch count and embeds" do
    num_patches = 4

    {encoder, _decoder} =
      Edifice.build(:mae,
        input_dim: @embed,
        embed_dim: @hidden,
        num_patches: num_patches,
        depth: 1,
        num_heads: 2,
        decoder_depth: 1,
        decoder_num_heads: 2
      )

    input = random_tensor({@batch, num_patches, @embed})
    {predict_fn, params} = build_and_init(encoder, %{"visible_patches" => input})
    output = predict_fn.(params, %{"visible_patches" => input})

    assert_finite!(output, "mae encoder")
    # Output should have same num_patches but embed_dim hidden size
    assert {@batch, ^num_patches, @hidden} = Nx.shape(output)
  end
end
