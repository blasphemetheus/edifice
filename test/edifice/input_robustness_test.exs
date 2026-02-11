defmodule Edifice.InputRobustnessTest do
  @moduledoc """
  Input robustness tests — verify architectures handle pathological inputs
  without producing NaN or Inf outputs. Tests zeros, large values, negatives,
  and mixed inputs across representative architectures.

  Strategy D: Catches numerical instability in softmax, layer norm, division,
  log operations, and attention score computation.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 2

  # Shared small dims
  @embed 16
  @hidden 8
  @seq_len 4
  @state_size 4
  @num_layers 1

  # Pathological input generators
  defp zeros(shape), do: Nx.broadcast(Nx.tensor(0.0, type: :f32), shape)
  defp ones(shape), do: Nx.broadcast(Nx.tensor(1.0, type: :f32), shape)
  defp large(shape), do: Nx.broadcast(Nx.tensor(100.0, type: :f32), shape)
  defp negative(shape), do: Nx.broadcast(Nx.tensor(-1.0, type: :f32), shape)
  defp large_negative(shape), do: Nx.broadcast(Nx.tensor(-100.0, type: :f32), shape)

  defp pathological_inputs(shape) do
    [
      {"zeros", zeros(shape)},
      {"ones", ones(shape)},
      {"large(100)", large(shape)},
      {"negative(-1)", negative(shape)},
      {"large_negative(-100)", large_negative(shape)}
    ]
  end

  # Run a model against all pathological inputs and assert finite output
  defp assert_robust(model, input_name, shape) do
    for {label, input} <- pathological_inputs(shape) do
      input_map = %{input_name => input}
      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "#{label}")
    end
  end

  # ── Sequence Models (representative sample) ──────────────────────

  @sequence_opts [
    embed_size: @embed,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: 4,
    num_heads: 2,
    dropout: 0.0
  ]

  # Test a diverse cross-section: SSM, attention, recurrent, linear attention
  @robust_sequence_archs [:mamba, :s4, :lstm, :gqa, :retnet, :fnet, :min_gru, :titans]

  for arch <- @robust_sequence_archs do
    @tag timeout: 120_000
    test "#{arch} handles pathological inputs" do
      model = Edifice.build(unquote(arch), @sequence_opts)
      assert_robust(model, "state_sequence", {@batch, @seq_len, @embed})
    end
  end

  # ── Feedforward ──────────────────────────────────────────────────

  @tag timeout: 120_000
  test "mlp handles pathological inputs" do
    model = Edifice.build(:mlp, input_size: @embed, hidden_sizes: [@hidden])
    assert_robust(model, "input", {@batch, @embed})
  end

  @tag timeout: 120_000
  test "tabnet handles pathological inputs" do
    model = Edifice.build(:tabnet, input_size: @embed, output_size: 4)
    assert_robust(model, "input", {@batch, @embed})
  end

  # ── Vision (representative) ──────────────────────────────────────

  @tag timeout: 120_000
  test "vit handles pathological inputs" do
    model =
      Edifice.build(:vit,
        image_size: 16,
        in_channels: 1,
        patch_size: 4,
        embed_dim: @hidden,
        depth: 1,
        num_heads: 2,
        dropout: 0.0
      )

    assert_robust(model, "image", {@batch, 1, 16, 16})
  end

  # ── Graph (representative) ──────────────────────────────────────

  @tag timeout: 120_000
  test "gcn handles pathological node features" do
    model =
      Edifice.build(:gcn,
        input_dim: @hidden,
        hidden_dim: @hidden,
        num_classes: 4,
        num_layers: @num_layers,
        dropout: 0.0
      )

    adj = random_tensor({@batch, 4, 4})

    for {label, nodes} <- pathological_inputs({@batch, 4, @hidden}) do
      input_map = %{"nodes" => nodes, "adjacency" => adj}
      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "gcn nodes=#{label}")
    end
  end

  @tag timeout: 120_000
  test "gcn handles zero adjacency matrix" do
    model =
      Edifice.build(:gcn,
        input_dim: @hidden,
        hidden_dim: @hidden,
        num_classes: 4,
        num_layers: @num_layers,
        dropout: 0.0
      )

    nodes = random_tensor({@batch, 4, @hidden})
    zero_adj = zeros({@batch, 4, 4})
    input_map = %{"nodes" => nodes, "adjacency" => zero_adj}
    {predict_fn, params} = build_and_init(model, input_map)
    output = predict_fn.(params, input_map)
    assert_finite!(output, "gcn zero_adjacency")
  end

  @tag timeout: 120_000
  test "gat handles pathological inputs" do
    model =
      Edifice.build(:gat,
        input_dim: @hidden,
        hidden_dim: @hidden,
        num_classes: 4,
        num_layers: @num_layers,
        num_heads: 2,
        dropout: 0.0
      )

    adj = random_tensor({@batch, 4, 4})

    for {label, nodes} <- pathological_inputs({@batch, 4, @hidden}) do
      input_map = %{"nodes" => nodes, "adjacency" => adj}
      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "gat nodes=#{label}")
    end
  end

  # ── Energy / Dynamic ──────────────────────────────────────────

  @tag timeout: 120_000
  test "ebm handles pathological inputs" do
    model = Edifice.build(:ebm, input_size: @embed)
    assert_robust(model, "input", {@batch, @embed})
  end

  @tag timeout: 120_000
  test "hopfield handles pathological inputs" do
    model = Edifice.build(:hopfield, input_dim: @embed)
    assert_robust(model, "input", {@batch, @embed})
  end

  # ── Probabilistic ──────────────────────────────────────────────

  @tag timeout: 120_000
  test "bayesian handles pathological inputs" do
    model = Edifice.build(:bayesian, input_size: @embed, output_size: 4)
    assert_robust(model, "input", {@batch, @embed})
  end

  # ── Generative (non-container components) ──────────────────────

  @tag timeout: 120_000
  test "vae decoder handles pathological latent inputs" do
    {_enc, decoder} = Edifice.build(:vae, input_size: @embed, latent_size: 4)
    assert_robust(decoder, "latent", {@batch, 4})
  end

  @tag timeout: 120_000
  test "gan generator handles pathological noise" do
    {gen, _disc} = Edifice.build(:gan, output_size: @embed, latent_size: 4)
    assert_robust(gen, "noise", {@batch, 4})
  end

  # ── Diffusion family ────────────────────────────────────────────

  @tag timeout: 120_000
  test "dit handles pathological inputs" do
    model =
      Edifice.build(:dit,
        input_dim: @embed,
        hidden_size: @hidden,
        depth: 1,
        num_heads: 2,
        dropout: 0.0
      )

    for {label, noisy} <- pathological_inputs({@batch, @embed}) do
      input_map = %{
        "noisy_input" => noisy,
        "timestep" => random_tensor({@batch})
      }

      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "dit input=#{label}")
    end
  end

  @tag timeout: 120_000
  test "score_sde handles pathological inputs" do
    model =
      Edifice.build(:score_sde, input_dim: @embed, hidden_size: @hidden, num_layers: @num_layers)

    for {label, noisy} <- pathological_inputs({@batch, @embed}) do
      input_map = %{
        "noisy_input" => noisy,
        "timestep" => random_tensor({@batch})
      }

      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "score_sde input=#{label}")
    end
  end

  # ── Contrastive ──────────────────────────────────────────────────

  @tag timeout: 120_000
  test "simclr handles pathological inputs" do
    model = Edifice.build(:simclr, encoder_dim: @embed, projection_dim: @hidden)
    assert_robust(model, "features", {@batch, @embed})
  end

  # ── Meta / PEFT ──────────────────────────────────────────────────

  @tag timeout: 120_000
  test "lora handles pathological inputs" do
    model = Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 4)
    assert_robust(model, "input", {@batch, @embed})
  end

  @tag timeout: 120_000
  test "adapter handles pathological inputs" do
    model = Edifice.build(:adapter, hidden_size: @hidden)
    assert_robust(model, "input", {@batch, @hidden})
  end

  # ── Memory ──────────────────────────────────────────────────────

  @tag timeout: 120_000
  test "ntm handles pathological inputs" do
    model =
      Edifice.build(:ntm,
        input_size: @embed,
        output_size: @hidden,
        memory_size: 4,
        memory_dim: 4,
        num_heads: 1
      )

    for {label, input} <- pathological_inputs({@batch, @embed}) do
      memory = random_tensor({@batch, 4, 4})
      input_map = %{"input" => input, "memory" => memory}
      {predict_fn, params} = build_and_init(model, input_map)
      output = predict_fn.(params, input_map)
      assert_finite!(output, "ntm input=#{label}")
    end
  end

  @tag timeout: 120_000
  test "ntm handles zero memory" do
    model =
      Edifice.build(:ntm,
        input_size: @embed,
        output_size: @hidden,
        memory_size: 4,
        memory_dim: 4,
        num_heads: 1
      )

    input = random_tensor({@batch, @embed})
    zero_memory = zeros({@batch, 4, 4})
    input_map = %{"input" => input, "memory" => zero_memory}
    {predict_fn, params} = build_and_init(model, input_map)
    output = predict_fn.(params, input_map)
    assert_finite!(output, "ntm zero_memory")
  end

  # ── Neuromorphic ──────────────────────────────────────────────────

  @tag timeout: 120_000
  test "ann2snn handles pathological inputs" do
    model = Edifice.build(:ann2snn, input_size: @embed, output_size: 4)
    assert_robust(model, "input", {@batch, @embed})
  end

  # ── Sets ──────────────────────────────────────────────────────────

  @tag timeout: 120_000
  test "deep_sets handles pathological inputs" do
    model = Edifice.build(:deep_sets, input_dim: 3, output_dim: 4)
    assert_robust(model, "input", {@batch, 8, 3})
  end
end
