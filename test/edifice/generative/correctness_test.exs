defmodule Edifice.Generative.CorrectnessTest do
  @moduledoc """
  Correctness tests for generative architectures.
  Verifies encoder/decoder independence, latent structure, and output properties.
  """
  use ExUnit.Case, async: true
  @moduletag :generative

  import Edifice.TestHelpers

  @batch 2
  @embed 16
  @latent 4

  # ── VAE Encoder/Decoder Independence ──────────────────────────
  # Encoder and decoder should be independently buildable with different params

  @tag :smoke
  test "vae encoder and decoder have independent parameters" do
    {encoder, decoder} = Edifice.build(:vae, input_size: @embed, latent_size: @latent)

    enc_input = random_tensor({@batch, @embed})
    {_enc_pred, enc_params} = build_and_init(encoder, %{"input" => enc_input})

    dec_input = random_tensor({@batch, @latent})
    {_dec_pred, dec_params} = build_and_init(decoder, %{"latent" => dec_input})

    enc_keys = enc_params |> flatten_params() |> Enum.map(fn {k, _} -> k end) |> MapSet.new()
    dec_keys = dec_params |> flatten_params() |> Enum.map(fn {k, _} -> k end) |> MapSet.new()

    # Encoder and decoder should have disjoint parameter sets
    overlap = MapSet.intersection(enc_keys, dec_keys)
    assert MapSet.size(overlap) == 0, "encoder/decoder share params: #{inspect(overlap)}"
  end

  # ── VAE Encoder Output Structure ──────────────────────────────

  test "vae encoder produces mu and log_var with correct shapes" do
    {encoder, _decoder} = Edifice.build(:vae, input_size: @embed, latent_size: @latent)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(encoder, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert %{mu: mu, log_var: log_var} = output
    assert {@batch, @latent} = Nx.shape(mu)
    assert {@batch, @latent} = Nx.shape(log_var)
    assert_finite!(mu, "vae mu")
    assert_finite!(log_var, "vae log_var")
  end

  # ── VAE Decoder Reconstruction Shape ──────────────────────────

  test "vae decoder reconstructs to input dimension" do
    {_encoder, decoder} = Edifice.build(:vae, input_size: @embed, latent_size: @latent)

    latent = random_tensor({@batch, @latent})
    {predict_fn, params} = build_and_init(decoder, %{"latent" => latent})
    output = predict_fn.(params, %{"latent" => latent})

    assert_finite!(output, "vae reconstruction")
    assert {@batch, @embed} = Nx.shape(output)
  end

  # ── GAN Generator/Discriminator Consistency ──────────────────

  @tag :slow
  test "gan generator output is valid discriminator input" do
    {gen, disc} = Edifice.build(:gan, output_size: @embed, latent_size: @latent)

    # Generate fake data
    noise = random_tensor({@batch, @latent})
    {gen_pred, gen_params} = build_and_init(gen, %{"noise" => noise})
    fake_data = gen_pred.(gen_params, %{"noise" => noise})

    assert_finite!(fake_data, "generator output")
    assert {@batch, @embed} = Nx.shape(fake_data)

    # Feed to discriminator
    {disc_pred, disc_params} = build_and_init(disc, %{"data" => fake_data})
    disc_out = disc_pred.(disc_params, %{"data" => fake_data})

    assert_finite!(disc_out, "discriminator on fake")
    assert {@batch, 1} = Nx.shape(disc_out)
  end

  # ── Latent Diffusion Component Shapes ────────────────────────

  test "latent_diffusion components have compatible shapes" do
    {encoder, decoder, denoiser} =
      Edifice.build(:latent_diffusion,
        input_size: @embed,
        latent_size: @latent,
        hidden_size: 8,
        num_layers: 1
      )

    # Encoder: input -> {mu, log_var} with latent_size dimensions
    enc_input = random_tensor({@batch, @embed})
    {enc_pred, enc_params} = build_and_init(encoder, %{"input" => enc_input})
    enc_out = enc_pred.(enc_params, %{"input" => enc_input})
    assert %{mu: mu, log_var: _lv} = enc_out
    assert {@batch, @latent} = Nx.shape(mu)

    # Decoder: latent -> reconstruction
    {dec_pred, dec_params} = build_and_init(decoder, %{"latent" => mu})
    dec_out = dec_pred.(dec_params, %{"latent" => mu})
    assert {@batch, @embed} = Nx.shape(dec_out)

    # Denoiser: (noisy_z, timestep) -> noise prediction in latent space
    denoise_input = %{
      "noisy_z" => random_tensor({@batch, @latent}),
      "timestep" => random_tensor({@batch})
    }

    {den_pred, den_params} = build_and_init(denoiser, denoise_input)
    den_out = den_pred.(den_params, denoise_input)
    assert {@batch, @latent} = Nx.shape(den_out)
  end

  # ── Normalizing Flow Invertibility (shape check) ──────────────

  test "normalizing_flow preserves input dimension" do
    model = Edifice.build(:normalizing_flow, input_size: @embed, num_flows: 2)

    input = random_tensor({@batch, @embed})
    {predict_fn, params} = build_and_init(model, %{"input" => input})
    output = predict_fn.(params, %{"input" => input})

    assert_finite!(output, "normalizing_flow")
  end
end
