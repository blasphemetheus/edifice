# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
Code.require_file("support/whisper_f64_validation.exs", __DIR__)
[reference_path] = System.argv()
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)["enc_final_norm"]
unless Nx.type(reference) == {:f, 64}, do: raise("Use the Rust eager-f64 reference")
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")
model = WhisperF64Validation.model(encoder)
{_init, predict} = Axon.build(model, mode: :inference)
actual = predict.(loaded, %{"mel_spectrogram" => fixture["mel_input"]})
unless Nx.type(actual) == {:f, 64}, do: raise("Expected f64 output")

for {label, left, right} <- [
      {"EXLA f64 vs LibTorch f64", actual, reference},
      {"LibTorch f64 vs original f32 fixture", reference, fixture["expected_encoder_output"]},
      {"EXLA f64 vs original f32 fixture", actual, fixture["expected_encoder_output"]}
    ] do
  error = Nx.abs(Nx.subtract(left, right))

  IO.inspect(%{
    comparison: label,
    max_absolute_error: Nx.to_number(Nx.reduce_max(error)),
    mean_absolute_error: Nx.to_number(Nx.mean(error)),
    values_at_or_above_1e_4: Nx.to_number(Nx.sum(Nx.greater_equal(error, 1.0e-4)))
  })
end
