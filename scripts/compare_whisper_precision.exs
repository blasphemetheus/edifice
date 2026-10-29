# AI-generated diagnostic. See docs/plans/pretrained-validation-fixes.md for attribution.
# devenv shell -- mix run scripts/compare_whisper_precision.exs
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
input = fixture["mel_input"]
expected = fixture["expected_encoder_output"]
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")

for type <- [{:f, 32}, {:f, 64}] do
  policy = Axon.MixedPrecision.create_policy(compute: type, params: type, output: type)
  model = Axon.MixedPrecision.apply_policy(encoder, policy)
  {init, predict} = Axon.build(model, mode: :inference)
  state = init.(%{"mel_spectrogram" => Nx.to_template(input)}, Axon.ModelState.empty())

  data =
    Map.merge(state.data, loaded.data, fn _, defaults, weights -> Map.merge(defaults, weights) end)

  actual = predict.(%{state | data: data}, %{"mel_spectrogram" => input})
  difference = Nx.abs(Nx.subtract(actual, expected))
  index = difference |> Nx.flatten() |> Nx.argmax() |> Nx.to_number()

  IO.inspect(%{
    compute: type,
    max_absolute_error: Nx.to_number(Nx.reduce_max(difference)),
    mean_absolute_error: Nx.to_number(Nx.mean(difference)),
    worst_flat_index: index,
    actual: actual |> Nx.flatten() |> Nx.take(index) |> Nx.to_number(),
    expected: expected |> Nx.flatten() |> Nx.take(index) |> Nx.to_number()
  })
end
