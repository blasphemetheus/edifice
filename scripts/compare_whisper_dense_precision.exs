# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
# Pass an eager reference produced by tools/whisper_reference.
[reference_path] = System.argv()
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")

IO.inspect(%{
  backend: :exla_host,
  nx: Application.spec(:nx, :vsn),
  axon: Application.spec(:axon, :vsn),
  input_shape: Nx.shape(fixture["mel_input"]),
  reference: reference_path
})

report = fn label, actual, expected ->
  error = Nx.abs(Nx.subtract(actual, expected))

  IO.inspect(%{
    experiment: label,
    max_absolute_error: Nx.to_number(Nx.reduce_max(error)),
    mean_absolute_error: Nx.to_number(Nx.mean(error)),
    values_above_1e_4: Nx.to_number(Nx.sum(Nx.greater_equal(error, 1.0e-4)))
  })
end

# Both variants start from the same f32 reference inputs and checkpoint weights.
# The f64 variant rounds back to f32 after the dot product and bias addition.
for type <- [:f32, :f64] do
  dense =
    Nx.Defn.jit(fn x, kernel, bias ->
      Axon.Layers.dense(Nx.as_type(x, type), Nx.as_type(kernel, type), Nx.as_type(bias, type))
      |> Nx.as_type(:f32)
    end)

  for layer <- 1..6, suffix <- ["up", "down"] do
    name = "enc_block_#{layer}_ffn_#{suffix}"
    source = if suffix == "up", do: "norm", else: "act"
    params = loaded.data[name]

    actual =
      dense.(reference["enc_block_#{layer}_ffn_#{source}"], params["kernel"], params["bias"])

    report.("isolated #{name} #{type}", actual, reference[name])
  end
end

# Change only selected dense nodes. Their inputs, stored parameters, and outputs
# remain f32; arithmetic inside each selected node uses f64. Other nodes retain
# their original policy. Returning only the final output preserves that test's
# compilation shape instead of exposing intermediate outputs.
policy = Axon.MixedPrecision.create_policy(compute: {:f, 64}, params: {:f, 32}, output: {:f, 32})

experiments = [
  {"baseline", fn _ -> false end},
  {"last_ffn_up", fn name -> name == "enc_block_6_ffn_up" end},
  {"all_ffn_up", &String.ends_with?(&1, "_ffn_up")},
  {"all_ffn_dense", &String.contains?(&1, "_ffn_")},
  {"all_dense", fn _ -> true end}
]

for {label, selected?} <- experiments do
  filter = fn node -> node.op == :dense and selected?.(node.name.(node.op_name, %{})) end
  selected = Enum.count(encoder.nodes, fn {_, node} -> filter.(node) end)
  IO.inspect(%{experiment: label, selected_dense_nodes: selected})
  model = Axon.MixedPrecision.apply_policy(encoder, policy, filter)
  {_init, predict} = Axon.build(model, mode: :inference)
  actual = predict.(loaded, %{"mel_spectrogram" => fixture["mel_input"]})
  report.(label, actual, fixture["expected_encoder_output"])
end
