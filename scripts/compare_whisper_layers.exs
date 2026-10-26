# AI-generated diagnostic; see tools/whisper_reference/README.md.
[reference_path] = System.argv()
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)
input = fixture["mel_input"]
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")
names = Map.new(encoder.nodes, fn {id, node} -> {node.name.(node.op_name, %{}), id} end)

outputs =
  Map.new(reference, fn {name, _} -> {name, %{encoder | output: Map.fetch!(names, name)}} end)

{init, predict} = Axon.build(Axon.container(outputs), mode: :inference)
state = init.(%{"mel_spectrogram" => Nx.to_template(input)}, Axon.ModelState.empty())

data =
  Map.merge(state.data, loaded.data, fn _, defaults, weights -> Map.merge(defaults, weights) end)

actual = predict.(%{state | data: data}, %{"mel_spectrogram" => input})

for name <- Enum.sort(Map.keys(reference)) do
  difference = Nx.abs(Nx.subtract(actual[name], reference[name]))

  IO.inspect(%{
    layer: name,
    max_absolute_error: Nx.to_number(Nx.reduce_max(difference)),
    mean_absolute_error: Nx.to_number(Nx.mean(difference))
  })
end

# Feed exactly the reference input to each normalization, separating local
# arithmetic differences from errors accumulated by earlier layers.
normalize = Nx.Defn.jit(&Axon.Layers.layer_norm/3)

for layer <- 1..6, part <- [:attn, :ffn] do
  name = "enc_block_#{layer}_#{part}_norm"

  source =
    case {part, layer} do
      {:attn, 1} -> "enc_pos_embed"
      {:attn, _} -> "enc_block_#{layer - 1}_ffn_residual"
      {:ffn, _} -> "enc_block_#{layer}_attn_residual"
    end

  params = loaded.data[name]
  local = normalize.(reference[source], params["gamma"], params["beta"])
  error = Nx.abs(Nx.subtract(local, reference[name]))

  IO.inspect(%{
    isolated_normalization: name,
    max_absolute_error: Nx.to_number(Nx.reduce_max(error)),
    mean_absolute_error: Nx.to_number(Nx.mean(error))
  })
end
