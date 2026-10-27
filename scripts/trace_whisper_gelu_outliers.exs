# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
[reference_path | options] = System.argv()
native? = options == ["native"]
unless options in [[], ["native"]], do: raise("usage: SCRIPT REFERENCE [native]")
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")
names = Map.new(encoder.nodes, fn {id, node} -> {node.name.(node.op_name, %{}), id} end)
expected = fixture["expected_encoder_output"]

reference_error =
  reference["enc_final_norm"]
  |> Nx.subtract(expected)
  |> Nx.abs()
  |> Nx.reduce_max()
  |> Nx.to_number()

if reference_error != 0, do: raise("Use the exact Rust SDPA reference")

IO.inspect(%{
  backend: :exla_host,
  nx: Application.spec(:nx, :vsn),
  axon: Application.spec(:axon, :vsn),
  reference: reference_path
})

report = fn label, actual ->
  error = Nx.abs(Nx.subtract(actual, expected))
  index = error |> Nx.flatten() |> Nx.argmax() |> Nx.to_number()
  at = fn tensor -> tensor |> Nx.flatten() |> Nx.take(index) |> Nx.to_number() end

  IO.inspect(%{
    experiment: label,
    max_absolute_error: Nx.to_number(Nx.reduce_max(error)),
    mean_absolute_error: Nx.to_number(Nx.mean(error)),
    values_at_or_above_1e_4: Nx.to_number(Nx.sum(Nx.greater_equal(error, 1.0e-4))),
    worst_frame: div(index, 512),
    worst_channel: rem(index, 512),
    actual: at.(actual),
    expected: at.(expected)
  })
end

run = fn model, boundary ->
  inputs = %{"mel_spectrogram" => fixture["mel_input"]}

  {model, inputs} =
    if boundary do
      tensor = Map.fetch!(reference, boundary)
      id = Map.fetch!(names, boundary)
      input = Axon.input("reference_input", shape: Nx.shape(tensor))
      node = %{Map.fetch!(input.nodes, input.output) | id: id}

      {%{model | nodes: Map.put(model.nodes, id, node)},
       Map.put(inputs, "reference_input", tensor)}
    else
      {model, inputs}
    end

  {_init, predict} = Axon.build(model, mode: :inference)
  predict.(loaded, inputs)
end

replace_gelu = fn include_stem ->
  nodes =
    Map.new(encoder.nodes, fn {id, node} ->
      name = node.name.(node.op_name, %{})

      if String.ends_with?(name, "_ffn_act") or
           (include_stem and name in ["enc_conv1_act", "enc_conv2_act"]) do
        operation = fn x, _opts ->
          Nx.multiply(Nx.multiply(x, 0.5), Nx.erfc(Nx.multiply(x, -0.7071067811865476)))
        end

        {id, %{node | op: operation}}
      else
        {id, node}
      end
    end)

  %{encoder | nodes: nodes}
end

report.("baseline", run.(encoder, nil))
unless native?, do: report.("ffn_gelu_erfc", run.(replace_gelu.(false), nil))
model = if native?, do: encoder, else: replace_gelu.(true)
label = if native?, do: "native", else: "all_gelu_erfc"
unless native?, do: report.(label, run.(model, nil))

boundaries =
  ["enc_conv1_act", "enc_conv2_act", "enc_pos_embed"] ++
    for(layer <- 1..6, part <- ["attn", "ffn"], do: "enc_block_#{layer}_#{part}_residual")

for boundary <- boundaries do
  report.("#{label} reference #{boundary}", run.(model, boundary))
end
