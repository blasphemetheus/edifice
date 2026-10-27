# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
[reference_path] = System.argv()
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")
names = Map.new(encoder.nodes, fn {id, node} -> {node.name.(node.op_name, %{}), id} end)
expected = fixture["expected_encoder_output"]

unless Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(reference["enc_final_norm"], expected)))) ==
         0 do
  raise "Use the Rust SDPA reference matching the original fixture exactly"
end

IO.inspect(%{
  backend: :exla_host,
  nx: Application.spec(:nx, :vsn),
  axon: Application.spec(:axon, :vsn),
  reference: reference_path
})

report = fn label, actual, expected ->
  error = Nx.abs(Nx.subtract(actual, expected))

  IO.inspect(%{
    experiment: label,
    max_absolute_error: Nx.to_number(Nx.reduce_max(error)),
    mean_absolute_error: Nx.to_number(Nx.mean(error)),
    values_at_or_above_1e_4: Nx.to_number(Nx.sum(Nx.greater_equal(error, 1.0e-4)))
  })
end

# Distinct inputs allow simultaneous replacement of the feed-forward branch and
# the residual branch. Both always use tensors from the same reference run.
run_model = fn base, output, replacements ->
  {model, inputs} =
    Enum.reduce(
      replacements,
      {%{base | output: Map.fetch!(names, output)}, %{"mel_spectrogram" => fixture["mel_input"]}},
      fn name, {model, inputs} ->
        tensor = Map.fetch!(reference, name)
        id = Map.fetch!(names, name)
        input_name = "reference_#{name}"
        source = Axon.input(input_name, shape: Nx.shape(tensor))
        node = %{Map.fetch!(source.nodes, source.output) | id: id}
        {%{model | nodes: Map.put(model.nodes, id, node)}, Map.put(inputs, input_name, tensor)}
      end
    )

  {_init, predict} = Axon.build(model, mode: :inference)
  predict.(loaded, inputs)
end

run = fn output, replacements -> run_model.(encoder, output, replacements) end

# Inspect the largest isolated activation difference without dumping tensors.
activation = run.("enc_block_5_ffn_act", ["enc_block_5_ffn_up"])
error = Nx.abs(Nx.subtract(activation, reference["enc_block_5_ffn_act"]))
index = error |> Nx.flatten() |> Nx.argmax() |> Nx.to_number()
at = fn tensor -> tensor |> Nx.flatten() |> Nx.take(index) |> Nx.to_number() end

IO.inspect(%{
  worst_gelu_index: index,
  input: at.(reference["enc_block_5_ffn_up"]),
  actual: at.(activation),
  expected: at.(reference["enc_block_5_ffn_act"])
})

entry = "enc_block_5_attn_residual"

operations = [
  "enc_block_5_ffn_norm",
  "enc_block_5_ffn_up",
  "enc_block_5_ffn_act",
  "enc_block_5_ffn_down",
  "enc_block_5_ffn_residual"
]

report.("baseline", run.("enc_final_norm", []), expected)

# Local differences: each operation receives identical reference inputs.
Enum.reduce(operations, entry, fn operation, previous ->
  replacements = Enum.uniq([entry, previous])
  report.("isolated #{operation}", run.(operation, replacements), reference[operation])
  operation
end)

# Move the boundary while holding the skip connection at the exact entry value.
for boundary <- [entry | operations] do
  replacements = Enum.uniq([entry, boundary])
  report.("boundary #{boundary} final", run.("enc_final_norm", replacements), expected)

  report.(
    "boundary #{boundary} block5",
    run.("enc_block_5_ffn_residual", replacements),
    reference["enc_block_5_ffn_residual"]
  )
end

# Boundary replacement also removes inherited error. Test arithmetic changes
# without replacements as a separate check on the suspected operation.

policy = Axon.MixedPrecision.create_policy(compute: {:f, 64}, params: {:f, 32}, output: {:f, 32})

for {label, selected} <- [
      {"block5_gelu_f64", ["enc_block_5_ffn_act"]},
      {"all_ffn_gelu_f64", for(layer <- 1..6, do: "enc_block_#{layer}_ffn_act")}
    ] do
  model =
    Axon.MixedPrecision.apply_policy(encoder, policy, fn node ->
      node.name.(node.op_name, %{}) in selected
    end)

  report.("#{label} full", run_model.(model, "enc_final_norm", []), expected)
  report.("#{label} exact_entry", run_model.(model, "enc_final_norm", [entry]), expected)

  report.(
    "#{label} isolated",
    run_model.(model, "enc_block_5_ffn_act", ["enc_block_5_ffn_up"]),
    reference["enc_block_5_ffn_act"]
  )
end

# Equivalent GELU expression using the complementary error function. This avoids
# adding one to erf and tests f32 arithmetic without substituting reference data.
for {label, selected} <- [
      {"block5_gelu_erfc", ["enc_block_5_ffn_act"]},
      {"all_ffn_gelu_erfc", for(layer <- 1..6, do: "enc_block_#{layer}_ffn_act")}
    ] do
  nodes =
    Map.new(encoder.nodes, fn {id, node} ->
      if node.name.(node.op_name, %{}) in selected do
        operation = fn x, _opts ->
          Nx.multiply(Nx.multiply(x, 0.5), Nx.erfc(Nx.multiply(x, -0.7071067811865476)))
        end

        {id, %{node | op: operation}}
      else
        {id, node}
      end
    end)

  model = %{encoder | nodes: nodes}
  report.("#{label} full", run_model.(model, "enc_final_norm", []), expected)
  report.("#{label} exact_entry", run_model.(model, "enc_final_norm", [entry]), expected)

  report.(
    "#{label} isolated",
    run_model.(model, "enc_block_5_ffn_act", ["enc_block_5_ffn_up"]),
    reference["enc_block_5_ffn_act"]
  )
end
