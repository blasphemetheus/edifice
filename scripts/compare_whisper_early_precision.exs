# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
# Use the SDPA reference from the Rust runner: it reproduces the original fixture.
[reference_path] = System.argv()
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")
inputs = %{"mel_spectrogram" => fixture["mel_input"]}
names = Map.new(encoder.nodes, fn {id, node} -> {node.name.(node.op_name, %{}), id} end)
policy = Axon.MixedPrecision.create_policy(compute: {:f, 64}, params: {:f, 32}, output: {:f, 32})

unless Nx.to_number(
         Nx.reduce_max(
           Nx.abs(Nx.subtract(reference["enc_final_norm"], fixture["expected_encoder_output"]))
         )
       ) == 0 do
  raise "Use a Rust SDPA reference that exactly reproduces the saved fixture"
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

predict = fn model, input ->
  {_init, run} = Axon.build(model, mode: :inference)
  run.(loaded, input)
end

# Replace a node with a supplied tensor while keeping downstream graph edges.
# This bypasses earlier calculations, without changing downstream parameters.
inject = fn model, name, tensor ->
  id = Map.fetch!(names, name)
  source = Axon.input("reference_input", shape: Nx.shape(tensor))
  node = %{Map.fetch!(source.nodes, source.output) | id: id}
  {%{model | nodes: Map.put(model.nodes, id, node)}, Map.put(inputs, "reference_input", tensor)}
end

experiments = [
  {"baseline", []},
  {"conv1", ["enc_conv1"]},
  {"conv2", ["enc_conv2"]},
  {"both_convolutions", ["enc_conv1", "enc_conv2"]},
  {"stem_with_gelu", ["enc_conv1", "enc_conv1_act", "enc_conv2", "enc_conv2_act"]},
  {"position_addition", ["enc_pos_embed"]},
  {"first_norm", ["enc_block_1_attn_norm"]},
  {"final_norm", ["enc_final_norm"]},
  {"all_norms", Enum.filter(Map.keys(names), &String.ends_with?(&1, "_norm"))}
]

{_, all_norms} = List.keyfind(experiments, "all_norms", 0)
experiments = experiments ++ [{"convolutions_and_norms", ["enc_conv1", "enc_conv2" | all_norms]}]

for {label, selected} <- experiments do
  model =
    Axon.MixedPrecision.apply_policy(encoder, policy, fn node ->
      node.name.(node.op_name, %{}) in selected
    end)

  IO.inspect(%{experiment: label, selected_nodes: Enum.sort(selected)})
  report.(label, predict.(model, inputs), fixture["expected_encoder_output"])
end

# Supply exact reference activations to find how much discrepancy the remaining
# f32 suffix creates. This intervention is separate from the precision variants.
for name <- [
      "enc_conv1_act",
      "enc_conv2_act",
      "enc_pos_embed",
      "enc_block_1_ffn_residual",
      "enc_block_2_ffn_residual",
      "enc_block_3_ffn_residual",
      "enc_block_4_ffn_residual",
      "enc_block_5_attn_residual",
      "enc_block_5_ffn_residual",
      "enc_block_6_ffn_residual"
    ] do
  {model, input} = inject.(encoder, name, Map.fetch!(reference, name))

  report.(
    "reference injection #{name}",
    predict.(model, input),
    fixture["expected_encoder_output"]
  )
end

# Compare each convolution+GELU and normalization on identical reference inputs.
sources = [
  {"enc_conv1_act", "enc_transpose", Nx.transpose(fixture["mel_input"], axes: [0, 2, 1])},
  {"enc_conv2_act", "enc_conv1_act", reference["enc_conv1_act"]}
]

norm_sources =
  for layer <- 1..6, part <- [:attn, :ffn] do
    source =
      case {part, layer} do
        {:attn, 1} -> "enc_pos_embed"
        {:attn, _} -> "enc_block_#{layer - 1}_ffn_residual"
        {:ffn, _} -> "enc_block_#{layer}_attn_residual"
      end

    {"enc_block_#{layer}_#{part}_norm", source, reference[source]}
  end

for {target, source, tensor} <-
      sources ++
        norm_sources ++
        [{"enc_final_norm", "enc_block_6_ffn_residual", reference["enc_block_6_ffn_residual"]}],
    precision <- [:f32, :f64] do
  model = %{encoder | output: Map.fetch!(names, target)}
  {model, input} = inject.(model, source, tensor)
  # Convolution comparisons keep GELU in f32, isolating convolution precision.
  selected = String.replace_suffix(target, "_act", "")

  model =
    if precision == :f64,
      do:
        Axon.MixedPrecision.apply_policy(model, policy, fn node ->
          node.name.(node.op_name, %{}) == selected
        end),
      else: model

  report.("isolated #{target} #{precision}", predict.(model, input), reference[target])
end
