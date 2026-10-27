# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
[reference_path] = System.argv()
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

report = fn label, actual, expected ->
  error = Nx.abs(Nx.subtract(actual, expected))

  IO.inspect(%{
    experiment: label,
    max_absolute_error: Nx.to_number(Nx.reduce_max(error)),
    mean_absolute_error: Nx.to_number(Nx.mean(error)),
    values_at_or_above_1e_4: Nx.to_number(Nx.sum(Nx.greater_equal(error, 1.0e-4)))
  })
end

run = fn model, output, replacements ->
  {model, inputs} =
    Enum.reduce(
      Enum.uniq(replacements),
      {%{model | output: Map.fetch!(names, output)},
       %{"mel_spectrogram" => fixture["mel_input"]}},
      fn name, {model, inputs} ->
        tensor = Map.fetch!(reference, name)
        id = Map.fetch!(names, name)
        source = Axon.input("reference_#{name}", shape: Nx.shape(tensor))
        node = %{Map.fetch!(source.nodes, source.output) | id: id}

        {%{model | nodes: Map.put(model.nodes, id, node)},
         Map.put(inputs, "reference_#{name}", tensor)}
      end
    )

  {_init, predict} = Axon.build(model, mode: :inference)
  predict.(loaded, inputs)
end

IO.inspect(%{
  nx: Application.spec(:nx, :vsn),
  exla: Application.spec(:exla, :vsn),
  xla: Application.spec(:xla, :vsn),
  backend: :exla_host
})

baseline = run.(encoder, "enc_final_norm", [])
report.("baseline", baseline, expected)
errors = Nx.abs(Nx.subtract(baseline, expected)) |> Nx.to_flat_list()

outliers =
  for {error, index} <- Enum.with_index(errors),
      error >= 1.0e-4,
      do: %{frame: div(index, 512), channel: rem(index, 512), absolute_error: error}

IO.inspect(%{baseline_outliers: outliers})
gamma = loaded.data["enc_final_norm"]["gamma"]

IO.inspect(%{
  final_norm_gamma_channel_323: Nx.to_number(Nx.take(gamma, 323)),
  final_norm_max_abs_gamma: Nx.to_number(Nx.reduce_max(Nx.abs(gamma)))
})

for layer <- 1..2 do
  prefix = "enc_block_#{layer}"
  entry = if layer == 1, do: "enc_pos_embed", else: "enc_block_#{layer - 1}_ffn_residual"
  norm = "#{prefix}_attn_norm"
  qkv = for part <- ["q", "k", "v"], do: "#{prefix}_attn_#{part}"

  stages =
    [{norm, [entry]}] ++
      Enum.map(qkv, &{&1, [norm]}) ++
      [{"#{prefix}_attn_compute", qkv}, {"#{prefix}_attn_out", ["#{prefix}_attn_compute"]}]

  for {target, sources} <- stages do
    report.("isolated #{target}", run.(encoder, target, sources), reference[target])
  end

  for {label, replacements} <- [
        {"entry", [entry]},
        {"norm", [entry, norm]},
        {"qkv", [entry | qkv]},
        {"compute", [entry, "#{prefix}_attn_compute"]},
        {"out", [entry, "#{prefix}_attn_out"]}
      ] do
    report.(
      "#{prefix} attention boundary #{label}",
      run.(encoder, "enc_final_norm", replacements),
      expected
    )
  end

  entry = "#{prefix}_attn_residual"

  Enum.reduce(["norm", "up", "act", "down", "residual"], entry, fn suffix, previous ->
    target = "#{prefix}_ffn_#{suffix}"
    report.("isolated #{target}", run.(encoder, target, [entry, previous]), reference[target])

    report.(
      "#{prefix} ffn boundary #{suffix}",
      run.(encoder, "enc_final_norm", [entry, target]),
      expected
    )

    target
  end)
end

# Change arithmetic without substituting reference activations. These f64
# policies retain f32 checkpoint parameters and round node outputs back to f32.
policy = Axon.MixedPrecision.create_policy(compute: {:f, 64}, params: {:f, 32}, output: {:f, 32})

for {label, selected} <- [
      {"conv1", ["enc_conv1"]},
      {"convolutions", ["enc_conv1", "enc_conv2"]},
      {"first_norm", ["enc_block_1_attn_norm"]},
      {"block1_ffn_up", ["enc_block_1_ffn_up"]},
      {"block1_attention",
       for(suffix <- ["q", "k", "v", "compute", "out"], do: "enc_block_1_attn_#{suffix}")},
      {"all_norms", Enum.filter(Map.keys(names), &String.ends_with?(&1, "_norm"))}
    ] do
  model =
    Axon.MixedPrecision.apply_policy(encoder, policy, fn node ->
      node.name.(node.op_name, %{}) in selected
    end)

  report.("f64 #{label}", run.(model, "enc_final_norm", []), expected)
end
