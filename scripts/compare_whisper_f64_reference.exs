# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
[reference_path] = System.argv()
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
reference = Safetensors.read!(reference_path)["enc_final_norm"]
unless Nx.type(reference) == {:f, 64}, do: raise("Use the Rust eager-f64 reference")
{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")

# The production attention helper intentionally computes softmax in f32.
# For this diagnostic, retain f64 through every attention operation as well.
nodes =
  Map.new(encoder.nodes, fn {id, node} ->
    if String.ends_with?(node.name.(node.op_name, %{}), "_attn_compute") do
      attention = fn q, k, v, _opts ->
        split = fn x -> Nx.transpose(Nx.reshape(x, {1, 1500, 8, 64}), axes: [0, 2, 1, 3]) end
        q = split.(q)
        k = split.(k)
        v = split.(v)
        scores = Nx.dot(Nx.divide(q, 8), [3], [0, 1], k, [3], [0, 1])
        exps = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
        weights = Nx.divide(exps, Nx.sum(exps, axes: [-1], keep_axes: true))

        Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
        |> Nx.transpose(axes: [0, 2, 1, 3])
        |> Nx.reshape({1, 1500, 512})
      end

      {id, %{node | op: attention}}
    else
      {id, node}
    end
  end)

policy = Axon.MixedPrecision.create_policy(compute: {:f, 64}, params: {:f, 64}, output: {:f, 64})
model = Axon.MixedPrecision.apply_policy(%{encoder | nodes: nodes}, policy)
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
