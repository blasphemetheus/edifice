# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
# Run after building the Rust reference tool. All generated tensors stay in /tmp.
Code.require_file("support/whisper_f64_validation.exs", __DIR__)
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)

defmodule WhisperInputValidation do
  def compare(label, actual, reference, atol, rtol) do
    unless Nx.shape(actual) == Nx.shape(reference), do: raise("Shape mismatch: #{label}")
    error = Nx.abs(Nx.subtract(actual, reference))
    bound = Nx.add(atol, Nx.multiply(rtol, Nx.abs(reference)))
    ratio = Nx.to_number(Nx.reduce_max(Nx.divide(error, bound)))

    IO.puts(
      "#{label} max_abs=#{Nx.to_number(Nx.reduce_max(error))} " <>
        "mean_abs=#{Nx.to_number(Nx.mean(error))} max_bound_ratio=#{ratio}"
    )

    ratio <= 1 and Nx.to_number(Nx.mean(error)) <= if(atol < 1.0e-6, do: 1.0e-11, else: 5.0e-6)
  end
end

directory = System.get_env("WHISPER_VALIDATION_DIR", "/tmp/whisper-input-validation")
File.mkdir_p!(directory)
fixture = Safetensors.read!("test/fixtures/numerical/whisper_encoder_reference.safetensors")
original = fixture["mel_input"]
shape = Nx.shape(original)
time = Nx.iota(shape, axis: 2, type: :f32)
frequency = Nx.iota(shape, axis: 1, type: :f32)
{noise, _} = Nx.Random.normal(Nx.Random.key(20_261_026), shape: shape)

synthetic_cases = [
  {"original", original},
  {"constant_zero", Nx.broadcast(0.0, shape)},
  {"constant_minus_one", Nx.broadcast(-1.0, shape)},
  {"scaled_original", Nx.multiply(original, 0.25)},
  {"structured", Nx.add(Nx.sin(Nx.divide(time, 47)), Nx.cos(Nx.divide(frequency, 13)))},
  {"noise", noise},
  {"step", Nx.select(Nx.less(time, 1500), -1.0, 1.0)},
  {"quiet_noise_holdout", Nx.subtract(Nx.multiply(noise, 0.1), 0.5)},
  {"reversed_original_holdout", Nx.reverse(original, axes: [2])},
  {"pulse_holdout",
   Nx.select(Nx.logical_and(Nx.greater(time, 1400), Nx.less(time, 1600)), 1.5, -0.5)}
]

# Positional arguments select recorded-audio feature fixtures instead of the
# synthetic suite. Generate them with the Rust whisper-audio executable.
cases =
  case System.argv() do
    [] ->
      synthetic_cases

    paths ->
      Enum.map(paths, fn path ->
        input = Safetensors.read!(path)["mel_input"]
        unless Nx.shape(input) == shape, do: raise("Expected #{inspect(shape)} in #{path}")
        {Path.basename(path, ".safetensors"), input}
      end)
  end

{encoder, _, loaded} = Edifice.Pretrained.from_hub("openai/whisper-base")
{_, predict32} = Axon.build(encoder, mode: :inference)
{_, predict64} = Axon.build(WhisperF64Validation.model(encoder), mode: :inference)
checkpoint = Path.expand("~/.cache/edifice/openai/whisper-base/model.safetensors")
binary = Path.expand("tools/whisper_reference/target/debug/whisper-reference")

torch_lib =
  Path.join(
    System.get_env(
      "LIBTORCH",
      Path.expand(".devenv/state/venv/lib/python3.12/site-packages/torch")
    ),
    "lib"
  )

library_path = torch_lib <> ":" <> System.get_env("LD_LIBRARY_PATH", "")

# Chosen from the first seven cases, before measuring the final three holdouts.
# The original saved fixture retains tighter bounds in the external ExUnit test.
IO.puts(
  "f32 bounds: atol=2e-3 rtol=1e-5 mean<=5e-6; f64 bounds: atol=1e-9 rtol=1e-10 mean<=1e-11"
)

passed =
  for {name, input} <- cases do
    input_path = Path.join(directory, "#{name}-input.safetensors")
    Safetensors.write!(input_path, %{"mel_input" => input})

    references =
      Map.new(["sdpa", "eager-f64"], fn mode ->
        output_path = Path.join(directory, "#{name}-#{mode}.safetensors")

        {output, status} =
          System.cmd(binary, [checkpoint, input_path, output_path, mode, "final-only"],
            env: [{"LD_LIBRARY_PATH", library_path}],
            stderr_to_stdout: true
          )

        if status != 0, do: raise("Rust reference failed: #{output}")
        {mode, Safetensors.read!(output_path)["enc_final_norm"]}
      end)

    actual32 = predict32.(loaded, %{"mel_spectrogram" => input})
    actual64 = predict64.(loaded, %{"mel_spectrogram" => input})

    Safetensors.write!(Path.join(directory, "#{name}-exla.safetensors"), %{
      "f32" => actual32,
      "f64" => actual64
    })

    unless Nx.type(actual64) == {:f, 64}, do: raise("Expected f64 output")

    checks = [
      {"EXLA64/Torch64", actual64, references["eager-f64"], 1.0e-9, 1.0e-10},
      {"EXLA32/Torch32", actual32, references["sdpa"], 2.0e-3, 1.0e-5},
      {"EXLA32/Torch64", actual32, references["eager-f64"], 2.0e-3, 1.0e-5},
      {"Torch32/Torch64", references["sdpa"], references["eager-f64"], 2.0e-3, 1.0e-5}
    ]

    results =
      for {label, actual, expected, atol, rtol} <- checks do
        WhisperInputValidation.compare("#{name} #{label}", actual, expected, atol, rtol)
      end

    Enum.all?(results)
  end

unless Enum.all?(passed), do: raise("Whisper validation exceeded the declared bounds")
IO.puts("PASS: all inputs satisfy the independent f64 and f32 checks")
