# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
# No checkpoint or audio fixture is needed.
Nx.default_backend({EXLA.Backend, client: :host})
Nx.Defn.default_options(compiler: EXLA, client: :host, precision: :highest)

values = [
  -59.02690505981445,
  -10.0,
  -5.0,
  -3.0,
  -1.0,
  0.0,
  1.0,
  3.0,
  5.0,
  10.0,
  59.02690505981445
]

x = Nx.tensor(values, type: :f32)
# Reference uses the actual rounded f32 inputs, evaluated with Erlang's f64 math.
rounded = Nx.to_flat_list(x)
expected_erf = Nx.tensor(Enum.map(rounded, &:math.erf(&1 / :math.sqrt(2.0))), type: :f32)

expected_gelu =
  Nx.tensor(Enum.map(rounded, &(0.5 * &1 * :math.erfc(-&1 / :math.sqrt(2.0)))), type: :f32)

erf_fn = fn x -> Nx.erf(Nx.divide(x, Nx.sqrt(Nx.tensor(2, type: Nx.type(x))))) end
gelu_fn = fn x -> Nx.divide(Nx.multiply(Nx.add(erf_fn.(x), 1), x), 2) end

erfc_fn = fn x ->
  Nx.multiply(Nx.multiply(x, 0.5), Nx.erfc(Nx.multiply(x, -0.7071067811865476)))
end

IO.inspect(%{
  nx: Application.spec(:nx, :vsn),
  exla: Application.spec(:exla, :vsn),
  xla_flags: System.get_env("XLA_FLAGS"),
  inputs: rounded
})

for {name, operation, expected} <- [
      {"erf_only", erf_fn, expected_erf},
      {"gelu_fused", gelu_fn, expected_gelu},
      {"gelu_erfc", erfc_fn, expected_gelu}
    ] do
  for {shape, input} <- [
        {:vector, x},
        {:model_shape, Nx.broadcast(Nx.reshape(x, {11, 1}), {11, 2048})}
      ] do
    actual = Nx.Defn.jit(operation).(input)
    actual = if shape == :vector, do: actual, else: Nx.take(actual, 0, axis: 1)

    IO.inspect(%{
      operation: name,
      shape: shape,
      actual: Nx.to_flat_list(actual),
      error: Nx.to_flat_list(Nx.subtract(actual, expected))
    })
  end
end

# Force host transfers between steps, preventing fusion across the boundary.
erf = Nx.Defn.jit(erf_fn).(x) |> Nx.backend_transfer(Nx.BinaryBackend)
sum = Nx.add(erf, 1) |> Nx.backend_transfer({EXLA.Backend, client: :host})
staged = Nx.Defn.jit(fn x, sum -> Nx.divide(Nx.multiply(x, sum), 2) end).(x, sum)

IO.inspect(%{
  operation: "gelu_staged_after_erf_and_add",
  actual: Nx.to_flat_list(staged),
  error: Nx.to_flat_list(Nx.subtract(staged, expected_gelu))
})
