# Build-time validation against NumPy: run this script, then compare the emitted
# matrix with numpy.linalg.eigvals. No Python dependency in the library itself.
Nx.default_backend(Nx.BinaryBackend)
{matrix, _} = Nx.Random.normal(Nx.Random.key(2026), shape: {500, 500}, type: :f64)
{mask, _} = Nx.Random.uniform(Nx.Random.key(2027), shape: {500, 500}, type: :f64)
matrix = Nx.select(Nx.greater(mask, 0.9), matrix, 0)
{elapsed, radius} = :timer.tc(fn -> Edifice.Utils.SpectralRadius.compute(matrix) end)
IO.puts("500x500 sparse radius=#{radius}, build-time eigenvalues=#{elapsed / 1_000_000}s")
[out] = System.argv()

File.write!(
  out,
  Jason.encode!(%{matrix: Nx.to_list(matrix), radius: radius, elapsed_us: elapsed})
)
