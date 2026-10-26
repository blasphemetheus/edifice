# Suppress noisy XLA JIT info logs ("Merging Dots", "BFC allocator", etc.)
Logger.configure(level: :warning)

# The default suite is CPU-sized (GPU-only cases are tagged :exla_only), so
# keep EXLA on its host client even when a GPU is present: tests that JIT with
# EXLA stay deterministic and never contend with training jobs for the GPU.
# Set EDIFICE_TEST_CLIENT=cuda to run the suite on the GPU.
if Code.ensure_loaded?(EXLA) do
  client = System.get_env("EDIFICE_TEST_CLIENT", "host") |> String.to_atom()
  Application.put_env(:exla, :default_client, client)
end

ExUnit.start(exclude: [:slow, :integration, :external, :exla_only, :known_issue])
