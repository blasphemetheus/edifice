# Per-frame step latency across the Edifice.Stateful backbones — the harness
# that picks the Melee bot's backbone (16.67 ms/frame budget at 60 FPS).
#
# CPU (EXLA host):      EXLA_TARGET=host mix run bench/step_latency.exs
# Idle GPU re-measure:  EXLA_TARGET=cuda mix run bench/step_latency.exs
# Pure BinaryBackend:   BENCH_EAGER=1 with exla absent (numbers not meaningful)
#
# When EXLA is available the step path is JIT-compiled (one executable per
# step instead of eager op-by-op dispatch); set BENCH_EAGER=1 to force the
# eager path for before/after comparison.
#
# Dims mirror the exphil deployment shape (embed 287, hidden 256, 2 layers).

archs = [:min_gru, :mamba, :gru, :lstm, :gated_ssm]

{compiler, mode_label} =
  cond do
    System.get_env("BENCH_EAGER") == "1" -> {nil, "eager op-by-op"}
    Code.ensure_loaded?(EXLA) -> {EXLA, "JIT-compiled (EXLA)"}
    true -> {nil, "eager op-by-op (EXLA not available)"}
  end

if Code.ensure_loaded?(EXLA) do
  # Run tensors on EXLA either way: the eager (BENCH_EAGER=1) numbers then
  # measure exactly the op-by-op dispatch overhead the JIT removes, and in
  # JIT mode the compiled fun's inputs live on the right device
  Nx.default_backend(EXLA.Backend)
end

IO.puts("Step-latency profile (batch=1, per-frame Edifice.step/5, #{mode_label})\n")

Edifice.Profile.compare(
  archs: archs,
  mode: :step,
  embed_dim: 287,
  hidden_size: 256,
  num_layers: 2,
  warmup: 10,
  iterations: 200,
  compiler: compiler
)
