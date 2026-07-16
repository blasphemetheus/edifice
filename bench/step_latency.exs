# Per-frame step latency across the Edifice.Stateful backbones — the harness
# that picks the Melee bot's backbone (16.67 ms/frame budget at 60 FPS).
#
# CPU (BinaryBackend):  mix run bench/step_latency.exs
# Idle GPU re-measure:  EXLA_TARGET=cuda mix run bench/step_latency.exs
#
# Dims mirror the exphil deployment shape (embed 287, hidden 256, 2 layers).

archs = [:min_gru, :mamba, :gru, :lstm, :gated_ssm]

IO.puts("Step-latency profile (batch=1, per-frame Edifice.step/4)\n")

Edifice.Profile.compare(
  archs: archs,
  mode: :step,
  embed_dim: 287,
  hidden_size: 256,
  num_layers: 2,
  warmup: 10,
  iterations: 200
)
