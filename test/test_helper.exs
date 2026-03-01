# Suppress noisy XLA JIT info logs ("Merging Dots", "BFC allocator", etc.)
Logger.configure(level: :warning)

ExUnit.start(exclude: [:slow, :integration, :external, :exla_only, :known_issue])
