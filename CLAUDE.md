# Edifice - Claude Code Notes

## Environment
- Development platform: NixOS (native, hostname `nixos_slanka`)
- GPU: NVIDIA RTX 5090 with proprietary drivers
- Dev environment: devenv (replaces old shell.nix / nix-shell workflow)
- Target: Elixir ~> 1.18, Erlang/OTP 27
- Project location: `/home/blewf/git/edifice`

## Running Commands
- Enter dev shell: `cd ~/git/edifice && devenv shell`
- Inside devenv shell, run `mix`, `elixir`, `erl`, `cargo` etc. directly — no wrappers needed.
- CUDA, Rust (for NIFs), Elixir, and Erlang are all provided by devenv.
- Example: `mix test`, `mix compile`, `iex -S mix`

## Testing — CRITICAL

**NEVER run the full test suite after editing 1-2 files.** It takes ~9 minutes on BinaryBackend. Use targeted runs:

| Changed File(s) | Run |
|-----------------|-----|
| `lib/edifice/<family>/<arch>.ex` | `mix test test/edifice/<family>/<arch>_test.exs` |
| `lib/edifice/blocks/*.ex` | `mix test test/edifice/blocks/` then `mix test --stale` |
| `lib/edifice.ex` (registry) | `mix test test/edifice/registry_integrity_test.exs` |
| Multiple files / unsure | `mix test --stale` (alias: `mix test.changed`) |
| Pre-commit / full validation | `mix test` |

Key commands:
- `mix test --stale` — only reruns tests whose module dependencies changed (1-2 files vs 254)
- `mix test --failed` — reruns only tests that failed last time
- `mix test test/edifice/recurrent/deep_res_lstm_test.exs:42` — single test by line
- `mix test --only recurrent` — run one architecture family
- See `docs/TESTING.md` for full documentation

## Numerical Precision Policy

**Loss/objective math (and therefore its backward) always computes in f32,
regardless of network compute precision.** Cast inputs to f32 at the loss
function's entry point — not at call sites, which individual code paths can
bypass.

Why: autodiff adjoints are more numerically fragile than forwards. Case
study (exphil, 2026-07-14): a bf16 BCE backward produced NaN gradients while
the forward loss was still finite, killing week-long training runs at
unpredictable steps; mixed precision (f32 master weights) did NOT help
because forward/backward still ran in bf16. The fix was a one-line f32 cast
at the loss entry point — loss ops are tiny next to the network, so the cost
is negligible.

Applies to any training utilities added here — especially SAE
reconstruction/sparsity objectives in `lib/edifice/interpretability/`
(watch exp/log/div in KL terms and normalizers).
