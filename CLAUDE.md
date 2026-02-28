# Edifice - Claude Code Notes

## Environment
- Development platform: WSL (Windows Subsystem for Linux) on Windows, NixOS distro
- Version manager: asdf (Erlang + Elixir)
- Target: Elixir ~> 1.18, Erlang/OTP 27

## Running Commands
- Use `nix-shell --run "<command>" /home/nixos/edifice/shell.nix` to run mix, erl, elixir, and other dev commands. The shell.nix provides Erlang 27, Elixir 1.18, and CUDA dependencies.
- Example: `nix-shell --run "mix test" /home/nixos/edifice/shell.nix`
- Example: `nix-shell --run "mix compile" /home/nixos/edifice/shell.nix`
- The bare PATH does not include Erlang/Elixir — always use nix-shell.

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

## WSL Setup Notes

### Opening WSL repos in VS Code from Windows

NixOS on WSL requires extra setup for VS Code's WSL extension to work:

1. **Enable `nix-ld`** — VS Code Server needs a standard dynamic linker that NixOS doesn't provide by default. Add to `/etc/nixos/configuration.nix`:
   ```nix
   programs.nix-ld.enable = true;
   ```
   Then rebuild: `sudo nixos-rebuild switch`

2. **Enable Windows interop** (if `code .` from WSL terminal doesn't work). Add to `/etc/nixos/configuration.nix`:
   ```nix
   wsl.interop.register = true;
   ```
   Then rebuild: `sudo nixos-rebuild switch`

3. **Open a repo** — from PowerShell:
   ```powershell
   code --remote wsl+NixOS /home/nixos/edifice
   ```
   Or from WSL terminal (once interop is working):
   ```bash
   code .
   ```
