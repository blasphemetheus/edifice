#!/usr/bin/env bash
# Compile a real downstream application using only Edifice's required dependencies.
# Run after mix deps.get. Path overrides reuse those sources without network access.
set -euo pipefail
export EDIFICE_SOURCE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export EDIFICE_DEPS_ROOT="${MIX_DEPS_PATH:-$EDIFICE_SOURCE_ROOT/deps}"
consumer=$(mktemp -d)
trap 'rm -rf "$consumer"' EXIT
cat > "$consumer/mix.exs" <<'ELIXIR'
defmodule OptionalExlaConsumer.MixProject do
  use Mix.Project
  def project do
    root = System.fetch_env!("EDIFICE_SOURCE_ROOT")
    deps_root = Path.expand(System.fetch_env!("EDIFICE_DEPS_ROOT"), root)
    required = for app <- [:nx, :axon, :polaris, :jason, :complex, :telemetry] do
      path = Path.join(deps_root, Atom.to_string(app))
      # Git dependencies checked out sparsely keep the monorepo subdirectory.
      path = if File.exists?(Path.join(path, "mix.exs")), do: path, else: Path.join(path, Atom.to_string(app))
      {app, path: path, override: true}
    end
    [app: :optional_exla_consumer, version: "0.1.0",
     deps: [{:edifice, path: root} | required]]
  end
  def application, do: []
end
ELIXIR
cat > "$consumer/check.exs" <<'ELIXIR'
if Code.ensure_loaded?(EXLA), do: raise("EXLA unexpectedly present")
if Edifice.Block.native_impl?(), do: raise("native EXLA implementation unexpectedly present")
Nx.default_backend(Nx.BinaryBackend)
model = Edifice.build(:min_gru, embed_dim: 3, hidden_size: 4, num_layers: 1)
{init, predict} = Axon.build(model)
input = %{"state_sequence" => Nx.broadcast(1.0, {1, 2, 3})}
state = init.(input, Axon.ModelState.empty())
output = predict.(state, input)
if Nx.shape(output) != {1, 4}, do: raise("unexpected output shape")
if Nx.to_number(Nx.any(Nx.is_nan(output))) == 1, do: raise("nonfinite output")
IO.puts("Edifice compiled and ran on BinaryBackend without EXLA")
ELIXIR
cd "$consumer"
unset MIX_DEPS_PATH MIX_BUILD_PATH EDIFICE_LOCAL_NX
export MIX_ENV=prod
mix deps.get --offline
mix deps.compile
mix run check.exs
