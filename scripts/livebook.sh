#!/usr/bin/env bash
# Start Livebook attached to the Edifice project node.
#
# This gives Livebook access to all compiled deps (including EXLA with CUDA)
# without needing Mix.install in notebooks.
#
# Usage:
#   From nix-shell, run two terminals:
#
#     Terminal 1:  ./scripts/livebook.sh node
#     Terminal 2:  ./scripts/livebook.sh livebook
#
#   Or use a single terminal (backgrounded node):
#
#     ./scripts/livebook.sh
#
# Then open the URL printed by Livebook and evaluate the "Attached to project"
# setup cell in any notebook.

set -euo pipefail
cd "$(dirname "$0")/.."

COOKIE="edifice_livebook"
NODE="edifice@127.0.0.1"

case "${1:-all}" in
  node)
    echo "Starting Edifice project node: ${NODE}"
    exec iex --name "$NODE" --cookie "$COOKIE" -S mix
    ;;
  livebook)
    echo "Starting Livebook attached to ${NODE}"
    exec env \
      LIVEBOOK_DEFAULT_RUNTIME="attached:${NODE}:${COOKIE}" \
      livebook server --cookie "$COOKIE"
    ;;
  all)
    echo "Starting Edifice node in background..."
    iex --name "$NODE" --cookie "$COOKIE" -S mix &
    NODE_PID=$!

    # Wait for the node to be ready
    sleep 3

    echo ""
    echo "Starting Livebook attached to ${NODE}"
    echo "Press Ctrl+C to stop both."
    echo ""

    trap "kill $NODE_PID 2>/dev/null; wait $NODE_PID 2>/dev/null" EXIT

    env \
      LIVEBOOK_DEFAULT_RUNTIME="attached:${NODE}:${COOKIE}" \
      livebook server --cookie "$COOKIE"
    ;;
  *)
    echo "Usage: $0 [node|livebook|all]"
    echo ""
    echo "  node      Start the project IEx node only"
    echo "  livebook  Start Livebook only (attach to existing node)"
    echo "  all       Start both (default)"
    exit 1
    ;;
esac
