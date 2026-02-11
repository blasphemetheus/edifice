# Sequence Model Comparison
# =========================
# Compare different sequence architectures on the same input.
# Shows that Edifice's consistent API makes swapping architectures trivial.
#
# Run with: mix run examples/sequence_comparison.exs

IO.puts("=== Sequence Model Comparison ===\n")

# ---------------------------------------------------------------
# Setup: common parameters for all models
# ---------------------------------------------------------------
# Imagine we have game state sequences: 60 frames, 128 features per frame.

embed_size = 128
hidden_size = 256
num_layers = 2
window_size = 60
batch_size = 4

# Create synthetic input: a batch of sequences
input = Nx.Random.key(0) |> Nx.Random.normal(shape: {batch_size, window_size, embed_size}) |> elem(0)

IO.puts("Input shape: #{inspect(Nx.shape(input))}")
IO.puts("  = #{batch_size} sequences of #{window_size} frames with #{embed_size} features each\n")

# ---------------------------------------------------------------
# Define architectures to compare
# ---------------------------------------------------------------
architectures = [
  {:lstm, "LSTM (classic recurrent)",
    [embed_size: embed_size, hidden_size: hidden_size, num_layers: num_layers, window_size: window_size]},

  {:min_gru, "MinGRU (parallel-scannable recurrent)",
    [embed_size: embed_size, hidden_size: hidden_size, num_layers: num_layers, window_size: window_size]},

  {:retnet, "RetNet (retention-based)",
    [embed_size: embed_size, hidden_size: hidden_size, num_layers: num_layers, num_heads: 4, window_size: window_size]},

  {:mamba, "Mamba (selective SSM)",
    [embed_size: embed_size, hidden_size: hidden_size, num_layers: num_layers, state_size: 16, window_size: window_size]},

  {:griffin, "Griffin (gated linear recurrence + local attention)",
    [embed_size: embed_size, hidden_size: hidden_size, num_layers: num_layers, window_size: window_size]}
]

# ---------------------------------------------------------------
# Build and run each architecture
# ---------------------------------------------------------------
IO.puts("Architecture                                  | Output Shape  | Parameters")
IO.puts("----------------------------------------------|---------------|----------")

count_params = fn count_params, data ->
  Enum.reduce(data, 0, fn
    {_k, %Nx.Tensor{} = t}, acc -> acc + Nx.size(t)
    {_k, %{} = nested}, acc -> acc + count_params.(count_params, nested)
    _, acc -> acc
  end)
end

_results =
  Enum.map(architectures, fn {name, description, opts} ->
    # Build the model through the registry
    model = Edifice.build(name, opts)

    # Compile
    {init_fn, predict_fn} = Axon.build(model)

    # Initialize and count parameters
    template = Nx.template({1, window_size, embed_size}, :f32)
    params = init_fn.(template, Axon.ModelState.empty())

    param_count = count_params.(count_params, params.data)

    # Run inference
    output = predict_fn.(params, input)

    # Format output
    shape_str = inspect(Nx.shape(output)) |> String.pad_trailing(13)
    param_str = param_count |> Integer.to_string() |> String.pad_leading(9)
    desc_padded = String.pad_trailing(description, 45)

    IO.puts("#{desc_padded} | #{shape_str} | #{param_str}")

    {name, output, param_count}
  end)

# ---------------------------------------------------------------
# Observations
# ---------------------------------------------------------------
IO.puts("")
IO.puts("Key observations:")
IO.puts("  - All models accept the same input shape: {#{batch_size}, #{window_size}, #{embed_size}}")
IO.puts("  - All models output the same shape: {#{batch_size}, #{hidden_size}}")
IO.puts("  - Swapping architecture = changing one atom in Edifice.build/2")
IO.puts("  - Parameter counts vary: different efficiency/capacity tradeoffs")

# ---------------------------------------------------------------
# Composing with a task head
# ---------------------------------------------------------------
IO.puts("\n--- Adding a classification head ---\n")

# Pick Mamba and add a 5-class head
model =
  Edifice.build(:mamba,
    embed_size: embed_size,
    hidden_size: hidden_size,
    num_layers: num_layers,
    state_size: 16,
    window_size: window_size
  )
  |> Axon.dense(5, name: "action_head")
  |> Axon.activation(:softmax)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, window_size, embed_size}, :f32), Axon.ModelState.empty())
predictions = predict_fn.(params, input)

IO.puts("Mamba + classification head:")
IO.puts("  Output shape: #{inspect(Nx.shape(predictions))}")
IO.puts("  First sample: #{Nx.to_flat_list(predictions[0]) |> Enum.map(&Float.round(&1, 4)) |> inspect()}")

IO.puts("\n=== Done ===")
