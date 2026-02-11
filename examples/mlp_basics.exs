# MLP Basics
# ===========
# The simplest Edifice example: build an MLP, initialize parameters,
# run inference, and add a classification head.
#
# Run with: mix run examples/mlp_basics.exs

IO.puts("=== MLP Basics ===\n")

# ---------------------------------------------------------------
# 1. Build a simple MLP
# ---------------------------------------------------------------
# This creates a computation graph (blueprint), not a trained model.
# No weights exist yet -- just a description of the network structure.

model = Edifice.Feedforward.MLP.build(
  input_size: 32,
  hidden_sizes: [128, 64],
  activation: :relu,
  dropout: 0.1
)

IO.puts("1. Built MLP: 32 -> 128 -> 64")

# ---------------------------------------------------------------
# 2. Compile the graph into functions
# ---------------------------------------------------------------
# Axon.build/1 returns two functions:
#   init_fn:    creates random parameters
#   predict_fn: runs the forward pass

{init_fn, predict_fn} = Axon.build(model)

IO.puts("2. Compiled model into init_fn and predict_fn")

# ---------------------------------------------------------------
# 3. Initialize parameters
# ---------------------------------------------------------------
# The template tells Axon what input shape to expect.
# nil in the first position means "any batch size."

template = Nx.template({1, 32}, :f32)
params = init_fn.(template, Axon.ModelState.empty())

# Let's see what parameters were created
param_count =
  params
  |> Axon.ModelState.data()
  |> Enum.reduce(0, fn {_name, layer_params}, acc ->
    layer_count =
      layer_params
      |> Map.values()
      |> Enum.map(&Nx.size/1)
      |> Enum.sum()

    acc + layer_count
  end)

IO.puts("3. Initialized #{param_count} parameters")
IO.puts("   Layer names: #{params |> Axon.ModelState.data() |> Map.keys() |> Enum.join(", ")}")

# ---------------------------------------------------------------
# 4. Run inference
# ---------------------------------------------------------------
# Feed data through the network. The batch dimension can be anything.

single_input = Nx.broadcast(0.5, {1, 32})
output = predict_fn.(params, single_input)
IO.puts("\n4. Single sample inference:")
IO.puts("   Input shape:  #{inspect(Nx.shape(single_input))}")
IO.puts("   Output shape: #{inspect(Nx.shape(output))}")

batch_input = Nx.broadcast(0.5, {8, 32})
batch_output = predict_fn.(params, batch_input)
IO.puts("\n   Batch inference:")
IO.puts("   Input shape:  #{inspect(Nx.shape(batch_input))}")
IO.puts("   Output shape: #{inspect(Nx.shape(batch_output))}")

# ---------------------------------------------------------------
# 5. Add a classification head
# ---------------------------------------------------------------
# Edifice models are composable Axon graphs. You can pipe them
# into additional layers.

num_classes = 5

classifier =
  model
  |> Axon.dense(num_classes, name: "classifier_head")
  |> Axon.activation(:softmax)

{cls_init, cls_predict} = Axon.build(classifier)
cls_params = cls_init.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())

predictions = cls_predict.(cls_params, batch_input)
IO.puts("\n5. With classification head (#{num_classes} classes):")
IO.puts("   Output shape: #{inspect(Nx.shape(predictions))}")
IO.puts("   First sample probabilities: #{Nx.to_flat_list(predictions[0]) |> Enum.map(&Float.round(&1, 4)) |> inspect()}")
IO.puts("   Sum of probabilities: #{predictions[0] |> Nx.sum() |> Nx.to_number() |> Float.round(4)}")

# ---------------------------------------------------------------
# 6. Compare MLP variants
# ---------------------------------------------------------------
# MLP with residual connections and layer normalization

model_fancy = Edifice.Feedforward.MLP.build(
  input_size: 32,
  hidden_sizes: [64, 64, 64, 64],
  activation: :silu,
  dropout: 0.1,
  residual: true,
  layer_norm: true
)

{fancy_init, fancy_predict} = Axon.build(model_fancy)
fancy_params = fancy_init.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())
fancy_output = fancy_predict.(fancy_params, batch_input)

IO.puts("\n6. Fancy MLP (residual + layer_norm, 4 layers, SiLU):")
IO.puts("   Output shape: #{inspect(Nx.shape(fancy_output))}")

IO.puts("\n=== Done ===")
