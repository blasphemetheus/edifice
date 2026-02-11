# Graph Classification
# ====================
# Classify synthetic graphs using a Graph Convolutional Network (GCN).
# Demonstrates Edifice's graph model pattern with map-based inputs
# containing node features and adjacency matrices.
#
# Run with: mix run examples/graph_classification.exs

IO.puts("=== Graph Classification ===\n")

# ---------------------------------------------------------------
# 1. Create synthetic graph data
# ---------------------------------------------------------------
# We'll create two types of graphs:
#   Type A: dense connections (high connectivity)
#   Type B: sparse connections (chain-like)
#
# Each graph has 10 nodes with 16 features per node.

num_graphs = 8
num_nodes = 10
input_dim = 16

key = Nx.Random.key(42)

# Node features: random for both types
{node_features, key} = Nx.Random.normal(key, shape: {num_graphs, num_nodes, input_dim})

# Adjacency matrices: mix of dense and sparse graphs
# Dense: many connections (upper triangle filled)
dense_adj =
  Nx.broadcast(1.0, {num_nodes, num_nodes})
  |> Nx.multiply(Nx.subtract(1.0, Nx.eye(num_nodes)))

# Sparse: chain connections only (each node connects to its neighbor)
sparse_adj =
  Enum.reduce(0..(num_nodes - 2), Nx.broadcast(0.0, {num_nodes, num_nodes}), fn i, acc ->
    acc
    |> Nx.indexed_put(Nx.tensor([i, i + 1]), Nx.tensor(1.0))
    |> Nx.indexed_put(Nx.tensor([i + 1, i]), Nx.tensor(1.0))
  end)

# Alternate: first 4 graphs dense, last 4 sparse
adjacency =
  Nx.concatenate([
    Nx.broadcast(dense_adj, {4, num_nodes, num_nodes}),
    Nx.broadcast(sparse_adj, {4, num_nodes, num_nodes})
  ])

IO.puts("1. Created #{num_graphs} synthetic graphs:")
IO.puts("   #{num_nodes} nodes, #{input_dim} features per node")
IO.puts("   Graphs 0-3: dense connectivity")
IO.puts("   Graphs 4-7: sparse (chain) connectivity")

# ---------------------------------------------------------------
# 2. Build a GCN classifier
# ---------------------------------------------------------------
# build_classifier adds global pooling + classification head on top
# of the GCN message-passing layers.

num_classes = 2

model = Edifice.Graph.GCN.build_classifier(
  input_dim: input_dim,
  hidden_dims: [64, 64],
  num_classes: num_classes,
  pool: :mean,
  dropout: 0.0
)

IO.puts("\n2. Built GCN classifier:")
IO.puts("   #{input_dim} -> [64, 64] -> pool -> #{num_classes} classes")

# ---------------------------------------------------------------
# 3. Compile and initialize
# ---------------------------------------------------------------
{init_fn, predict_fn} = Axon.build(model)

# Graph models use map-based inputs
template = %{
  "nodes" => Nx.template({num_graphs, num_nodes, input_dim}, :f32),
  "adjacency" => Nx.template({num_graphs, num_nodes, num_nodes}, :f32)
}

params = init_fn.(template, Axon.ModelState.empty())

IO.puts("\n3. Initialized model parameters")

# ---------------------------------------------------------------
# 4. Run inference
# ---------------------------------------------------------------
input = %{
  "nodes" => node_features,
  "adjacency" => adjacency
}

predictions = predict_fn.(params, input)

IO.puts("\n4. Classification results (untrained -- random predictions):")
IO.puts("   Output shape: #{inspect(Nx.shape(predictions))}")
IO.puts("   (#{num_graphs} graphs x #{num_classes} class logits)\n")

# Show per-graph predictions
for i <- 0..(num_graphs - 1) do
  logits = predictions[i] |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 3))
  graph_type = if i < 4, do: "dense ", else: "sparse"
  IO.puts("   Graph #{i} (#{graph_type}): logits #{inspect(logits)}")
end

# ---------------------------------------------------------------
# 5. Compare GCN vs GAT
# ---------------------------------------------------------------
IO.puts("\n5. Comparing graph architectures:\n")

graph_architectures = [
  {"GCN (spectral convolution)", fn ->
    Edifice.Graph.GCN.build_classifier(
      input_dim: input_dim, hidden_dims: [64, 64],
      num_classes: num_classes, pool: :mean
    )
  end},
  {"GAT (graph attention)", fn ->
    Edifice.Graph.GAT.build_classifier(
      input_dim: input_dim, hidden_dims: [64, 64],
      num_classes: num_classes, num_heads: 4, pool: :mean
    )
  end},
  {"GIN (graph isomorphism)", fn ->
    Edifice.Graph.GIN.build_classifier(
      input_dim: input_dim, hidden_dims: [64, 64],
      num_classes: num_classes, pool: :mean
    )
  end}
]

for {name, build_fn} <- graph_architectures do
  arch_model = build_fn.()
  {arch_init, arch_predict} = Axon.build(arch_model)
  arch_params = arch_init.(template, Axon.ModelState.empty())
  output = arch_predict.(arch_params, input)
  IO.puts("   #{String.pad_trailing(name, 35)} output: #{inspect(Nx.shape(output))}")
end

# ---------------------------------------------------------------
# 6. DeepSets for unordered data
# ---------------------------------------------------------------
IO.puts("\n6. Bonus: DeepSets for unordered set data:\n")

set_model = Edifice.Sets.DeepSets.build(
  input_dim: 3,
  hidden_dim: 64,
  output_dim: 10,
  pool: :mean
)

{set_init, set_predict} = Axon.build(set_model)
set_params = set_init.(Nx.template({4, 20, 3}, :f32), Axon.ModelState.empty())

# 4 sets of 20 3D points
{point_clouds, _key} = Nx.Random.normal(key, shape: {4, 20, 3})
set_output = set_predict.(set_params, point_clouds)

IO.puts("   Input:  4 sets of 20 points in 3D = #{inspect(Nx.shape(point_clouds))}")
IO.puts("   Output: 4 set-level predictions   = #{inspect(Nx.shape(set_output))}")

IO.puts("\n=== Done ===")
