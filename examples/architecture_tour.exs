# Architecture Tour
# =================
# A quick tour of Edifice's architecture families using the unified registry.
# Builds one model from each major family to demonstrate breadth and
# the consistent build/init/predict API.
#
# Run with: mix run examples/architecture_tour.exs

IO.puts("=== Edifice Architecture Tour ===\n")

# ---------------------------------------------------------------
# 1. What's available?
# ---------------------------------------------------------------
all = Edifice.list_architectures()
families = Edifice.list_families()

IO.puts("1. Edifice contains #{length(all)} architectures across #{map_size(families)} families:\n")

for {family, archs} <- Enum.sort(families) do
  names = Enum.map(archs, &Atom.to_string/1) |> Enum.join(", ")
  IO.puts("   #{String.pad_trailing(Atom.to_string(family), 16)} (#{length(archs)}): #{names}")
end

# ---------------------------------------------------------------
# 2. Build one from each family
# ---------------------------------------------------------------
IO.puts("\n2. Building one architecture from each family:\n")

# Helper to count parameters
count_params_rec = fn count_params_rec, data ->
  Enum.reduce(data, 0, fn
    {_k, %Nx.Tensor{} = t}, acc -> acc + Nx.size(t)
    {_k, %{} = nested}, acc -> acc + count_params_rec.(count_params_rec, nested)
    _, acc -> acc
  end)
end

count_params = fn params -> count_params_rec.(count_params_rec, params.data) end

# Sequence models: {batch, seq_len, features} -> {batch, hidden}
sequence_models = [
  {:mamba, "Mamba (SSM)", [embed_size: 64, hidden_size: 128, state_size: 16, num_layers: 2, window_size: 30]},
  {:gla, "GLA (Gated Linear Attention)", [embed_size: 64, hidden_size: 128, num_heads: 4, num_layers: 2, window_size: 30]},
  {:lstm, "LSTM (Recurrent)", [embed_size: 64, hidden_size: 128, num_layers: 2, window_size: 30]},
  {:retnet, "RetNet (Retention)", [embed_size: 64, hidden_size: 128, num_heads: 4, num_layers: 2, window_size: 30]},
  {:rwkv, "RWKV-7 (Linear Attention)", [embed_size: 64, hidden_size: 128, num_heads: 4, num_layers: 2, window_size: 30]}
]

seq_template = Nx.template({1, 30, 64}, :f32)
seq_input = Nx.Random.key(0) |> Nx.Random.normal(shape: {2, 30, 64}) |> elem(0)

IO.puts("   --- Sequence Models: {2, 30, 64} input ---")
IO.puts("   #{"Architecture"  |> String.pad_trailing(30)} | #{"Output" |> String.pad_trailing(13)} | Parameters")
IO.puts("   #{String.duplicate("-", 30)}-+-#{String.duplicate("-", 13)}-+----------")

for {name, desc, opts} <- sequence_models do
  model = Edifice.build(name, opts)
  {init_fn, predict_fn} = Axon.build(model)
  params = init_fn.(seq_template, Axon.ModelState.empty())
  output = predict_fn.(params, seq_input)
  pcount = count_params.(params)

  IO.puts("   #{String.pad_trailing(desc, 30)} | #{inspect(Nx.shape(output)) |> String.pad_trailing(13)} | #{pcount}")
end

# Feedforward models: {batch, features} -> {batch, hidden}
IO.puts("\n   --- Feedforward Models: {4, 32} input ---")

ff_models = [
  {:mlp, "MLP", [input_size: 32, hidden_sizes: [128, 64]]},
  {:tabnet, "TabNet", [input_size: 32, hidden_size: 64, output_size: 16]}
]

ff_template = Nx.template({1, 32}, :f32)
ff_input = Nx.Random.key(1) |> Nx.Random.normal(shape: {4, 32}) |> elem(0)

IO.puts("   #{"Architecture" |> String.pad_trailing(30)} | #{"Output" |> String.pad_trailing(13)} | Parameters")
IO.puts("   #{String.duplicate("-", 30)}-+-#{String.duplicate("-", 13)}-+----------")

for {name, desc, opts} <- ff_models do
  model = Edifice.build(name, opts)
  {init_fn, predict_fn} = Axon.build(model)
  params = init_fn.(ff_template, Axon.ModelState.empty())
  output = predict_fn.(params, ff_input)
  pcount = count_params.(params)

  IO.puts("   #{String.pad_trailing(desc, 30)} | #{inspect(Nx.shape(output)) |> String.pad_trailing(13)} | #{pcount}")
end

# ---------------------------------------------------------------
# 3. Generative models (tuple returns)
# ---------------------------------------------------------------
IO.puts("\n   --- Generative Models (return tuples) ---\n")

# VAE
{encoder, decoder} = Edifice.Generative.VAE.build(
  input_size: 256,
  latent_size: 16,
  encoder_sizes: [128, 64],
  decoder_sizes: [64, 128]
)

{enc_init, _enc_pred} = Axon.build(encoder)
{dec_init, _dec_pred} = Axon.build(decoder)
enc_params = enc_init.(Nx.template({1, 256}, :f32), Axon.ModelState.empty())
dec_params = dec_init.(Nx.template({1, 16}, :f32), Axon.ModelState.empty())
enc_count = count_params.(enc_params)
dec_count = count_params.(dec_params)

IO.puts("   VAE:  encoder #{enc_count} params, decoder #{dec_count} params")

# GAN
{generator, discriminator} = Edifice.Generative.GAN.build(
  latent_size: 64,
  output_size: 256,
  gen_sizes: [128, 256],
  disc_sizes: [256, 128]
)

{gen_init, _gen_pred} = Axon.build(generator)
{disc_init, _disc_pred} = Axon.build(discriminator)
gen_params = gen_init.(Nx.template({1, 64}, :f32), Axon.ModelState.empty())
disc_params = disc_init.(Nx.template({1, 256}, :f32), Axon.ModelState.empty())
gen_count = count_params.(gen_params)
disc_count = count_params.(disc_params)

IO.puts("   GAN:  generator #{gen_count} params, discriminator #{disc_count} params")

# ---------------------------------------------------------------
# 4. Graph models (map inputs)
# ---------------------------------------------------------------
IO.puts("\n   --- Graph Models (map inputs) ---\n")

graph_template = %{
  "nodes" => Nx.template({2, 8, 16}, :f32),
  "adjacency" => Nx.template({2, 8, 8}, :f32)
}

graph_input = %{
  "nodes" => Nx.Random.key(2) |> Nx.Random.normal(shape: {2, 8, 16}) |> elem(0),
  "adjacency" => Nx.eye(8) |> Nx.broadcast({2, 8, 8})
}

graph_archs = [
  {"GCN", Edifice.Graph.GCN.build_classifier(input_dim: 16, hidden_dims: [32, 32], num_classes: 3, pool: :mean)},
  {"GAT", Edifice.Graph.GAT.build(input_dim: 16, hidden_dim: 8, num_classes: 3, num_heads: 2)},
  {"GIN", Edifice.Graph.GIN.build(input_dim: 16, hidden_dims: [32, 32], num_classes: 3, pool: :mean)}
]

for {name, model} <- graph_archs do
  {init_fn, predict_fn} = Axon.build(model)
  params = init_fn.(graph_template, Axon.ModelState.empty())
  output = predict_fn.(params, graph_input)
  pcount = count_params.(params)

  IO.puts("   #{String.pad_trailing(name, 6)} #{inspect(Nx.shape(output))} with #{pcount} params")
end

# ---------------------------------------------------------------
# 5. The registry makes experimentation easy
# ---------------------------------------------------------------
IO.puts("\n3. The point: architecture search is a loop\n")

IO.puts("""
   for arch <- [:mamba, :retnet, :griffin, :lstm, :gla] do
     model = Edifice.build(arch, embed_size: 128, hidden_size: 256, ...)
     # ... train and evaluate
   end

   One dependency. One API. 103 architectures.
""")

IO.puts("=== Done ===")
