# VAE Generation
# ==============
# Build a Variational Autoencoder, encode data to latent space,
# sample, decode, and compute the VAE loss.
#
# This demonstrates Edifice's generative model pattern where
# build/1 returns a tuple of models (encoder, decoder).
#
# Run with: mix run examples/vae_generation.exs

alias Edifice.Generative.VAE

IO.puts("=== VAE Generation ===\n")

# ---------------------------------------------------------------
# 1. Build the VAE
# ---------------------------------------------------------------
# VAE.build returns {encoder, decoder} -- two separate Axon models.
# The encoder maps input -> {mu, log_var} (distribution parameters).
# The decoder maps latent vector -> reconstruction.

input_size = 784  # e.g., flattened 28x28 image
latent_size = 32

{encoder, decoder} = VAE.build(
  input_size: input_size,
  latent_size: latent_size,
  encoder_sizes: [512, 256],
  decoder_sizes: [256, 512]
)

IO.puts("1. Built VAE:")
IO.puts("   Encoder: #{input_size} -> [512, 256] -> latent #{latent_size}")
IO.puts("   Decoder: latent #{latent_size} -> [256, 512] -> #{input_size}")

# ---------------------------------------------------------------
# 2. Compile and initialize both models
# ---------------------------------------------------------------
{enc_init, enc_predict} = Axon.build(encoder)
{dec_init, dec_predict} = Axon.build(decoder)

enc_params = enc_init.(Nx.template({1, input_size}, :f32), Axon.ModelState.empty())
dec_params = dec_init.(Nx.template({1, latent_size}, :f32), Axon.ModelState.empty())

IO.puts("\n2. Initialized encoder and decoder parameters")

# ---------------------------------------------------------------
# 3. Encode: data -> latent distribution
# ---------------------------------------------------------------
# Create synthetic "images" (random data as placeholder)
batch_size = 8
data = Nx.Random.key(42) |> Nx.Random.uniform(shape: {batch_size, input_size}) |> elem(0)

# The encoder outputs mu and log_var (the distribution parameters)
%{mu: mu, log_var: log_var} = enc_predict.(enc_params, data)

IO.puts("\n3. Encoded #{batch_size} samples to latent distributions:")
IO.puts("   mu shape:      #{inspect(Nx.shape(mu))}")
IO.puts("   log_var shape:  #{inspect(Nx.shape(log_var))}")
IO.puts("   mu range:       [#{Nx.reduce_min(mu) |> Nx.to_number() |> Float.round(3)}, #{Nx.reduce_max(mu) |> Nx.to_number() |> Float.round(3)}]")

# ---------------------------------------------------------------
# 4. Sample from the latent space
# ---------------------------------------------------------------
# The reparameterization trick: z = mu + eps * exp(0.5 * log_var)
# This makes sampling differentiable for backpropagation.

z = VAE.reparameterize(mu, log_var)

IO.puts("\n4. Sampled latent vectors via reparameterization trick:")
IO.puts("   z shape: #{inspect(Nx.shape(z))}")
IO.puts("   z range: [#{Nx.reduce_min(z) |> Nx.to_number() |> Float.round(3)}, #{Nx.reduce_max(z) |> Nx.to_number() |> Float.round(3)}]")

# ---------------------------------------------------------------
# 5. Decode: latent vector -> reconstruction
# ---------------------------------------------------------------
reconstruction = dec_predict.(dec_params, z)

IO.puts("\n5. Decoded latent vectors to reconstructions:")
IO.puts("   Reconstruction shape: #{inspect(Nx.shape(reconstruction))}")
IO.puts("   (should match input shape: {#{batch_size}, #{input_size}})")

# ---------------------------------------------------------------
# 6. Compute VAE loss
# ---------------------------------------------------------------
# The VAE loss has two parts:
#   - Reconstruction loss: how well the decoder reconstructs the input
#   - KL divergence: how close the learned posterior is to the prior N(0,I)

kl_loss = VAE.kl_divergence(mu, log_var)
total_loss = VAE.loss(reconstruction, data, mu, log_var, beta: 1.0)

IO.puts("\n6. Loss computation:")
IO.puts("   KL divergence:  #{kl_loss |> Nx.to_number() |> Float.round(4)}")
IO.puts("   Total VAE loss: #{total_loss |> Nx.to_number() |> Float.round(4)}")
IO.puts("   (untrained model -- loss will decrease with training)")

# ---------------------------------------------------------------
# 7. Generate new samples from the prior
# ---------------------------------------------------------------
# To generate new data, sample z from the prior N(0, I)
# and decode it -- no encoder needed.

num_generated = 4
{prior_z, _key} = Nx.Random.normal(Nx.Random.key(99), shape: {num_generated, latent_size})
generated = dec_predict.(dec_params, prior_z)

IO.puts("\n7. Generated #{num_generated} new samples from prior N(0, I):")
IO.puts("   Generated shape: #{inspect(Nx.shape(generated))}")
IO.puts("   (random weights = random outputs; training would produce realistic samples)")

# ---------------------------------------------------------------
# 8. Beta-VAE: controlling the latent space
# ---------------------------------------------------------------
# Higher beta = stronger regularization = more organized latent space
# Lower beta = better reconstructions but less regular latent space

for beta <- [0.1, 1.0, 5.0] do
  loss = VAE.loss(reconstruction, data, mu, log_var, beta: beta)
  IO.puts("   beta=#{beta}: loss = #{loss |> Nx.to_number() |> Float.round(4)}")
end

IO.puts("\n=== Done ===")
