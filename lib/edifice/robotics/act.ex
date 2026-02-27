defmodule Edifice.Robotics.ACT do
  @moduledoc """
  ACT: Action Chunking with Transformers for robot imitation learning.

  ACT predicts entire action *chunks* (sequences of future actions) rather than
  single timestep actions. This drastically reduces compounding error in
  imitation learning: small mistakes don't propagate across every timestep.
  A Conditional VAE (CVAE) encoder captures task variation (e.g. "pick from
  left" vs "pick from right"), and a Transformer decoder autoregressively
  generates the action chunk conditioned on image observations and the latent
  style variable z.

  ## Motivation

  Standard behavior cloning predicts `a_t` from `o_t` at each timestep. Errors
  compound over the trajectory. ACT instead predicts `[a_t, a_{t+1}, ...,
  a_{t+chunk_size-1}]` in one shot, executing only the first few actions
  before re-planning. The CVAE latent z captures multimodal action
  distributions (different valid ways to perform a task).

  ## Architecture

  ```
  Training:
    obs [batch, obs_dim]  +  actions [batch, chunk_size, action_dim]
           |                              |
           +----------+-------------------+
                      |
                CVAE Encoder (MLP)
                      |
                 mu, log_var
                      |
             reparameterize -> z [batch, latent_dim]
                      |
           +----------+-------------------+
           |                              |
    obs [batch, obs_dim]                  z
           |                              |
           +--------> Transformer Decoder (cross-attn on obs, autoregressive)
                              |
                    pred_actions [batch, chunk_size, action_dim]

  Inference:
    z ~ N(0, I)  (no encoder needed)
    obs -> Transformer Decoder -> action chunk
  ```

  ## Usage

      {encoder, decoder} = ACT.build(
        obs_dim: 512,
        action_dim: 7,
        chunk_size: 100,
        latent_dim: 32
      )

      # Training: encode actions to get latent, decode to reconstruct
      %{mu: mu, log_var: log_var} = encoder_predict.(encoder_params, %{"obs" => obs, "actions" => actions})
      {z, key} = ACT.reparameterize(mu, log_var, key)
      pred = decoder_predict.(decoder_params, %{"obs" => obs, "z" => z})

      # Inference: sample z from prior
      z = Nx.Random.normal(key, shape: {batch, latent_dim})
      pred = decoder_predict.(decoder_params, %{"obs" => obs, "z" => z})

      # Loss
      loss = ACT.act_loss(pred_actions, target_actions, mu, log_var)

  ## References

  - Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware" (2023) — https://arxiv.org/abs/2304.13705
  - ALOHA project: https://tonyzhaozh.github.io/aloha/
  """

  import Nx.Defn

  alias Edifice.Blocks.TransformerBlock

  @default_chunk_size 100
  @default_hidden_dim 256
  @default_num_heads 8
  @default_num_layers 6
  @default_latent_dim 32
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:obs_dim, pos_integer()}
          | {:action_dim, pos_integer()}
          | {:chunk_size, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:latent_dim, pos_integer()}
          | {:dropout, float()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build the ACT model (CVAE encoder + Transformer decoder).

  ## Options

    - `:obs_dim` - Observation feature dimension (required)
    - `:action_dim` - Action dimension per timestep (required)
    - `:chunk_size` - Number of future actions to predict (default: 100)
    - `:hidden_dim` - Transformer hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of decoder layers (default: 6)
    - `:latent_dim` - CVAE latent dimension (default: 32)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    `{encoder, decoder}` tuple:
    - Encoder: inputs `"obs"` and `"actions"` -> `%{mu: ..., log_var: ...}`
    - Decoder: inputs `"obs"` and `"z"` -> `[batch, chunk_size, action_dim]`
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    encoder = build_encoder(opts)
    decoder = build_decoder(opts)
    {encoder, decoder}
  end

  @doc """
  Build the CVAE encoder.

  Maps (observation, action_sequence) to a latent distribution (mu, log_var).

  ## Options

    Same as `build/1`.

  ## Returns

    Axon model with inputs `"obs"` `[batch, obs_dim]` and `"actions"`
    `[batch, chunk_size, action_dim]`, outputting `%{mu: ..., log_var: ...}`.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    obs_dim = Keyword.fetch!(opts, :obs_dim)
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)

    obs = Axon.input("obs", shape: {nil, obs_dim})
    actions = Axon.input("actions", shape: {nil, chunk_size, action_dim})

    # Flatten actions: [batch, chunk_size * action_dim]
    actions_flat =
      Axon.nx(
        actions,
        fn a ->
          {batch, cs, ad} = Nx.shape(a)
          Nx.reshape(a, {batch, cs * ad})
        end,
        name: "flatten_actions"
      )

    # Concatenate obs and flattened actions
    concat = Axon.concatenate([obs, actions_flat], axis: -1, name: "encoder_concat")

    # MLP encoder trunk
    trunk =
      concat
      |> Axon.dense(hidden_dim, name: "encoder_dense_1")
      |> Axon.activation(:relu, name: "encoder_act_1")
      |> Axon.dense(hidden_dim, name: "encoder_dense_2")
      |> Axon.activation(:relu, name: "encoder_act_2")

    # Latent distribution parameters
    mu = Axon.dense(trunk, latent_dim, name: "mu")
    log_var = Axon.dense(trunk, latent_dim, name: "log_var")

    Axon.container(%{mu: mu, log_var: log_var})
  end

  @doc """
  Build the Transformer decoder.

  Takes observation features and latent z, outputs action chunk.

  ## Options

    Same as `build/1`.

  ## Returns

    Axon model with inputs `"obs"` `[batch, obs_dim]` and `"z"`
    `[batch, latent_dim]`, outputting `[batch, chunk_size, action_dim]`.
  """
  @spec build_decoder([build_opt()]) :: Axon.t()
  def build_decoder(opts \\ []) do
    obs_dim = Keyword.fetch!(opts, :obs_dim)
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    obs = Axon.input("obs", shape: {nil, obs_dim})
    z = Axon.input("z", shape: {nil, latent_dim})

    # Project obs and z to hidden_dim
    obs_proj = Axon.dense(obs, hidden_dim, name: "obs_proj")
    z_proj = Axon.dense(z, hidden_dim, name: "z_proj")

    # Combine obs and z as conditioning: [batch, 2, hidden_dim]
    # This serves as the "memory" for cross-attention in the decoder
    context =
      Axon.layer(
        fn obs_p, z_p, _opts ->
          # Stack along seq dimension: [batch, 2, hidden_dim]
          Nx.stack([obs_p, z_p], axis: 1)
        end,
        [obs_proj, z_proj],
        name: "stack_context",
        op_name: :stack_context
      )

    # Learnable query embeddings for each chunk position
    # Shape: [chunk_size, hidden_dim] broadcast to [batch, chunk_size, hidden_dim]
    query_embed =
      Axon.param("query_embed", {chunk_size, hidden_dim}, initializer: :glorot_uniform)

    queries =
      Axon.layer(
        fn _obs, query_e, _opts ->
          # Broadcast query_embed to batch: assume batch from obs
          # We need batch size, so use a trick: add dummy obs contribution
          query_e
        end,
        [obs, query_embed],
        name: "query_init",
        op_name: :query_init
      )

    # Broadcast queries to batch size
    queries =
      Axon.layer(
        fn obs_tensor, q, _opts ->
          batch_size = Nx.axis_size(obs_tensor, 0)
          Nx.broadcast(Nx.new_axis(q, 0), {batch_size, Nx.axis_size(q, 0), Nx.axis_size(q, 1)})
        end,
        [obs, queries],
        name: "broadcast_queries",
        op_name: :broadcast_queries
      )

    # Transformer decoder layers with cross-attention to context
    decoded =
      TransformerBlock.stack(queries, context, num_layers,
        attention_fn: fn x_norm, name ->
          multi_head_attention(x_norm, x_norm, x_norm, hidden_dim, num_heads, name)
        end,
        cross_attention_fn: fn q_norm, ctx, name ->
          multi_head_attention(q_norm, ctx, ctx, hidden_dim, num_heads, name)
        end,
        hidden_size: hidden_dim,
        custom_ffn: fn x_norm, name ->
          x_norm
          |> Axon.dense(hidden_dim * 4, name: "#{name}_up")
          |> Axon.activation(:gelu, name: "#{name}_act")
          |> Axon.dense(hidden_dim, name: "#{name}_down")
        end,
        dropout: dropout,
        name: "decoder_layer"
      )

    # Final projection to action space
    Axon.dense(decoded, action_dim, name: "action_head")
  end

  # Multi-head attention helper
  defp multi_head_attention(query, key, value, hidden_dim, num_heads, name) do
    head_dim = div(hidden_dim, num_heads)

    q = Axon.dense(query, hidden_dim, name: "#{name}_q")
    k = Axon.dense(key, hidden_dim, name: "#{name}_k")
    v = Axon.dense(value, hidden_dim, name: "#{name}_v")

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          compute_attention(q_t, k_t, v_t, num_heads, head_dim)
        end,
        [q, k, v],
        name: "#{name}_compute",
        op_name: :mha_compute
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out")
  end

  defp compute_attention(q, k, v, num_heads, head_dim) do
    {batch, q_len, _} = Nx.shape(q)
    {_, kv_len, _} = Nx.shape(k)

    # Reshape to [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, q_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

    # Apply to values
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, q_len, head_dim] -> [batch, q_len, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, q_len, num_heads * head_dim})
  end

  @doc """
  CVAE encoder forward pass: observation + actions -> (mu, log_var).

  Convenience wrapper that builds and runs the encoder.

  ## Parameters

    - `obs` - Observation tensor `[batch, obs_dim]`
    - `actions` - Action sequence tensor `[batch, chunk_size, action_dim]`
    - `opts` - Same options as `build/1`

  ## Returns

    `{mu, log_var}` tuple of tensors, each `[batch, latent_dim]`.
  """
  @spec encode(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def encode(obs, actions, opts) do
    # This creates an Axon subgraph for encoding
    obs_dim = Keyword.fetch!(opts, :obs_dim)
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)

    # Flatten actions
    actions_flat =
      Axon.nx(
        actions,
        fn a ->
          {batch, cs, ad} = Nx.shape(a)
          Nx.reshape(a, {batch, cs * ad})
        end,
        name: "encode_flatten_actions"
      )

    # Verify dimensions match
    _expected_flat = chunk_size * action_dim
    _expected_concat = obs_dim + chunk_size * action_dim

    concat = Axon.concatenate([obs, actions_flat], axis: -1, name: "encode_concat")

    trunk =
      concat
      |> Axon.dense(hidden_dim, name: "encode_dense_1")
      |> Axon.activation(:relu, name: "encode_act_1")
      |> Axon.dense(hidden_dim, name: "encode_dense_2")
      |> Axon.activation(:relu, name: "encode_act_2")

    mu = Axon.dense(trunk, latent_dim, name: "encode_mu")
    log_var = Axon.dense(trunk, latent_dim, name: "encode_log_var")

    Axon.container(%{mu: mu, log_var: log_var})
  end

  @doc """
  Transformer decoder forward: (obs, z) -> action_chunk.

  Convenience wrapper that creates an Axon subgraph for decoding.

  ## Parameters

    - `obs` - Axon node for observation `[batch, obs_dim]`
    - `z` - Axon node for latent `[batch, latent_dim]`
    - `opts` - Same options as `build/1`

  ## Returns

    Axon node outputting `[batch, chunk_size, action_dim]`.
  """
  @spec decode(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def decode(obs, z, opts) do
    action_dim = Keyword.fetch!(opts, :action_dim)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    obs_proj = Axon.dense(obs, hidden_dim, name: "decode_obs_proj")
    z_proj = Axon.dense(z, hidden_dim, name: "decode_z_proj")

    context =
      Axon.layer(
        fn obs_p, z_p, _opts -> Nx.stack([obs_p, z_p], axis: 1) end,
        [obs_proj, z_proj],
        name: "decode_stack_context",
        op_name: :stack_context
      )

    query_embed =
      Axon.param("decode_query_embed", {chunk_size, hidden_dim}, initializer: :glorot_uniform)

    queries =
      Axon.layer(
        fn obs_tensor, q, _opts ->
          batch_size = Nx.axis_size(obs_tensor, 0)
          Nx.broadcast(Nx.new_axis(q, 0), {batch_size, Nx.axis_size(q, 0), Nx.axis_size(q, 1)})
        end,
        [obs, query_embed],
        name: "decode_broadcast_queries",
        op_name: :broadcast_queries
      )

    decoded =
      TransformerBlock.stack(queries, context, num_layers,
        attention_fn: fn x_norm, name ->
          multi_head_attention(x_norm, x_norm, x_norm, hidden_dim, num_heads, name)
        end,
        cross_attention_fn: fn q_norm, ctx, name ->
          multi_head_attention(q_norm, ctx, ctx, hidden_dim, num_heads, name)
        end,
        hidden_size: hidden_dim,
        custom_ffn: fn x_norm, name ->
          x_norm
          |> Axon.dense(hidden_dim * 4, name: "#{name}_up")
          |> Axon.activation(:gelu, name: "#{name}_act")
          |> Axon.dense(hidden_dim, name: "#{name}_down")
        end,
        dropout: dropout,
        name: "decode_layer"
      )

    Axon.dense(decoded, action_dim, name: "decode_action_head")
  end

  # ============================================================================
  # Loss and Reparameterization
  # ============================================================================

  @doc """
  Reparameterization trick: sample z from q(z|x) = N(mu, sigma^2).

  Computes `z = mu + eps * exp(0.5 * log_var)` where `eps ~ N(0, I)`.

  ## Parameters

    - `mu` - Mean `[batch, latent_dim]`
    - `log_var` - Log variance `[batch, latent_dim]`
    - `key` - PRNG key

  ## Returns

    `{z, new_key}` tuple.
  """
  @spec reparameterize(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  defn reparameterize(mu, log_var, key) do
    std = Nx.exp(0.5 * log_var)
    {eps, key} = Nx.Random.normal(key, shape: Nx.shape(mu), type: Nx.type(mu))
    {mu + eps * std, key}
  end

  @doc """
  ACT loss: MSE reconstruction + beta * KL divergence.

  ## Parameters

    - `pred_actions` - Predicted action chunk `[batch, chunk_size, action_dim]`
    - `target_actions` - Ground truth actions `[batch, chunk_size, action_dim]`
    - `mu` - Encoder mean `[batch, latent_dim]`
    - `log_var` - Encoder log variance `[batch, latent_dim]`

  ## Options

    - `:beta` - KL weight (default: 1.0)

  ## Returns

    Scalar loss tensor.
  """
  @spec act_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def act_loss(pred_actions, target_actions, mu, log_var, opts \\ []) do
    beta = Keyword.get(opts, :beta, 1.0)
    do_act_loss(pred_actions, target_actions, mu, log_var, beta)
  end

  defnp do_act_loss(pred_actions, target_actions, mu, log_var, beta) do
    # MSE reconstruction loss
    mse = Nx.mean(Nx.pow(pred_actions - target_actions, 2))

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_per_sample = -0.5 * Nx.sum(1.0 + log_var - Nx.pow(mu, 2) - Nx.exp(log_var), axes: [-1])
    kl = Nx.mean(kl_per_sample)

    mse + beta * kl
  end

  @doc "Get output size (action_dim * chunk_size flattened, or action_dim per step)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts) do
    Keyword.fetch!(opts, :action_dim)
  end
end
