defmodule Edifice.Blocks.AdaptiveNorm do
  @moduledoc """
  Adaptive Layer Normalization (AdaLN / AdaLN-Zero).

  Conditional normalization where scale and shift parameters are predicted
  from a conditioning signal (e.g., timestep embedding, class label). Used
  in Diffusion Transformers (DiT) and class-conditional generation.

  ## Variants

  - **AdaLN**: Replace fixed gamma/beta with condition-predicted parameters
  - **AdaLN-Zero**: Also predict a gating factor alpha, initialized to zero

  ## Formula

      AdaLN(x, c) = gamma(c) * LayerNorm(x) + beta(c)
      AdaLN-Zero(x, c) = alpha(c) * (gamma(c) * LayerNorm(x) + beta(c))

  ## Usage

      # AdaLN conditioning on timestep embedding
      output = AdaptiveNorm.layer(input, condition,
        hidden_size: 256,
        mode: :adaln_zero
      )

  ## References
  - "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
  - https://arxiv.org/abs/2212.09748
  """

  require Axon

  @doc """
  Build an AdaLN / AdaLN-Zero layer.

  ## Parameters
    - `input` - Input tensor Axon node [batch, ..., hidden_size]
    - `condition` - Conditioning signal Axon node [batch, cond_dim]

  ## Options
    - `:hidden_size` - Feature dimension (required)
    - `:mode` - :adaln or :adaln_zero (default: :adaln_zero)
    - `:name` - Layer name prefix (default: "adaptive_norm")
  """
  @spec layer(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def layer(input, condition, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    mode = Keyword.get(opts, :mode, :adaln_zero)
    name = Keyword.get(opts, :name, "adaptive_norm")

    # Predict scale, shift (and gate for adaln_zero) from condition
    num_params = if mode == :adaln_zero, do: hidden_size * 3, else: hidden_size * 2
    params = Axon.dense(condition, num_params, name: "#{name}_params")

    # Apply LayerNorm then modulate
    normed = Axon.layer_norm(input, name: "#{name}_ln")

    Axon.layer(
      &adaptive_norm_impl/3,
      [normed, params],
      name: name,
      hidden_size: hidden_size,
      mode: mode,
      op_name: :adaptive_norm
    )
  end

  defp adaptive_norm_impl(normed, params, opts) do
    hidden_size = opts[:hidden_size]
    mode = opts[:mode] || :adaln_zero

    if mode == :adaln_zero do
      gamma = Nx.slice_along_axis(params, 0, hidden_size, axis: -1)
      beta = Nx.slice_along_axis(params, hidden_size, hidden_size, axis: -1)
      alpha = Nx.slice_along_axis(params, hidden_size * 2, hidden_size, axis: -1)

      # Expand for broadcasting if normed has more dims
      {gamma, beta, alpha} =
        if tuple_size(Nx.shape(normed)) == 3 do
          {Nx.new_axis(gamma, 1), Nx.new_axis(beta, 1), Nx.new_axis(alpha, 1)}
        else
          {gamma, beta, alpha}
        end

      modulated = Nx.add(Nx.multiply(Nx.add(1.0, gamma), normed), beta)
      Nx.multiply(alpha, modulated)
    else
      gamma = Nx.slice_along_axis(params, 0, hidden_size, axis: -1)
      beta = Nx.slice_along_axis(params, hidden_size, hidden_size, axis: -1)

      {gamma, beta} =
        if tuple_size(Nx.shape(normed)) == 3 do
          {Nx.new_axis(gamma, 1), Nx.new_axis(beta, 1)}
        else
          {gamma, beta}
        end

      Nx.add(Nx.multiply(Nx.add(1.0, gamma), normed), beta)
    end
  end
end
