defmodule Edifice.OutputSizeSweepTest do
  @moduledoc """
  Discovers all architecture modules that export output_size/1 and verifies
  each returns a positive integer (or tuple for special cases like GaussianSplat).
  Pure keyword functions — runs in milliseconds.
  """
  use ExUnit.Case, async: true

  # Broad opts that satisfy most output_size/1 signatures.
  @test_opts [
    embed_dim: 16,
    hidden_size: 8,
    input_size: 16,
    input_dim: 16,
    obs_size: 16,
    num_layers: 1,
    num_heads: 2,
    num_kv_heads: 2,
    head_dim: 4,
    seq_len: 4,
    window_size: 4,
    state_size: 4,
    dropout: 0.0,
    image_size: 16,
    in_channels: 1,
    patch_size: 4,
    depth: 1,
    num_classes: 4,
    output_size: 4,
    latent_size: 4,
    action_dim: 4,
    action_horizon: 4,
    encoder_dim: 16,
    projection_dim: 8,
    num_experts: 2,
    vocab_size: 32,
    hidden_sizes: [8],
    rank: 4,
    num_codebooks: 2,
    memory_size: 4,
    memory_dim: 4,
    num_memories: 4,
    num_filters: 8,
    num_rbf: 8,
    edge_dim: 4,
    in_node_features: 8,
    hidden_dim: 8,
    num_encoder_layers: 1,
    num_decoder_layers: 1,
    ffn_dim: 8,
    num_queries: 4,
    n_mels: 16,
    audio_len: 4,
    conditioning_size: 16,
    target_layer_sizes: [{16, 8}],
    action_size: 4,
    state_dim: 16,
    context_len: 4,
    obs_dim: 16,
    latent_dim: 4,
    chunk_size: 4,
    coord_dim: 3,
    dir_dim: 3,
    num_gaussians: 4,
    img_dim: 16,
    txt_dim: 16,
    img_tokens: 4,
    txt_tokens: 4,
    feature_dim: 8,
    text_hidden_size: 8,
    predictor_embed_dim: 8,
    encoder_depth: 1,
    predictor_depth: 1,
    num_patches: 4,
    decoder_depth: 1,
    decoder_num_heads: 2,
    num_predictions: 2,
    max_byte_len: 4,
    scorer_hidden: 8,
    head_hidden: 8,
    teacher_dim: 8,
    vision_dim: 16,
    llm_dim: 8,
    num_visual_tokens: 4,
    text_seq_len: 4,
    out_channels: 1,
    hidden_channels: 8,
    modes: 4,
    num_sensors: 8,
    key_dim: 16,
    value_dim: 8,
    num_tables: 2,
    num_buckets: 4,
    block_size: 2,
    pe_dim: 4,
    rwse_walk_length: 4,
    scales: [1, 2],
    base_dim: 16,
    num_interactions: 1,
    reservoir_size: 8,
    head_size: 4,
    dims: [8],
    depths: [1],
    num_latents: 4
  ]

  # Architecture-specific overrides for output_size/1
  @output_size_overrides %{
    gaussian_splat: [num_gaussians: 4, image_size: 8]
  }

  # Build list at compile time with Code.ensure_loaded
  @archs_with_output_size (for arch <- Edifice.list_architectures(),
                               module = Edifice.module_for(arch),
                               Code.ensure_loaded?(module),
                               function_exported?(module, :output_size, 1) do
                             {arch, module}
                           end)

  for {arch, module} <- @archs_with_output_size do
    test "#{arch} output_size/1 returns valid size" do
      module = unquote(module)
      arch = unquote(arch)
      overrides = Map.get(@output_size_overrides, arch, [])
      opts = Keyword.merge(@test_opts, overrides)

      result = module.output_size(opts)

      assert valid_output_size?(result),
             "#{arch}: output_size/1 returned #{inspect(result)}, " <>
               "expected pos_integer or tuple of pos_integers"
    end
  end

  defp valid_output_size?(n) when is_integer(n) and n > 0, do: true
  # Some architectures return atoms for dynamic/grid-dependent sizes
  defp valid_output_size?(a) when is_atom(a), do: true

  defp valid_output_size?(t) when is_tuple(t) do
    t |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 > 0))
  end

  defp valid_output_size?(_), do: false
end
