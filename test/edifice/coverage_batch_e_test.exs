defmodule Edifice.CoverageBatchETest do
  @moduledoc """
  Coverage tests for mid-coverage modules: MultiHead, Mamba, Hybrid, GQA, SNN, HGRN, S5, LinearTransformer.

  Targets uncovered code paths: utility functions, alternative attention modes,
  scan algorithm branches, build option variants, and cache initialization.
  """
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  @batch 2
  @seq_len 8
  @embed 16

  # ==========================================================================
  # MultiHead (70.81%) - Need more attention variants and build paths
  # ==========================================================================
  describe "MultiHead build/1 with hidden_size mapping" do
    alias Edifice.Attention.MultiHead

    test "build/1 maps hidden_size to head_dim" do
      # build/1 with :hidden_size option triggers the mapping branch
      model =
        MultiHead.build(
          embed_dim: @embed,
          hidden_size: 16,
          num_heads: 2,
          window_size: @seq_len,
          seq_len: @seq_len,
          num_layers: 1
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      # hidden_size = 16 => head_dim = 16 / 2 = 8; output is [batch, num_heads * head_dim]
      assert Nx.shape(output) == {@batch, 16}
    end

    test "build/1 without hidden_size uses default head_dim" do
      model =
        MultiHead.build(
          embed_dim: @embed,
          num_heads: 2,
          head_dim: 8,
          window_size: @seq_len,
          seq_len: @seq_len,
          num_layers: 1
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 16}
    end
  end

  describe "MultiHead build_hybrid/1" do
    alias Edifice.Attention.MultiHead

    test "builds and runs hybrid LSTM + attention model" do
      model =
        MultiHead.build_hybrid(
          embed_dim: @embed,
          lstm_hidden: @embed,
          lstm_layers: 1,
          num_heads: 2,
          head_dim: 8,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 16}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "MultiHead build_hybrid_mlp/1" do
    alias Edifice.Attention.MultiHead

    test "builds and runs hybrid model with MLP layers" do
      model =
        MultiHead.build_hybrid_mlp(
          embed_dim: @embed,
          lstm_hidden: @embed,
          lstm_layers: 1,
          num_heads: 2,
          head_dim: 8,
          dropout: 0.0,
          mlp_sizes: [16, 8],
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      # Last MLP layer is size 8
      assert Nx.shape(output) == {@batch, 8}
    end
  end

  describe "MultiHead self_attention chunked path" do
    alias Edifice.Attention.MultiHead

    test "self_attention with chunked: true" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        MultiHead.self_attention(input,
          hidden_size: @embed,
          num_heads: 2,
          chunked: true,
          chunk_size: 4,
          causal: true
        )

      assert %Axon{} = layer

      {init_fn, predict_fn} = Axon.build(layer)
      inp = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end

    test "self_attention with memory_efficient: true" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        MultiHead.self_attention(input,
          hidden_size: @embed,
          num_heads: 2,
          memory_efficient: true,
          chunk_size: 4,
          causal: true
        )

      assert %Axon{} = layer

      {init_fn, predict_fn} = Axon.build(layer)
      inp = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end

    test "self_attention with qk_layernorm and chunked" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        MultiHead.self_attention(input,
          hidden_size: @embed,
          num_heads: 2,
          qk_layernorm: true,
          chunked: true,
          chunk_size: 4,
          causal: true
        )

      assert %Axon{} = layer

      {init_fn, predict_fn} = Axon.build(layer)
      inp = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end
  end

  describe "MultiHead sliding_window_attention variants" do
    alias Edifice.Attention.MultiHead

    test "sliding_window_attention with chunked: true" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        MultiHead.sliding_window_attention(input,
          window_size: @seq_len,
          num_heads: 2,
          head_dim: div(@embed, 2),
          chunked: true,
          chunk_size: 4
        )

      assert %Axon{} = layer

      {init_fn, predict_fn} = Axon.build(layer)
      inp = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end

    test "sliding_window_attention with memory_efficient: true" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        MultiHead.sliding_window_attention(input,
          window_size: @seq_len,
          num_heads: 2,
          head_dim: div(@embed, 2),
          memory_efficient: true,
          chunk_size: 4
        )

      assert %Axon{} = layer

      {init_fn, predict_fn} = Axon.build(layer)
      inp = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end

    test "sliding_window_attention with qk_layernorm: true" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        MultiHead.sliding_window_attention(input,
          window_size: @seq_len,
          num_heads: 2,
          head_dim: div(@embed, 2),
          qk_layernorm: true
        )

      assert %Axon{} = layer

      {init_fn, predict_fn} = Axon.build(layer)
      inp = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)
      assert Nx.shape(output) == {@batch, @seq_len, @embed}
    end
  end

  describe "MultiHead chunked_attention with 3D mask" do
    alias Edifice.Attention.MultiHead

    test "chunked_attention with 3D batch mask" do
      q = Nx.broadcast(0.1, {@batch, @seq_len, @embed}) |> Nx.as_type(:f32)
      k = Nx.broadcast(0.1, {@batch, @seq_len, @embed}) |> Nx.as_type(:f32)
      v = Nx.broadcast(0.5, {@batch, @seq_len, @embed}) |> Nx.as_type(:f32)

      # 3D mask: [batch, seq_q, seq_k]
      mask =
        MultiHead.causal_mask(@seq_len)
        |> Nx.broadcast({@batch, @seq_len, @seq_len})

      result = MultiHead.chunked_attention(q, k, v, chunk_size: 4, mask: mask)
      assert Nx.shape(result) == {@batch, @seq_len, @embed}
    end
  end

  # ==========================================================================
  # Mamba (70.83%) - Need scan algorithm branches and utility functions
  # ==========================================================================
  describe "Mamba sequential scan (short seq_len <= 32)" do
    alias Edifice.SSM.Mamba

    test "build with short seq_len triggers sequential scan" do
      # seq_len 8 <= 32, so sequential scan is used
      model =
        Mamba.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "Mamba Blelloch scan (long seq_len > 32)" do
    alias Edifice.SSM.Mamba

    @tag timeout: 600_000
    test "build with long seq_len triggers Blelloch scan" do
      long_seq = 64

      model =
        Mamba.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          window_size: long_seq,
          seq_len: long_seq
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, long_seq, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "Mamba utility functions" do
    alias Edifice.SSM.Mamba

    test "param_count/1 returns positive integer" do
      count =
        Mamba.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "recommended_defaults/0 returns keyword list" do
      defaults = Mamba.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :state_size)
      assert Keyword.has_key?(defaults, :window_size)
    end

    test "output_size/1 returns hidden_size" do
      assert Mamba.output_size(hidden_size: 64) == 64
    end

    test "output_size/0 returns default hidden_size" do
      assert Mamba.output_size() == 256
    end
  end

  # ==========================================================================
  # Hybrid (70.99%) - Need alternative attention modes and utility functions
  # ==========================================================================
  describe "Hybrid with use_sliding_window: false (full attention)" do
    alias Edifice.SSM.Hybrid

    test "build with full attention (no sliding window)" do
      model =
        Hybrid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          attention_every: 3,
          state_size: 4,
          num_heads: 2,
          head_dim: div(@embed, 2),
          window_size: @seq_len,
          seq_len: @seq_len,
          use_sliding_window: false,
          dropout: 0.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "Hybrid with pre_norm: false (post-norm variant)" do
    alias Edifice.SSM.Hybrid

    test "build with post-norm" do
      model =
        Hybrid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          attention_every: 3,
          state_size: 4,
          num_heads: 2,
          head_dim: div(@embed, 2),
          window_size: @seq_len,
          seq_len: @seq_len,
          pre_norm: false,
          dropout: 0.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  describe "Hybrid with chunked and memory-efficient attention" do
    alias Edifice.SSM.Hybrid

    test "build with chunked_attention: true" do
      model =
        Hybrid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          attention_every: 3,
          state_size: 4,
          num_heads: 2,
          head_dim: div(@embed, 2),
          window_size: @seq_len,
          seq_len: @seq_len,
          chunked_attention: true,
          chunk_size: 4,
          dropout: 0.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with memory_efficient_attention: true" do
      model =
        Hybrid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          attention_every: 3,
          state_size: 4,
          num_heads: 2,
          head_dim: div(@embed, 2),
          window_size: @seq_len,
          seq_len: @seq_len,
          memory_efficient_attention: true,
          chunk_size: 4,
          dropout: 0.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  describe "Hybrid with embed_dim == hidden_size (no input projection)" do
    alias Edifice.SSM.Hybrid

    test "build skips input projection when embed == hidden" do
      model =
        Hybrid.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          attention_every: 3,
          state_size: 4,
          num_heads: 2,
          head_dim: div(@embed, 2),
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  describe "Hybrid utility functions" do
    alias Edifice.SSM.Hybrid

    test "layer_pattern/1 returns correct pattern" do
      pattern = Hybrid.layer_pattern(num_layers: 6, attention_every: 3)
      assert pattern == [:mamba, :mamba, :attention, :mamba, :mamba, :attention]
    end

    test "layer_pattern/1 with defaults" do
      pattern = Hybrid.layer_pattern()
      assert is_list(pattern)
      assert length(pattern) == 6
      assert :mamba in pattern
      assert :attention in pattern
    end

    test "param_count/1 returns positive integer" do
      count =
        Hybrid.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 6,
          attention_every: 3,
          state_size: 4,
          num_heads: 2,
          head_dim: div(@embed, 2)
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count/1 with embed_dim == hidden_size (no input proj)" do
      count =
        Hybrid.param_count(
          embed_dim: 256,
          hidden_size: 256,
          num_layers: 3
        )

      assert is_integer(count)
      assert count > 0
    end

    test "recommended_defaults/0 returns keyword list" do
      defaults = Hybrid.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :attention_every)
      assert Keyword.has_key?(defaults, :pre_norm)
      assert Keyword.has_key?(defaults, :qk_layernorm)
    end

    test "output_size/1 returns hidden_size" do
      assert Hybrid.output_size(hidden_size: 64) == 64
    end

    test "output_size/0 returns default" do
      assert Hybrid.output_size() == 256
    end
  end

  describe "Hybrid build_mamba_layer/2 post-norm branch" do
    alias Edifice.SSM.Hybrid

    test "build_mamba_layer with pre_norm: false triggers post-norm" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        Hybrid.build_mamba_layer(input,
          hidden_size: @embed,
          state_size: 4,
          expand_factor: 2,
          conv_size: 4,
          dropout: 0.0,
          pre_norm: false,
          name: "test_mamba"
        )

      assert %Axon{} = layer
    end

    test "build_mamba_layer with dropout: 0 skips dropout" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        Hybrid.build_mamba_layer(input,
          hidden_size: @embed,
          state_size: 4,
          dropout: 0.0,
          pre_norm: true,
          name: "test_mamba_nodrop"
        )

      assert %Axon{} = layer
    end
  end

  describe "Hybrid build_attention_layer/2 variants" do
    alias Edifice.SSM.Hybrid

    test "build_attention_layer with pre_norm: false" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        Hybrid.build_attention_layer(input,
          hidden_size: @embed,
          attn_hidden_dim: @embed,
          num_heads: 2,
          head_dim: div(@embed, 2),
          dropout: 0.0,
          pre_norm: false,
          use_sliding_window: false,
          name: "test_attn_postnorm"
        )

      assert %Axon{} = layer
    end

    test "build_attention_layer with different attn_hidden_dim triggers projections" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        Hybrid.build_attention_layer(input,
          hidden_size: @embed,
          attn_hidden_dim: 32,
          num_heads: 2,
          head_dim: 16,
          dropout: 0.0,
          pre_norm: true,
          use_sliding_window: false,
          name: "test_attn_proj"
        )

      assert %Axon{} = layer
    end

    test "build_attention_layer with dropout: 0 skips dropout layers" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        Hybrid.build_attention_layer(input,
          hidden_size: @embed,
          num_heads: 2,
          head_dim: div(@embed, 2),
          dropout: 0.0,
          pre_norm: true,
          use_sliding_window: true,
          window_size: @seq_len,
          name: "test_attn_nodrop"
        )

      assert %Axon{} = layer
    end
  end

  # ==========================================================================
  # GQA (72.13%) - Need utility functions
  # ==========================================================================
  describe "GQA utility functions" do
    alias Edifice.Attention.GQA

    test "param_count/1 returns positive integer" do
      count =
        GQA.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          num_heads: 4,
          num_kv_heads: 2,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count/1 with embed_dim == hidden_size (no input proj)" do
      count =
        GQA.param_count(
          embed_dim: 256,
          hidden_size: 256,
          num_heads: 8,
          num_kv_heads: 2,
          num_layers: 4
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count/1 with embed_dim != hidden_size (with input proj)" do
      count_with_proj =
        GQA.param_count(
          embed_dim: 32,
          hidden_size: 256,
          num_heads: 8,
          num_kv_heads: 2,
          num_layers: 4
        )

      count_without_proj =
        GQA.param_count(
          embed_dim: 256,
          hidden_size: 256,
          num_heads: 8,
          num_kv_heads: 2,
          num_layers: 4
        )

      assert count_with_proj > count_without_proj
    end

    test "recommended_defaults/0 returns keyword list" do
      defaults = GQA.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_kv_heads)
      assert Keyword.has_key?(defaults, :num_layers)
    end

    test "output_size/0 returns default hidden_size" do
      assert GQA.output_size() == 256
    end
  end

  describe "GQA build with different num_kv_heads" do
    alias Edifice.Attention.GQA

    test "build with num_kv_heads: 1 (MQA mode)" do
      model =
        GQA.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_heads: 4,
          num_kv_heads: 1,
          num_layers: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with num_kv_heads == num_heads (MHA mode)" do
      model =
        GQA.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_heads: 4,
          num_kv_heads: 4,
          num_layers: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # SNN (74.29%) - Need different timesteps and hidden_sizes
  # ==========================================================================
  describe "SNN with different timesteps" do
    alias Edifice.Neuromorphic.SNN

    test "build with num_timesteps: 5" do
      model =
        SNN.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          num_timesteps: 5,
          tau: 2.0,
          threshold: 1.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"input" => Nx.broadcast(0.5, {@batch, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 4}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "build with num_timesteps: 50" do
      model =
        SNN.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          num_timesteps: 50,
          tau: 2.0,
          threshold: 1.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"input" => Nx.broadcast(0.5, {@batch, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 4}
    end

    test "build with different tau (fast decay)" do
      model =
        SNN.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4,
          num_timesteps: 10,
          tau: 0.5,
          threshold: 0.5
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"input" => Nx.broadcast(0.5, {@batch, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 4}
    end

    test "build with multiple hidden layers" do
      model =
        SNN.build(
          input_size: @embed,
          hidden_sizes: [16, 8, 4],
          output_size: 2,
          num_timesteps: 10,
          tau: 2.0,
          threshold: 1.0
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"input" => Nx.broadcast(0.5, {@batch, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 2}
    end
  end

  describe "SNN standalone functions" do
    alias Edifice.Neuromorphic.SNN

    test "lif_neuron/4 returns valid membrane and spikes" do
      membrane = Nx.broadcast(0.0, {2, 4})
      input_current = Nx.broadcast(0.8, {2, 4})

      {new_membrane, spikes} = SNN.lif_neuron(membrane, input_current, 0.9, 1.0)

      assert Nx.shape(new_membrane) == {2, 4}
      assert Nx.shape(spikes) == {2, 4}
      assert Nx.all(Nx.is_nan(new_membrane) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "surrogate_gradient/2 returns values in [0, 1]" do
      x = Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
      result = SNN.surrogate_gradient(x, 10.0)
      assert Nx.shape(result) == {5}

      # All values should be between 0 and 1 (sigmoid output)
      assert Nx.all(Nx.greater_equal(result, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(result, 1.0)) |> Nx.to_number() == 1
    end

    test "rate_decode/1 averages over timestep axis" do
      # [batch, timesteps, hidden]
      spike_train = Nx.broadcast(0.5, {2, 10, 4})
      result = SNN.rate_decode(spike_train)
      assert Nx.shape(result) == {2, 4}
    end
  end

  # ==========================================================================
  # HGRN (74.49%) - Need utility functions and init_cache
  # ==========================================================================
  describe "HGRN utility functions" do
    alias Edifice.Attention.HGRN

    test "param_count/1 returns positive integer" do
      count =
        HGRN.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 3,
          state_expansion: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count/1 with embed == hidden (no input proj)" do
      count_same =
        HGRN.param_count(
          embed_dim: 256,
          hidden_size: 256,
          num_layers: 3,
          state_expansion: 2
        )

      count_diff =
        HGRN.param_count(
          embed_dim: 32,
          hidden_size: 256,
          num_layers: 3,
          state_expansion: 2
        )

      assert count_diff > count_same
    end

    test "recommended_defaults/0 returns keyword list" do
      defaults = HGRN.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :state_expansion)
      assert Keyword.has_key?(defaults, :num_layers)
    end

    test "output_size/0 returns default" do
      assert HGRN.output_size() == 256
    end
  end

  describe "HGRN init_cache/1" do
    alias Edifice.Attention.HGRN

    test "init_cache returns properly structured map" do
      cache = HGRN.init_cache(batch_size: 2, hidden_size: 16, state_expansion: 2, num_layers: 3)

      assert is_map(cache)
      assert cache.step == 0
      assert is_map(cache.layers)
      assert map_size(cache.layers) == 3

      # Each layer should have an h tensor
      layer_1 = cache.layers["layer_1"]
      assert is_map(layer_1)
      assert Nx.shape(layer_1.h) == {2, 32}
    end

    test "init_cache with defaults" do
      cache = HGRN.init_cache()
      assert cache.step == 0
      assert cache.config.hidden_size == 256
      assert cache.config.state_expansion == 2
      assert cache.config.num_layers == 6
      assert map_size(cache.layers) == 6
    end
  end

  describe "HGRN build with different state_expansion" do
    alias Edifice.Attention.HGRN

    test "build with state_expansion: 1" do
      model =
        HGRN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          state_expansion: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with state_expansion: 4" do
      model =
        HGRN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          state_expansion: 4,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # S5 (76.09%) - Need utility functions and init_cache
  # ==========================================================================
  describe "S5 utility functions" do
    alias Edifice.SSM.S5

    test "param_count/1 returns positive integer" do
      count =
        S5.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 8,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count/1 with embed == hidden (no input proj)" do
      count_same =
        S5.param_count(
          embed_dim: 256,
          hidden_size: 256,
          state_size: 64,
          num_layers: 4
        )

      count_diff =
        S5.param_count(
          embed_dim: 32,
          hidden_size: 256,
          state_size: 64,
          num_layers: 4
        )

      assert count_diff > count_same
    end

    test "recommended_defaults/0 returns keyword list" do
      defaults = S5.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :state_size)
      assert Keyword.has_key?(defaults, :num_layers)
    end

    test "output_size/0 returns default" do
      assert S5.output_size() == 256
    end
  end

  describe "S5 init_cache/1" do
    alias Edifice.SSM.S5

    test "init_cache returns properly structured map" do
      cache = S5.init_cache(batch_size: 2, state_size: 8, num_layers: 3)

      assert is_map(cache)
      assert cache.step == 0
      assert is_map(cache.layers)
      assert map_size(cache.layers) == 3

      # Each layer should have an h tensor
      layer_1 = cache.layers["layer_1"]
      assert is_map(layer_1)
      assert Nx.shape(layer_1.h) == {2, 8}
    end

    test "init_cache with defaults" do
      cache = S5.init_cache()
      assert cache.step == 0
      assert cache.config.state_size == 64
      assert cache.config.num_layers == 4
      assert map_size(cache.layers) == 4
    end
  end

  describe "S5 build with different state_size" do
    alias Edifice.SSM.S5

    test "build with small state_size: 4" do
      model =
        S5.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 4,
          num_layers: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with large state_size: 32" do
      model =
        S5.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 32,
          num_layers: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with embed_dim == hidden_size (skip input proj)" do
      model =
        S5.build(
          embed_dim: @embed,
          hidden_size: @embed,
          state_size: 8,
          num_layers: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # LinearTransformer (76.92%) - Need utility functions
  # ==========================================================================
  describe "LinearTransformer utility functions" do
    alias Edifice.Attention.LinearTransformer

    test "param_count/1 returns positive integer" do
      count =
        LinearTransformer.param_count(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count/1 with embed == hidden (no input proj)" do
      count_same =
        LinearTransformer.param_count(
          embed_dim: 256,
          hidden_size: 256,
          num_layers: 4
        )

      count_diff =
        LinearTransformer.param_count(
          embed_dim: 32,
          hidden_size: 256,
          num_layers: 4
        )

      assert count_diff > count_same
    end

    test "recommended_defaults/0 returns keyword list" do
      defaults = LinearTransformer.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
    end

    test "output_size/0 returns default" do
      assert LinearTransformer.output_size() == 256
    end
  end

  describe "LinearTransformer build with different num_heads" do
    alias Edifice.Attention.LinearTransformer

    test "build with num_heads: 2 (fewer heads)" do
      model =
        LinearTransformer.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          num_heads: 2,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with num_heads: 1 (single head)" do
      model =
        LinearTransformer.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          num_heads: 1,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "build with embed_dim != hidden_size" do
      model =
        LinearTransformer.build(
          embed_dim: @embed,
          hidden_size: 32,
          num_layers: 1,
          num_heads: 4,
          dropout: 0.0,
          window_size: @seq_len,
          seq_len: @seq_len
        )

      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)
      input = %{"state_sequence" => Nx.broadcast(0.5, {@batch, @seq_len, @embed})}
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 32}
    end
  end
end
