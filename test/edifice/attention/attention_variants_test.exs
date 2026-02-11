defmodule Edifice.Attention.AttentionVariantsTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.{GQA, Perceiver, FNet, LinearTransformer, Nystromformer, Performer}

  # Small dimensions for fast testing
  @batch 2
  @seq_len 8
  @embed_size 32
  @hidden_size 32

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
    input
  end

  # ============================================================================
  # GQA Tests
  # ============================================================================

  describe "GQA" do
    @gqa_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      num_heads: 4,
      num_kv_heads: 2,
      num_layers: 2,
      dropout: 0.0,
      window_size: @seq_len,
      seq_len: @seq_len
    ]

    test "build/1 returns an Axon model" do
      model = GQA.build(@gqa_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = GQA.build(@gqa_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = GQA.build(@gqa_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns hidden_size" do
      assert GQA.output_size(@gqa_opts) == @hidden_size
    end
  end

  # ============================================================================
  # Perceiver Tests
  # ============================================================================

  describe "Perceiver" do
    @perceiver_opts [
      input_dim: @embed_size,
      latent_dim: @hidden_size,
      num_latents: 8,
      num_layers: 2,
      num_cross_layers: 1,
      num_heads: 4,
      dropout: 0.0
    ]

    test "build/1 returns an Axon model" do
      model = Perceiver.build(@perceiver_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Perceiver.build(@perceiver_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Perceiver.build(@perceiver_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns latent_dim" do
      assert Perceiver.output_size(@perceiver_opts) == @hidden_size
    end
  end

  # ============================================================================
  # FNet Tests
  # ============================================================================

  describe "FNet" do
    @fnet_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      num_layers: 2,
      dropout: 0.0,
      window_size: @seq_len,
      seq_len: @seq_len
    ]

    test "build/1 returns an Axon model" do
      model = FNet.build(@fnet_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = FNet.build(@fnet_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = FNet.build(@fnet_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns hidden_size" do
      assert FNet.output_size(@fnet_opts) == @hidden_size
    end
  end

  # ============================================================================
  # LinearTransformer Tests
  # ============================================================================

  describe "LinearTransformer" do
    @linear_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      num_layers: 2,
      num_heads: 4,
      dropout: 0.0,
      window_size: @seq_len,
      seq_len: @seq_len
    ]

    test "build/1 returns an Axon model" do
      model = LinearTransformer.build(@linear_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = LinearTransformer.build(@linear_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = LinearTransformer.build(@linear_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns hidden_size" do
      assert LinearTransformer.output_size(@linear_opts) == @hidden_size
    end
  end

  # ============================================================================
  # Nystromformer Tests
  # ============================================================================

  describe "Nystromformer" do
    @nystrom_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      num_landmarks: 4,
      num_layers: 2,
      num_heads: 4,
      dropout: 0.0,
      window_size: @seq_len,
      seq_len: @seq_len
    ]

    test "build/1 returns an Axon model" do
      model = Nystromformer.build(@nystrom_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Nystromformer.build(@nystrom_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Nystromformer.build(@nystrom_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns hidden_size" do
      assert Nystromformer.output_size(@nystrom_opts) == @hidden_size
    end
  end

  # ============================================================================
  # Performer Tests
  # ============================================================================

  describe "Performer" do
    @performer_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      num_features: 16,
      num_layers: 2,
      num_heads: 4,
      dropout: 0.0,
      window_size: @seq_len,
      seq_len: @seq_len
    ]

    test "build/1 returns an Axon model" do
      model = Performer.build(@performer_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Performer.build(@performer_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Performer.build(@performer_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns hidden_size" do
      assert Performer.output_size(@performer_opts) == @hidden_size
    end
  end
end
