defmodule Edifice.Generative.JanusFlowTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.JanusFlow

  @opts [
    latent_channels: 4,
    latent_size: 8,
    encoder_dim: 16,
    hidden_size: 32,
    num_heads: 4,
    num_layers: 2,
    text_seq_len: 4,
    num_convnext_blocks: 1,
    dropout: 0.0,
    patch_size: 2
  ]

  describe "build/1" do
    test "returns {model, nil} tuple" do
      {model, nil} = JanusFlow.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      {model, nil} = JanusFlow.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "z_t" => Nx.divide(Nx.iota({2, 8, 8, 4}, type: :f32), 2048),
        "timestep" => Nx.tensor([0.5, 0.3]),
        "text_embed" => Nx.divide(Nx.iota({2, 4, 32}, type: :f32), 256)
      }

      params = init_fn.(input, %{})
      out = predict_fn.(params, input)
      # Output should match input latent shape: [batch, H, W, latent_ch]
      assert Nx.shape(out) == {2, 8, 8, 4}
    end

    test "outputs are finite" do
      {model, nil} = JanusFlow.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "z_t" => Nx.divide(Nx.iota({2, 8, 8, 4}, type: :f32), 2048),
        "timestep" => Nx.tensor([0.5, 0.3]),
        "text_embed" => Nx.divide(Nx.iota({2, 4, 32}, type: :f32), 256)
      }

      params = init_fn.(input, %{})
      out = predict_fn.(params, input)
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      {model, nil} = JanusFlow.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "z_t" => Nx.divide(Nx.iota({1, 8, 8, 4}, type: :f32), 1024),
        "timestep" => Nx.tensor([0.5]),
        "text_embed" => Nx.divide(Nx.iota({1, 4, 32}, type: :f32), 128)
      }

      params = init_fn.(input, %{})
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {1, 8, 8, 4}
    end

    test "two ConvNeXt blocks" do
      opts = Keyword.put(@opts, :num_convnext_blocks, 2)
      {model, nil} = JanusFlow.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "z_t" => Nx.divide(Nx.iota({2, 8, 8, 4}, type: :f32), 2048),
        "timestep" => Nx.tensor([0.5, 0.3]),
        "text_embed" => Nx.divide(Nx.iota({2, 4, 32}, type: :f32), 256)
      }

      params = init_fn.(input, %{})
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 8, 8, 4}
    end

    test "different latent_channels" do
      opts = Keyword.put(@opts, :latent_channels, 8)
      {model, nil} = JanusFlow.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "z_t" => Nx.divide(Nx.iota({2, 8, 8, 8}, type: :f32), 4096),
        "timestep" => Nx.tensor([0.5, 0.3]),
        "text_embed" => Nx.divide(Nx.iota({2, 4, 32}, type: :f32), 256)
      }

      params = init_fn.(input, %{})
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 8, 8, 8}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert JanusFlow.output_size(hidden_size: 256) == 256
    end

    test "uses default" do
      assert JanusFlow.output_size([]) == 128
    end
  end

  describe "Edifice.build/2" do
    test "builds via registry" do
      {model, nil} = Edifice.build(:janus_flow, @opts)
      assert %Axon{} = model
    end
  end
end
