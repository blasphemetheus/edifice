defmodule Edifice.CoverageBatchCTest do
  @moduledoc "Coverage tests: VQVAE, NormalizingFlow, Zamba, FNet, ODESolver, KAN, Bayesian, ScoreSDE, ResNet, LatentDiffusion, GAT"
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  @batch 2
  @seq_len 8
  @embed 16

  # ==========================================================================
  # VQVAE (59%) - Need encoder/decoder build + forward
  # ==========================================================================
  describe "VQVAE" do
    alias Edifice.Generative.VQVAE

    test "build returns encoder/decoder tuple" do
      {encoder, decoder} =
        VQVAE.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_embeddings: 8,
          commitment_cost: 0.25
        )

      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end

    test "encoder forward pass" do
      {encoder, _decoder} =
        VQVAE.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_embeddings: 8
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      # Encoder outputs latent codes
      {b, _} = Nx.shape(output)
      assert b == @batch
    end

    test "decoder forward pass" do
      {_encoder, decoder} =
        VQVAE.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_embeddings: 8
        )

      {init_fn, predict_fn} = Axon.build(decoder)
      params = init_fn.(Nx.template({@batch, 8}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 8}))
      {b, _} = Nx.shape(output)
      assert b == @batch
    end
  end

  # ==========================================================================
  # NormalizingFlow (60%) - Need forward pass
  # ==========================================================================
  describe "NormalizingFlow" do
    alias Edifice.Generative.NormalizingFlow

    test "build returns Axon model" do
      model =
        NormalizingFlow.build(
          input_size: @embed,
          hidden_size: 16,
          num_flows: 2
        )

      assert %Axon{} = model
    end

    test "forward pass" do
      model =
        NormalizingFlow.build(
          input_size: @embed,
          hidden_size: 16,
          num_flows: 2
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      {b, _} = Nx.shape(output)
      assert b == @batch
    end
  end

  # ==========================================================================
  # Zamba (61%) - Need build + forward
  # ==========================================================================
  describe "Zamba" do
    alias Edifice.SSM.Zamba

    test "build returns Axon model" do
      model =
        Zamba.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          attention_every: 2,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "forward pass" do
      model =
        Zamba.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 2,
          attention_every: 2,
          seq_len: @seq_len,
          state_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # FNet (63%) - Need build + forward
  # ==========================================================================
  describe "FNet" do
    alias Edifice.Attention.FNet

    test "build returns Axon model" do
      model =
        FNet.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "forward pass" do
      model =
        FNet.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # KAN (65%) - Need chebyshev, fourier, rbf bases
  # ==========================================================================
  describe "KAN" do
    alias Edifice.Feedforward.KAN

    test "build with :chebyshev basis" do
      model =
        KAN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          basis: :chebyshev
        )

      assert %Axon{} = model
    end

    test "build with :fourier basis" do
      model =
        KAN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          basis: :fourier
        )

      assert %Axon{} = model
    end

    test "build with :rbf basis" do
      model =
        KAN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          basis: :rbf
        )

      assert %Axon{} = model
    end

    test "forward pass with :chebyshev basis" do
      model =
        KAN.build(
          embed_dim: @embed,
          hidden_size: @embed,
          num_layers: 1,
          seq_len: @seq_len,
          basis: :chebyshev
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed}))
      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # Bayesian (66%) - Need bayesian_dense
  # ==========================================================================
  describe "Bayesian" do
    alias Edifice.Probabilistic.Bayesian

    test "build creates model" do
      model =
        Bayesian.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4
        )

      assert %Axon{} = model
    end

    test "forward pass" do
      model =
        Bayesian.build(
          input_size: @embed,
          hidden_sizes: [8],
          output_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      assert Nx.shape(output) == {@batch, 4}
    end
  end

  # ==========================================================================
  # ScoreSDE (66%) - Need VE-SDE build
  # ==========================================================================
  describe "ScoreSDE" do
    alias Edifice.Generative.ScoreSDE

    test "build with sde_type: :ve" do
      model =
        ScoreSDE.build(
          input_dim: @embed,
          hidden_size: 16,
          num_layers: 1,
          sde_type: :ve
        )

      assert %Axon{} = model
    end

    test "forward pass with :ve SDE" do
      model =
        ScoreSDE.build(
          input_dim: @embed,
          hidden_size: 16,
          num_layers: 1,
          sde_type: :ve
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "noisy_input" => Nx.template({@batch, @embed}, :f32),
            "timestep" => Nx.template({@batch}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noisy_input" => Nx.broadcast(0.5, {@batch, @embed}),
          "timestep" => Nx.tensor([0.3, 0.7], type: :f32)
        })

      assert Nx.shape(output) == {@batch, @embed}
    end
  end

  # ==========================================================================
  # ResNet (68%) - Need bottleneck blocks
  # ==========================================================================
  describe "ResNet" do
    alias Edifice.Convolutional.ResNet

    test "build with :bottleneck block_type" do
      model =
        ResNet.build(
          input_shape: {nil, 8, 8, 3},
          num_classes: 4,
          block_sizes: [1],
          block_type: :bottleneck,
          initial_channels: 8
        )

      assert %Axon{} = model
    end

    test "forward pass with bottleneck" do
      model =
        ResNet.build(
          input_shape: {nil, 8, 8, 3},
          num_classes: 4,
          block_sizes: [1],
          block_type: :bottleneck,
          initial_channels: 8
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 8, 8, 3}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 8, 8, 3}))
      assert Nx.shape(output) == {@batch, 4}
    end

    test "build with dropout" do
      model =
        ResNet.build(
          input_shape: {nil, 8, 8, 3},
          num_classes: 4,
          block_sizes: [1],
          initial_channels: 8,
          dropout: 0.1
        )

      assert %Axon{} = model
    end
  end

  # ==========================================================================
  # LatentDiffusion (69%) - Need build + forward of encoder/decoder/denoiser
  # ==========================================================================
  describe "LatentDiffusion" do
    alias Edifice.Generative.LatentDiffusion

    test "build returns tuple of 3 models" do
      {encoder, decoder, denoiser} =
        LatentDiffusion.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      assert %Axon{} = encoder
      assert %Axon{} = decoder
      assert %Axon{} = denoiser
    end

    test "encoder forward pass returns mu/log_var map" do
      {encoder, _decoder, _denoiser} =
        LatentDiffusion.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({@batch, @embed}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @embed}))
      # Encoder returns %{mu: tensor, log_var: tensor} via Axon.container
      assert %{mu: mu, log_var: log_var} = output
      assert Nx.shape(mu) == {@batch, 8}
      assert Nx.shape(log_var) == {@batch, 8}
    end

    test "decoder forward pass" do
      {_encoder, decoder, _denoiser} =
        LatentDiffusion.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      {init_fn, predict_fn} = Axon.build(decoder)
      params = init_fn.(Nx.template({@batch, 8}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 8}))
      assert Nx.shape(output) == {@batch, @embed}
    end

    test "denoiser forward pass" do
      {_encoder, _decoder, denoiser} =
        LatentDiffusion.build(
          input_size: @embed,
          latent_size: 8,
          hidden_size: 16,
          num_layers: 1,
          num_steps: 10
        )

      {init_fn, predict_fn} = Axon.build(denoiser)

      # Denoiser inputs: "noisy_z" and "timestep"
      params =
        init_fn.(
          %{
            "noisy_z" => Nx.template({@batch, 8}, :f32),
            "timestep" => Nx.template({@batch}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noisy_z" => Nx.broadcast(0.5, {@batch, 8}),
          "timestep" => Nx.tensor([3, 5], type: :f32)
        })

      assert Nx.shape(output) == {@batch, 8}
    end
  end

  # ==========================================================================
  # GAT (71%) - Need multi-layer + forward pass
  # ==========================================================================
  describe "GAT" do
    alias Edifice.Graph.GAT

    test "build with num_layers > 1" do
      model =
        GAT.build(
          input_dim: 8,
          hidden_size: 4,
          num_heads: 2,
          num_classes: 3,
          num_layers: 3
        )

      assert %Axon{} = model
    end

    test "forward pass" do
      model =
        GAT.build(
          input_dim: 8,
          hidden_size: 4,
          num_heads: 2,
          num_classes: 3,
          num_layers: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      num_nodes = 4

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch, num_nodes, 8}, :f32),
            "adjacency" => Nx.template({@batch, num_nodes, num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      adj = Nx.eye(num_nodes) |> Nx.broadcast({@batch, num_nodes, num_nodes})
      nodes = Nx.broadcast(0.5, {@batch, num_nodes, 8})

      output = predict_fn.(params, %{"nodes" => nodes, "adjacency" => adj})
      {b, n, c} = Nx.shape(output)
      assert b == @batch
      assert n == num_nodes
      assert c == 3
    end
  end

  # ==========================================================================
  # ODESolver (65%) - Need dopri5 solver
  # ==========================================================================
  describe "ODESolver" do
    alias Edifice.Utils.ODESolver

    test "solve with :dopri5 solver" do
      # Simple linear ODE: dy/dt = -y, y(0) = 1
      # ODESolver.solve(f, t0, t1, x0, opts) — f is fn(t, x) -> dx
      dynamics_fn = fn _t, y -> Nx.negate(y) end
      y0 = Nx.tensor([1.0])

      result = ODESolver.solve(dynamics_fn, 0.0, 1.0, y0, solver: :dopri5)
      # Should approach exp(-1) ~ 0.368
      val = Nx.to_number(Nx.squeeze(result))
      assert abs(val - :math.exp(-1)) < 0.1
    end
  end
end
