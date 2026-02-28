defmodule Edifice.Misc.MiscRemainingTest do
  use ExUnit.Case, async: true
  @moduletag :misc

  alias Edifice.Convolutional.Conv
  alias Edifice.Energy.EBM
  alias Edifice.Graph.GAT
  alias Edifice.Graph.MessagePassing
  alias Edifice.Memory.NTM
  alias Edifice.Meta.Capsule
  alias Edifice.Meta.Hypernetwork
  alias Edifice.Neuromorphic.SNN
  alias Edifice.Probabilistic.Bayesian
  alias Edifice.Probabilistic.MCDropout
  alias Edifice.Sets.PointNet

  @batch_size 2

  # ============================================================================
  # Conv Tests
  # ============================================================================

  describe "Conv.build_conv1d/1" do
    @conv_input_size 8
    @conv_seq_len 10

    test "produces correct output shape" do
      model =
        Conv.build_conv1d(
          input_size: @conv_input_size,
          channels: [16, 32],
          kernel_sizes: 3
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @conv_seq_len, @conv_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @conv_seq_len, @conv_input_size})}
      output = predict_fn.(params, input)

      {b, _seq, channels} = Nx.shape(output)
      assert b == @batch_size
      assert channels == 32
    end

    test "output contains finite values" do
      model =
        Conv.build_conv1d(
          input_size: @conv_input_size,
          channels: [16],
          kernel_sizes: 3
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @conv_seq_len, @conv_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @conv_seq_len, @conv_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # GAT Tests
  # ============================================================================

  describe "GAT.build/1" do
    @gat_input_dim 8
    @gat_num_nodes 4
    @gat_num_classes 3

    test "produces correct output shape" do
      model =
        GAT.build(
          input_dim: @gat_input_dim,
          hidden_size: 4,
          num_heads: 2,
          num_classes: @gat_num_classes,
          num_layers: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch_size, @gat_num_nodes, @gat_input_dim}, :f32),
        "adjacency" => Nx.template({@batch_size, @gat_num_nodes, @gat_num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj = Nx.broadcast(1.0, {@batch_size, @gat_num_nodes, @gat_num_nodes})

      input = %{
        "nodes" => Nx.broadcast(0.5, {@batch_size, @gat_num_nodes, @gat_input_dim}),
        "adjacency" => adj
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @gat_num_nodes, @gat_num_classes}
    end

    test "output contains finite values" do
      model =
        GAT.build(
          input_dim: @gat_input_dim,
          hidden_size: 4,
          num_heads: 2,
          num_classes: @gat_num_classes,
          num_layers: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch_size, @gat_num_nodes, @gat_input_dim}, :f32),
        "adjacency" => Nx.template({@batch_size, @gat_num_nodes, @gat_num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj = Nx.broadcast(1.0, {@batch_size, @gat_num_nodes, @gat_num_nodes})

      input = %{
        "nodes" => Nx.broadcast(0.5, {@batch_size, @gat_num_nodes, @gat_input_dim}),
        "adjacency" => adj
      }

      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # MessagePassing Tests
  # ============================================================================

  describe "MessagePassing.message_passing_layer/4" do
    @mp_feature_dim 8
    @mp_output_dim 16
    @mp_num_nodes 4

    test "produces correct output shape" do
      nodes_input = Axon.input("nodes", shape: {nil, @mp_num_nodes, @mp_feature_dim})
      adj_input = Axon.input("adjacency", shape: {nil, @mp_num_nodes, @mp_num_nodes})

      model =
        MessagePassing.message_passing_layer(nodes_input, adj_input, @mp_output_dim,
          name: "mp",
          activation: :relu,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch_size, @mp_num_nodes, @mp_feature_dim}, :f32),
        "adjacency" => Nx.template({@batch_size, @mp_num_nodes, @mp_num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "nodes" => Nx.broadcast(0.5, {@batch_size, @mp_num_nodes, @mp_feature_dim}),
        "adjacency" => Nx.broadcast(1.0, {@batch_size, @mp_num_nodes, @mp_num_nodes})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @mp_num_nodes, @mp_output_dim}
    end
  end

  # ============================================================================
  # PointNet Tests
  # ============================================================================

  describe "PointNet.build/1" do
    @pn_input_dim 3
    @pn_num_points 16
    @pn_num_classes 5

    test "produces correct output shape" do
      model =
        PointNet.build(
          input_dim: @pn_input_dim,
          num_classes: @pn_num_classes,
          hidden_dims: [16, 32],
          global_dims: [32, 16],
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @pn_num_points, @pn_input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @pn_num_points, @pn_input_dim})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @pn_num_classes}
    end

    test "output contains finite values" do
      model =
        PointNet.build(
          input_dim: @pn_input_dim,
          num_classes: @pn_num_classes,
          hidden_dims: [16, 32],
          global_dims: [32, 16],
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @pn_num_points, @pn_input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @pn_num_points, @pn_input_dim})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # EBM Tests
  # ============================================================================

  describe "EBM.build/1" do
    @ebm_input_size 8

    test "produces scalar energy per sample" do
      model =
        EBM.build(
          input_size: @ebm_input_size,
          hidden_sizes: [16, 8]
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @ebm_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @ebm_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, 1}
    end

    test "output contains finite values" do
      model =
        EBM.build(
          input_size: @ebm_input_size,
          hidden_sizes: [16, 8]
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @ebm_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @ebm_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # Bayesian Tests
  # ============================================================================

  describe "Bayesian.build/1" do
    @bay_input_size 8
    @bay_output_size 4

    test "produces correct output shape" do
      model =
        Bayesian.build(
          input_size: @bay_input_size,
          hidden_sizes: [16],
          output_size: @bay_output_size
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @bay_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @bay_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @bay_output_size}
    end

    test "output contains finite values" do
      model =
        Bayesian.build(
          input_size: @bay_input_size,
          hidden_sizes: [16],
          output_size: @bay_output_size
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @bay_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @bay_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # MCDropout Tests
  # ============================================================================

  describe "MCDropout.build/1" do
    @mc_input_size 8
    @mc_output_size 4

    test "produces correct output shape" do
      model =
        MCDropout.build(
          input_size: @mc_input_size,
          hidden_sizes: [16],
          output_size: @mc_output_size
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @mc_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @mc_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @mc_output_size}
    end

    test "output contains finite values" do
      model =
        MCDropout.build(
          input_size: @mc_input_size,
          hidden_sizes: [16],
          output_size: @mc_output_size
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @mc_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @mc_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # NTM Tests
  # ============================================================================

  describe "NTM.build/1" do
    @ntm_input_size 8
    @ntm_memory_size 16
    @ntm_memory_dim 4

    test "produces correct output shape" do
      model =
        NTM.build(
          input_size: @ntm_input_size,
          memory_size: @ntm_memory_size,
          memory_dim: @ntm_memory_dim,
          controller_size: 16,
          num_heads: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "input" => Nx.template({@batch_size, @ntm_input_size}, :f32),
        "memory" => Nx.template({@batch_size, @ntm_memory_size, @ntm_memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "input" => Nx.broadcast(0.5, {@batch_size, @ntm_input_size}),
        "memory" => Nx.broadcast(0.1, {@batch_size, @ntm_memory_size, @ntm_memory_dim})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @ntm_input_size}
    end

    test "output contains finite values" do
      model =
        NTM.build(
          input_size: @ntm_input_size,
          memory_size: @ntm_memory_size,
          memory_dim: @ntm_memory_dim,
          controller_size: 16,
          num_heads: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "input" => Nx.template({@batch_size, @ntm_input_size}, :f32),
        "memory" => Nx.template({@batch_size, @ntm_memory_size, @ntm_memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "input" => Nx.broadcast(0.5, {@batch_size, @ntm_input_size}),
        "memory" => Nx.broadcast(0.1, {@batch_size, @ntm_memory_size, @ntm_memory_dim})
      }

      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # Hypernetwork Tests
  # ============================================================================

  describe "Hypernetwork.build/1" do
    @hyper_cond_size 8
    @hyper_input_size 4

    test "produces correct output shape" do
      model =
        Hypernetwork.build(
          conditioning_size: @hyper_cond_size,
          target_layer_sizes: [{@hyper_input_size, 8}, {8, 4}],
          hidden_sizes: [16],
          input_size: @hyper_input_size
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "conditioning" => Nx.template({@batch_size, @hyper_cond_size}, :f32),
        "data_input" => Nx.template({@batch_size, @hyper_input_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "conditioning" => Nx.broadcast(0.5, {@batch_size, @hyper_cond_size}),
        "data_input" => Nx.broadcast(0.3, {@batch_size, @hyper_input_size})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, 4}
    end

    test "output contains finite values" do
      model =
        Hypernetwork.build(
          conditioning_size: @hyper_cond_size,
          target_layer_sizes: [{@hyper_input_size, 8}],
          hidden_sizes: [16],
          input_size: @hyper_input_size
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "conditioning" => Nx.template({@batch_size, @hyper_cond_size}, :f32),
        "data_input" => Nx.template({@batch_size, @hyper_input_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "conditioning" => Nx.broadcast(0.5, {@batch_size, @hyper_cond_size}),
        "data_input" => Nx.broadcast(0.3, {@batch_size, @hyper_input_size})
      }

      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # Capsule Tests
  # ============================================================================

  describe "Capsule.build/1" do
    @caps_height 28
    @caps_width 28
    @caps_channels 1
    @caps_num_digit 5

    test "produces correct output shape" do
      model =
        Capsule.build(
          input_shape: {nil, @caps_height, @caps_width, @caps_channels},
          num_primary_caps: 4,
          primary_cap_dim: 4,
          num_digit_caps: @caps_num_digit,
          digit_cap_dim: 8,
          routing_iterations: 2,
          conv_channels: 16,
          conv_kernel: 3
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch_size, @caps_height, @caps_width, @caps_channels}, :f32)
          },
          Axon.ModelState.empty()
        )

      input = %{
        "input" => Nx.broadcast(0.5, {@batch_size, @caps_height, @caps_width, @caps_channels})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @caps_num_digit}
    end
  end

  # ============================================================================
  # SNN Tests
  # ============================================================================

  describe "SNN.build/1" do
    @snn_input_size 8
    @snn_output_size 4

    test "produces correct output shape" do
      model =
        SNN.build(
          input_size: @snn_input_size,
          hidden_sizes: [16],
          output_size: @snn_output_size,
          num_timesteps: 5
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @snn_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @snn_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @snn_output_size}
    end

    test "output contains finite values" do
      model =
        SNN.build(
          input_size: @snn_input_size,
          hidden_sizes: [16],
          output_size: @snn_output_size,
          num_timesteps: 5
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @snn_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @snn_input_size})}
      output = predict_fn.(params, input)

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end

    test "different hidden sizes produce same output shape" do
      model =
        SNN.build(
          input_size: @snn_input_size,
          hidden_sizes: [32, 16],
          output_size: @snn_output_size,
          num_timesteps: 5
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @snn_input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"input" => Nx.broadcast(0.5, {@batch_size, @snn_input_size})}
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch_size, @snn_output_size}
    end
  end
end
