defmodule Edifice.RecurrentCoverageTest do
  use ExUnit.Case, async: true

  @moduletag timeout: 180_000

  alias Edifice.Recurrent

  @batch 2
  @embed 16
  @hidden 16
  @seq_len 4

  describe "build/1" do
    test "builds LSTM model" do
      model =
        Recurrent.build(
          embed_dim: @embed,
          hidden_size: @hidden,
          cell_type: :lstm,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "builds GRU model" do
      model =
        Recurrent.build(
          embed_dim: @embed,
          hidden_size: @hidden,
          cell_type: :gru,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "supports return_sequences" do
      model =
        Recurrent.build(
          embed_dim: @embed,
          hidden_size: @hidden,
          seq_len: @seq_len,
          return_sequences: true
        )

      assert %Axon{} = model
    end

    test "supports truncated BPTT" do
      model =
        Recurrent.build(
          embed_dim: @embed,
          hidden_size: @hidden,
          seq_len: @seq_len,
          truncate_bptt: 2
        )

      assert %Axon{} = model
    end

    test "supports multi-layer with dropout" do
      model =
        Recurrent.build(
          embed_dim: @embed,
          hidden_size: @hidden,
          num_layers: 2,
          dropout: 0.2,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end
  end

  describe "build_backbone/2" do
    test "builds backbone without layer norm" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      model =
        Recurrent.build_backbone(input,
          hidden_size: @hidden,
          input_layer_norm: false,
          use_layer_norm: false
        )

      assert %Axon{} = model
    end
  end

  describe "build_recurrent_layer/4" do
    test "builds LSTM layer returning sequences" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})
      layer = Recurrent.build_recurrent_layer(input, @hidden, :lstm, return_sequences: true)
      assert %Axon{} = layer
    end

    test "builds GRU layer returning last timestep" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @embed})

      layer =
        Recurrent.build_recurrent_layer(input, @hidden, :gru,
          return_sequences: false,
          use_layer_norm: false
        )

      assert %Axon{} = layer
    end
  end

  describe "build_stateful/1" do
    for cell_type <- [:lstm, :gru] do
      test "builds stateful #{cell_type} model" do
        model =
          Recurrent.build_stateful(
            embed_dim: @embed,
            hidden_size: @hidden,
            cell_type: unquote(cell_type)
          )

        assert %Axon{} = model
      end
    end
  end

  describe "build_hybrid/1" do
    test "builds hybrid RNN + MLP model" do
      model =
        Recurrent.build_hybrid(
          embed_dim: @embed,
          recurrent_size: @hidden,
          mlp_sizes: [16],
          num_recurrent_layers: 1,
          dropout: 0.1
        )

      assert %Axon{} = model
    end
  end

  describe "initial_hidden/2" do
    test "LSTM returns tuple of zero tensors" do
      {h, c} = Recurrent.initial_hidden(2, hidden_size: 8, cell_type: :lstm)
      assert Nx.shape(h) == {2, 8}
      assert Nx.shape(c) == {2, 8}
      assert Nx.to_number(Nx.sum(h)) == 0.0
    end

    test "GRU returns single zero tensor" do
      h = Recurrent.initial_hidden(3, hidden_size: 8, cell_type: :gru)
      assert Nx.shape(h) == {3, 8}
    end
  end

  describe "frames_to_sequence/1" do
    test "stacks 1D frames" do
      frames = [
        Nx.tensor([1.0, 2.0]),
        Nx.tensor([3.0, 4.0]),
        Nx.tensor([5.0, 6.0])
      ]

      seq = Recurrent.frames_to_sequence(frames)
      assert Nx.shape(seq) == {1, 3, 2}
    end

    test "stacks 2D frames (batched)" do
      frames = [
        Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      ]

      seq = Recurrent.frames_to_sequence(frames)
      assert Nx.shape(seq) == {2, 2, 2}
    end
  end

  describe "pad_sequence/3" do
    test "pads short sequence" do
      seq = Nx.broadcast(1.0, {@batch, 3, @embed})
      padded = Recurrent.pad_sequence(seq, 5)
      assert Nx.shape(padded) == {@batch, 5, @embed}
    end

    test "truncates long sequence" do
      seq = Nx.broadcast(1.0, {@batch, 10, @embed})
      truncated = Recurrent.pad_sequence(seq, 4)
      assert Nx.shape(truncated) == {@batch, 4, @embed}
    end

    test "returns unchanged when matching length" do
      seq = Nx.broadcast(1.0, {@batch, 5, @embed})
      result = Recurrent.pad_sequence(seq, 5)
      assert Nx.shape(result) == {@batch, 5, @embed}
    end
  end

  describe "utility functions" do
    test "output_size returns hidden_size" do
      assert Recurrent.output_size(hidden_size: 128) == 128
      assert Recurrent.output_size() == 256
    end

    test "cell_types returns supported types" do
      assert Recurrent.cell_types() == [:lstm, :gru]
    end
  end

  describe "apply_gradient_truncation/2" do
    test "builds truncation layer" do
      input = Axon.input("state_sequence", shape: {nil, 10, @embed})
      layer = Recurrent.apply_gradient_truncation(input, 5)
      assert %Axon{} = layer
    end
  end
end
