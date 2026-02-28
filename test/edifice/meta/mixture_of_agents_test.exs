defmodule Edifice.Meta.MixtureOfAgentsTest do
  use ExUnit.Case, async: true
  @moduletag :meta
  @moduletag timeout: 120_000

  alias Edifice.Meta.MixtureOfAgents

  @batch_size 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  describe "build/1" do
    test "produces correct output shape" do
      model =
        MixtureOfAgents.build(
          embed_dim: @embed_dim,
          num_proposers: 2,
          proposer_hidden_size: @hidden_size,
          aggregator_hidden_size: @hidden_size,
          proposer_layers: 1,
          aggregator_layers: 1,
          num_heads: 4,
          dropout: 0.0,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output is finite" do
      model =
        MixtureOfAgents.build(
          embed_dim: @embed_dim,
          num_proposers: 2,
          proposer_hidden_size: @hidden_size,
          aggregator_hidden_size: @hidden_size,
          proposer_layers: 1,
          aggregator_layers: 1,
          num_heads: 4,
          dropout: 0.0,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with different num_proposers" do
      model =
        MixtureOfAgents.build(
          embed_dim: @embed_dim,
          num_proposers: 3,
          proposer_hidden_size: @hidden_size,
          aggregator_hidden_size: @hidden_size,
          proposer_layers: 1,
          aggregator_layers: 1,
          num_heads: 4,
          dropout: 0.0,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with embed_dim different from proposer_hidden_size" do
      model =
        MixtureOfAgents.build(
          embed_dim: 24,
          num_proposers: 2,
          proposer_hidden_size: @hidden_size,
          aggregator_hidden_size: @hidden_size,
          proposer_layers: 1,
          aggregator_layers: 1,
          num_heads: 4,
          dropout: 0.0,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, 24}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 24})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns aggregator_hidden_size" do
      assert MixtureOfAgents.output_size(aggregator_hidden_size: 128) == 128
      assert MixtureOfAgents.output_size() == 256
    end
  end
end
