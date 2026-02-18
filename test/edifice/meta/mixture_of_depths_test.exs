defmodule Edifice.Meta.MixtureOfDepthsTest do
  use ExUnit.Case, async: true
  @moduletag timeout: 120_000

  alias Edifice.Meta.MixtureOfDepths

  @batch_size 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  describe "build/1" do
    test "produces correct output shape" do
      model =
        MixtureOfDepths.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 2,
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
        MixtureOfDepths.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 1,
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

    test "with different capacity_ratio" do
      model =
        MixtureOfDepths.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 1,
          capacity_ratio: 0.25,
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

    test "with embed_dim different from hidden_size" do
      model =
        MixtureOfDepths.build(
          embed_dim: 24,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 1,
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
    test "returns hidden_size" do
      assert MixtureOfDepths.output_size(hidden_size: 128) == 128
      assert MixtureOfDepths.output_size() == 256
    end
  end
end
