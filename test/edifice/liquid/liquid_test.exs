defmodule Edifice.LiquidTest do
  use ExUnit.Case, async: true

  alias Edifice.Liquid

  @embed_dim 16
  @hidden_size 32
  @seq_len 4
  @batch_size 2

  describe "build/1" do
    test "produces correct output shape" do
      model =
        Liquid.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0,
          integration_steps: 1,
          solver: :euler
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @embed_dim}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      # Output is last timestep: [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "works when embed_dim equals hidden_size (no projection)" do
      model =
        Liquid.build(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1,
          window_size: @seq_len,
          seq_len: @seq_len,
          dropout: 0.0,
          integration_steps: 1,
          solver: :euler
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @hidden_size}, :f32),
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size}))

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Liquid.output_size(hidden_size: 128) == 128
    end

    test "uses default of 256" do
      assert Liquid.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns a positive integer" do
      count =
        Liquid.param_count(
          embed_dim: 64,
          hidden_size: 128,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "scales with num_layers" do
      count_2 = Liquid.param_count(embed_dim: 64, hidden_size: 128, num_layers: 2)
      count_4 = Liquid.param_count(embed_dim: 64, hidden_size: 128, num_layers: 4)

      assert count_4 > count_2
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = Liquid.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :solver)
    end
  end

  describe "high_accuracy_defaults/0" do
    test "uses dopri5 solver" do
      defaults = Liquid.high_accuracy_defaults()
      assert Keyword.get(defaults, :solver) == :dopri5
    end
  end

  describe "init_cache/1" do
    test "creates cache with correct structure" do
      cache = Liquid.init_cache(batch_size: 2, hidden_size: 64, num_layers: 3)

      assert %{layers: layers, step: 0, config: config} = cache
      assert map_size(layers) == 3
      assert config.hidden_size == 64
      assert config.num_layers == 3
    end

    test "hidden states are initialized to zeros" do
      cache = Liquid.init_cache(batch_size: 2, hidden_size: 32, num_layers: 1)
      layer = cache.layers["layer_1"]

      assert Nx.shape(layer.h) == {2, 32}
      assert Nx.to_number(Nx.sum(layer.h)) == 0.0
    end
  end
end
