defmodule Edifice.Vision.NeRFTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.NeRF

  @batch 2

  @base_opts [
    hidden_size: 64,
    num_layers: 4,
    skip_layer: 2,
    num_frequencies: 4,
    use_viewdir: true
  ]

  describe "build/1 with viewdir" do
    test "returns an Axon model" do
      model = NeRF.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces [batch, 4] output shape" do
      model = NeRF.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      coords = Nx.broadcast(0.5, {@batch, 3})
      dirs = Nx.broadcast(0.1, {@batch, 3})

      params =
        init_fn.(
          %{
            "coordinates" => Nx.template({@batch, 3}, :f32),
            "directions" => Nx.template({@batch, 3}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"coordinates" => coords, "directions" => dirs})

      assert Nx.shape(output) == {@batch, 4}
    end

    test "output contains finite values" do
      model = NeRF.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      coords = Nx.broadcast(0.5, {@batch, 3})
      dirs = Nx.broadcast(0.1, {@batch, 3})

      params =
        init_fn.(
          %{
            "coordinates" => Nx.template({@batch, 3}, :f32),
            "directions" => Nx.template({@batch, 3}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"coordinates" => coords, "directions" => dirs})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "RGB values are in [0, 1] due to sigmoid" do
      model = NeRF.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      coords = Nx.broadcast(0.5, {@batch, 3})
      dirs = Nx.broadcast(0.1, {@batch, 3})

      params =
        init_fn.(
          %{
            "coordinates" => Nx.template({@batch, 3}, :f32),
            "directions" => Nx.template({@batch, 3}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"coordinates" => coords, "directions" => dirs})

      # First 3 channels are RGB (sigmoid bounded)
      rgb = Nx.slice_along_axis(output, 0, 3, axis: 1)
      assert Nx.all(Nx.greater_equal(rgb, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(rgb, 1.0)) |> Nx.to_number() == 1
    end
  end

  describe "build/1 without viewdir" do
    test "produces [batch, 4] output shape" do
      opts = Keyword.put(@base_opts, :use_viewdir, false)
      model = NeRF.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      coords = Nx.broadcast(0.5, {@batch, 3})

      params =
        init_fn.(
          %{"coordinates" => Nx.template({@batch, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"coordinates" => coords})
      assert Nx.shape(output) == {@batch, 4}
    end

    test "output contains finite values" do
      opts = Keyword.put(@base_opts, :use_viewdir, false)
      model = NeRF.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      coords = Nx.broadcast(0.5, {@batch, 3})

      params =
        init_fn.(
          %{"coordinates" => Nx.template({@batch, 3}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"coordinates" => coords})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with custom dimensions" do
    test "works with custom coord_dim" do
      opts = @base_opts |> Keyword.put(:coord_dim, 5) |> Keyword.put(:use_viewdir, false)
      model = NeRF.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      coords = Nx.broadcast(0.5, {@batch, 5})

      params =
        init_fn.(
          %{"coordinates" => Nx.template({@batch, 5}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"coordinates" => coords})
      assert Nx.shape(output) == {@batch, 4}
    end
  end

  describe "output_size/1" do
    test "always returns 4" do
      assert NeRF.output_size() == 4
      assert NeRF.output_size(@base_opts) == 4
    end
  end
end
