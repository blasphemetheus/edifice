defmodule Edifice.Recurrent.NewArchitecturesTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.DeltaNet
  alias Edifice.Recurrent.MinGRU
  alias Edifice.Recurrent.MinLSTM
  alias Edifice.Recurrent.Titans
  alias Edifice.Recurrent.TTT

  @batch_size 2
  @seq_len 8
  @embed_size 32
  @hidden_size 32

  # ============================================================================
  # MinGRU
  # ============================================================================

  describe "MinGRU.build/1" do
    test "returns an Axon model" do
      model = MinGRU.build(embed_size: @embed_size, hidden_size: @hidden_size, seq_len: @seq_len)
      assert %Axon{} = model
    end

    test "forward pass produces expected output shape" do
      model =
        MinGRU.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output values are finite" do
      model =
        MinGRU.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "MinGRU.output_size/1" do
    test "returns hidden_size" do
      assert MinGRU.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert MinGRU.output_size() == 256
    end
  end

  # ============================================================================
  # MinLSTM
  # ============================================================================

  describe "MinLSTM.build/1" do
    test "returns an Axon model" do
      model = MinLSTM.build(embed_size: @embed_size, hidden_size: @hidden_size, seq_len: @seq_len)
      assert %Axon{} = model
    end

    test "forward pass produces expected output shape" do
      model =
        MinLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(43)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output values are finite" do
      model =
        MinLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(43)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "MinLSTM.output_size/1" do
    test "returns hidden_size" do
      assert MinLSTM.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert MinLSTM.output_size() == 256
    end
  end

  # ============================================================================
  # DeltaNet
  # ============================================================================

  describe "DeltaNet.build/1" do
    test "returns an Axon model" do
      model =
        DeltaNet.build(embed_size: @embed_size, hidden_size: @hidden_size, seq_len: @seq_len)

      assert %Axon{} = model
    end

    test "forward pass produces expected output shape" do
      model =
        DeltaNet.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(44)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output values are finite" do
      model =
        DeltaNet.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(44)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "DeltaNet.output_size/1" do
    test "returns hidden_size" do
      assert DeltaNet.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert DeltaNet.output_size() == 256
    end
  end

  # ============================================================================
  # TTT
  # ============================================================================

  describe "TTT.build/1" do
    test "returns an Axon model" do
      model = TTT.build(embed_size: @embed_size, hidden_size: @hidden_size, seq_len: @seq_len)
      assert %Axon{} = model
    end

    test "forward pass produces expected output shape" do
      model =
        TTT.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          inner_size: 16,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(45)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output values are finite" do
      model =
        TTT.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          inner_size: 16,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(45)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "TTT.output_size/1" do
    test "returns hidden_size" do
      assert TTT.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert TTT.output_size() == 256
    end
  end

  # ============================================================================
  # Titans
  # ============================================================================

  describe "Titans.build/1" do
    test "returns an Axon model" do
      model = Titans.build(embed_size: @embed_size, hidden_size: @hidden_size, seq_len: @seq_len)
      assert %Axon{} = model
    end

    test "forward pass produces expected output shape" do
      model =
        Titans.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          memory_size: 16,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(46)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output values are finite" do
      model =
        Titans.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          memory_size: 16,
          num_layers: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(46)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "Titans.output_size/1" do
    test "returns hidden_size" do
      assert Titans.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert Titans.output_size() == 256
    end
  end
end
