defmodule Edifice.SSM.SSMNewTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.BiMamba
  alias Edifice.SSM.H3
  alias Edifice.SSM.Hyena
  alias Edifice.SSM.S4
  alias Edifice.SSM.S4D

  @batch 2
  @seq_len 8
  @embed_size 32
  @hidden_size 32
  @state_size 8
  @num_layers 2

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
    input
  end

  # ============================================================================
  # S4 Tests
  # ============================================================================

  describe "S4.build/1" do
    @s4_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      state_size: @state_size,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = S4.build(@s4_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = S4.build(@s4_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = S4.build(@s4_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "S4.output_size/1" do
    test "returns hidden_size" do
      assert S4.output_size(hidden_size: @hidden_size) == @hidden_size
    end
  end

  # ============================================================================
  # S4D Tests
  # ============================================================================

  describe "S4D.build/1" do
    @s4d_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      state_size: @state_size,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = S4D.build(@s4d_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = S4D.build(@s4d_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = S4D.build(@s4d_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "S4D.output_size/1" do
    test "returns hidden_size" do
      assert S4D.output_size(hidden_size: @hidden_size) == @hidden_size
    end
  end

  # ============================================================================
  # H3 Tests
  # ============================================================================

  describe "H3.build/1" do
    @h3_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      state_size: @state_size,
      conv_size: 4,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = H3.build(@h3_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = H3.build(@h3_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = H3.build(@h3_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "H3.output_size/1" do
    test "returns hidden_size" do
      assert H3.output_size(hidden_size: @hidden_size) == @hidden_size
    end
  end

  # ============================================================================
  # Hyena Tests
  # ============================================================================

  describe "Hyena.build/1" do
    @hyena_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      order: 2,
      filter_size: 16,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = Hyena.build(@hyena_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Hyena.build(@hyena_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Hyena.build(@hyena_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "Hyena.output_size/1" do
    test "returns hidden_size" do
      assert Hyena.output_size(hidden_size: @hidden_size) == @hidden_size
    end
  end

  # ============================================================================
  # BiMamba Tests
  # ============================================================================

  describe "BiMamba.build/1" do
    @bimamba_opts [
      embed_size: @embed_size,
      hidden_size: @hidden_size,
      state_size: @state_size,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = BiMamba.build(@bimamba_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = BiMamba.build(@bimamba_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = BiMamba.build(@bimamba_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "concat combine mode produces correct output shape" do
      opts = Keyword.put(@bimamba_opts, :combine, :concat)
      model = BiMamba.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "BiMamba.output_size/1" do
    test "returns hidden_size" do
      assert BiMamba.output_size(hidden_size: @hidden_size) == @hidden_size
    end
  end
end
