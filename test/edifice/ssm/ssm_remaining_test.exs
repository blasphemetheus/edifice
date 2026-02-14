defmodule Edifice.SSM.SSMRemainingTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.Hybrid
  alias Edifice.SSM.HybridBuilder
  alias Edifice.SSM.MambaCumsum
  alias Edifice.SSM.MambaHillisSteele
  alias Edifice.SSM.Zamba

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @state_size 8
  @num_layers 2

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  # ============================================================================
  # MambaCumsum Tests
  # ============================================================================

  describe "MambaCumsum.build/1" do
    @cumsum_opts [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      state_size: @state_size,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = MambaCumsum.build(@cumsum_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MambaCumsum.build(@cumsum_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = MambaCumsum.build(@cumsum_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "MambaCumsum.output_size/1" do
    test "returns hidden_size" do
      assert MambaCumsum.output_size(hidden_size: 64) == 64
    end
  end

  # ============================================================================
  # MambaHillisSteele Tests
  # ============================================================================

  describe "MambaHillisSteele.build/1" do
    @hs_opts [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      state_size: @state_size,
      num_layers: @num_layers,
      window_size: @seq_len
    ]

    test "builds an Axon model" do
      model = MambaHillisSteele.build(@hs_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MambaHillisSteele.build(@hs_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = MambaHillisSteele.build(@hs_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "MambaHillisSteele.output_size/1" do
    test "returns hidden_size" do
      assert MambaHillisSteele.output_size(hidden_size: 64) == 64
    end
  end

  # ============================================================================
  # Hybrid (Jamba) Tests
  # ============================================================================

  describe "Hybrid.build/1" do
    @hybrid_opts [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_layers: 3,
      attention_every: 3,
      state_size: @state_size,
      num_heads: 2,
      head_dim: 16,
      window_size: @seq_len,
      dropout: 0.0
    ]

    test "builds an Axon model" do
      model = Hybrid.build(@hybrid_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Hybrid.build(@hybrid_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Hybrid.build(@hybrid_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "Hybrid.output_size/1" do
    test "returns hidden_size" do
      assert Hybrid.output_size(hidden_size: 64) == 64
    end
  end

  # ============================================================================
  # Zamba Tests
  # ============================================================================

  describe "Zamba.build/1" do
    @zamba_opts [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_layers: 3,
      attention_every: 3,
      state_size: @state_size,
      num_heads: 2,
      head_dim: 16,
      window_size: @seq_len,
      dropout: 0.0
    ]

    test "builds an Axon model" do
      model = Zamba.build(@zamba_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Zamba.build(@zamba_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Zamba.build(@zamba_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert output
             |> Nx.is_nan()
             |> Nx.any()
             |> Nx.to_number() == 0
    end
  end

  describe "Zamba.output_size/1" do
    test "returns hidden_size" do
      assert Zamba.output_size(hidden_size: 64) == 64
    end
  end

  # ============================================================================
  # HybridBuilder Tests
  # ============================================================================

  describe "HybridBuilder.build/2" do
    @builder_opts [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      dropout: 0.0,
      seq_len: @seq_len
    ]

    test "builds a jamba_like model" do
      pattern = HybridBuilder.pattern(:jamba_like, 3)
      model = HybridBuilder.build(pattern, @builder_opts)
      assert %Axon{} = model
    end

    test "builds a zamba_like model" do
      pattern = HybridBuilder.pattern(:zamba_like, 3)
      model = HybridBuilder.build(pattern, @builder_opts)
      assert %Axon{} = model
    end

    @tag :slow
    test "forward pass produces correct output shape" do
      pattern = HybridBuilder.pattern(:jamba_like, 3)
      model = HybridBuilder.build(pattern, @builder_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "HybridBuilder.pattern/2" do
    test "returns list of layer types" do
      pattern = HybridBuilder.pattern(:jamba_like, 6)
      assert is_list(pattern)
      assert length(pattern) == 6
    end

    test "jamba_like includes both mamba and attention" do
      pattern = HybridBuilder.pattern(:jamba_like, 6)
      assert :mamba in pattern
      assert :attention in pattern
    end
  end
end
