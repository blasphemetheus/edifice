defmodule Edifice.Attention.NystromformerCoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Attention.Nystromformer.
  Targets uncovered code paths: param_count, recommended_defaults,
  seq_len <= num_landmarks branch (short sequences), different
  landmark counts, different head/layer configurations.
  """
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.Nystromformer

  @batch 2
  @seq_len 8
  @embed_dim 16
  @hidden_size 16
  @num_heads 4

  # ============================================================================
  # Short sequence (seq_len <= num_landmarks)
  # ============================================================================

  describe "short sequence (seq_len <= num_landmarks)" do
    test "uses all positions as landmarks when seq_len <= num_landmarks" do
      # seq_len=4, num_landmarks=8 -> should use all positions as landmarks
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 8,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: 4
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, 4, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, 4, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "seq_len exactly equals num_landmarks" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: @seq_len,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # Different landmark counts
  # ============================================================================

  describe "different landmark counts" do
    test "with 2 landmarks" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 2,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "with many landmarks" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 16,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: 16
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, 16, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, 16, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # Different layer counts
  # ============================================================================

  describe "different layer counts" do
    test "single layer" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 4,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "three layers" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 4,
          num_layers: 3,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # Different head counts
  # ============================================================================

  describe "different head counts" do
    test "with 2 heads" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 4,
          num_layers: 1,
          num_heads: 2,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # Dropout paths
  # ============================================================================

  describe "dropout" do
    test "with dropout > 0" do
      model =
        Nystromformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_landmarks: 4,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.2,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # embed_dim == hidden_size
  # ============================================================================

  describe "embed_dim == hidden_size" do
    test "skips input projection" do
      model =
        Nystromformer.build(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_landmarks: 4,
          num_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # param_count/1
  # ============================================================================

  describe "param_count/1" do
    test "returns a positive integer" do
      count =
        Nystromformer.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "more layers increases param count" do
      count_1 =
        Nystromformer.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1
        )

      count_4 =
        Nystromformer.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 4
        )

      assert count_4 > count_1
    end

    test "embed_dim == hidden_size has no input projection cost" do
      count_same =
        Nystromformer.param_count(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1
        )

      count_diff =
        Nystromformer.param_count(
          embed_dim: @embed_dim + 8,
          hidden_size: @hidden_size,
          num_layers: 1
        )

      assert count_diff > count_same
    end
  end

  # ============================================================================
  # output_size/1
  # ============================================================================

  describe "output_size/1" do
    test "returns default when no opts given" do
      assert Nystromformer.output_size() == 256
    end

    test "returns custom hidden_size" do
      assert Nystromformer.output_size(hidden_size: 128) == 128
    end
  end

  # ============================================================================
  # recommended_defaults/0
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = Nystromformer.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_landmarks)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :dropout)
    end
  end
end
