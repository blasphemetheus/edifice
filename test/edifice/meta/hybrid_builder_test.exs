defmodule Edifice.Meta.HybridBuilderTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.HybridBuilder

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  defp base_opts do
    [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_heads: 4,
      head_dim: 8,
      num_layers: 4,
      state_size: 8,
      expand_factor: 2,
      conv_size: 3,
      dropout: 0.0,
      window_size: @seq_len,
      seq_len: @seq_len
    ]
  end

  defp run_model(model) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    params =
      init_fn.(
        Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        Axon.ModelState.empty()
      )

    predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim}))
  end

  describe "build/1 with attention_every (interleaved)" do
    test "produces correct output shape" do
      model = HybridBuilder.build(base_opts() ++ [attention_every: 2])
      output = run_model(model)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "produces finite values" do
      model = HybridBuilder.build(base_opts() ++ [attention_every: 2])
      output = run_model(model)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with ratio" do
    test "produces correct output shape with 3:1 ratio" do
      model = HybridBuilder.build(base_opts() ++ [ratio: {3, 1}])
      output = run_model(model)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "produces correct output shape with 9:1 ratio" do
      opts = base_opts() |> Keyword.put(:num_layers, 10)
      model = HybridBuilder.build(opts ++ [ratio: {9, 1}])
      output = run_model(model)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 with explicit schedule" do
    test "uses custom schedule" do
      schedule = [:mamba, :mamba, :attn, :mamba]
      model = HybridBuilder.build(base_opts() ++ [schedule: schedule])
      output = run_model(model)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "supports multi-backbone schedule" do
      schedule = [:mamba, :gru, :attn, :mamba]
      model = HybridBuilder.build(base_opts() ++ [schedule: schedule])
      output = run_model(model)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 with parallel mode" do
    test "produces correct output shape" do
      model = HybridBuilder.build(base_opts() ++ [mode: :parallel])
      output = run_model(model)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "produces finite values" do
      model = HybridBuilder.build(base_opts() ++ [mode: :parallel])
      output = run_model(model)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "resolve_schedule/1" do
    test "uses explicit schedule as-is" do
      schedule = [:mamba, :mamba, :attn]
      assert HybridBuilder.resolve_schedule(schedule: schedule) == schedule
    end

    test "resolves ratio to correct pattern" do
      schedule = HybridBuilder.resolve_schedule(num_layers: 10, ratio: {9, 1}, backbone: :mamba)
      assert length(schedule) == 10
      assert Enum.count(schedule, &(&1 == :attn)) == 1
      assert Enum.count(schedule, &(&1 == :mamba)) == 9
    end

    test "resolves attention_every to correct pattern" do
      schedule = HybridBuilder.resolve_schedule(num_layers: 6, attention_every: 3, backbone: :gru)
      assert schedule == [:gru, :gru, :attn, :gru, :gru, :attn]
    end

    test "default schedule has attention" do
      schedule = HybridBuilder.resolve_schedule(num_layers: 8)
      assert Enum.any?(schedule, &(&1 == :attn))
      assert Enum.any?(schedule, &(&1 == :mamba))
    end
  end

  describe "describe_schedule/1" do
    test "describes interleaved schedule" do
      desc = HybridBuilder.describe_schedule(num_layers: 10, ratio: {9, 1})
      assert desc.mode == :interleaved
      assert desc.num_attention == 1
      assert desc.num_backbone == 9
      assert desc.backbone_pct == 90.0
    end

    test "describes parallel schedule" do
      desc = HybridBuilder.describe_schedule(num_layers: 6, mode: :parallel)
      assert desc.mode == :parallel
      assert desc.num_backbone == 6
      assert desc.num_attention == 6
    end
  end

  describe "preset/1" do
    test "nemotron_h preset is valid" do
      preset = HybridBuilder.preset(:nemotron_h)
      assert Keyword.get(preset, :ratio) == {9, 1}
      assert Keyword.get(preset, :mode) == :interleaved
    end

    test "parallel preset is valid" do
      preset = HybridBuilder.preset(:parallel)
      assert Keyword.get(preset, :mode) == :parallel
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert HybridBuilder.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert HybridBuilder.output_size([]) == 256
    end
  end
end
