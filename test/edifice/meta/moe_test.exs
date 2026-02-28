defmodule Edifice.Meta.MoETest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.MoE

  @input_size 32
  @batch_size 2
  @seq_len 4

  describe "build/1" do
    test "produces correct output shape with top_k routing" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 64,
          output_size: @input_size,
          num_experts: 4,
          top_k: 2,
          routing: :top_k,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "produces correct output shape with switch routing" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 64,
          output_size: @input_size,
          num_experts: 4,
          routing: :switch,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "produces correct output shape with soft routing" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 64,
          output_size: @input_size,
          num_experts: 4,
          routing: :soft,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "produces correct output shape with hash routing" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: 64,
          output_size: @input_size,
          num_experts: 4,
          routing: :hash,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end

    test "output_size defaults to input_size" do
      model =
        MoE.build(
          input_size: @input_size,
          num_experts: 4,
          top_k: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @input_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size}))
      assert Nx.shape(output) == {@batch_size, @seq_len, @input_size}
    end
  end

  describe "compute_aux_loss/3" do
    test "returns scalar loss" do
      num_experts = 4
      router_probs = Nx.broadcast(0.5, {@batch_size, @seq_len, num_experts})

      expert_mask =
        Nx.tensor([
          [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]],
          [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
        ])

      loss = MoE.compute_aux_loss(router_probs, expert_mask)
      assert Nx.shape(loss) == {}
    end

    test "respects load_balance_weight" do
      num_experts = 4
      router_probs = Nx.broadcast(0.5, {@batch_size, @seq_len, num_experts})
      expert_mask = Nx.broadcast(1.0, {@batch_size, @seq_len, num_experts})

      loss_small = MoE.compute_aux_loss(router_probs, expert_mask, load_balance_weight: 0.001)
      loss_large = MoE.compute_aux_loss(router_probs, expert_mask, load_balance_weight: 1.0)

      assert Nx.to_number(loss_large) > Nx.to_number(loss_small)
    end
  end

  describe "estimate_speedup/3" do
    test "returns expected speedup ratio" do
      # 8 experts, top-2 -> 4x speedup on expert layers
      # With 50% expert fraction: 1 / (0.5 + 0.5/4) = 1/0.625 = 1.6x
      speedup = MoE.estimate_speedup(8, 2, 0.5)
      assert_in_delta speedup, 1.6, 0.01
    end

    test "higher expert fraction yields more speedup" do
      speedup_low = MoE.estimate_speedup(8, 2, 0.25)
      speedup_high = MoE.estimate_speedup(8, 2, 0.75)
      assert speedup_high > speedup_low
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MoE.recommended_defaults()
      assert Keyword.has_key?(defaults, :num_experts)
      assert Keyword.has_key?(defaults, :top_k)
      assert Keyword.has_key?(defaults, :routing)
    end
  end
end
