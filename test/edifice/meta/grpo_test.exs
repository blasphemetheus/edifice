defmodule Edifice.Meta.GRPOTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.GRPO

  describe "build/1" do
    test "builds GRPO policy model with default options" do
      model =
        GRPO.build(
          hidden_size: 64,
          num_layers: 2,
          num_heads: 2,
          vocab_size: 100
        )

      assert %Axon{} = model
    end

    test "model produces output with correct shape" do
      vocab_size = 100
      hidden_size = 32
      seq_len = 8
      batch_size = 2

      model =
        GRPO.build(
          hidden_size: hidden_size,
          num_layers: 1,
          num_heads: 2,
          vocab_size: vocab_size,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({batch_size, seq_len}, :s64),
          %{}
        )

      input = Nx.broadcast(1, {batch_size, seq_len}) |> Nx.as_type(:s64)
      output = predict_fn.(params, %{"tokens" => input})

      assert {^batch_size, ^seq_len, ^vocab_size} = Nx.shape(output)
    end
  end

  describe "compute_advantages/2" do
    test "computes group-relative advantages from 2D rewards" do
      # 2 prompts, 4 responses each
      rewards =
        Nx.tensor([
          [1.0, 2.0, 3.0, 4.0],
          [10.0, 20.0, 30.0, 40.0]
        ])

      advantages = GRPO.compute_advantages(rewards)

      assert {2, 4} = Nx.shape(advantages)

      # Check normalization: mean should be ~0, std ~1 within each group
      mean_per_group = Nx.mean(advantages, axes: [1])
      std_per_group = Nx.standard_deviation(advantages, axes: [1])

      # Mean should be close to 0
      assert Nx.to_number(Nx.all(Nx.less(Nx.abs(mean_per_group), 1.0e-5))) == 1
      # Std should be close to 1
      assert Nx.to_number(Nx.all(Nx.less(Nx.abs(Nx.subtract(std_per_group, 1.0)), 0.1))) == 1
    end

    test "computes advantages from 1D rewards with group_size" do
      # 8 total rewards, group size 4 -> 2 groups
      rewards = Nx.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])

      advantages = GRPO.compute_advantages(rewards, group_size: 4)

      assert {8} = Nx.shape(advantages)
    end

    test "handles single-element groups gracefully" do
      rewards = Nx.tensor([[5.0], [10.0]])

      advantages = GRPO.compute_advantages(rewards)

      # Single-element groups: std is ~0, so advantage is ~0
      assert {2, 1} = Nx.shape(advantages)
    end
  end

  describe "loss/3" do
    test "computes simple policy gradient loss" do
      log_probs = Nx.tensor([-1.0, -2.0, -1.5, -2.5])
      advantages = Nx.tensor([1.0, -1.0, 0.5, -0.5])

      loss = GRPO.loss(log_probs, advantages)

      assert is_struct(loss, Nx.Tensor)
      assert Nx.shape(loss) == {}
    end

    test "computes PPO-clipped loss when clip_range provided" do
      log_probs = Nx.tensor([-1.0, -2.0])
      old_log_probs = Nx.tensor([-1.1, -1.9])
      advantages = Nx.tensor([1.0, -1.0])

      loss =
        GRPO.loss(log_probs, advantages,
          clip_range: 0.2,
          old_log_probs: old_log_probs
        )

      assert is_struct(loss, Nx.Tensor)
      assert Nx.shape(loss) == {}
    end

    test "loss responds to advantage signs" do
      log_probs = Nx.tensor([-1.0, -1.0])

      # Positive advantage should encourage the action
      loss_pos = GRPO.loss(log_probs, Nx.tensor([1.0, 1.0]))

      # Negative advantage should discourage the action
      loss_neg = GRPO.loss(log_probs, Nx.tensor([-1.0, -1.0]))

      # The losses should have opposite signs (before negation in the function)
      # After negation, positive advantage leads to more negative loss (encouragement)
      assert Nx.to_number(loss_pos) != Nx.to_number(loss_neg)
    end
  end

  describe "compute_logprobs/3" do
    test "computes per-sequence log probabilities" do
      batch_size = 2
      seq_len = 4
      vocab_size = 10

      logits = Nx.broadcast(0.1, {batch_size, seq_len, vocab_size})
      targets = Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

      log_probs = GRPO.compute_logprobs(logits, targets)

      assert {^batch_size} = Nx.shape(log_probs)
      # Log probs should be negative
      assert Nx.to_number(Nx.all(Nx.less(log_probs, 0.0))) == 1
    end
  end

  describe "default_group_size/0" do
    test "returns default group size" do
      assert GRPO.default_group_size() == 8
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = GRPO.recommended_defaults()

      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :group_size)
    end
  end
end
