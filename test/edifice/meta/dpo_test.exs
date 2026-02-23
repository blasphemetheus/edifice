defmodule Edifice.Meta.DPOTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.DPO

  describe "build/1" do
    test "builds DPO policy model with default options" do
      model = DPO.build(
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

      model = DPO.build(
        hidden_size: hidden_size,
        num_layers: 1,
        num_heads: 2,
        vocab_size: vocab_size,
        dropout: 0.0
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        Nx.template({batch_size, seq_len}, :s64),
        %{}
      )

      input = Nx.broadcast(1, {batch_size, seq_len}) |> Nx.as_type(:s64)
      output = predict_fn.(params, %{"tokens" => input})

      assert {^batch_size, ^seq_len, ^vocab_size} = Nx.shape(output)
    end
  end

  describe "loss/5" do
    test "computes DPO loss with default beta" do
      # Simulated log probabilities
      policy_chosen = Nx.tensor([-1.0, -1.5, -2.0])
      policy_rejected = Nx.tensor([-2.0, -2.5, -3.0])
      ref_chosen = Nx.tensor([-1.2, -1.7, -2.2])
      ref_rejected = Nx.tensor([-2.2, -2.7, -3.2])

      loss = DPO.loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

      assert is_struct(loss, Nx.Tensor)
      assert Nx.shape(loss) == {}
      # Loss should be non-negative
      assert Nx.to_number(Nx.greater_equal(loss, 0.0)) == 1
    end

    test "computes loss with custom beta" do
      # Use different log probs so the log ratios don't cancel
      policy_chosen = Nx.tensor([-1.0, -1.5])
      policy_rejected = Nx.tensor([-2.0, -2.5])
      ref_chosen = Nx.tensor([-1.5, -2.0])
      ref_rejected = Nx.tensor([-1.5, -2.0])

      loss_low_beta = DPO.loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: 0.01)
      loss_high_beta = DPO.loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: 1.0)

      # Different betas should give different losses
      # (unless log ratios are equal, which they're not here)
      diff = Nx.abs(Nx.subtract(loss_low_beta, loss_high_beta))
      assert Nx.to_number(Nx.greater(diff, 0.001)) == 1
    end

    test "loss is lower when policy prefers chosen over rejected more than reference" do
      # Policy strongly prefers chosen
      policy_chosen = Nx.tensor([-0.5])
      policy_rejected = Nx.tensor([-3.0])

      # Reference is neutral
      ref_chosen = Nx.tensor([-1.5])
      ref_rejected = Nx.tensor([-1.5])

      loss = DPO.loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: 0.1)

      # With policy preferring chosen more than ref, loss should be relatively low
      # The loss is -log(sigmoid(beta * diff)), which for positive diff is small
      assert Nx.to_number(loss) < 1.0
    end
  end

  describe "compute_logprobs/3" do
    test "computes per-sequence log probabilities" do
      batch_size = 2
      seq_len = 4
      vocab_size = 10

      # Random logits
      logits = Nx.broadcast(0.1, {batch_size, seq_len, vocab_size})
      targets = Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

      log_probs = DPO.compute_logprobs(logits, targets)

      assert {^batch_size} = Nx.shape(log_probs)
      # Log probs should be negative
      assert Nx.to_number(Nx.all(Nx.less(log_probs, 0.0))) == 1
    end

    test "respects mask for padding" do
      logits = Nx.broadcast(0.1, {2, 4, 10})
      targets = Nx.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
      mask = Nx.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]) |> Nx.as_type(:f32)

      log_probs_masked = DPO.compute_logprobs(logits, targets, mask)
      log_probs_unmasked = DPO.compute_logprobs(logits, targets)

      # Masked should have different values for the second sequence
      # The masked version zeros out the last two positions, giving higher (less negative) log prob
      masked_val = log_probs_masked |> Nx.slice([1], [1]) |> Nx.squeeze() |> Nx.to_number()
      unmasked_val = log_probs_unmasked |> Nx.slice([1], [1]) |> Nx.squeeze() |> Nx.to_number()

      assert masked_val > unmasked_val
    end
  end

  describe "default_beta/0" do
    test "returns default beta value" do
      assert DPO.default_beta() == 0.1
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = DPO.recommended_defaults()

      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :vocab_size)
      assert Keyword.has_key?(defaults, :beta)
    end
  end
end
