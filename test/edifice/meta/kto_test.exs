defmodule Edifice.Meta.KTOTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.KTO

  @batch 4
  @seq_len 8
  @vocab_size 16
  @hidden_size 16
  @num_heads 2
  @num_layers 2

  @base_opts [
    vocab_size: @vocab_size,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    dropout: 0.0
  ]

  defp build_and_init do
    model = KTO.build(@base_opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    params = init_fn.(Nx.template({@batch, @seq_len}, :s64), Axon.ModelState.empty())
    {predict_fn, params}
  end

  describe "KTO.build/1" do
    test "returns an Axon model" do
      assert %Axon{} = KTO.build(@base_opts)
    end

    test "output shape is [batch, seq_len, vocab_size]" do
      {predict_fn, params} = build_and_init()
      tokens = Nx.broadcast(1, {@batch, @seq_len})
      output = predict_fn.(params, %{"tokens" => tokens})
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "output contains finite values" do
      {predict_fn, params} = build_and_init()
      tokens = Nx.broadcast(1, {@batch, @seq_len})
      output = predict_fn.(params, %{"tokens" => tokens})
      assert Nx.all(Nx.logical_not(Nx.is_nan(output))) |> Nx.to_number() == 1
    end
  end

  describe "KTO.kto_loss/4" do
    test "returns a scalar tensor" do
      policy_lp = Nx.tensor([-1.0, -2.0, -1.5, -0.5])
      ref_lp = Nx.tensor([-1.2, -1.8, -1.6, -0.8])
      labels = Nx.tensor([1, 0, 1, 0])
      loss = KTO.kto_loss(policy_lp, ref_lp, labels)
      assert Nx.shape(loss) == {}
    end

    test "loss is finite" do
      policy_lp = Nx.tensor([-1.0, -2.0, -1.5, -0.5])
      ref_lp = Nx.tensor([-1.2, -1.8, -1.6, -0.8])
      labels = Nx.tensor([1, 0, 1, 0])
      loss = KTO.kto_loss(policy_lp, ref_lp, labels)
      assert Nx.to_number(Nx.is_nan(loss)) == 0
    end

    test "all-desirable labels: loss bounded by −λ_D" do
      # When all are desirable, loss = −mean(σ(reward)) in (−1, 0)
      policy_lp = Nx.tensor([-1.0, -0.9, -1.1, -0.8])
      ref_lp = Nx.tensor([-1.0, -1.0, -1.0, -1.0])
      labels = Nx.tensor([1, 1, 1, 1])
      loss = KTO.kto_loss(policy_lp, ref_lp, labels)
      loss_val = Nx.to_number(loss)
      # σ ∈ (0, 1) so loss ∈ (−1, 0)
      assert loss_val > -1.0
      assert loss_val < 0.0
    end

    test "all-undesirable labels: loss bounded by −λ_U" do
      policy_lp = Nx.tensor([-1.0, -0.9, -1.1, -0.8])
      ref_lp = Nx.tensor([-1.0, -1.0, -1.0, -1.0])
      labels = Nx.tensor([0, 0, 0, 0])
      loss = KTO.kto_loss(policy_lp, ref_lp, labels)
      loss_val = Nx.to_number(loss)
      assert loss_val > -1.0
      assert loss_val < 0.0
    end

    test "desirable_weight scales desirable loss" do
      policy_lp = Nx.tensor([-1.0, -0.8])
      ref_lp = Nx.tensor([-1.0, -1.0])
      labels = Nx.tensor([1, 1])
      loss_w1 = KTO.kto_loss(policy_lp, ref_lp, labels, desirable_weight: 1.0) |> Nx.to_number()
      loss_w2 = KTO.kto_loss(policy_lp, ref_lp, labels, desirable_weight: 2.0) |> Nx.to_number()
      # Doubling weight doubles the magnitude of the (negative) loss
      assert_in_delta loss_w2, loss_w1 * 2, 1.0e-5
    end

    test "undesirable_weight scales undesirable loss" do
      policy_lp = Nx.tensor([-1.0, -0.8])
      ref_lp = Nx.tensor([-1.0, -1.0])
      labels = Nx.tensor([0, 0])
      loss_w1 = KTO.kto_loss(policy_lp, ref_lp, labels, undesirable_weight: 1.0) |> Nx.to_number()
      loss_w2 = KTO.kto_loss(policy_lp, ref_lp, labels, undesirable_weight: 2.0) |> Nx.to_number()
      assert_in_delta loss_w2, loss_w1 * 2, 1.0e-5
    end

    test "beta scales the KL contribution" do
      policy_lp = Nx.tensor([-0.5, -1.5])
      ref_lp = Nx.tensor([-1.0, -1.0])
      labels = Nx.tensor([1, 0])
      loss_b001 = KTO.kto_loss(policy_lp, ref_lp, labels, beta: 0.01) |> Nx.to_number()
      loss_b100 = KTO.kto_loss(policy_lp, ref_lp, labels, beta: 100.0) |> Nx.to_number()

      # Very small beta → rewards ≈ 0 → sigmoid(0) ≈ 0.5; large beta → saturated → differ
      refute_in_delta loss_b001, loss_b100, 0.01
    end
  end

  describe "KTO.compute_logprobs/3" do
    test "returns [batch] tensor" do
      logits = Nx.broadcast(0.0, {@batch, @seq_len, @vocab_size})
      targets = Nx.broadcast(0, {@batch, @seq_len}) |> Nx.as_type(:s64)
      lp = KTO.compute_logprobs(logits, targets)
      assert Nx.shape(lp) == {@batch}
    end

    test "masking zeros out padded positions" do
      logits = Nx.broadcast(0.0, {@batch, @seq_len, @vocab_size})
      targets = Nx.broadcast(0, {@batch, @seq_len}) |> Nx.as_type(:s64)
      mask_full = Nx.broadcast(1, {@batch, @seq_len})

      mask_half =
        Nx.concatenate(
          [
            Nx.broadcast(1, {@batch, div(@seq_len, 2)}),
            Nx.broadcast(0, {@batch, div(@seq_len, 2)})
          ],
          axis: 1
        )

      lp_full = KTO.compute_logprobs(logits, targets, mask_full) |> Nx.to_flat_list()
      lp_half = KTO.compute_logprobs(logits, targets, mask_half) |> Nx.to_flat_list()
      # Half-masked should have smaller magnitude (less sequence summed)
      Enum.zip(lp_full, lp_half)
      |> Enum.each(fn {f, h} ->
        # log-probs are negative; summing fewer gives larger (closer to 0)
        assert f < h
      end)
    end
  end

  describe "KTO.default_beta/0" do
    test "returns 0.1" do
      assert KTO.default_beta() == 0.1
    end
  end
end
