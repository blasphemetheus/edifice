defmodule Edifice.Generative.MARTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.MAR

  @batch 2
  @seq_len 8
  @vocab_size 16
  @embed_dim 16
  @num_heads 2
  @num_layers 2

  @base_opts [
    vocab_size: @vocab_size,
    embed_dim: @embed_dim,
    num_layers: @num_layers,
    num_heads: @num_heads,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp build_and_init do
    model = MAR.build(@base_opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    tokens = Nx.broadcast(0, {@batch, @seq_len})
    params = init_fn.(Nx.template({@batch, @seq_len}, :s64), Axon.ModelState.empty())
    {predict_fn, params, tokens}
  end

  describe "MAR.build/1" do
    test "returns an Axon model" do
      assert %Axon{} = MAR.build(@base_opts)
    end

    test "output shape is [batch, seq_len, vocab_size]" do
      {predict_fn, params, tokens} = build_and_init()
      output = predict_fn.(params, %{"tokens" => tokens})
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "output contains finite values" do
      {predict_fn, params, tokens} = build_and_init()
      output = predict_fn.(params, %{"tokens" => tokens})
      assert Nx.all(Nx.logical_not(Nx.is_nan(output))) |> Nx.to_number() == 1
    end

    test "requires vocab_size" do
      assert_raise KeyError, fn -> MAR.build([]) end
    end
  end

  describe "MAR.mar_loss/3" do
    test "returns a scalar" do
      logits = Nx.broadcast(0.1, {@batch, @seq_len, @vocab_size})
      targets = Nx.broadcast(1, {@batch, @seq_len}) |> Nx.as_type(:s64)
      mask = Nx.broadcast(1, {@batch, @seq_len})
      loss = MAR.mar_loss(logits, targets, mask)
      assert Nx.shape(loss) == {}
    end

    test "loss is zero when mask is all zeros" do
      logits = Nx.broadcast(0.1, {@batch, @seq_len, @vocab_size})
      targets = Nx.broadcast(1, {@batch, @seq_len}) |> Nx.as_type(:s64)
      mask = Nx.broadcast(0, {@batch, @seq_len})
      loss = MAR.mar_loss(logits, targets, mask)
      # Numerator is 0, denominator is eps → ~0
      assert Nx.to_number(loss) < 1.0e-5
    end

    test "loss is positive when mask covers all positions" do
      logits = Nx.broadcast(0.1, {@batch, @seq_len, @vocab_size})
      targets = Nx.broadcast(1, {@batch, @seq_len}) |> Nx.as_type(:s64)
      mask = Nx.broadcast(1, {@batch, @seq_len})
      loss = MAR.mar_loss(logits, targets, mask)
      assert Nx.to_number(loss) > 0.0
    end

    test "partial mask only counts masked positions" do
      logits_uniform = Nx.broadcast(0.0, {@batch, @seq_len, @vocab_size})
      targets = Nx.broadcast(0, {@batch, @seq_len}) |> Nx.as_type(:s64)

      # Mask only first half
      half_mask =
        Nx.concatenate(
          [
            Nx.broadcast(1, {@batch, div(@seq_len, 2)}),
            Nx.broadcast(0, {@batch, div(@seq_len, 2)})
          ],
          axis: 1
        )

      full_mask = Nx.broadcast(1, {@batch, @seq_len})

      loss_half = MAR.mar_loss(logits_uniform, targets, half_mask) |> Nx.to_number()
      loss_full = MAR.mar_loss(logits_uniform, targets, full_mask) |> Nx.to_number()

      # Both compute same per-position CE (uniform logits); loss should be equal
      assert_in_delta loss_half, loss_full, 1.0e-4
    end
  end

  describe "MAR.sample_mask_ratio/0" do
    test "returns a float in [0, 1]" do
      ratio = MAR.sample_mask_ratio()
      assert is_float(ratio)
      assert ratio >= 0.0
      assert ratio <= 1.0
    end

    test "multiple calls produce different values (stochastic)" do
      ratios = for _ <- 1..20, do: MAR.sample_mask_ratio()
      unique = Enum.uniq(ratios)
      # With 20 draws it's extremely unlikely all are identical
      assert length(unique) > 1
    end
  end

  describe "MAR.iterative_decode/3" do
    test "returns [1, seq_len] token tensor" do
      model = MAR.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({1, @seq_len}, :s64), Axon.ModelState.empty())

      result =
        MAR.iterative_decode(model, params,
          num_steps: 3,
          seq_len: @seq_len,
          vocab_size: @vocab_size
        )

      assert Nx.shape(result) == {1, @seq_len}
    end

    test "all tokens are valid vocabulary ids after decoding" do
      model = MAR.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({1, @seq_len}, :s64), Axon.ModelState.empty())

      result =
        MAR.iterative_decode(model, params,
          num_steps: 4,
          seq_len: @seq_len,
          vocab_size: @vocab_size
        )

      flat = Nx.to_flat_list(result)
      assert Enum.all?(flat, fn t -> t >= 0 and t < @vocab_size end)
    end
  end
end
