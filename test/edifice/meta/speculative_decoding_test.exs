defmodule Edifice.Meta.SpeculativeDecodingTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.SpeculativeDecoding

  @embed_dim 32

  @opts [
    embed_dim: @embed_dim,
    draft_model_opts: [hidden_size: 16, num_layers: 1, num_heads: 2, num_kv_heads: 1],
    verifier_model_opts: [hidden_size: 32, num_layers: 2, num_heads: 4, num_kv_heads: 2]
  ]

  describe "build/1" do
    test "returns a 2-tuple of Axon models" do
      {draft, verifier} = SpeculativeDecoding.build(@opts)
      assert %Axon{} = draft
      assert %Axon{} = verifier
    end

    test "draft and verifier can run forward passes" do
      {draft, verifier} = SpeculativeDecoding.build(@opts)

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {2, 12, @embed_dim})

      # Draft
      {init_fn, predict_fn} = Axon.build(draft)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({2, 12, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      draft_out = predict_fn.(params, %{"state_sequence" => input})
      assert {2, 16} = Nx.shape(draft_out)

      # Verifier
      {init_fn2, predict_fn2} = Axon.build(verifier)

      params2 =
        init_fn2.(
          %{"state_sequence" => Nx.template({2, 12, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      verifier_out = predict_fn2.(params2, %{"state_sequence" => input})
      assert {2, 32} = Nx.shape(verifier_out)
    end
  end

  describe "accept_reject/2" do
    test "all tokens match — accepts all" do
      draft = Nx.tensor([1, 2, 3, 4, 5])
      verifier = Nx.tensor([1, 2, 3, 4, 5])

      accepted = SpeculativeDecoding.accept_reject(draft, verifier)
      assert Nx.to_number(accepted) == 5
    end

    test "first token mismatch — accepts 0" do
      draft = Nx.tensor([1, 2, 3, 4, 5])
      verifier = Nx.tensor([9, 2, 3, 4, 5])

      accepted = SpeculativeDecoding.accept_reject(draft, verifier)
      assert Nx.to_number(accepted) == 0
    end

    test "mismatch at position 3 — accepts 3" do
      draft = Nx.tensor([1, 2, 3, 4, 5])
      verifier = Nx.tensor([1, 2, 3, 9, 5])

      accepted = SpeculativeDecoding.accept_reject(draft, verifier)
      assert Nx.to_number(accepted) == 3
    end

    test "all different — accepts 0" do
      draft = Nx.tensor([1, 2, 3, 4, 5])
      verifier = Nx.tensor([6, 7, 8, 9, 10])

      accepted = SpeculativeDecoding.accept_reject(draft, verifier)
      assert Nx.to_number(accepted) == 0
    end

    test "batched accept_reject" do
      draft = Nx.tensor([[1, 2, 3], [1, 2, 3]])
      verifier = Nx.tensor([[1, 2, 3], [1, 9, 3]])

      accepted = SpeculativeDecoding.accept_reject(draft, verifier)
      assert Nx.to_flat_list(accepted) == [3, 1]
    end
  end

  describe "output_size/1" do
    test "returns verifier hidden_size" do
      assert SpeculativeDecoding.output_size(@opts) == 32
    end
  end
end
