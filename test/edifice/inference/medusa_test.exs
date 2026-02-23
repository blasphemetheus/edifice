defmodule Edifice.Inference.MedusaTest do
  use ExUnit.Case, async: true

  alias Edifice.Inference.Medusa

  @batch_size 2
  @base_hidden_dim 64
  @vocab_size 256

  describe "Medusa.build/1" do
    test "produces correct output shape with 4 heads" do
      model = Medusa.build(
        base_hidden_dim: @base_hidden_dim,
        vocab_size: @vocab_size,
        num_medusa_heads: 4,
        medusa_num_layers: 1
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"hidden_states" => Nx.template({@batch_size, @base_hidden_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @base_hidden_dim})
      output = predict_fn.(params, %{"hidden_states" => input})

      # Output should be a map with head_1..head_4
      assert is_map(output)
      assert Map.has_key?(output, :head_1)
      assert Map.has_key?(output, :head_4)
      refute Map.has_key?(output, :head_5)

      # Each head outputs [batch, vocab_size]
      assert Nx.shape(output.head_1) == {@batch_size, @vocab_size}
      assert Nx.shape(output.head_4) == {@batch_size, @vocab_size}
    end

    test "with 2 heads and 2 layers per head" do
      model = Medusa.build(
        base_hidden_dim: @base_hidden_dim,
        vocab_size: @vocab_size,
        num_medusa_heads: 2,
        medusa_num_layers: 2
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"hidden_states" => Nx.template({@batch_size, @base_hidden_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @base_hidden_dim})
      output = predict_fn.(params, %{"hidden_states" => input})

      assert Map.has_key?(output, :head_1)
      assert Map.has_key?(output, :head_2)
      refute Map.has_key?(output, :head_3)

      assert Nx.shape(output.head_1) == {@batch_size, @vocab_size}
    end
  end

  describe "Medusa.medusa_heads/2" do
    test "builds heads as Axon subgraph" do
      input = Axon.input("hidden_states", shape: {nil, @base_hidden_dim})

      heads = Medusa.medusa_heads(input,
        base_hidden_dim: @base_hidden_dim,
        vocab_size: @vocab_size,
        num_medusa_heads: 3
      )

      {init_fn, predict_fn} = Axon.build(heads, mode: :inference)

      params =
        init_fn.(
          %{"hidden_states" => Nx.template({@batch_size, @base_hidden_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @base_hidden_dim})
      output = predict_fn.(params, %{"hidden_states" => input_data})

      assert Map.has_key?(output, :head_1)
      assert Map.has_key?(output, :head_2)
      assert Map.has_key?(output, :head_3)
      assert Nx.shape(output.head_1) == {@batch_size, @vocab_size}
    end
  end

  describe "Medusa.build_tree_candidates/2" do
    test "generates candidate tree with correct shape" do
      # Simulate logits from 3 heads
      head_logits = %{
        head_1: Nx.broadcast(0.0, {1, @vocab_size}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0, 2.0, 3.0]])),
        head_2: Nx.broadcast(0.0, {1, @vocab_size}) |> Nx.put_slice([0, 0], Nx.tensor([[2.0, 1.0, 3.0]])),
        head_3: Nx.broadcast(0.0, {1, @vocab_size}) |> Nx.put_slice([0, 0], Nx.tensor([[3.0, 2.0, 1.0]]))
      }

      {candidates, tree_indices} = Medusa.build_tree_candidates(head_logits, top_k: 2)

      # candidates: [num_cands, 3] (3 heads)
      assert Nx.rank(candidates) == 2
      assert Nx.axis_size(candidates, 1) == 3

      # tree_indices: [num_cands]
      assert Nx.rank(tree_indices) == 1
      assert Nx.axis_size(tree_indices, 0) == Nx.axis_size(candidates, 0)
    end

    test "top_k limits candidate tokens per head" do
      # Create logits where top tokens are clearly defined
      head_logits = %{
        head_1: Nx.iota({1, 10}) |> Nx.as_type(:f32),
        head_2: Nx.iota({1, 10}) |> Nx.as_type(:f32)
      }

      {candidates, _tree_indices} = Medusa.build_tree_candidates(head_logits, top_k: 3)

      # With top_k=3 and 2 heads, we get 3*3=9 candidates max (but capped)
      assert Nx.axis_size(candidates, 0) <= 9
    end
  end

  describe "Medusa.tree_decoding_mask/1" do
    test "produces correct mask shape" do
      # 4 candidates, each of length 3
      candidates = Nx.iota({4, 3}, type: :s64)

      mask = Medusa.tree_decoding_mask(candidates)

      # Total positions = 4 * 3 = 12
      assert Nx.shape(mask) == {12, 12}
      assert Nx.type(mask) == {:u, 8}
    end

    test "mask is causal within each candidate path" do
      # 2 candidates, each of length 2
      candidates = Nx.tensor([[1, 2], [3, 4]])
      mask = Medusa.tree_decoding_mask(candidates)

      # Position 0 (cand 0, depth 0) can attend to position 0 only
      # Position 1 (cand 0, depth 1) can attend to positions 0 and 1
      # Position 2 (cand 1, depth 0) can attend to position 2 only
      # Position 3 (cand 1, depth 1) can attend to positions 2 and 3

      # Check diagonal is always true (self-attention)
      for i <- 0..3 do
        assert Nx.to_number(mask[i][i]) == 1
      end

      # Check cross-candidate attention is blocked
      # Position 0 shouldn't attend to position 2 or 3
      assert Nx.to_number(mask[0][2]) == 0
      assert Nx.to_number(mask[0][3]) == 0
    end
  end

  describe "Medusa.output_size/1" do
    test "returns vocab_size" do
      assert Medusa.output_size(vocab_size: 32000) == 32000
      assert Medusa.output_size(vocab_size: 50257) == 50257
    end
  end
end
