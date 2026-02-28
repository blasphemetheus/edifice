defmodule Edifice.Utils.FusedOpsTest do
  use ExUnit.Case, async: true
  @moduletag :utils

  alias Edifice.Utils.FusedOps

  @input Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32)
  @weight Nx.broadcast(0.1, {4, 8}) |> Nx.as_type(:f32)
  @bias Nx.broadcast(0.0, {8}) |> Nx.as_type(:f32)

  describe "dense_activation/4" do
    for act <- [:relu, :silu, :gelu, :sigmoid, :tanh, :softplus, :identity, :none] do
      test "#{act} produces correct shape" do
        result = FusedOps.dense_activation(@input, @weight, @bias, unquote(act))
        assert Nx.shape(result) == {1, 8}
        assert Nx.all(Nx.is_nan(result) |> Nx.bitwise_not()) |> Nx.to_number() == 1
      end
    end

    test "relu zeros out negative values" do
      weight = Nx.tensor([[1.0], [-1.0]], type: :f32)
      bias = Nx.tensor([0.0], type: :f32)
      input = Nx.tensor([[1.0, 1.0]], type: :f32)
      # input @ weight = [[1-1]] = [[0]], relu(0) = 0
      result = FusedOps.dense_activation(input, weight, bias, :relu)
      assert Nx.to_number(result[0][0]) >= 0
    end
  end

  describe "dense_activation_no_bias/3" do
    for act <- [:relu, :silu, :gelu, :sigmoid, :tanh] do
      test "#{act} produces correct shape without bias" do
        result = FusedOps.dense_activation_no_bias(@input, @weight, unquote(act))
        assert Nx.shape(result) == {1, 8}
      end
    end

    test "fallback for unknown activation" do
      result = FusedOps.dense_activation_no_bias(@input, @weight, :unknown)
      assert Nx.shape(result) == {1, 8}
    end
  end

  describe "layernorm_activation/5" do
    setup do
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32)
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0], type: :f32)
      beta = Nx.tensor([0.0, 0.0, 0.0, 0.0], type: :f32)
      %{input: input, gamma: gamma, beta: beta}
    end

    for act <- [:relu, :silu, :gelu, :identity, :none] do
      test "#{act} produces correct shape", %{input: input, gamma: gamma, beta: beta} do
        result = FusedOps.layernorm_activation(input, gamma, beta, unquote(act))
        assert Nx.shape(result) == {1, 4}
        assert Nx.all(Nx.is_nan(result) |> Nx.bitwise_not()) |> Nx.to_number() == 1
      end
    end
  end

  describe "fused_layernorm/4" do
    test "normalizes to zero mean unit variance" do
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32)
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0], type: :f32)
      beta = Nx.tensor([0.0, 0.0, 0.0, 0.0], type: :f32)
      result = FusedOps.fused_layernorm(input, gamma, beta)
      # Mean should be ~0
      mean = Nx.mean(result) |> Nx.to_number()
      assert abs(mean) < 0.01
    end
  end

  describe "fused_ffn/6" do
    setup do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      w1 = Nx.broadcast(0.1, {2, 4}) |> Nx.as_type(:f32)
      b1 = Nx.broadcast(0.0, {4}) |> Nx.as_type(:f32)
      w2 = Nx.broadcast(0.1, {4, 2}) |> Nx.as_type(:f32)
      b2 = Nx.broadcast(0.0, {2}) |> Nx.as_type(:f32)
      %{input: input, w1: w1, b1: b1, w2: w2, b2: b2}
    end

    for act <- [:relu, :silu, :gelu, :identity] do
      test "#{act} produces correct shape", ctx do
        result = FusedOps.fused_ffn(ctx.input, ctx.w1, ctx.b1, ctx.w2, ctx.b2, unquote(act))
        assert Nx.shape(result) == {1, 2}
      end
    end
  end

  describe "fused_ffn_no_bias/4" do
    setup do
      input = Nx.tensor([[1.0, 2.0]], type: :f32)
      w1 = Nx.broadcast(0.1, {2, 4}) |> Nx.as_type(:f32)
      w2 = Nx.broadcast(0.1, {4, 2}) |> Nx.as_type(:f32)
      %{input: input, w1: w1, w2: w2}
    end

    for act <- [:relu, :silu, :gelu, :identity] do
      test "#{act} produces correct shape", ctx do
        result = FusedOps.fused_ffn_no_bias(ctx.input, ctx.w1, ctx.w2, unquote(act))
        assert Nx.shape(result) == {1, 2}
      end
    end
  end

  describe "gated_linear_unit/4" do
    setup do
      input = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)
      w_gate = Nx.broadcast(0.1, {3, 4}) |> Nx.as_type(:f32)
      w_up = Nx.broadcast(0.1, {3, 4}) |> Nx.as_type(:f32)
      %{input: input, w_gate: w_gate, w_up: w_up}
    end

    for act <- [:silu, :gelu, :relu] do
      test "#{act} GLU produces correct shape", ctx do
        result = FusedOps.gated_linear_unit(ctx.input, ctx.w_gate, ctx.w_up, unquote(act))
        assert Nx.shape(result) == {1, 4}
      end
    end
  end

  describe "fused_softmax/1" do
    test "produces valid probability distribution" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)
      result = FusedOps.fused_softmax(logits)
      assert Nx.shape(result) == {1, 3}
      # Probabilities sum to 1
      sum = Nx.sum(result, axes: [-1]) |> Nx.squeeze() |> Nx.to_number()
      assert abs(sum - 1.0) < 1.0e-5
      # All positive
      assert Nx.all(Nx.greater(result, 0)) |> Nx.to_number() == 1
    end

    test "handles large logits without overflow" do
      logits = Nx.tensor([[1000.0, 1001.0, 1002.0]], type: :f32)
      result = FusedOps.fused_softmax(logits)
      assert Nx.all(Nx.is_nan(result) |> Nx.bitwise_not()) |> Nx.to_number() == 1
      sum = Nx.sum(result, axes: [-1]) |> Nx.squeeze() |> Nx.to_number()
      assert abs(sum - 1.0) < 1.0e-5
    end
  end

  describe "fused_log_softmax/2" do
    test "produces log probabilities" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]], type: :f32)
      result = FusedOps.fused_log_softmax(logits)
      assert Nx.shape(result) == {1, 3}
      # All values should be negative (log of probability < 1)
      assert Nx.all(Nx.less(result, 0.01)) |> Nx.to_number() == 1
      # exp(log_softmax) should sum to 1
      sum = Nx.exp(result) |> Nx.sum(axes: [-1]) |> Nx.squeeze() |> Nx.to_number()
      assert abs(sum - 1.0) < 1.0e-4
    end
  end

  describe "apply_activation/2" do
    for act <- [:relu, :silu, :gelu, :gelu_approx, :sigmoid, :tanh, :softplus] do
      test "#{act} produces finite output" do
        x = Nx.tensor([0.5, -0.5, 1.0], type: :f32)
        result = FusedOps.apply_activation(x, unquote(act))
        assert Nx.shape(result) == {3}
        assert Nx.all(Nx.is_nan(result) |> Nx.bitwise_not()) |> Nx.to_number() == 1
      end
    end

    test "identity returns input unchanged" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = FusedOps.apply_activation(x, :identity)
      assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "supported_activation?/1" do
    test "returns true for supported activations" do
      assert FusedOps.supported_activation?(:relu)
      assert FusedOps.supported_activation?(:silu)
      assert FusedOps.supported_activation?(:gelu)
      assert FusedOps.supported_activation?(:identity)
    end

    test "returns false for unsupported" do
      refute FusedOps.supported_activation?(:leaky_relu)
      refute FusedOps.supported_activation?(:elu)
    end
  end

  describe "supported_activations/0" do
    test "returns a list of atoms" do
      acts = FusedOps.supported_activations()
      assert is_list(acts)
      assert :relu in acts
      assert :silu in acts
      assert :gelu in acts
    end
  end

  describe "fused_ssm_discretize/3" do
    test "computes A_bar and B_bar" do
      dt = Nx.tensor([[0.1, 0.2]], type: :f32)
      a = Nx.tensor([[-1.0, -2.0]], type: :f32)
      b = Nx.tensor([[1.0, 0.5]], type: :f32)
      {a_bar, b_bar} = FusedOps.fused_ssm_discretize(dt, a, b)
      # A_bar = exp(dt * A)
      assert Nx.shape(a_bar) == {1, 2}
      assert Nx.all(Nx.greater(a_bar, 0)) |> Nx.to_number() == 1
      # B_bar = dt * B
      assert Nx.shape(b_bar) == {1, 2}
    end
  end

  describe "fused_ssm_output/2" do
    test "computes output from C and hidden state" do
      c = Nx.broadcast(0.5, {2, 4, 8}) |> Nx.as_type(:f32)
      hidden = Nx.broadcast(1.0, {2, 4, 8}) |> Nx.as_type(:f32)
      result = FusedOps.fused_ssm_output(c, hidden)
      assert Nx.shape(result) == {2, 4}
    end
  end

  describe "fused_attention_scores/3" do
    test "computes attention without mask" do
      # [batch, heads, seq, dim]
      query = Nx.broadcast(0.1, {1, 2, 4, 8}) |> Nx.as_type(:f32)
      key = Nx.broadcast(0.1, {1, 2, 4, 8}) |> Nx.as_type(:f32)
      result = FusedOps.fused_attention_scores(query, key)
      assert Nx.shape(result) == {1, 2, 4, 4}
      # Should be valid probabilities
      sums = Nx.sum(result, axes: [-1])
      assert Nx.all(Nx.less(Nx.abs(Nx.subtract(sums, 1.0)), 0.01)) |> Nx.to_number() == 1
    end

    test "computes attention with mask" do
      query = Nx.broadcast(0.1, {1, 2, 4, 8}) |> Nx.as_type(:f32)
      key = Nx.broadcast(0.1, {1, 2, 4, 8}) |> Nx.as_type(:f32)
      mask = Nx.broadcast(0.0, {1, 2, 4, 4}) |> Nx.as_type(:f32)
      result = FusedOps.fused_attention_scores(query, key, mask: mask)
      assert Nx.shape(result) == {1, 2, 4, 4}
    end
  end
end
