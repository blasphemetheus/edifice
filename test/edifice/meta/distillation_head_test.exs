defmodule Edifice.Meta.DistillationHeadTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.DistillationHead

  @batch 2
  @seq_len 8
  @student_dim 32
  @teacher_dim 64
  @hidden_size 48

  @opts [
    embed_dim: @student_dim,
    teacher_dim: @teacher_dim,
    hidden_size: @hidden_size,
    num_layers: 2,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @student_dim})
    input
  end

  describe "build/1" do
    test "produces correct output shape [batch, seq_len, teacher_dim]" do
      model = DistillationHead.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @student_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @seq_len, @teacher_dim}
    end

    test "output is finite" do
      model = DistillationHead.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @student_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "distillation_loss/3" do
    test "returns a scalar" do
      teacher_logits = Nx.broadcast(0.5, {@batch, @seq_len, 64})
      student_logits = Nx.broadcast(0.3, {@batch, @seq_len, 64})

      loss = DistillationHead.distillation_loss(teacher_logits, student_logits)
      assert Nx.shape(loss) == {}
    end

    test "loss is non-negative" do
      teacher_logits = Nx.broadcast(0.5, {@batch, @seq_len, 64})
      student_logits = Nx.broadcast(0.3, {@batch, @seq_len, 64})

      loss = DistillationHead.distillation_loss(teacher_logits, student_logits)
      assert Nx.to_number(loss) >= 0
    end

    test "loss is zero when teacher equals student" do
      logits = Nx.broadcast(0.5, {@batch, @seq_len, 64})
      loss = DistillationHead.distillation_loss(logits, logits)
      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-4
    end
  end

  describe "hidden_state_loss/2" do
    test "returns a scalar" do
      student = Nx.broadcast(0.5, {@batch, @seq_len, @teacher_dim})
      teacher = Nx.broadcast(0.3, {@batch, @seq_len, @teacher_dim})

      loss = DistillationHead.hidden_state_loss(student, teacher)
      assert Nx.shape(loss) == {}
    end

    test "loss is zero when inputs match" do
      hidden = Nx.broadcast(0.5, {@batch, @seq_len, @teacher_dim})
      loss = DistillationHead.hidden_state_loss(hidden, hidden)
      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-6
    end
  end

  describe "output_size/1" do
    test "returns teacher_dim" do
      assert DistillationHead.output_size(@opts) == @teacher_dim
    end
  end
end
