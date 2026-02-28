defmodule Edifice.Recurrent.NativeRecurrenceTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent.NativeRecurrence

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_layers 2

  defp base_opts(recurrence_type) do
    [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_layers: @num_layers,
      recurrence_type: recurrence_type,
      seq_len: @seq_len,
      dropout: 0.0
    ]
  end

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1 with :elu_gru" do
    test "produces correct output shape" do
      model = NativeRecurrence.build(base_opts(:elu_gru))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      model = NativeRecurrence.build(base_opts(:elu_gru))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with :real_gru" do
    test "produces correct output shape" do
      model = NativeRecurrence.build(base_opts(:real_gru))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      model = NativeRecurrence.build(base_opts(:real_gru))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with :diag_linear" do
    test "produces correct output shape" do
      model = NativeRecurrence.build(base_opts(:diag_linear))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      model = NativeRecurrence.build(base_opts(:diag_linear))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert NativeRecurrence.output_size(hidden_size: 128) == 128
    end
  end
end
