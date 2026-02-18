defmodule Edifice.Meta.RLHFHeadTest do
  use ExUnit.Case, async: true
  @moduletag timeout: 120_000

  alias Edifice.Meta.RLHFHead

  @batch_size 2
  @seq_len 8
  @input_size 32
  @hidden_size 32

  describe "build/1 with :reward head" do
    test "produces correct output shape" do
      model =
        RLHFHead.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          head_type: :reward,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      # Scalar per batch element
      assert Nx.shape(output) == {@batch_size}
    end

    test "output is finite" do
      model =
        RLHFHead.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          head_type: :reward,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "defaults to :reward head_type" do
      model =
        RLHFHead.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size}
    end
  end

  describe "build/1 with :dpo head" do
    test "produces correct output shape" do
      model =
        RLHFHead.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          head_type: :dpo,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "chosen" => Nx.template({@batch_size, @seq_len, @input_size}, :f32),
        "rejected" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "chosen" => Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size}),
        "rejected" => Nx.broadcast(0.3, {@batch_size, @seq_len, @input_size})
      }

      output = predict_fn.(params, input)

      # Preference logit: scalar per batch element
      assert Nx.shape(output) == {@batch_size}
    end

    test "output is finite" do
      model =
        RLHFHead.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          head_type: :dpo,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "chosen" => Nx.template({@batch_size, @seq_len, @input_size}, :f32),
        "rejected" => Nx.template({@batch_size, @seq_len, @input_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "chosen" => Nx.broadcast(0.5, {@batch_size, @seq_len, @input_size}),
        "rejected" => Nx.broadcast(0.3, {@batch_size, @seq_len, @input_size})
      }

      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "always returns 1" do
      assert RLHFHead.output_size() == 1
      assert RLHFHead.output_size(hidden_size: 512) == 1
    end
  end
end
