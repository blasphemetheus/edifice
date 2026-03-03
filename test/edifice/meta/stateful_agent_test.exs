defmodule Edifice.Meta.StatefulAgentTest do
  use ExUnit.Case, async: true
  @moduletag :meta
  @moduletag timeout: 120_000

  alias Edifice.Meta.StatefulAgent

  @batch_size 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @state_size 16

  defp build_and_run(opts) do
    model = StatefulAgent.build(opts)
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    state_size = Keyword.get(opts, :state_size, @state_size)

    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    params =
      init_fn.(
        %{
          "state_sequence" => Nx.template({@batch_size, @seq_len, embed_dim}, :f32),
          "agent_state" => Nx.template({@batch_size, state_size}, :f32)
        },
        Axon.ModelState.empty()
      )

    input = Nx.broadcast(0.5, {@batch_size, @seq_len, embed_dim})
    zero_state = Nx.broadcast(0.0, {@batch_size, state_size})

    result = predict_fn.(params, %{
      "state_sequence" => input,
      "agent_state" => zero_state
    })

    {result, params, predict_fn}
  end

  defp default_opts(overrides \\ []) do
    Keyword.merge(
      [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        num_layers: 1,
        num_heads: 4,
        state_size: @state_size,
        state_mode: :compressive,
        dropout: 0.0,
        window_size: @seq_len
      ],
      overrides
    )
  end

  describe "build/1 with compressive state" do
    test "produces correct output shapes" do
      {result, _params, _predict_fn} = build_and_run(default_opts())
      {output, new_state} = result

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.shape(new_state) == {@batch_size, @state_size}
    end

    test "output and state are finite" do
      {result, _params, _predict_fn} = build_and_run(default_opts())
      {output, new_state} = result

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(new_state) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "state changes across turns with different inputs" do
      {_result, params, predict_fn} = build_and_run(default_opts())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch_size, @seq_len, @embed_dim}, type: {:f, 32})
      {input2, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch_size, @seq_len, @embed_dim}, type: {:f, 32})
      zero_state = Nx.broadcast(0.0, {@batch_size, @state_size})

      # Turn 1
      {_out1, state1} = predict_fn.(params, %{
        "state_sequence" => input1,
        "agent_state" => zero_state
      })

      # Turn 2: different input + state from turn 1
      {_out2, state2} = predict_fn.(params, %{
        "state_sequence" => input2,
        "agent_state" => state1
      })

      # States should differ (different inputs + accumulated memory)
      diff = Nx.subtract(state1, state2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "State should change between turns (diff=#{diff})"
    end
  end

  describe "build/1 with ema state" do
    test "produces correct output shapes" do
      {result, _params, _predict_fn} = build_and_run(default_opts(state_mode: :ema))
      {output, new_state} = result

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.shape(new_state) == {@batch_size, @state_size}
    end

    test "output and state are finite" do
      {result, _params, _predict_fn} = build_and_run(default_opts(state_mode: :ema))
      {output, new_state} = result

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(new_state) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with gru state" do
    test "produces correct output shapes" do
      {result, _params, _predict_fn} = build_and_run(default_opts(state_mode: :gru))
      {output, new_state} = result

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.shape(new_state) == {@batch_size, @state_size}
    end

    test "output and state are finite" do
      {result, _params, _predict_fn} = build_and_run(default_opts(state_mode: :gru))
      {output, new_state} = result

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(new_state) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "state changes across turns with different inputs" do
      {_result, params, predict_fn} = build_and_run(default_opts(state_mode: :gru))

      key = Nx.Random.key(99)
      {input1, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch_size, @seq_len, @embed_dim}, type: {:f, 32})
      {input2, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch_size, @seq_len, @embed_dim}, type: {:f, 32})
      zero_state = Nx.broadcast(0.0, {@batch_size, @state_size})

      {_out1, state1} = predict_fn.(params, %{
        "state_sequence" => input1,
        "agent_state" => zero_state
      })

      {_out2, state2} = predict_fn.(params, %{
        "state_sequence" => input2,
        "agent_state" => state1
      })

      diff = Nx.subtract(state1, state2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "GRU state should change between turns (diff=#{diff})"
    end
  end

  describe "output_size/1" do
    test "returns {hidden_size, state_size} tuple" do
      assert StatefulAgent.output_size(hidden_size: 128, state_size: 64) == {128, 64}
      assert StatefulAgent.output_size() == {128, 64}
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = StatefulAgent.recommended_defaults()
      assert Keyword.has_key?(defaults, :state_size)
      assert Keyword.has_key?(defaults, :state_mode)
    end
  end

  describe "registry" do
    test "registered as :stateful_agent" do
      assert Edifice.module_for(:stateful_agent) == Edifice.Meta.StatefulAgent
    end

    test "appears in meta family" do
      families = Edifice.list_families()
      meta = Map.get(families, :meta)
      assert :stateful_agent in meta
    end
  end
end
