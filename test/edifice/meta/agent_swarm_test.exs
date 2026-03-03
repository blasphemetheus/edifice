defmodule Edifice.Meta.AgentSwarmTest do
  use ExUnit.Case, async: true
  @moduletag :meta
  @moduletag timeout: 120_000

  alias Edifice.Meta.AgentSwarm

  @batch_size 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  defp build_and_run(opts) do
    model = AgentSwarm.build(opts)
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    params =
      init_fn.(
        %{"state_sequence" => Nx.template({@batch_size, @seq_len, embed_dim}, :f32)},
        Axon.ModelState.empty()
      )

    input = Nx.broadcast(0.5, {@batch_size, @seq_len, embed_dim})
    output = predict_fn.(params, %{"state_sequence" => input})
    {output, params}
  end

  defp default_opts(overrides \\ []) do
    Keyword.merge(
      [
        embed_dim: @embed_dim,
        num_agents: 2,
        agent_hidden_size: @hidden_size,
        agent_layers: 1,
        communication_rounds: 1,
        aggregator_hidden_size: @hidden_size,
        aggregator_layers: 1,
        num_heads: 4,
        dropout: 0.0,
        window_size: @seq_len
      ],
      overrides
    )
  end

  describe "build/1" do
    test "produces correct output shape" do
      {output, _params} = build_and_run(default_opts())
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output is finite" do
      {output, _params} = build_and_run(default_opts())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with different num_agents" do
      for n <- [2, 3, 5] do
        {output, _params} = build_and_run(default_opts(num_agents: n))
        assert Nx.shape(output) == {@batch_size, @hidden_size}
      end
    end

    test "with multiple communication rounds" do
      {output, _params} = build_and_run(default_opts(communication_rounds: 3))
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with communication_gate enabled" do
      {output, _params} = build_and_run(default_opts(communication_gate: true))
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "without agent embeddings" do
      {output, _params} = build_and_run(default_opts(use_agent_embeddings: false))
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with embed_dim different from agent_hidden_size" do
      opts = default_opts(embed_dim: 24, agent_hidden_size: @hidden_size)
      model = AgentSwarm.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, 24}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 24})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with larger aggregator than agents" do
      {output, _params} =
        build_and_run(default_opts(agent_hidden_size: 16, aggregator_hidden_size: 64))

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "produces different outputs for different inputs" do
      opts = default_opts()
      model = AgentSwarm.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input_a = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
      input_b = Nx.broadcast(-0.5, {@batch_size, @seq_len, @embed_dim})

      output_a = predict_fn.(params, %{"state_sequence" => input_a})
      output_b = predict_fn.(params, %{"state_sequence" => input_b})

      max_diff = Nx.subtract(output_a, output_b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert max_diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end

  describe "output_size/1" do
    test "returns aggregator_hidden_size" do
      assert AgentSwarm.output_size(aggregator_hidden_size: 128) == 128
      assert AgentSwarm.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = AgentSwarm.recommended_defaults()
      assert Keyword.has_key?(defaults, :num_agents)
      assert Keyword.has_key?(defaults, :communication_rounds)
      assert Keyword.has_key?(defaults, :use_agent_embeddings)
    end
  end

  describe "registry" do
    test "registered as :agent_swarm" do
      assert Edifice.module_for(:agent_swarm) == Edifice.Meta.AgentSwarm
    end

    test "appears in meta family" do
      families = Edifice.list_families()
      meta = Map.get(families, :meta)
      assert :agent_swarm in meta
    end
  end
end
