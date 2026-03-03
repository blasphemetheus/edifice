defmodule Edifice.Meta.MessagePassingAgentsTest do
  use ExUnit.Case, async: true
  @moduletag :meta
  @moduletag timeout: 120_000

  alias Edifice.Meta.MessagePassingAgents

  @batch_size 2
  @seq_len 8
  @embed_dim 32
  @agent_hidden 32
  @num_agents 3

  defp fully_connected_adj do
    Nx.broadcast(1.0, {@batch_size, @num_agents, @num_agents})
  end

  defp build_and_run(opts, adj \\ nil) do
    num_agents = Keyword.get(opts, :num_agents, @num_agents)
    model = MessagePassingAgents.build(opts)
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    adj_tensor = adj || Nx.broadcast(1.0, {@batch_size, num_agents, num_agents})

    params =
      init_fn.(
        %{
          "state_sequence" => Nx.template({@batch_size, @seq_len, embed_dim}, :f32),
          "adjacency" => Nx.template({@batch_size, num_agents, num_agents}, :f32)
        },
        Axon.ModelState.empty()
      )

    input = Nx.broadcast(0.5, {@batch_size, @seq_len, embed_dim})

    output =
      predict_fn.(params, %{
        "state_sequence" => input,
        "adjacency" => adj_tensor
      })

    {output, params, predict_fn}
  end

  defp default_opts(overrides \\ []) do
    Keyword.merge(
      [
        embed_dim: @embed_dim,
        num_agents: @num_agents,
        agent_hidden_size: @agent_hidden,
        agent_layers: 1,
        message_rounds: 2,
        output_size: @agent_hidden,
        num_heads: 4,
        dropout: 0.0,
        aggregation: :mean,
        pool_mode: :mean,
        window_size: @seq_len
      ],
      overrides
    )
  end

  describe "build/1 with fully-connected topology" do
    test "produces correct output shape" do
      {output, _params, _predict_fn} = build_and_run(default_opts())
      assert Nx.shape(output) == {@batch_size, @agent_hidden}
    end

    test "output is finite" do
      {output, _params, _predict_fn} = build_and_run(default_opts())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "produces different outputs for different inputs" do
      opts = default_opts()
      model = MessagePassingAgents.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{
            "state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_agents, @num_agents}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {input_a, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch_size, @seq_len, @embed_dim}, type: {:f, 32})
      {input_b, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch_size, @seq_len, @embed_dim}, type: {:f, 32})

      adj = fully_connected_adj()

      output_a = predict_fn.(params, %{"state_sequence" => input_a, "adjacency" => adj})
      output_b = predict_fn.(params, %{"state_sequence" => input_b, "adjacency" => adj})

      max_diff = Nx.subtract(output_a, output_b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert max_diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end

  describe "build/1 with custom adjacency" do
    test "ring topology produces correct shape" do
      ring_adj =
        Nx.tensor([
          [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
          ],
          [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
          ]
        ])

      {output, _params, _predict_fn} = build_and_run(default_opts(), ring_adj)
      assert Nx.shape(output) == {@batch_size, @agent_hidden}
    end

    test "sparse adjacency produces finite output" do
      sparse_adj =
        Nx.tensor([
          [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ],
          [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ]
        ])

      {output, _params, _predict_fn} = build_and_run(default_opts(), sparse_adj)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with sum aggregation" do
    test "produces correct output shape" do
      {output, _params, _predict_fn} = build_and_run(default_opts(aggregation: :sum))
      assert Nx.shape(output) == {@batch_size, @agent_hidden}
    end

    test "output is finite" do
      {output, _params, _predict_fn} = build_and_run(default_opts(aggregation: :sum))
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with output_size different from agent_hidden" do
    test "produces correct output shape" do
      {output, _params, _predict_fn} = build_and_run(default_opts(output_size: 64))
      assert Nx.shape(output) == {@batch_size, 64}
    end
  end

  describe "build/1 with max pooling" do
    test "produces correct output shape" do
      {output, _params, _predict_fn} = build_and_run(default_opts(pool_mode: :max))
      assert Nx.shape(output) == {@batch_size, @agent_hidden}
    end
  end

  describe "output_size/1" do
    test "returns output_size when specified" do
      assert MessagePassingAgents.output_size(output_size: 128) == 128
    end

    test "falls back to agent_hidden_size" do
      assert MessagePassingAgents.output_size(agent_hidden_size: 96) == 96
    end

    test "default is 64" do
      assert MessagePassingAgents.output_size() == 64
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MessagePassingAgents.recommended_defaults()
      assert Keyword.has_key?(defaults, :num_agents)
      assert Keyword.has_key?(defaults, :message_rounds)
      assert Keyword.has_key?(defaults, :aggregation)
    end
  end

  describe "registry" do
    test "registered as :message_passing_agents" do
      assert Edifice.module_for(:message_passing_agents) == Edifice.Meta.MessagePassingAgents
    end

    test "appears in meta family" do
      families = Edifice.list_families()
      meta = Map.get(families, :meta)
      assert :message_passing_agents in meta
    end
  end
end
