defmodule Edifice.Meta.RouterNetworkTest do
  use ExUnit.Case, async: true
  @moduletag :meta
  @moduletag timeout: 120_000

  alias Edifice.Meta.RouterNetwork

  @batch_size 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  defp build_and_run(opts) do
    model = RouterNetwork.build(opts)
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
        num_specialists: 2,
        specialist_hidden_size: @hidden_size,
        specialist_layers: 1,
        num_heads: 4,
        dropout: 0.0,
        routing: :soft,
        window_size: @seq_len
      ],
      overrides
    )
  end

  describe "build/1 with soft routing" do
    test "produces correct output shape" do
      {output, _params} = build_and_run(default_opts())
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output is finite" do
      {output, _params} = build_and_run(default_opts())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with different num_specialists" do
      for n <- [2, 3, 5] do
        {output, _params} = build_and_run(default_opts(num_specialists: n))
        assert Nx.shape(output) == {@batch_size, @hidden_size}
      end
    end

    test "with output_hidden_size different from specialist_hidden_size" do
      {output, _params} =
        build_and_run(default_opts(specialist_hidden_size: 16, output_hidden_size: 64))

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "with embed_dim different from specialist_hidden_size" do
      opts = default_opts(embed_dim: 24)
      model = RouterNetwork.build(opts)
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

    test "produces different outputs for different inputs" do
      opts = default_opts()
      model = RouterNetwork.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)

      {input_a, key} =
        Nx.Random.uniform(key, -1.0, 1.0,
          shape: {@batch_size, @seq_len, @embed_dim},
          type: {:f, 32}
        )

      {input_b, _key} =
        Nx.Random.uniform(key, -1.0, 1.0,
          shape: {@batch_size, @seq_len, @embed_dim},
          type: {:f, 32}
        )

      output_a = predict_fn.(params, %{"state_sequence" => input_a})
      output_b = predict_fn.(params, %{"state_sequence" => input_b})

      max_diff = Nx.subtract(output_a, output_b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert max_diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end

  describe "build/1 with top_k routing" do
    test "produces correct output shape" do
      {output, _params} = build_and_run(default_opts(num_specialists: 4, routing: {:top_k, 2}))
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output is finite" do
      {output, _params} = build_and_run(default_opts(num_specialists: 4, routing: {:top_k, 2}))
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "top_k=1 selects single specialist" do
      {output, _params} = build_and_run(default_opts(num_specialists: 3, routing: {:top_k, 1}))
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns output_hidden_size" do
      assert RouterNetwork.output_size(output_hidden_size: 64) == 64
      assert RouterNetwork.output_size(specialist_hidden_size: 96) == 96
      assert RouterNetwork.output_size() == 128
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = RouterNetwork.recommended_defaults()
      assert Keyword.has_key?(defaults, :num_specialists)
      assert Keyword.has_key?(defaults, :routing)
      assert Keyword.has_key?(defaults, :router_hidden_size)
    end
  end

  describe "registry" do
    test "registered as :router_network" do
      assert Edifice.module_for(:router_network) == Edifice.Meta.RouterNetwork
    end

    test "appears in meta family" do
      families = Edifice.list_families()
      meta = Map.get(families, :meta)
      assert :router_network in meta
    end
  end
end
