defmodule Edifice.Recurrent.MultiTimescaleRecurrenceTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent.MultiTimescaleRecurrence

  @batch 2
  @seq_len 16
  @embed_dim 8
  @output_size 16

  defp build_and_run(opts \\ []) do
    default_opts = [
      embed_dim: @embed_dim,
      scales: [
        %{stride: 1, hidden_size: 8, num_layers: 1},
        %{stride: 4, hidden_size: 8, num_layers: 1}
      ],
      output_size: @output_size,
      dropout: 0.0,
      window_size: @seq_len
    ]

    model = MultiTimescaleRecurrence.build(Keyword.merge(default_opts, opts))
    {init_fn, predict_fn} = Axon.build(model)

    input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
    templates = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
    params = init_fn.(templates, Axon.ModelState.empty())
    output = predict_fn.(params, %{"state_sequence" => input})
    {output, params, predict_fn}
  end

  describe "build/1" do
    test "produces correct output shape" do
      {output, _, _} = build_and_run()
      assert Nx.shape(output) == {@batch, @output_size}
    end

    test "works with single scale (no subsampling)" do
      {output, _, _} =
        build_and_run(
          scales: [%{stride: 1, hidden_size: 12, num_layers: 1}],
          output_size: 8
        )

      assert Nx.shape(output) == {@batch, 8}
    end

    test "works with three timescales" do
      # seq_len=16 divisible by 1, 4, 8
      {output, _, _} =
        build_and_run(
          scales: [
            %{stride: 1, hidden_size: 8, num_layers: 1},
            %{stride: 4, hidden_size: 8, num_layers: 1},
            %{stride: 8, hidden_size: 4, num_layers: 1}
          ],
          output_size: 12
        )

      assert Nx.shape(output) == {@batch, 12}
    end

    test "different inputs produce different outputs" do
      model =
        MultiTimescaleRecurrence.build(
          embed_dim: @embed_dim,
          scales: [%{stride: 1, hidden_size: 8, num_layers: 1}],
          output_size: @output_size,
          dropout: 0.0,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      templates = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(templates, Axon.ModelState.empty())

      input_a = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      input_b = Nx.broadcast(-0.5, {@batch, @seq_len, @embed_dim})

      out_a = predict_fn.(params, %{"state_sequence" => input_a})
      out_b = predict_fn.(params, %{"state_sequence" => input_b})

      diff = Nx.subtract(out_a, out_b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-5
    end

    test "multi-layer GRU cores" do
      {output, _, _} =
        build_and_run(
          scales: [%{stride: 1, hidden_size: 8, num_layers: 2}]
        )

      assert Nx.shape(output) == {@batch, @output_size}
    end
  end

  describe "output_size/1" do
    test "returns configured output size" do
      assert MultiTimescaleRecurrence.output_size(output_size: 512) == 512
    end

    test "defaults to 256" do
      assert MultiTimescaleRecurrence.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with 3 scales" do
      defaults = MultiTimescaleRecurrence.recommended_defaults()
      assert length(Keyword.get(defaults, :scales)) == 3
      assert Keyword.get(defaults, :embed_dim) == 288
    end
  end

  describe "registry" do
    test "registered in Edifice" do
      assert Edifice.module_for(:multi_timescale_recurrence) ==
               Edifice.Recurrent.MultiTimescaleRecurrence
    end

    test "in recurrent family" do
      families = Edifice.list_families()
      assert :multi_timescale_recurrence in families[:recurrent]
    end
  end
end
