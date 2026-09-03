defmodule Edifice.Recurrent.CarryBackboneTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent

  @batch 2
  @embed_dim 16
  @hidden 24
  @layers 2

  defp build_carry_model do
    input = Axon.input("state_sequence", shape: {nil, nil, @embed_dim})

    Recurrent.build_backbone_with_carry(input,
      hidden_size: @hidden,
      num_layers: @layers,
      dropout: 0.0
    )
  end

  defp init(model, seq_len) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    params =
      init_fn.(
        %{
          "state_sequence" => Nx.template({@batch, seq_len, @embed_dim}, :f32),
          "initial_hidden" => Nx.template({@batch, @layers, @hidden}, :f32)
        },
        Axon.ModelState.empty()
      )

    {params, predict_fn}
  end

  test "outputs full sequence and stacked final hidden" do
    seq_len = 10
    model = build_carry_model()
    {params, predict_fn} = init(model, seq_len)

    x = Nx.iota({@batch, seq_len, @embed_dim}, type: :f32) |> Nx.divide(100)
    h0 = Nx.broadcast(0.0, {@batch, @layers, @hidden})

    %{output: out, hidden: hf} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h0})

    assert Nx.shape(out) == {@batch, seq_len, @hidden}
    assert Nx.shape(hf) == {@batch, @layers, @hidden}
  end

  test "two carried chunks == one long unroll (the BPTT equivalence law)" do
    seq_len = 12
    half = 6
    model = build_carry_model()
    {params, predict_fn} = init(model, seq_len)

    x = Nx.iota({@batch, seq_len, @embed_dim}, type: :f32) |> Nx.divide(100)
    h0 = Nx.broadcast(0.0, {@batch, @layers, @hidden})

    # One long unroll
    %{output: out_full, hidden: hf_full} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h0})

    # Two chunks, feeding chunk 1's final hidden into chunk 2
    x1 = Nx.slice_along_axis(x, 0, half, axis: 1)
    x2 = Nx.slice_along_axis(x, half, half, axis: 1)

    %{output: out1, hidden: h_mid} =
      predict_fn.(params, %{"state_sequence" => x1, "initial_hidden" => h0})

    %{output: out2, hidden: hf_chunked} =
      predict_fn.(params, %{"state_sequence" => x2, "initial_hidden" => h_mid})

    out_chunked = Nx.concatenate([out1, out2], axis: 1)

    assert_all_close(out_chunked, out_full)
    assert_all_close(hf_chunked, hf_full)
  end

  test "zero h0 differs from carried h0 (state actually flows)" do
    seq_len = 6
    model = build_carry_model()
    {params, predict_fn} = init(model, seq_len)

    x = Nx.iota({@batch, seq_len, @embed_dim}, type: :f32) |> Nx.divide(100)
    h0 = Nx.broadcast(0.0, {@batch, @layers, @hidden})

    %{hidden: h_after} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h0})

    %{output: out_zero} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h0})

    %{output: out_carried} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h_after})

    refute Nx.all_close(out_zero, out_carried, atol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "per-row reset: zeroed rows match a fresh start" do
    seq_len = 6
    model = build_carry_model()
    {params, predict_fn} = init(model, seq_len)

    x = Nx.iota({@batch, seq_len, @embed_dim}, type: :f32) |> Nx.divide(100)
    h0 = Nx.broadcast(0.0, {@batch, @layers, @hidden})

    %{hidden: h_after} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h0})

    # Zero only row 0 (as the trainer does for is_resetting rows)
    mask = Nx.tensor([[0.0], [1.0]]) |> Nx.reshape({@batch, 1, 1})
    h_masked = Nx.multiply(h_after, mask)

    %{output: out_masked} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h_masked})

    %{output: out_zero} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h0})

    %{output: out_carried} =
      predict_fn.(params, %{"state_sequence" => x, "initial_hidden" => h_after})

    # Row 0 behaves like a fresh start, row 1 like a carried row
    assert_all_close(out_masked[0], out_zero[0])
    assert_all_close(out_masked[1], out_carried[1])
  end

  test "GRU kernel param names match the carryless build (trunk transplant)" do
    model = build_carry_model()
    {init_fn, _} = Axon.build(model, mode: :inference)

    params =
      init_fn.(
        %{
          "state_sequence" => Nx.template({@batch, 8, @embed_dim}, :f32),
          "initial_hidden" => Nx.template({@batch, @layers, @hidden}, :f32)
        },
        Axon.ModelState.empty()
      )

    keys = params.data |> Map.keys() |> MapSet.new()
    assert MapSet.member?(keys, "gru_1")
    assert MapSet.member?(keys, "gru_2")
    assert MapSet.member?(keys, "gru_1_ln")
    assert MapSet.member?(keys, "input_ln")
    # No RNG-key initial-state param in carry mode
    refute MapSet.member?(keys, "gru_1_h_hidden_state")
  end

  defp assert_all_close(a, b) do
    assert Nx.all_close(a, b, atol: 1.0e-5, rtol: 1.0e-5) |> Nx.to_number() == 1,
           "tensors differ: #{inspect(Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number())}"
  end
end
