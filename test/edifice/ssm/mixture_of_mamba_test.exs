defmodule Edifice.SSM.MixtureOfMambaTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  import Edifice.TestHelpers

  alias Edifice.SSM.MixtureOfMamba

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 16
  @state_size 4
  @num_modalities 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    state_size: @state_size,
    num_layers: 1,
    num_modalities: @num_modalities,
    expand_factor: 2,
    window_size: @seq_len
  ]

  defp build_and_run(opts) do
    model = MixtureOfMamba.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    embed = opts[:embed_dim] || @embed_dim
    seq = opts[:window_size] || @seq_len

    template = %{
      "state_sequence" => Nx.template({@batch, seq, embed}, :f32),
      "modality_mask" => Nx.template({@batch, seq}, :s32)
    }

    params = init_fn.(template, Axon.ModelState.empty())

    # Create mixed modality input (alternating 0s and 1s)
    input = %{
      "state_sequence" => random_tensor({@batch, seq, embed}),
      "modality_mask" => modality_mask(@batch, seq, opts[:num_modalities] || @num_modalities)
    }

    output = predict_fn.(params, input)
    {model, output}
  end

  # Create alternating modality mask
  defp modality_mask(batch, seq_len, num_modalities) do
    Nx.iota({batch, seq_len}, axis: 1)
    |> Nx.remainder(num_modalities)
    |> Nx.as_type(:s32)
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = MixtureOfMamba.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output values are finite" do
      {_model, output} = build_and_run(@opts)
      assert_finite!(output)
    end

    test "works with multiple layers" do
      {_model, output} = build_and_run(Keyword.put(@opts, :num_layers, 2))
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "works with 3 modalities" do
      opts = Keyword.put(@opts, :num_modalities, 3)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "handles uniform modality mask (all same modality)" do
      model = MixtureOfMamba.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "modality_mask" => Nx.template({@batch, @seq_len}, :s32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "state_sequence" => random_tensor({@batch, @seq_len, @embed_dim}),
        "modality_mask" => Nx.broadcast(Nx.tensor(0, type: :s32), {@batch, @seq_len})
      }

      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MixtureOfMamba.output_size(hidden_size: 128) == 128
    end

    test "returns default when no option" do
      assert MixtureOfMamba.output_size([]) == 256
    end
  end

  describe "registry integration" do
    test "Edifice.build(:mixture_of_mamba, ...) works" do
      model = Edifice.build(:mixture_of_mamba, @opts)
      assert %Axon{} = model
    end
  end
end
