defmodule Edifice.Meta.MixtureOfTransformersTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.MixtureOfTransformers

  @batch 2
  @seq_len 8
  @vocab_size 32
  @hidden_size 32
  @num_heads 4
  @num_modalities 2

  @small_opts [
    vocab_size: @vocab_size,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: 2,
    intermediate_size: 64,
    num_modalities: @num_modalities,
    seq_len: @seq_len
  ]

  defp random_input(num_modalities) do
    key = Nx.Random.key(42)

    {tokens, key} =
      Nx.Random.randint(key, 0, @vocab_size, shape: {@batch, @seq_len}, type: :s64)

    # Create mutually exclusive modality masks
    # Assign each position to a random modality
    {rand_vals, _key} =
      Nx.Random.randint(key, 0, num_modalities, shape: {@batch, @seq_len}, type: :s64)

    # One-hot encode: [batch, seq, num_modalities]
    modality_mask =
      Nx.equal(Nx.new_axis(rand_vals, -1), Nx.iota({num_modalities}))
      |> Nx.as_type(:f32)

    %{
      "tokens" => tokens,
      "modality_mask" => modality_mask
    }
  end

  defp build_and_run(opts) do
    num_modalities = Keyword.get(opts, :num_modalities, @num_modalities)
    model = MixtureOfTransformers.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "tokens" => Nx.template({@batch, @seq_len}, :s64),
      "modality_mask" => Nx.template({@batch, @seq_len, num_modalities}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_input(num_modalities))
    {model, output}
  end

  describe "MixtureOfTransformers.build/1" do
    test "returns an Axon model" do
      model = MixtureOfTransformers.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@small_opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@small_opts)
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "configuration variants" do
    test "three modalities" do
      opts = Keyword.put(@small_opts, :num_modalities, 3)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "single layer" do
      opts = Keyword.put(@small_opts, :num_layers, 1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "larger vocab" do
      opts = Keyword.put(@small_opts, :vocab_size, 64)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, 64}
    end
  end

  describe "modality masking" do
    test "uniform mask produces same shape" do
      model = MixtureOfTransformers.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "tokens" => Nx.template({@batch, @seq_len}, :s64),
        "modality_mask" => Nx.template({@batch, @seq_len, @num_modalities}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {tokens, _key} = Nx.Random.randint(key, 0, @vocab_size, shape: {@batch, @seq_len}, type: :s64)

      # All tokens belong to modality 0
      mask = Nx.broadcast(0.0, {@batch, @seq_len, @num_modalities})
      mask = Nx.put_slice(mask, [0, 0, 0], Nx.broadcast(1.0, {@batch, @seq_len, 1}))

      output = predict_fn.(params, %{"tokens" => tokens, "modality_mask" => mask})
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns vocab_size" do
      assert MixtureOfTransformers.output_size(vocab_size: 32000) == 32000
      assert MixtureOfTransformers.output_size(vocab_size: 50257) == 50257
    end
  end
end
