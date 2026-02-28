defmodule Edifice.Attention.RLATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Attention.RLA

  @moduletag timeout: 120_000

  @batch 2
  @seq_len 4
  @embed_dim 16
  @hidden_size 16
  @num_heads 2
  @num_layers 2

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    window_size: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    random_tensor({@batch, @seq_len, @embed_dim})
  end

  defp build_and_run(opts) do
    model = RLA.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    params =
      init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

    output = predict_fn.(params, random_input())
    {model, output}
  end

  describe "RLA variant" do
    @rla_opts Keyword.put(@base_opts, :variant, :rla)

    test "builds an Axon model" do
      model = RLA.build(@rla_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@rla_opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@rla_opts)
      assert_finite!(output, "rla_output")
    end

    test "works with multiple layers" do
      opts = Keyword.put(@rla_opts, :num_layers, 3)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output, "rla_multi_layer")
    end

    test "works with different embed_dim than hidden_size" do
      opts = Keyword.merge(@rla_opts, embed_dim: 24)
      model = RLA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, 24}, :f32), Axon.ModelState.empty())

      input = random_tensor({@batch, @seq_len, 24}, 99)
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "RDN variant" do
    @rdn_opts Keyword.put(@base_opts, :variant, :rdn)

    test "builds an Axon model" do
      model = RLA.build(@rdn_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@rdn_opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@rdn_opts)
      assert_finite!(output, "rdn_output")
    end

    test "works with custom clip threshold" do
      opts = Keyword.put(@rdn_opts, :clip_threshold, 0.5)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output, "rdn_clipped")
    end
  end

  describe "default variant" do
    test "defaults to :rla when variant not specified" do
      {_model, output} = build_and_run(@base_opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert RLA.output_size(@base_opts) == @hidden_size
    end

    test "returns default when no opts" do
      assert RLA.output_size([]) == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = RLA.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :variant)
      assert Keyword.has_key?(defaults, :clip_threshold)
      assert Keyword.has_key?(defaults, :dropout)
      assert Keyword.has_key?(defaults, :window_size)
    end
  end
end
