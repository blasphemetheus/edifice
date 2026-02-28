defmodule Edifice.Generative.CaDDiTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.CaDDi

  @batch 2
  @seq_len 8
  @vocab_size 32
  @hidden_size 32
  @num_heads 4
  @num_layers 2
  @num_diffusion_steps 16
  @context_window 4
  @total_len @context_window * @seq_len

  @small_opts [
    vocab_size: @vocab_size,
    seq_len: @seq_len,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: @num_heads,
    num_diffusion_steps: @num_diffusion_steps,
    context_window: @context_window,
    intermediate_size: 64
  ]

  defp random_input(total_len) do
    key = Nx.Random.key(42)

    {tokens, _key} =
      Nx.Random.randint(key, 0, @vocab_size, shape: {@batch, total_len}, type: :s64)

    %{"trajectory" => tokens}
  end

  defp build_and_run(opts) do
    model = CaDDi.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    effective_steps =
      min(
        Keyword.get(opts, :num_diffusion_steps, @num_diffusion_steps),
        Keyword.get(opts, :context_window, @context_window)
      )

    total = effective_steps * Keyword.fetch!(opts, :seq_len)

    template = %{"trajectory" => Nx.template({@batch, total}, :s64)}
    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_input(total))
    {model, output, total}
  end

  describe "CaDDi.build/1 block variant" do
    test "returns an Axon model" do
      model = CaDDi.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output, _total} = build_and_run(@small_opts)
      assert Nx.shape(output) == {@batch, @total_len, @vocab_size}
    end

    test "output contains finite values" do
      {_model, output, _total} = build_and_run(@small_opts)
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "CaDDi.build/1 AR variant" do
    test "forward pass produces correct shape" do
      opts = Keyword.put(@small_opts, :variant, :ar)
      {_model, output, _total} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @total_len, @vocab_size}
    end

    test "output contains finite values" do
      opts = Keyword.put(@small_opts, :variant, :ar)
      {_model, output, _total} = build_and_run(opts)
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "context window" do
    test "clamps to min(T, context_window)" do
      # T=3, window=4 -> effective=3, total=24
      opts = Keyword.merge(@small_opts, num_diffusion_steps: 3, context_window: 4)
      {_model, output, total} = build_and_run(opts)
      assert total == 3 * @seq_len
      assert Nx.shape(output) == {@batch, total, @vocab_size}
    end
  end

  describe "output_size/1" do
    test "returns vocab_size" do
      assert CaDDi.output_size(vocab_size: 50_257) == 50_257
    end
  end
end
