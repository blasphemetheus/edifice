defmodule Edifice.Generative.MDLMTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.MDLM

  @batch 2
  @seq_len 8
  @vocab_size 32
  @hidden_size 16
  @num_heads 2
  @num_layers 2

  @small_opts [
    vocab_size: @vocab_size,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: @num_heads,
    seq_len: @seq_len,
    mlp_ratio: 2,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)

    # Token indices in [0, vocab_size)
    {tokens, key} = Nx.Random.randint(key, 0, @vocab_size, shape: {@batch, @seq_len}, type: :s64)

    # Timestep in (0, 1)
    {timestep, _key} = Nx.Random.uniform(key, shape: {@batch})

    %{
      "masked_tokens" => tokens,
      "timestep" => timestep
    }
  end

  describe "MDLM.build/1" do
    test "returns an Axon model" do
      model = MDLM.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MDLM.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "masked_tokens" => Nx.template({@batch, @seq_len}, :s64),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      # Output should be logits: [batch, seq_len, vocab_size]
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "output contains finite values" do
      model = MDLM.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "masked_tokens" => Nx.template({@batch, @seq_len}, :s64),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      output = predict_fn.(params, random_input())

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "noise schedules" do
    test "log_linear_schedule returns values in (0, 1)" do
      t = Nx.linspace(0.01, 0.99, n: 10)
      alpha = MDLM.log_linear_schedule(t)

      min_val = Nx.reduce_min(alpha) |> Nx.to_number()
      max_val = Nx.reduce_max(alpha) |> Nx.to_number()

      assert min_val > 0.0
      assert max_val < 1.0
    end

    test "log_linear_schedule is monotonically increasing" do
      t = Nx.linspace(0.01, 0.99, n: 20)
      alpha = MDLM.log_linear_schedule(t)

      values = Nx.to_flat_list(alpha)

      pairs = Enum.zip(Enum.drop(values, -1), Enum.drop(values, 1))
      assert Enum.all?(pairs, fn {a, b} -> b > a end)
    end

    test "cosine_schedule returns values in (0, 1)" do
      t = Nx.linspace(0.01, 0.99, n: 10)
      alpha = MDLM.cosine_schedule(t)

      min_val = Nx.reduce_min(alpha) |> Nx.to_number()
      max_val = Nx.reduce_max(alpha) |> Nx.to_number()

      assert min_val > 0.0
      assert max_val < 1.0
    end

    test "cosine_schedule is monotonically increasing" do
      t = Nx.linspace(0.01, 0.99, n: 20)
      alpha = MDLM.cosine_schedule(t)

      values = Nx.to_flat_list(alpha)

      pairs = Enum.zip(Enum.drop(values, -1), Enum.drop(values, 1))
      assert Enum.all?(pairs, fn {a, b} -> b > a end)
    end
  end

  describe "configuration variants" do
    test "works with different mlp_ratio" do
      opts = Keyword.put(@small_opts, :mlp_ratio, 3)
      model = MDLM.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "masked_tokens" => Nx.template({@batch, @seq_len}, :s64),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "works with dropout" do
      opts = Keyword.put(@small_opts, :dropout, 0.1)
      model = MDLM.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "masked_tokens" => Nx.template({@batch, @seq_len}, :s64),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build via registry" do
      model = Edifice.build(:mdlm, @small_opts)
      assert %Axon{} = model
    end
  end

  describe "output_size/1" do
    test "returns vocab_size" do
      assert MDLM.output_size(vocab_size: 50_257) == 50_257
      assert MDLM.output_size(vocab_size: 128) == 128
    end
  end
end
