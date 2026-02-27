defmodule Edifice.Blocks.TransformerBlockTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.{CrossAttention, TransformerBlock}

  @batch 2
  @seq_len 8
  @hidden 32
  @num_heads 4
  @mem_len 6

  describe "layer/3 (3-sublayer encoder-decoder block)" do
    test "produces correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      memory = Axon.input("memory", shape: {nil, @mem_len, @hidden})

      output =
        TransformerBlock.layer(input, memory,
          attention_fn: fn x, name -> self_attn(x, name) end,
          cross_attention_fn: fn q, mem, name ->
            CrossAttention.layer(q, mem,
              hidden_size: @hidden,
              num_heads: @num_heads,
              name: name
            )
          end,
          hidden_size: @hidden,
          name: "dec_block"
        )

      {init_fn, predict_fn} = Axon.build(output, mode: :inference)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "memory" => Nx.template({@batch, @mem_len, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      result =
        predict_fn.(params, %{
          "input" => Nx.broadcast(0.1, {@batch, @seq_len, @hidden}),
          "memory" => Nx.broadcast(0.2, {@batch, @mem_len, @hidden})
        })

      assert Nx.shape(result) == {@batch, @seq_len, @hidden}
    end

    test "output values are finite" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      memory = Axon.input("memory", shape: {nil, @mem_len, @hidden})

      output =
        TransformerBlock.layer(input, memory,
          attention_fn: fn x, name -> self_attn(x, name) end,
          cross_attention_fn: fn q, mem, name ->
            CrossAttention.layer(q, mem,
              hidden_size: @hidden,
              num_heads: @num_heads,
              name: name
            )
          end,
          hidden_size: @hidden,
          name: "dec_block"
        )

      {init_fn, predict_fn} = Axon.build(output, mode: :inference)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "memory" => Nx.template({@batch, @mem_len, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {inp, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      {mem, _key} = Nx.Random.normal(key, shape: {@batch, @mem_len, @hidden})

      result = predict_fn.(params, %{"input" => inp, "memory" => mem})

      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "works with custom_ffn" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      memory = Axon.input("memory", shape: {nil, @mem_len, @hidden})

      output =
        TransformerBlock.layer(input, memory,
          attention_fn: fn x, name -> self_attn(x, name) end,
          cross_attention_fn: fn q, mem, name ->
            CrossAttention.layer(q, mem,
              hidden_size: @hidden,
              num_heads: @num_heads,
              name: name
            )
          end,
          hidden_size: @hidden,
          custom_ffn: fn x, name ->
            x
            |> Axon.dense(@hidden * 4, name: "#{name}_up")
            |> Axon.activation(:relu, name: "#{name}_act")
            |> Axon.dense(@hidden, name: "#{name}_down")
          end,
          name: "dec_custom"
        )

      {init_fn, predict_fn} = Axon.build(output, mode: :inference)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "memory" => Nx.template({@batch, @mem_len, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      result =
        predict_fn.(params, %{
          "input" => Nx.broadcast(0.1, {@batch, @seq_len, @hidden}),
          "memory" => Nx.broadcast(0.2, {@batch, @mem_len, @hidden})
        })

      assert Nx.shape(result) == {@batch, @seq_len, @hidden}
    end

    test "works with rms_norm" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      memory = Axon.input("memory", shape: {nil, @mem_len, @hidden})

      output =
        TransformerBlock.layer(input, memory,
          attention_fn: fn x, name -> self_attn(x, name) end,
          cross_attention_fn: fn q, mem, name ->
            CrossAttention.layer(q, mem,
              hidden_size: @hidden,
              num_heads: @num_heads,
              name: name
            )
          end,
          hidden_size: @hidden,
          norm: :rms_norm,
          name: "dec_rms"
        )

      {init_fn, predict_fn} = Axon.build(output, mode: :inference)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "memory" => Nx.template({@batch, @mem_len, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      result =
        predict_fn.(params, %{
          "input" => Nx.broadcast(0.1, {@batch, @seq_len, @hidden}),
          "memory" => Nx.broadcast(0.2, {@batch, @mem_len, @hidden})
        })

      assert Nx.shape(result) == {@batch, @seq_len, @hidden}
    end
  end

  describe "stack/4 (3-sublayer multi-layer)" do
    test "stacks N layers with correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      memory = Axon.input("memory", shape: {nil, @mem_len, @hidden})

      output =
        TransformerBlock.stack(input, memory, 3,
          attention_fn: fn x, name -> self_attn(x, name) end,
          cross_attention_fn: fn q, mem, name ->
            CrossAttention.layer(q, mem,
              hidden_size: @hidden,
              num_heads: @num_heads,
              name: name
            )
          end,
          hidden_size: @hidden,
          name: "dec"
        )

      {init_fn, predict_fn} = Axon.build(output, mode: :inference)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "memory" => Nx.template({@batch, @mem_len, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      result =
        predict_fn.(params, %{
          "input" => Nx.broadcast(0.1, {@batch, @seq_len, @hidden}),
          "memory" => Nx.broadcast(0.2, {@batch, @mem_len, @hidden})
        })

      assert Nx.shape(result) == {@batch, @seq_len, @hidden}
    end

    test "single layer stack matches layer/3" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      memory = Axon.input("memory", shape: {nil, @mem_len, @hidden})

      opts = [
        attention_fn: fn x, name -> self_attn(x, name) end,
        cross_attention_fn: fn q, mem, name ->
          CrossAttention.layer(q, mem,
            hidden_size: @hidden,
            num_heads: @num_heads,
            name: name
          )
        end,
        hidden_size: @hidden,
        name: "test"
      ]

      single = TransformerBlock.layer(input, memory, Keyword.put(opts, :name, "test_block_1"))
      stacked = TransformerBlock.stack(input, memory, 1, opts)

      # Both should build successfully and produce same shape
      {init1, pred1} = Axon.build(single, mode: :inference)
      {init2, pred2} = Axon.build(stacked, mode: :inference)

      inp_data = %{
        "input" => Nx.broadcast(0.5, {@batch, @seq_len, @hidden}),
        "memory" => Nx.broadcast(0.5, {@batch, @mem_len, @hidden})
      }

      templates = %{
        "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
        "memory" => Nx.template({@batch, @mem_len, @hidden}, :f32)
      }

      p1 = init1.(templates, Axon.ModelState.empty())
      p2 = init2.(templates, Axon.ModelState.empty())

      r1 = pred1.(p1, inp_data)
      r2 = pred2.(p2, inp_data)

      assert Nx.shape(r1) == Nx.shape(r2)
    end
  end

  # Simple self-attention helper for tests
  defp self_attn(input, name) do
    q = Axon.dense(input, @hidden, name: "#{name}_q")
    k = Axon.dense(input, @hidden, name: "#{name}_k")
    v = Axon.dense(input, @hidden, name: "#{name}_v")

    head_dim = div(@hidden, @num_heads)

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          {batch, q_len, _} = Nx.shape(q_t)
          {_, kv_len, _} = Nx.shape(k_t)

          q_r =
            q_t
            |> Nx.reshape({batch, q_len, @num_heads, head_dim})
            |> Nx.transpose(axes: [0, 2, 1, 3])

          k_r =
            k_t
            |> Nx.reshape({batch, kv_len, @num_heads, head_dim})
            |> Nx.transpose(axes: [0, 2, 1, 3])

          v_r =
            v_t
            |> Nx.reshape({batch, kv_len, @num_heads, head_dim})
            |> Nx.transpose(axes: [0, 2, 1, 3])

          scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q_r)))
          scores = Nx.divide(Nx.dot(q_r, [3], [0, 1], k_r, [3], [0, 1]), scale)

          max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
          exp_s = Nx.exp(Nx.subtract(scores, max_s))
          weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-9))

          output = Nx.dot(weights, [3], [0, 1], v_r, [2], [0, 1])

          output
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, q_len, @num_heads * head_dim})
        end,
        [q, k, v],
        name: "#{name}_compute",
        op_name: :self_attention
      )

    Axon.dense(attended, @hidden, name: "#{name}_out")
  end
end
