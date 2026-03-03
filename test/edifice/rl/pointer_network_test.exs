defmodule Edifice.RL.PointerNetworkTest do
  use ExUnit.Case, async: true
  @moduletag :rl

  alias Edifice.RL.PointerNetwork

  @batch 2
  @hidden_size 16
  @entity_dim 8
  @num_entities 5
  @key_dim 8

  describe "build/1 (no mask)" do
    test "produces correct output shape" do
      model =
        PointerNetwork.build(
          hidden_size: @hidden_size,
          entity_dim: @entity_dim,
          key_dim: @key_dim
        )

      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{
        "query" => Nx.broadcast(0.5, {@batch, @hidden_size}),
        "keys" => Nx.broadcast(0.5, {@batch, @num_entities, @entity_dim})
      }

      templates =
        Map.new(inputs, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

      params = init_fn.(templates, Axon.ModelState.empty())
      output = predict_fn.(params, inputs)

      assert Nx.shape(output) == {@batch, @num_entities}
    end

    test "different queries produce different selections" do
      model =
        PointerNetwork.build(
          hidden_size: @hidden_size,
          entity_dim: @entity_dim,
          key_dim: @key_dim
        )

      {init_fn, predict_fn} = Axon.build(model)

      keys = Nx.broadcast(0.5, {@batch, @num_entities, @entity_dim})

      inputs_a = %{
        "query" => Nx.broadcast(1.0, {@batch, @hidden_size}),
        "keys" => keys
      }

      inputs_b = %{
        "query" => Nx.broadcast(-1.0, {@batch, @hidden_size}),
        "keys" => keys
      }

      templates = %{
        "query" => Nx.template({@batch, @hidden_size}, :f32),
        "keys" => Nx.template({@batch, @num_entities, @entity_dim}, :f32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      out_a = predict_fn.(params, inputs_a)
      out_b = predict_fn.(params, inputs_b)

      diff = Nx.subtract(out_a, out_b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-5
    end
  end

  describe "build/1 (with mask)" do
    test "masked entities get large negative logits" do
      model =
        PointerNetwork.build(
          hidden_size: @hidden_size,
          entity_dim: @entity_dim,
          key_dim: @key_dim,
          use_mask: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      # Mask: only first 3 entities valid
      mask = Nx.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], type: :f32)

      inputs = %{
        "query" => Nx.broadcast(0.5, {@batch, @hidden_size}),
        "keys" => Nx.broadcast(0.5, {@batch, @num_entities, @entity_dim}),
        "mask" => mask
      }

      templates =
        Map.new(inputs, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

      params = init_fn.(templates, Axon.ModelState.empty())
      output = predict_fn.(params, inputs)

      assert Nx.shape(output) == {@batch, @num_entities}

      # Masked positions should have very negative logits
      # Check batch 0, entity 3 (masked) vs entity 0 (valid)
      logit_valid = output |> Nx.slice([0, 0], [1, 1]) |> Nx.squeeze() |> Nx.to_number()
      logit_masked = output |> Nx.slice([0, 3], [1, 1]) |> Nx.squeeze() |> Nx.to_number()

      assert logit_masked < logit_valid - 1.0e6
    end
  end

  describe "works with different entity counts" do
    test "3 entities" do
      model =
        PointerNetwork.build(
          hidden_size: @hidden_size,
          entity_dim: @entity_dim,
          key_dim: @key_dim
        )

      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{
        "query" => Nx.broadcast(0.5, {1, @hidden_size}),
        "keys" => Nx.broadcast(0.5, {1, 3, @entity_dim})
      }

      templates = %{
        "query" => Nx.template({1, @hidden_size}, :f32),
        "keys" => Nx.template({1, 3, @entity_dim}, :f32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())
      output = predict_fn.(params, inputs)

      assert Nx.shape(output) == {1, 3}
    end
  end

  describe "output_size/1" do
    test "returns :dynamic" do
      assert PointerNetwork.output_size() == :dynamic
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = PointerNetwork.recommended_defaults()
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :use_mask) == true
    end
  end

  describe "registry" do
    test "registered in Edifice" do
      assert Edifice.module_for(:pointer_network) == Edifice.RL.PointerNetwork
    end

    test "in rl family" do
      families = Edifice.list_families()
      assert :pointer_network in families[:rl]
    end
  end
end
