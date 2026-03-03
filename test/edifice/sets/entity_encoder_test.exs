defmodule Edifice.Sets.EntityEncoderTest do
  use ExUnit.Case, async: true
  @moduletag :sets

  alias Edifice.Sets.EntityEncoder

  @batch 2
  @num_entities 5
  @entity_dim 8
  @num_types 4
  @hidden 16

  defp build_and_run(opts \\ []) do
    default_opts = [
      entity_dim: @entity_dim,
      num_types: @num_types,
      hidden_size: @hidden,
      num_heads: 2,
      num_layers: 1,
      dropout: 0.0
    ]

    model = EntityEncoder.build(Keyword.merge(default_opts, opts))
    {init_fn, predict_fn} = Axon.build(model)

    features = Nx.broadcast(0.5, {@batch, @num_entities, @entity_dim})
    types = Nx.tensor([[0, 1, 2, 3, 0], [1, 2, 0, 3, 1]])

    templates = %{
      "entity_features" => Nx.template({@batch, @num_entities, @entity_dim}, :f32),
      "entity_types" => Nx.template({@batch, @num_entities}, :s64)
    }

    params = init_fn.(templates, Axon.ModelState.empty())
    output = predict_fn.(params, %{"entity_features" => features, "entity_types" => types})
    {output, params, predict_fn}
  end

  describe "build/1" do
    test "produces correct output shapes (mean pooling)" do
      {output, _, _} = build_and_run(pool_mode: :mean)

      assert Nx.shape(output.global) == {@batch, @hidden}
      assert Nx.shape(output.entities) == {@batch, @num_entities, @hidden}
    end

    test "produces correct output shapes (max pooling)" do
      {output, _, _} = build_and_run(pool_mode: :max)

      assert Nx.shape(output.global) == {@batch, @hidden}
      assert Nx.shape(output.entities) == {@batch, @num_entities, @hidden}
    end

    test "produces correct output shapes (attention pooling)" do
      {output, _, _} = build_and_run(pool_mode: :attention)

      assert Nx.shape(output.global) == {@batch, @hidden}
      assert Nx.shape(output.entities) == {@batch, @num_entities, @hidden}
    end

    test "works with multiple layers" do
      {output, _, _} = build_and_run(num_layers: 3)

      assert Nx.shape(output.global) == {@batch, @hidden}
    end

    test "different entity types produce different representations" do
      model =
        EntityEncoder.build(
          entity_dim: @entity_dim,
          num_types: @num_types,
          hidden_size: @hidden,
          num_heads: 2,
          num_layers: 1,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      # Same features, different types
      features = Nx.broadcast(1.0, {1, 3, @entity_dim})
      types_a = Nx.tensor([[0, 0, 0]])
      types_b = Nx.tensor([[1, 2, 3]])

      templates = %{
        "entity_features" => Nx.template({1, 3, @entity_dim}, :f32),
        "entity_types" => Nx.template({1, 3}, :s64)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      out_a = predict_fn.(params, %{"entity_features" => features, "entity_types" => types_a})
      out_b = predict_fn.(params, %{"entity_features" => features, "entity_types" => types_b})

      # Different types should produce different outputs
      diff = Nx.subtract(out_a.global, out_b.global) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-5
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert EntityEncoder.output_size(hidden_size: 128) == 128
    end

    test "defaults to 64" do
      assert EntityEncoder.output_size() == 64
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = EntityEncoder.recommended_defaults()
      assert Keyword.get(defaults, :entity_dim) == 16
      assert Keyword.get(defaults, :num_types) == 8
    end
  end

  describe "registry" do
    test "registered in Edifice" do
      assert Edifice.module_for(:entity_encoder) == Edifice.Sets.EntityEncoder
    end

    test "in sets family" do
      families = Edifice.list_families()
      assert :entity_encoder in families[:sets]
    end
  end
end
