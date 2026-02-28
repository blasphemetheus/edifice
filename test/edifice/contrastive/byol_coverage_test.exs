defmodule Edifice.Contrastive.BYOLCoverageTest do
  @moduledoc """
  Coverage tests for Edifice.Contrastive.BYOL.
  Covers option variations, EMA update branches, loss edge cases,
  and standalone build functions not exercised by existing tests.
  """
  use ExUnit.Case, async: true
  @moduletag :contrastive

  alias Edifice.Contrastive.BYOL

  @batch 2
  @encoder_dim 16
  @projection_dim 8
  @predictor_dim 4
  @hidden_size 16

  # ============================================================================
  # build_online/1 standalone
  # ============================================================================

  describe "build_online/1" do
    test "builds online network with custom dims" do
      online =
        BYOL.build_online(
          encoder_dim: @encoder_dim,
          projection_dim: @projection_dim,
          predictor_dim: @predictor_dim,
          hidden_size: @hidden_size
        )

      assert %Axon{} = online

      {init_fn, predict_fn} = Axon.build(online)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(200)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @encoder_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @projection_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "online has predictor layers" do
      online =
        BYOL.build_online(
          encoder_dim: @encoder_dim,
          projection_dim: @projection_dim,
          predictor_dim: @predictor_dim,
          hidden_size: @hidden_size
        )

      {init_fn, _} = Axon.build(online)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "predictor_fc1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "predictor_fc2"))
    end

    @tag :slow
    test "online uses default projection_dim and predictor_dim" do
      online = BYOL.build_online(encoder_dim: @encoder_dim)

      {init_fn, predict_fn} = Axon.build(online)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(201)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @encoder_dim})
      output = predict_fn.(params, input)

      # Default projection_dim is 256
      assert Nx.shape(output) == {@batch, 256}
    end
  end

  # ============================================================================
  # build_target/1 standalone
  # ============================================================================

  describe "build_target/1" do
    test "builds target network without predictor" do
      target =
        BYOL.build_target(
          encoder_dim: @encoder_dim,
          projection_dim: @projection_dim,
          hidden_size: @hidden_size
        )

      assert %Axon{} = target

      {init_fn, predict_fn} = Axon.build(target)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      # Target should NOT have predictor layers
      param_keys = Map.keys(params.data)
      refute Enum.any?(param_keys, &String.contains?(&1, "predictor"))

      key = Nx.Random.key(202)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @encoder_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @projection_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    @tag :slow
    test "target uses default projection_dim" do
      target = BYOL.build_target(encoder_dim: @encoder_dim)

      {init_fn, predict_fn} = Axon.build(target)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(203)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @encoder_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 256}
    end
  end

  # ============================================================================
  # EMA Update variations
  # ============================================================================

  describe "ema_update/3" do
    test "momentum=0.0 copies online params entirely" do
      online_params = %{
        "online_encoder_fc1" => 10.0,
        "online_proj_fc1" => 20.0
      }

      target_params = %{
        "target_encoder_fc1" => 0.0,
        "target_proj_fc1" => 0.0
      }

      updated = BYOL.ema_update(online_params, target_params, momentum: 0.0)

      # EMA: 0.0 * target + 1.0 * online = online
      assert_in_delta Nx.to_number(updated["target_encoder_fc1"]), 10.0, 0.01
      assert_in_delta Nx.to_number(updated["target_proj_fc1"]), 20.0, 0.01
    end

    test "momentum=1.0 keeps target params unchanged" do
      online_params = %{
        "online_encoder_fc1" => 10.0,
        "online_proj_fc1" => 20.0
      }

      target_params = %{
        "target_encoder_fc1" => 5.0,
        "target_proj_fc1" => 15.0
      }

      updated = BYOL.ema_update(online_params, target_params, momentum: 1.0)

      # EMA: 1.0 * target + 0.0 * online = target
      assert_in_delta Nx.to_number(updated["target_encoder_fc1"]), 5.0, 0.01
      assert_in_delta Nx.to_number(updated["target_proj_fc1"]), 15.0, 0.01
    end

    test "default momentum is 0.996" do
      online_params = %{
        "online_encoder_fc1" => 100.0
      }

      target_params = %{
        "target_encoder_fc1" => 0.0
      }

      updated = BYOL.ema_update(online_params, target_params)

      # EMA: 0.996 * 0 + 0.004 * 100 = 0.4
      assert_in_delta Nx.to_number(updated["target_encoder_fc1"]), 0.4, 0.01
    end

    test "handles unmatched target keys (error branch)" do
      # Target has a key that doesn't have a matching online key
      online_params = %{
        "online_encoder_fc1" => 10.0
      }

      target_params = %{
        "target_encoder_fc1" => 5.0,
        "target_extra_layer" => 99.0
      }

      updated = BYOL.ema_update(online_params, target_params, momentum: 0.5)

      # target_encoder_fc1 should be blended
      assert_in_delta Nx.to_number(updated["target_encoder_fc1"]), 7.5, 0.01

      # target_extra_layer has no matching online key -> remains unchanged
      assert_in_delta Nx.to_number(updated["target_extra_layer"]), 99.0, 0.01
    end

    test "handles nested map params via ema_blend recursion" do
      online_params = %{
        "online_encoder_fc1" => %{"kernel" => 10.0, "bias" => 2.0}
      }

      target_params = %{
        "target_encoder_fc1" => %{"kernel" => 0.0, "bias" => 0.0}
      }

      updated = BYOL.ema_update(online_params, target_params, momentum: 0.5)

      inner = updated["target_encoder_fc1"]
      assert is_map(inner)
      assert_in_delta Nx.to_number(inner["kernel"]), 5.0, 0.01
      assert_in_delta Nx.to_number(inner["bias"]), 1.0, 0.01
    end

    test "ema_blend with nested map where inner key doesn't match" do
      online_params = %{
        "online_encoder_fc1" => %{"kernel" => 10.0}
      }

      target_params = %{
        "target_encoder_fc1" => %{"kernel" => 0.0, "bias" => 99.0}
      }

      updated = BYOL.ema_update(online_params, target_params, momentum: 0.5)

      inner = updated["target_encoder_fc1"]
      assert_in_delta Nx.to_number(inner["kernel"]), 5.0, 0.01
      # bias has no matching online key -> stays same
      assert_in_delta Nx.to_number(inner["bias"]), 99.0, 0.01
    end
  end

  # ============================================================================
  # Loss function edge cases
  # ============================================================================

  describe "loss/2" do
    test "loss of identical inputs is close to zero" do
      key = Nx.Random.key(210)
      {pred, _} = Nx.Random.uniform(key, shape: {@batch, @projection_dim})

      loss = BYOL.loss(pred, pred)

      assert Nx.shape(loss) == {}
      # Same input normalized -> same direction -> cosine similarity = 1 -> loss ~ 0
      assert Nx.to_number(loss) < 0.01
    end

    test "loss of orthogonal inputs is close to 2" do
      # Create two vectors that are roughly orthogonal
      pred =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

      proj =
        Nx.tensor([
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        ])

      loss = BYOL.loss(pred, proj)

      # Orthogonal -> cosine = 0 -> loss = 2
      assert_in_delta Nx.to_number(loss), 2.0, 0.01
    end

    test "loss of opposite inputs is close to 4" do
      pred =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

      proj =
        Nx.tensor([
          [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

      loss = BYOL.loss(pred, proj)

      # Opposite -> cosine = -1 -> loss = 2 - 2*(-1) = 4
      assert_in_delta Nx.to_number(loss), 4.0, 0.01
    end

    test "loss with random inputs is bounded" do
      key = Nx.Random.key(211)
      {pred, key} = Nx.Random.normal(key, shape: {8, @projection_dim})
      {proj, _} = Nx.Random.normal(key, shape: {8, @projection_dim})

      loss = BYOL.loss(pred, proj)

      val = Nx.to_number(loss)
      assert val >= 0.0
      assert val <= 4.0
    end
  end

  # ============================================================================
  # Default accessor functions
  # ============================================================================

  describe "default functions" do
    test "default_projection_dim returns 256" do
      assert BYOL.default_projection_dim() == 256
    end

    test "default_predictor_dim returns 64" do
      assert BYOL.default_predictor_dim() == 64
    end

    test "default_momentum returns 0.996" do
      assert BYOL.default_momentum() == 0.996
    end

    test "default_hidden_size returns 256" do
      assert BYOL.default_hidden_size() == 256
    end
  end

  # ============================================================================
  # Output size
  # ============================================================================

  describe "output_size/1" do
    test "returns custom projection_dim" do
      assert BYOL.output_size(projection_dim: 128) == 128
    end

    test "returns default when no opts" do
      assert BYOL.output_size() == 256
    end

    test "ignores unrelated opts" do
      assert BYOL.output_size(encoder_dim: 999, projection_dim: 32) == 32
    end
  end

  # ============================================================================
  # Different hidden_size and predictor_dim combos
  # ============================================================================

  describe "dimension variations" do
    test "large predictor_dim" do
      {online, target} =
        BYOL.build(
          encoder_dim: @encoder_dim,
          projection_dim: @projection_dim,
          predictor_dim: 16,
          hidden_size: @hidden_size
        )

      {init_fn, predict_fn} = Axon.build(online)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(220)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @encoder_dim})
      online_out = predict_fn.(params, input)

      assert Nx.shape(online_out) == {@batch, @projection_dim}

      {init_fn_t, predict_fn_t} = Axon.build(target)
      params_t = init_fn_t.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())
      target_out = predict_fn_t.(params_t, input)

      assert Nx.shape(target_out) == {@batch, @projection_dim}
    end

    test "small hidden_size=8" do
      {online, _target} =
        BYOL.build(
          encoder_dim: @encoder_dim,
          projection_dim: @projection_dim,
          predictor_dim: @predictor_dim,
          hidden_size: 8
        )

      {init_fn, predict_fn} = Axon.build(online)
      params = init_fn.(Nx.template({@batch, @encoder_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(221)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @encoder_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @projection_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
