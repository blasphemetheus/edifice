defmodule Edifice.RL.AutoregressiveHeadTest do
  use ExUnit.Case, async: true
  @moduletag :rl

  alias Edifice.RL.AutoregressiveHead

  @batch 2
  @hidden_size 16
  @components [
    %{name: "buttons", num_actions: 4, embed_dim: 8},
    %{name: "stick_x", num_actions: 5, embed_dim: 8},
    %{name: "stick_y", num_actions: 5, embed_dim: 8}
  ]

  describe "build/1 (training with teacher forcing)" do
    test "produces correct output shapes" do
      model =
        AutoregressiveHead.build(
          hidden_size: @hidden_size,
          components: @components,
          component_hidden: 16,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{
        "hidden" => Nx.broadcast(0.5, {@batch, @hidden_size}),
        "action_buttons" => Nx.tensor([0, 2]),
        "action_stick_x" => Nx.tensor([3, 1]),
        "action_stick_y" => Nx.tensor([4, 0])
      }

      templates =
        Map.new(inputs, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

      params = init_fn.(templates, Axon.ModelState.empty())
      output = predict_fn.(params, inputs)

      assert Nx.shape(output.buttons) == {@batch, 4}
      assert Nx.shape(output.stick_x) == {@batch, 5}
      assert Nx.shape(output.stick_y) == {@batch, 5}
    end

    test "later components depend on earlier actions" do
      model =
        AutoregressiveHead.build(
          hidden_size: @hidden_size,
          components: @components,
          component_hidden: 16,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      hidden = Nx.broadcast(0.5, {@batch, @hidden_size})

      # Same hidden, different previous actions
      inputs_a = %{
        "hidden" => hidden,
        "action_buttons" => Nx.tensor([0, 0]),
        "action_stick_x" => Nx.tensor([0, 0]),
        "action_stick_y" => Nx.tensor([0, 0])
      }

      inputs_b = %{
        "hidden" => hidden,
        "action_buttons" => Nx.tensor([3, 3]),
        "action_stick_x" => Nx.tensor([4, 4]),
        "action_stick_y" => Nx.tensor([4, 4])
      }

      templates =
        Map.new(inputs_a, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

      params = init_fn.(templates, Axon.ModelState.empty())

      out_a = predict_fn.(params, inputs_a)
      out_b = predict_fn.(params, inputs_b)

      # First component (buttons) should be the same — only conditioned on hidden
      diff_buttons =
        Nx.subtract(out_a.buttons, out_b.buttons)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      assert diff_buttons < 1.0e-5

      # Later components should differ — conditioned on different previous actions
      diff_stick_x =
        Nx.subtract(out_a.stick_x, out_b.stick_x)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      assert diff_stick_x > 1.0e-5
    end
  end

  describe "build_inference/1" do
    test "produces logits and actions" do
      model =
        AutoregressiveHead.build_inference(
          hidden_size: @hidden_size,
          components: @components,
          component_hidden: 16,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{"hidden" => Nx.broadcast(0.5, {@batch, @hidden_size})}
      templates = %{"hidden" => Nx.template({@batch, @hidden_size}, :f32)}

      params = init_fn.(templates, Axon.ModelState.empty())
      output = predict_fn.(params, inputs)

      # Has both logits and actions
      assert Nx.shape(output.logits.buttons) == {@batch, 4}
      assert Nx.shape(output.logits.stick_x) == {@batch, 5}
      assert Nx.shape(output.actions.buttons) == {@batch}
      assert Nx.shape(output.actions.stick_x) == {@batch}
    end
  end

  describe "total_actions/1" do
    test "sums across components" do
      assert AutoregressiveHead.total_actions(components: @components) == 14
    end
  end

  describe "melee_defaults/0" do
    test "returns 6 components" do
      defaults = AutoregressiveHead.melee_defaults()
      assert length(Keyword.get(defaults, :components)) == 6
    end
  end

  describe "registry" do
    test "registered in Edifice" do
      assert Edifice.module_for(:autoregressive_head) == Edifice.RL.AutoregressiveHead
    end

    test "in rl family" do
      families = Edifice.list_families()
      assert :autoregressive_head in families[:rl]
    end
  end
end
