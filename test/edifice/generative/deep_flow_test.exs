defmodule Edifice.Generative.DeepFlowTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.DeepFlow

  @batch 2
  @input_size 8
  @patch_size 2
  @in_channels 4
  @hidden_size 32
  @num_heads 4
  @num_layers 4
  @num_branches 2

  @num_patches div(@input_size, @patch_size) * div(@input_size, @patch_size)
  @patch_dim @patch_size * @patch_size * @in_channels

  @small_opts [
    input_size: @input_size,
    patch_size: @patch_size,
    in_channels: @in_channels,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: @num_heads,
    num_branches: @num_branches,
    mlp_ratio: 2.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {latent, key} = Nx.Random.normal(key, shape: {@batch, @num_patches, @patch_dim})
    {timestep, _key} = Nx.Random.uniform(key, shape: {@batch})

    %{
      "noisy_latent" => latent,
      "timestep" => timestep
    }
  end

  defp build_and_run(opts) do
    model = DeepFlow.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "noisy_latent" => Nx.template({@batch, @num_patches, @patch_dim}, :f32),
      "timestep" => Nx.template({@batch}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_input())
    {model, output}
  end

  describe "DeepFlow.build/1" do
    test "returns an Axon container" do
      model = DeepFlow.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces container with velocity and branch_velocities" do
      {_model, output} = build_and_run(@small_opts)
      assert is_map(output)
      assert Map.has_key?(output, :velocity)
      assert Map.has_key?(output, :branch_velocities)
    end

    test "final velocity has correct shape" do
      {_model, output} = build_and_run(@small_opts)
      assert Nx.shape(output.velocity) == {@batch, @num_patches, @patch_dim}
    end

    test "branch_velocities is a tuple with one entry per branch" do
      {_model, output} = build_and_run(@small_opts)
      assert is_tuple(output.branch_velocities)
      assert tuple_size(output.branch_velocities) == @num_branches

      Enum.each(Tuple.to_list(output.branch_velocities), fn vel ->
        assert Nx.shape(vel) == {@batch, @num_patches, @patch_dim}
      end)
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@small_opts)
      refute Nx.any(Nx.is_nan(output.velocity)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output.velocity)) |> Nx.to_number() == 1
    end
  end

  describe "configuration variants" do
    test "works with 3 branches" do
      opts = Keyword.merge(@small_opts, num_branches: 3, num_layers: 6)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output.velocity) == {@batch, @num_patches, @patch_dim}

      assert tuple_size(output.branch_velocities) == 3
    end

    test "works with class conditioning" do
      opts = Keyword.put(@small_opts, :num_classes, 10)
      model = DeepFlow.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_latent" => Nx.template({@batch, @num_patches, @patch_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32),
        "class_label" => Nx.template({@batch}, :s64)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = Map.put(random_input(), "class_label", Nx.tensor([3, 7]))
      output = predict_fn.(params, input)
      assert Nx.shape(output.velocity) == {@batch, @num_patches, @patch_dim}
    end
  end

  describe "output_size/1" do
    test "returns total output dimension" do
      # 32/2 * 32/2 * (2*2*4) = 16*16*16 = 4096
      assert DeepFlow.output_size(input_size: 32, patch_size: 2, in_channels: 4) == 4096
    end
  end
end
