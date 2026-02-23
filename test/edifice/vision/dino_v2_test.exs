defmodule Edifice.Vision.DINOv2Test do
  use ExUnit.Case, async: true

  alias Edifice.Vision.DINOv2

  @batch 2
  @channels 3
  @image_size 32
  @patch_size 8
  @embed_dim 64
  @num_heads 4
  @num_layers 2
  @num_register_tokens 2
  @head_output_dim 128

  @base_opts [
    image_size: @image_size,
    patch_size: @patch_size,
    in_channels: @channels,
    embed_dim: @embed_dim,
    num_heads: @num_heads,
    num_layers: @num_layers,
    num_register_tokens: @num_register_tokens,
    head_hidden_dim: 128,
    head_bottleneck_dim: 32,
    head_output_dim: @head_output_dim
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  defp build_and_predict_student(opts) do
    {student, _teacher} = DINOv2.build(opts)
    {init_fn, predict_fn} = Axon.build(student)

    params =
      init_fn.(
        Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
        Axon.ModelState.empty()
      )

    predict_fn.(params, random_image())
  end

  describe "build/1" do
    test "returns student and teacher models" do
      {student, teacher} = DINOv2.build(@base_opts)
      assert %Axon{} = student
      assert %Axon{} = teacher
    end

    test "student produces correct output shape" do
      output = build_and_predict_student(@base_opts)
      assert Nx.shape(output) == {@batch, @head_output_dim}
    end

    test "output contains finite values" do
      output = build_and_predict_student(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works without register tokens" do
      opts = Keyword.put(@base_opts, :num_register_tokens, 0)
      output = build_and_predict_student(opts)
      assert Nx.shape(output) == {@batch, @head_output_dim}
    end
  end

  describe "build_backbone/1" do
    test "builds single backbone without head" do
      opts = Keyword.merge(@base_opts, prefix: "test", include_head: false)
      model = DINOv2.build_backbone(opts)
      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())
      assert Nx.shape(output) == {@batch, @embed_dim}
    end
  end

  describe "dino_loss/3" do
    test "computes loss between student and teacher outputs" do
      key = Nx.Random.key(123)
      {student_out, key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})

      loss = DINOv2.dino_loss(student_out, teacher_out)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "loss is finite" do
      key = Nx.Random.key(456)
      {student_out, key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})

      loss = DINOv2.dino_loss(student_out, teacher_out)
      refute Nx.to_number(Nx.is_nan(loss)) == 1
      refute Nx.to_number(Nx.is_infinity(loss)) == 1
    end

    test "accepts custom temperatures" do
      key = Nx.Random.key(789)
      {student_out, key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})

      loss = DINOv2.dino_loss(student_out, teacher_out, student_temp: 0.2, teacher_temp: 0.05)
      assert Nx.shape(loss) == {}
    end
  end

  describe "update_center/3" do
    test "updates center with EMA" do
      key = Nx.Random.key(111)
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @head_output_dim})
      center = Nx.broadcast(0.0, {@head_output_dim})

      new_center = DINOv2.update_center(teacher_out, center, momentum: 0.9)
      assert Nx.shape(new_center) == {@head_output_dim}
    end
  end

  describe "koleo_loss/1" do
    test "computes KoLeo regularization" do
      key = Nx.Random.key(222)
      num_patches = div(@image_size, @patch_size) * div(@image_size, @patch_size)
      {patch_tokens, _key} = Nx.Random.normal(key, shape: {@batch, num_patches, @embed_dim})

      loss = DINOv2.koleo_loss(patch_tokens)
      assert Nx.shape(loss) == {}
    end
  end

  describe "update_teacher/3" do
    test "updates teacher params via EMA" do
      # Create simple param maps
      student_params = %{
        "student_fc" => %{"kernel" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]])}
      }

      teacher_params = %{
        "teacher_fc" => %{"kernel" => Nx.tensor([[0.0, 0.0], [0.0, 0.0]])}
      }

      updated = DINOv2.update_teacher(student_params, teacher_params, momentum: 0.5)
      expected = Nx.tensor([[0.5, 1.0], [1.5, 2.0]])

      assert_all_close(updated["teacher_fc"]["kernel"], expected)
    end
  end

  describe "output_size/1" do
    test "returns head_output_dim" do
      assert DINOv2.output_size(@base_opts) == @head_output_dim
    end
  end

  describe "recommended_defaults/1" do
    test "returns defaults for small size" do
      defaults = DINOv2.recommended_defaults(:small)
      assert defaults[:embed_dim] == 384
      assert defaults[:num_heads] == 6
    end

    test "returns defaults for base size" do
      defaults = DINOv2.recommended_defaults(:base)
      assert defaults[:embed_dim] == 768
      assert defaults[:num_heads] == 12
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all(Nx.less_equal(Nx.abs(Nx.subtract(a, b)), atol)) |> Nx.to_number() == 1
  end
end
