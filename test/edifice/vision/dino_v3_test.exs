defmodule Edifice.Vision.DINOv3Test do
  use ExUnit.Case, async: true

  alias Edifice.Vision.DINOv3

  # Small config for fast tests — head_dim must be divisible by 4 for axial RoPE
  @batch 2
  @channels 3
  @image_size 32
  @patch_size 8
  @embed_dim 64
  @num_heads 4
  @num_layers 2
  @num_register_tokens 2
  @dino_num_prototypes 32
  @ibot_num_prototypes 16
  @num_patches div(@image_size, @patch_size) * div(@image_size, @patch_size)

  @base_opts [
    image_size: @image_size,
    patch_size: @patch_size,
    in_channels: @channels,
    embed_dim: @embed_dim,
    num_heads: @num_heads,
    num_layers: @num_layers,
    num_register_tokens: @num_register_tokens,
    dino_hidden_dim: 64,
    dino_bottleneck_dim: 32,
    dino_num_prototypes: @dino_num_prototypes,
    ibot_hidden_dim: 64,
    ibot_bottleneck_dim: 32,
    ibot_num_prototypes: @ibot_num_prototypes
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  defp build_and_predict_student(opts) do
    {student, _teacher} = DINOv3.build(opts)
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
      {student, teacher} = DINOv3.build(@base_opts)
      assert %Axon{} = student
      assert %Axon{} = teacher
    end

    test "student produces correct output shapes (container)" do
      output = build_and_predict_student(@base_opts)

      assert %{dino: dino, ibot: ibot} = output
      assert Nx.shape(dino) == {@batch, @dino_num_prototypes}
      assert Nx.shape(ibot) == {@batch, @num_patches, @ibot_num_prototypes}
    end

    test "output contains finite values" do
      %{dino: dino, ibot: ibot} = build_and_predict_student(@base_opts)

      assert Nx.all(Nx.is_nan(dino) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(ibot) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works without register tokens" do
      opts = Keyword.put(@base_opts, :num_register_tokens, 0)
      %{dino: dino, ibot: ibot} = build_and_predict_student(opts)

      assert Nx.shape(dino) == {@batch, @dino_num_prototypes}
      assert Nx.shape(ibot) == {@batch, @num_patches, @ibot_num_prototypes}
    end

    test "works with SwiGLU FFN" do
      opts = Keyword.put(@base_opts, :ffn_type, :swiglu)
      %{dino: dino} = build_and_predict_student(opts)

      assert Nx.shape(dino) == {@batch, @dino_num_prototypes}
    end
  end

  describe "build_backbone/1 without head" do
    test "returns CLS features" do
      opts = Keyword.merge(@base_opts, prefix: "test", include_head: false)
      model = DINOv3.build_backbone(opts)
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
      {student_out, key} = Nx.Random.normal(key, shape: {@batch, @dino_num_prototypes})
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @dino_num_prototypes})

      loss = DINOv3.dino_loss(student_out, teacher_out)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "loss is finite" do
      key = Nx.Random.key(456)
      {student_out, key} = Nx.Random.normal(key, shape: {@batch, @dino_num_prototypes})
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @dino_num_prototypes})

      loss = DINOv3.dino_loss(student_out, teacher_out)
      refute Nx.to_number(Nx.is_nan(loss)) == 1
      refute Nx.to_number(Nx.is_infinity(loss)) == 1
    end

    test "accepts custom temperatures" do
      key = Nx.Random.key(789)
      {student_out, key} = Nx.Random.normal(key, shape: {@batch, @dino_num_prototypes})
      {teacher_out, _key} = Nx.Random.normal(key, shape: {@batch, @dino_num_prototypes})

      loss = DINOv3.dino_loss(student_out, teacher_out, student_temp: 0.2, teacher_temp: 0.05)
      assert Nx.shape(loss) == {}
    end
  end

  describe "ibot_loss/4" do
    test "computes patch-level loss with mask" do
      key = Nx.Random.key(100)
      {s_patch, key} = Nx.Random.normal(key, shape: {@batch, @num_patches, @ibot_num_prototypes})
      {t_patch, key} = Nx.Random.normal(key, shape: {@batch, @num_patches, @ibot_num_prototypes})
      # Mask half the patches
      {mask_vals, _key} = Nx.Random.uniform(key, shape: {@batch, @num_patches})
      mask = Nx.greater(mask_vals, 0.5)

      loss = DINOv3.ibot_loss(s_patch, t_patch, mask)
      assert Nx.shape(loss) == {}
      refute Nx.to_number(Nx.is_nan(loss)) == 1
    end

    test "loss is zero when no patches are masked" do
      key = Nx.Random.key(200)
      {s_patch, key} = Nx.Random.normal(key, shape: {@batch, @num_patches, @ibot_num_prototypes})
      {t_patch, _key} = Nx.Random.normal(key, shape: {@batch, @num_patches, @ibot_num_prototypes})
      mask = Nx.broadcast(Nx.tensor(0, type: :u8), {@batch, @num_patches})

      loss = DINOv3.ibot_loss(s_patch, t_patch, mask)
      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-5
    end
  end

  describe "gram_anchoring_loss/2" do
    test "returns zero for identical prototypes" do
      protos = Nx.iota({8, 4}, type: :f32) |> Nx.add(1.0)
      loss = DINOv3.gram_anchoring_loss(protos, protos)
      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-5
    end

    test "returns positive loss for different prototypes" do
      key = Nx.Random.key(300)
      {p1, key} = Nx.Random.normal(key, shape: {8, 4})
      {p2, _key} = Nx.Random.normal(key, shape: {8, 4})

      loss = DINOv3.gram_anchoring_loss(p1, p2)
      assert Nx.to_number(loss) > 0
    end
  end

  describe "koleo_loss/1" do
    test "computes KoLeo regularization" do
      key = Nx.Random.key(400)
      {patch_tokens, _key} = Nx.Random.normal(key, shape: {@batch, @num_patches, @embed_dim})

      loss = DINOv3.koleo_loss(patch_tokens)
      assert Nx.shape(loss) == {}
      refute Nx.to_number(Nx.is_nan(loss)) == 1
    end
  end

  describe "update_teacher/3" do
    test "updates teacher params via EMA" do
      student_params = %{
        "student_fc" => %{"kernel" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]])}
      }

      teacher_params = %{
        "teacher_fc" => %{"kernel" => Nx.tensor([[0.0, 0.0], [0.0, 0.0]])}
      }

      updated = DINOv3.update_teacher(student_params, teacher_params, momentum: 0.5)
      expected = Nx.tensor([[0.5, 1.0], [1.5, 2.0]])

      assert Nx.all(
               Nx.less_equal(
                 Nx.abs(Nx.subtract(updated["teacher_fc"]["kernel"], expected)),
                 1.0e-5
               )
             )
             |> Nx.to_number() == 1
    end
  end

  describe "recommended_defaults/1" do
    test "returns defaults for small size" do
      defaults = DINOv3.recommended_defaults(:small)
      assert defaults[:embed_dim] == 384
      assert defaults[:num_heads] == 6
      assert defaults[:patch_size] == 16
      assert defaults[:ffn_type] == :mlp
    end

    test "returns defaults for large size with SwiGLU" do
      defaults = DINOv3.recommended_defaults(:large)
      assert defaults[:embed_dim] == 1024
      assert defaults[:num_heads] == 16
      assert defaults[:ffn_type] == :swiglu
    end

    test "returns defaults for all sizes" do
      for size <- [:small, :base, :large, :huge, :giant] do
        defaults = DINOv3.recommended_defaults(size)
        assert is_list(defaults)
        assert defaults[:patch_size] == 16
      end
    end
  end
end
