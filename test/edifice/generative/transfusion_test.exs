defmodule Edifice.Generative.TransfusionTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.Transfusion

  @batch 2
  @text_len 4
  @image_len 4
  @seq_len @text_len + @image_len
  @embed_dim 32
  @hidden_size 32
  @num_heads 4
  @num_layers 2
  @vocab_size 20
  @patch_dim 16

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    vocab_size: @vocab_size,
    patch_dim: @patch_dim,
    dropout: 0.0
  ]

  # First @text_len positions are text (0), remainder are image (1).
  defp modality_mask do
    text_part = Nx.broadcast(Nx.tensor(0, type: :s32), {@batch, @text_len})
    img_part = Nx.broadcast(Nx.tensor(1, type: :s32), {@batch, @image_len})
    Nx.concatenate([text_part, img_part], axis: 1)
  end

  defp random_inputs(seed) do
    key = Nx.Random.key(seed)
    {seq, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    {timestep, _key} = Nx.Random.uniform(key, shape: {@batch})

    %{
      "sequence" => seq,
      "modality_mask" => modality_mask(),
      "timestep" => timestep
    }
  end

  defp build_model_and_predict(opts \\ @opts, seed \\ 42) do
    model = Transfusion.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
      "modality_mask" => Nx.template({@batch, @seq_len}, :s32),
      "timestep" => Nx.template({@batch}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_inputs(seed))
    {output, params}
  end

  # ============================================================================
  # build/1
  # ============================================================================

  describe "build/1" do
    test "returns an Axon model" do
      model = Transfusion.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shapes" do
      {output, _} = build_model_and_predict()

      assert Nx.shape(output.text_logits) == {@batch, @seq_len, @vocab_size}
      assert Nx.shape(output.image_pred) == {@batch, @seq_len, @patch_dim}
    end

    test "output contains finite values" do
      {output, _} = build_model_and_predict()

      for tensor <- [output.text_logits, output.image_pred] do
        assert Nx.all(Nx.is_nan(tensor) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(tensor) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    test "works with embed_dim == hidden_size (no input projection)" do
      # Same embed_dim and hidden_size: skips the dense projection layer
      opts = Keyword.merge(@opts, embed_dim: @hidden_size)
      {output, _} = build_model_and_predict(opts)
      assert Nx.shape(output.text_logits) == {@batch, @seq_len, @vocab_size}
    end

    test "works with single layer" do
      opts = Keyword.put(@opts, :num_layers, 1)
      {output, _} = build_model_and_predict(opts)
      assert Nx.shape(output.text_logits) == {@batch, @seq_len, @vocab_size}
    end

    test "works when entire sequence is text" do
      model = Transfusion.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      all_text_mask = Nx.broadcast(Nx.tensor(0, type: :s32), {@batch, @seq_len})
      key = Nx.Random.key(99)
      {seq, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      {ts, _} = Nx.Random.uniform(key, shape: {@batch})

      template = %{
        "sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "modality_mask" => Nx.template({@batch, @seq_len}, :s32),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      output =
        predict_fn.(params, %{
          "sequence" => seq,
          "modality_mask" => all_text_mask,
          "timestep" => ts
        })

      assert Nx.shape(output.text_logits) == {@batch, @seq_len, @vocab_size}
    end

    test "works when entire sequence is image" do
      model = Transfusion.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      all_img_mask = Nx.broadcast(Nx.tensor(1, type: :s32), {@batch, @seq_len})
      key = Nx.Random.key(77)
      {seq, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      {ts, _} = Nx.Random.uniform(key, shape: {@batch})

      template = %{
        "sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "modality_mask" => Nx.template({@batch, @seq_len}, :s32),
        "timestep" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      output =
        predict_fn.(params, %{
          "sequence" => seq,
          "modality_mask" => all_img_mask,
          "timestep" => ts
        })

      assert Nx.shape(output.image_pred) == {@batch, @seq_len, @patch_dim}
    end
  end

  # ============================================================================
  # build_mixed_mask/3
  # ============================================================================

  describe "build_mixed_mask/3" do
    test "returns tensor of correct shape" do
      mask = Transfusion.build_mixed_mask(@text_len, @image_len)
      assert Nx.shape(mask) == {@seq_len, @seq_len}
    end

    test "text-to-future-text is masked (causal)" do
      mask = Transfusion.build_mixed_mask(4, 0)
      # In a text-only sequence the mask should be strictly lower-triangular
      # mask[0, 1] should be false (token 0 cannot attend to token 1)
      assert Nx.to_number(mask[0][1]) == 0
      assert Nx.to_number(mask[1][0]) == 1
    end

    test "image-to-image is always allowed (bidirectional)" do
      mask = Transfusion.build_mixed_mask(0, 4)
      # All positions are image, so all pairs should be allowed
      assert Nx.all(mask) |> Nx.to_number() == 1
    end

    test "image attends to preceding text" do
      # 2 text tokens then 2 image patches
      mask = Transfusion.build_mixed_mask(2, 2)
      # Image patch at position 2 should attend to text at position 0
      assert Nx.to_number(mask[2][0]) == 1
      # Image patch at position 2 should attend to image patch at position 3
      assert Nx.to_number(mask[2][3]) == 1
    end

    test "text cannot attend to future image patches" do
      # 2 text, 2 image  →  total 4
      mask = Transfusion.build_mixed_mask(2, 2)
      # Text at position 0 cannot attend to image at position 2
      assert Nx.to_number(mask[0][2]) == 0
      assert Nx.to_number(mask[1][3]) == 0
    end

    test "diagonal (self-attention) is always allowed" do
      mask = Transfusion.build_mixed_mask(3, 3)

      for i <- 0..5 do
        assert Nx.to_number(mask[i][i]) == 1
      end
    end
  end

  # ============================================================================
  # transfusion_loss/4
  # ============================================================================

  describe "transfusion_loss/4" do
    setup do
      key = Nx.Random.key(42)
      {text_logits, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @vocab_size})
      {image_pred, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @patch_dim})
      {image_targets, _key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @patch_dim})

      text_targets = Nx.broadcast(Nx.tensor(0, type: :s64), {@batch, @seq_len})
      text_mask = Nx.broadcast(Nx.tensor(1.0), {@batch, @seq_len})
      image_mask = Nx.broadcast(Nx.tensor(1.0), {@batch, @seq_len})

      targets = %{
        text_targets: text_targets,
        image_targets: image_targets,
        text_mask: text_mask,
        image_mask: image_mask
      }

      %{
        text_logits: text_logits,
        image_pred: image_pred,
        targets: targets
      }
    end

    test "returns a scalar tensor", %{text_logits: tl, image_pred: ip, targets: t} do
      loss = Transfusion.transfusion_loss(tl, ip, t)
      assert Nx.shape(loss) == {}
    end

    test "loss is positive", %{text_logits: tl, image_pred: ip, targets: t} do
      loss = Transfusion.transfusion_loss(tl, ip, t)
      assert Nx.to_number(loss) > 0
    end

    test "image MSE loss is zero when predictions match targets", %{text_logits: tl, targets: t} do
      perfect_pred = t[:image_targets]
      # Zero image mask, full text mask → only text loss contributes
      no_image_targets = Map.put(t, :image_mask, Nx.broadcast(Nx.tensor(0.0), {@batch, @seq_len}))
      loss_no_image = Transfusion.transfusion_loss(tl, perfect_pred, no_image_targets)

      # Only text loss now
      no_text_targets = Map.put(t, :text_mask, Nx.broadcast(Nx.tensor(0.0), {@batch, @seq_len}))
      loss_no_text = Transfusion.transfusion_loss(tl, perfect_pred, no_text_targets)

      # With perfect image predictions and all-image mask, image loss should be ~0
      assert Nx.to_number(loss_no_text) < 1.0e-4
      # Text loss should be > 0 (random logits)
      assert Nx.to_number(loss_no_image) > 0
    end

    test "text_weight and image_weight scale the losses", %{
      text_logits: tl,
      image_pred: ip,
      targets: t
    } do
      loss_1x = Transfusion.transfusion_loss(tl, ip, t, text_weight: 1.0, image_weight: 1.0)
      loss_2x = Transfusion.transfusion_loss(tl, ip, t, text_weight: 2.0, image_weight: 2.0)

      # Double weights should roughly double the loss
      ratio = Nx.to_number(loss_2x) / Nx.to_number(loss_1x)
      assert_in_delta ratio, 2.0, 0.01
    end

    test "loss is finite", %{text_logits: tl, image_pred: ip, targets: t} do
      loss = Transfusion.transfusion_loss(tl, ip, t)
      assert Nx.is_nan(loss) |> Nx.to_number() == 0
      assert Nx.is_infinity(loss) |> Nx.to_number() == 0
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Transfusion.output_size(@opts) == @hidden_size
    end

    test "returns default when no opts" do
      assert Transfusion.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = Transfusion.param_count(@opts)
      assert is_integer(count)
      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = Transfusion.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :vocab_size)
      assert Keyword.has_key?(defaults, :patch_dim)
    end
  end
end
