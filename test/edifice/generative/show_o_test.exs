defmodule Edifice.Generative.ShowOTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.ShowO

  @opts [
    vocab_size: 64,
    codebook_size: 16,
    hidden_size: 32,
    num_heads: 4,
    num_layers: 2,
    intermediate_size: 64,
    qk_norm: true
  ]

  describe "build/1" do
    test "produces correct output shape" do
      model = ShowO.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({2, 8}, :s64),
            "modality_mask" => Nx.template({2, 8}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[1, 2, 3, 50, 51, 52, 53, 54], [4, 5, 6, 50, 51, 52, 53, 54]])
      modality_mask = Nx.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})
      assert Nx.shape(out) == {2, 8, 64}
    end

    test "outputs are finite" do
      model = ShowO.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({2, 6}, :s64),
            "modality_mask" => Nx.template({2, 6}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[1, 2, 50, 51, 52, 53], [3, 4, 50, 51, 52, 53]])
      modality_mask = Nx.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})

      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = ShowO.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({1, 6}, :s64),
            "modality_mask" => Nx.template({1, 6}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[1, 2, 50, 51, 52, 53]])
      modality_mask = Nx.tensor([[0, 0, 1, 1, 1, 1]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})
      assert Nx.shape(out) == {1, 6, 64}
    end

    test "text-only mode (all zeros modality mask)" do
      model = ShowO.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({2, 4}, :s64),
            "modality_mask" => Nx.template({2, 4}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
      modality_mask = Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})
      assert Nx.shape(out) == {2, 4, 64}
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
    end

    test "image-only mode (all ones modality mask)" do
      model = ShowO.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({2, 4}, :s64),
            "modality_mask" => Nx.template({2, 4}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[50, 51, 52, 53], [54, 55, 56, 57]])
      modality_mask = Nx.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})
      assert Nx.shape(out) == {2, 4, 64}
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
    end

    test "without QK-norm" do
      opts = Keyword.put(@opts, :qk_norm, false)
      model = ShowO.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({2, 6}, :s64),
            "modality_mask" => Nx.template({2, 6}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[1, 2, 50, 51, 52, 53], [3, 4, 50, 51, 52, 53]])
      modality_mask = Nx.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})
      assert Nx.shape(out) == {2, 6, 64}
    end

    test "different vocab and hidden size" do
      opts =
        @opts
        |> Keyword.put(:vocab_size, 128)
        |> Keyword.put(:hidden_size, 64)
        |> Keyword.put(:num_heads, 8)

      model = ShowO.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input_ids" => Nx.template({2, 4}, :s64),
            "modality_mask" => Nx.template({2, 4}, :s64)
          },
          %{}
        )

      input_ids = Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
      modality_mask = Nx.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

      out = predict_fn.(params, %{"input_ids" => input_ids, "modality_mask" => modality_mask})
      assert Nx.shape(out) == {2, 4, 128}
    end
  end

  describe "cosine_mask_ratio/2" do
    test "starts at 1.0 (fully masked)" do
      assert ShowO.cosine_mask_ratio(0, 18) == 1.0
    end

    test "ends near 0.0 (fully unmasked)" do
      assert abs(ShowO.cosine_mask_ratio(18, 18)) < 1.0e-6
    end

    test "monotonically decreasing" do
      ratios = for step <- 0..18, do: ShowO.cosine_mask_ratio(step, 18)
      pairs = Enum.zip(ratios, tl(ratios))
      assert Enum.all?(pairs, fn {a, b} -> a >= b end)
    end
  end

  describe "build_omni_mask/2" do
    test "text-only is causal" do
      mask = ShowO.build_omni_mask(4, 0)
      assert Nx.shape(mask) == {4, 4}

      # Lower triangle should be true (causal)
      expected =
        Nx.tensor(
          [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
          ],
          type: :u8
        )

      assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "image-only is fully connected" do
      mask = ShowO.build_omni_mask(0, 4)
      assert Nx.shape(mask) == {4, 4}
      # All ones
      assert Nx.all(mask) |> Nx.to_number() == 1
    end

    test "mixed text+image has correct structure" do
      mask = ShowO.build_omni_mask(2, 2)
      assert Nx.shape(mask) == {4, 4}

      # Row 0 (text): [1, 0, 0, 0] - only self
      # Row 1 (text): [1, 1, 0, 0] - causal
      # Row 2 (image): [1, 1, 1, 1] - sees all text + all image
      # Row 3 (image): [1, 1, 1, 1] - sees all text + all image
      expected =
        Nx.tensor(
          [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
          ],
          type: :u8
        )

      assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert ShowO.output_size(hidden_size: 256) == 256
      assert ShowO.output_size(hidden_size: 512) == 512
    end

    test "uses default" do
      assert ShowO.output_size([]) == 256
    end
  end

  describe "Edifice.build/2" do
    test "builds show_o via registry" do
      model = Edifice.build(:show_o, @opts)
      assert %Axon{} = model
    end
  end
end
