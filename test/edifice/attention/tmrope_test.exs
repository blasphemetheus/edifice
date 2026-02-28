defmodule Edifice.Attention.TMRoPETest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.TMRoPE

  @batch 2
  @seq_len 16
  @embed_dim 32

  defp random_tensor(shape) do
    key = Nx.Random.key(42)
    {t, _key} = Nx.Random.uniform(key, shape: shape)
    t
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = TMRoPE.build(embed_dim: @embed_dim)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shapes" do
      model = TMRoPE.build(embed_dim: @embed_dim)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "tmrope_query" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "tmrope_key" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "tmrope_positions" => Nx.template({@batch, @seq_len}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "tmrope_query" => random_tensor({@batch, @seq_len, @embed_dim}),
        "tmrope_key" => random_tensor({@batch, @seq_len, @embed_dim}),
        "tmrope_positions" => Nx.iota({@batch, @seq_len}, type: :f32)
      }

      output = predict_fn.(params, input)

      assert is_map(output)
      assert Map.has_key?(output, :query)
      assert Map.has_key?(output, :key)
      assert Nx.shape(output.query) == {@batch, @seq_len, @embed_dim}
      assert Nx.shape(output.key) == {@batch, @seq_len, @embed_dim}
    end

    test "output contains finite values" do
      model = TMRoPE.build(embed_dim: @embed_dim)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "tmrope_query" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "tmrope_key" => Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        "tmrope_positions" => Nx.template({@batch, @seq_len}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      input = %{
        "tmrope_query" => random_tensor({@batch, @seq_len, @embed_dim}),
        "tmrope_key" => random_tensor({@batch, @seq_len, @embed_dim}),
        "tmrope_positions" => Nx.iota({@batch, @seq_len}, type: :f32)
      }

      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output.query) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output.key) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "accepts custom temporal_scaling" do
      model = TMRoPE.build(embed_dim: @embed_dim, temporal_scaling: 0.5)
      assert %Axon{} = model
    end
  end

  describe "tmrope_freqs/2" do
    test "returns tensor of shape [embed_dim / 2]" do
      freqs = TMRoPE.tmrope_freqs(@embed_dim)
      assert Nx.shape(freqs) == {div(@embed_dim, 2)}
    end

    test "all frequencies are positive" do
      freqs = TMRoPE.tmrope_freqs(@embed_dim)
      assert Nx.all(Nx.greater(freqs, 0.0)) |> Nx.to_number() == 1
    end

    test "temporal_scaling affects frequencies" do
      freqs_1 = TMRoPE.tmrope_freqs(@embed_dim, temporal_scaling: 1.0)
      freqs_half = TMRoPE.tmrope_freqs(@embed_dim, temporal_scaling: 0.5)

      # Frequencies should be scaled
      ratio = Nx.divide(freqs_half, freqs_1) |> Nx.mean() |> Nx.to_number()
      assert_in_delta(ratio, 0.5, 0.01)
    end
  end

  describe "assign_positions/2" do
    test "assigns uniform positions for text modality" do
      metadata = [{:text, [start_idx: 0, end_idx: 5, time: 0.0]}]
      positions = TMRoPE.assign_positions(5, metadata)

      assert Nx.shape(positions) == {5}
      assert Nx.to_flat_list(positions) == [0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "assigns uniform positions for image modality" do
      metadata = [{:image, [start_idx: 0, end_idx: 4, time: 1.5]}]
      positions = TMRoPE.assign_positions(4, metadata)

      assert Nx.shape(positions) == {4}
      assert Nx.to_flat_list(positions) == [1.5, 1.5, 1.5, 1.5]
    end

    test "assigns per-frame positions for video modality" do
      metadata = [
        {:video, [start_idx: 0, end_idx: 6, patches_per_frame: 2, frame_times: [0.0, 1.0, 2.0]]}
      ]

      positions = TMRoPE.assign_positions(6, metadata)

      assert Nx.shape(positions) == {6}
      # 2 patches per frame, 3 frames
      assert Nx.to_flat_list(positions) == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    end

    test "handles mixed modalities" do
      metadata = [
        {:text, [start_idx: 0, end_idx: 2, time: 0.0]},
        {:image, [start_idx: 2, end_idx: 4, time: 1.0]},
        {:video, [start_idx: 4, end_idx: 8, patches_per_frame: 2, frame_times: [2.0, 3.0]]}
      ]

      positions = TMRoPE.assign_positions(8, metadata)

      expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
      assert Nx.to_flat_list(positions) == expected
    end

    test "handles audio modality same as text" do
      metadata = [{:audio, [start_idx: 0, end_idx: 3, time: 2.5]}]
      positions = TMRoPE.assign_positions(3, metadata)

      assert Nx.to_flat_list(positions) == [2.5, 2.5, 2.5]
    end
  end

  describe "apply_tmrope/4" do
    test "returns query and key tensors of same shape" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})
      positions = Nx.iota({@batch, @seq_len}, type: :f32)

      {q_rot, k_rot} = TMRoPE.apply_tmrope(q, k, positions)

      assert Nx.shape(q_rot) == Nx.shape(q)
      assert Nx.shape(k_rot) == Nx.shape(k)
    end

    test "handles 1D position input by broadcasting" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})
      positions = Nx.iota({@seq_len}, type: :f32)

      {q_rot, k_rot} = TMRoPE.apply_tmrope(q, k, positions)

      assert Nx.shape(q_rot) == Nx.shape(q)
      assert Nx.shape(k_rot) == Nx.shape(k)
    end

    test "output contains finite values" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})
      positions = Nx.iota({@batch, @seq_len}, type: :f32)

      {q_rot, k_rot} = TMRoPE.apply_tmrope(q, k, positions)

      assert Nx.all(Nx.is_nan(q_rot) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(k_rot) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "rotation is norm-preserving" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})
      positions = Nx.iota({@batch, @seq_len}, type: :f32)

      {q_rot, k_rot} = TMRoPE.apply_tmrope(q, k, positions)

      q_norm = Nx.sum(Nx.pow(q, 2), axes: [-1])
      q_rot_norm = Nx.sum(Nx.pow(q_rot, 2), axes: [-1])
      k_norm = Nx.sum(Nx.pow(k, 2), axes: [-1])
      k_rot_norm = Nx.sum(Nx.pow(k_rot, 2), axes: [-1])

      q_diff = Nx.subtract(q_norm, q_rot_norm) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      k_diff = Nx.subtract(k_norm, k_rot_norm) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert q_diff < 1.0e-4
      assert k_diff < 1.0e-4
    end

    test "same temporal position produces same rotation for different indices" do
      # Two tokens at same temporal position should have same rotation
      q = Nx.broadcast(1.0, {@batch, 4, @embed_dim})
      k = Nx.broadcast(1.0, {@batch, 4, @embed_dim})
      # All positions = 1.0
      positions = Nx.broadcast(1.0, {@batch, 4})

      {q_rot, _k_rot} = TMRoPE.apply_tmrope(q, k, positions)

      # All sequence positions should have same output since temporal position is same
      first = Nx.slice_along_axis(q_rot, 0, 1, axis: 1)
      second = Nx.slice_along_axis(q_rot, 1, 1, axis: 1)

      diff = Nx.subtract(first, second) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "different temporal positions produce different rotations" do
      q = Nx.broadcast(1.0, {@batch, 4, @embed_dim})
      k = Nx.broadcast(1.0, {@batch, 4, @embed_dim})
      # Different positions
      positions = Nx.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])

      {q_rot, _k_rot} = TMRoPE.apply_tmrope(q, k, positions)

      first = Nx.slice_along_axis(q_rot, 0, 1, axis: 1)
      second = Nx.slice_along_axis(q_rot, 1, 1, axis: 1)

      diff = Nx.subtract(first, second) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 0.0
    end
  end

  describe "frame_times/2" do
    test "generates sequential frame times" do
      times = TMRoPE.frame_times(3)
      assert times == [0.0, 1.0, 2.0]
    end

    test "respects start_time option" do
      times = TMRoPE.frame_times(3, start_time: 5.0)
      assert times == [5.0, 6.0, 7.0]
    end

    test "respects frame_interval option" do
      times = TMRoPE.frame_times(3, frame_interval: 0.5)
      assert times == [0.0, 0.5, 1.0]
    end

    test "combines start_time and frame_interval" do
      times = TMRoPE.frame_times(3, start_time: 2.0, frame_interval: 0.5)
      assert times == [2.0, 2.5, 3.0]
    end
  end

  describe "output_size/1" do
    test "returns embed_dim" do
      assert TMRoPE.output_size(embed_dim: 64) == 64
      assert TMRoPE.output_size(embed_dim: 128) == 128
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = TMRoPE.recommended_defaults()

      assert Keyword.has_key?(defaults, :modalities)
      assert Keyword.has_key?(defaults, :max_position)
      assert Keyword.has_key?(defaults, :temporal_scaling)
      assert Keyword.has_key?(defaults, :base)
      assert defaults[:modalities] == [:text, :image, :video]
      assert defaults[:temporal_scaling] == 1.0
    end
  end
end
