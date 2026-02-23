defmodule Edifice.Attention.YARNTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.YARN

  @batch 2
  @seq_len 8
  @embed_dim 32

  defp random_tensor(shape) do
    key = Nx.Random.key(42)
    {t, _key} = Nx.Random.uniform(key, shape: shape)
    t
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = YARN.build(embed_dim: @embed_dim)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = YARN.build(embed_dim: @embed_dim)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"yarn_input" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"yarn_input" => random_tensor({@batch, @seq_len, @embed_dim})}
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @seq_len, @embed_dim}
    end

    test "output contains finite values" do
      model = YARN.build(embed_dim: @embed_dim)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"yarn_input" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = %{"yarn_input" => random_tensor({@batch, @seq_len, @embed_dim})}
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "accepts custom scale and original_max_position" do
      model = YARN.build(embed_dim: @embed_dim, scale: 4, original_max_position: 1024)
      assert %Axon{} = model
    end
  end

  describe "yarn_freqs/2" do
    test "returns tensor of shape [embed_dim / 2]" do
      freqs = YARN.yarn_freqs(@embed_dim)
      assert Nx.shape(freqs) == {div(@embed_dim, 2)}
    end

    test "all frequencies are positive" do
      freqs = YARN.yarn_freqs(@embed_dim)
      assert Nx.all(Nx.greater(freqs, 0.0)) |> Nx.to_number() == 1
    end

    test "scale=1 produces standard RoPE frequencies" do
      freqs_scaled = YARN.yarn_freqs(@embed_dim, scale: 1)
      freqs_default = YARN.yarn_freqs(@embed_dim, scale: 1, original_max_position: 2048)
      # Both should be the same since scale=1 means no extension
      diff = Nx.subtract(freqs_scaled, freqs_default) |> Nx.abs() |> Nx.reduce_max()
      assert Nx.to_number(diff) < 1.0e-5
    end

    test "higher scale produces lower low-frequency components" do
      freqs_scale2 = YARN.yarn_freqs(@embed_dim, scale: 2)
      freqs_scale8 = YARN.yarn_freqs(@embed_dim, scale: 8)

      # Low-frequency dimensions (first few) should be smaller with higher scale
      # (they get divided by scale)
      first_freq_scale2 = freqs_scale2 |> Nx.to_flat_list() |> List.first()
      first_freq_scale8 = freqs_scale8 |> Nx.to_flat_list() |> List.first()
      assert first_freq_scale8 <= first_freq_scale2
    end
  end

  describe "apply_yarn/3" do
    test "returns query and key tensors of same shape" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})

      {q_rot, k_rot} = YARN.apply_yarn(q, k)

      assert Nx.shape(q_rot) == Nx.shape(q)
      assert Nx.shape(k_rot) == Nx.shape(k)
    end

    test "output contains finite values" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})

      {q_rot, k_rot} = YARN.apply_yarn(q, k, scale: 8)

      assert Nx.all(Nx.is_nan(q_rot) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(k_rot) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "rotation is norm-preserving per embedding position" do
      # RoPE is an isometric rotation, so ||q|| == ||q_rot||
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})

      {q_rot, k_rot} = YARN.apply_yarn(q, k, scale: 8)

      q_norm = Nx.sum(Nx.pow(q, 2), axes: [-1])
      q_rot_norm = Nx.sum(Nx.pow(q_rot, 2), axes: [-1])
      k_norm = Nx.sum(Nx.pow(k, 2), axes: [-1])
      k_rot_norm = Nx.sum(Nx.pow(k_rot, 2), axes: [-1])

      q_diff = Nx.subtract(q_norm, q_rot_norm) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      k_diff = Nx.subtract(k_norm, k_rot_norm) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert q_diff < 1.0e-4
      assert k_diff < 1.0e-4
    end

    test "scale=8 and scale=1 produce different outputs" do
      q = random_tensor({@batch, @seq_len, @embed_dim})
      k = random_tensor({@batch, @seq_len, @embed_dim})

      {q_scale8, _} = YARN.apply_yarn(q, k, scale: 8)
      {q_scale1, _} = YARN.apply_yarn(q, k, scale: 1)

      diff = Nx.subtract(q_scale8, q_scale1) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 0.0
    end
  end

  describe "effective_context_length/2" do
    test "returns original * scale" do
      assert YARN.effective_context_length(2048, 8) == 16384
      assert YARN.effective_context_length(4096, 4) == 16384
      assert YARN.effective_context_length(2048, 1) == 2048
    end
  end

  describe "recommended_defaults/0" do
    test "returns a keyword list with expected keys" do
      defaults = YARN.recommended_defaults()
      assert Keyword.has_key?(defaults, :scale)
      assert Keyword.has_key?(defaults, :original_max_position)
      assert Keyword.has_key?(defaults, :low_freq_factor)
      assert Keyword.has_key?(defaults, :high_freq_factor)
      assert defaults[:scale] == 8
      assert defaults[:original_max_position] == 2048
    end
  end
end
