defmodule Edifice.Blocks.RoPEYarnTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.RoPE

  @dim 64
  @max_seq_len 128

  describe "precompute_freqs/3 with YARN scaling" do
    test "shape is preserved with YARN scaling" do
      {cos_plain, sin_plain} = RoPE.precompute_freqs(@dim, @max_seq_len)

      {cos_yarn, sin_yarn} =
        RoPE.precompute_freqs(@dim, @max_seq_len,
          scaling_type: :yarn,
          target_length: 8192,
          original_length: 4096
        )

      assert Nx.shape(cos_yarn) == Nx.shape(cos_plain)
      assert Nx.shape(sin_yarn) == Nx.shape(sin_plain)
      assert Nx.shape(cos_yarn) == {@max_seq_len, div(@dim, 2)}
    end

    test "scale=1.0 (target == original) is a no-op" do
      {cos_plain, sin_plain} = RoPE.precompute_freqs(@dim, @max_seq_len)

      {cos_yarn, sin_yarn} =
        RoPE.precompute_freqs(@dim, @max_seq_len,
          scaling_type: :yarn,
          target_length: 4096,
          original_length: 4096
        )

      assert Nx.all_close(cos_plain, cos_yarn, atol: 1.0e-5) |> Nx.to_number() == 1
      assert Nx.all_close(sin_plain, sin_yarn, atol: 1.0e-5) |> Nx.to_number() == 1
    end

    test "scaling_type :none is identical to default" do
      {cos_default, sin_default} = RoPE.precompute_freqs(@dim, @max_seq_len)

      {cos_none, sin_none} =
        RoPE.precompute_freqs(@dim, @max_seq_len, scaling_type: :none)

      assert Nx.all_close(cos_default, cos_none) |> Nx.to_number() == 1
      assert Nx.all_close(sin_default, sin_none) |> Nx.to_number() == 1
    end

    test "YARN produces different freqs when target > original" do
      {cos_plain, _sin_plain} = RoPE.precompute_freqs(@dim, @max_seq_len)

      {cos_yarn, _sin_yarn} =
        RoPE.precompute_freqs(@dim, @max_seq_len,
          scaling_type: :yarn,
          target_length: 16384,
          original_length: 4096
        )

      # They should differ (at least some frequencies are scaled)
      refute Nx.all_close(cos_plain, cos_yarn, atol: 1.0e-3) |> Nx.to_number() == 1
    end
  end

  describe "yarn_scale_freqs/3" do
    test "returns tensor of same shape as input freqs" do
      half_dim = div(@dim, 2)

      freqs =
        Nx.pow(
          10_000.0,
          Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), @dim))
        )
        |> Nx.as_type(:f32)

      scaled = RoPE.yarn_scale_freqs(freqs, @dim, target_length: 8192, original_length: 4096)
      assert Nx.shape(scaled) == Nx.shape(freqs)
    end

    test "all scaled frequencies are finite" do
      half_dim = div(@dim, 2)

      freqs =
        Nx.pow(
          10_000.0,
          Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), @dim))
        )
        |> Nx.as_type(:f32)

      scaled = RoPE.yarn_scale_freqs(freqs, @dim, target_length: 8192, original_length: 4096)
      assert Nx.all(Nx.is_nan(scaled) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(scaled) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
