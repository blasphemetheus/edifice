defmodule Edifice.Blocks.KVCacheTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.KVCache

  @batch 2
  @num_layers 4
  @num_heads 4
  @head_dim 8

  describe "init/1" do
    test "creates cache with correct structure" do
      cache =
        KVCache.init(
          batch_size: @batch,
          num_layers: @num_layers,
          num_heads: @num_heads,
          head_dim: @head_dim
        )

      assert is_map(cache)
      assert cache.position == 0
      assert cache.num_layers == @num_layers
      assert cache.max_seq_len == 2048
      assert map_size(cache.cache) == @num_layers
    end

    test "pre-allocated tensors have correct shape" do
      cache =
        KVCache.init(
          batch_size: @batch,
          num_layers: 2,
          num_heads: @num_heads,
          head_dim: @head_dim,
          max_seq_len: 64
        )

      {k, v} = cache.cache[0]
      assert Nx.shape(k) == {@batch, @num_heads, 64, @head_dim}
      assert Nx.shape(v) == {@batch, @num_heads, 64, @head_dim}
    end

    test "respects custom max_seq_len" do
      cache =
        KVCache.init(
          batch_size: 1,
          num_layers: 1,
          num_heads: 2,
          head_dim: 4,
          max_seq_len: 128
        )

      assert cache.max_seq_len == 128
      {k, _v} = cache.cache[0]
      assert Nx.axis_size(k, 2) == 128
    end

    test "respects custom type" do
      cache =
        KVCache.init(
          batch_size: 1,
          num_layers: 1,
          num_heads: 2,
          head_dim: 4,
          type: :f16
        )

      {k, _v} = cache.cache[0]
      assert Nx.type(k) == {:f, 16}
    end

    test "initializes all layers" do
      cache =
        KVCache.init(
          batch_size: 1,
          num_layers: 6,
          num_heads: 2,
          head_dim: 4
        )

      for i <- 0..5 do
        assert Map.has_key?(cache.cache, i)
      end
    end
  end

  describe "update/4" do
    setup do
      cache =
        KVCache.init(
          batch_size: @batch,
          num_layers: 2,
          num_heads: @num_heads,
          head_dim: @head_dim,
          max_seq_len: 32
        )

      %{cache: cache}
    end

    test "appends new K/V and returns valid portion", %{cache: cache} do
      new_k = Nx.broadcast(1.0, {@batch, @num_heads, 3, @head_dim})
      new_v = Nx.broadcast(2.0, {@batch, @num_heads, 3, @head_dim})

      {updated, cached_k, cached_v} = KVCache.update(cache, 0, new_k, new_v)

      assert updated.position == 3
      assert Nx.shape(cached_k) == {@batch, @num_heads, 3, @head_dim}
      assert Nx.shape(cached_v) == {@batch, @num_heads, 3, @head_dim}
    end

    test "successive updates accumulate", %{cache: cache} do
      # First update: 3 tokens
      new_k1 = Nx.broadcast(1.0, {@batch, @num_heads, 3, @head_dim})
      new_v1 = Nx.broadcast(1.0, {@batch, @num_heads, 3, @head_dim})
      {cache, _, _} = KVCache.update(cache, 0, new_k1, new_v1)

      # Second update: 2 more tokens
      new_k2 = Nx.broadcast(2.0, {@batch, @num_heads, 2, @head_dim})
      new_v2 = Nx.broadcast(2.0, {@batch, @num_heads, 2, @head_dim})
      {cache, cached_k, cached_v} = KVCache.update(cache, 0, new_k2, new_v2)

      assert cache.position == 5
      assert Nx.shape(cached_k) == {@batch, @num_heads, 5, @head_dim}
      assert Nx.shape(cached_v) == {@batch, @num_heads, 5, @head_dim}
    end

    test "different layers are independent", %{cache: cache} do
      new_k = Nx.broadcast(1.0, {@batch, @num_heads, 2, @head_dim})
      new_v = Nx.broadcast(1.0, {@batch, @num_heads, 2, @head_dim})

      {cache, _, _} = KVCache.update(cache, 0, new_k, new_v)

      # Layer 1 should still have no cached data at position 0
      {k1, v1} = KVCache.get(cache, 1)
      # Position was globally updated to 2, so get returns first 2 entries
      assert Nx.shape(k1) == {@batch, @num_heads, 2, @head_dim}
      # But layer 1 data should be zeros
      assert Nx.to_number(Nx.reduce_max(Nx.abs(k1))) < 1.0e-6
      assert Nx.to_number(Nx.reduce_max(Nx.abs(v1))) < 1.0e-6
    end
  end

  describe "seq_length/1" do
    test "returns 0 for fresh cache" do
      cache =
        KVCache.init(
          batch_size: 1,
          num_layers: 1,
          num_heads: 2,
          head_dim: 4
        )

      assert KVCache.seq_length(cache) == 0
    end

    test "returns correct length after updates" do
      cache =
        KVCache.init(
          batch_size: 1,
          num_layers: 1,
          num_heads: 2,
          head_dim: 4
        )

      new_k = Nx.broadcast(1.0, {1, 2, 5, 4})
      new_v = Nx.broadcast(1.0, {1, 2, 5, 4})
      {cache, _, _} = KVCache.update(cache, 0, new_k, new_v)

      assert KVCache.seq_length(cache) == 5
    end
  end

  describe "get/2" do
    test "returns valid K/V for a layer" do
      cache =
        KVCache.init(
          batch_size: @batch,
          num_layers: 2,
          num_heads: @num_heads,
          head_dim: @head_dim,
          max_seq_len: 32
        )

      new_k = Nx.broadcast(3.0, {@batch, @num_heads, 4, @head_dim})
      new_v = Nx.broadcast(7.0, {@batch, @num_heads, 4, @head_dim})
      {cache, _, _} = KVCache.update(cache, 0, new_k, new_v)

      {k, v} = KVCache.get(cache, 0)
      assert Nx.shape(k) == {@batch, @num_heads, 4, @head_dim}
      assert_in_delta Nx.to_number(Nx.mean(k)), 3.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.mean(v)), 7.0, 1.0e-5
    end
  end

  describe "reset/1" do
    test "resets position to 0" do
      cache =
        KVCache.init(
          batch_size: 1,
          num_layers: 1,
          num_heads: 2,
          head_dim: 4
        )

      new_k = Nx.broadcast(1.0, {1, 2, 3, 4})
      new_v = Nx.broadcast(1.0, {1, 2, 3, 4})
      {cache, _, _} = KVCache.update(cache, 0, new_k, new_v)
      assert KVCache.seq_length(cache) == 3

      cache = KVCache.reset(cache)
      assert KVCache.seq_length(cache) == 0
    end
  end

  describe "build_cached_attention/1" do
    test "returns a function" do
      attn_fn =
        KVCache.build_cached_attention(
          num_heads: @num_heads,
          head_dim: @head_dim,
          layer_idx: 0
        )

      assert is_function(attn_fn, 4)
    end

    test "computes attention with cache" do
      cache =
        KVCache.init(
          batch_size: @batch,
          num_layers: 1,
          num_heads: @num_heads,
          head_dim: @head_dim,
          max_seq_len: 32
        )

      attn_fn =
        KVCache.build_cached_attention(
          num_heads: @num_heads,
          head_dim: @head_dim,
          layer_idx: 0
        )

      hidden = @num_heads * @head_dim
      q = Nx.broadcast(0.5, {@batch, @num_heads, 3, @head_dim})
      k = Nx.broadcast(0.5, {@batch, @num_heads, 3, @head_dim})
      v = Nx.broadcast(0.5, {@batch, @num_heads, 3, @head_dim})

      {output, updated_cache} = attn_fn.(q, k, v, cache)

      assert Nx.shape(output) == {@batch, 3, hidden}
      assert KVCache.seq_length(updated_cache) == 3
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end
end
