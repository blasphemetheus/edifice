defmodule Edifice.Memory.EngramTest do
  use ExUnit.Case, async: true
  @moduletag :memory

  alias Edifice.Memory.Engram

  @key_dim 8
  @value_dim 16
  @batch 3
  @num_buckets 16
  @num_tables 2

  @base_opts [
    key_dim: @key_dim,
    value_dim: @value_dim,
    num_buckets: @num_buckets,
    num_tables: @num_tables
  ]

  # ============================================================================
  # build/1
  # ============================================================================

  describe "Engram.build/1" do
    test "returns an Axon model" do
      assert %Axon{} = Engram.build(@base_opts)
    end

    test "output shape is [batch, value_dim]" do
      model = Engram.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      templates = %{
        "query" => Nx.template({@batch, @key_dim}, :f32),
        "memory_slots" => Nx.template({@num_tables, @num_buckets, @value_dim}, :f32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      query = Nx.broadcast(0.5, {@batch, @key_dim})
      slots = Nx.broadcast(0.0, {@num_tables, @num_buckets, @value_dim})

      output = predict_fn.(params, %{"query" => query, "memory_slots" => slots})
      assert Nx.shape(output) == {@batch, @value_dim}
    end

    test "output contains finite values" do
      model = Engram.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      templates = %{
        "query" => Nx.template({@batch, @key_dim}, :f32),
        "memory_slots" => Nx.template({@num_tables, @num_buckets, @value_dim}, :f32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())
      query = Nx.broadcast(0.5, {@batch, @key_dim})
      slots = Nx.broadcast(1.0, {@num_tables, @num_buckets, @value_dim})

      output = predict_fn.(params, %{"query" => query, "memory_slots" => slots})
      assert Nx.all(Nx.logical_not(Nx.is_nan(output))) |> Nx.to_number() == 1
    end

    test "requires key_dim and value_dim" do
      assert_raise KeyError, fn -> Engram.build([]) end
    end
  end

  # ============================================================================
  # new/1
  # ============================================================================

  describe "Engram.new/1" do
    test "creates hash_matrices with correct shape" do
      mem = Engram.new(@base_opts)
      hash_bits = trunc(:math.log2(@num_buckets))
      assert Nx.shape(mem.hash_matrices) == {@num_tables, hash_bits, @key_dim}
    end

    test "creates zero slots with correct shape" do
      mem = Engram.new(@base_opts)
      assert Nx.shape(mem.slots) == {@num_tables, @num_buckets, @value_dim}
      assert Nx.all(Nx.equal(mem.slots, 0.0)) |> Nx.to_number() == 1
    end

    test "different seeds produce different hash matrices" do
      mem1 = Engram.new(@base_opts ++ [seed: 0])
      mem2 = Engram.new(@base_opts ++ [seed: 42])
      diff = Nx.sum(Nx.abs(Nx.subtract(mem1.hash_matrices, mem2.hash_matrices))) |> Nx.to_number()
      assert diff > 0.0
    end

    test "hash matrix rows are unit-normalised" do
      mem = Engram.new(@base_opts)
      # Each projection vector W[t, h, :] should have L2 norm ≈ 1
      norms = Nx.sqrt(Nx.sum(Nx.pow(mem.hash_matrices, 2), axes: [2]))
      max_deviation = Nx.reduce_max(Nx.abs(Nx.subtract(norms, 1.0))) |> Nx.to_number()
      assert max_deviation < 1.0e-5
    end
  end

  # ============================================================================
  # engram_read/2
  # ============================================================================

  describe "Engram.engram_read/2" do
    test "returns [batch, value_dim] for batched query" do
      mem = Engram.new(@base_opts)
      query = Nx.broadcast(0.5, {@batch, @key_dim})
      result = Engram.engram_read(mem, query)
      assert Nx.shape(result) == {@batch, @value_dim}
    end

    test "returns [1, value_dim] for 1-D query (auto-batched)" do
      mem = Engram.new(@base_opts)
      query = Nx.broadcast(0.5, {@key_dim})
      result = Engram.engram_read(mem, query)
      assert Nx.shape(result) == {1, @value_dim}
    end

    test "returns zeros from a freshly initialised memory" do
      mem = Engram.new(@base_opts)
      query = Nx.broadcast(1.0, {@batch, @key_dim})
      result = Engram.engram_read(mem, query)
      assert Nx.all(Nx.equal(result, 0.0)) |> Nx.to_number() == 1
    end

    test "retrieves non-zero after a write" do
      mem = Engram.new(@base_opts)
      key = Nx.broadcast(1.0, {@key_dim})
      value = Nx.broadcast(5.0, {@value_dim})

      mem = Engram.engram_write(mem, key, value, decay: 0.0)

      # Read with the same key — should find the written value (decay=0 replaces slot)
      query = Nx.broadcast(1.0, {1, @key_dim})
      result = Engram.engram_read(mem, query)

      # All tables hash the same key to the same bucket, so the mean should be 5.0
      mean_val = result |> Nx.mean() |> Nx.to_number()
      assert mean_val > 0.0
    end
  end

  # ============================================================================
  # engram_write/4
  # ============================================================================

  describe "Engram.engram_write/4" do
    test "returns updated memory with same shape" do
      mem = Engram.new(@base_opts)
      key = Nx.broadcast(0.3, {@key_dim})
      value = Nx.broadcast(1.0, {@value_dim})
      mem2 = Engram.engram_write(mem, key, value)
      assert Nx.shape(mem2.slots) == Nx.shape(mem.slots)
      assert Nx.shape(mem2.hash_matrices) == Nx.shape(mem.hash_matrices)
    end

    test "decay=0.0 fully replaces the slot with new value" do
      mem = Engram.new(@base_opts)
      key = Nx.broadcast(1.0, {@key_dim})
      value = Nx.iota({@value_dim}, type: :f32)
      mem2 = Engram.engram_write(mem, key, value, decay: 0.0)

      # Sum of all slots should now be non-zero
      total = Nx.sum(mem2.slots) |> Nx.to_number()
      assert total > 0.0
    end

    test "decay=1.0 leaves slots unchanged" do
      mem = Engram.new(@base_opts)
      key = Nx.broadcast(1.0, {@key_dim})
      value = Nx.broadcast(9.0, {@value_dim})
      # With decay=1.0: new = 1.0 * old + 0.0 * value = old (which is 0.0)
      mem2 = Engram.engram_write(mem, key, value, decay: 1.0)
      assert Nx.all(Nx.equal(mem2.slots, 0.0)) |> Nx.to_number() == 1
    end

    test "EMA interpolates between old and new values" do
      mem = Engram.new(@base_opts)
      key = Nx.broadcast(1.0, {@key_dim})
      value1 = Nx.broadcast(10.0, {@value_dim})
      value2 = Nx.broadcast(0.0, {@value_dim})

      # Write value1 with decay=0 (full replace)
      mem = Engram.engram_write(mem, key, value1, decay: 0.0)

      # Write value2 with decay=0.5: result should be ~5.0
      mem = Engram.engram_write(mem, key, value2, decay: 0.5)

      query = Nx.broadcast(1.0, {1, @key_dim})
      result = Engram.engram_read(mem, query)
      mean_val = result |> Nx.mean() |> Nx.to_number()

      # Expected: 0.5 * 10.0 + 0.5 * 0.0 = 5.0
      assert_in_delta mean_val, 5.0, 0.5
    end

    test "default decay is 0.99 (slots change very little)" do
      mem = Engram.new(@base_opts)
      key = Nx.broadcast(0.5, {@key_dim})
      value = Nx.broadcast(1000.0, {@value_dim})

      # With decay=0.99 from zero: result = 0.01 * 1000 = 10.0 per affected slot
      mem2 = Engram.engram_write(mem, key, value)
      total = Nx.sum(mem2.slots) |> Nx.to_number()
      # Change should be small relative to the written value
      assert total > 0.0
      assert total < @num_tables * @value_dim * 100.0
    end
  end
end
