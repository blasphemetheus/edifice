defmodule Edifice.SSM.MambaSSDCoverage2Test do
  @moduledoc """
  Extended coverage tests for Edifice.SSM.MambaSSD.

  Targets the ~54% of uncovered code: matmul scan paths, chunked scan with
  remainder, inter-chunk propagation, cumulative products, intra-chunk scan
  branching, and utility functions.
  """
  use ExUnit.Case, async: true

  @moduletag timeout: 300_000

  alias Edifice.SSM.MambaSSD

  # Small dims for speed
  @batch 2
  @embed 16
  @hidden 16
  @state_size 4
  @num_layers 1

  defp build_and_run(opts) do
    model = MambaSSD.build(opts)
    assert %Axon{} = model

    seq = Keyword.fetch!(opts, :seq_len)
    embed = Keyword.fetch!(opts, :embed_dim)

    template = Nx.template({@batch, seq, embed}, :f32)
    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    key = Nx.Random.key(42)
    {input, _} = Nx.Random.uniform(key, shape: {@batch, seq, embed})

    output = predict_fn.(params, input)
    {model, params, output}
  end

  defp base_opts(overrides) do
    defaults = [
      embed_dim: @embed,
      hidden_size: @hidden,
      state_size: @state_size,
      num_layers: @num_layers
    ]

    Keyword.merge(defaults, overrides)
  end

  # ============================================================================
  # Inference mode (ssd_scan) paths
  # ============================================================================

  describe "inference mode - single chunk (seq_len <= chunk_size)" do
    test "seq_len fits in one chunk, uses Common.sequential_scan" do
      # seq_len=4 <= chunk_size=16 and chunk_len <= 4 -> sequential_scan
      opts = base_opts(seq_len: 4, chunk_size: 16, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "seq_len fits in one chunk but > 4, uses blelloch_scan" do
      # seq_len=8 <= chunk_size=16 but chunk_len > 4 -> blelloch_scan
      opts = base_opts(seq_len: 8, chunk_size: 16, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "inference mode - multi-chunk (seq_len > chunk_size)" do
    test "evenly divisible, no remainder" do
      # seq_len=8, chunk_size=4 -> 2 full chunks, 0 remainder
      opts = base_opts(seq_len: 8, chunk_size: 4, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with remainder" do
      # seq_len=10, chunk_size=4 -> 2 full chunks + 2 remainder
      opts = base_opts(seq_len: 10, chunk_size: 4, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "3+ chunks triggers inter-chunk propagation with non-adjacent products" do
      # seq_len=12, chunk_size=4 -> 3 full chunks
      # The 3rd chunk's inter-chunk propagation needs products spanning chunk 0->2
      opts = base_opts(seq_len: 12, chunk_size: 4, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "many chunks with remainder exercises full inter-chunk paths" do
      # seq_len=14, chunk_size=4 -> 3 full chunks + 2 remainder
      opts = base_opts(seq_len: 14, chunk_size: 4, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "intra_chunk_scan uses sequential for very small chunks" do
      # chunk_size=3 (<= 4) -> intra_chunk_scan will use Common.sequential_scan
      # seq_len=6, chunk_size=3 -> 2 full chunks
      opts = base_opts(seq_len: 6, chunk_size: 3, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "intra_chunk_scan uses blelloch for larger chunks" do
      # chunk_size=5 (> 4) -> intra_chunk_scan will use Common.blelloch_scan
      # seq_len=10, chunk_size=5 -> 2 full chunks
      opts = base_opts(seq_len: 10, chunk_size: 5, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Training mode (ssd_matmul_scan) paths
  # ============================================================================

  describe "training mode - single chunk (seq_len <= chunk_size)" do
    test "small seq uses ssd_matmul_chunk directly" do
      opts = base_opts(seq_len: 4, chunk_size: 16, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "seq_len == chunk_size stays in single chunk" do
      opts = base_opts(seq_len: 8, chunk_size: 8, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "training mode - multi-chunk (chunked_ssd_matmul)" do
    test "evenly divisible, no remainder" do
      # seq_len=8, chunk_size=4 -> 2 full chunks, 0 remainder
      opts = base_opts(seq_len: 8, chunk_size: 4, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with remainder exercises remainder branch" do
      # seq_len=10, chunk_size=4 -> 2 full chunks + 2 remainder
      opts = base_opts(seq_len: 10, chunk_size: 4, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "3+ chunks exercises full inter-chunk propagation and compute_inter_chunk_products" do
      # seq_len=12, chunk_size=4 -> 3 full chunks
      # Chunk idx=2 will reference chunks 0 and 1, exercising
      # compute_inter_chunk_products with both adjacent (products == []) and
      # non-adjacent (products != []) cases
      opts = base_opts(seq_len: 12, chunk_size: 4, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "4 chunks with remainder hits all inter-chunk code paths" do
      # seq_len=14, chunk_size=3 -> 4 full chunks + 2 remainder
      # Exercises: remainder branch for last chunk in both chunk_outputs and
      # inter-chunk reduce, compute_inter_chunk_products with products spanning
      # multiple chunks (products list is non-empty), and the remainder check
      # inside inter-chunk propagation
      opts = base_opts(seq_len: 14, chunk_size: 3, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "many chunks exercises non-adjacent inter-chunk products" do
      # seq_len=20, chunk_size=4 -> 5 full chunks, 0 remainder
      # Chunk idx=4 references chunks 0-3, lots of inter-chunk products
      opts = base_opts(seq_len: 20, chunk_size: 4, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Multi-layer tests (exercises Common.build_model residual/dropout paths)
  # ============================================================================

  describe "multi-layer configurations" do
    test "inference mode with 2 layers" do
      opts = base_opts(seq_len: 8, chunk_size: 4, num_layers: 2, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "training mode with 2 layers" do
      opts = base_opts(seq_len: 8, chunk_size: 4, num_layers: 2, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "with dropout between layers" do
      opts =
        base_opts(
          seq_len: 8,
          chunk_size: 4,
          num_layers: 3,
          dropout: 0.1,
          training_mode: false
        )

      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "embed_dim != hidden_size triggers input projection" do
      opts =
        base_opts(
          embed_dim: 12,
          hidden_size: @hidden,
          seq_len: 8,
          chunk_size: 4,
          training_mode: false
        )

      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end
  end

  # ============================================================================
  # Determinism and numerical stability
  # ============================================================================

  describe "determinism" do
    test "same seed produces same output in inference mode" do
      opts = base_opts(seq_len: 8, chunk_size: 4, training_mode: false)
      model = MambaSSD.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = Nx.template({@batch, 8, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 8, @embed})

      out1 = predict_fn.(params, input)
      out2 = predict_fn.(params, input)

      assert Nx.all(Nx.equal(out1, out2)) |> Nx.to_number() == 1
    end

    test "same seed produces same output in training mode" do
      opts = base_opts(seq_len: 8, chunk_size: 4, training_mode: true)
      model = MambaSSD.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = Nx.template({@batch, 8, @embed}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 8, @embed})

      out1 = predict_fn.(params, input)
      out2 = predict_fn.(params, input)

      assert Nx.all(Nx.equal(out1, out2)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Utility functions
  # ============================================================================

  describe "param_count/1" do
    test "returns positive integer" do
      count = MambaSSD.param_count(embed_dim: 64, hidden_size: 32, num_layers: 2)
      assert is_integer(count)
      assert count > 0
    end

    test "scales with num_layers" do
      count1 = MambaSSD.param_count(embed_dim: 32, hidden_size: 32, num_layers: 1)
      count2 = MambaSSD.param_count(embed_dim: 32, hidden_size: 32, num_layers: 2)
      assert count2 > count1
    end

    test "no input_proj when embed_dim == hidden_size" do
      count = MambaSSD.param_count(embed_dim: 32, hidden_size: 32, num_layers: 1)
      # When embed_dim == hidden_size, input_proj should be 0
      # Just verify it returns a positive integer
      assert is_integer(count)
      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MambaSSD.recommended_defaults()
      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :chunk_size)
      assert Keyword.has_key?(defaults, :training_mode)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert defaults[:training_mode] == false
      assert defaults[:chunk_size] == 16
    end
  end

  describe "training_defaults/0" do
    test "returns keyword list with training-specific values" do
      defaults = MambaSSD.training_defaults()
      assert is_list(defaults)
      assert defaults[:training_mode] == true
      assert defaults[:chunk_size] == 32
    end
  end

  describe "output_size/1" do
    test "returns hidden_size from opts" do
      assert MambaSSD.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      size = MambaSSD.output_size()
      assert is_integer(size)
      assert size > 0
    end
  end

  # ============================================================================
  # Edge cases
  # ============================================================================

  describe "edge cases" do
    test "chunk_size=2 forces maximum chunking" do
      # Each chunk has 2 positions, maximizing inter-chunk propagation
      # chunk_len=2 <= 4 -> sequential_scan in intra_chunk_scan
      opts = base_opts(seq_len: 8, chunk_size: 2, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "chunk_size=2 in training mode" do
      opts = base_opts(seq_len: 8, chunk_size: 2, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "chunk_size=2 exercises small chunk intra_chunk_scan" do
      # 2 <= 4, so intra_chunk_scan uses sequential
      opts = base_opts(seq_len: 6, chunk_size: 2, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "seq_len=1 minimal sequence" do
      opts = base_opts(seq_len: 1, chunk_size: 4, training_mode: false)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end

    test "seq_len=1 in training mode" do
      opts = base_opts(seq_len: 1, chunk_size: 4, training_mode: true)
      {_model, _params, output} = build_and_run(opts)

      assert Nx.shape(output) == {@batch, @hidden}
    end
  end
end
