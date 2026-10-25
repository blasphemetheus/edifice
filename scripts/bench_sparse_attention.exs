# Run with the desired Nx/EXLA backend configured, e.g. mix run this file.
# Compares gathered execution against dense execution of the SAME corrected mask.
defmodule SparseAttentionBench do
  alias Edifice.Attention.{NSA, SparseAttention}

  def dense(q, k, v, mask) do
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(:math.sqrt(Nx.axis_size(q, 3)))
    Nx.dot(SparseAttention.masked_softmax(scores, mask), [3], [0, 1], v, [2], [0, 1])
  end

  def selected_mask(blocks, time, size) do
    token_blocks = Nx.quotient(Nx.iota({time}), size)
    selected = Nx.equal(Nx.new_axis(blocks, -1), token_blocks) |> Nx.any(axes: [-2])
    causal = Nx.greater_equal(Nx.iota({time, 1}, axis: 0), Nx.iota({1, time}, axis: 1))
    Nx.logical_and(selected, causal)
  end

  def run(q, k, v, architecture, execution) do
    size = 16
    topk = 2
    window = 32
    compression = 8
    time = Nx.axis_size(q, 2)

    {compressed, blocks} =
      case architecture do
        :moba ->
          {nil, SparseAttention.route(q, k, size, topk, true)}

        :nsa ->
          {output, probabilities} = SparseAttention.compressed(q, k, v, compression)
          scores = SparseAttention.compression_scores(probabilities, time, compression, size)
          {output, SparseAttention.select_blocks(scores, size, topk, true)}
      end

    selected =
      case execution do
        :gathered -> SparseAttention.selected(q, k, v, blocks, size, true)
        :dense -> dense(q, k, v, selected_mask(blocks, time, size))
      end

    case architecture do
      :moba ->
        selected

      :nsa ->
        local =
          case execution do
            :gathered ->
              SparseAttention.window(q, k, v, window)

            :dense ->
              distance = Nx.subtract(Nx.iota({time, 1}, axis: 0), Nx.iota({1, time}, axis: 1))

              dense(
                q,
                k,
                v,
                Nx.logical_and(Nx.greater_equal(distance, 0), Nx.less(distance, window))
              )
          end

        Nx.divide(Nx.add(Nx.add(compressed, selected), local), 3)
    end
  end

  def measure(fun, args) do
    output = apply(Nx.Defn.jit(fun), args)
    Nx.to_binary(output)
    compiled = Nx.Defn.jit(fn q, k, v -> Nx.sum(fun.(q, k, v)) end)
    for _ <- 1..5, do: Nx.to_binary(apply(compiled, args))

    samples =
      for _ <- 1..30 do
        {us, _} = :timer.tc(fn -> Nx.to_binary(apply(compiled, args)) end)
        us
      end
      |> Enum.sort()

    {output, %{median_us: Enum.at(samples, 15), p95_us: Enum.at(samples, 28)}}
  end

  def main do
    results =
      for time <- [64, 256, 1024, 4096], architecture <- [:moba, :nsa] do
        {q, key} = Nx.Random.normal(Nx.Random.key(42), shape: {1, 4, time, 32})
        {k, key} = Nx.Random.normal(key, shape: {1, 4, time, 32})
        {v, _} = Nx.Random.normal(key, shape: {1, 4, time, 32})
        {a, gathered} = measure(&run(&1, &2, &3, architecture, :gathered), [q, k, v])
        {b, dense} = measure(&run(&1, &2, &3, architecture, :dense), [q, k, v])

        unless Nx.to_number(Nx.all_close(a, b, atol: 1.0e-5, rtol: 1.0e-5)) == 1,
          do:
            raise(
              "dense parity failed: #{architecture} T=#{time} max_error=#{Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(a, b))))}"
            )

        if architecture == :nsa do
          direct =
            Nx.Defn.jit(fn q, k, v ->
              NSA.attention(q, k, v, Nx.tensor([0.0, 0, 0]),
                block_size: 16,
                num_selected_blocks: 2,
                window_size: 32,
                compression_ratio: 8
              )
            end).(q, k, v)

          unless Nx.to_number(Nx.all_close(a, direct, atol: 1.0e-5, rtol: 1.0e-5)) == 1,
            do: raise("NSA benchmark drift")
        end

        row = %{
          architecture: architecture,
          time: time,
          gathered: gathered,
          dense_same_mask: dense,
          parity: true
        }

        IO.inspect(row)
        row
      end

    if path = System.get_env("SPARSE_OUT") do
      File.write!(
        path,
        Jason.encode!(
          %{
            backend: inspect(Nx.default_backend()),
            precision: :highest,
            shape: "B=1 H=4 D=32",
            block_size: 16,
            topk: 2,
            window: 32,
            compression: 8,
            samples: 30,
            timing:
              "Resident inputs; compiled output reduction and scalar host synchronization; warmup/compilation excluded",
            results: results
          },
          pretty: true
        )
      )
    end
  end
end

SparseAttentionBench.main()
