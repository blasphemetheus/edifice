# AI-generated diagnostic; attribution and poem: tools/whisper_reference/README.md.
defmodule WhisperF64Validation do
  @moduledoc "Keeps every operation in f64 for batch-one Whisper-base validation."
  def model(encoder) do
    # The production attention helper intentionally computes softmax in f32.
    # For this diagnostic, retain f64 through every attention operation as well.
    nodes =
      Map.new(encoder.nodes, fn {id, node} ->
        if String.ends_with?(node.name.(node.op_name, %{}), "_attn_compute") do
          attention = fn q, k, v, _opts ->
            split = fn x -> Nx.transpose(Nx.reshape(x, {1, 1500, 8, 64}), axes: [0, 2, 1, 3]) end
            q = split.(q)
            k = split.(k)
            v = split.(v)
            scores = Nx.dot(Nx.divide(q, 8), [3], [0, 1], k, [3], [0, 1])
            exps = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
            weights = Nx.divide(exps, Nx.sum(exps, axes: [-1], keep_axes: true))

            Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
            |> Nx.transpose(axes: [0, 2, 1, 3])
            |> Nx.reshape({1, 1500, 512})
          end

          {id, %{node | op: attention}}
        else
          {id, node}
        end
      end)

    policy =
      Axon.MixedPrecision.create_policy(compute: {:f, 64}, params: {:f, 64}, output: {:f, 64})

    Axon.MixedPrecision.apply_policy(%{encoder | nodes: nodes}, policy)
  end
end
