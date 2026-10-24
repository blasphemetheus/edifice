defmodule Edifice.CUDA.RLABackwardTest do
  use ExUnit.Case, async: true
  alias Edifice.CUDA.FusedScan

  for variant <- [:rla, :rdn], threshold <- [0.05, 1.0] do
    test "#{variant} backward matches finite differences with clip #{threshold}" do
      variant = unquote(variant)
      threshold = unquote(threshold)
      Process.put(:__edifice_force_fallback__, true)

      vectors =
        for scale <- [0.2, 0.3, 0.4],
            do: Nx.multiply(Nx.add(Nx.iota({1, 3, 1, 2}, type: :f32), 1), scale)

      gates = for value <- [0.7, 0.2, 0.1], do: Nx.broadcast(value, {1, 3, 1})
      inputs = vectors ++ gates
      # Only the last output contributes: earlier inputs must receive gradients
      # through the recurrent states, including both retrieval paths in RDN.
      upstream = Nx.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[0.5, -0.2]]]])

      grads =
        apply(FusedScan, :rla_backward_fallback, inputs ++ [upstream, variant, threshold])
        |> Tuple.to_list()

      objective = fn args ->
        apply(FusedScan, :rla_scan, args ++ [[variant: variant, clip_threshold: threshold]])
        |> Nx.multiply(upstream)
        |> Nx.sum()
        |> Nx.to_number()
      end

      for {input, arg} <- Enum.with_index(inputs), index <- 0..(Nx.size(input) - 1) do
        direction = Nx.equal(Nx.iota({Nx.size(input)}), index) |> Nx.reshape(Nx.shape(input))
        delta = Nx.multiply(direction, 0.001)
        plus = List.replace_at(inputs, arg, Nx.add(input, delta))
        minus = List.replace_at(inputs, arg, Nx.subtract(input, delta))
        numerical = (objective.(plus) - objective.(minus)) / 0.002
        actual = Enum.at(Nx.to_flat_list(Enum.at(grads, arg)), index)
        assert_in_delta actual, numerical, 0.0002
      end
    end
  end
end
