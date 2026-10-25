defmodule Edifice.Stateful.Replay do
  @moduledoc """
  Compiled replay of a nonempty `[batch, time, features]` chunk into carry.
  A chronological tuple of `[batch, features]` tensors is stacked in the graph.

  Returns the final state; outputs are deliberately discarded for cold-start
  padding and window resynchronization. The loop is compiled once, rather than
  unrolling the graph or dispatching a separate executable for each timestep.
  """
  import Nx.Defn

  @doc false
  defn run(params, state, frames, opts) do
    frames = stack_frames(frames)

    {_, state, _, _} =
      while {i = 0, state, params, frames}, i < Nx.axis_size(frames, 1) do
        frame = frames |> Nx.slice_along_axis(i, 1, axis: 1) |> Nx.squeeze(axes: [1])
        {_, next_state} = advance(params, state, frame, opts)
        {i + 1, next_state, params, frames}
      end

    state
  end

  @doc false
  defn run_rows(params, current, initial, frames, rows, count, opts) do
    frames = stack_frames(frames)
    selected = map_state(initial, fn leaf -> Nx.take(leaf, rows, axis: 0) end)
    frames = Nx.take(frames, rows, axis: 0)
    replayed = run(params, selected, frames, opts)
    merge_rows(current, replayed, rows, Nx.iota(Nx.shape(rows)) < count)
  end

  deftransformp(stack_frames(%Nx.Tensor{} = frames), do: frames)

  deftransformp(stack_frames(frames) when is_tuple(frames) and tuple_size(frames) > 0,
    do: Nx.stack(Tuple.to_list(frames), axis: 1)
  )

  deftransformp(map_state(%Nx.Tensor{} = leaf, fun), do: fun.(leaf))

  deftransformp(map_state(state, fun),
    do: Map.new(state, fn {key, value} -> {key, map_state(value, fun)} end)
  )

  deftransformp merge_rows(%Nx.Tensor{} = current, replayed, rows, active) do
    shape = List.to_tuple([Nx.axis_size(rows, 0) | List.duplicate(1, Nx.rank(current) - 1)])
    active = Nx.broadcast(Nx.reshape(active, shape), Nx.shape(replayed))
    updates = Nx.select(active, replayed, Nx.take(current, rows, axis: 0))
    Nx.indexed_put(current, Nx.new_axis(rows, 1), updates, axes: [0])
  end

  deftransformp(merge_rows(current, replayed, rows, active),
    do:
      Map.new(current, fn {key, value} ->
        {key, merge_rows(value, replayed[key], rows, active)}
      end)
  )

  deftransformp advance(params, state, frame, opts) do
    module = Keyword.fetch!(opts, :module)
    module.step(params, state, frame)
  end
end
