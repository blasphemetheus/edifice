defmodule Edifice.Stateful.Replay do
  @moduledoc """
  Compiled replay of a nonempty `[batch, time, features]` chunk into carry.

  Returns the final state; outputs are deliberately discarded for cold-start
  padding and window resynchronization. The loop is compiled once, rather than
  unrolling the graph or dispatching a separate executable for each timestep.
  """
  import Nx.Defn

  @doc false
  defn run(params, state, frames, opts) do
    {_, state, _, _} =
      while {i = 0, state, params, frames}, i < Nx.axis_size(frames, 1) do
        frame = frames |> Nx.slice_along_axis(i, 1, axis: 1) |> Nx.squeeze(axes: [1])
        {_, next_state} = advance(params, state, frame, opts)
        {i + 1, next_state, params, frames}
      end

    state
  end

  deftransformp advance(params, state, frame, opts) do
    module = Keyword.fetch!(opts, :module)
    module.step(params, state, frame)
  end
end
