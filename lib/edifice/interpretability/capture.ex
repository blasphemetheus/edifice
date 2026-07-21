defmodule Edifice.Interpretability.Capture do
  @moduledoc """
  Generic activation capture for a frozen network (INTERP_AUDIT
  structural gap 3; the architecture-generic half of exphil's
  `ExPhil.Interp.Activations` — replay parsing, embeddings, labels, and
  caching stay with the consumer).

  Works with any model — single-output (a trunk) or an
  `Axon.container` of named sites (see `Edifice.SSM.MambaSSD.build_probe/1`).
  Batches stream through the frozen params; per-site outputs are cast
  to f32, copied to `Nx.BinaryBackend` per batch (accelerator buffers
  otherwise accumulate until BEAM GC — the exphil step-~3000 OOM
  lesson), and concatenated along axis 0.
  """

  @doc """
  Run `model` with `params` over an enumerable of input batches.

  Batches are whatever the model's `predict_fn` accepts (a bare tensor
  for single-input models, or an input map). Options:

    - `:only` - list of site names to keep (default: all)
    - `:compiler` - `Axon.build` compiler (default EXLA if loaded)

  Returns `%{site => tensor}` with sites concatenated across batches;
  single-output models yield `%{"output" => tensor}`.
  """
  def run(model, %Axon.ModelState{} = params, batches, opts \\ []) do
    only = Keyword.get(opts, :only)

    build_opts =
      case Keyword.get(opts, :compiler) do
        nil -> if Code.ensure_loaded?(EXLA), do: [compiler: EXLA], else: []
        c -> [compiler: c]
      end

    {_init_fn, predict_fn} = Axon.build(model, [mode: :inference] ++ build_opts)

    to_bin = fn t -> t |> Nx.as_type(:f32) |> Nx.backend_copy(Nx.BinaryBackend) end

    per_batch =
      Enum.map(batches, fn batch ->
        out = predict_fn.(params, batch)

        sites =
          case out do
            %Nx.Tensor{} = t -> %{"output" => t}
            %{} = m -> m
          end

        sites
        |> filter_sites(only)
        |> Map.new(fn {k, v} -> {k, to_bin.(v)} end)
      end)

    case per_batch do
      [] ->
        %{}

      [first | _] ->
        Map.new(first, fn {site, _} ->
          {site, per_batch |> Enum.map(& &1[site]) |> Nx.concatenate()}
        end)
    end
  end

  defp filter_sites(sites, nil), do: sites
  defp filter_sites(sites, only), do: Map.take(sites, only)
end
