defmodule Edifice.Interpretability.Diffing do
  @moduledoc """
  Stage-wise model diffing over SAE dictionaries (Anthropic, Dec 2024;
  adopted per exphil INTERP_NEXT_RESEARCH_2026-07-20).

  Protocol: fit an SAE on model A's activations
  (`SAETrainer.fit/3`), fine-tune it briefly on model B's
  (`SAETrainer.finetune/3`), then rank features by how far their
  decoder directions moved. Features that moved most are what B learned
  or unlearned relative to A — at a fraction of a full crosscoder's
  cost, reusing the existing dictionary.

  Decoder rows are unit-normed by the trainer (renorm every step), so
  row-wise dot products ARE cosines.
  """

  @doc """
  Per-feature decoder-direction shift between two parameter states of
  the SAME build (e.g. before/after `SAETrainer.finetune/3`).

  Returns a list sorted by ascending cosine (most-moved first):
  `[%{feature: i, cosine: c}, ...]`. Multiple decoder layers
  (transcoders) contribute the MINIMUM cosine per feature index.
  """
  def decoder_shift(%Axon.ModelState{data: a}, %Axon.ModelState{data: b}) do
    keys = decoder_keys(a)

    if keys == [], do: raise(ArgumentError, "no decoder layers in params")

    per_key =
      Enum.map(keys, fn key ->
        ka = a[key]["kernel"] |> Nx.as_type(:f32)
        kb = b[key]["kernel"] |> Nx.as_type(:f32)

        if Nx.shape(ka) != Nx.shape(kb) do
          raise ArgumentError,
                "decoder #{key} shape mismatch: #{inspect(Nx.shape(ka))} vs #{inspect(Nx.shape(kb))}"
        end

        # Rows are unit-norm — renormalize defensively, then row-dot
        cos = Nx.sum(Nx.multiply(unit_rows(ka), unit_rows(kb)), axes: [1])
        Nx.to_flat_list(cos)
      end)

    per_key
    |> Enum.zip()
    |> Enum.with_index()
    |> Enum.map(fn {cosines, i} ->
      %{feature: i, cosine: cosines |> Tuple.to_list() |> Enum.min() |> Float.round(4)}
    end)
    |> Enum.sort_by(& &1.cosine)
  end

  @doc """
  Features whose decoder direction moved beyond `1 - min_cosine`
  (default 0.05, i.e. cosine < 0.95) — the "what did B learn?" set.
  """
  def moved_features(shift, min_cosine \\ 0.95) do
    Enum.filter(shift, &(&1.cosine < min_cosine))
  end

  @doc """
  Per-feature firing-rate delta between two fitted results on their own
  activation sets: `rate_b - rate_a`, `{dict_size}` tensor. A feature
  that moved AND fires much more in B is a strong candidate for
  "B-exclusive"; pair with `decoder_shift/2`.

  Takes `{result, activations}` pairs (result from `SAETrainer.fit/3` /
  `finetune/3`).
  """
  def firing_shift({result_a, acts_a}, {result_b, acts_b}) do
    Nx.subtract(firing_rate(result_b, acts_b), firing_rate(result_a, acts_a))
  end

  defp firing_rate(%{model: model, params: params}, acts) do
    {_init, predict_fn} = Axon.build(model, mode: :inference)
    input_key = model |> Axon.get_inputs() |> Map.keys() |> hd()

    out = predict_fn.(params, %{input_key => Nx.as_type(acts, :f32)})
    out.hidden |> Nx.greater(0.0) |> Nx.as_type(:f32) |> Nx.mean(axes: [0])
  end

  defp unit_rows(kernel) do
    norms =
      kernel
      |> Nx.pow(2)
      |> Nx.sum(axes: [1], keep_axes: true)
      |> Nx.sqrt()
      |> Nx.max(1.0e-8)

    Nx.divide(kernel, norms)
  end

  defp decoder_keys(data) do
    data |> Map.keys() |> Enum.filter(&String.ends_with?(&1, "decoder")) |> Enum.sort()
  end
end
