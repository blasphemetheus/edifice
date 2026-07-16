defmodule Edifice.Interpretability.Attribution do
  @moduledoc """
  Input attribution (gradient × input): which input dimensions drove this
  prediction?

  Differentiates a model's chosen-class logit w.r.t. the input and
  aggregates `|grad × input|`, optionally into named dimension groups.
  Ported from exphil's attribution core (the architecture-generic part;
  exphil keeps its Melee head map and embedding-perturbation dim
  discovery).

  ## Usage

      {_init, predict_fn} = Axon.build(model, mode: :inference)

      sal = Attribution.saliency(predict_fn, params, inputs)
      # => {n, input_dim} — |grad × input| per row

      shares = Attribution.group_shares(sal, %{position: [0, 1], velocity: [2, 3]})
      # => %{position: {n} fraction, velocity: {n} fraction}

  For models whose output is a tuple/container of heads, pass `:select`
  to pick the logits tensor:

      Attribution.saliency(predict_fn, params, inputs, select: &elem(&1, 1))
  """

  @doc """
  Gradient × input saliency over a batch.

  The scalar objective per row is the logit at that row's argmax class,
  with the class choice held constant under `stop_grad` — "how sensitive
  is the decision the model actually made?".

  ## Arguments

    - `predict_fn` / `params` - from `Axon.build/2`
    - `inputs` - `{n, d}` or `{n, window, d}` input tensor

  ## Options

    - `:select` - function extracting the `{n, num_classes}` logits tensor
      from the model output (default: identity)
    - `:compiler` - `Nx.Defn` compiler (default `Nx.Defn.Evaluator`; pass
      `EXLA` for real workloads)

  ## Returns

  `{n, d}` — `|grad × input|`, summed over the window axis when the input
  is `{n, window, d}`.
  """
  def saliency(predict_fn, params, inputs, opts \\ []) do
    select = Keyword.get(opts, :select, & &1)
    compiler = Keyword.get(opts, :compiler, Nx.Defn.Evaluator)

    # params passed as a jit argument (closure-captured device tensors can
    # crash the trace); the argmax one-hot is computed INSIDE the trace
    # under stop_grad so the chosen class is a constant
    grads =
      Nx.Defn.jit_apply(
        fn p, s ->
          Nx.Defn.grad(s, fn s2 ->
            logits = select.(predict_fn.(p, s2))

            onehot =
              logits
              |> Nx.Defn.Kernel.stop_grad()
              |> Nx.argmax(axis: 1)
              |> Nx.new_axis(1)
              |> then(&Nx.equal(Nx.iota(Nx.shape(logits), axis: 1), &1))
              |> Nx.as_type(Nx.type(logits))

            Nx.sum(Nx.multiply(logits, onehot))
          end)
        end,
        [params, inputs],
        compiler: compiler
      )

    sal = Nx.abs(Nx.multiply(grads, inputs))

    case Nx.rank(sal) do
      2 -> sal
      3 -> Nx.sum(sal, axes: [1])
      r -> raise ArgumentError, "expected rank-2 or rank-3 inputs, got rank #{r}"
    end
  end

  @doc """
  Aggregate per-dim saliency `{n, d}` into per-group shares.

  `dim_groups` maps group names to lists of dim indices. Returns
  `%{group => {n} tensor}` — each group's fraction of total saliency per
  row. Groups need not partition the dims; shares sum to 1 only if they do.
  """
  def group_shares(sal, dim_groups) do
    total = Nx.sum(sal, axes: [1]) |> Nx.max(1.0e-9)

    Map.new(dim_groups, fn {name, idx} ->
      share =
        case idx do
          [] ->
            Nx.broadcast(0.0, Nx.shape(total))

          _ ->
            sal
            |> Nx.take(Nx.tensor(idx), axis: 1)
            |> Nx.sum(axes: [1])
            |> Nx.divide(total)
        end

      {name, share}
    end)
  end
end
