defmodule Edifice.Interpretability.Probe do
  @moduledoc """
  Linear probing harness: is a feature linearly decodable from frozen
  activations?

  Ported from exphil's probe machinery (the architecture-generic core;
  Melee-specific suites, replay splitting, and lead-time curves stay in
  exphil). Implements the audit-mandated probe hygiene
  (INTERP_AUDIT_2026-07-15, remediation step 3):

  - **Balanced accuracy** (mean per-class recall) as the headline metric —
    plain accuracy rewards predicting the majority class on imbalanced
    targets.
  - **Class-weighted cross-entropy** (inverse frequency) from raw logits
    with the max-shift log-sum-exp trick — no baked-in softmax.
  - **Z-score standardization with train-set statistics only** (no eval
    leakage).
  - **Label masking**: rows labeled `-1` are dropped (e.g. "feature not
    applicable here").
  - **Single-class guard**: with fewer than 2 classes present in eval,
    balanced accuracy is degenerate (predicting the only class scores 1.0)
    and is reported as `nil` instead.
  - **Majority baseline** reported alongside, so a probe can never look
    better than it is.
  - **Shuffled-label control** (`shuffled_control/6`, Hewitt-Liang style):
    a sound probe setup scores ~chance on shuffled train labels.
  - **f32 at entry** per the repo loss-precision policy.

  Training is full-batch gradient descent with momentum inside a single
  `defn` while-loop (the gradient is taken *inside* the loop body, not
  through the `while` — safe w.r.t. nx#1747).

  ## Performance note (from the exphil port)

  Pass `compiler: EXLA` for real workloads — the default
  `Nx.Defn.Evaluator` never touches an accelerator (safe everywhere) but
  is slow at scale. Just as important: the heavy ROW operations
  (`Nx.take` masking, standardization) run on the tensors' current
  backend, and BinaryBackend's single-threaded gather turns `{32k, 256}`
  takes into tens of CPU-minutes (observed in exphil 2026-07-14 — this was
  the unexplained "probe training burned 20+ CPU-min" issue: it wasn't the
  training loop, it was row gathers on BinaryBackend). Copy `x`/`y` to an
  accelerator backend first (`Nx.backend_copy(t, EXLA.Backend)`) for large
  captures. `fit_eval/6` reports `train_ms` so regressions here are visible.
  """

  import Nx.Defn

  @default_steps 300
  @default_lr 0.05
  @default_l2 1.0e-4

  @doc """
  Fit a linear probe on `{x, y}` and evaluate on `{x_eval, y_eval}`.

  ## Arguments

    - `x` / `x_eval` - `{n, d}` activations (cast to f32 at entry)
    - `y` / `y_eval` - `{n}` integer labels; `-1` rows are masked out
    - `num_classes` - number of classes (2 for binary features)

  ## Options

    - `:steps` - gradient steps (default #{@default_steps})
    - `:lr` - learning rate (default #{@default_lr})
    - `:l2` - L2 penalty on weights (default #{@default_l2})
    - `:compiler` - `Nx.Defn` compiler for the train/predict loops
      (default `Nx.Defn.Evaluator`; pass `EXLA` for real workloads)
    - `:seed` - RNG seed for weight init (default 42)

  ## Returns

  `%{balanced_accuracy, accuracy, majority_baseline, per_class_recall,
  n_train, n_eval, train_ms, params}` — all metric fields `nil` when a
  split is empty after masking.
  """
  def fit_eval(x, y, x_eval, y_eval, num_classes, opts \\ []) do
    steps = Keyword.get(opts, :steps, @default_steps)
    lr = Keyword.get(opts, :lr, @default_lr)
    l2 = Keyword.get(opts, :l2, @default_l2)
    compiler = Keyword.get(opts, :compiler, Nx.Defn.Evaluator)
    seed = Keyword.get(opts, :seed, 42)

    # f32 at entry per the repo loss-precision policy; labels to s64
    x = Nx.as_type(x, :f32)
    x_eval = Nx.as_type(x_eval, :f32)
    y = Nx.as_type(y, :s64)
    y_eval = Nx.as_type(y_eval, :s64)

    masked_train = mask_rows(x, y)
    masked_eval = mask_rows(x_eval, y_eval)

    if masked_train == :empty or masked_eval == :empty do
      %{
        balanced_accuracy: nil,
        accuracy: nil,
        majority_baseline: nil,
        per_class_recall: nil,
        n_train: if(masked_train == :empty, do: 0, else: Nx.axis_size(elem(masked_train, 0), 0)),
        n_eval: if(masked_eval == :empty, do: 0, else: Nx.axis_size(elem(masked_eval, 0), 0)),
        train_ms: nil,
        params: nil
      }
    else
      {x, y} = masked_train
      {x_eval, y_eval} = masked_eval
      n_train = Nx.axis_size(x, 0)
      n_eval = Nx.axis_size(x_eval, 0)
      # Standardize with train statistics only
      mean = Nx.mean(x, axes: [0], keep_axes: true)
      std = Nx.standard_deviation(x, axes: [0], keep_axes: true) |> Nx.max(1.0e-6)
      xs = Nx.divide(Nx.subtract(x, mean), std)
      xs_eval = Nx.divide(Nx.subtract(x_eval, mean), std)

      # Inverse-frequency class weights from the train labels
      counts = class_counts(y, num_classes)
      weights = Nx.divide(n_train / num_classes, Nx.max(counts, 1))

      d = Nx.axis_size(x, 1)

      key = Nx.Random.key(seed)
      {w, _} = Nx.Random.normal(key, 0.0, 0.01, shape: {d, num_classes}, type: :f32)
      b = Nx.broadcast(Nx.tensor(0.0, type: :f32), {num_classes})

      y_onehot = Nx.equal(Nx.new_axis(y, 1), Nx.iota({1, num_classes})) |> Nx.as_type(:f32)

      {train_us, {w, b}} =
        :timer.tc(fn ->
          Nx.Defn.jit_apply(
            fn xs, y_onehot, weights, w, b ->
              train_loop(xs, y_onehot, weights, w, b, steps: steps, lr: lr, l2: l2)
            end,
            [xs, y_onehot, weights, w, b],
            compiler: compiler
          )
        end)

      pred = Nx.Defn.jit_apply(&predict_n/3, [w, b, xs_eval], compiler: compiler)
      per_class = per_class_recall(pred, y_eval, num_classes)

      recalls = per_class |> Enum.reject(&is_nil/1)

      # Guard: with <2 classes present in eval, balanced accuracy is
      # degenerate (predicting the only class scores 1.0)
      balanced =
        if length(recalls) >= 2, do: Enum.sum(recalls) / length(recalls), else: nil

      %{
        balanced_accuracy: balanced,
        accuracy: Nx.mean(Nx.equal(pred, y_eval)) |> Nx.to_number(),
        majority_baseline: majority_baseline(y_eval, num_classes),
        per_class_recall: per_class,
        n_train: n_train,
        n_eval: n_eval,
        train_ms: Float.round(train_us / 1_000.0, 1),
        params: %{w: w, b: b, mean: mean, std: std}
      }
    end
  end

  @doc """
  Predict class labels for new activations using a fitted probe's `:params`.
  """
  def predict(%{w: w, b: b, mean: mean, std: std}, x, opts \\ []) do
    compiler = Keyword.get(opts, :compiler, Nx.Defn.Evaluator)
    xs = x |> Nx.as_type(:f32) |> Nx.subtract(mean) |> Nx.divide(std)
    Nx.Defn.jit_apply(&predict_n/3, [w, b, xs], compiler: compiler)
  end

  @doc """
  Hewitt-Liang-style control: fit on SHUFFLED train labels, evaluate on the
  true eval labels. A sound probe setup scores ~chance here; anything above
  chance means the probe (not the representation) is doing the work.
  """
  def shuffled_control(x, y, x_eval, y_eval, num_classes, opts \\ []) do
    shuffle_seed = Keyword.get(opts, :shuffle_seed, 7)

    n = Nx.axis_size(y, 0)
    :rand.seed(:exsss, {shuffle_seed, shuffle_seed, shuffle_seed})
    perm = 0..(n - 1) |> Enum.shuffle() |> Nx.tensor(type: :s64)
    y_shuffled = Nx.take(Nx.as_type(y, :s64), perm)

    fit_eval(x, y_shuffled, x_eval, y_eval, num_classes, opts)
  end

  @doc """
  Mean balanced accuracy across fit results (nil-safe). Accepts a list of
  result maps or a map of `%{feature => result}`.
  """
  def mean_balanced_accuracy(results) when is_map(results) and not is_struct(results) do
    results |> Map.values() |> mean_balanced_accuracy()
  end

  def mean_balanced_accuracy(results) when is_list(results) do
    scores =
      results
      |> Enum.map(& &1.balanced_accuracy)
      |> Enum.reject(&is_nil/1)

    Enum.sum(scores) / max(length(scores), 1)
  end

  # ============================================================================
  # Private — Nx internals
  # ============================================================================

  # Returns {x, y} with -1-labeled rows dropped, or :empty (nx 0.11 cannot
  # represent zero-size tensors)
  defp mask_rows(x, y) do
    case mask_to_indices(Nx.greater_equal(y, 0)) do
      :empty -> :empty
      keep -> {Nx.take(x, keep, axis: 0), Nx.take(y, keep, axis: 0)}
    end
  end

  defp mask_to_indices(mask) do
    n = Nx.axis_size(mask, 0)
    count = Nx.sum(mask) |> Nx.to_number()

    if count == 0 do
      :empty
    else
      mask
      |> Nx.as_type(:s64)
      |> Nx.multiply(Nx.iota({n}) |> Nx.add(1))
      |> Nx.to_flat_list()
      |> Enum.filter(&(&1 > 0))
      |> Enum.map(&(&1 - 1))
      |> Nx.tensor(type: :s64)
    end
  end

  defp class_counts(y, num_classes) do
    Nx.equal(Nx.new_axis(y, 1), Nx.iota({1, num_classes}))
    |> Nx.sum(axes: [0])
    |> Nx.as_type(:f32)
  end

  defp majority_baseline(y_eval, num_classes) do
    counts = class_counts(y_eval, num_classes)
    # Balanced accuracy of always predicting the majority class:
    # recall = 1 for that class, 0 elsewhere → 1/num_present
    present = Nx.greater(counts, 0) |> Nx.sum() |> Nx.to_number()
    1.0 / max(present, 1)
  end

  defp per_class_recall(pred, y, num_classes) do
    Enum.map(0..(num_classes - 1), fn c ->
      in_class = Nx.equal(y, c)
      total = Nx.sum(in_class) |> Nx.to_number()

      if total == 0 do
        nil
      else
        hit = Nx.logical_and(in_class, Nx.equal(pred, c)) |> Nx.sum() |> Nx.to_number()
        hit / total
      end
    end)
  end

  defnp predict_n(w, b, xs) do
    xs |> Nx.dot(w) |> Nx.add(b) |> Nx.argmax(axis: 1)
  end

  defnp train_loop(xs, y_onehot, weights, w0, b0, opts \\ []) do
    steps = opts[:steps]
    lr = opts[:lr]
    l2 = opts[:l2]

    {{w, b}, _} =
      while {{w = w0, b = b0}, {xs, y_onehot, weights, mw = w0 * 0.0, mb = b0 * 0.0}},
            _i <- 0..(steps - 1) do
        # grad INSIDE the while body (per-iteration), never through the
        # while — nx#1747 makes grad-of-while unreliable
        {gw, gb} = grad({w, b}, fn {w, b} -> ce_loss(w, b, xs, y_onehot, weights, l2) end)

        mw = 0.9 * mw + gw
        mb = 0.9 * mb + gb
        {{w - lr * mw, b - lr * mb}, {xs, y_onehot, weights, mw, mb}}
      end

    {w, b}
  end

  # Class-weighted cross-entropy from raw logits (max-shift log-sum-exp; no
  # softmax layer involved, so logit stability is preserved)
  defnp ce_loss(w, b, xs, y_onehot, weights, l2) do
    logits = Nx.dot(xs, w) + b
    max_l = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = logits - max_l
    log_probs = shifted - Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))

    row_w = Nx.sum(y_onehot * Nx.reshape(weights, {1, :auto}), axes: [1])
    nll = -Nx.sum(log_probs * y_onehot, axes: [1])

    Nx.sum(nll * row_w) / Nx.sum(row_w) + l2 * Nx.sum(w * w)
  end
end
