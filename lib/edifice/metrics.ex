defmodule Edifice.Metrics do
  @moduledoc """
  Extended metrics for model evaluation via Scholar.

  Provides classification and regression metrics beyond what Axon.Metrics
  offers. Requires the optional `scholar` dependency.

  ## Usage

      # After training
      predictions = predict_fn.(params, test_data)
      targets = test_labels

      Edifice.Metrics.classification_report(targets, predictions, num_classes: 10)
      Edifice.Metrics.confusion_matrix(targets, predictions, num_classes: 10)

  ## Requirements

  Add `{:scholar, "~> 0.4", optional: true}` to your deps.
  """

  @doc """
  Compute a classification report with per-class precision, recall, and F1.

  ## Parameters

    * `y_true` - Ground truth labels `[n]` (integer class indices)
    * `y_pred` - Predicted probabilities `[n, num_classes]` or class indices `[n]`

  ## Options

    * `:num_classes` - Number of classes (required)

  ## Returns

    A map with `:precision`, `:recall`, `:f1` (per-class tensors) and
    `:accuracy` (scalar).
  """
  def classification_report(y_true, y_pred, opts \\ []) do
    ensure_scholar!()
    num_classes = Keyword.fetch!(opts, :num_classes)

    # Convert probabilities to class indices if needed
    pred_indices = to_class_indices(y_pred)
    true_indices = to_class_indices(y_true)

    cm = apply(Scholar.Metrics.Classification, :confusion_matrix, [true_indices, pred_indices, [num_classes: num_classes]])

    # Per-class metrics from confusion matrix
    tp = Nx.take_diagonal(cm)
    fp = Nx.subtract(Nx.sum(cm, axes: [0]), tp)
    fn_ = Nx.subtract(Nx.sum(cm, axes: [1]), tp)

    precision = Nx.divide(tp, Nx.add(Nx.add(tp, fp), 1.0e-8))
    recall = Nx.divide(tp, Nx.add(Nx.add(tp, fn_), 1.0e-8))
    f1 = Nx.divide(Nx.multiply(2.0, Nx.multiply(precision, recall)),
                    Nx.add(Nx.add(precision, recall), 1.0e-8))

    total = Nx.size(true_indices)
    correct = Nx.sum(Nx.equal(true_indices, pred_indices)) |> Nx.to_number()
    accuracy = correct / total

    report = %{
      precision: precision,
      recall: recall,
      f1: f1,
      accuracy: accuracy,
      confusion_matrix: cm
    }

    # Print summary
    IO.puts("\n[Metrics] Classification Report (#{num_classes} classes)")
    IO.puts("  Accuracy: #{Float.round(accuracy * 100, 1)}%")

    for c <- 0..(num_classes - 1) do
      p = precision[c] |> Nx.to_number() |> Float.round(3)
      r = recall[c] |> Nx.to_number() |> Float.round(3)
      f = f1[c] |> Nx.to_number() |> Float.round(3)
      IO.puts("  Class #{c}: precision=#{p} recall=#{r} f1=#{f}")
    end

    IO.puts("")

    report
  end

  @doc """
  Compute confusion matrix.

  ## Options

    * `:num_classes` - Number of classes (required)
  """
  def confusion_matrix(y_true, y_pred, opts \\ []) do
    ensure_scholar!()
    num_classes = Keyword.fetch!(opts, :num_classes)

    pred_indices = to_class_indices(y_pred)
    true_indices = to_class_indices(y_true)

    apply(Scholar.Metrics.Classification, :confusion_matrix, [true_indices, pred_indices, [num_classes: num_classes]])
  end

  defp to_class_indices(tensor) do
    case Nx.rank(tensor) do
      1 -> Nx.as_type(tensor, :s32)
      2 -> Nx.argmax(tensor, axis: -1) |> Nx.as_type(:s32)
      _ -> Nx.argmax(tensor, axis: -1) |> Nx.as_type(:s32)
    end
  end

  defp ensure_scholar! do
    unless Code.ensure_loaded?(Scholar.Metrics.Classification) do
      raise RuntimeError,
            "Edifice.Metrics requires Scholar. Add {:scholar, \"~> 0.4\", optional: true} to your deps."
    end
  end
end
