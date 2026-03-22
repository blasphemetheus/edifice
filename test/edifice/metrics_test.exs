defmodule Edifice.MetricsTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO

  alias Edifice.Metrics

  describe "classification_report/3" do
    test "computes per-class precision, recall, f1" do
      # 3 classes, 6 samples
      y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
      y_pred = Nx.tensor([0, 1, 1, 1, 2, 0])

      output =
        capture_io(fn ->
          report = Metrics.classification_report(y_true, y_pred, num_classes: 3)

          assert is_map(report)
          assert is_float(report.accuracy)
          assert report.accuracy > 0
          assert Nx.shape(report.precision) == {3}
          assert Nx.shape(report.recall) == {3}
          assert Nx.shape(report.f1) == {3}
          assert Nx.shape(report.confusion_matrix) == {3, 3}
        end)

      assert output =~ "Classification Report"
      assert output =~ "Accuracy:"
    end

    test "handles probability predictions" do
      y_true = Nx.tensor([0, 1, 2])
      y_pred = Nx.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])

      output =
        capture_io(fn ->
          report = Metrics.classification_report(y_true, y_pred, num_classes: 3)
          assert report.accuracy == 1.0
        end)

      assert output =~ "100.0%"
    end
  end

  describe "confusion_matrix/3" do
    test "computes confusion matrix" do
      y_true = Nx.tensor([0, 0, 1, 1])
      y_pred = Nx.tensor([0, 1, 0, 1])

      cm = Metrics.confusion_matrix(y_true, y_pred, num_classes: 2)

      assert Nx.shape(cm) == {2, 2}
      # Diagonal: correct predictions
      assert Nx.to_number(cm[0][0]) == 1
      assert Nx.to_number(cm[1][1]) == 1
    end
  end
end
