defmodule Edifice.Display.TrainingPlot do
  @moduledoc """
  Training curve visualization via VegaLite.

  Provides real-time training plots in Livebook and static chart
  generation for loss curves, metric histories, and learning rate
  schedules.

  ## Real-time in Livebook

      widget = Edifice.Display.TrainingPlot.live_widget()

      loop =
        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> Edifice.Display.TrainingPlot.attach(widget)

      Axon.Loop.run(loop, data, %{}, epochs: 10)

  ## Static Charts

      # From collected metrics
      Edifice.Display.TrainingPlot.loss_curve(metrics_history)
      Edifice.Display.TrainingPlot.lr_schedule(schedule_fn, total_steps: 1000)

  ## Requirements

  Requires `kino_vega_lite` (dev dependency). Functions gracefully
  return `:no_kino` when Kino is not available.
  """

  @doc """
  Create a live VegaLite widget for real-time training visualization.

  Returns a `Kino.VegaLite` widget. Display it in a Livebook cell,
  then pass it to `attach/2` to stream data during training.

  Returns `:no_kino` if Kino.VegaLite is not available.
  """
  def live_widget do
    if kino_vega_lite_available?() do
      spec = %{
        "$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
        "width" => 600,
        "height" => 300,
        "mark" => "line",
        "encoding" => %{
          "x" => %{"field" => "step", "type" => "quantitative"},
          "y" => %{"field" => "value", "type" => "quantitative"},
          "color" => %{"field" => "metric", "type" => "nominal"}
        }
      }

      apply(Kino.VegaLite, :new, [spec])
    else
      :no_kino
    end
  end

  @doc """
  Attach real-time training plots to an Axon.Loop.

  Streams loss and metric values to a `Kino.VegaLite` widget during
  training. Call `live_widget/0` first to create the widget.

  ## Parameters

    * `loop` - Axon.Loop to attach to
    * `widget` - Kino.VegaLite widget from `live_widget/0`

  ## Options

    * `:metrics` - List of metric names to plot (default: `["loss"]`)
    * `:every` - Plot every N steps (default: 1)
  """
  def attach(loop, widget, opts \\ [])

  def attach(loop, :no_kino, _opts), do: loop

  def attach(loop, widget, opts) do
    metrics = Keyword.get(opts, :metrics, ["loss"])
    every = Keyword.get(opts, :every, 1)

    Axon.Loop.handle_event(loop, :iteration_completed, fn state ->
      step = state.iteration

      if rem(step, every) == 0 do
        Enum.each(metrics, fn metric_name ->
          value =
            case metric_name do
              "loss" ->
                loss = get_in(state.step_state, [:loss])
                if loss, do: Nx.to_number(loss), else: nil

              name ->
                case get_in(state.metrics, [name]) do
                  nil -> nil
                  %Nx.Tensor{} = t -> Nx.to_number(t)
                  v when is_number(v) -> v
                  _ -> nil
                end
            end

          if value do
            apply(Kino.VegaLite, :push, [widget, %{
              "step" => step,
              "value" => value,
              "metric" => metric_name
            }])
          end
        end)
      end

      {:continue, state}
    end)
  end

  @doc """
  Generate a static loss curve from a list of `{step, loss}` tuples.

  Returns a VegaLite spec (map) suitable for rendering in Livebook
  or exporting as JSON.

  ## Examples

      data = [{0, 2.3}, {100, 1.5}, {200, 0.8}, {300, 0.4}]
      Edifice.Display.TrainingPlot.loss_curve(data)
  """
  def loss_curve(data) when is_list(data) do
    values = Enum.map(data, fn {step, loss} -> %{"step" => step, "loss" => loss} end)

    %{
      "$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
      "width" => 600,
      "height" => 300,
      "title" => "Training Loss",
      "data" => %{"values" => values},
      "mark" => %{"type" => "line", "point" => true},
      "encoding" => %{
        "x" => %{"field" => "step", "type" => "quantitative", "title" => "Step"},
        "y" => %{"field" => "loss", "type" => "quantitative", "title" => "Loss"}
      }
    }
  end

  @doc """
  Visualize a learning rate schedule.

  Evaluates the schedule function at each step and returns a VegaLite spec.

  ## Options

    * `:total_steps` - Number of steps to plot (default: 1000)
    * `:sample_every` - Sample every N steps (default: 10)
  """
  def lr_schedule(schedule_fn, opts \\ []) when is_function(schedule_fn, 1) do
    total_steps = Keyword.get(opts, :total_steps, 1000)
    sample_every = Keyword.get(opts, :sample_every, 10)

    values =
      0..total_steps//sample_every
      |> Enum.map(fn step ->
        lr = schedule_fn.(Nx.tensor(step)) |> Nx.to_number()
        %{"step" => step, "lr" => lr}
      end)

    %{
      "$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
      "width" => 600,
      "height" => 200,
      "title" => "Learning Rate Schedule",
      "data" => %{"values" => values},
      "mark" => "line",
      "encoding" => %{
        "x" => %{"field" => "step", "type" => "quantitative", "title" => "Step"},
        "y" => %{"field" => "lr", "type" => "quantitative", "title" => "Learning Rate"}
      }
    }
  end

  defp kino_vega_lite_available? do
    Code.ensure_loaded?(Kino.VegaLite)
  end
end
