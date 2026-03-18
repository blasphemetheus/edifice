defmodule Edifice.Display.Heatmap do
  @moduledoc """
  Heatmap visualization for model weights, activations, and gradients.

  Wraps `Nx.to_heatmap/2` with model-aware utilities that make it easy
  to visualize parameter distributions, spot dead neurons, and diagnose
  vanishing/exploding gradients.

  ## Weight Visualization

      model = Edifice.build(:mlp, input_size: 16, hidden_sizes: [32], output_size: 8)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 16}, :f32), Axon.ModelState.empty())

      # Heatmap of a single weight matrix
      Edifice.Display.Heatmap.weights(params, "dense_0.kernel")

      # All weight matrices
      Edifice.Display.Heatmap.all_weights(params)

  ## Gradient Visualization

  Use with `Edifice.Training.Monitor` to capture gradients during training,
  then visualize:

      # After a training step
      Edifice.Display.Heatmap.gradients(grads)

  ## Terminal Support

  Heatmaps use ANSI 256-color backgrounds. Best viewed in terminals with
  256-color support (iTerm2, kitty, Alacritty, most modern terminals).
  Falls back to ASCII digits (0-9) when ANSI is unavailable.
  """

  require Logger

  import Nx.Defn

  # ============================================================================
  # Weight heatmaps
  # ============================================================================

  @doc """
  Display a heatmap of a specific parameter tensor.

  ## Parameters

    * `params` - Model parameters (map or `Axon.ModelState`)
    * `key_path` - Dot-separated path to the parameter (e.g., `"dense_0.kernel"`)

  ## Options

    * `:label` - Custom label (default: the key path)

  ## Examples

      Edifice.Display.Heatmap.weights(params, "dense_0.kernel")
  """
  def weights(params, key_path, opts \\ []) do
    label = Keyword.get(opts, :label, key_path)
    data = extract_params(params)

    case get_by_path(data, key_path) do
      %Nx.Tensor{} = tensor ->
        print_labeled_heatmap(tensor, label)

      nil ->
        IO.puts("[Heatmap] Parameter '#{key_path}' not found")
        IO.puts("  Available: #{Enum.join(list_param_paths(data), ", ")}")

      other ->
        IO.puts("[Heatmap] '#{key_path}' is not a tensor: #{inspect(other)}")
    end
  end

  @doc """
  Display heatmaps of all weight matrices in the model.

  Only shows 2D+ tensors (skips biases and scalars by default).

  ## Options

    * `:min_rank` - Minimum tensor rank to display (default: 2)
    * `:max_params` - Skip tensors with more than this many elements (default: 10_000).
      Large tensors are slow to render.
    * `:pattern` - Only show parameters matching this string (default: nil — show all)
  """
  def all_weights(params, opts \\ []) do
    min_rank = Keyword.get(opts, :min_rank, 2)
    max_params = Keyword.get(opts, :max_params, 10_000)
    pattern = Keyword.get(opts, :pattern, nil)
    data = extract_params(params)

    paths = list_param_paths(data)

    shown =
      Enum.filter(paths, fn path ->
        tensor = get_by_path(data, path)

        match_pattern =
          case pattern do
            nil -> true
            p -> String.contains?(path, to_string(p))
          end

        match_pattern and
          is_struct(tensor, Nx.Tensor) and
          tuple_size(Nx.shape(tensor)) >= min_rank and
          Nx.size(tensor) <= max_params
      end)

    if shown == [] do
      IO.puts("[Heatmap] No matching weight tensors found")
    else
      IO.puts("[Heatmap] Showing #{length(shown)} weight tensors\n")

      Enum.each(shown, fn path ->
        tensor = get_by_path(data, path)
        print_labeled_heatmap(tensor, path)
        IO.puts("")
      end)
    end
  end

  # ============================================================================
  # Gradient heatmaps
  # ============================================================================

  @doc """
  Display heatmaps of gradient tensors.

  Visualizes gradient magnitudes to diagnose vanishing/exploding gradients.
  Large values appear bright, small values appear dark.

  ## Parameters

    * `grads` - Gradient map (from `value_and_grad` or `Nx.Defn.grad`)

  ## Options

    * `:min_rank` - Minimum tensor rank to display (default: 2)
    * `:max_params` - Skip tensors with more than this many elements (default: 10_000)
    * `:abs` - Show absolute values (default: `true`). Makes it easier to spot
      magnitude patterns regardless of sign.
  """
  def gradients(grads, opts \\ []) do
    min_rank = Keyword.get(opts, :min_rank, 2)
    max_params = Keyword.get(opts, :max_params, 10_000)
    show_abs = Keyword.get(opts, :abs, true)
    data = extract_params(grads)

    paths = list_param_paths(data)

    shown =
      Enum.filter(paths, fn path ->
        tensor = get_by_path(data, path)

        is_struct(tensor, Nx.Tensor) and
          tuple_size(Nx.shape(tensor)) >= min_rank and
          Nx.size(tensor) <= max_params
      end)

    if shown == [] do
      IO.puts("[Heatmap] No gradient tensors to display")
    else
      IO.puts("[Heatmap] Gradient magnitudes (#{length(shown)} tensors)\n")

      Enum.each(shown, fn path ->
        tensor = get_by_path(data, path)
        display_tensor = if show_abs, do: Nx.abs(tensor), else: tensor

        norm = tensor |> Nx.pow(2) |> Nx.sum() |> Nx.to_number() |> :math.sqrt()
        label = "#{path} (norm=#{Float.round(norm, 4)})"

        print_labeled_heatmap(display_tensor, label)
        IO.puts("")
      end)
    end
  end

  # ============================================================================
  # Gradient capture via runtime_call
  # ============================================================================

  @doc """
  Capture and display a gradient heatmap inside defn via `runtime_call`.

  Logs a heatmap of the gradient tensor to the terminal during graph
  execution. Returns the tensor unchanged (passthrough).

  Use this inside `value_and_grad` or after computing gradients to
  visualize gradient flow in real time.

  ## Options

    * `:label` - Label for the heatmap (default: `"grad"`)
    * `:abs` - Show absolute values (default: `true`)
    * `:every` - Only display every N calls (default: 1)

  ## Examples

      defn training_step(params, input) do
        {loss, grads} = value_and_grad(params, &loss_fn.(&1, input))
        # Visualize a specific gradient
        grads = put_in(grads["dense"]["kernel"],
          Edifice.Display.Heatmap.capture_grad(grads["dense"]["kernel"],
            label: "dense.kernel grad"))
        {loss, grads}
      end
  """
  defn capture_grad(tensor, opts \\ []) do
    Nx.runtime_call(tensor, tensor, opts, &__MODULE__.capture_grad_callback/2)
  end

  @doc false
  def capture_grad_callback(tensor, opts) do
    label = Keyword.get(opts, :label, "grad")
    show_abs = Keyword.get(opts, :abs, true)
    every = Keyword.get(opts, :every, 1)

    if should_log?(label, every) do
      display = if show_abs, do: Nx.abs(tensor), else: tensor
      print_labeled_heatmap(display, label)
    end

    tensor
  end

  # ============================================================================
  # Summary statistics heatmap
  # ============================================================================

  @doc """
  Display a summary heatmap showing per-layer gradient or weight norms.

  Creates a 1D heatmap where each cell represents one parameter tensor's
  L2 norm. Useful for a quick overview of gradient/weight magnitude
  distribution across layers.

  ## Parameters

    * `params_or_grads` - Parameter or gradient map
    * `:label` - Display label (default: `"norms"`)
  """
  def norm_summary(params_or_grads, opts \\ []) do
    label = Keyword.get(opts, :label, "norms")
    data = extract_params(params_or_grads)
    paths = list_param_paths(data)

    tensors =
      paths
      |> Enum.map(fn path -> {path, get_by_path(data, path)} end)
      |> Enum.filter(fn {_path, t} -> is_struct(t, Nx.Tensor) end)

    if tensors == [] do
      IO.puts("[Heatmap] No tensors found for norm summary")
    else
      norms =
        Enum.map(tensors, fn {_path, t} ->
          t |> Nx.as_type(:f32) |> Nx.pow(2) |> Nx.sum() |> Nx.to_number() |> :math.sqrt()
        end)

      norm_tensor = Nx.tensor(norms) |> Nx.reshape({1, length(norms)})

      IO.puts("[Heatmap] #{label} — #{length(tensors)} parameters")

      Enum.zip(paths, norms)
      |> Enum.each(fn {path, norm} ->
        IO.puts("  #{String.pad_trailing(path, 30)} #{Float.round(norm, 4)}")
      end)

      IO.puts("")
      IO.inspect(Nx.to_heatmap(norm_tensor), label: label)
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp print_labeled_heatmap(tensor, label) do
    shape = Nx.shape(tensor)
    type = Nx.type(tensor)
    IO.puts("#{label} #{inspect(shape)} #{Nx.Type.to_string(type)}")

    case tuple_size(shape) do
      0 ->
        IO.puts("  scalar: #{Nx.to_number(tensor)}")

      1 ->
        # Reshape 1D to 2D for better visualization
        len = elem(shape, 0)
        cols = min(len, 64)
        rows = div(len + cols - 1, cols)
        padded_size = rows * cols

        display =
          if padded_size > len do
            Nx.pad(tensor, 0.0, [{0, padded_size - len, 0}])
          else
            tensor
          end
          |> Nx.reshape({rows, cols})

        IO.inspect(Nx.to_heatmap(display))

      _ ->
        # For 3D+, show just the first 2D slice
        display =
          case tuple_size(shape) do
            2 -> tensor
            _ -> tensor |> Nx.reshape({Nx.size(tensor) |> div(elem(shape, tuple_size(shape) - 1)), elem(shape, tuple_size(shape) - 1)})
          end

        IO.inspect(Nx.to_heatmap(display))
    end
  end

  defp extract_params(%Axon.ModelState{data: data}), do: data
  defp extract_params(%{} = map), do: map

  defp get_by_path(map, path) when is_binary(path) do
    keys = String.split(path, ".")
    get_in(map, keys)
  end

  defp list_param_paths(map, prefix \\ []) do
    Enum.flat_map(map, fn
      {k, %Nx.Tensor{}} ->
        [Enum.join(prefix ++ [k], ".")]

      {k, v} when is_map(v) and not is_struct(v) ->
        list_param_paths(v, prefix ++ [k])

      _ ->
        []
    end)
  end

  defp should_log?(label, 1), do: bump_counter(label) || true

  defp should_log?(label, every) do
    count = bump_counter(label)
    rem(count, every) == 0
  end

  defp bump_counter(label) do
    key = {__MODULE__, :counter, label}
    count = Process.get(key, 0) + 1
    Process.put(key, count)
    count
  end
end
