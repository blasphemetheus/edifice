defmodule Edifice.Display do
  @moduledoc """
  Rendering utilities for visualizing Axon model graphs.

  Three output formats are supported:

    * **table** — Keras-style summary with layer names, shapes, and parameter counts.
    * **tree** — Lightweight ASCII tree showing the operation hierarchy.
    * **mermaid** — Raw Mermaid flowchart text for pasting into GitHub markdown or mermaid.live.

  All functions accept an `%Axon{}` struct and return an iodata-friendly string.

  ## Examples

      model = Edifice.build(:mlp, input_size: 16, hidden_sizes: [32], output_size: 8)

      Edifice.Display.as_tree(model) |> IO.puts()
      Edifice.Display.as_mermaid(model) |> IO.puts()
      Edifice.Display.as_table(model) |> IO.puts()
  """

  # -------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------

  @doc """
  Render model as an ASCII tree showing operation hierarchy.

  Does not require shape computation — works on graph structure alone.

  ## Options

    * `:name` — label printed as the tree root (default `"Model"`)

  ## Examples

      model = Axon.input("input", shape: {nil, 16}) |> Axon.dense(32)
      Edifice.Display.as_tree(model)
  """
  @spec as_tree(Axon.t(), keyword()) :: String.t()
  def as_tree(%Axon{output: id, nodes: nodes}, opts \\ []) do
    name = Keyword.get(opts, :name, "Model")
    {lines, _op_counts} = tree_walk(id, nodes, 0, MapSet.new(), %{})
    Enum.join([name | lines], "\n")
  end

  @doc """
  Render model as raw Mermaid flowchart text.

  Requires input templates to compute shapes. Templates are built
  automatically from the model's input nodes.

  ## Options

    * `:name` — unused (kept for API symmetry)
    * `:direction` — `:top_down` (default) or `:left_right`

  ## Examples

      model = Axon.input("input", shape: {nil, 16}) |> Axon.dense(32)
      Edifice.Display.as_mermaid(model)
  """
  @spec as_mermaid(Axon.t(), keyword()) :: String.t()
  def as_mermaid(%Axon{output: id, nodes: nodes} = model, opts \\ []) do
    direction = if Keyword.get(opts, :direction, :top_down) == :left_right, do: "LR", else: "TD"
    templates = build_input_templates(model)
    {_root, {cache, _op_counts, edges}} = mermaid_walk(id, nodes, templates, {%{}, %{}, []})

    node_lines =
      cache
      |> Map.values()
      |> Enum.map_join(";\n", &mermaid_node_entry/1)

    edge_lines = Enum.map_join(edges, ";\n", fn {from, to} -> "#{from.id} --> #{to.id}" end)

    "graph #{direction};\n#{node_lines};\n#{edge_lines};"
  end

  @doc """
  Render model as a plain-text table with layer shapes and parameter counts.

  Builds input templates automatically from the model's input nodes.

  ## Options

    * `:name` — table title (default `"Model"`)
  """
  @spec as_table(Axon.t(), keyword()) :: String.t()
  def as_table(%Axon{output: id, nodes: nodes} = model, opts \\ []) do
    name = Keyword.get(opts, :name, "Model")
    templates = build_input_templates(model)

    {rows, _cache, _op_counts, info} =
      table_walk(id, nodes, templates, %{}, %{}, %{params: 0, bytes: 0})

    header = ["Layer", "Output Shape", "Parameters"]

    col_widths =
      Enum.reduce([header | rows], [0, 0, 0], fn row, widths ->
        Enum.zip_with(row, widths, fn cell, w -> max(String.length(cell), w) end)
      end)

    sep = Enum.map_join(col_widths, "-+-", fn w -> String.duplicate("-", w + 2) end)

    fmt_row = fn row ->
      row
      |> Enum.zip(col_widths)
      |> Enum.map_join(" | ", fn {cell, w} -> String.pad_trailing(cell, w) end)
    end

    lines = [
      name,
      sep,
      fmt_row.(header),
      String.replace(sep, "-", "="),
      Enum.map_join(rows, "\n", fmt_row),
      sep,
      "Total Parameters: #{info.params}",
      "Total Parameters Memory: #{readable_size(info.bytes)}"
    ]

    Enum.join(lines, "\n")
  end

  @doc """
  Format the result of a `build/1` call (single Axon or tuple) in the given format.

  Returns a string with each component labeled.
  """
  @spec format_build_result(Axon.t() | tuple(), atom(), keyword()) :: String.t()
  def format_build_result(result, format, opts \\ [])

  def format_build_result(%Axon{} = model, format, opts) do
    render(model, format, opts)
  end

  def format_build_result(tuple, format, opts) when is_tuple(tuple) do
    elements = Tuple.to_list(tuple)
    labels = component_labels(length(elements))

    elements
    |> Enum.zip(labels)
    |> Enum.map_join("\n\n", fn {model, label} ->
      "=== #{label} ===\n" <> render(model, format, Keyword.put(opts, :name, label))
    end)
  end

  # -------------------------------------------------------------------
  # Input template generation
  # -------------------------------------------------------------------

  @doc false
  @spec build_input_templates(Axon.t()) :: map()
  def build_input_templates(%Axon{} = model) do
    model
    |> Axon.get_inputs()
    |> Map.new(fn {name, shape} ->
      concrete =
        if shape do
          shape
          |> Tuple.to_list()
          |> Enum.map(fn
            nil -> 1
            n -> n
          end)
          |> List.to_tuple()
        else
          {1}
        end

      {name, Nx.template(concrete, :f32)}
    end)
  end

  # -------------------------------------------------------------------
  # ASCII tree internals
  # -------------------------------------------------------------------

  defp tree_walk(id, nodes, depth, visited, op_counts) do
    if MapSet.member?(visited, id) do
      {[], op_counts}
    else
      visited = MapSet.put(visited, id)
      %Axon.Node{op_name: op_name, parent: parents, name: name_fn} = nodes[id]
      label = name_fn.(op_name, op_counts)
      op_counts = Map.update(op_counts, op_name, 1, fn c -> c + 1 end)

      prefix = tree_prefix(depth)
      line = "#{prefix}#{label} (#{op_name})"

      {child_lines, op_counts} =
        parents
        |> Enum.reduce({[], op_counts}, fn pid, {acc, oc} ->
          {lines, oc} = tree_walk(pid, nodes, depth + 1, visited, oc)
          {acc ++ lines, oc}
        end)

      {[line | child_lines], op_counts}
    end
  end

  defp tree_prefix(0), do: ""

  defp tree_prefix(depth) do
    indent = String.duplicate("│   ", max(depth - 1, 0))
    indent <> "└── "
  end

  # -------------------------------------------------------------------
  # Mermaid internals (mirrors Axon.Display.as_graph without Kino)
  # -------------------------------------------------------------------

  defp mermaid_walk(id, nodes, templates, {cache, op_counts, edges} = acc) do
    case cache do
      %{^id => entry} ->
        {entry, acc}

      %{} ->
        %Axon.Node{id: ^id, op_name: op, name: name_fn, parent: parents} = nodes[id]

        {inputs, {cache, op_counts, edges}} =
          Enum.map_reduce(parents, {cache, op_counts, edges}, fn pid, acc ->
            mermaid_walk(pid, nodes, templates, acc)
          end)

        name = name_fn.(op, op_counts)
        shape = safe_output_shape(id, nodes, templates)
        entry = %{id: id, op: op, name: name, shape: shape}

        edges = Enum.reduce(inputs, edges, fn from, acc -> [{from, entry} | acc] end)
        op_counts = Map.update(op_counts, op, 1, fn c -> c + 1 end)

        {entry, {Map.put(cache, id, entry), op_counts, edges}}
    end
  end

  defp safe_output_shape(id, nodes, templates) do
    Axon.get_output_shape(%Axon{output: id, nodes: nodes}, templates)
    |> expand_shape()
  rescue
    _ -> nil
  end

  defp expand_shape(%Nx.Tensor{} = t), do: Nx.shape(t)

  defp expand_shape(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&expand_shape/1) |> List.to_tuple()

  defp expand_shape(other), do: other

  defp mermaid_node_entry(%{id: id, op: :input, name: name, shape: shape}) do
    ~s'#{id}[/"#{name} (:input) #{inspect(shape)}"/]'
  end

  defp mermaid_node_entry(%{id: id, op: op, name: name, shape: shape}) do
    ~s'#{id}["#{name} (#{inspect(op)}) #{inspect(shape)}"]'
  end

  # -------------------------------------------------------------------
  # Table internals
  # -------------------------------------------------------------------

  defp table_walk(id, nodes, templates, cache, op_counts, info) do
    case cache do
      %{^id => _} ->
        {[], cache, op_counts, info}

      %{} ->
        %Axon.Node{
          id: ^id,
          op_name: op_name,
          parent: parents,
          name: name_fn,
          parameters: params,
          policy: %{params: params_policy}
        } = nodes[id]

        {parent_rows, cache, op_counts, info} =
          Enum.reduce(parents, {[], cache, op_counts, info}, fn pid, {rows, c, oc, inf} ->
            {new_rows, c, oc, inf} = table_walk(pid, nodes, templates, c, oc, inf)
            {rows ++ new_rows, c, oc, inf}
          end)

        name = name_fn.(op_name, op_counts)
        op_counts = Map.update(op_counts, op_name, 1, fn c -> c + 1 end)

        shape = safe_output_shape(id, nodes, templates)
        shape_str = format_shape(shape)

        # Compute parameters
        bitsize =
          case params_policy do
            nil -> 32
            {_, bs} -> bs
          end

        parent_shapes = Enum.map(parents, fn pid -> safe_output_shape(pid, nodes, templates) end)

        {param_str, num_params} = format_params(params, parent_shapes)

        param_bytes = num_params * div(bitsize, 8)
        info = %{info | params: info.params + num_params, bytes: info.bytes + param_bytes}

        row = ["#{name} (#{op_name})", shape_str, param_str]
        cache = Map.put(cache, id, true)

        {parent_rows ++ [row], cache, op_counts, info}
    end
  end

  defp format_shape(nil), do: "?"
  defp format_shape(%Nx.Tensor{} = t), do: inspect(Nx.shape(t))
  defp format_shape(shape) when is_tuple(shape), do: inspect(shape)
  defp format_shape(other), do: inspect(other)

  defp format_params(params, parent_shapes) do
    total =
      Enum.reduce(params, 0, fn
        %Axon.Parameter{shape: {:tuple, shape_fns}}, acc ->
          Enum.reduce(shape_fns, acc, fn sfn, a ->
            a + safe_param_size(sfn, parent_shapes)
          end)

        %Axon.Parameter{template: shape_fn}, acc when is_function(shape_fn) ->
          acc + safe_param_size(shape_fn, parent_shapes)

        _, acc ->
          acc
      end)

    str = if total == 0, do: "0", else: "#{total}"
    {str, total}
  end

  defp safe_param_size(shape_fn, parent_shapes) do
    Nx.size(apply(shape_fn, parent_shapes))
  rescue
    _ -> 0
  end

  defp readable_size(n) when n < 1_000, do: "#{n} bytes"
  defp readable_size(n) when n < 1_000_000, do: "#{:io_lib.format(~c"~.2f", [n / 1_000])} KB"

  defp readable_size(n) when n < 1_000_000_000,
    do: "#{:io_lib.format(~c"~.2f", [n / 1_000_000])} MB"

  defp readable_size(n), do: "#{:io_lib.format(~c"~.2f", [n / 1_000_000_000])} GB"

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------

  defp render(model, :tree, opts), do: as_tree(model, opts)
  defp render(model, :mermaid, opts), do: as_mermaid(model, opts)
  defp render(model, :table, opts), do: as_table(model, opts)

  defp component_labels(2), do: ["Component 1", "Component 2"]
  defp component_labels(3), do: ["Component 1", "Component 2", "Component 3"]
  defp component_labels(n), do: Enum.map(1..n, &"Component #{&1}")
end
