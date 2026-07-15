if Code.ensure_loaded?(Kino.SmartCell) do
  defmodule Edifice.SmartCell.ModelExplorer do
    @moduledoc """
    Livebook Smart Cell for browsing, configuring, and building Edifice architectures.

    ## Usage

    In a Livebook setup cell:

        Edifice.SmartCell.ModelExplorer.register()

    Then insert an "Edifice Model Explorer" cell from the Smart Cell menu.
    """

    use Kino.JS, assets_path: nil
    use Kino.JS.Live
    use Kino.SmartCell, name: "Edifice Model Explorer"

    # ── Option schemas ──────────────────────────────────────────────────

    @common_options [
      %{key: "embed_dim", label: "Embed Dim", type: "number", default: "256"},
      %{key: "hidden_size", label: "Hidden Size", type: "number", default: "256"},
      %{key: "num_layers", label: "Layers", type: "number", default: "2"},
      %{key: "num_heads", label: "Heads", type: "number", default: "4"},
      %{key: "seq_len", label: "Seq Length", type: "number", default: "32"},
      %{key: "dropout", label: "Dropout", type: "number", default: "0.0"}
    ]

    @family_options %{
      "ssm" => [
        %{key: "state_size", label: "State Size", type: "number", default: "16"}
      ],
      "vision" => [
        %{key: "image_size", label: "Image Size", type: "number", default: "32"},
        %{key: "patch_size", label: "Patch Size", type: "number", default: "4"},
        %{key: "in_channels", label: "Channels", type: "number", default: "3"}
      ],
      "convolutional" => [
        %{key: "image_size", label: "Image Size", type: "number", default: "32"},
        %{key: "in_channels", label: "Channels", type: "number", default: "3"}
      ],
      "generative" => [
        %{key: "latent_size", label: "Latent Size", type: "number", default: "32"}
      ],
      "graph" => [
        %{key: "num_classes", label: "Classes", type: "number", default: "4"}
      ],
      "meta" => [
        %{key: "rank", label: "Rank", type: "number", default: "4"},
        %{key: "num_experts", label: "Experts", type: "number", default: "4"}
      ],
      "audio" => [
        %{key: "vocab_size", label: "Vocab Size", type: "number", default: "32"}
      ]
    }

    # Architectures that return tuples with labels for destructuring
    @tuple_archs %{
      "vae" => ~w(encoder decoder),
      "vq_vae" => ~w(encoder decoder),
      "mae" => ~w(encoder decoder),
      "magvit2" => ~w(encoder decoder),
      "tar_flow" => ~w(encoder decoder),
      "star_flow" => ~w(encoder decoder),
      "act" => ~w(encoder decoder),
      "whisper" => ~w(encoder decoder),
      "gan" => ~w(generator discriminator),
      "simclr" => ~w(backbone projection),
      "byol" => ~w(online target),
      "barlow_twins" => ~w(backbone projection),
      "vicreg" => ~w(backbone projection),
      "jepa" => ~w(context_encoder predictor),
      "temporal_jepa" => ~w(context_encoder predictor),
      "vjepa2" => ~w(context_encoder predictor),
      "normalizing_flow" => ~w(flow_model log_det),
      "speculative_decoding" => ~w(draft verifier),
      "world_model" => ~w(encoder dynamics reward),
      "byte_latent_transformer" => ~w(encoder latent_transformer decoder)
    }

    @doc "Register this Smart Cell with Livebook."
    def register do
      Kino.SmartCell.register(__MODULE__)
    end

    # ── Callbacks ───────────────────────────────────────────────────────

    @impl true
    def init(attrs, ctx) do
      families = load_families()
      family = attrs["family"] || "transformer"

      archs = Map.get(families, family, [])
      architecture = attrs["architecture"] || List.first(archs) || ""

      default_opts = default_options_for(family)
      options = Map.merge(default_opts, attrs["options"] || %{})

      ctx =
        assign(ctx,
          family: family,
          architecture: architecture,
          options: options,
          variable: Kino.SmartCell.prefixed_var_name("model", attrs["variable"]),
          summary: nil,
          families: families
        )

      {:ok, ctx}
    end

    @impl true
    def handle_connect(ctx) do
      payload = %{
        families: ctx.assigns.families,
        family: ctx.assigns.family,
        architectures: Map.get(ctx.assigns.families, ctx.assigns.family, []),
        architecture: ctx.assigns.architecture,
        options: ctx.assigns.options,
        variable: ctx.assigns.variable,
        summary: ctx.assigns.summary,
        common_options: @common_options,
        family_options: Map.get(@family_options, ctx.assigns.family, [])
      }

      {:ok, payload, ctx}
    end

    @impl true
    def handle_event("update_family", family, ctx) do
      archs = Map.get(ctx.assigns.families, family, [])
      architecture = List.first(archs) || ""
      options = default_options_for(family)

      ctx =
        assign(ctx,
          family: family,
          architecture: architecture,
          options: options,
          summary: nil
        )

      broadcast_event(ctx, "update", %{
        family: family,
        architectures: archs,
        architecture: architecture,
        options: options,
        family_options: Map.get(@family_options, family, []),
        summary: nil
      })

      {:noreply, ctx}
    end

    @impl true
    def handle_event("update_arch", architecture, ctx) do
      ctx = assign(ctx, architecture: architecture, summary: nil)
      broadcast_event(ctx, "update", %{architecture: architecture, summary: nil})
      {:noreply, ctx}
    end

    @impl true
    def handle_event("update_option", %{"key" => key, "value" => value}, ctx) do
      options = Map.put(ctx.assigns.options, key, value)
      ctx = assign(ctx, options: options)
      {:noreply, ctx}
    end

    @impl true
    def handle_event("update_variable", variable, ctx) do
      ctx = assign(ctx, variable: variable)
      {:noreply, ctx}
    end

    @impl true
    def to_attrs(ctx) do
      %{
        "family" => ctx.assigns.family,
        "architecture" => ctx.assigns.architecture,
        "options" => ctx.assigns.options,
        "variable" => ctx.assigns.variable
      }
    end

    @impl true
    def to_source(attrs) do
      arch = attrs["architecture"]
      variable = Kino.SmartCell.valid_variable_name?(attrs["variable"]) && attrs["variable"]
      variable = variable || "model"

      opts_ast = build_opts_ast(attrs["family"], attrs["options"])
      arch_atom =
        try do
          String.to_existing_atom(arch)
        rescue
          ArgumentError -> raise ArgumentError, "unknown architecture: #{inspect(arch)}"
        end

      build_call =
        case opts_ast do
          [] ->
            quote do
              Edifice.build(unquote(arch_atom))
            end

          opts ->
            quote do
              Edifice.build(unquote(arch_atom), unquote(opts))
            end
        end

      lhs = build_lhs(arch, variable)

      ast =
        quote do
          unquote(lhs) = unquote(build_call)
        end

      Kino.SmartCell.quoted_to_string(ast)
    end

    @impl true
    def scan_eval_result(server, {:ok, result}) do
      summary = compute_summary(result)

      if summary do
        send(server, {:summary, summary})
      end
    end

    def scan_eval_result(_server, _error), do: :ok

    @impl true
    def handle_info({:summary, summary}, ctx) do
      ctx = assign(ctx, summary: summary)
      broadcast_event(ctx, "update", %{summary: summary})
      {:noreply, ctx}
    end

    # ── Private helpers ─────────────────────────────────────────────────

    defp load_families do
      Edifice.list_families()
      |> Enum.sort_by(fn {name, _} -> Atom.to_string(name) end)
      |> Enum.map(fn {family, archs} ->
        {Atom.to_string(family), Enum.map(archs, &Atom.to_string/1)}
      end)
      |> Map.new()
    end

    defp default_options_for(family) do
      all_opts = @common_options ++ Map.get(@family_options, family, [])
      Map.new(all_opts, fn %{key: k, default: d} -> {k, d} end)
    end

    defp build_opts_ast(family, options) when is_map(options) do
      defaults = default_options_for(family)

      visible_keys =
        (@common_options ++ Map.get(@family_options, family, []))
        |> Enum.map(& &1.key)
        |> MapSet.new()

      options
      |> Enum.filter(fn {k, v} ->
        MapSet.member?(visible_keys, k) and v != "" and Map.get(defaults, k) != v
      end)
      |> Enum.sort_by(fn {k, _} -> k end)
      |> Enum.map(fn {k, v} ->
        {String.to_atom(k), parse_number(v)}
      end)
    end

    defp parse_number(val) when is_binary(val) do
      case Float.parse(val) do
        {f, ""} ->
          if f == trunc(f), do: trunc(f), else: f

        _ ->
          val
      end
    end

    defp parse_number(val), do: val

    defp build_lhs(arch, variable) do
      case Map.get(@tuple_archs, arch) do
        nil ->
          Macro.var(String.to_atom(variable), nil)

        labels ->
          vars = Enum.map(labels, &Macro.var(String.to_atom(&1), nil))
          {:{}, [], vars}
      end
    end

    defp compute_summary(%Axon{} = model) do
      layer_count =
        Axon.reduce_nodes(model, 0, fn _node, acc -> acc + 1 end)

      table_str = Edifice.Display.as_table(model)

      # Extract param count and memory from last two lines of the table
      lines = String.split(table_str, "\n")

      {params, memory} =
        case Enum.take(lines, -2) do
          [params_line, memory_line] ->
            params =
              case Regex.run(~r/(\d[\d,]*)/, params_line) do
                [_, n] -> String.replace(n, ",", "")
                _ -> "?"
              end

            memory =
              case Regex.run(~r/:\s*(.+)$/, memory_line) do
                [_, m] -> String.trim(m)
                _ -> "?"
              end

            {params, memory}

          _ ->
            {"?", "?"}
        end

      %{layers: layer_count, params: params, memory: memory}
    rescue
      _ -> nil
    end

    defp compute_summary(tuple) when is_tuple(tuple) do
      case Tuple.to_list(tuple) do
        [%Axon{} = first | _] -> compute_summary(first)
        _ -> nil
      end
    end

    defp compute_summary(_), do: nil

    # ── Assets ──────────────────────────────────────────────────────────

    asset "main.js" do
      """
      export function init(ctx, payload) {
        ctx.importCSS("main.css");

        const container = document.createElement("div");
        container.className = "edifice-explorer";

        container.innerHTML = `
          <div class="header">
            <span class="title">Edifice Model Explorer</span>
            <div class="var-field">
              <label for="variable">Variable:</label>
              <input id="variable" type="text" value="${payload.variable}" />
            </div>
          </div>
          <div class="selectors">
            <div class="field">
              <label for="family">Family</label>
              <select id="family"></select>
            </div>
            <div class="field">
              <label for="architecture">Architecture</label>
              <select id="architecture"></select>
            </div>
          </div>
          <div class="options" id="options-grid"></div>
          <div class="summary" id="summary" style="display:none"></div>
        `;

        ctx.root.appendChild(container);

        const familySelect = container.querySelector("#family");
        const archSelect = container.querySelector("#architecture");
        const variableInput = container.querySelector("#variable");
        const optionsGrid = container.querySelector("#options-grid");
        const summaryDiv = container.querySelector("#summary");

        let currentOptions = payload.options;

        function populateFamilySelect(families, selected) {
          familySelect.innerHTML = "";
          Object.keys(families).sort().forEach(f => {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f.replace(/_/g, " ");
            opt.selected = f === selected;
            familySelect.appendChild(opt);
          });
        }

        function populateArchSelect(archs, selected) {
          archSelect.innerHTML = "";
          archs.forEach(a => {
            const opt = document.createElement("option");
            opt.value = a;
            opt.textContent = a.replace(/_/g, " ");
            opt.selected = a === selected;
            archSelect.appendChild(opt);
          });
        }

        function renderOptions(commonOpts, familyOpts, values) {
          optionsGrid.innerHTML = "";
          const allOpts = commonOpts.concat(familyOpts);
          allOpts.forEach(opt => {
            const div = document.createElement("div");
            div.className = "opt-field";
            const label = document.createElement("label");
            label.textContent = opt.label;
            label.setAttribute("for", "opt-" + opt.key);
            const input = document.createElement("input");
            input.id = "opt-" + opt.key;
            input.type = "number";
            input.step = opt.key === "dropout" ? "0.1" : "1";
            input.value = values[opt.key] !== undefined ? values[opt.key] : opt.default;
            input.addEventListener("change", (e) => {
              currentOptions[opt.key] = e.target.value;
              ctx.pushEvent("update_option", { key: opt.key, value: e.target.value });
            });
            div.appendChild(label);
            div.appendChild(input);
            optionsGrid.appendChild(div);
          });
        }

        function renderSummary(summary) {
          if (summary) {
            summaryDiv.style.display = "block";
            const params = Number(summary.params).toLocaleString();
            summaryDiv.textContent =
              summary.layers + " layers | " + params + " params | " + summary.memory;
          } else {
            summaryDiv.style.display = "none";
          }
        }

        // Initial render
        populateFamilySelect(payload.families, payload.family);
        populateArchSelect(payload.architectures, payload.architecture);
        renderOptions(payload.common_options, payload.family_options, payload.options);
        renderSummary(payload.summary);

        // Events
        familySelect.addEventListener("change", (e) => {
          ctx.pushEvent("update_family", e.target.value);
        });

        archSelect.addEventListener("change", (e) => {
          ctx.pushEvent("update_arch", e.target.value);
        });

        variableInput.addEventListener("change", (e) => {
          ctx.pushEvent("update_variable", e.target.value);
        });

        // Server updates
        ctx.handleEvent("update", (data) => {
          if (data.architectures) {
            populateArchSelect(data.architectures, data.architecture || "");
          }
          if (data.architecture !== undefined) {
            archSelect.value = data.architecture;
          }
          if (data.options) {
            currentOptions = data.options;
          }
          if (data.family_options !== undefined) {
            renderOptions(
              payload.common_options,
              data.family_options,
              currentOptions
            );
          }
          if (data.summary !== undefined) {
            renderSummary(data.summary);
          }
        });

        ctx.handleSync(() => {
          // Flush pending input values
          optionsGrid.querySelectorAll("input").forEach(input => {
            const key = input.id.replace("opt-", "");
            if (currentOptions[key] !== input.value) {
              currentOptions[key] = input.value;
              ctx.pushEvent("update_option", { key: key, value: input.value });
            }
          });
        });
      }
      """
    end

    asset "main.css" do
      """
      .edifice-explorer {
        font-family: var(--livebook-font-family, Inter, system-ui, sans-serif);
        font-size: 14px;
        color: var(--livebook-text-color, #1e1e1e);
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .edifice-explorer .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--livebook-border-color, #e0e0e0);
      }

      .edifice-explorer .title {
        font-weight: 600;
        font-size: 15px;
      }

      .edifice-explorer .var-field {
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .edifice-explorer .var-field input {
        width: 100px;
        padding: 4px 8px;
        border: 1px solid var(--livebook-border-color, #ccc);
        border-radius: 4px;
        font-family: var(--livebook-code-font-family, monospace);
        font-size: 13px;
      }

      .edifice-explorer .selectors {
        display: flex;
        gap: 16px;
      }

      .edifice-explorer .field {
        display: flex;
        flex-direction: column;
        gap: 4px;
        flex: 1;
      }

      .edifice-explorer .field label {
        font-size: 12px;
        font-weight: 500;
        color: var(--livebook-secondary-text-color, #666);
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }

      .edifice-explorer .field select {
        padding: 6px 8px;
        border: 1px solid var(--livebook-border-color, #ccc);
        border-radius: 4px;
        background: var(--livebook-input-bg, #fff);
        font-size: 14px;
        text-transform: capitalize;
      }

      .edifice-explorer .options {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px 16px;
      }

      .edifice-explorer .opt-field {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .edifice-explorer .opt-field label {
        font-size: 11px;
        font-weight: 500;
        color: var(--livebook-secondary-text-color, #666);
      }

      .edifice-explorer .opt-field input {
        padding: 4px 8px;
        border: 1px solid var(--livebook-border-color, #ccc);
        border-radius: 4px;
        font-size: 13px;
        font-family: var(--livebook-code-font-family, monospace);
        width: 100%;
        box-sizing: border-box;
      }

      .edifice-explorer .summary {
        padding: 8px 12px;
        background: var(--livebook-surface-color, #f5f5f5);
        border-radius: 4px;
        font-family: var(--livebook-code-font-family, monospace);
        font-size: 13px;
        color: var(--livebook-secondary-text-color, #555);
      }
      """
    end
  end
end
