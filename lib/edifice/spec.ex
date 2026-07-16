defmodule Edifice.Spec do
  @moduledoc """
  Self-describing model manifest: the architecture and build options a model
  was constructed with, plus library versions for provenance.

  A spec captures everything `Edifice.build/2` needs to reconstruct the same
  model graph later. Embedding it in a checkpoint (see `Edifice.Checkpoint.save/3`
  with the `:spec` option) makes the checkpoint self-describing: loading via
  `Edifice.Checkpoint.load_model/2` rebuilds the model from the stored spec and
  validates parameter shapes, instead of silently rebuilding with default
  options and producing garbage outputs on shape mismatch.

  ## Quick Start

      {model, spec} = Edifice.build_with_spec(:mamba, embed_dim: 287, state_size: 32)
      # ... train ...
      Edifice.Checkpoint.save(params, "mamba.nx", spec: spec)

      # Later — no need to remember the build options:
      {model, params} = Edifice.Checkpoint.load_model("mamba.nx")
  """

  @enforce_keys [:arch, :build_opts, :edifice_version, :nx_version, :created_at]
  defstruct [:arch, :build_opts, :edifice_version, :nx_version, :created_at, external: false]

  @type t :: %__MODULE__{
          arch: atom(),
          build_opts: keyword(),
          edifice_version: String.t(),
          nx_version: String.t(),
          created_at: DateTime.t(),
          external: boolean()
        }

  @doc """
  Create a spec for a registered architecture.

  `build_opts` should be the options that reproduce the model via
  `Edifice.build(arch, build_opts)`. Prefer `Edifice.build_with_spec/3`,
  which records the fully-merged options automatically.

  ## Options

    * `:created_at` - `DateTime` to stamp (default: `DateTime.utc_now()`)
    * `:edifice_version` / `:nx_version` - version string overrides
      (defaults read from the loaded applications)
    * `:external` - set `true` to skip the architecture-registry check for
      a COMPOSITE model owned by a downstream library (e.g. a policy that
      wraps an edifice backbone in its own heads). An external spec still
      carries build opts + provenance and still powers
      `Edifice.Checkpoint.validate_shapes!/3` (the caller supplies the
      rebuilt model), but `Edifice.Checkpoint.load_model/2` cannot rebuild
      it — the owning library must rebuild from `build_opts` itself.

  Raises `ArgumentError` for an unknown architecture (unless `external: true`)
  or for build options that cannot survive serialization (functions, PIDs,
  references, tensors).
  """
  @spec new(atom(), keyword(), keyword()) :: t()
  def new(arch, build_opts \\ [], opts \\ []) when is_atom(arch) do
    unless Keyword.get(opts, :external, false) do
      # Raises with the list of available architectures on unknown arch
      _module = Edifice.module_for(arch)
    end

    validate_serializable!(build_opts)

    %__MODULE__{
      arch: arch,
      build_opts: build_opts,
      edifice_version: Keyword.get_lazy(opts, :edifice_version, fn -> app_version(:edifice) end),
      nx_version: Keyword.get_lazy(opts, :nx_version, fn -> app_version(:nx) end),
      created_at: Keyword.get_lazy(opts, :created_at, &DateTime.utc_now/0),
      external: Keyword.get(opts, :external, false)
    }
  end

  @doc """
  Convert a spec to a plain map for embedding in serialized metadata.

  `created_at` becomes an ISO 8601 string so the map contains only atoms
  already interned by loaded Edifice modules plus primitive terms — safe to
  round-trip through `:erlang.binary_to_term(bin, [:safe])`.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = spec) do
    %{
      arch: spec.arch,
      build_opts: spec.build_opts,
      edifice_version: spec.edifice_version,
      nx_version: spec.nx_version,
      created_at: DateTime.to_iso8601(spec.created_at),
      external: spec.external
    }
  end

  @doc """
  Reconstruct a spec from a map produced by `to_map/1`.

  Tolerant: returns `{:ok, spec}` or `{:error, reason}` — never raises.
  Missing version fields default to `"unknown"`.
  """
  @spec from_map(term()) :: {:ok, t()} | {:error, String.t()}
  def from_map(%{arch: arch, build_opts: build_opts} = map) when is_atom(arch) do
    external = Map.get(map, :external, false) == true

    cond do
      not keyword_list?(build_opts) ->
        {:error, "build_opts is not a keyword list: #{inspect(build_opts)}"}

      not external and not known_arch?(arch) ->
        {:error, "unknown architecture #{inspect(arch)}"}

      true ->
        case parse_created_at(Map.get(map, :created_at)) do
          {:ok, created_at} ->
            {:ok,
             %__MODULE__{
               arch: arch,
               build_opts: build_opts,
               edifice_version: Map.get(map, :edifice_version) || "unknown",
               nx_version: Map.get(map, :nx_version) || "unknown",
               created_at: created_at,
               external: external
             }}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  def from_map(%{} = map) do
    {:error, "missing required keys :arch and :build_opts in #{inspect(Map.keys(map))}"}
  end

  def from_map(other) do
    {:error, "expected a spec map, got: #{inspect(other)}"}
  end

  # ---------------------------------------------------------------------------
  # Internals
  # ---------------------------------------------------------------------------

  defp known_arch?(arch) do
    _ = Edifice.module_for(arch)
    true
  rescue
    ArgumentError -> false
  end

  defp parse_created_at(nil), do: {:ok, ~U[1970-01-01 00:00:00Z]}
  defp parse_created_at(%DateTime{} = dt), do: {:ok, dt}

  defp parse_created_at(iso) when is_binary(iso) do
    case DateTime.from_iso8601(iso) do
      {:ok, dt, _offset} -> {:ok, dt}
      {:error, reason} -> {:error, "invalid created_at #{inspect(iso)}: #{inspect(reason)}"}
    end
  end

  defp parse_created_at(other), do: {:error, "invalid created_at: #{inspect(other)}"}

  defp keyword_list?(list), do: is_list(list) and Keyword.keyword?(list)

  defp app_version(app) do
    case Application.spec(app, :vsn) do
      vsn when is_list(vsn) -> List.to_string(vsn)
      vsn when is_binary(vsn) -> vsn
      _ -> "unknown"
    end
  end

  defp validate_serializable!(build_opts) do
    unless keyword_list?(build_opts) do
      raise ArgumentError, "build_opts must be a keyword list, got: #{inspect(build_opts)}"
    end

    offenders =
      Enum.filter(build_opts, fn {_key, value} -> not serializable?(value) end)

    if offenders != [] do
      keys = Enum.map(offenders, &elem(&1, 0))

      raise ArgumentError,
            "build_opts contain values that cannot survive checkpoint serialization " <>
              "(functions, PIDs, references, ports, or tensors) under keys: #{inspect(keys)}. " <>
              "A spec with unreproducible options is worse than none — pass only plain terms."
    end
  end

  defp serializable?(value)
       when is_atom(value) or is_number(value) or is_binary(value),
       do: true

  defp serializable?(value) when is_list(value), do: Enum.all?(value, &serializable?/1)

  defp serializable?(value) when is_tuple(value),
    do: value |> Tuple.to_list() |> Enum.all?(&serializable?/1)

  defp serializable?(%Nx.Tensor{}), do: false
  defp serializable?(%_struct{}), do: false

  defp serializable?(value) when is_map(value) do
    Enum.all?(value, fn {k, v} -> serializable?(k) and serializable?(v) end)
  end

  defp serializable?(_other), do: false
end
