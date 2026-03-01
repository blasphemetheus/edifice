defmodule Edifice.Pretrained.Hub do
  @moduledoc """
  Download pretrained weights from HuggingFace Hub with local caching.

  Fetches `.safetensors` checkpoint files for a given repository, caching
  them locally so subsequent calls are instant. Supports both single-file
  and sharded (multi-file) models.

  ## Prerequisites

  Requires the optional `req` dependency:

      {:req, "~> 0.5", optional: true}

  ## Usage

      # Download a single-file model
      {:ok, [path]} = Edifice.Pretrained.Hub.download("google/vit-base-patch16-224")

      # Bang variant
      [path] = Edifice.Pretrained.Hub.download!("google/vit-base-patch16-224")

      # Then load with a key map
      model_state = Edifice.Pretrained.load(MyKeyMap, path)

      # Sharded models return multiple paths
      paths = Edifice.Pretrained.Hub.download!("bigscience/bloom")
      model_state = Edifice.Pretrained.load_sharded(MyKeyMap, paths)

  ## Options

    - `:revision` — Git revision (branch, tag, commit). Default: `"main"`.
    - `:cache_dir` — Override cache directory. Default: `~/.cache/edifice`.
    - `:force` — Re-download even if cached. Default: `false`.
    - `:token` — HuggingFace API token for private/gated models.

  """

  require Logger

  @hf_base_url "https://huggingface.co"
  @default_cache_dir Path.expand("~/.cache/edifice")
  @index_filename "model.safetensors.index.json"
  @single_filename "model.safetensors"

  @type download_opt ::
          {:revision, String.t()}
          | {:cache_dir, Path.t()}
          | {:force, boolean()}
          | {:token, String.t()}

  @doc """
  Downloads SafeTensors checkpoint files for the given HuggingFace repo.

  Returns `{:ok, paths}` where `paths` is a list of local file paths,
  or `{:error, reason}`.

  ## Examples

      {:ok, [path]} = Hub.download("google/vit-base-patch16-224")
      {:ok, paths} = Hub.download("bigscience/bloom", token: "hf_...")

  """
  @spec download(String.t(), [download_opt()]) :: {:ok, [Path.t()]} | {:error, String.t()}
  def download(repo_id, opts \\ []) do
    ensure_req!()

    revision = Keyword.get(opts, :revision, "main")
    force = Keyword.get(opts, :force, false)
    headers = auth_headers(opts)

    # Try sharded index first
    case fetch_file(repo_id, @index_filename, revision, headers) do
      {:ok, index_body} ->
        filenames = parse_shard_index(index_body)
        download_files(repo_id, filenames, revision, headers, force, opts)

      {:error, :not_found} ->
        # Single file model
        download_files(repo_id, [@single_filename], revision, headers, force, opts)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Like `download/2` but raises on error.

  Returns a list of local file paths.

  ## Examples

      [path] = Hub.download!("google/vit-base-patch16-224")

  """
  @spec download!(String.t(), [download_opt()]) :: [Path.t()]
  def download!(repo_id, opts \\ []) do
    case download(repo_id, opts) do
      {:ok, paths} -> paths
      {:error, reason} -> raise RuntimeError, "Hub download failed: #{reason}"
    end
  end

  @doc """
  Returns the local cache directory for a given repo.

  ## Examples

      Hub.cache_path("google/vit-base-patch16-224")
      #=> "~/.cache/edifice/google/vit-base-patch16-224"

      Hub.cache_path("google/vit-base-patch16-224", cache_dir: "/tmp/models")
      #=> "/tmp/models/google/vit-base-patch16-224"

  """
  @spec cache_path(String.t(), keyword()) :: Path.t()
  def cache_path(repo_id, opts \\ []) do
    cache_dir = Keyword.get(opts, :cache_dir, @default_cache_dir)
    Path.join(cache_dir, repo_id)
  end

  @doc """
  Parses a SafeTensors shard index JSON and returns unique sorted filenames.

  The index JSON has a `"weight_map"` field mapping parameter names to shard
  filenames. This extracts the unique set of shard files needed.

  ## Examples

      Hub.parse_shard_index(~s({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}))
      #=> ["shard-1.safetensors", "shard-2.safetensors"]

  """
  @spec parse_shard_index(String.t()) :: [String.t()]
  def parse_shard_index(json_body) do
    json_body
    |> Jason.decode!()
    |> Map.fetch!("weight_map")
    |> Map.values()
    |> Enum.uniq()
    |> Enum.sort()
  end

  # -- Private --

  defp download_files(repo_id, filenames, revision, headers, force, opts) do
    total = length(filenames)

    paths =
      filenames
      |> Enum.with_index(1)
      |> Enum.map(fn {filename, n} ->
        local_path = file_cache_path(repo_id, filename, opts)

        if not force and File.exists?(local_path) do
          Logger.info("Cached #{filename} (#{n}/#{total})")
          local_path
        else
          Logger.info("Downloading #{filename} (#{n}/#{total})...")
          download_to_file(repo_id, filename, revision, headers, local_path)
        end
      end)

    {:ok, paths}
  end

  defp download_to_file(repo_id, filename, revision, headers, local_path) do
    url = file_url(repo_id, filename, revision)
    File.mkdir_p!(Path.dirname(local_path))

    part_path = local_path <> ".part"

    response = apply(Req, :get!, [[url: url, headers: headers, into: File.stream!(part_path)]])

    case response.status do
      200 ->
        File.rename!(part_path, local_path)
        local_path

      status when status in [401, 403] ->
        File.rm(part_path)

        raise RuntimeError,
              "Access denied (HTTP #{status}) for #{repo_id}/#{filename}. " <>
                "This model may be private or gated. " <>
                "Pass a token: Hub.download!(\"#{repo_id}\", token: \"hf_...\")"

      status ->
        File.rm(part_path)
        raise RuntimeError, "Failed to download #{filename}: HTTP #{status}"
    end
  end

  defp fetch_file(repo_id, filename, revision, headers) do
    url = file_url(repo_id, filename, revision)
    response = apply(Req, :get!, [[url: url, headers: headers]])

    case response.status do
      200 ->
        {:ok, response.body}

      404 ->
        {:error, :not_found}

      status when status in [401, 403] ->
        {:error,
         "Access denied (HTTP #{status}) for #{repo_id}. " <>
           "This model may be private or gated. " <>
           "Pass a token: Hub.download!(\"#{repo_id}\", token: \"hf_...\")"}

      status ->
        {:error, "HTTP #{status} fetching #{filename} from #{repo_id}"}
    end
  end

  defp file_url(repo_id, filename, revision) do
    "#{@hf_base_url}/#{repo_id}/resolve/#{revision}/#{filename}"
  end

  defp file_cache_path(repo_id, filename, opts) do
    Path.join(cache_path(repo_id, opts), filename)
  end

  defp auth_headers(opts) do
    case Keyword.get(opts, :token) do
      nil -> []
      token -> [{"authorization", "Bearer #{token}"}]
    end
  end

  defp ensure_req! do
    unless Code.ensure_loaded?(Req) do
      raise RuntimeError, """
      The :req package is required for HuggingFace Hub downloads.

      Add it to your mix.exs dependencies:

          {:req, "~> 0.5"}

      Then run: mix deps.get
      """
    end
  end
end
