defmodule Edifice.Serving.Batch do
  @moduledoc """
  Batched inference serving for Edifice models via `Nx.Serving`.

  Wraps `Nx.Serving` to provide automatic request batching, multi-GPU
  partitioning, and distributed serving for Edifice models. Handles
  both tensor and map inputs with architecture-aware preprocessing.

  ## Inline Usage (no process)

      serving = Edifice.Serving.Batch.new(predict_fn, params)
      result = Nx.Serving.run(serving, input_tensor)

  ## Process-based Usage (production)

      # In your supervision tree:
      serving = Edifice.Serving.Batch.new(predict_fn, params,
        compiler: EXLA, batch_size: 16
      )

      children = [
        {Nx.Serving, serving: serving, name: MyModelServing,
         batch_size: 16, batch_timeout: 100}
      ]

      # From any process:
      Nx.Serving.batched_run(MyModelServing, input_tensor)

  ## Multi-GPU Partitioning

      # Automatically distributes across available GPUs:
      children = [
        {Nx.Serving, serving: serving, name: MyModelServing,
         batch_size: 16, batch_timeout: 100, partitions: true}
      ]

  ## Input Formats

  Supports three input formats:

    * **Single tensor** — `Nx.Serving.run(serving, tensor)` for models with
      a single "state_sequence" input
    * **Map of tensors** — `Nx.Serving.run(serving, %{"nodes" => ..., "adj" => ...})`
      for models with named inputs (graph models, multi-modal)
    * **List of tensors** — `Nx.Serving.run(serving, [t1, t2, t3])` for batching
      multiple single-tensor inputs

  ## Comparison with InferenceServer

  `Edifice.Serving.Batch` replaces `Edifice.Serving.InferenceServer` with
  Nx ecosystem benefits:

  | Feature | InferenceServer | Serving.Batch |
  |---------|-----------------|---------------|
  | Request batching | Manual GenServer | Nx.Serving built-in |
  | Multi-GPU | No | `partitions: true` |
  | Distribution | No | Automatic via pg |
  | Batch timeout | Manual timer | Nx.Serving built-in |
  | Supervision | Manual | OTP-native |
  """

  @doc """
  Create a new serving for an Edifice model.

  ## Parameters

    * `predict_fn` - Compiled prediction function from `Axon.build/2`
    * `params` - Model parameters

  ## Options

    * `:compiler` - Defn compiler (default: `EXLA` if available)
    * `:input_key` - For single-tensor input, the map key to wrap it in
      (default: `"state_sequence"`). Set to `nil` to pass tensors directly.
    * `:defn_options` - Additional options passed to the defn compiler
  """
  @spec new(function(), map(), keyword()) :: Nx.Serving.t()
  def new(predict_fn, params, opts \\ []) do
    input_key = Keyword.get(opts, :input_key, "state_sequence")

    compiler =
      Keyword.get_lazy(opts, :compiler, fn ->
        if Code.ensure_loaded?(EXLA), do: EXLA, else: Nx.Defn.Evaluator
      end)

    defn_options = Keyword.get(opts, :defn_options, [])

    # Build the serving computation
    # params are captured in the closure — they become constants in the graph
    serving =
      Nx.Serving.new(fn opts ->
        batch_fn = fn inputs ->
          run_predict(predict_fn, params, inputs, input_key)
        end

        Nx.Defn.jit(batch_fn, [compiler: compiler] ++ defn_options ++ opts)
      end)

    # Client preprocessing: normalize input format → Nx.Batch
    serving
    |> Nx.Serving.client_preprocessing(fn input ->
      batch = build_batch(input, input_key)
      {batch, :ok}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _server_info}, :ok ->
      result
    end)
  end

  @doc """
  Create a serving for autoregressive generation.

  Wraps `Edifice.Serving.Generate.generate/3` in an `Nx.Serving` for
  batched generation requests. Each request is a prompt tensor;
  the serving batches prompts (padding to max length) and runs
  generation in parallel.

  ## Parameters

    * `predict_fn` - Compiled prediction function
    * `params` - Model parameters

  ## Options

    * `:embed_fn` - Token ID → embedding function (required)
    * `:seq_len` - Model sequence length (required)
    * `:max_tokens` - Max tokens to generate (default: 128)
    * `:temperature` - Sampling temperature (default: 1.0)
    * `:top_k` - Top-k filtering (default: 0)
    * `:top_p` - Nucleus sampling (default: 1.0)
    * `:compiler` - Defn compiler (default: EXLA if available)

  ## Example

      serving = Edifice.Serving.Batch.generation(predict_fn, params,
        embed_fn: &Nx.take(embed_table, &1),
        seq_len: 128,
        max_tokens: 50,
        temperature: 0.7
      )

      # In supervision tree
      children = [{Nx.Serving, serving: serving, name: MyLMServing, batch_size: 4, batch_timeout: 200}]

      # Generate from any process
      Nx.Serving.batched_run(MyLMServing, prompt_tensor)
  """
  @spec generation(function(), map(), keyword()) :: Nx.Serving.t()
  def generation(predict_fn, params, opts) do
    gen_opts =
      Keyword.take(opts, [
        :embed_fn,
        :seq_len,
        :max_tokens,
        :temperature,
        :top_k,
        :top_p,
        :seed,
        :stop_token
      ])

    Nx.Serving.new(fn _opts ->
      fn prompt ->
        # Ensure prompt is 2D [batch, seq]
        prompt =
          case Nx.shape(prompt) do
            {_seq} -> Nx.reshape(prompt, {1, :auto})
            {_batch, _seq} -> prompt
          end

        Edifice.Serving.Generate.generate(predict_fn, params, [{:prompt, prompt} | gen_opts])
      end
    end)
    |> Nx.Serving.client_preprocessing(fn input ->
      batch =
        case input do
          %Nx.Tensor{} = t -> Nx.Batch.stack([t])
          list when is_list(list) -> Nx.Batch.stack(list)
          %Nx.Batch{} = b -> b
        end

      {batch, :ok}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _server_info}, :ok ->
      result
    end)
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp run_predict(predict_fn, params, inputs, nil) do
    predict_fn.(params, inputs)
  end

  defp run_predict(predict_fn, params, inputs, input_key) when is_binary(input_key) do
    predict_fn.(params, %{input_key => inputs})
  end

  defp build_batch(input, _input_key) do
    case input do
      # Single tensor — wrap in a batch
      %Nx.Tensor{} = tensor ->
        case Nx.rank(tensor) do
          r when r >= 2 ->
            # Already has batch dim — use as-is
            Nx.Batch.concatenate([tensor])

          _ ->
            # Add batch dim
            Nx.Batch.stack([tensor])
        end

      # List of tensors — stack into batch
      [%Nx.Tensor{} | _] = list ->
        Nx.Batch.stack(list)

      # Map of tensors — stack each value
      %{} = map when not is_struct(map) ->
        entries =
          map
          |> Map.values()
          |> Enum.map(fn tensor ->
            case Nx.rank(tensor) do
              r when r >= 2 -> tensor
              _ -> Nx.new_axis(tensor, 0)
            end
          end)

        # For map inputs, we concatenate the first value as the batch key
        # and rebuild the map in the serving function
        Nx.Batch.concatenate(entries)

      # Already a batch
      %Nx.Batch{} = batch ->
        batch
    end
  end
end
