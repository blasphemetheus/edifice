defmodule Edifice.Quantization.FP8 do
  @moduledoc """
  FP8 quantization for inference.

  Provides utilities to quantize model weights to 8-bit floating point
  for ~4x memory reduction (vs f32) or ~2x (vs bf16) with minimal
  accuracy loss on inference.

  Two FP8 formats are supported:

  - **`:e4m3fn`** — 4 exponent bits, 3 mantissa bits, no infinities.
    Higher precision, range [-448, 448]. **Recommended for weights.**
  - **`:e5m2`** — 5 exponent bits, 2 mantissa bits, with infinities.
    Wider range but lower precision. Better for activations/gradients.

  ## Quick Start

      model = Edifice.build(:decoder_only, embed_dim: 256, ...)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template, Axon.ModelState.empty())

      # Quantize weights to FP8
      q_params = Edifice.Quantization.FP8.quantize(params)

      # Run inference (dequantizes on the fly)
      output = predict_fn.(Edifice.Quantization.FP8.dequantize(q_params), input)

      # Or use the convenience wrapper
      q_predict_fn = Edifice.Quantization.FP8.wrap_inference(predict_fn)
      output = q_predict_fn.(q_params, input)

  ## Memory Savings

      Edifice.Quantization.FP8.report(params)
      # Original: 12.50 MB (f32) → Quantized: 3.13 MB (e4m3fn) — 4.0x reduction

  ## With MixedPrecision

  FP8 quantization composes with mixed precision. Quantize the params
  after applying a precision policy:

      model
      |> Edifice.MixedPrecision.apply(:bf16)
      |> Axon.build(mode: :inference)
      # ... init params, then quantize ...
  """

  require Logger

  @type format :: :e4m3fn | :e5m2
  @type quantized_params :: map()

  @e4m3fn_type {:f8_e4m3fn, 8}
  @e5m2_type {:f, 8}

  # ============================================================================
  # Quantization
  # ============================================================================

  @doc """
  Quantize model parameters to FP8.

  Converts all floating-point weight tensors to FP8 format with per-tensor
  scale factors stored alongside. Non-float tensors (e.g., integer indices)
  are left unchanged.

  ## Options

    * `:format` - FP8 format to use (default: `:e4m3fn`)
      * `:e4m3fn` — higher precision, range [-448, 448], recommended for weights
      * `:e5m2` — wider range, lower precision
    * `:skip` - List of parameter name patterns to skip (kept in original precision).
      Useful for keeping embeddings or final logit projections in higher precision.
      Patterns are matched against the full key path (e.g., `"embedding"`, `"head"`).

  ## Returns

  A map with the same structure as the input, where each float tensor is
  replaced by a map `%{tensor: fp8_tensor, scale: f32_scalar, original_type: type}`.
  """
  @spec quantize(map(), keyword()) :: quantized_params()
  def quantize(params, opts \\ []) do
    format = Keyword.get(opts, :format, :e4m3fn)
    skip_patterns = Keyword.get(opts, :skip, [])
    fp8_type = format_to_type(format)

    deep_map(params, [], fn tensor, key_path ->
      if should_skip?(key_path, skip_patterns) or not float_type?(tensor) do
        tensor
      else
        quantize_tensor(tensor, fp8_type)
      end
    end)
  end

  @doc """
  Dequantize FP8 parameters back to their original precision.

  Reconstructs full-precision tensors from the quantized representation.
  Call this before passing params to a predict function.
  """
  @spec dequantize(quantized_params()) :: map()
  def dequantize(params) do
    deep_map(params, [], fn value, _key_path ->
      case value do
        %{tensor: tensor, scale: scale, original_type: orig_type} ->
          tensor
          |> Nx.as_type({:f, 32})
          |> Nx.multiply(scale)
          |> Nx.as_type(orig_type)

        other ->
          other
      end
    end)
  end

  @doc """
  Wrap a predict function to auto-dequantize FP8 params before inference.

  Returns a function with the same `(params, input) -> output` signature
  that dequantizes on the fly.

  ## Examples

      q_predict = Edifice.Quantization.FP8.wrap_inference(predict_fn)
      output = q_predict.(quantized_params, input)
  """
  @spec wrap_inference(function()) :: function()
  def wrap_inference(predict_fn) do
    fn params, input ->
      predict_fn.(dequantize(params), input)
    end
  end

  # ============================================================================
  # Analysis and reporting
  # ============================================================================

  @doc """
  Report memory savings from FP8 quantization.

  ## Parameters

    * `params` - Original (unquantized) model parameters

  ## Options

    * `:format` - FP8 format for estimation (default: `:e4m3fn`)

  ## Returns

  A map with byte counts and prints a human-readable summary.
  """
  @spec report(map(), keyword()) :: map()
  def report(params, opts \\ []) do
    format = Keyword.get(opts, :format, :e4m3fn)

    {original_bytes, quantized_bytes, tensor_count} =
      count_bytes(params, format)

    ratio = if quantized_bytes > 0, do: Float.round(original_bytes / quantized_bytes, 1), else: 0.0

    result = %{
      original_bytes: original_bytes,
      quantized_bytes: quantized_bytes,
      scale_overhead_bytes: tensor_count * 4,
      tensor_count: tensor_count,
      ratio: ratio
    }

    IO.puts(
      "[FP8] Original: #{readable_size(original_bytes)} → " <>
        "Quantized: #{readable_size(quantized_bytes)} " <>
        "(#{format}) — #{ratio}x reduction " <>
        "(#{tensor_count} tensors, #{readable_size(tensor_count * 4)} scale overhead)"
    )

    result
  end

  @doc """
  Estimate FP8 memory savings without actually quantizing.

  Faster than `report/2` — doesn't traverse tensor data.
  """
  @spec estimate_savings(map(), keyword()) :: map()
  def estimate_savings(params, opts \\ []) do
    format = Keyword.get(opts, :format, :e4m3fn)
    {original_bytes, quantized_bytes, tensor_count} = count_bytes(params, format)
    ratio = if quantized_bytes > 0, do: Float.round(original_bytes / quantized_bytes, 1), else: 0.0

    %{
      original_bytes: original_bytes,
      quantized_bytes: quantized_bytes,
      ratio: ratio,
      tensor_count: tensor_count
    }
  end

  # ============================================================================
  # Per-tensor quantization
  # ============================================================================

  defp quantize_tensor(tensor, fp8_type) do
    original_type = Nx.type(tensor)

    # Compute per-tensor scale: scale = max(|tensor|) / max_representable
    abs_max =
      tensor
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.as_type({:f, 32})
      |> Nx.to_number()

    max_fp8 = max_representable(fp8_type)

    scale =
      if abs_max == 0.0 do
        1.0
      else
        abs_max / max_fp8
      end

    # Scale down, cast to FP8
    scaled_tensor =
      tensor
      |> Nx.as_type({:f, 32})
      |> Nx.divide(scale)
      |> Nx.as_type(fp8_type)

    %{
      tensor: scaled_tensor,
      scale: Nx.tensor(scale, type: {:f, 32}),
      original_type: original_type
    }
  end

  defp max_representable(@e4m3fn_type), do: 448.0
  defp max_representable(@e5m2_type), do: 57344.0

  # ============================================================================
  # Helpers
  # ============================================================================

  defp format_to_type(:e4m3fn), do: @e4m3fn_type
  defp format_to_type(:e5m2), do: @e5m2_type

  defp float_type?(%Nx.Tensor{type: {t, _}}) when t in [:f, :bf, :f8_e4m3fn], do: true
  defp float_type?(_), do: false

  defp should_skip?(_key_path, []), do: false

  defp should_skip?(key_path, patterns) do
    path_str = Enum.join(key_path, ".")

    Enum.any?(patterns, fn pattern ->
      String.contains?(path_str, to_string(pattern))
    end)
  end

  defp deep_map(map, key_path, fun) when is_map(map) and not is_struct(map) do
    Map.new(map, fn
      {k, %Nx.Tensor{} = t} ->
        {k, fun.(t, key_path ++ [k])}

      {k, %{tensor: _, scale: _, original_type: _} = quantized} ->
        {k, fun.(quantized, key_path ++ [k])}

      {k, v} when is_map(v) ->
        {k, deep_map(v, key_path ++ [k], fun)}

      other ->
        other
    end)
  end

  defp deep_map(other, _key_path, _fun), do: other

  defp count_bytes(params, format) do
    _fp8_type = format_to_type(format)
    fp8_elem_size = 1

    flatten_tensors(params)
    |> Enum.reduce({0, 0, 0}, fn tensor, {orig_acc, quant_acc, count_acc} ->
      if float_type?(tensor) do
        num_elements = Nx.size(tensor)
        {_, bits} = Nx.type(tensor)
        orig_elem_size = div(bits, 8)

        {
          orig_acc + num_elements * orig_elem_size,
          quant_acc + num_elements * fp8_elem_size + 4,
          count_acc + 1
        }
      else
        {_, bits} = Nx.type(tensor)
        size = Nx.size(tensor) * div(bits, 8)
        {orig_acc + size, quant_acc + size, count_acc}
      end
    end)
  end

  defp flatten_tensors(map) when is_map(map) and not is_struct(map) do
    Enum.flat_map(map, fn
      {_k, %Nx.Tensor{} = t} -> [t]
      {_k, v} when is_map(v) -> flatten_tensors(v)
      _ -> []
    end)
  end

  defp flatten_tensors(_), do: []

  defp readable_size(n) when n < 1_000, do: "#{n} B"
  defp readable_size(n) when n < 1_000_000, do: "#{Float.round(n / 1_024, 2)} KB"
  defp readable_size(n), do: "#{Float.round(n / 1_048_576, 2)} MB"
end
