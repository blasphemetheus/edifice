defmodule Edifice.Utils.Quantization do
  @moduledoc """
  Post-Training Quantization Toolkit.

  Provides utilities for quantizing trained model weights to lower precision,
  reducing model size and enabling faster inference. Supports three methods:

  ## Methods

  ### 1. RTN (Round-to-Nearest)
  Simplest baseline: scale weights to fit in target bit width, then round.
  Fast but can lose significant accuracy for aggressive quantization.

  ```
  w_quant = round(w / scale) * scale
  where scale = max(|w|) / (2^(bits-1) - 1)
  ```

  ### 2. GPTQ (Accurate Post-Training Quantization)
  Quantizes weights column-by-column, using the inverse Hessian to optimally
  distribute quantization error across remaining columns. Calibration quality
  determines how well this preserves model outputs at low precision.

  ```
  for each column j:
    quantize w[:,j]
    error = w[:,j] - w_quant[:,j]
    w[:,j+1:] += error * H_inv[j, j+1:] / H_inv[j,j]  (compensate)
  ```

  ### 3. AWQ (Activation-Aware Weight Quantization)
  Searches for per-channel scaling factors that minimize quantization error
  when weighted by activation magnitudes. Channels with large activations
  get more precision.

  ```
  scale = (act_magnitude)^alpha    (alpha searched in [0, 1])
  w_scaled = w * scale
  quantize w_scaled, then divide by scale at runtime
  ```

  ## Usage

      # Simple round-to-nearest at 8-bit
      quantized_params = Quantization.quantize_weights(params, bits: 8)

      # GPTQ with captured inputs to each dense layer
      calibration = %{["dense", "kernel"] => [dense_input_batch]}
      quantized_params = Quantization.gptq(params, calibration, bits: 4)

      # AWQ with activation statistics
      quantized_params = Quantization.awq(params, act_stats, bits: 4)

  ## References

  - Frantar et al., "GPTQ: Accurate Post-Training Quantization for GPT" (ICLR 2023)
  - Lin et al., "AWQ: Activation-aware Weight Quantization" (MLSys 2024)
  """

  @typedoc "Quantization method."
  @type method :: :rtn | :gptq | :awq

  @typedoc "Options for quantization."
  @type quant_opt ::
          {:bits, 2 | 3 | 4 | 8}
          | {:group_size, pos_integer()}
          | {:symmetric, boolean()}

  # ============================================================================
  # Round-to-Nearest (RTN) Quantization
  # ============================================================================

  @doc """
  Quantize model parameters using round-to-nearest.

  ## Parameters

    - `params` - Model parameters (nested map from `Axon.build` + training)
    - `opts` - Options:
      - `:bits` - Target bit width (default: 8)
      - `:group_size` - Group size for per-group quantization (default: -1, per-tensor)
      - `:symmetric` - Use symmetric quantization (default: true)
      - `:skip_patterns` - List of name patterns to skip (e.g., ["norm", "bias"])

  ## Returns

    A map with `:quantized_params` (quantized weights), `:scales` (per-tensor/group scales),
    and `:zero_points` (for asymmetric quantization).
  """
  @spec quantize_weights(map(), keyword()) :: map()
  def quantize_weights(params, opts \\ []) do
    bits = Keyword.get(opts, :bits, 8)
    symmetric = Keyword.get(opts, :symmetric, true)
    skip_patterns = Keyword.get(opts, :skip_patterns, ["bias"])

    {quantized, scales, zero_points} =
      deep_map_params(params, fn name, tensor ->
        if should_skip?(name, skip_patterns) or Nx.rank(tensor) < 2 do
          {tensor, nil, nil}
        else
          rtn_quantize_tensor(tensor, bits, symmetric)
        end
      end)

    %{
      quantized_params: quantized,
      scales: scales,
      zero_points: zero_points,
      bits: bits,
      method: :rtn
    }
  end

  # ============================================================================
  # GPTQ (Accurate Post-Training Quantization)
  # ============================================================================

  @doc """
  Quantize dense kernels using GPTQ with full Hessian error compensation.

  Kernels use Axon's `{input_features, output_features}` layout. Calibration
  is a map from full parameter paths to nonempty lists of input activation
  batches, e.g. `%{["dense", "kernel"] => [activations]}`. Each batch has shape
  `{..., input_features}`. Capture the actual inputs to each layer, not just
  the model's input; this function does not execute a model to collect them.
  A bare list of batches is accepted only when exactly one kernel is selected.
  Missing calibration raises rather than silently falling back to RTN.

  Uses the uncentered input covariance, diagonal damping, an upper Cholesky
  factor of its inverse, and sequential quantization error compensation.
  Unobserved input channels are zeroed. Only rank-two kernels are supported;
  skip other tensors explicitly using `:skip_patterns`.

  ## Options

    - `:bits` — 2, 3, 4 or 8 (default: 4), symmetric signed quantization
    - `:group_size` — contiguous input channels per static scale group
      (default: 128); `-1` uses all input channels. Scales are per output channel.
    - `:block_size` — input channels per error-update block (default: 128)
    - `:dampening` — positive fraction of mean Hessian diagonal (default: 0.01)
    - `:skip_patterns` — substrings of parameter paths to preserve (default: ["bias"])

  Returns the same structure as `quantize_weights/2`, consumable by
  `dequantize/1`. Codes are stored as s8 tensors (not bit-packed), with scales
  expanded to kernel shape. `bits` describes the quantization grid, not actual
  storage compression. This offline routine is not intended to run inside defn.
  """
  @spec gptq(map(), map() | list(Nx.Tensor.t()), keyword()) :: map()
  def gptq(params, calibration_inputs, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        bits: 4,
        group_size: 128,
        block_size: 128,
        dampening: 0.01,
        skip_patterns: ["bias"]
      )

    bits = opts[:bits]
    group_size = opts[:group_size]
    block_size = opts[:block_size]
    dampening = opts[:dampening]
    unless bits in [2, 3, 4, 8], do: raise(ArgumentError, "GPTQ bits must be 2, 3, 4 or 8")

    unless is_integer(group_size) and (group_size == -1 or group_size > 0),
      do: raise(ArgumentError, "GPTQ group_size must be -1 or a positive integer")

    unless is_integer(block_size) and block_size > 0,
      do: raise(ArgumentError, "GPTQ block_size must be a positive integer")

    unless is_number(dampening) and dampening > 0,
      do: raise(ArgumentError, "GPTQ dampening must be positive")

    selected = gptq_kernels(params, [], opts[:skip_patterns])

    calibration =
      cond do
        is_map(calibration_inputs) and not is_struct(calibration_inputs) ->
          calibration_inputs

        is_list(calibration_inputs) and length(selected) == 1 ->
          [{path, _}] = selected
          %{path => calibration_inputs}

        true ->
          raise ArgumentError,
                "GPTQ requires calibration keyed by full parameter paths; a list is only valid for one kernel"
      end

    results =
      Map.new(selected, fn {path, tensor} ->
        batches = Map.get(calibration, path)
        hessian = calibration_hessian!(batches, Nx.axis_size(tensor, 0), path)
        {path, gptq_quantize_tensor(tensor, hessian, bits, dampening, group_size, block_size)}
      end)

    {quantized, scales, zero_points} = gptq_rebuild(params, results, [])

    %{
      quantized_params: quantized,
      scales: scales,
      zero_points: zero_points,
      bits: bits,
      method: :gptq
    }
  end

  # ============================================================================
  # AWQ (Activation-Aware Weight Quantization)
  # ============================================================================

  @doc """
  Quantize parameters using activation-aware scaling.

  AWQ finds per-channel scaling factors that minimize quantization error
  when weighted by activation magnitudes.

  ## Parameters

    - `params` - Model parameters
    - `activation_stats` - Map of layer name to mean activation magnitudes
      (e.g., from running calibration data through the model)
    - `opts` - Options:
      - `:bits` - Target bit width (default: 4)
      - `:alpha_steps` - Number of alpha values to search (default: 20)
      - `:skip_patterns` - Name patterns to skip

  ## Returns

    Map with `:quantized_params`, `:scales`, `:zero_points`, and `:channel_scales`.
  """
  @spec awq(map(), map(), keyword()) :: map()
  def awq(params, activation_stats, opts \\ []) do
    bits = Keyword.get(opts, :bits, 4)
    alpha_steps = Keyword.get(opts, :alpha_steps, 20)
    skip_patterns = Keyword.get(opts, :skip_patterns, ["bias"])

    {quantized, scales, zero_points} =
      deep_map_params(params, fn name, tensor ->
        if should_skip?(name, skip_patterns) or Nx.rank(tensor) < 2 do
          {tensor, nil, nil}
        else
          act_mag = Map.get(activation_stats, name, nil)
          awq_quantize_tensor(tensor, act_mag, bits, alpha_steps)
        end
      end)

    %{
      quantized_params: quantized,
      scales: scales,
      zero_points: zero_points,
      bits: bits,
      method: :awq
    }
  end

  # ============================================================================
  # Dequantization
  # ============================================================================

  @doc """
  Dequantize a quantization result back to full-precision weights.

  Useful for inference when the backend doesn't support low-precision ops.
  """
  @spec dequantize(map()) :: map()
  def dequantize(%{quantized_params: qparams, scales: scales, zero_points: zps}) do
    deep_zip_map(qparams, scales, zps, fn qw, scale, zp ->
      if scale == nil do
        qw
      else
        if zp != nil and zp != 0 do
          Nx.multiply(Nx.subtract(qw, zp), scale)
        else
          Nx.multiply(qw, scale)
        end
      end
    end)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Compute the compression ratio for a quantization result.
  """
  @spec compression_ratio(map()) :: float()
  def compression_ratio(%{bits: bits}) do
    32.0 / bits
  end

  @doc """
  Estimate the memory savings in bytes for quantized vs original.
  """
  @spec memory_savings(map(), map()) :: %{
          original_bytes: integer(),
          quantized_bytes: integer(),
          saved_bytes: integer()
        }
  def memory_savings(%{bits: bits}, original_params) do
    total_elements = count_elements(original_params)
    # f32
    original_bytes = total_elements * 4
    quantized_bytes = div(total_elements * bits, 8)

    %{
      original_bytes: original_bytes,
      quantized_bytes: quantized_bytes,
      saved_bytes: original_bytes - quantized_bytes
    }
  end

  # ============================================================================
  # Private: RTN Implementation
  # ============================================================================

  defp rtn_quantize_tensor(tensor, bits, symmetric) do
    {qmin, qmax} = quant_range(bits, symmetric)

    if symmetric do
      abs_max = Nx.reduce_max(Nx.abs(tensor))
      scale = Nx.divide(abs_max, qmax)
      safe_scale = Nx.max(scale, 1.0e-8)

      quantized =
        tensor
        |> Nx.divide(safe_scale)
        |> Nx.round()
        |> Nx.clip(qmin, qmax)

      {quantized, safe_scale, nil}
    else
      wmin = Nx.reduce_min(tensor)
      wmax = Nx.reduce_max(tensor)
      range = Nx.subtract(wmax, wmin)
      scale = Nx.divide(range, qmax - qmin)
      safe_scale = Nx.max(scale, 1.0e-8)
      zero_point = Nx.round(Nx.negate(Nx.divide(wmin, safe_scale))) |> Nx.add(qmin)

      quantized =
        tensor
        |> Nx.divide(safe_scale)
        |> Nx.add(zero_point)
        |> Nx.round()
        |> Nx.clip(qmin, qmax)

      {quantized, safe_scale, zero_point}
    end
  end

  # ============================================================================
  # Private: GPTQ Implementation
  # ============================================================================

  # Algorithm reference: https://github.com/IST-DASLab/gptq/blob/main/gptq.py
  # Operate on Axon input rows (the reference GPTQ implementation uses columns).
  # Static groups keep scale selection independent of error-update block size.
  defp gptq_quantize_tensor(tensor, hessian, bits, dampening, group_size, block_size) do
    {n, outputs} = Nx.shape(tensor)
    weights = Nx.as_type(tensor, :f32)
    finite!(weights, "GPTQ weights")
    dead = Nx.equal(Nx.take_diagonal(hessian), 0)
    weights = Nx.select(Nx.broadcast(Nx.new_axis(dead, 1), {n, outputs}), 0.0, weights)
    hessian = Nx.add(hessian, Nx.make_diagonal(Nx.as_type(dead, :f32)))
    damp = Nx.multiply(dampening, Nx.mean(Nx.take_diagonal(hessian)))
    hessian = Nx.add(hessian, Nx.multiply(damp, Nx.eye(n)))
    upper = hessian |> Nx.LinAlg.invert() |> Nx.LinAlg.cholesky() |> Nx.transpose()
    finite!(upper, "GPTQ inverse Hessian factor")
    {qmin, qmax} = quant_range(bits, true)
    group_size = if group_size == -1, do: n, else: group_size

    scales =
      for start <- Enum.take_every(0..(n - 1), group_size) do
        count = min(group_size, n - start)

        weights
        |> Nx.slice_along_axis(start, count, axis: 0)
        |> Nx.abs()
        |> Nx.reduce_max(axes: [0], keep_axes: true)
        |> Nx.divide(qmax)
        |> Nx.max(1.0e-8)
        |> Nx.broadcast({count, outputs})
      end
      |> Nx.concatenate(axis: 0)

    {_, codes} =
      Enum.reduce(
        Enum.take_every(0..(n - 1), block_size),
        {weights, Nx.broadcast(Nx.s8(0), {n, outputs})},
        fn start, {w, codes} ->
          count = min(block_size, n - start)
          finish = start + count
          local = Nx.slice_along_axis(w, start, count, axis: 0)

          {_, codes, errors} =
            Enum.reduce(0..(count - 1), {local, codes, []}, fn offset, {local, q, errors} ->
              i = start + offset
              row = Nx.slice_along_axis(local, offset, 1, axis: 0)
              scale = Nx.slice_along_axis(scales, i, 1, axis: 0)
              code = row |> Nx.divide(scale) |> Nx.round() |> Nx.clip(qmin, qmax)
              error = Nx.divide(Nx.subtract(row, Nx.multiply(code, scale)), upper[i][i])
              factors = Nx.slice(upper, [i, i], [1, count - offset]) |> Nx.transpose()
              tail = Nx.slice_along_axis(local, offset, count - offset, axis: 0)
              local = Nx.put_slice(local, [offset, 0], Nx.subtract(tail, Nx.dot(factors, error)))
              {local, Nx.put_slice(q, [i, 0], Nx.as_type(code, :s8)), [error | errors]}
            end)

          w =
            if finish < n do
              errors = errors |> Enum.reverse() |> Nx.concatenate(axis: 0)
              factors = Nx.slice(upper, [start, finish], [count, n - finish]) |> Nx.transpose()
              tail = Nx.slice_along_axis(w, finish, n - finish, axis: 0)
              Nx.put_slice(w, [finish, 0], Nx.subtract(tail, Nx.dot(factors, errors)))
            else
              w
            end

          {w, codes}
        end
      )

    {codes, scales, nil}
  end

  defp calibration_hessian!(batches, features, path) do
    unless is_list(batches) and batches != [],
      do: raise(ArgumentError, "missing or empty GPTQ calibration for #{inspect(path)}")

    {sum, samples} =
      Enum.reduce(batches, {Nx.broadcast(0.0, {features, features}), 0}, fn batch, {sum, count} ->
        unless is_struct(batch, Nx.Tensor) and Nx.rank(batch) >= 2 and
                 Nx.axis_size(batch, -1) == features,
               do:
                 raise(
                   ArgumentError,
                   "GPTQ calibration for #{inspect(path)} must have shape {..., #{features}}"
                 )

        x = batch |> Nx.as_type(:f32) |> Nx.reshape({:auto, features})
        finite!(x, "GPTQ calibration for #{inspect(path)}")
        {Nx.add(sum, Nx.dot(Nx.transpose(x), x)), count + Nx.axis_size(x, 0)}
      end)

    hessian = Nx.divide(sum, samples)
    finite!(hessian, "GPTQ Hessian for #{inspect(path)}")
    hessian
  end

  defp finite!(tensor, label) do
    if Nx.any(Nx.logical_or(Nx.is_nan(tensor), Nx.is_infinity(tensor))) |> Nx.to_number() == 1,
      do: raise(ArgumentError, "#{label} must be finite; check calibration and dampening")
  end

  defp gptq_kernels(params, prefix, skip_patterns) do
    Enum.flat_map(params, fn {key, value} ->
      path = prefix ++ [key]

      cond do
        is_map(value) and not is_struct(value) ->
          gptq_kernels(value, path, skip_patterns)

        should_skip?(Enum.join(path, "."), skip_patterns) or Nx.rank(value) < 2 ->
          []

        Nx.rank(value) == 2 ->
          [{path, value}]

        true ->
          raise ArgumentError,
                "GPTQ supports only dense rank-two kernels; skip #{inspect(path)} explicitly"
      end
    end)
  end

  defp gptq_rebuild(params, results, prefix) do
    Enum.reduce(params, {%{}, %{}, %{}}, fn {key, value}, {q, s, z} ->
      path = prefix ++ [key]

      {qv, sv, zv} =
        if is_map(value) and not is_struct(value),
          do: gptq_rebuild(value, results, path),
          else: Map.get(results, path, {value, nil, nil})

      {Map.put(q, key, qv), Map.put(s, key, sv), Map.put(z, key, zv)}
    end)
  end

  # ============================================================================
  # Private: AWQ Implementation
  # ============================================================================

  defp awq_quantize_tensor(tensor, act_magnitudes, bits, alpha_steps) do
    {qmin, qmax} = quant_range(bits, true)

    if act_magnitudes == nil do
      # No activation info, fall back to RTN
      rtn_quantize_tensor(tensor, bits, true)
    else
      # Search for optimal alpha
      best_alpha = search_optimal_alpha(tensor, act_magnitudes, bits, alpha_steps)

      # Compute per-channel scale: s = act_mag^alpha
      channel_scale = Nx.pow(Nx.max(act_magnitudes, 1.0e-6), best_alpha)

      # Scale weights
      scaled_tensor = Nx.multiply(tensor, Nx.reshape(channel_scale, {:auto, 1}))

      # Quantize scaled weights
      abs_max = Nx.reduce_max(Nx.abs(scaled_tensor))
      scale = Nx.divide(abs_max, qmax)
      safe_scale = Nx.max(scale, 1.0e-8)

      quantized =
        scaled_tensor
        |> Nx.divide(safe_scale)
        |> Nx.round()
        |> Nx.clip(qmin, qmax)

      # The effective scale includes channel_scale for dequant:
      # dequant = quantized * safe_scale / channel_scale
      {quantized, safe_scale, nil}
    end
  end

  defp search_optimal_alpha(tensor, act_mag, bits, alpha_steps) do
    # Grid search over alpha in [0, 1]
    {qmin, qmax} = quant_range(bits, true)

    alphas = for i <- 0..alpha_steps, do: i / alpha_steps

    {best_alpha, _best_error} =
      Enum.reduce(alphas, {0.5, :infinity}, fn alpha, {best_a, best_e} ->
        channel_scale = Nx.pow(Nx.max(act_mag, 1.0e-6), alpha)
        scaled = Nx.multiply(tensor, Nx.reshape(channel_scale, {:auto, 1}))

        abs_max = Nx.reduce_max(Nx.abs(scaled))
        scale = Nx.max(Nx.divide(abs_max, qmax), 1.0e-8)

        quantized = scaled |> Nx.divide(scale) |> Nx.round() |> Nx.clip(qmin, qmax)
        dequantized = Nx.multiply(quantized, scale)

        # Unscale
        restored = Nx.divide(dequantized, Nx.reshape(channel_scale, {:auto, 1}))
        error = Nx.mean(Nx.pow(Nx.subtract(tensor, restored), 2)) |> Nx.to_number()

        if best_e == :infinity or error < best_e do
          {alpha, error}
        else
          {best_a, best_e}
        end
      end)

    best_alpha
  end

  # ============================================================================
  # Private: Helpers
  # ============================================================================

  defp quant_range(bits, symmetric) do
    if symmetric do
      qmax = :math.pow(2, bits - 1) - 1
      {-qmax, qmax}
    else
      {0, :math.pow(2, bits) - 1}
    end
  end

  defp should_skip?(name, patterns) do
    Enum.any?(patterns, fn pattern ->
      String.contains?(to_string(name), pattern)
    end)
  end

  defp deep_map_params(params, fun) when is_map(params) do
    Enum.reduce(params, {%{}, %{}, %{}}, fn {key, value}, {q_acc, s_acc, z_acc} ->
      if is_map(value) and not is_struct(value, Nx.Tensor) do
        {q_inner, s_inner, z_inner} = deep_map_params(value, fun)
        {Map.put(q_acc, key, q_inner), Map.put(s_acc, key, s_inner), Map.put(z_acc, key, z_inner)}
      else
        {quantized, scale, zp} = fun.(key, value)
        {Map.put(q_acc, key, quantized), Map.put(s_acc, key, scale), Map.put(z_acc, key, zp)}
      end
    end)
  end

  defp deep_zip_map(qparams, scales, zps, fun) when is_map(qparams) do
    Map.new(qparams, fn {key, qval} ->
      sval = Map.get(scales, key)
      zval = Map.get(zps, key)

      if is_map(qval) and not is_struct(qval, Nx.Tensor) do
        {key, deep_zip_map(qval, sval || %{}, zval || %{}, fun)}
      else
        {key, fun.(qval, sval, zval)}
      end
    end)
  end

  defp count_elements(params) when is_map(params) do
    Enum.reduce(params, 0, fn {_key, value}, acc ->
      if is_map(value) and not is_struct(value, Nx.Tensor) do
        acc + count_elements(value)
      else
        acc + Nx.size(value)
      end
    end)
  end
end
