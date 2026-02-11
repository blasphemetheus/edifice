defmodule Edifice.Utils.FusedOps do
  @moduledoc """
  Fused operations for GPU kernel optimization.

  ## Overview

  Modern GPUs are often memory-bandwidth limited rather than compute-limited.
  By fusing multiple operations into a single kernel, we:
  - Read input tensors once instead of multiple times
  - Write output tensors once instead of multiple times
  - Reduce kernel launch overhead
  - Enable better register utilization

  ## Example: Unfused vs Fused

  ```
  # Unfused (3 memory accesses per element):
  x = dense(input)      # read input, write x
  x = add_bias(x, b)    # read x, write x
  x = relu(x)           # read x, write x

  # Fused (1 memory access per element):
  x = dense_bias_relu(input, w, b)  # read input, write x
  ```

  ## Available Fusions

  - `dense_activation/4` - Dense + bias + activation in one pass
  - `layernorm_activation/5` - LayerNorm + activation
  - `fused_ffn/6` - Full FFN block (dense -> activation -> dense)
  - `gated_dense/4` - Gated linear unit (dense * sigmoid(dense))
  - `fused_softmax/2` - Numerically stable softmax with max subtraction

  ## Speedup Expectations

  | Fusion | Memory Reduction | Expected Speedup |
  |--------|------------------|------------------|
  | dense_activation | 33% | 15-25% |
  | layernorm_activation | 40% | 20-30% |
  | fused_ffn | 50% | 25-35% |
  | gated_dense | 50% | 30-40% |

  ## Note on Activation Dispatch

  Since atoms can't be passed to JIT-compiled functions, activation functions
  are dispatched at call time (not JIT time). This still allows XLA to fuse
  the operations within each specific activation path.
  """

  import Nx.Defn

  # ==========================================================================
  # Dense + Activation Fusions
  # ==========================================================================

  @doc """
  Fused dense layer with activation.

  Combines: `activation(input @ weight + bias)` into a single operation.

  ## Arguments

  - `input` - Input tensor of shape `[batch, in_features]` or `[batch, seq, in_features]`
  - `weight` - Weight matrix of shape `[in_features, out_features]`
  - `bias` - Bias vector of shape `[out_features]` (or nil for no bias)
  - `activation` - Activation function atom (:relu, :silu, :gelu, :sigmoid, :tanh)

  ## Examples

      iex> input = Nx.iota({2, 4}, type: :f32)
      iex> weight = Nx.broadcast(0.1, {4, 8})
      iex> bias = Nx.broadcast(0.0, {8})
      iex> FusedOps.dense_activation(input, weight, bias, :relu)
  """
  @spec dense_activation(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def dense_activation(input, weight, bias, activation) do
    # Dispatch to specific defn based on activation (compile-time dispatch)
    case activation do
      :relu -> dense_relu(input, weight, bias)
      :silu -> dense_silu(input, weight, bias)
      :gelu -> dense_gelu(input, weight, bias)
      :sigmoid -> dense_sigmoid(input, weight, bias)
      :tanh -> dense_tanh(input, weight, bias)
      :softplus -> dense_softplus(input, weight, bias)
      :identity -> dense_identity(input, weight, bias)
      :none -> dense_identity(input, weight, bias)
      _ -> dense_identity(input, weight, bias)
    end
  end

  # Specific defn implementations for each activation
  defn dense_relu(input, weight, bias) do
    input |> Nx.dot(weight) |> Nx.add(bias) |> Nx.max(0)
  end

  defn dense_silu(input, weight, bias) do
    x = input |> Nx.dot(weight) |> Nx.add(bias)
    Nx.multiply(x, Nx.sigmoid(x))
  end

  defn dense_gelu(input, weight, bias) do
    x = input |> Nx.dot(weight) |> Nx.add(bias)
    gelu_impl(x)
  end

  defn dense_sigmoid(input, weight, bias) do
    input |> Nx.dot(weight) |> Nx.add(bias) |> Nx.sigmoid()
  end

  defn dense_tanh(input, weight, bias) do
    input |> Nx.dot(weight) |> Nx.add(bias) |> Nx.tanh()
  end

  defn dense_softplus(input, weight, bias) do
    x = input |> Nx.dot(weight) |> Nx.add(bias)
    Nx.log1p(Nx.exp(x))
  end

  defn dense_identity(input, weight, bias) do
    input |> Nx.dot(weight) |> Nx.add(bias)
  end

  @doc """
  Fused dense without bias, directly into activation.

  Slightly faster than `dense_activation/4` when bias is not needed.
  """
  @spec dense_activation_no_bias(Nx.Tensor.t(), Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def dense_activation_no_bias(input, weight, activation) do
    case activation do
      :relu -> dense_relu_no_bias(input, weight)
      :silu -> dense_silu_no_bias(input, weight)
      :gelu -> dense_gelu_no_bias(input, weight)
      :sigmoid -> dense_sigmoid_no_bias(input, weight)
      :tanh -> dense_tanh_no_bias(input, weight)
      _ -> Nx.dot(input, weight)
    end
  end

  defn dense_relu_no_bias(input, weight) do
    input |> Nx.dot(weight) |> Nx.max(0)
  end

  defn dense_silu_no_bias(input, weight) do
    x = Nx.dot(input, weight)
    Nx.multiply(x, Nx.sigmoid(x))
  end

  defn dense_gelu_no_bias(input, weight) do
    input |> Nx.dot(weight) |> gelu_impl()
  end

  defn dense_sigmoid_no_bias(input, weight) do
    input |> Nx.dot(weight) |> Nx.sigmoid()
  end

  defn dense_tanh_no_bias(input, weight) do
    input |> Nx.dot(weight) |> Nx.tanh()
  end

  # ==========================================================================
  # LayerNorm + Activation Fusions
  # ==========================================================================

  @doc """
  Fused layer normalization with activation.

  Combines normalization, scale/shift, and activation into a single pass.
  Uses FP32 internally for numerical stability (important for BF16 training).

  ## Arguments

  - `input` - Input tensor
  - `gamma` - Scale parameter (per-feature)
  - `beta` - Shift parameter (per-feature)
  - `activation` - Activation function atom
  - `opts` - Options:
    - `:epsilon` - Small constant for numerical stability (default: 1.0e-5)
    - `:axis` - Axis to normalize over (default: -1)

  ## Implementation Note

  Even when input is BF16, we compute mean/variance in FP32 to avoid
  precision loss, then cast back to input precision for the output.
  """
  @spec layernorm_activation(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), atom(), keyword()) ::
          Nx.Tensor.t()
  def layernorm_activation(input, gamma, beta, activation, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, 1.0e-5)
    # Note: axis is always -1 for fused layer norm (standard for layer normalization)
    # This is required because defn needs compile-time constant axes

    case activation do
      :relu -> layernorm_relu_impl(input, gamma, beta, epsilon)
      :silu -> layernorm_silu_impl(input, gamma, beta, epsilon)
      :gelu -> layernorm_gelu_impl(input, gamma, beta, epsilon)
      :identity -> fused_layernorm_impl(input, gamma, beta, epsilon)
      :none -> fused_layernorm_impl(input, gamma, beta, epsilon)
      _ -> fused_layernorm_impl(input, gamma, beta, epsilon)
    end
  end

  # Layer norm implementations with hardcoded axis=-1 (last axis)
  # This is the standard for layer normalization across features
  defn layernorm_relu_impl(input, gamma, beta, epsilon) do
    input_type = Nx.type(input)
    x = Nx.as_type(input, :f32)
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.variance(x, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(variance, epsilon)))
    scaled = Nx.add(Nx.multiply(normalized, gamma), beta)
    activated = Nx.max(scaled, 0)
    Nx.as_type(activated, input_type)
  end

  defn layernorm_silu_impl(input, gamma, beta, epsilon) do
    input_type = Nx.type(input)
    x = Nx.as_type(input, :f32)
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.variance(x, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(variance, epsilon)))
    scaled = Nx.add(Nx.multiply(normalized, gamma), beta)
    activated = Nx.multiply(scaled, Nx.sigmoid(scaled))
    Nx.as_type(activated, input_type)
  end

  defn layernorm_gelu_impl(input, gamma, beta, epsilon) do
    input_type = Nx.type(input)
    x = Nx.as_type(input, :f32)
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.variance(x, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(variance, epsilon)))
    scaled = Nx.add(Nx.multiply(normalized, gamma), beta)
    activated = gelu_impl(scaled)
    Nx.as_type(activated, input_type)
  end

  @doc """
  Fused layer normalization without activation.

  Useful as a building block or when you need normalized values
  before a custom operation.

  Note: Uses axis=-1 (last axis) which is the standard for layer normalization.
  """
  @spec fused_layernorm(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def fused_layernorm(input, gamma, beta, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, 1.0e-5)
    # axis option is accepted but ignored - always uses -1 for compile-time optimization
    fused_layernorm_impl(input, gamma, beta, epsilon)
  end

  defn fused_layernorm_impl(input, gamma, beta, epsilon) do
    input_type = Nx.type(input)

    # Cast to FP32 for stable computation
    x = Nx.as_type(input, :f32)

    # Compute mean and variance (always over last axis)
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.variance(x, axes: [-1], keep_axes: true)

    # Normalize, scale, shift
    normalized = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(variance, epsilon)))
    result = Nx.add(Nx.multiply(normalized, gamma), beta)

    # Cast back
    Nx.as_type(result, input_type)
  end

  # ==========================================================================
  # Feed-Forward Network (FFN) Fusion
  # ==========================================================================

  @doc """
  Fused feed-forward network block.

  Combines the common FFN pattern: `dense -> activation -> dense` into an
  optimized sequence. This is the standard pattern in transformer FFN layers.

  ## Arguments

  - `input` - Input tensor `[batch, seq, hidden]` or `[batch, hidden]`
  - `w1` - First weight matrix `[hidden, ffn_dim]` (expansion)
  - `b1` - First bias `[ffn_dim]`
  - `w2` - Second weight matrix `[ffn_dim, hidden]` (projection)
  - `b2` - Second bias `[hidden]`
  - `activation` - Activation function (typically :gelu or :silu)

  ## Memory Efficiency

  Standard unfused FFN:
  - Read input, write intermediate1 (after dense1)
  - Read intermediate1, write intermediate2 (after activation)
  - Read intermediate2, write output (after dense2)
  - Total: 6 tensor accesses

  Fused FFN with XLA optimization:
  - XLA can often fuse dense1+activation into single kernel
  - Reduces to ~4 tensor accesses
  - Register-level optimization for activation
  """
  @spec fused_ffn(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          atom()
        ) :: Nx.Tensor.t()
  def fused_ffn(input, w1, b1, w2, b2, activation) do
    case activation do
      :relu -> fused_ffn_relu(input, w1, b1, w2, b2)
      :silu -> fused_ffn_silu(input, w1, b1, w2, b2)
      :gelu -> fused_ffn_gelu(input, w1, b1, w2, b2)
      _ -> fused_ffn_identity(input, w1, b1, w2, b2)
    end
  end

  defn fused_ffn_relu(input, w1, b1, w2, b2) do
    hidden = input |> Nx.dot(w1) |> Nx.add(b1) |> Nx.max(0)
    hidden |> Nx.dot(w2) |> Nx.add(b2)
  end

  defn fused_ffn_silu(input, w1, b1, w2, b2) do
    x = input |> Nx.dot(w1) |> Nx.add(b1)
    hidden = Nx.multiply(x, Nx.sigmoid(x))
    hidden |> Nx.dot(w2) |> Nx.add(b2)
  end

  defn fused_ffn_gelu(input, w1, b1, w2, b2) do
    hidden = input |> Nx.dot(w1) |> Nx.add(b1) |> gelu_impl()
    hidden |> Nx.dot(w2) |> Nx.add(b2)
  end

  defn fused_ffn_identity(input, w1, b1, w2, b2) do
    hidden = input |> Nx.dot(w1) |> Nx.add(b1)
    hidden |> Nx.dot(w2) |> Nx.add(b2)
  end

  @doc """
  Fused FFN without biases.

  Some architectures (like LLaMA) don't use biases in FFN layers.
  """
  @spec fused_ffn_no_bias(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def fused_ffn_no_bias(input, w1, w2, activation) do
    case activation do
      :relu -> fused_ffn_relu_no_bias(input, w1, w2)
      :silu -> fused_ffn_silu_no_bias(input, w1, w2)
      :gelu -> fused_ffn_gelu_no_bias(input, w1, w2)
      _ -> input |> Nx.dot(w1) |> Nx.dot(w2)
    end
  end

  defn fused_ffn_relu_no_bias(input, w1, w2) do
    input |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2)
  end

  defn fused_ffn_silu_no_bias(input, w1, w2) do
    x = Nx.dot(input, w1)
    x |> Nx.multiply(Nx.sigmoid(x)) |> Nx.dot(w2)
  end

  defn fused_ffn_gelu_no_bias(input, w1, w2) do
    input |> Nx.dot(w1) |> gelu_impl() |> Nx.dot(w2)
  end

  # ==========================================================================
  # Gated Linear Unit (GLU) Fusions
  # ==========================================================================

  @doc """
  Fused Gated Linear Unit.

  Computes: `activation(x @ w1 + b1) * (x @ w2 + b2)`

  This is used in SwiGLU (SiLU gate), GeGLU (GELU gate), and ReGLU (ReLU gate)
  which are common in modern transformers like LLaMA, PaLM, etc.

  ## Arguments

  - `input` - Input tensor
  - `w_gate` - Weight for the gate path
  - `w_up` - Weight for the up-projection path
  - `activation` - Gate activation (:silu for SwiGLU, :gelu for GeGLU)

  ## Implementation

  Instead of two separate dense layers followed by element-wise multiply,
  we structure the computation to enable better memory access patterns.
  """
  @spec gated_linear_unit(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def gated_linear_unit(input, w_gate, w_up, activation) do
    case activation do
      :silu -> swiglu(input, w_gate, w_up)
      :gelu -> geglu(input, w_gate, w_up)
      :relu -> reglu(input, w_gate, w_up)
      _ -> swiglu(input, w_gate, w_up)
    end
  end

  @doc """
  Fused SwiGLU (SiLU-gated linear unit).

  Convenience function for the most common GLU variant.
  Used in LLaMA, Mistral, and other modern LLMs.
  """
  defn swiglu(input, w_gate, w_up) do
    gate = Nx.dot(input, w_gate)
    gate_activated = Nx.multiply(gate, Nx.sigmoid(gate))
    up = Nx.dot(input, w_up)
    Nx.multiply(gate_activated, up)
  end

  @doc """
  Fused GeGLU (GELU-gated linear unit).

  Used in some transformer variants.
  """
  defn geglu(input, w_gate, w_up) do
    gate = Nx.dot(input, w_gate)
    gate_activated = gelu_impl(gate)
    up = Nx.dot(input, w_up)
    Nx.multiply(gate_activated, up)
  end

  @doc """
  Fused ReGLU (ReLU-gated linear unit).
  """
  defn reglu(input, w_gate, w_up) do
    gate = Nx.dot(input, w_gate)
    gate_activated = Nx.max(gate, 0)
    up = Nx.dot(input, w_up)
    Nx.multiply(gate_activated, up)
  end

  # ==========================================================================
  # Softmax and Attention Fusions
  # ==========================================================================

  @doc """
  Fused stable softmax.

  Combines max subtraction (for numerical stability) with exp and normalization
  in a way that XLA can optimize into fewer memory passes.

  ## Implementation

  Standard softmax: 4 passes (max, subtract, exp, sum, divide)
  Fused softmax: XLA can reduce to 2-3 passes with better register usage

  Note: Always computes over axis=-1 (last axis), which is standard for softmax
  over class logits.
  """
  @spec fused_softmax(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def fused_softmax(logits, _opts \\ []) do
    # axis option is accepted but ignored - always uses -1 for compile-time optimization
    fused_softmax_impl(logits)
  end

  defn fused_softmax_impl(logits) do
    input_type = Nx.type(logits)

    # Compute in FP32 for numerical stability
    logits_f32 = Nx.as_type(logits, :f32)

    # Max subtraction for stability (prevents overflow in exp)
    max_logits = Nx.reduce_max(logits_f32, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits_f32, max_logits)

    # Exp and normalize
    exp_shifted = Nx.exp(shifted)
    sum_exp = Nx.sum(exp_shifted, axes: [-1], keep_axes: true)
    probs = Nx.divide(exp_shifted, sum_exp)

    # Cast back
    Nx.as_type(probs, input_type)
  end

  @doc """
  Fused log-softmax.

  More numerically stable than `log(softmax(x))` and enables better fusion.
  Used in cross-entropy loss computation.

  log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

  Note: Always computes over axis=-1 (last axis).
  """
  @spec fused_log_softmax(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def fused_log_softmax(logits, _opts \\ []) do
    # axis option is accepted but ignored - always uses -1 for compile-time optimization
    fused_log_softmax_impl(logits)
  end

  defn fused_log_softmax_impl(logits) do
    input_type = Nx.type(logits)

    # Compute in FP32
    logits_f32 = Nx.as_type(logits, :f32)

    # Stable log-softmax
    max_logits = Nx.reduce_max(logits_f32, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits_f32, max_logits)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    log_probs = Nx.subtract(shifted, log_sum_exp)

    # Cast back
    Nx.as_type(log_probs, input_type)
  end

  @doc """
  Fused scaled dot-product attention scores.

  Computes: `softmax(Q @ K^T / sqrt(d_k))`

  Fuses the scaling, matmul, and softmax operations.
  """
  @spec fused_attention_scores(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def fused_attention_scores(query, key, opts \\ []) do
    mask = Keyword.get(opts, :mask, nil)

    if mask do
      fused_attention_scores_masked(query, key, mask)
    else
      fused_attention_scores_unmasked(query, key)
    end
  end

  defn fused_attention_scores_unmasked(query, key) do
    # Get head dimension for scaling
    d_k = Nx.axis_size(query, -1)
    scale = Nx.rsqrt(Nx.as_type(d_k, Nx.type(query)))

    # Q @ K^T with scaling (fused)
    scores = query
    |> Nx.dot([3], [0, 1], key, [3], [0, 1])
    |> Nx.multiply(scale)

    # Softmax over keys dimension (axis=-1)
    fused_softmax_impl(scores)
  end

  defn fused_attention_scores_masked(query, key, mask) do
    d_k = Nx.axis_size(query, -1)
    scale = Nx.rsqrt(Nx.as_type(d_k, Nx.type(query)))

    scores = query
    |> Nx.dot([3], [0, 1], key, [3], [0, 1])
    |> Nx.multiply(scale)
    |> Nx.add(mask)

    fused_softmax_impl(scores)
  end

  # ==========================================================================
  # SSM (State Space Model) Fusions
  # ==========================================================================

  @doc """
  Fused discretization step for SSM.

  Computes: `A_bar = exp(dt * A)` and `B_bar = dt * B`

  These operations happen every timestep in selective scan SSM models.
  """
  defn fused_ssm_discretize(dt, a, b) do
    # A_bar = exp(dt * A) - the decay factor
    a_bar = Nx.exp(Nx.multiply(dt, a))

    # B_bar = dt * B - the input scaling
    b_bar = Nx.multiply(dt, b)

    {a_bar, b_bar}
  end

  @doc """
  Fused SSM output computation.

  Computes: `y = sum(C * h)` where h is the hidden state.
  Fuses the element-wise multiply and reduction.
  """
  defn fused_ssm_output(c, hidden_state) do
    # Element-wise multiply and sum over state dimension
    # c: [batch, seq, state]
    # hidden_state: [batch, seq, hidden, state]
    # output: [batch, seq, hidden]
    Nx.sum(Nx.multiply(c, hidden_state), axes: [-1])
  end

  # ==========================================================================
  # Activation Functions
  # ==========================================================================

  @doc """
  Apply activation function by atom.

  This uses runtime dispatch to select the appropriate defn implementation,
  which still allows XLA to inline and optimize each activation path.
  """
  @spec apply_activation(Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def apply_activation(x, activation) do
    case activation do
      :relu -> apply_relu(x)
      :silu -> apply_silu(x)
      :gelu -> apply_gelu(x)
      :gelu_approx -> apply_gelu_approx(x)
      :sigmoid -> apply_sigmoid(x)
      :tanh -> apply_tanh(x)
      :softplus -> apply_softplus(x)
      :identity -> x
      :none -> x
      _ -> x
    end
  end

  defn apply_relu(x), do: Nx.max(x, 0)
  defn apply_silu(x), do: Nx.multiply(x, Nx.sigmoid(x))
  defn apply_gelu(x), do: gelu_impl(x)
  defn apply_gelu_approx(x), do: gelu_approx_impl(x)
  defn apply_sigmoid(x), do: Nx.sigmoid(x)
  defn apply_tanh(x), do: Nx.tanh(x)
  defn apply_softplus(x), do: Nx.log1p(Nx.exp(x))

  # Exact GELU: x * CDF_normal(x) where CDF is the CDF of standard normal
  defnp gelu_impl(x) do
    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Nx.multiply(x, Nx.multiply(0.5, Nx.add(1.0, Nx.erf(Nx.multiply(x, 0.7071067811865476)))))
  end

  # Approximate GELU (faster, used by GPT-2)
  defnp gelu_approx_impl(x) do
    # GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    inner = Nx.multiply(0.7978845608028654, Nx.add(x, Nx.multiply(0.044715, Nx.pow(x, 3))))
    Nx.multiply(0.5, Nx.multiply(x, Nx.add(1.0, Nx.tanh(inner))))
  end

  # ==========================================================================
  # Utility Functions
  # ==========================================================================

  @doc """
  Check if an activation function is supported.
  """
  @spec supported_activation?(atom()) :: boolean()
  def supported_activation?(activation) do
    activation in [:relu, :silu, :gelu, :gelu_approx, :sigmoid, :tanh, :softplus, :identity, :none]
  end

  @doc """
  List all supported activation functions.
  """
  @spec supported_activations() :: [atom()]
  def supported_activations do
    [:relu, :silu, :gelu, :gelu_approx, :sigmoid, :tanh, :softplus, :identity, :none]
  end
end
