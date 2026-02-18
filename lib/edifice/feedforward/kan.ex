defmodule Edifice.Feedforward.KAN do
  @moduledoc """
  KAN: Kolmogorov-Arnold Networks with learnable activation functions.

  Implements KAN from "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024).
  Based on the Kolmogorov-Arnold representation theorem: any multivariate
  continuous function can be represented as compositions of univariate functions.

  ## Key Innovation: Learnable Edge Activations

  Unlike MLPs with fixed activations on nodes, KAN has learnable activations on edges:

  ```
  MLP:  y = W2 * sigma(W1 * x)           # Fixed sigma (ReLU, etc.)
  KAN:  y = Sum_j Phi_j(x_j)             # Learnable Phi_j per edge
  ```

  Each edge activation is parameterized as:
  ```
  Phi(x) = w_base * SiLU(x) + w_spline * spline(x)
  ```

  ## Basis Function Options

  This implementation supports multiple basis functions:

  | Basis | Formula | Params | Speed |
  |-------|---------|--------|-------|
  | `:bspline` (default) | Sum c*B_k(x) (cubic B-spline) | O(oig) | Medium |
  | `:sine` | Sum A*sin(omega*x + phi) | O(oig) | Fast |
  | `:chebyshev` | Sum c*Tn(x) | O(oig) | Fast |
  | `:fourier` | Sum (a*cos + b*sin) | O(2oig) | Medium |
  | `:rbf` | Sum w*exp(-||x-mu||^2/2sigma^2) | O(oig) | Medium |

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |       KAN Block                      |
  |  LayerNorm -> KAN Layer -> Residual  |
  |  LayerNorm -> KAN Layer -> Residual  |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      # Build KAN backbone
      model = KAN.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        grid_size: 8,
        basis: :sine
      )

  ## Comparison with MLP

  | Aspect | MLP | KAN |
  |--------|-----|-----|
  | Activation | Fixed on nodes | Learnable on edges |
  | Interpretability | Low | High (visualizable) |
  | Parameters | O(n^2) | O(n^2*g) where g=grid |
  | Best for | General tasks | Symbolic/scientific |

  ## References
  - Paper: https://arxiv.org/abs/2404.19756
  - SineKAN: https://www.frontiersin.org/articles/10.3389/frai.2024.1462952
  - GitHub: https://github.com/KindXiaoming/pykan
  """

  require Axon
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default grid size (number of basis functions)"
  @spec default_grid_size() :: pos_integer()
  def default_grid_size, do: 8

  @doc "Default basis function type"
  @spec default_basis() :: atom()
  def default_basis, do: :bspline

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  @doc "Epsilon for numerical stability"
  @spec eps() :: float()
  def eps, do: 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a KAN model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of KAN blocks (default: 4)
    - `:grid_size` - Number of basis functions per edge (default: 8)
    - `:basis` - Basis function type: :bspline, :sine, :chebyshev, :fourier, :rbf (default: :bspline)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)
    - `:base_weight` - Weight for base SiLU activation (default: 0.5)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack KAN blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block_opts = Keyword.merge(opts, layer_idx: layer_idx)
        block = build_kan_block(acc, block_opts)

        # Dropout between blocks
        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "dropout_#{layer_idx}")
        else
          block
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # KAN Block
  # ============================================================================

  @doc """
  Build a single KAN block.

  KAN block structure:
  1. LayerNorm -> KAN Layer -> Residual
  2. LayerNorm -> KAN Layer (wider) -> Residual
  """
  @spec build_kan_block(Axon.t(), keyword()) :: Axon.t()
  def build_kan_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "kan_block_#{layer_idx}"

    # First KAN layer (same dimension)
    kan1_normed = Axon.layer_norm(input, name: "#{name}_norm1")
    kan1_out = build_kan_layer(kan1_normed, hidden_size, Keyword.put(opts, :name, "#{name}_kan1"))

    # Residual
    after_kan1 = Axon.add(input, kan1_out, name: "#{name}_residual1")

    # Second KAN layer (expand then contract, like FFN)
    kan2_normed = Axon.layer_norm(after_kan1, name: "#{name}_norm2")
    inner_size = hidden_size * 2

    kan2_up =
      build_kan_layer(kan2_normed, inner_size, Keyword.put(opts, :name, "#{name}_kan2_up"))

    kan2_down =
      build_kan_layer(kan2_up, hidden_size, Keyword.put(opts, :name, "#{name}_kan2_down"))

    # Residual
    Axon.add(after_kan1, kan2_down, name: "#{name}_residual2")
  end

  # ============================================================================
  # KAN Layer
  # ============================================================================

  @doc """
  Build a KAN layer with learnable edge activations.

  KAN layer computes:
  ```
  y_i = Sum_j Phi_ij(x_j)
  ```

  Where each Phi_ij is approximated as:
  ```
  Phi(x) = w_base * SiLU(x) + w_spline * Sum sin(omega*x)
  ```

  This implementation uses a combination of:
  1. Base activation: SiLU(x) for gradient flow
  2. Learnable activation: Multi-frequency sine basis projected through dense layers
  """
  @spec build_kan_layer(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def build_kan_layer(input, out_size, opts \\ []) do
    grid_size = Keyword.get(opts, :grid_size, default_grid_size())
    basis = Keyword.get(opts, :basis, default_basis())
    name = Keyword.get(opts, :name, "kan_layer")

    # Base activation path: linear + SiLU
    base = Axon.dense(input, out_size, name: "#{name}_base_proj")
    base_activated = Axon.activation(base, :silu, name: "#{name}_base_silu")

    # Spline/basis path: project to expanded space, apply basis fn, project back
    freq_size = out_size * grid_size
    freq_proj = Axon.dense(input, freq_size, name: "#{name}_freq_proj")

    # Apply selected basis function
    basis_activated =
      case basis do
        :bspline ->
          # Cubic B-spline basis evaluation over uniform knot grid
          # Project input to grid_size basis evaluations, then weight by learnable coefficients
          Axon.nx(
            freq_proj,
            fn x ->
              # x: [batch, seq_len, out_size * grid_size]
              # Treat as grid_size groups of out_size values
              # Map input to [0, 1] range for B-spline evaluation
              x_norm = Nx.sigmoid(x)
              bspline_basis_eval(x_norm, grid_size)
            end,
            name: "#{name}_bspline"
          )

        :sine ->
          Axon.nx(freq_proj, &Nx.sin/1, name: "#{name}_sine")

        :chebyshev ->
          # Chebyshev polynomials require input in [-1, 1]
          Axon.nx(
            freq_proj,
            fn x ->
              clamped = Nx.clip(x, -1.0, 1.0)
              Nx.cos(Nx.multiply(Nx.acos(clamped), 1.0))
            end,
            name: "#{name}_chebyshev"
          )

        :fourier ->
          # Fourier: both sin and cos components
          Axon.nx(
            freq_proj,
            fn x ->
              half = div(Nx.axis_size(x, -1), 2)
              sin_part = Nx.sin(Nx.slice_along_axis(x, 0, half, axis: -1))
              cos_part = Nx.cos(Nx.slice_along_axis(x, half, half, axis: -1))
              Nx.concatenate([sin_part, cos_part], axis: -1)
            end,
            name: "#{name}_fourier"
          )

        :rbf ->
          # RBF: exp(-x^2 / 2) — Gaussian basis
          Axon.nx(
            freq_proj,
            fn x ->
              Nx.exp(Nx.negate(Nx.divide(Nx.pow(x, 2), 2.0)))
            end,
            name: "#{name}_rbf"
          )
      end

    # Project back to output size (learns amplitude weights)
    spline_out = Axon.dense(basis_activated, out_size, name: "#{name}_spline_proj")

    # Learnable mixing weights: output = w_base * base + w_spline * spline
    base_weight_param =
      Axon.param("#{name}_base_w", {1, 1, out_size},
        initializer: fn shape, _type, _key -> Nx.broadcast(0.5, shape) end
      )

    spline_weight_param =
      Axon.param("#{name}_spline_w", {1, 1, out_size},
        initializer: fn shape, _type, _key -> Nx.broadcast(0.5, shape) end
      )

    combined =
      Axon.layer(
        fn base_val, spline_val, w_base, w_spline, _opts ->
          Nx.add(Nx.multiply(w_base, base_val), Nx.multiply(w_spline, spline_val))
        end,
        [base_activated, spline_out, base_weight_param, spline_weight_param],
        name: "#{name}_combine",
        op_name: :kan_combine
      )

    # Final layer norm for stability
    Axon.layer_norm(combined, name: "#{name}_norm")
  end

  # ============================================================================
  # Basis Functions (for reference/future use)
  # ============================================================================

  @doc """
  Compute sine basis functions.

  SineKAN: y = Sum A * sin(omega * x + phi)
  """
  defn sine_basis(x, frequencies, phases) do
    # x: [batch, seq, in_size]
    # frequencies: [grid_size] or [out, in, grid]
    # phases: [grid_size] or [out, in, grid]

    # Compute sin(omega * x + phi) for each frequency
    # Broadcasting: x[..., in] * freq[grid] -> [..., in, grid]
    x_expanded = Nx.new_axis(x, -1)
    angles = Nx.add(Nx.multiply(x_expanded, frequencies), phases)
    Nx.sin(angles)
  end

  @doc """
  Compute Chebyshev polynomial basis functions.

  ChebyKAN: y = Sum c * Tn(x)
  where T0(x) = 1, T1(x) = x, Tn+1(x) = 2x*Tn(x) - Tn-1(x)
  """
  defn chebyshev_basis(x, _order) do
    # x should be in [-1, 1] for Chebyshev
    x_clamped = Nx.clip(x, -1.0, 1.0)

    # Compute T_0 through T_order using recurrence
    # T_0 = 1, T_1 = x, T_{n+1} = 2*x*T_n - T_{n-1}
    t0 = Nx.broadcast(1.0, Nx.shape(x_clamped))
    t1 = x_clamped

    # Build up polynomial values
    # For simplicity, we compute a few fixed orders
    t2 = Nx.subtract(Nx.multiply(2.0, Nx.multiply(x_clamped, t1)), t0)
    t3 = Nx.subtract(Nx.multiply(2.0, Nx.multiply(x_clamped, t2)), t1)
    t4 = Nx.subtract(Nx.multiply(2.0, Nx.multiply(x_clamped, t3)), t2)

    # Stack basis functions
    Nx.stack([t0, t1, t2, t3, t4], axis: -1)
  end

  @doc """
  Compute RBF (Radial Basis Function) basis.

  y = Sum w * exp(-||x - mu||^2 / 2*sigma^2)
  """
  defn rbf_basis(x, centers, sigma) do
    # x: [batch, seq, in_size]
    # centers: [grid_size]
    # sigma: scalar

    x_expanded = Nx.new_axis(x, -1)
    centers_expanded = Nx.reshape(centers, {1, 1, 1, Nx.axis_size(centers, 0)})

    # Squared distance
    diff = Nx.subtract(x_expanded, centers_expanded)
    sq_dist = Nx.multiply(diff, diff)

    # RBF: exp(-d^2 / 2*sigma^2)
    Nx.exp(Nx.divide(Nx.negate(sq_dist), 2.0 * sigma * sigma))
  end

  # ============================================================================
  # B-spline Basis Evaluation
  # ============================================================================

  @doc false
  # Evaluate cubic B-spline basis functions on input in [0, 1].
  # Uses the unrolled Cox-de Boor recurrence for order 3 (cubic).
  # Returns weighted basis activations suitable for downstream projection.
  @spec bspline_basis_eval(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def bspline_basis_eval(x, grid_size) do
    # Uniform knot grid from 0 to 1 with (grid_size + 4) knots for cubic B-splines
    # This gives grid_size basis functions, each nonzero over 4 knot spans
    num_knots = grid_size + 4

    # Scale x to knot grid: x in [0, 1] -> position in knot spans
    # x: [batch, seq, features]
    x_scaled = Nx.multiply(x, (num_knots - 1) * 1.0)

    # For each basis function k, compute B_{k,3}(x)
    # Cubic B-spline on uniform grid: piecewise cubic polynomial
    # B_{k,3}(t) where t = x_scaled - k (local coordinate)
    # Nonzero for t in [0, 4), with pieces:
    #   [0,1): t^3 / 6
    #   [1,2): (-3t^3 + 12t^2 - 12t + 4) / 6
    #   [2,3): (3t^3 - 24t^2 + 60t - 44) / 6
    #   [3,4): (4-t)^3 / 6

    # Evaluate all grid_size basis functions in parallel
    # For each k in 0..grid_size-1, compute B_k(x_scaled)
    basis_values =
      for k <- 0..(grid_size - 1) do
        t = Nx.subtract(x_scaled, k * 1.0)

        # Clamp to [0, 4] for numerical safety
        t_clamped = Nx.clip(t, 0.0, 4.0)

        # Compute all four pieces
        p0 = Nx.divide(Nx.pow(t_clamped, 3), 6.0)

        t1 = Nx.subtract(t_clamped, 1.0)

        p1 =
          Nx.divide(
            Nx.add(
              Nx.add(
                Nx.multiply(-3.0, Nx.pow(t1, 3)),
                Nx.multiply(3.0, Nx.pow(t1, 2))
              ),
              Nx.add(Nx.multiply(3.0, t1), 1.0)
            ),
            6.0
          )

        t2 = Nx.subtract(t_clamped, 2.0)

        p2 =
          Nx.divide(
            Nx.add(
              Nx.add(
                Nx.multiply(3.0, Nx.pow(t2, 3)),
                Nx.multiply(-3.0, Nx.pow(t2, 2))
              ),
              Nx.add(Nx.multiply(-3.0, t2), 1.0)
            ),
            6.0
          )

        p3 = Nx.divide(Nx.pow(Nx.subtract(4.0, t_clamped), 3), 6.0)

        # Select the right piece based on which interval t falls in
        in_0_1 = Nx.logical_and(Nx.greater_equal(t, 0.0), Nx.less(t, 1.0))
        in_1_2 = Nx.logical_and(Nx.greater_equal(t, 1.0), Nx.less(t, 2.0))
        in_2_3 = Nx.logical_and(Nx.greater_equal(t, 2.0), Nx.less(t, 3.0))
        in_3_4 = Nx.logical_and(Nx.greater_equal(t, 3.0), Nx.less_equal(t, 4.0))

        val = Nx.multiply(Nx.select(in_0_1, 1.0, 0.0), p0)
        val = Nx.add(val, Nx.multiply(Nx.select(in_1_2, 1.0, 0.0), p1))
        val = Nx.add(val, Nx.multiply(Nx.select(in_2_3, 1.0, 0.0), p2))
        Nx.add(val, Nx.multiply(Nx.select(in_3_4, 1.0, 0.0), p3))
      end

    # Multiply each basis evaluation element-wise with x (to modulate)
    # and sum — the downstream projection layer learns the control point weights
    Enum.reduce(basis_values, Nx.broadcast(0.0, Nx.shape(x)), &Nx.add/2)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a KAN model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc """
  Calculate approximate parameter count for a KAN model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    grid_size = Keyword.get(opts, :grid_size, default_grid_size())

    # KAN layer params (using v2 implementation):
    # - base_proj: in * out
    # - freq_proj: in * (out * grid)
    # - spline_proj: (out * grid) * out
    kan_layer_params = fn in_size, out_size ->
      # base_proj
      # freq_proj
      # spline_proj
      in_size * out_size +
        in_size * (out_size * grid_size) +
        out_size * grid_size * out_size
    end

    # Per block:
    # - kan1: hidden -> hidden
    # - kan2_up: hidden -> 2*hidden
    # - kan2_down: 2*hidden -> hidden
    inner_size = hidden_size * 2

    block_params =
      kan_layer_params.(hidden_size, hidden_size) +
        kan_layer_params.(hidden_size, inner_size) +
        kan_layer_params.(inner_size, hidden_size)

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + num_layers * block_params
  end

  @doc """
  Get recommended defaults for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      grid_size: 8,
      basis: :bspline,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
