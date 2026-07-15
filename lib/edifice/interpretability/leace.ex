defmodule Edifice.Interpretability.LEACE do
  @moduledoc """
  LEACE (LEAst-squares Concept Erasure) — Belrose et al., ICML 2023.

  Removes ALL linear information about a concept `Z` from representations
  `X` with the least-squares-minimal affine edit:

      x' = x − W⁺ P W (x − μ)      P = U Uᵀ,  U = colspace(W Σ_xz)

  where `W = Σ_xx^(-1/2)` whitens X. After erasure, no *linear* probe can
  recover Z from X' (guaranteed — the cross-covariance of X' with Z is
  exactly zero), while everything linearly independent of Z is preserved
  as much as possible.

  ## Usage

      # Closed-form fit (no training): x {n, d}, z {n, k}
      eraser = LEACE.fit(x, z)

      # Erase in plain Nx
      x_clean = LEACE.erase(eraser, x)

      # Or as an Axon transformation with frozen constants (insert between
      # a trunk and downstream heads)
      model = trunk |> LEACE.apply_layer(eraser) |> heads

  ## Trainable variant

  `build_trainable/1` (aliased by `build/1` for the registry) builds the
  older learnable low-rank-subtraction scaffold (`x − x·W₁·W₂` with two
  unconstrained dense layers). NOTE: as shipped it is NOT LEACE — this
  library provides no fitting procedure for it, its random initialization
  erases a random subspace, and nothing constrains it to a projection. It
  is kept only as a building block for external adversarial-training
  setups. Prefer `fit/3` + `erase/2` / `apply_layer/2`.

  First production use: exphil (2026-07-14) erased the "copy signal"
  (trunk subspace predicting the policy's previous buttons) to cure
  causal-confusion pathologies in an imitation-learned Melee policy.

  ## References

  - Belrose et al., "LEACE: Perfect linear concept erasure in closed form"
    (ICML 2023)
  """

  @default_concept_dim 1

  @doc """
  Closed-form LEACE fit from representations `x` `{n, d}` and concept
  labels `z` `{n, k}` (any numeric types; centered internally).

  ## Options

    - `:shrinkage` - relative ridge added to Σ_xx for stable inversion
      (default 1.0e-4)
    - `:rank` - cap the erased subspace rank (default: numerical rank of
      the whitened cross-covariance, at most k)
    - `:decomp_backend` - backend for the {d, d} eigh/SVD stage (default
      `Nx.BinaryBackend`; accelerator backends may spend many minutes
      XLA-compiling the unrolled decomposition graphs)

  Returns `%{mu: {d} f32, a: {d, d} f32, rank: integer}` (BinaryBackend).

  Note: the O(n·d²) covariance products run on the INPUT tensors' backend —
  put `x`/`z` on your accelerator for large n.
  """
  def fit(x, z, opts \\ []) do
    shrink = Keyword.get(opts, :shrinkage, 1.0e-4)
    decomp_backend = Keyword.get(opts, :decomp_backend, Nx.BinaryBackend)

    x = Nx.as_type(x, :f64)
    z = Nx.as_type(z, :f64)

    n = Nx.axis_size(x, 0)
    d = Nx.axis_size(x, 1)

    mu_x = Nx.mean(x, axes: [0])
    mu_z = Nx.mean(z, axes: [0])
    xc = Nx.subtract(x, mu_x)
    zc = Nx.subtract(z, mu_z)

    sigma_xx = Nx.dot(Nx.transpose(xc), xc) |> Nx.divide(n)
    sigma_xz = Nx.dot(Nx.transpose(xc), zc) |> Nx.divide(n)

    sigma_xx = Nx.backend_transfer(sigma_xx, decomp_backend)
    sigma_xz = Nx.backend_transfer(sigma_xz, decomp_backend)
    mu_x = Nx.backend_transfer(mu_x, decomp_backend)

    prev_backend = Nx.default_backend()
    Nx.default_backend(decomp_backend)

    try do
      fit_small_stage(sigma_xx, sigma_xz, mu_x, d, shrink, opts)
    after
      Nx.default_backend(prev_backend)
    end
  end

  defp fit_small_stage(sigma_xx, sigma_xz, mu_x, d, shrink, opts) do
    mean_var = Nx.take_diagonal(sigma_xx) |> Nx.mean()
    ridge = Nx.multiply(mean_var, shrink)
    sigma_xx = Nx.add(sigma_xx, Nx.multiply(Nx.eye(d, type: :f64), ridge))

    # Whitening via CHOLESKY (Σ = LLᵀ, W = L⁻¹, W⁺ = L) — exact,
    # non-iterative triangular ops. eigh-based ZCA whitening fails at high
    # eigenvalue spread on iterative backends (silently under-converged
    # eigh produced a guarantee-violating eraser on real 256-d data,
    # 2026-07-14). The erasure guarantee is whitener-agnostic; the edit is
    # minimal in the L-whitened metric rather than exactly ZCA-minimal.
    l = Nx.LinAlg.cholesky(sigma_xx)
    eye = Nx.eye(d, type: :f64)
    w = Nx.LinAlg.triangular_solve(l, eye, lower: true)
    w_pinv = l

    wxz = Nx.dot(w, sigma_xz)
    {u, s, _vt} = Nx.LinAlg.svd(wxz, full_matrices?: false)

    tol = Nx.multiply(Nx.reduce_max(s), 1.0e-6)
    keep = Nx.greater(s, tol) |> Nx.as_type(:f64)

    keep =
      case Keyword.get(opts, :rank) do
        nil -> keep
        r -> Nx.multiply(keep, Nx.less(Nx.iota(Nx.shape(s)), r) |> Nx.as_type(:f64))
      end

    p = Nx.dot(Nx.multiply(u, Nx.reshape(keep, {1, :auto})), Nx.transpose(u))
    a = w_pinv |> Nx.dot(p) |> Nx.dot(w)

    %{
      mu: Nx.as_type(mu_x, :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      a: Nx.as_type(a, :f32) |> Nx.backend_transfer(Nx.BinaryBackend),
      rank: Nx.sum(keep) |> Nx.to_number() |> trunc()
    }
  end

  @doc """
  Erase the fitted concept from representations `{n, d}`:
  `x − (x − μ) Aᵀ`.
  """
  def erase(%{mu: mu, a: a}, x) do
    mu = Nx.as_type(mu, Nx.type(x))
    a = Nx.as_type(a, Nx.type(x))
    xc = Nx.subtract(x, mu)
    Nx.subtract(x, Nx.dot(xc, Nx.transpose(a)))
  end

  @doc """
  Wrap a fitted eraser as an Axon transformation over an existing node —
  frozen constants, no learnable parameters. Insert between a trunk and
  its heads:

      trunk |> LEACE.apply_layer(eraser) |> heads
  """
  def apply_layer(%Axon{} = input, %{mu: _, a: _} = eraser) do
    Axon.nx(input, fn x -> erase(eraser, x) end, name: "leace_erase")
  end

  @typedoc "Options for `build_trainable/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:concept_dim, pos_integer()}

  @doc """
  Build the TRAINABLE low-rank-subtraction scaffold (NOT closed-form
  LEACE — see the moduledoc). `x − x·W₁·W₂` with learnable, unconstrained
  `W₁ {d, k}`, `W₂ {k, d}`. Requires an external training objective
  (e.g. adversarial probe minimization) that this library does not
  provide; at initialization it subtracts a RANDOM subspace.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:concept_dim` - Dimension of the subtracted subspace
      (default: #{@default_concept_dim})

  Returns an Axon model mapping `[batch, input_size]` to
  `[batch, input_size]`.
  """
  @spec build_trainable([build_opt()]) :: Axon.t()
  def build_trainable(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    concept_dim = Keyword.get(opts, :concept_dim, @default_concept_dim)

    input = Axon.input("leace_input", shape: {nil, input_size})

    concept_proj = Axon.dense(input, concept_dim, name: "leace_concept_proj", use_bias: false)

    concept_recon =
      Axon.dense(concept_proj, input_size, name: "leace_concept_recon", use_bias: false)

    Axon.layer(
      fn x, recon, _opts -> Nx.subtract(x, recon) end,
      [input, concept_recon],
      name: "leace_erase",
      op_name: :concept_erase
    )
  end

  @doc """
  Alias for `build_trainable/1` (kept for the registry and existing
  callers). The name previously implied this was LEACE; it is only the
  trainable scaffold — see `build_trainable/1`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []), do: build_trainable(opts)

  @doc "Get the output size of the eraser (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
