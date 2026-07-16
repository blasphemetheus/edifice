defmodule Edifice.Stateful do
  @moduledoc """
  Behaviour for O(1) per-frame stateful inference on recurrent backbones.

  Full-sequence models re-encode a window of frames every frame — O(window)
  work for O(1) new information. Architectures with a linear recurrence
  (Mamba, GRU, MinGRU, GatedSSM, ...) admit constant-time stepping instead:
  carry a small recurrent state, feed one frame, get one output.

  ## The contract

    * `c:init_state/2` builds the zero (or model-defined initial) state for a
      batch. It receives the trained params because some architectures derive
      their initial hidden state from stored parameters.
    * `c:step/3` advances one frame: `[batch, embed_dim]` in,
      `{[batch, hidden], new_state}` out.

  **Correctness pin:** iterating `step/3` over a sequence must reproduce the
  architecture's full-sequence forward outputs (same params, same inputs) to
  ~1.0e-5 on `Nx.BinaryBackend`. See `test/edifice/stateful/` for the
  equivalence property tests that hold implementations to this.

  ## State is a plain Nx container

  Implementations must represent state as a (possibly nested) map whose
  leaves are all `Nx.Tensor`s — no config scalars, no functions. Everything
  a `step/3` needs must be derivable from the state tensors' shapes and the
  param map. This makes snapshot/restore trivial, which is the rollback
  primitive for netplay: frames get re-simulated when remote inputs arrive
  late, so the policy must restore its state to frame k and replay
  deterministically.

      state_k = Edifice.Stateful.snapshot(state)      # cheap copy
      binary  = Edifice.Stateful.serialize(state)     # wire format
      state   = Edifice.Stateful.deserialize(binary)  # restore

  ## Dispatch by architecture name

      params = ...                                    # trained params
      state = Edifice.init_state(:min_gru, params, build_opts ++ [batch_size: 1])
      {out, state} = Edifice.step(:min_gru, params, state, frame)
  """

  @typedoc "Nested map with Nx.Tensor leaves only."
  @type state :: map()

  @typedoc "Trained parameters: an `Axon.ModelState` or its plain data map."
  @type params :: Axon.ModelState.t() | map()

  @doc """
  Build the initial recurrent state.

  `opts` are the architecture's build options (the ones passed to `build/1`,
  or `spec.build_opts` from a checkpoint manifest) plus `:batch_size`
  (default 1).
  """
  @callback init_state(params(), opts :: keyword()) :: state()

  @doc """
  Advance one frame.

  `frame` is `[batch, embed_dim]`. Returns `{output, new_state}` where
  `output` is `[batch, hidden_size]` — the same tensor the full-sequence
  forward would emit at this position.
  """
  @callback step(params(), state(), frame :: Nx.Tensor.t()) ::
              {Nx.Tensor.t(), state()}

  @doc """
  Fetch (or compile and cache) a JIT-compiled `step/3` for a module.

  Eager op-by-op dispatch dominates step latency on accelerator backends
  (launch + sync per Nx op); compiling the whole step into one executable
  is what makes the 16.67 ms frame budget reachable. The compiled fun is
  cached in `:persistent_term` keyed by `{module, compiler}` — the
  compiler itself re-uses its compilation cache across calls with the
  same tensor shapes, so each (shapes, types) signature compiles once.

  The compiled fun lives OUTSIDE the state on purpose: state must remain
  a plain Nx container so the snapshot/serialize/restore rollback
  properties keep holding.

  Callers normally don't use this directly — pass `compiler:` to
  `Edifice.step/5` / `Edifice.init_state/3` instead.
  """
  @spec jit_step(module(), module()) :: (params(), state(), Nx.Tensor.t() -> {Nx.Tensor.t(), state()})
  def jit_step(module, compiler) do
    key = {__MODULE__, :jit_step, module, compiler}

    case :persistent_term.get(key, nil) do
      nil ->
        fun = Nx.Defn.jit(&module.step/3, compiler: compiler)
        :persistent_term.put(key, fun)
        fun

      fun ->
        fun
    end
  end

  @doc false
  def clear_jit_cache do
    for {{__MODULE__, :jit_step, _, _} = key, _} <- :persistent_term.get() do
      :persistent_term.erase(key)
    end

    :ok
  end

  @doc """
  Whether an architecture (atom or module) implements the stateful contract.
  """
  @spec stateful?(atom() | module()) :: boolean()
  def stateful?(arch_or_module) do
    module = resolve_module(arch_or_module)

    module != nil and Code.ensure_loaded?(module) and
      function_exported?(module, :step, 3) and
      function_exported?(module, :init_state, 2)
  end

  @doc """
  Cheap point-in-time copy of a state (the rollback snapshot primitive).

  Copies every tensor leaf off any device buffer via `Nx.backend_copy/1`,
  so later steps cannot mutate the snapshot through aliasing.
  """
  @spec snapshot(state()) :: state()
  def snapshot(state), do: map_leaves(state, &Nx.backend_copy/1)

  @doc """
  Serialize a state to a binary (the netplay wire format).

  Round-trips through `Nx.serialize/1`; `deserialize/1` restores a state
  that continues bit-identically on the same backend.
  """
  @spec serialize(state()) :: binary()
  def serialize(state) do
    state
    |> snapshot()
    |> Nx.serialize()
    |> IO.iodata_to_binary()
  end

  @doc """
  Restore a state serialized with `serialize/1`.
  """
  @spec deserialize(binary()) :: state()
  def deserialize(binary) when is_binary(binary), do: Nx.deserialize(binary)

  @doc false
  @spec assert_plain_container!(state()) :: :ok
  def assert_plain_container!(state) do
    map_leaves(state, fn
      %Nx.Tensor{} = t ->
        t

      other ->
        raise ArgumentError,
              "stateful state must contain only Nx.Tensor leaves, found: #{inspect(other)}"
    end)

    :ok
  end

  defp map_leaves(%Nx.Tensor{} = t, fun), do: fun.(t)

  defp map_leaves(%{} = map, fun) when not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, map_leaves(v, fun)} end)
  end

  defp map_leaves(other, fun), do: fun.(other)

  defp resolve_module(arch) when is_atom(arch) do
    # Try the registry first (arch atoms like :min_gru), fall back to
    # treating the atom as a module name.
    Edifice.module_for(arch)
  rescue
    ArgumentError -> if module_like?(arch), do: arch, else: nil
  end

  defp module_like?(atom) do
    match?("Elixir." <> _, Atom.to_string(atom))
  end
end
