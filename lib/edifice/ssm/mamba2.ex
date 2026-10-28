defmodule Edifice.SSM.Mamba2 do
  @moduledoc """
  Canonical Mamba-2 mixer with portable Nx execution and recurrent stepping.

  Implements the standard full-SSM block from state-spaces/mamba: joint
  `[z, x, B, C, dt]` projection, causal convolution over x/B/C, learned scalar
  decay per head, grouped B/C, learned D skip, and gated group RMS normalization.
  Unlike `MambaSSD`, A is learned and dt is not clipped after softplus.

  `build_block/2` is the reference mixer alone (no outer norm or residual).
  `build/1` stacks pre-RMS-normalized residual mixers for frame-vector inputs.
  Sequence execution projects and convolves the whole sequence before a compiled
  sequential recurrence. It does not use the upstream fused SSD CUDA algorithm.

  Supported reference configuration: full inner width in the SSM, per-head D,
  RMS normalization after SiLU gating, no dt limit, no packed sequences or
  tensor parallelism. Partial-SSM/MLP branches and alternate normalization modes
  are rejected rather than silently approximated. f32 is the validated compute
  type; reduced precision has not been certified.

  See `docs/mamba2.md` for reference provenance, weight conversion and validation.
  """
  import Nx.Defn
  alias Edifice.Stateful.Ops
  alias Edifice.Blocks.RMSNorm
  @behaviour Edifice.Stateful

  @doc """
  Build a frame-vector backbone. Options:

    * `:embed_dim` - input width (required)
    * `:hidden_size` - model width (default 256)
    * `:num_layers` - residual blocks (default 4)
    * `:state_size` - state coordinates per head (default 128)
    * `:head_dim` - channels per head (default 64)
    * `:num_groups` - shared B/C groups, dividing the head count (default 1)
    * `:expand_factor` - inner/model width ratio (default 2)
    * `:conv_size` - causal convolution width (default 4)
    * `:bias` / `:conv_bias` - projection/convolution bias (default false/true)
    * `:output_mode` - `:last` (default) or `:all`

  No dropout is inserted. A nonzero `:dropout` is rejected.
  """
  def build(opts \\ []) do
    cfg = config(opts)
    embed = Keyword.fetch!(opts, :embed_dim)
    layers = Keyword.get(opts, :num_layers, 4)

    unless is_integer(layers) and layers > 0,
      do: raise(ArgumentError, "num_layers must be positive")

    input = Axon.input("state_sequence", shape: {nil, nil, embed})

    x =
      if embed == cfg.width,
        do: input,
        else: Axon.dense(input, cfg.width, name: "input_projection")

    x =
      Enum.reduce(1..layers, x, fn i, x ->
        name = "mamba2_#{i}"
        norm = RMSNorm.layer(x, hidden_size: cfg.width, epsilon: 1.0e-5, name: "#{name}_norm")
        Axon.add(x, build_block(norm, Keyword.put(opts, :name, name)))
      end)

    x = RMSNorm.layer(x, hidden_size: cfg.width, epsilon: 1.0e-5, name: "final_norm")

    case Keyword.get(opts, :output_mode, :last) do
      :all -> x
      :last -> Axon.nx(x, &Nx.take(&1, Nx.axis_size(&1, 1) - 1, axis: 1))
      _ -> raise ArgumentError, "output_mode must be :last or :all"
    end
  end

  @doc "Build one reference mixer without an outer residual or normalization."
  def build_block(input, opts \\ []) do
    cfg = config(opts)
    projection = 2 * cfg.inner + 2 * cfg.groups * cfg.state + cfg.heads
    channels = cfg.inner + 2 * cfg.groups * cfg.state

    uniform = fn bound ->
      fn shape, type, key ->
        {tensor, _} = Nx.Random.uniform(key, -bound, bound, shape: shape, type: type)
        tensor
      end
    end

    param = fn name, shape, init -> Axon.param(name, shape, initializer: init) end

    bias = fn name, shape, enabled, bound ->
      if enabled, do: param.(name, shape, uniform.(bound)), else: Axon.constant(Nx.tensor(0.0))
    end

    a_init = fn shape, type, key ->
      {a, _} = Nx.Random.uniform(key, 1.0, 16.0, shape: shape, type: type)
      Nx.log(a)
    end

    dt_init = fn shape, type, key ->
      {v, _} = Nx.Random.uniform(key, :math.log(0.001), :math.log(0.1), shape: shape, type: type)
      dt = Nx.exp(v)
      Nx.add(dt, Nx.log(Nx.negate(Nx.expm1(Nx.negate(dt)))))
    end

    args = [
      input,
      param.("in_kernel", {cfg.width, projection}, uniform.(1 / :math.sqrt(cfg.width))),
      bias.("in_bias", {projection}, cfg.bias, 1 / :math.sqrt(cfg.width)),
      param.("conv_kernel", {cfg.conv, 1, channels}, uniform.(1 / :math.sqrt(cfg.conv))),
      bias.("conv_bias", {channels}, cfg.conv_bias, 1 / :math.sqrt(cfg.conv)),
      param.("A_log", {cfg.heads}, a_init),
      param.("dt_bias", {cfg.heads}, dt_init),
      param.("D", {cfg.heads}, :ones),
      param.("norm", {cfg.inner}, :ones),
      param.("out_kernel", {cfg.inner, cfg.width}, uniform.(1 / :math.sqrt(cfg.inner))),
      bias.("out_bias", {cfg.width}, cfg.bias, 1 / :math.sqrt(cfg.inner))
    ]

    Axon.layer(
      fn x, wi, bi, wc, bc, a, dt, d, norm, wo, bo, _opts ->
        p = %{
          "in_kernel" => wi,
          "in_bias" => bi,
          "conv_kernel" => wc,
          "conv_bias" => bc,
          "A_log" => a,
          "dt_bias" => dt,
          "D" => d,
          "norm" => norm,
          "out_kernel" => wo,
          "out_bias" => bo
        }

        initial = block_state(p, Nx.axis_size(x, 0), cfg.state, Nx.type(x))
        {y, _} = forward(p, x, initial)
        y
      end,
      args,
      name: Keyword.get(opts, :name, "mamba2"),
      op_name: :mamba2
    )
  end

  @doc "Convert an unsharded upstream mixer state dict (without a layer prefix) to block parameters."
  def from_reference(weights) do
    p = %{
      "in_kernel" => Nx.transpose(Map.fetch!(weights, "in_proj.weight")),
      "conv_kernel" => Nx.transpose(Map.fetch!(weights, "conv1d.weight"), axes: [2, 1, 0]),
      "out_kernel" => Nx.transpose(Map.fetch!(weights, "out_proj.weight")),
      "A_log" => Map.fetch!(weights, "A_log"),
      "dt_bias" => Map.fetch!(weights, "dt_bias"),
      "D" => Map.fetch!(weights, "D"),
      "norm" => Map.fetch!(weights, "norm.weight")
    }

    Enum.reduce(
      [{"in_proj.bias", "in_bias"}, {"conv1d.bias", "conv_bias"}, {"out_proj.bias", "out_bias"}],
      p,
      fn {from, to}, p ->
        if Map.has_key?(weights, from), do: Map.put(p, to, weights[from]), else: p
      end
    )
  end

  @doc "Zero state for a standalone mixer; convolution memory contains the previous k-1 projected inputs."
  def block_state(p, batch, state_size, type \\ :f32) do
    unless is_integer(batch) and batch > 0 and is_integer(state_size) and state_size > 0,
      do: raise(ArgumentError, "batch and state_size must be positive integers")

    heads = Nx.axis_size(p["A_log"], 0)
    inner = Nx.axis_size(p["out_kernel"], 0)
    {k, 1, channels} = Nx.shape(p["conv_kernel"])

    unless rem(inner, heads) == 0 and channels > inner and
             rem(channels - inner, 2 * state_size) == 0,
           do: raise(ArgumentError, "incompatible Mamba-2 state or projection dimensions")

    groups = div(channels - inner, 2 * state_size)
    unless rem(heads, groups) == 0, do: raise(ArgumentError, "B/C groups must divide heads")
    width = Nx.axis_size(p["out_kernel"], 1)

    for {key, shape} <- [
          {"in_kernel", {width, inner + channels + heads}},
          {"dt_bias", {heads}},
          {"D", {heads}},
          {"norm", {inner}}
        ] do
      unless Nx.shape(Map.fetch!(p, key)) == shape,
        do:
          raise(
            ArgumentError,
            "incompatible Mamba-2 parameter #{key}; expected #{inspect(shape)}"
          )
    end

    %{
      h: Nx.broadcast(Nx.tensor(0, type: type), {batch, heads, div(inner, heads), state_size}),
      conv: Nx.broadcast(Nx.tensor(0, type: type), {batch, max(k - 1, 1), channels})
    }
  end

  @doc "Run a sequence through one mixer with explicit initial/final state; accepts block parameters."
  def forward(p, x, state) do
    {batch, length, _} = Nx.shape(x)
    {_, heads, width, n} = Nx.shape(state.h)
    inner = heads * width
    channels = Nx.axis_size(state.conv, 2)
    groups = div(channels - inner, 2 * n)
    zxc = dense(x, p, "in")
    z = Nx.slice_along_axis(zxc, 0, inner, axis: 2)
    xc = Nx.slice_along_axis(zxc, inner, channels, axis: 2)

    dt =
      Nx.slice_along_axis(zxc, inner + channels, heads, axis: 2)
      |> Nx.add(p["dt_bias"])
      |> Ops.softplus()

    k = Nx.axis_size(p["conv_kernel"], 0)
    history = if k == 1, do: xc, else: Nx.concatenate([state.conv, xc], axis: 1)
    conv = if k == 1, do: state.conv, else: Nx.slice_along_axis(history, length, k - 1, axis: 1)

    activated =
      Axon.Layers.conv(history, p["conv_kernel"], Map.get(p, "conv_bias", 0.0),
        padding: :valid,
        feature_group_size: channels,
        channels: :last
      )
      |> Ops.silu()

    xh =
      activated
      |> Nx.slice_along_axis(0, inner, axis: 2)
      |> Nx.reshape({batch, length, heads, width})

    b = group_values(activated, inner, groups, n, heads)
    c = group_values(activated, inner + groups * n, groups, n, heads)
    {y, h} = scan(xh, b, c, dt, Nx.negate(Nx.exp(p["A_log"])), p["D"], state.h)
    y = Nx.reshape(y, {batch, length, inner})
    {finish(p, y, z, groups), %{h: h, conv: conv}}
  end

  defp group_values(x, offset, groups, n, heads) do
    {batch, length, _} = Nx.shape(x)
    indices = Nx.quotient(Nx.iota({heads}), div(heads, groups))

    x
    |> Nx.slice_along_axis(offset, groups * n, axis: 2)
    |> Nx.reshape({batch, length, groups, n})
    |> Nx.take(indices, axis: 2)
  end

  defnp scan(x, b, c, dt, a, d, initial) do
    output = Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), Nx.shape(x))

    {_, _, _, _, _, _, _, h, output} =
      while {t = 0, x, b, c, dt, a, d, h = initial, output}, t < Nx.axis_size(x, 1) do
        xt = Nx.take(x, t, axis: 1)
        delta = Nx.take(dt, t, axis: 1)
        decay = Nx.exp(delta * a) |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        drive = Nx.new_axis(delta, -1) * xt
        h = decay * h + Nx.new_axis(drive, -1) * Nx.new_axis(Nx.take(b, t, axis: 1), 2)

        y =
          Nx.sum(h * Nx.new_axis(Nx.take(c, t, axis: 1), 2), axes: [-1]) +
            Nx.reshape(d, {1, :auto, 1}) * xt

        output = Nx.put_slice(output, [0, t, 0, 0], Nx.new_axis(y, 1))
        {t + 1, x, b, c, dt, a, d, h, output}
      end

    {output, h}
  end

  defp finish(p, y, z, groups) do
    shape = Nx.shape(y)

    gated =
      Nx.multiply(y, Ops.silu(z))
      |> Nx.reshape(
        (Tuple.to_list(Tuple.delete_at(shape, tuple_size(shape) - 1)) ++
           [groups, div(Nx.axis_size(y, -1), groups)])
        |> List.to_tuple()
      )

    scaled =
      Nx.divide(
        gated,
        Nx.sqrt(Nx.add(Nx.mean(Nx.multiply(gated, gated), axes: [-1], keep_axes: true), 1.0e-5))
      )

    scaled |> Nx.reshape(shape) |> Nx.multiply(p["norm"]) |> dense(p, "out")
  end

  defp dense(x, p, name),
    do: Nx.add(Nx.dot(x, p["#{name}_kernel"]), Map.get(p, "#{name}_bias", 0.0))

  @doc "Advance a standalone mixer by one frame with constant-size carried state."
  def block_step(p, state, frame) do
    # Length-one convolution + recurrence preserves the same parameter contract.
    {y, state} = forward(p, Nx.new_axis(frame, 1), state)
    {Nx.squeeze(y, axes: [1]), state}
  end

  @impl true
  def init_state(params, opts \\ []) do
    cfg = config(opts)
    params = Ops.unwrap_params(params)
    layers = Keyword.get(opts, :num_layers, 4)
    batch = Keyword.get(opts, :batch_size, 1)

    unless is_integer(layers) and layers > 0,
      do: raise(ArgumentError, "num_layers must be positive")

    first =
      block_state(params["mamba2_1"], batch, cfg.state, Nx.type(params["mamba2_1"]["in_kernel"]))

    Map.new(first, fn {key, tensor} ->
      {key, Nx.stack(List.duplicate(tensor, layers), axis: 1)}
    end)
  end

  @impl true
  def step(params, state, frame) do
    params = Ops.unwrap_params(params)

    x =
      if params["input_projection"], do: Ops.dense(frame, params["input_projection"]), else: frame

    {y, states} =
      Enum.reduce(1..Nx.axis_size(state.h, 1), {x, []}, fn i, {x, states} ->
        name = "mamba2_#{i}"
        normed = RMSNorm.apply(x, params["#{name}_norm"]["#{name}_norm_gamma"], epsilon: 1.0e-5)
        s = Map.new(state, fn {key, value} -> {key, Nx.take(value, i - 1, axis: 1)} end)
        {y, s} = block_step(params[name], s, normed)
        {Nx.add(x, y), [s | states]}
      end)

    state =
      Map.new(state, fn {key, _} ->
        {key, states |> Enum.reverse() |> Enum.map(& &1[key]) |> Nx.stack(axis: 1)}
      end)

    {RMSNorm.apply(y, params["final_norm"]["final_norm_gamma"], epsilon: 1.0e-5), state}
  end

  def output_size(opts), do: Keyword.get(opts, :hidden_size, 256)

  defp config(opts) do
    width = Keyword.get(opts, :hidden_size, 256)
    expand = Keyword.get(opts, :expand_factor, 2)
    head = Keyword.get(opts, :head_dim, 64)
    groups = Keyword.get(opts, :num_groups, 1)
    state = Keyword.get(opts, :state_size, 128)
    conv = Keyword.get(opts, :conv_size, 4)

    unless Enum.all?([width, expand, head, groups, state, conv], &(is_integer(&1) and &1 > 0)),
      do: raise(ArgumentError, "Mamba-2 dimensions must be positive integers")

    inner = width * expand

    unless rem(inner, head) == 0 and rem(div(inner, head), groups) == 0,
      do:
        raise(ArgumentError, "head_dim must divide inner width and num_groups must divide heads")

    for {key, expected} <- [
          d_ssm: inner,
          rmsnorm: true,
          norm_before_gate: false,
          d_has_hdim: false,
          dt_limit: {0.0, :infinity},
          dropout: 0.0
        ] do
      unless Keyword.get(opts, key, expected) == expected,
        do: raise(ArgumentError, "unsupported Mamba-2 option #{key}")
    end

    %{
      width: width,
      inner: inner,
      heads: div(inner, head),
      groups: groups,
      state: state,
      conv: conv,
      bias: Keyword.get(opts, :bias, false),
      conv_bias: Keyword.get(opts, :conv_bias, true)
    }
  end
end
