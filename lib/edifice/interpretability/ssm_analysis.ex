defmodule Edifice.Interpretability.SSMAnalysis do
  @moduledoc """
  Analyses over captured SSM sites (`build_probe/1` -> `Capture.run/4`):

  - **Hidden-attention maps** (Ali, Zimerman & Wolf, arXiv:2403.01590):
    the selective scan is exactly a data-dependent lower-triangular
    attention `alpha[t,s] = C_t . (prod_{k=s+1}^t A_bar_k) B_bar_s` —
    per decision frame t, a weight over source frames s answering
    "which past inputs did this output depend on". Validatable against
    known reaction structure when the consumer has ground truth.

  - **Timescale map** (Stuffed-Mamba framing, arXiv:2410.07145): the
    per-channel effective decay `mean(A_bar)` over real data, and its
    horizon `-1/log(a)` in frames — which channels are reaction-speed
    and which hold episode memory.

  All functions consume captured `B`/`C`/`dt` site tensors
  (`{n, seq, state}` / `{n, seq, hidden}`) and mirror
  `Edifice.SSM.Common.discretize_ssm/4`'s math exactly (dt clamp, fixed
  negative-diagonal A, per-channel-dt B discretization). If Common's
  discretization changes, change this module with it. `dt` must be the
  post-softplus capture, before the discretization clamp.

  These are conditional scan coefficients with B/C/dt held fixed, not a
  full-model Jacobian or normalized attention probabilities. They describe
  the Common fixed-A recurrence with zero initial state, excluding
  convolution, gates, residual paths, and output projections. Fused scans
  must use the same fixed A and timestep clamp for these coefficients to apply.
  """

  import Nx.Defn

  alias Edifice.SSM.Common

  @dt_min Common.dt_min()
  @dt_max Common.dt_max()

  @doc """
  Hidden-attention rows for decision frame `t` (0-indexed): weight of
  each source frame `s <= t`, per row of the capture.

  Inputs are the captured sites for ONE layer: `b` `{n, seq, state}`,
  `c` `{n, seq, state}`, `dt` `{n, seq, hidden}`. Returns
  `{n, t + 1}` — attention mass per source frame, hidden/state
  contracted (`sum_h |sum_j C[t,j] . decay[t,s,h,j] . B_bar[s,j]|`).

  Frame convention matches capture row alignment: row i's frame t is
  the window's last position, so `t = window - 1` gives "what the
  decision at row i attended to inside its window".
  """
  def frame_attention(b, c, dt, t) do
    frame_kernel(b, c, dt, t) |> Nx.abs() |> Nx.sum(axes: [2])
  end

  @doc """
  Signed conditional scan coefficients `{batch, t + 1, hidden}`.

  With zero initial state, `sum_s kernel[s,h] * x[s,h]` reconstructs the
  Common scan output at frame `t`. Keeping the signs preserves cancellation
  that the `frame_attention/4` magnitude summary intentionally removes.
  Future frames are excluded before computation, including their gradients.
  """
  def frame_kernel(b, c, dt, t) do
    unless Nx.rank(b) == 3 and Nx.rank(c) == 3 and Nx.rank(dt) == 3,
      do: raise(ArgumentError, "B, C and dt must be rank-three capture tensors")

    {batch, seq, _state} = Nx.shape(b)

    unless Nx.shape(c) == Nx.shape(b) and Nx.axis_size(dt, 0) == batch and
             Nx.axis_size(dt, 1) == seq,
           do: raise(ArgumentError, "B, C and dt capture dimensions must agree")

    unless is_integer(t) and t >= 0 and t < seq,
      do: raise(ArgumentError, "t=#{inspect(t)} out of range for seq=#{seq}")

    frame_kernel_n(b, c, dt, t: t)
  end

  defnp frame_kernel_n(b, c, dt, opts \\ []) do
    t = opts[:t]
    state = Nx.axis_size(b, 2)
    b = b |> Nx.slice_along_axis(0, t + 1, axis: 1) |> Nx.as_type(:f32)
    c_t = c |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.as_type(:f32)

    dt =
      dt
      |> Nx.slice_along_axis(0, t + 1, axis: 1)
      |> Nx.as_type(:f32)
      |> Nx.clip(@dt_min, @dt_max)

    # Sum dt strictly AFTER each source s, through t. Computing log decay
    # directly avoids log(exp(dt*A)) underflow and future-frame overflow.
    after_source =
      dt
      |> Nx.reverse(axes: [1])
      |> Nx.cumulative_sum(axis: 1)
      |> Nx.reverse(axes: [1])

    after_source = Nx.max(after_source - dt, 0.0)
    a_diag = -(Nx.iota({state}) + 1.0)
    decay = Nx.exp(Nx.new_axis(after_source, 3) * Nx.reshape(a_diag, {1, 1, 1, state}))
    b_bar = Nx.new_axis(dt, 3) * Nx.new_axis(b, 2)
    Nx.sum(Nx.new_axis(c_t, 2) * decay * b_bar, axes: [3])
  end

  @doc """
  Per-channel timescale map from a captured `dt` site `{n, seq, hidden}`.

  Returns `%{a_bar_mean, horizon_frames}`, both `{hidden, state}`:
  mean decay factor per channel over all rows/frames, and the effective
  memory horizon `-1 / log(a_bar_mean)` in frames (frames until a
  stored value decays to 1/e under that mean multiplier). This is a summary
  of one-step decays, not an exact horizon for a time-varying trajectory.
  """
  def timescale_map(dt, state_size \\ nil) do
    state = state_size || Common.default_state_size()

    unless Nx.rank(dt) == 3 and is_integer(state) and state > 0,
      do: raise(ArgumentError, "dt must be rank three and state_size must be positive")

    {a_bar_mean, horizon} = timescale_n(dt, state: state)
    %{a_bar_mean: a_bar_mean, horizon_frames: horizon}
  end

  defnp timescale_n(dt, opts \\ []) do
    state = opts[:state]
    dt = dt |> Nx.as_type(:f32) |> Nx.clip(@dt_min, @dt_max)
    a_diag = -(Nx.iota({state}) + 1.0)
    log_a = Nx.new_axis(dt, 3) * Nx.reshape(a_diag, {1, 1, 1, state})
    # Keep the mean in log space so large state indices still have a finite
    # horizon even when exp(log_a) underflows to zero.
    samples = Nx.axis_size(dt, 0) * Nx.axis_size(dt, 1)
    log_mean = Nx.logsumexp(log_a, axes: [0, 1]) - Nx.log(samples)
    {Nx.exp(log_mean), -1.0 / log_mean}
  end
end
