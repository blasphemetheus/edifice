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

  All functions consume the RAW captured `B`/`C`/`dt` site tensors
  (`{n, seq, state}` / `{n, seq, hidden}`) and mirror
  `Edifice.SSM.Common.discretize_ssm/4`'s math exactly (dt clamp, fixed
  negative-diagonal A, mean-dt B discretization). If Common's
  discretization changes, change this module with it.
  """

  import Nx.Defn

  alias Edifice.SSM.Common

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
    seq = Nx.axis_size(b, 1)

    if t >= seq do
      raise ArgumentError, "t=#{t} out of range for seq=#{seq}"
    end

    frame_attention_n(b, c, dt, t: t)
  end

  defnp frame_attention_n(b, c, dt, opts \\ []) do
    t = opts[:t]
    state = Nx.axis_size(b, 2)

    dt = Nx.clip(dt, Common.dt_min(), Common.dt_max())

    # A_bar = exp(dt * a_diag): {n, seq, hidden, state}
    a_diag = -(Nx.iota({state}) + 1.0)
    a_bar = Nx.exp(Nx.new_axis(dt, 3) * Nx.reshape(a_diag, {1, 1, 1, state}))

    # log-space cumulative decay along seq
    log_cum = Nx.cumulative_sum(Nx.log(a_bar), axis: 1)

    # decay from s (exclusive) to t: exp(log_cum[t] - log_cum[s])
    # {n, 1, hidden, state} - {n, seq, hidden, state} -> broadcast over s
    log_at_t = Nx.take(log_cum, Nx.tensor([0]) + t, axis: 1)
    decay = Nx.exp(log_at_t - log_cum)

    # B_bar[s] = mean-dt discretization (mirrors discretize_ssm)
    dt_mean = Nx.mean(dt, axes: [2], keep_axes: true)
    b_bar = Nx.new_axis(dt_mean, 3) * Nx.new_axis(b, 2)

    # alpha[s, h] = sum_j C[t, j] * decay[s, h, j] * B_bar[s, 1->h, j]
    c_t = c |> Nx.take(Nx.tensor([0]) + t, axis: 1) |> Nx.new_axis(2)
    contrib = Nx.sum(c_t * decay * b_bar, axes: [3])

    # Attention mass per source frame: |.| summed over hidden
    mass = contrib |> Nx.abs() |> Nx.sum(axes: [2])

    # Causal: zero out s > t, keep s in 0..t
    Nx.slice_along_axis(mass, 0, t + 1, axis: 1)
  end

  @doc """
  Per-channel timescale map from a captured `dt` site `{n, seq, hidden}`.

  Returns `%{a_bar_mean, horizon_frames}`, both `{hidden, state}`:
  mean decay factor per channel over all rows/frames, and the effective
  memory horizon `-1 / log(a_bar_mean)` in frames (frames until a
  stored value decays to 1/e).
  """
  def timescale_map(dt, state_size \\ nil) do
    state = state_size || Common.default_state_size()
    {a_bar_mean, horizon} = timescale_n(dt, state: state)
    %{a_bar_mean: a_bar_mean, horizon_frames: horizon}
  end

  defnp timescale_n(dt, opts \\ []) do
    state = opts[:state]
    dt = Nx.clip(dt, Common.dt_min(), Common.dt_max())
    hidden = Nx.axis_size(dt, 2)

    a_diag = -(Nx.iota({state}) + 1.0)

    a_bar = Nx.exp(Nx.new_axis(dt, 3) * Nx.reshape(a_diag, {1, 1, 1, state}))
    a_mean = Nx.mean(a_bar, axes: [0, 1])

    horizon = -1.0 / Nx.log(Nx.clip(a_mean, 1.0e-12, 1.0 - 1.0e-7))

    {Nx.reshape(a_mean, {hidden, state}), Nx.reshape(horizon, {hidden, state})}
  end
end
