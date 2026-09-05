defimpl EXLA.CustomCall, for: Edifice.Block.FusedOp do
  # The in-graph native tier for Edifice's fused CUDA kernels (the
  # "plugs in later" impl promised in Edifice.Block's moduledoc).
  #
  # Registering this defimpl flips `Edifice.Block.native_impl?/0` to
  # true, which lights up the :custom_call arm of every FusedScan
  # dispatch — so each op clause here must EXACTLY mirror what its
  # kernel supports and `:skip` everything else (a :skip compiles the
  # pure default fun in-graph, same as before this impl existed).
  #
  # The call targets ("exla_fused_<op>_<precision>") must be linked
  # into the loaded EXLA native library (the fork compiles
  # native/cuda/*.cu with -DEXLA_FFI). A target emitted here without
  # its handler registered fails at XLA compile time with an unknown
  # custom-call error — loud, not silent.
  #
  # Scope (2026-09-05): GRU forward + backward only (f32), the pair
  # verified end-to-end via carry-backbone/backward-kernel tests.
  # Constraints mirror the kernels: hidden <= 1024 (whole hidden dim in
  # one CUDA block); backward additionally seq_len <= 256 (MAX_SEQ_LEN
  # thread-local arrays).

  # ACTIVATION GATE: emitting a call target whose FFI handler is not
  # linked into the loaded EXLA library fails the whole XLA compile —
  # so the impl stays inert until EDIFICE_FUSED_CUSTOM_CALL=1 is set
  # explicitly (i.e., you are running an EXLA fork built with the
  # native/cuda kernels). Default: every op :skip, exactly the
  # pre-defimpl behavior.

  alias EXLA.CustomCall.Spec

  @max_hidden 1024
  @max_backward_seq 256

  defp enabled?, do: System.get_env("EDIFICE_FUSED_CUSTOM_CALL") == "1"

  # :fused_gru_scan operands: [wx {B,T,3H}, R {H,3H}, h0 {B,H}, bhn {H}]
  def call(%{op: :fused_gru_scan}, _out, [wx | _], %{platform: :cuda}) do
    with true <- enabled?(),
         {:f, 32} <- wx.type,
         {_b, _t, h3} <- wx.shape,
         true <- div(h3, 3) <= @max_hidden do
      {:ok, %Spec{call_target_name: "exla_fused_gru_scan_f32"}}
    else
      _ -> :skip
    end
  end

  # :fused_gru_scan_backward operands:
  # [wx, R, h0, bhn, forward_out, grad_output]
  def call(%{op: :fused_gru_scan_backward}, _out, [wx | _], %{platform: :cuda}) do
    with true <- enabled?(),
         {:f, 32} <- wx.type,
         {_b, t, h3} <- wx.shape,
         true <- div(h3, 3) <= @max_hidden and t <= @max_backward_seq do
      {:ok, %Spec{call_target_name: "exla_fused_gru_scan_backward_f32"}}
    else
      _ -> :skip
    end
  end

  # :fused_selective_scan operands: [x {B,T,H}, dt {B,T,H}, A {H,S},
  # B {B,T,S}, C {B,T,S}]. Kernel keeps per-(batch,hidden) state in
  # registers sized MAX_STATE=32 and SILENTLY TRUNCATES beyond — the
  # state guard here is a correctness gate, not a perf one. No hidden
  # or seq_len limit (threads are independent; backward stages h_prev
  # through a global-memory workspace).
  @max_ssm_state 32

  def call(%{op: :fused_selective_scan}, _out, [x, _dt, a | _], %{platform: :cuda}) do
    with true <- enabled?(),
         {:f, 32} <- x.type,
         {_h, s} <- a.shape,
         true <- s <= @max_ssm_state do
      {:ok, %Spec{call_target_name: "exla_fused_selective_scan_f32"}}
    else
      _ -> :skip
    end
  end

  # :fused_selective_scan_backward operands: [x, dt, A, B, C, grad_output]
  def call(%{op: :fused_selective_scan_backward}, _out, [x, _dt, a | _], %{platform: :cuda}) do
    with true <- enabled?(),
         {:f, 32} <- x.type,
         {_h, s} <- a.shape,
         true <- s <= @max_ssm_state do
      {:ok, %Spec{call_target_name: "exla_fused_selective_scan_backward_f32"}}
    else
      _ -> :skip
    end
  end

  def call(_block, _out, _args, _client), do: :skip
end
