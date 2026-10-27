if Code.ensure_loaded?(EXLA.CustomCall) do
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
    # Edifice.CUDA.Library loads the separately built handlers before a
    # supported call target is emitted. EXLA itself does not need rebuilding.
    #
    # Scope: selective-scan forward + backward (f32). The GRU handlers were
    # dropped on 2026-10-26: the kernel (one block per batch row, R@h
    # recomputed from global memory each step) is 3-10x slower forward and
    # 10-50x slower backward than XLA's compiled fallback on an RTX 5090
    # (docs/kernel_tier_benchmarks.md).

    # ACTIVATION GATE: emitting a call target whose FFI handler is not
    # linked into the loaded EXLA library fails the whole XLA compile —
    # so the impl stays inert until EDIFICE_FUSED_CUSTOM_CALL=1 is set
    # explicitly after `mix edifice.cuda.build`. Default: every op :skip, exactly the
    # pre-defimpl behavior.

    alias EXLA.CustomCall.Spec

    defp enabled? do
      if System.get_env("EDIFICE_FUSED_CUSTOM_CALL") == "1" do
        Edifice.CUDA.Library.ensure_loaded!()
        true
      else
        false
      end
    end

    # :fused_selective_scan operands: [x {B,T,H}, dt {B,T,H}, A {H,S},
    # B {B,T,S}, C {B,T,S}]. Kernel keeps per-(batch,hidden) state in
    # registers sized MAX_STATE=32 and SILENTLY TRUNCATES beyond — the
    # state guard here is a correctness gate, not a perf one. No hidden
    # or seq_len limit (threads are independent; backward stages h_prev
    # through a global-memory workspace).
    @max_ssm_state 32

    def call(%{op: :fused_selective_scan}, out, [x, _dt, a | _] = args, %{platform: :cuda}) do
      with {:f, 32} <- x.type,
           {_h, s} <- a.shape,
           true <- s <= @max_ssm_state,
           true <- enabled?() do
        {:ok, spec("exla_fused_selective_scan_f32", args, out)}
      else
        _ -> :skip
      end
    end

    # :fused_selective_scan_backward operands: [x, dt, A, B, C, grad_output]
    def call(%{op: :fused_selective_scan_backward}, out, [x, _dt, a | _] = args, %{
          platform: :cuda
        }) do
      with {:f, 32} <- x.type,
           {_h, s} <- a.shape,
           true <- s <= @max_ssm_state,
           true <- enabled?() do
        {:ok, spec("exla_fused_selective_scan_backward_f32", args, out)}
      else
        _ -> :skip
      end
    end

    def call(_block, _out, _args, _client), do: :skip

    # LAYOUT PIN. EXLA emits `stablehlo.custom_call` without layout
    # constraints, so XLA's layout assignment may hand the handler operands
    # or results in whatever layout their consumers prefer (observed with
    # the former GRU handler: `f32[B,T,H]{2,0,1}`, i.e. T-major memory, for
    # the scan result feeding a reduce and a last-step slice). The kernels
    # read and write row-major and cannot tell. At batch 1 a permutation of
    # the batch axis is byte-identical, so the corruption only appears at
    # batch > 1 (the 2-layer GRU model was off by 7.2 at B=32 and exact at
    # B=1 on 2026-10-26). Pinning row-major on every operand and result
    # makes XLA insert the copies instead. `out` is nil when the protocol is
    # probed directly (tests); real lowerings always pass the output templates.
    defp spec(target, args, out) do
      attributes =
        [{"operand_layouts", layouts(args)}] ++
          case out && Nx.Defn.Composite.flatten_list([out]) do
            [_ | _] = outs -> [{"result_layouts", layouts(outs)}]
            _ -> []
          end

      %Spec{call_target_name: target, attributes: attributes}
    end

    defp layouts(templates), do: "[" <> Enum.map_join(templates, ", ", &row_major/1) <> "]"

    defp row_major(%{shape: shape}) do
      case tuple_size(shape) do
        0 -> "dense<> : tensor<0xindex>"
        rank -> "dense<[#{Enum.join((rank - 1)..0//-1, ", ")}]> : tensor<#{rank}xindex>"
      end
    end
  end
end
