; ModuleID = '__compute_module_multiply_erf_fusion_kernel_module'
source_filename = "__compute_module_multiply_erf_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @multiply_erf_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 1
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 2
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  call void @multiply_erf_fusion_wrapped(ptr %5, ptr %7, i64 %11, i64 %13, i64 %15)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @multiply_erf_fusion_wrapped(ptr noalias align 64 dereferenceable(44) %0, ptr noalias align 64 dereferenceable(44) %1, i64 %2, i64 %3, i64 %4) #1 {
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ %29, %9 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 11
  br i1 %8, label %9, label %30

9:                                                ; preds = %6
  %10 = getelementptr inbounds [11 x float], ptr %0, i32 0, i64 %7
  %11 = load float, ptr %10, align 4, !invariant.load !3
  %12 = fmul float %11, 0x3FE6A09E60000000
  %13 = call float @llvm.maximum.f32(float %12, float 0xC00DF38D00000000)
  %14 = call float @llvm.minimum.f32(float %13, float 0x400DF38D00000000)
  %15 = fmul float %14, %14
  %16 = call float @llvm.fma.f32(float 0x3F2E05AA20000000, float %15, float 0x3F6BEBB440000000)
  %17 = call float @llvm.fma.f32(float %16, float %15, float 0x3FAA16DD60000000)
  %18 = call float @llvm.fma.f32(float %17, float %15, float 0x3FC7B4E800000000)
  %19 = call float @llvm.fma.f32(float %18, float %15, float 0x3FF20DD740000000)
  %20 = fmul float %14, %19
  %21 = call float @llvm.fma.f32(float 0xBE7FA720C0000000, float %15, float 0x3EF8B11BE0000000)
  %22 = call float @llvm.fma.f32(float %21, float %15, float 0x3F50ADA500000000)
  %23 = call float @llvm.fma.f32(float %22, float %15, float 0x3F8CD0FA80000000)
  %24 = call float @llvm.fma.f32(float %23, float %15, float 0x3FBC698420000000)
  %25 = call float @llvm.fma.f32(float %24, float %15, float 0x3FDFD68940000000)
  %26 = call float @llvm.fma.f32(float %25, float %15, float 1.000000e+00)
  %27 = fdiv float %20, %26
  %28 = getelementptr inbounds [11 x float], ptr %1, i32 0, i64 %7
  store float %27, ptr %28, align 4
  %29 = add i64 %7, 1
  br label %6

30:                                               ; preds = %6
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimum.f32(float, float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 44}
