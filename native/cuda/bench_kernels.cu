// Standalone latency benchmark for fused scan kernels.
//
// Measures kernel execution time using CUDA events (GPU-side timing).
// Reports average, min, and median over multiple iterations.
//
// Build: nvcc -arch=sm_75 -o build/bench_kernels bench_kernels.cu build/fused_mingru_scan.o build/fused_minlstm_scan.o build/fused_native_rec_scan.o build/fused_liquid_scan.o -O3
// Run:   ./build/bench_kernels

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

extern "C" int fused_mingru_scan_launch(
    cudaStream_t stream,
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden);

extern "C" int fused_minlstm_scan_launch(
    cudaStream_t stream,
    const float* forget_gates, const float* input_gates,
    const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden);

extern "C" int fused_elu_gru_scan_launch(
    cudaStream_t stream,
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden);

extern "C" int fused_real_gru_scan_launch(
    cudaStream_t stream,
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden);

extern "C" int fused_diag_linear_scan_launch(
    cudaStream_t stream,
    const float* a_vals, const float* b_vals, const float* h0,
    float* output, int batch, int seq_len, int hidden);

extern "C" int fused_liquid_scan_launch(
    cudaStream_t stream,
    const float* tau, const float* activation, const float* h0,
    float* output, int batch, int seq_len, int hidden);

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct BenchResult {
    float avg_ms;
    float min_ms;
    float median_ms;
};

// ============================================================================
// Generic 2-input benchmark (MinGRU, ELU-GRU, Real-GRU, DiagLinear, Liquid)
// ============================================================================

typedef int (*scan_2input_launch_fn)(cudaStream_t, const float*, const float*,
                                      const float*, float*, int, int, int);

BenchResult bench_2input(scan_2input_launch_fn launch_fn,
                         int batch, int seq_len, int hidden,
                         int warmup, int iters,
                         float in1_lo, float in1_hi,
                         float in2_lo, float in2_hi) {
    int bth = batch * seq_len * hidden;
    int bh = batch * hidden;

    float *d_in1, *d_in2, *d_h0, *d_output;
    CUDA_CHECK(cudaMalloc(&d_in1, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in2, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h0, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bth * sizeof(float)));

    float* h_buf = (float*)malloc(bth * sizeof(float));
    srand(42);
    for (int i = 0; i < bth; i++) h_buf[i] = (float)rand() / RAND_MAX * (in1_hi - in1_lo) + in1_lo;
    CUDA_CHECK(cudaMemcpy(d_in1, h_buf, bth * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < bth; i++) h_buf[i] = (float)rand() / RAND_MAX * (in2_hi - in2_lo) + in2_lo;
    CUDA_CHECK(cudaMemcpy(d_in2, h_buf, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_h0, 0, bh * sizeof(float)));
    free(h_buf);

    for (int i = 0; i < warmup; i++) {
        launch_fn(0, d_in1, d_in2, d_h0, d_output, batch, seq_len, hidden);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(iters);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_fn(0, d_in1, d_in2, d_h0, d_output, batch, seq_len, hidden);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_h0); cudaFree(d_output);

    std::sort(times.begin(), times.end());
    float sum = 0;
    for (auto t : times) sum += t;

    return {sum / iters, times[0], times[iters / 2]};
}

// ============================================================================
// MinGRU benchmark (preserving original interface)
// ============================================================================

BenchResult bench_mingru(int batch, int seq_len, int hidden, int warmup, int iters) {
    return bench_2input(fused_mingru_scan_launch, batch, seq_len, hidden,
                        warmup, iters, 0.0f, 1.0f, -1.0f, 1.0f);
}

// ============================================================================
// MinLSTM benchmark (3 inputs — custom)
// ============================================================================

BenchResult bench_minlstm(int batch, int seq_len, int hidden, int warmup, int iters) {
    int bth = batch * seq_len * hidden;
    int bh = batch * hidden;

    float *d_fgates, *d_igates, *d_candidates, *d_h0, *d_output;
    CUDA_CHECK(cudaMalloc(&d_fgates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_igates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h0, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bth * sizeof(float)));

    float* h_buf = (float*)malloc(bth * sizeof(float));
    srand(123);
    for (int i = 0; i < bth; i++) h_buf[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_fgates, h_buf, bth * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < bth; i++) h_buf[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_igates, h_buf, bth * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < bth; i++) h_buf[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    CUDA_CHECK(cudaMemcpy(d_candidates, h_buf, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_h0, 0, bh * sizeof(float)));
    free(h_buf);

    for (int i = 0; i < warmup; i++) {
        fused_minlstm_scan_launch(0, d_fgates, d_igates, d_candidates, d_h0, d_output, batch, seq_len, hidden);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(iters);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        fused_minlstm_scan_launch(0, d_fgates, d_igates, d_candidates, d_h0, d_output, batch, seq_len, hidden);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_fgates); cudaFree(d_igates); cudaFree(d_candidates); cudaFree(d_h0); cudaFree(d_output);

    std::sort(times.begin(), times.end());
    float sum = 0;
    for (auto t : times) sum += t;

    return {sum / iters, times[0], times[iters / 2]};
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %.0f MHz, %zu MB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.clockRate / 1000.0f, prop.totalGlobalMem / (1024*1024));

    const int WARMUP = 100;
    const int ITERS  = 500;

    printf("=== Fused Scan Kernel Benchmarks ===\n");
    printf("(%d warmup, %d timed iterations, GPU-side timing)\n\n", WARMUP, ITERS);

    printf("Target: < 16ms total inference (60 FPS)\n");
    printf("Note: kernel time is ONLY the scan — matmuls are separate XLA ops\n\n");

    struct Config { int batch; int seq_len; int hidden; const char* label; };
    Config configs[] = {
        {1,   1, 256, "inference seq=1   (single step)"},
        {1,  32, 256, "inference seq=32  (target config)"},
        {1,  64, 256, "inference seq=64  (extended)"},
        {1, 128, 256, "inference seq=128 (long context)"},
        {1,  32, 512, "inference seq=32  (large hidden)"},
        {32, 32, 256, "training  seq=32  (batch=32)"},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    printf("%-42s %10s %10s %10s\n", "Config", "Avg (ms)", "Min (ms)", "Med (ms)");
    printf("%-42s %10s %10s %10s\n",
           "------", "--------", "--------", "--------");

    printf("\n--- MinGRU ---\n");
    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto r = bench_mingru(cfg.batch, cfg.seq_len, cfg.hidden, WARMUP, ITERS);
        printf("%-42s %10.3f %10.3f %10.3f\n", cfg.label, r.avg_ms, r.min_ms, r.median_ms);
    }

    printf("\n--- MinLSTM ---\n");
    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto r = bench_minlstm(cfg.batch, cfg.seq_len, cfg.hidden, WARMUP, ITERS);
        printf("%-42s %10.3f %10.3f %10.3f\n", cfg.label, r.avg_ms, r.min_ms, r.median_ms);
    }

    printf("\n--- ELU-GRU (NativeRecurrence) ---\n");
    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto r = bench_2input(fused_elu_gru_scan_launch, cfg.batch, cfg.seq_len, cfg.hidden,
                              WARMUP, ITERS, -3.0f, 3.0f, -2.0f, 2.0f);
        printf("%-42s %10.3f %10.3f %10.3f\n", cfg.label, r.avg_ms, r.min_ms, r.median_ms);
    }

    printf("\n--- Real-GRU (NativeRecurrence) ---\n");
    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto r = bench_2input(fused_real_gru_scan_launch, cfg.batch, cfg.seq_len, cfg.hidden,
                              WARMUP, ITERS, -3.0f, 3.0f, -1.0f, 1.0f);
        printf("%-42s %10.3f %10.3f %10.3f\n", cfg.label, r.avg_ms, r.min_ms, r.median_ms);
    }

    printf("\n--- Diag-Linear (NativeRecurrence) ---\n");
    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto r = bench_2input(fused_diag_linear_scan_launch, cfg.batch, cfg.seq_len, cfg.hidden,
                              WARMUP, ITERS, -3.0f, 3.0f, -1.0f, 1.0f);
        printf("%-42s %10.3f %10.3f %10.3f\n", cfg.label, r.avg_ms, r.min_ms, r.median_ms);
    }

    printf("\n--- Liquid (exact solver) ---\n");
    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto r = bench_2input(fused_liquid_scan_launch, cfg.batch, cfg.seq_len, cfg.hidden,
                              WARMUP, ITERS, 0.1f, 10.0f, -1.0f, 1.0f);
        printf("%-42s %10.3f %10.3f %10.3f\n", cfg.label, r.avg_ms, r.min_ms, r.median_ms);
    }

    printf("\n--- Comparison with plan doc targets ---\n");
    auto mingru_target = bench_mingru(1, 32, 256, WARMUP, ITERS);
    auto minlstm_target = bench_minlstm(1, 32, 256, WARMUP, ITERS);
    printf("MinGRU  scan kernel: %.3f ms  (plan: 84.75ms total -> target ~12ms scan)\n", mingru_target.median_ms);
    printf("MinLSTM scan kernel: %.3f ms  (plan: 74.79ms total -> target ~14ms scan)\n", minlstm_target.median_ms);
    printf("\nRemember: total inference = scan_kernel * num_layers + matmul_time + overhead\n");

    return 0;
}
