// Standalone test for fused scan kernels.
//
// Compiles and runs independently — no EXLA or Elixir dependencies.
// Compares kernel output against a CPU reference implementation.
//
// Build: nvcc -arch=sm_75 -o test_kernels test_kernels.cu -O3
// Run:   ./test_kernels

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Import the launch wrappers
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

// ============================================================================
// CPU reference implementations
// ============================================================================

void mingru_scan_cpu(
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float state = h0[b * hidden + h];
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * hidden + t * hidden + h;
                float z = gates[idx];
                float c = candidates[idx];
                state = (1.0f - z) * state + z * c;
                output[idx] = state;
            }
        }
    }
}

void minlstm_scan_cpu(
    const float* forget_gates, const float* input_gates,
    const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    const float eps = 1.0e-6f;
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float state = h0[b * hidden + h];
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * hidden + t * hidden + h;
                float f = forget_gates[idx];
                float i = input_gates[idx];
                float c = candidates[idx];
                float sum = f + i + eps;
                state = (f / sum) * state + (i / sum) * c;
                output[idx] = state;
            }
        }
    }
}

// CPU sigmoid/elu for NativeRecurrence reference implementations
static inline float cpu_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float cpu_elu(float x) { return (x >= 0.0f) ? x : (expf(x) - 1.0f); }

void elu_gru_scan_cpu(
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float state = h0[b * hidden + h];
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * hidden + t * hidden + h;
                float z = cpu_sigmoid(gates[idx]);
                float c = 1.0f + cpu_elu(candidates[idx]);
                state = (1.0f - z) * state + z * c;
                output[idx] = state;
            }
        }
    }
}

void real_gru_scan_cpu(
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float state = h0[b * hidden + h];
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * hidden + t * hidden + h;
                float z = cpu_sigmoid(gates[idx]);
                float c = candidates[idx];
                state = (1.0f - z) * state + z * c;
                output[idx] = state;
            }
        }
    }
}

void diag_linear_scan_cpu(
    const float* a_vals, const float* b_vals, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float state = h0[b * hidden + h];
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * hidden + t * hidden + h;
                float a = cpu_sigmoid(a_vals[idx]);
                float bv = b_vals[idx];
                state = a * state + bv;
                output[idx] = state;
            }
        }
    }
}

void liquid_scan_cpu(
    const float* tau, const float* activation, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden; h++) {
            float state = h0[b * hidden + h];
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * hidden + t * hidden + h;
                float tau_t = tau[idx];
                float act_t = activation[idx];
                float decay = expf(-1.0f / tau_t);
                state = act_t + (state - act_t) * decay;
                output[idx] = state;
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

float rand_float() { return (float)rand() / RAND_MAX; }

float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Tests
// ============================================================================

bool test_mingru(int batch, int seq_len, int hidden) {
    int bth = batch * seq_len * hidden;
    int bh = batch * hidden;

    // Allocate host memory
    float* h_gates      = (float*)malloc(bth * sizeof(float));
    float* h_candidates = (float*)malloc(bth * sizeof(float));
    float* h_h0         = (float*)malloc(bh * sizeof(float));
    float* h_output_gpu = (float*)malloc(bth * sizeof(float));
    float* h_output_cpu = (float*)malloc(bth * sizeof(float));

    // Initialize with random sigmoid-range values for gates
    srand(42);
    for (int i = 0; i < bth; i++) {
        h_gates[i] = rand_float();      // Already in [0,1] (post-sigmoid)
        h_candidates[i] = rand_float() * 2.0f - 1.0f;  // [-1, 1]
    }
    for (int i = 0; i < bh; i++) h_h0[i] = 0.0f;

    // CPU reference
    mingru_scan_cpu(h_gates, h_candidates, h_h0, h_output_cpu, batch, seq_len, hidden);

    // GPU
    float *d_gates, *d_candidates, *d_h0, *d_output;
    CUDA_CHECK(cudaMalloc(&d_gates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h0, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bth * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gates, h_gates, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates, h_candidates, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0, h_h0, bh * sizeof(float), cudaMemcpyHostToDevice));

    int err = fused_mingru_scan_launch(0, d_gates, d_candidates, d_h0, d_output, batch, seq_len, hidden);
    if (err != 0) {
        printf("  FAIL: kernel launch error %d\n", err);
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bth * sizeof(float), cudaMemcpyDeviceToHost));

    float diff = max_abs_diff(h_output_cpu, h_output_gpu, bth);
    bool pass = diff < 1.0e-4f;

    printf("  MinGRU [B=%d, T=%d, H=%d]: max_diff=%.2e %s\n",
           batch, seq_len, hidden, diff, pass ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_gates); cudaFree(d_candidates); cudaFree(d_h0); cudaFree(d_output);
    free(h_gates); free(h_candidates); free(h_h0); free(h_output_gpu); free(h_output_cpu);

    return pass;
}

bool test_minlstm(int batch, int seq_len, int hidden) {
    int bth = batch * seq_len * hidden;
    int bh = batch * hidden;

    float* h_fgates     = (float*)malloc(bth * sizeof(float));
    float* h_igates     = (float*)malloc(bth * sizeof(float));
    float* h_candidates = (float*)malloc(bth * sizeof(float));
    float* h_h0         = (float*)malloc(bh * sizeof(float));
    float* h_output_gpu = (float*)malloc(bth * sizeof(float));
    float* h_output_cpu = (float*)malloc(bth * sizeof(float));

    srand(123);
    for (int i = 0; i < bth; i++) {
        h_fgates[i] = rand_float();
        h_igates[i] = rand_float();
        h_candidates[i] = rand_float() * 2.0f - 1.0f;
    }
    for (int i = 0; i < bh; i++) h_h0[i] = 0.0f;

    minlstm_scan_cpu(h_fgates, h_igates, h_candidates, h_h0, h_output_cpu, batch, seq_len, hidden);

    float *d_fgates, *d_igates, *d_candidates, *d_h0, *d_output;
    CUDA_CHECK(cudaMalloc(&d_fgates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_igates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidates, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h0, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bth * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_fgates, h_fgates, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_igates, h_igates, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates, h_candidates, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0, h_h0, bh * sizeof(float), cudaMemcpyHostToDevice));

    int launch_err = fused_minlstm_scan_launch(0, d_fgates, d_igates, d_candidates, d_h0, d_output, batch, seq_len, hidden);
    if (launch_err != 0) {
        printf("  FAIL: kernel launch error %d\n", launch_err);
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bth * sizeof(float), cudaMemcpyDeviceToHost));

    float diff = max_abs_diff(h_output_cpu, h_output_gpu, bth);
    bool pass = diff < 1.0e-4f;

    printf("  MinLSTM [B=%d, T=%d, H=%d]: max_diff=%.2e %s\n",
           batch, seq_len, hidden, diff, pass ? "PASS" : "FAIL");

    cudaFree(d_fgates); cudaFree(d_igates); cudaFree(d_candidates); cudaFree(d_h0); cudaFree(d_output);
    free(h_fgates); free(h_igates); free(h_candidates); free(h_h0); free(h_output_gpu); free(h_output_cpu);

    return pass;
}

// Generic test for 2-input kernels (elu_gru, real_gru, diag_linear, liquid)
// Takes raw pre-activation inputs (kernel applies nonlinearities internally)
typedef int (*scan_launch_fn)(cudaStream_t, const float*, const float*, const float*,
                               float*, int, int, int);
typedef void (*scan_cpu_fn)(const float*, const float*, const float*,
                            float*, int, int, int);

bool test_2input_kernel(
    const char* name,
    scan_launch_fn gpu_launch,
    scan_cpu_fn cpu_ref,
    int batch, int seq_len, int hidden,
    unsigned int seed,
    float input1_lo, float input1_hi,
    float input2_lo, float input2_hi
) {
    int bth = batch * seq_len * hidden;
    int bh = batch * hidden;

    float* h_in1        = (float*)malloc(bth * sizeof(float));
    float* h_in2        = (float*)malloc(bth * sizeof(float));
    float* h_h0         = (float*)malloc(bh * sizeof(float));
    float* h_output_gpu = (float*)malloc(bth * sizeof(float));
    float* h_output_cpu = (float*)malloc(bth * sizeof(float));

    srand(seed);
    for (int i = 0; i < bth; i++) {
        h_in1[i] = rand_float() * (input1_hi - input1_lo) + input1_lo;
        h_in2[i] = rand_float() * (input2_hi - input2_lo) + input2_lo;
    }
    for (int i = 0; i < bh; i++) h_h0[i] = 0.0f;

    cpu_ref(h_in1, h_in2, h_h0, h_output_cpu, batch, seq_len, hidden);

    float *d_in1, *d_in2, *d_h0, *d_output;
    CUDA_CHECK(cudaMalloc(&d_in1, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in2, bth * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h0, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bth * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in1, h_in1, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in2, h_in2, bth * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0, h_h0, bh * sizeof(float), cudaMemcpyHostToDevice));

    int err = gpu_launch(0, d_in1, d_in2, d_h0, d_output, batch, seq_len, hidden);
    if (err != 0) {
        printf("  FAIL: %s kernel launch error %d\n", name, err);
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bth * sizeof(float), cudaMemcpyDeviceToHost));

    float diff = max_abs_diff(h_output_cpu, h_output_gpu, bth);
    bool pass = diff < 1.0e-4f;

    printf("  %s [B=%d, T=%d, H=%d]: max_diff=%.2e %s\n",
           name, batch, seq_len, hidden, diff, pass ? "PASS" : "FAIL");

    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_h0); cudaFree(d_output);
    free(h_in1); free(h_in2); free(h_h0); free(h_output_gpu); free(h_output_cpu);

    return pass;
}

int main() {
    printf("=== Fused Scan Kernel Tests ===\n\n");

    int pass = 0, fail = 0;

    // Test various configurations from the plan doc edge cases
    printf("MinGRU tests:\n");
    (test_mingru(1, 1, 64)    ? pass++ : fail++);  // seq=1 (single step)
    (test_mingru(1, 32, 256)  ? pass++ : fail++);  // target config
    (test_mingru(2, 32, 256)  ? pass++ : fail++);  // batch=2
    (test_mingru(1, 64, 128)  ? pass++ : fail++);  // longer seq
    (test_mingru(1, 256, 64)  ? pass++ : fail++);  // stress test seq
    (test_mingru(32, 32, 256) ? pass++ : fail++);  // batch=32 (training)
    (test_mingru(1, 32, 512)  ? pass++ : fail++);  // large hidden

    printf("\nMinLSTM tests:\n");
    (test_minlstm(1, 1, 64)    ? pass++ : fail++);
    (test_minlstm(1, 32, 256)  ? pass++ : fail++);
    (test_minlstm(2, 32, 256)  ? pass++ : fail++);
    (test_minlstm(1, 64, 128)  ? pass++ : fail++);
    (test_minlstm(1, 256, 64)  ? pass++ : fail++);
    (test_minlstm(32, 32, 256) ? pass++ : fail++);
    (test_minlstm(1, 32, 512)  ? pass++ : fail++);

    // NativeRecurrence kernels — inputs are raw pre-activations
    printf("\nELU-GRU tests:\n");
    // Input ranges: gates in [-3,3] (pre-sigmoid), candidates in [-2,2] (pre-elu)
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        1, 1, 64, 200, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        1, 32, 256, 201, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        2, 32, 256, 202, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        1, 64, 128, 203, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        1, 256, 64, 204, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        32, 32, 256, 205, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);
    (test_2input_kernel("ELU-GRU", fused_elu_gru_scan_launch, elu_gru_scan_cpu,
                        1, 32, 512, 206, -3.0f, 3.0f, -2.0f, 2.0f) ? pass++ : fail++);

    printf("\nReal-GRU tests:\n");
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        1, 1, 64, 300, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        1, 32, 256, 301, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        2, 32, 256, 302, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        1, 64, 128, 303, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        1, 256, 64, 304, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        32, 32, 256, 305, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Real-GRU", fused_real_gru_scan_launch, real_gru_scan_cpu,
                        1, 32, 512, 306, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);

    printf("\nDiag-Linear tests:\n");
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        1, 1, 64, 400, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        1, 32, 256, 401, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        2, 32, 256, 402, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        1, 64, 128, 403, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        1, 256, 64, 404, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        32, 32, 256, 405, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("DiagLinear", fused_diag_linear_scan_launch, diag_linear_scan_cpu,
                        1, 32, 512, 406, -3.0f, 3.0f, -1.0f, 1.0f) ? pass++ : fail++);

    printf("\nLiquid tests:\n");
    // tau in [0.1, 10.0] (post-softplus), activation in [-1, 1]
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        1, 1, 64, 500, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        1, 32, 256, 501, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        2, 32, 256, 502, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        1, 64, 128, 503, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        1, 256, 64, 504, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        32, 32, 256, 505, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);
    (test_2input_kernel("Liquid", fused_liquid_scan_launch, liquid_scan_cpu,
                        1, 32, 512, 506, 0.1f, 10.0f, -1.0f, 1.0f) ? pass++ : fail++);

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
