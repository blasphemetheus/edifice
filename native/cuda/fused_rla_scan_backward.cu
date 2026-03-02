// Fused RLA/RDN (Residual Linear Attention) Scan Backward Kernel
//
// Two-pass backward for the dual-state RLA recurrence. Both variants share
// the same kernel, dispatched by the `variant` parameter.
//
// Forward recurrence (Variant 0 — RLA, moving average):
//   retrieval_s = S_{t-1} @ k
//   r_error = clip(v - retrieval_s, -c, c)
//   S_t = alpha * S_{t-1} + beta * outer(v, k)
//   R_t = alpha * R_{t-1} + gamma * outer(r_error, k)
//   o_t = (S_t + R_t) @ q
//
// Forward recurrence (Variant 1 — RDN, delta rule):
//   retrieval_s = S_{t-1} @ k
//   retrieval_r = R_{t-1} @ k
//   r_error = clip(v - retrieval_s, -c, c)
//   S_t = alpha * S_{t-1} + beta * outer(v - retrieval_s, k)
//   R_t = alpha * R_{t-1} + gamma * outer(r_error - retrieval_r, k)
//   o_t = (S_t + R_t) @ q
//
// Backward approach:
//   Pass 1 (forward): Recompute S and R states. Store per-timestep scalars:
//     - retrieval_s_i[t], r_error_i[t], clip_mask_i[t]
//     - retrieval_r_i[t] (variant 1 only, alias re-used for variant 0)
//   Pass 2 (reverse): Walk t=T-1..0, accumulate dS_row and dR_row per thread,
//     undo S/R updates to recover previous states, compute all gradients.
//
// Inputs:
//   q:           [B, T, H, d]
//   k:           [B, T, H, d]
//   v:           [B, T, H, d]
//   alpha:       [B, T, H]    — decay gate
//   beta:        [B, T, H]    — base update rate
//   gamma:       [B, T, H]    — residual update rate
//   forward_out: [B, T, H, d] — forward pass outputs (not strictly needed but kept for API consistency)
//   grad_output: [B, T, H, d]
//
// Outputs:
//   grad_q:     [B, T, H, d]
//   grad_k:     [B, T, H, d]
//   grad_v:     [B, T, H, d]
//   grad_alpha: [B, T, H]    — uses atomicAdd for cross-thread reduction
//   grad_beta:  [B, T, H]    — uses atomicAdd for cross-thread reduction
//   grad_gamma: [B, T, H]    — uses atomicAdd for cross-thread reduction
//
// Thread layout: one block per (batch, head), head_dim threads.
// Each thread owns row i of S[d][d] and R[d][d].

#include <cuda_runtime.h>
#include "precision.cuh"

#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_rla_scan_backward_kernel(
    const io_type* __restrict__ q,            // [B, T, H, d]
    const io_type* __restrict__ k,            // [B, T, H, d]
    const io_type* __restrict__ v,            // [B, T, H, d]
    const io_type* __restrict__ alpha,        // [B, T, H]
    const io_type* __restrict__ beta,         // [B, T, H]
    const io_type* __restrict__ gamma,        // [B, T, H]
    const io_type* __restrict__ forward_out,  // [B, T, H, d]
    const io_type* __restrict__ grad_output,  // [B, T, H, d]
    io_type* __restrict__ grad_q,             // [B, T, H, d]
    io_type* __restrict__ grad_k,             // [B, T, H, d]
    io_type* __restrict__ grad_v,             // [B, T, H, d]
    io_type* __restrict__ grad_alpha,         // [B, T, H]
    io_type* __restrict__ grad_beta,          // [B, T, H]
    io_type* __restrict__ grad_gamma,         // [B, T, H]
    int seq_len,
    int num_heads,
    int head_dim,
    int variant,              // 0 = RLA, 1 = RDN
    float clip_threshold
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;

    if (i >= head_dim) return;

    // Shared memory layout:
    //   S[d][d]       — base state matrix
    //   R[d][d]       — residual state matrix
    //   k_shared[d]   — broadcast k vector
    //   q_shared[d]   — broadcast q vector (reused for various reductions)
    //   temp_shared[d] — scratch for dk cross-thread reduction
    //   reduce_buf[3]  — for alpha/beta/gamma scalar gradient reduction
    extern __shared__ float smem[];
    float* S = smem;                                          // [d][d]
    float* R = S + head_dim * head_dim;                       // [d][d]
    float* k_shared = R + head_dim * head_dim;                // [d]
    float* q_shared = k_shared + head_dim;                    // [d]
    float* temp_shared = q_shared + head_dim;                 // [d]
    float* reduce_buf = temp_shared + head_dim;               // [3] for alpha, beta, gamma

    // 4D strides: [B, T, H, d]
    int THd = seq_len * num_heads * head_dim;
    int Hd  = num_heads * head_dim;
    int d   = head_dim;
    int base_bh = b * THd + h * d;

    // 3D strides: [B, T, H]  (gate tensors)
    int gate_BH = seq_len * num_heads;
    int gate_TH = num_heads;
    int gate_base = b * gate_BH + h;

    // ========================================
    // Pass 1: Forward — recompute S, R states and store intermediates
    // ========================================
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
        R[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    // Per-thread per-timestep intermediates
    float local_retrieval_s[MAX_SEQ_LEN];  // retrieval from S for thread row i
    float local_retrieval_r[MAX_SEQ_LEN];  // retrieval from R for thread row i (variant 1)
    float local_r_error[MAX_SEQ_LEN];      // clipped error
    float local_clip_mask[MAX_SEQ_LEN];    // 1.0 if within clip range, 0.0 if clipped

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * Hd;
        int gate_idx = gate_base + t * gate_TH;

        // Load k and v into shared memory
        k_shared[i] = IO_LOAD(k, offset + i);
        __syncthreads();

        float alpha_val = IO_LOAD(alpha, gate_idx);
        float beta_val  = IO_LOAD(beta, gate_idx);
        float gamma_val = IO_LOAD(gamma, gate_idx);
        float v_i = IO_LOAD(v, offset + i);

        // Compute retrieval_s = S_{t-1} @ k (before S update)
        // Note: S currently holds S_{t-1} (state from previous iteration)
        float retrieval_s = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval_s += S[i * head_dim + j] * k_shared[j];
        }
        local_retrieval_s[t] = retrieval_s;

        // Compute retrieval_r = R_{t-1} @ k (variant 1 only)
        float retrieval_r = 0.0f;
        if (variant == 1) {
            for (int j = 0; j < head_dim; j++) {
                retrieval_r += R[i * head_dim + j] * k_shared[j];
            }
        }
        local_retrieval_r[t] = retrieval_r;

        // Residual error with clipping
        float raw_error = v_i - retrieval_s;
        float r_error = fminf(fmaxf(raw_error, -clip_threshold), clip_threshold);
        float clip_mask = (fabsf(raw_error) <= clip_threshold) ? 1.0f : 0.0f;
        local_r_error[t] = r_error;
        local_clip_mask[t] = clip_mask;

        // Update S and R
        if (variant == 0) {
            // RLA: S = alpha*S + beta*outer(v, k)
            //      R = alpha*R + gamma*outer(r_error, k)
            float beta_v_i = beta_val * v_i;
            float gamma_r_i = gamma_val * r_error;
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] = alpha_val * S[i * head_dim + j]
                                    + beta_v_i * k_shared[j];
                R[i * head_dim + j] = alpha_val * R[i * head_dim + j]
                                    + gamma_r_i * k_shared[j];
            }
        } else {
            // RDN: S = alpha*S + beta*outer(v - retrieval_s, k)
            //      R = alpha*R + gamma*outer(r_error - retrieval_r, k)
            float delta_s_i = v_i - retrieval_s;  // same as raw_error
            float delta_r_i = r_error - retrieval_r;
            float beta_ds_i = beta_val * delta_s_i;
            float gamma_dr_i = gamma_val * delta_r_i;
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] = alpha_val * S[i * head_dim + j]
                                    + beta_ds_i * k_shared[j];
                R[i * head_dim + j] = alpha_val * R[i * head_dim + j]
                                    + gamma_dr_i * k_shared[j];
            }
        }
        __syncthreads();
    }

    // ========================================
    // Pass 2: Backward — reverse iterate, accumulate dS and dR
    // ========================================
    // dS_row[j] and dR_row[j]: per-thread gradient accumulators for row i of
    // the state matrices. These propagate backward through the recurrence.
    float dS_row[128];
    float dR_row[128];
    for (int j = 0; j < head_dim; j++) {
        dS_row[j] = 0.0f;
        dR_row[j] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int offset = base_bh + t * Hd;
        int gate_idx = gate_base + t * gate_TH;

        float q_i = IO_LOAD(q, offset + i);
        float k_i = IO_LOAD(k, offset + i);
        float v_i = IO_LOAD(v, offset + i);
        float do_i = IO_LOAD(grad_output, offset + i);
        float alpha_val = IO_LOAD(alpha, gate_idx);
        float beta_val  = IO_LOAD(beta, gate_idx);
        float gamma_val = IO_LOAD(gamma, gate_idx);

        float retrieval_s_i = local_retrieval_s[t];
        float retrieval_r_i = local_retrieval_r[t];
        float r_error_i = local_r_error[t];
        float clip_mask_i = local_clip_mask[t];

        // ---- grad_q: o_t = (S_t + R_t) @ q  =>  dq_i = (S_t + R_t)^T[i] @ do ----
        // dq_i = sum_j( (S[j][i] + R[j][i]) * do[j] )
        q_shared[i] = do_i;
        __syncthreads();

        float dq_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            dq_i += (S[j * head_dim + i] + R[j * head_dim + i]) * q_shared[j];
        }
        IO_STORE(grad_q, offset + i, dq_i);

        // ---- dS, dR from output: o_t = (S_t + R_t) @ q  =>  d(S_t + R_t) += outer(do, q) ----
        k_shared[i] = q_i;  // reuse k_shared to broadcast q
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += do_i * k_shared[j];
            dR_row[j] += do_i * k_shared[j];
        }

        // Load k into shared memory for update gradient computations
        k_shared[i] = k_i;
        __syncthreads();

        // ========================================
        // Variant-specific backward through update equations
        // ========================================

        // We need to compute:
        //   dk contributions from the outer products
        //   dv contributions
        //   d(scalar gate) contributions (alpha, beta, gamma)
        //   dS_row and dR_row propagation through alpha decay

        if (variant == 0) {
            // ----- Variant 0 (RLA) backward -----
            // S_t = alpha * S_{t-1} + beta * outer(v, k)
            // R_t = alpha * R_{t-1} + gamma * outer(r_error, k)

            // --- From S update: dS @ d(beta * outer(v, k)) ---
            // d(beta * v_i * k_j) / dk_j summed over i: sum_i(dS[i][j] * beta * v_i)
            // d(beta * v_i * k_j) / dv_i: sum_j(dS[i][j] * beta * k_j)  = beta * (dS_row . k)
            // d(beta * v_i * k_j) / dbeta: sum_{i,j}(dS[i][j] * v_i * k_j)

            float dS_dot_k = 0.0f;  // sum_j(dS_row[j] * k_shared[j])
            for (int j = 0; j < head_dim; j++) {
                dS_dot_k += dS_row[j] * k_shared[j];
            }
            float dv_from_S = beta_val * dS_dot_k;

            // --- From R update: dR @ d(gamma * outer(r_error, k)) ---
            // d(gamma * r_error_i * k_j) / dr_error_i = gamma * (dR_row . k)
            float dR_dot_k = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                dR_dot_k += dR_row[j] * k_shared[j];
            }
            float d_r_error_i = gamma_val * dR_dot_k;

            // r_error = clip(v - retrieval_s, -c, c)
            // d(v - retrieval_s) from clipping: d_r_error * clip_mask
            float d_raw_error = d_r_error_i * clip_mask_i;

            // dv from r_error path: d(v - retrieval_s)/dv * d_raw_error = d_raw_error
            float dv_from_R = d_raw_error;

            // d_retrieval_s from r_error path: d(v - retrieval_s)/d(retrieval_s) = -d_raw_error
            float d_retrieval_s = -d_raw_error;

            // Total dv
            float dv_i = dv_from_S + dv_from_R;
            IO_STORE(grad_v, offset + i, dv_i);

            // --- dk from outer(v, k) in S update ---
            // dk_j += sum_i(dS[i][j] * beta * v_i)
            // Reduction across threads: each thread contributes dS_row[j] * beta_val * v_i
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    temp_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&temp_shared[j], dS_row[j] * beta_val * v_i);
            }
            __syncthreads();

            // Save S-contribution to dk, prepare for R-contribution
            float dk_from_S_j = temp_shared[i];

            // --- dk from outer(r_error, k) in R update ---
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    q_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&q_shared[j], dR_row[j] * gamma_val * r_error_i);
            }
            __syncthreads();

            float dk_from_R_outer_j = q_shared[i];

            // --- dk from retrieval_s = S_{t-1} @ k (via r_error path) ---
            // d_retrieval_s contributes to dS_{t-1} and dk
            // dk_j += sum_i(S_{t-1}[i][j] * d_retrieval_s_i)
            // But we need S_{t-1}. We'll undo the update first, then compute.

            // Undo S update: S_{t-1} = (S_t - beta * outer(v, k)) / alpha
            // First undo the outer product addition
            float beta_v_i = beta_val * v_i;
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] -= beta_v_i * k_shared[j];
            }
            __syncthreads();

            // Now S holds alpha * S_{t-1}. Undo alpha scaling.
            // But first, compute dk from retrieval_s through S_gated = alpha * S_{t-1}
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    temp_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            // S[i][j] currently = alpha * S_{t-1}[i][j]
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&temp_shared[j], S[i * head_dim + j] * d_retrieval_s);
            }
            __syncthreads();

            // Note: S currently holds alpha * S_{t-1}, so
            // sum_i(S[i][j] * d_retrieval_s_i) = alpha * sum_i(S_{t-1}[i][j] * d_retrieval_s_i)
            // But retrieval used S_{t-1} (before alpha scaling in this timestep).
            // Actually, looking at the forward: retrieval_s uses the state BEFORE the update
            // at timestep t, which is S_{t-1}. And S[i][j] currently = alpha * S_{t-1}[i][j].
            // So we need to divide by alpha. But it's simpler to note that after undoing
            // the outer product, S = alpha*S_{t-1}, so S_{t-1}[i][j] = S[i][j]/alpha.
            // We can scale temp_shared by 1/alpha.
            float dk_from_retrieval_j = (alpha_val != 0.0f) ? temp_shared[i] / alpha_val : 0.0f;

            float dk_i = dk_from_S_j + dk_from_R_outer_j + dk_from_retrieval_j;
            IO_STORE(grad_k, offset + i, dk_i);

            // --- dS propagation through retrieval_s path ---
            // retrieval_s_i = sum_j(S_{t-1}[i][j] * k[j])
            // dS_{t-1}[i][j] += d_retrieval_s_i * k[j]
            for (int j = 0; j < head_dim; j++) {
                dS_row[j] += d_retrieval_s * k_shared[j];
            }

            // Undo R update similarly: R_{t-1} = (R_t - gamma * outer(r_error, k)) / alpha
            float gamma_r_i = gamma_val * r_error_i;
            for (int j = 0; j < head_dim; j++) {
                R[i * head_dim + j] -= gamma_r_i * k_shared[j];
            }
            __syncthreads();

            // --- grad_alpha: S_t = alpha * S_{t-1} + ... ---
            // d_alpha from S: sum_{i,j}(dS[i][j] * S_{t-1}[i][j])
            // d_alpha from R: sum_{i,j}(dR[i][j] * R_{t-1}[i][j])
            // S currently = alpha * S_{t-1}, R currently = alpha * R_{t-1}
            float d_alpha_partial = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                float s_prev_ij = (alpha_val != 0.0f) ? S[i * head_dim + j] / alpha_val : 0.0f;
                float r_prev_ij = (alpha_val != 0.0f) ? R[i * head_dim + j] / alpha_val : 0.0f;
                d_alpha_partial += dS_row[j] * s_prev_ij + dR_row[j] * r_prev_ij;
            }

            // --- grad_beta: sum_{i,j}(dS[i][j] * v_i * k_j) ---
            // Each thread contributes: v_i * sum_j(dS_row[j] * k_shared[j]) = v_i * dS_dot_k
            float d_beta_partial = v_i * dS_dot_k;

            // --- grad_gamma: sum_{i,j}(dR[i][j] * r_error_i * k_j) ---
            float d_gamma_partial = r_error_i * dR_dot_k;

            // Reduce alpha, beta, gamma gradients across threads
            if (i == 0) {
                reduce_buf[0] = 0.0f;
                reduce_buf[1] = 0.0f;
                reduce_buf[2] = 0.0f;
            }
            __syncthreads();
            atomicAdd(&reduce_buf[0], d_alpha_partial);
            atomicAdd(&reduce_buf[1], d_beta_partial);
            atomicAdd(&reduce_buf[2], d_gamma_partial);
            __syncthreads();

            if (i == 0) {
                IO_STORE(grad_alpha, gate_idx, reduce_buf[0]);
                IO_STORE(grad_beta,  gate_idx, reduce_buf[1]);
                IO_STORE(grad_gamma, gate_idx, reduce_buf[2]);
            }

            // --- dS_row, dR_row propagation through alpha decay ---
            for (int j = 0; j < head_dim; j++) {
                dS_row[j] *= alpha_val;
                dR_row[j] *= alpha_val;
            }

            // Undo alpha decay to get S_{t-1} and R_{t-1}
            if (alpha_val != 0.0f) {
                float inv_alpha = 1.0f / alpha_val;
                for (int j = 0; j < head_dim; j++) {
                    S[i * head_dim + j] *= inv_alpha;
                    R[i * head_dim + j] *= inv_alpha;
                }
            }
            __syncthreads();

        } else {
            // ----- Variant 1 (RDN — delta rule) backward -----
            // S_t = alpha * S_{t-1} + beta * outer(v - retrieval_s, k)
            // R_t = alpha * R_{t-1} + gamma * outer(r_error - retrieval_r, k)
            //
            // Let delta_s = v - retrieval_s,  delta_r = r_error - retrieval_r
            float delta_s_i = v_i - retrieval_s_i;
            float delta_r_i = r_error_i - retrieval_r_i;

            // --- From S update: dS @ d(beta * outer(delta_s, k)) ---
            float dS_dot_k = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                dS_dot_k += dS_row[j] * k_shared[j];
            }
            // d(delta_s_i) from S path = beta * dS_dot_k
            float d_delta_s_from_S = beta_val * dS_dot_k;

            // --- From R update: dR @ d(gamma * outer(delta_r, k)) ---
            float dR_dot_k = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                dR_dot_k += dR_row[j] * k_shared[j];
            }
            // d(delta_r_i) from R path = gamma * dR_dot_k
            float d_delta_r_from_R = gamma_val * dR_dot_k;

            // --- Backprop through delta_s = v - retrieval_s ---
            // dv from delta_s: d_delta_s
            // d_retrieval_s from delta_s: -d_delta_s
            float d_retrieval_s_from_S = -d_delta_s_from_S;

            // --- Backprop through delta_r = r_error - retrieval_r ---
            // d_r_error from delta_r: d_delta_r
            // d_retrieval_r from delta_r: -d_delta_r
            float d_r_error_from_R = d_delta_r_from_R;
            float d_retrieval_r = -d_delta_r_from_R;

            // --- Backprop through r_error = clip(v - retrieval_s, -c, c) ---
            // d(v - retrieval_s) = d_r_error * clip_mask
            float d_raw_error = d_r_error_from_R * clip_mask_i;
            float d_retrieval_s_from_R = -d_raw_error;
            float dv_from_R_clip = d_raw_error;

            // Total d_retrieval_s
            float d_retrieval_s = d_retrieval_s_from_S + d_retrieval_s_from_R;

            // Total dv
            float dv_i = d_delta_s_from_S + dv_from_R_clip;
            IO_STORE(grad_v, offset + i, dv_i);

            // --- dk from outer(delta_s, k) in S update ---
            // dk_j += sum_i(dS[i][j] * beta * delta_s_i)
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    temp_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&temp_shared[j], dS_row[j] * beta_val * delta_s_i);
            }
            __syncthreads();

            float dk_from_S_j = temp_shared[i];

            // --- dk from outer(delta_r, k) in R update ---
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    q_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&q_shared[j], dR_row[j] * gamma_val * delta_r_i);
            }
            __syncthreads();

            float dk_from_R_outer_j = q_shared[i];

            // Undo S update: remove the outer product to get alpha * S_{t-1}
            float beta_ds_i = beta_val * delta_s_i;
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] -= beta_ds_i * k_shared[j];
            }
            __syncthreads();

            // Undo R update: remove the outer product to get alpha * R_{t-1}
            float gamma_dr_i = gamma_val * delta_r_i;
            for (int j = 0; j < head_dim; j++) {
                R[i * head_dim + j] -= gamma_dr_i * k_shared[j];
            }
            __syncthreads();

            // --- dk from retrieval_s = S_{t-1} @ k ---
            // S currently = alpha * S_{t-1}
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    temp_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&temp_shared[j], S[i * head_dim + j] * d_retrieval_s);
            }
            __syncthreads();
            float dk_from_retr_s_j = (alpha_val != 0.0f) ? temp_shared[i] / alpha_val : 0.0f;

            // --- dk from retrieval_r = R_{t-1} @ k ---
            // R currently = alpha * R_{t-1}
            if (i == 0) {
                for (int j = 0; j < head_dim; j++) {
                    q_shared[j] = 0.0f;
                }
            }
            __syncthreads();
            for (int j = 0; j < head_dim; j++) {
                atomicAdd(&q_shared[j], R[i * head_dim + j] * d_retrieval_r);
            }
            __syncthreads();
            float dk_from_retr_r_j = (alpha_val != 0.0f) ? q_shared[i] / alpha_val : 0.0f;

            float dk_i = dk_from_S_j + dk_from_R_outer_j + dk_from_retr_s_j + dk_from_retr_r_j;
            IO_STORE(grad_k, offset + i, dk_i);

            // --- dS propagation through retrieval_s and retrieval_r ---
            // dS_{t-1}[i][j] += d_retrieval_s_i * k[j]
            for (int j = 0; j < head_dim; j++) {
                dS_row[j] += d_retrieval_s * k_shared[j];
            }
            // dR_{t-1}[i][j] += d_retrieval_r_i * k[j]
            for (int j = 0; j < head_dim; j++) {
                dR_row[j] += d_retrieval_r * k_shared[j];
            }

            // --- grad_alpha ---
            // d_alpha from S: sum_{i,j}(dS[i][j] * S_{t-1}[i][j])
            // d_alpha from R: sum_{i,j}(dR[i][j] * R_{t-1}[i][j])
            float d_alpha_partial = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                float s_prev_ij = (alpha_val != 0.0f) ? S[i * head_dim + j] / alpha_val : 0.0f;
                float r_prev_ij = (alpha_val != 0.0f) ? R[i * head_dim + j] / alpha_val : 0.0f;
                d_alpha_partial += dS_row[j] * s_prev_ij + dR_row[j] * r_prev_ij;
            }

            // --- grad_beta: sum_{i,j}(dS[i][j] * delta_s_i * k_j) ---
            float d_beta_partial = delta_s_i * dS_dot_k;

            // --- grad_gamma: sum_{i,j}(dR[i][j] * delta_r_i * k_j) ---
            float d_gamma_partial = delta_r_i * dR_dot_k;

            // Reduce alpha, beta, gamma gradients across threads
            if (i == 0) {
                reduce_buf[0] = 0.0f;
                reduce_buf[1] = 0.0f;
                reduce_buf[2] = 0.0f;
            }
            __syncthreads();
            atomicAdd(&reduce_buf[0], d_alpha_partial);
            atomicAdd(&reduce_buf[1], d_beta_partial);
            atomicAdd(&reduce_buf[2], d_gamma_partial);
            __syncthreads();

            if (i == 0) {
                IO_STORE(grad_alpha, gate_idx, reduce_buf[0]);
                IO_STORE(grad_beta,  gate_idx, reduce_buf[1]);
                IO_STORE(grad_gamma, gate_idx, reduce_buf[2]);
            }

            // --- dS_row, dR_row propagation through alpha decay ---
            for (int j = 0; j < head_dim; j++) {
                dS_row[j] *= alpha_val;
                dR_row[j] *= alpha_val;
            }

            // Undo alpha decay to get S_{t-1} and R_{t-1}
            if (alpha_val != 0.0f) {
                float inv_alpha = 1.0f / alpha_val;
                for (int j = 0; j < head_dim; j++) {
                    S[i * head_dim + j] *= inv_alpha;
                    R[i * head_dim + j] *= inv_alpha;
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concat [grad_q(B*T*H*d) | grad_k(B*T*H*d) | grad_v(B*T*H*d) | grad_alpha(B*T*H) | grad_beta(B*T*H) | grad_gamma(B*T*H)]
int fused_rla_scan_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* alpha, const io_type* beta, const io_type* gamma,
    const io_type* forward_out, const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int num_heads, int head_dim,
    int variant, float clip_threshold
) {
    int total_4d = batch * seq_len * num_heads * head_dim;
    int total_3d = batch * seq_len * num_heads;
    io_type* grad_q     = output_concat;
    io_type* grad_k     = output_concat + total_4d;
    io_type* grad_v     = output_concat + 2 * total_4d;
    io_type* grad_alpha = output_concat + 3 * total_4d;
    io_type* grad_beta  = output_concat + 3 * total_4d + total_3d;
    io_type* grad_gamma = output_concat + 3 * total_4d + 2 * total_3d;

    // Zero the scalar gradient buffers (they use atomicAdd)
    cudaMemsetAsync(grad_alpha, 0, total_3d * sizeof(io_type), stream);
    cudaMemsetAsync(grad_beta,  0, total_3d * sizeof(io_type), stream);
    cudaMemsetAsync(grad_gamma, 0, total_3d * sizeof(io_type), stream);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // Shared memory: S[d][d] + R[d][d] + k_shared[d] + q_shared[d] + temp_shared[d] + reduce_buf[3]
    size_t smem_bytes = 2 * (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + 3 * sizeof(float);

    fused_rla_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, alpha, beta, gamma, forward_out, grad_output,
        grad_q, grad_k, grad_v, grad_alpha, grad_beta, grad_gamma,
        seq_len, num_heads, head_dim,
        variant, clip_threshold
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration (for EXLA fork)
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_rla_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,            // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k,            // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> v,            // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> alpha,        // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> beta,         // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> gamma,        // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> forward_out,  // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> grad_output,  // [B, T, H, d]
    ffi::AnyBuffer variant_buf,                // scalar i32
    ffi::AnyBuffer clip_buf,                   // scalar f32
    ffi::ResultBuffer<FFI_IO_TYPE> grad_q,     // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_k,     // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_v,     // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_alpha, // [B, T, H]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_beta,  // [B, T, H]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_gamma  // [B, T, H]
) {
    int32_t variant = reinterpret_cast<const int32_t*>(variant_buf.untyped_data())[0];
    float clip_threshold = reinterpret_cast<const float*>(clip_buf.untyped_data())[0];
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    int total_3d = batch * seq_len * num_heads;

    // Zero scalar gradient buffers (atomicAdd targets)
    cudaMemsetAsync(grad_alpha->untyped_data(), 0, total_3d * sizeof(io_type), stream);
    cudaMemsetAsync(grad_beta->untyped_data(),  0, total_3d * sizeof(io_type), stream);
    cudaMemsetAsync(grad_gamma->untyped_data(), 0, total_3d * sizeof(io_type), stream);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = 2 * (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + 3 * sizeof(float);

    fused_rla_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<const io_type*>(gamma.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_q->untyped_data()),
        reinterpret_cast<io_type*>(grad_k->untyped_data()),
        reinterpret_cast<io_type*>(grad_v->untyped_data()),
        reinterpret_cast<io_type*>(grad_alpha->untyped_data()),
        reinterpret_cast<io_type*>(grad_beta->untyped_data()),
        reinterpret_cast<io_type*>(grad_gamma->untyped_data()),
        seq_len, num_heads, head_dim,
        static_cast<int>(variant), clip_threshold
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_rla_scan_backward, fused_rla_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // gamma
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Arg<ffi::AnyBuffer>()             // variant (scalar i32)
        .Arg<ffi::AnyBuffer>()             // clip_threshold (scalar f32)
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_q
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_k
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_v
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_alpha
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_beta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_gamma
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_rla_scan_backward_" PRECISION_SUFFIX, "CUDA",
    fused_rla_scan_backward);

#endif  // EXLA_FFI
