/*
 * Edifice CUDA NIF Bridge
 *
 * Minimal C NIF that dlopen's libedifice_cuda_kernels.so and exposes
 * fused scan kernels as Erlang NIF functions. Takes device pointer
 * integers (from Nx.to_pointer) and dimensions, launches the fused
 * kernel, and returns the output pointer integer.
 *
 * Memory management: Each cudaMalloc'd output is wrapped in an Erlang
 * NIF resource. The resource's destructor calls cudaFree when the BEAM
 * garbage collects the reference. The Elixir side must hold the resource
 * reference for the lifetime of the Nx tensor wrapping the pointer.
 *
 * Marked ERL_NIF_DIRTY_JOB_IO_BOUND so GPU blocking doesn't stall
 * the BEAM scheduler.
 */

#define _GNU_SOURCE  /* for dladdr / Dl_info */

#include <erl_nif.h>
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* CUDA runtime types — avoid pulling in full cuda_runtime.h */
typedef void* cudaStream_t;
typedef int cudaError_t;

/* Function pointer types matching the C-linkage launch wrappers */
typedef int (*mingru_launch_fn)(
    cudaStream_t stream,
    const float* gates,
    const float* candidates,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
);

typedef int (*minlstm_launch_fn)(
    cudaStream_t stream,
    const float* forget_gates,
    const float* input_gates,
    const float* candidates,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
);

/* NativeRecurrence + Liquid launch wrappers (same 2-input signature) */
typedef int (*scan_2input_launch_fn)(
    cudaStream_t stream,
    const float* input1,
    const float* input2,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
);

/* Delta rule scan launch wrappers (DeltaNet / GatedDeltaNet) */
typedef int (*delta_net_launch_fn)(
    cudaStream_t stream,
    const float* q, const float* k, const float* v, const float* beta,
    float* output,
    int batch, int seq_len, int num_heads, int head_dim
);

typedef int (*gated_delta_net_launch_fn)(
    cudaStream_t stream,
    const float* q, const float* k, const float* v, const float* beta,
    const float* alpha,
    float* output,
    int batch, int seq_len, int num_heads, int head_dim
);

/* cudaMalloc / cudaFree — resolved from libcudart */
typedef cudaError_t (*cuda_malloc_fn)(void** devPtr, size_t size);
typedef cudaError_t (*cuda_free_fn)(void* devPtr);
typedef cudaError_t (*cuda_device_synchronize_fn)(void);

/* Resolved function pointers (set on NIF load) */
static mingru_launch_fn  s_mingru_launch  = NULL;
static minlstm_launch_fn s_minlstm_launch = NULL;
static scan_2input_launch_fn s_elu_gru_launch     = NULL;
static scan_2input_launch_fn s_real_gru_launch    = NULL;
static scan_2input_launch_fn s_diag_linear_launch = NULL;
static scan_2input_launch_fn s_liquid_launch      = NULL;
static scan_2input_launch_fn s_linear_launch     = NULL;
static delta_net_launch_fn       s_delta_net_launch       = NULL;
static gated_delta_net_launch_fn s_gated_delta_net_launch = NULL;
static cuda_malloc_fn    s_cuda_malloc    = NULL;
static cuda_free_fn      s_cuda_free      = NULL;
static cuda_device_synchronize_fn s_cuda_sync = NULL;

static void* s_kernels_handle = NULL;
static void* s_cudart_handle  = NULL;

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

static ERL_NIF_TERM make_error(ErlNifEnv* env, const char* reason) {
    return enif_make_tuple2(env,
        enif_make_atom(env, "error"),
        enif_make_string(env, reason, ERL_NIF_LATIN1));
}

/* ========================================================================== */
/* GPU Buffer Resource — ensures cudaFree on GC                               */
/* ========================================================================== */

/* NIF resource type for tracking GPU allocations */
static ErlNifResourceType* s_gpu_buffer_type = NULL;

typedef struct {
    void* device_ptr;  /* GPU pointer from cudaMalloc */
} GpuBuffer;

/* Called by BEAM GC when the resource reference count drops to zero */
static void gpu_buffer_dtor(ErlNifEnv* env, void* obj) {
    (void)env;
    GpuBuffer* buf = (GpuBuffer*)obj;
    if (buf->device_ptr && s_cuda_free) {
        s_cuda_free(buf->device_ptr);
        buf->device_ptr = NULL;
    }
}

/*
 * Allocate a GPU buffer, wrap it in a NIF resource, and return
 * {:ok, pointer_int, resource_ref} or {:error, reason}.
 *
 * The caller must hold resource_ref for the lifetime of any Nx tensor
 * wrapping pointer_int. When resource_ref is GC'd, cudaFree is called.
 */
static ERL_NIF_TERM alloc_gpu_buffer(ErlNifEnv* env, size_t bytes) {
    void* dev_ptr = NULL;
    cudaError_t err = s_cuda_malloc(&dev_ptr, bytes);
    if (err != 0) {
        return make_error(env, "cudaMalloc failed for output buffer");
    }

    /* Create a NIF resource that will call cudaFree in its destructor */
    GpuBuffer* buf = enif_alloc_resource(s_gpu_buffer_type, sizeof(GpuBuffer));
    if (!buf) {
        s_cuda_free(dev_ptr);
        return make_error(env, "failed to allocate NIF resource");
    }
    buf->device_ptr = dev_ptr;

    /* Make an Erlang term referencing this resource */
    ERL_NIF_TERM ref = enif_make_resource(env, buf);

    /* Release our ownership — the term now holds the only reference.
     * When the BEAM GCs the term, gpu_buffer_dtor fires. */
    enif_release_resource(buf);

    return enif_make_tuple3(env,
        enif_make_atom(env, "ok"),
        enif_make_uint64(env, (uint64_t)(uintptr_t)dev_ptr),
        ref);
}

/* ========================================================================== */
/* NIF: fused_mingru_scan                                                     */
/* ========================================================================== */

/*
 * fused_mingru_scan(gates_ptr, candidates_ptr, h0_ptr, batch, seq_len, hidden)
 *   -> {:ok, output_ptr, gc_ref} | {:error, reason}
 *
 * All pointer args are uint64 device pointer addresses.
 * gc_ref is a NIF resource — hold it to prevent cudaFree until done.
 */
static ERL_NIF_TERM nif_fused_mingru_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t gates_ptr, cand_ptr, h0_ptr;
    int batch, seq_len, hidden;

    if (!s_mingru_launch)
        return make_error(env, "kernel library not loaded");

    if (!enif_get_uint64(env, argv[0], &gates_ptr) ||
        !enif_get_uint64(env, argv[1], &cand_ptr) ||
        !enif_get_uint64(env, argv[2], &h0_ptr) ||
        !enif_get_int(env, argv[3], &batch) ||
        !enif_get_int(env, argv[4], &seq_len) ||
        !enif_get_int(env, argv[5], &hidden))
    {
        return enif_make_badarg(env);
    }

    if (batch <= 0 || seq_len <= 0 || hidden <= 0)
        return make_error(env, "dimensions must be positive");

    /* Allocate output buffer on GPU, wrapped in a GC-tracked resource */
    size_t out_bytes = (size_t)batch * seq_len * hidden * sizeof(float);
    ERL_NIF_TERM alloc_result = alloc_gpu_buffer(env, out_bytes);

    /* Check if allocation succeeded */
    int arity;
    const ERL_NIF_TERM* tuple;
    if (!enif_get_tuple(env, alloc_result, &arity, &tuple) || arity < 2) {
        return alloc_result;  /* propagate error */
    }

    /* Check for :error atom */
    char atom_buf[8];
    if (enif_get_atom(env, tuple[0], atom_buf, sizeof(atom_buf), ERL_NIF_LATIN1)
        && strcmp(atom_buf, "error") == 0) {
        return alloc_result;
    }

    /* Extract the device pointer from {:ok, ptr, ref} */
    uint64_t out_ptr;
    enif_get_uint64(env, tuple[1], &out_ptr);

    /* Launch the fused kernel on the default stream */
    int launch_err = s_mingru_launch(
        NULL,  /* default stream */
        (const float*)(uintptr_t)gates_ptr,
        (const float*)(uintptr_t)cand_ptr,
        (const float*)(uintptr_t)h0_ptr,
        (float*)(uintptr_t)out_ptr,
        batch, seq_len, hidden
    );

    if (launch_err != 0) {
        /* Resource ref goes out of scope → GC will call cudaFree */
        return make_error(env, "kernel launch failed");
    }

    /* Synchronize — blocks this dirty scheduler thread until GPU finishes */
    cudaError_t err = s_cuda_sync();
    if (err != 0) {
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return alloc_result;  /* {:ok, output_ptr, gc_ref} */
}

/* ========================================================================== */
/* NIF: fused_minlstm_scan                                                    */
/* ========================================================================== */

/*
 * fused_minlstm_scan(forget_ptr, input_ptr, cand_ptr, h0_ptr,
 *                    batch, seq_len, hidden)
 *   -> {:ok, output_ptr, gc_ref} | {:error, reason}
 */
static ERL_NIF_TERM nif_fused_minlstm_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t forget_ptr, input_ptr, cand_ptr, h0_ptr;
    int batch, seq_len, hidden;

    if (!s_minlstm_launch)
        return make_error(env, "kernel library not loaded");

    if (!enif_get_uint64(env, argv[0], &forget_ptr) ||
        !enif_get_uint64(env, argv[1], &input_ptr) ||
        !enif_get_uint64(env, argv[2], &cand_ptr) ||
        !enif_get_uint64(env, argv[3], &h0_ptr) ||
        !enif_get_int(env, argv[4], &batch) ||
        !enif_get_int(env, argv[5], &seq_len) ||
        !enif_get_int(env, argv[6], &hidden))
    {
        return enif_make_badarg(env);
    }

    if (batch <= 0 || seq_len <= 0 || hidden <= 0)
        return make_error(env, "dimensions must be positive");

    size_t out_bytes = (size_t)batch * seq_len * hidden * sizeof(float);
    ERL_NIF_TERM alloc_result = alloc_gpu_buffer(env, out_bytes);

    int arity;
    const ERL_NIF_TERM* tuple;
    if (!enif_get_tuple(env, alloc_result, &arity, &tuple) || arity < 2) {
        return alloc_result;
    }

    char atom_buf[8];
    if (enif_get_atom(env, tuple[0], atom_buf, sizeof(atom_buf), ERL_NIF_LATIN1)
        && strcmp(atom_buf, "error") == 0) {
        return alloc_result;
    }

    uint64_t out_ptr;
    enif_get_uint64(env, tuple[1], &out_ptr);

    int launch_err = s_minlstm_launch(
        NULL,
        (const float*)(uintptr_t)forget_ptr,
        (const float*)(uintptr_t)input_ptr,
        (const float*)(uintptr_t)cand_ptr,
        (const float*)(uintptr_t)h0_ptr,
        (float*)(uintptr_t)out_ptr,
        batch, seq_len, hidden
    );

    if (launch_err != 0) {
        return make_error(env, "kernel launch failed");
    }

    cudaError_t err = s_cuda_sync();
    if (err != 0) {
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return alloc_result;  /* {:ok, output_ptr, gc_ref} */
}

/* ========================================================================== */
/* NIF: fused 2-input scans (ELU-GRU, Real-GRU, DiagLinear, Liquid)          */
/* ========================================================================== */

/*
 * Generic NIF for 2-input scan kernels.
 * fused_*_scan(input1_ptr, input2_ptr, h0_ptr, batch, seq_len, hidden)
 *   -> {:ok, output_ptr, gc_ref} | {:error, reason}
 */
static ERL_NIF_TERM nif_fused_2input_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[],
    scan_2input_launch_fn launch_fn, const char* kernel_name)
{
    uint64_t in1_ptr, in2_ptr, h0_ptr;
    int batch, seq_len, hidden;

    if (!launch_fn) {
        char msg[64];
        snprintf(msg, sizeof(msg), "%s kernel not loaded", kernel_name);
        return make_error(env, msg);
    }

    if (!enif_get_uint64(env, argv[0], &in1_ptr) ||
        !enif_get_uint64(env, argv[1], &in2_ptr) ||
        !enif_get_uint64(env, argv[2], &h0_ptr) ||
        !enif_get_int(env, argv[3], &batch) ||
        !enif_get_int(env, argv[4], &seq_len) ||
        !enif_get_int(env, argv[5], &hidden))
    {
        return enif_make_badarg(env);
    }

    if (batch <= 0 || seq_len <= 0 || hidden <= 0)
        return make_error(env, "dimensions must be positive");

    size_t out_bytes = (size_t)batch * seq_len * hidden * sizeof(float);
    ERL_NIF_TERM alloc_result = alloc_gpu_buffer(env, out_bytes);

    int arity;
    const ERL_NIF_TERM* tuple;
    if (!enif_get_tuple(env, alloc_result, &arity, &tuple) || arity < 2) {
        return alloc_result;
    }

    char atom_buf[8];
    if (enif_get_atom(env, tuple[0], atom_buf, sizeof(atom_buf), ERL_NIF_LATIN1)
        && strcmp(atom_buf, "error") == 0) {
        return alloc_result;
    }

    uint64_t out_ptr;
    enif_get_uint64(env, tuple[1], &out_ptr);

    int launch_err = launch_fn(
        NULL,
        (const float*)(uintptr_t)in1_ptr,
        (const float*)(uintptr_t)in2_ptr,
        (const float*)(uintptr_t)h0_ptr,
        (float*)(uintptr_t)out_ptr,
        batch, seq_len, hidden
    );

    if (launch_err != 0) {
        return make_error(env, "kernel launch failed");
    }

    cudaError_t err = s_cuda_sync();
    if (err != 0) {
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return alloc_result;
}

static ERL_NIF_TERM nif_fused_elu_gru_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return nif_fused_2input_scan(env, argc, argv, s_elu_gru_launch, "elu_gru");
}

static ERL_NIF_TERM nif_fused_real_gru_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return nif_fused_2input_scan(env, argc, argv, s_real_gru_launch, "real_gru");
}

static ERL_NIF_TERM nif_fused_diag_linear_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return nif_fused_2input_scan(env, argc, argv, s_diag_linear_launch, "diag_linear");
}

static ERL_NIF_TERM nif_fused_liquid_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return nif_fused_2input_scan(env, argc, argv, s_liquid_launch, "liquid");
}

static ERL_NIF_TERM nif_fused_linear_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return nif_fused_2input_scan(env, argc, argv, s_linear_launch, "linear");
}

/* ========================================================================== */
/* NIF: fused delta net scan (DeltaNet — no alpha)                            */
/* ========================================================================== */

/*
 * fused_delta_net_scan(q_ptr, k_ptr, v_ptr, beta_ptr,
 *                      batch, seq_len, num_heads, head_dim)
 *   -> {:ok, output_ptr, gc_ref} | {:error, reason}
 */
static ERL_NIF_TERM nif_fused_delta_net_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t q_ptr, k_ptr, v_ptr, beta_ptr;
    int batch, seq_len, num_heads, head_dim;

    if (!s_delta_net_launch)
        return make_error(env, "delta_net kernel not loaded");

    if (!enif_get_uint64(env, argv[0], &q_ptr) ||
        !enif_get_uint64(env, argv[1], &k_ptr) ||
        !enif_get_uint64(env, argv[2], &v_ptr) ||
        !enif_get_uint64(env, argv[3], &beta_ptr) ||
        !enif_get_int(env, argv[4], &batch) ||
        !enif_get_int(env, argv[5], &seq_len) ||
        !enif_get_int(env, argv[6], &num_heads) ||
        !enif_get_int(env, argv[7], &head_dim))
    {
        return enif_make_badarg(env);
    }

    if (batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0)
        return make_error(env, "dimensions must be positive");

    size_t out_bytes = (size_t)batch * seq_len * num_heads * head_dim * sizeof(float);
    ERL_NIF_TERM alloc_result = alloc_gpu_buffer(env, out_bytes);

    int arity;
    const ERL_NIF_TERM* tuple;
    if (!enif_get_tuple(env, alloc_result, &arity, &tuple) || arity < 2) {
        return alloc_result;
    }

    char atom_buf[8];
    if (enif_get_atom(env, tuple[0], atom_buf, sizeof(atom_buf), ERL_NIF_LATIN1)
        && strcmp(atom_buf, "error") == 0) {
        return alloc_result;
    }

    uint64_t out_ptr;
    enif_get_uint64(env, tuple[1], &out_ptr);

    int launch_err = s_delta_net_launch(
        NULL,
        (const float*)(uintptr_t)q_ptr,
        (const float*)(uintptr_t)k_ptr,
        (const float*)(uintptr_t)v_ptr,
        (const float*)(uintptr_t)beta_ptr,
        (float*)(uintptr_t)out_ptr,
        batch, seq_len, num_heads, head_dim
    );

    if (launch_err != 0) {
        return make_error(env, "delta_net kernel launch failed");
    }

    cudaError_t err = s_cuda_sync();
    if (err != 0) {
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return alloc_result;
}

/* ========================================================================== */
/* NIF: fused gated delta net scan (GatedDeltaNet — with alpha)               */
/* ========================================================================== */

/*
 * fused_gated_delta_net_scan(q_ptr, k_ptr, v_ptr, beta_ptr, alpha_ptr,
 *                            batch, seq_len, num_heads, head_dim)
 *   -> {:ok, output_ptr, gc_ref} | {:error, reason}
 */
static ERL_NIF_TERM nif_fused_gated_delta_net_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t q_ptr, k_ptr, v_ptr, beta_ptr, alpha_ptr;
    int batch, seq_len, num_heads, head_dim;

    if (!s_gated_delta_net_launch)
        return make_error(env, "gated_delta_net kernel not loaded");

    if (!enif_get_uint64(env, argv[0], &q_ptr) ||
        !enif_get_uint64(env, argv[1], &k_ptr) ||
        !enif_get_uint64(env, argv[2], &v_ptr) ||
        !enif_get_uint64(env, argv[3], &beta_ptr) ||
        !enif_get_uint64(env, argv[4], &alpha_ptr) ||
        !enif_get_int(env, argv[5], &batch) ||
        !enif_get_int(env, argv[6], &seq_len) ||
        !enif_get_int(env, argv[7], &num_heads) ||
        !enif_get_int(env, argv[8], &head_dim))
    {
        return enif_make_badarg(env);
    }

    if (batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0)
        return make_error(env, "dimensions must be positive");

    size_t out_bytes = (size_t)batch * seq_len * num_heads * head_dim * sizeof(float);
    ERL_NIF_TERM alloc_result = alloc_gpu_buffer(env, out_bytes);

    int arity;
    const ERL_NIF_TERM* tuple;
    if (!enif_get_tuple(env, alloc_result, &arity, &tuple) || arity < 2) {
        return alloc_result;
    }

    char atom_buf[8];
    if (enif_get_atom(env, tuple[0], atom_buf, sizeof(atom_buf), ERL_NIF_LATIN1)
        && strcmp(atom_buf, "error") == 0) {
        return alloc_result;
    }

    uint64_t out_ptr;
    enif_get_uint64(env, tuple[1], &out_ptr);

    int launch_err = s_gated_delta_net_launch(
        NULL,
        (const float*)(uintptr_t)q_ptr,
        (const float*)(uintptr_t)k_ptr,
        (const float*)(uintptr_t)v_ptr,
        (const float*)(uintptr_t)beta_ptr,
        (const float*)(uintptr_t)alpha_ptr,
        (float*)(uintptr_t)out_ptr,
        batch, seq_len, num_heads, head_dim
    );

    if (launch_err != 0) {
        return make_error(env, "gated_delta_net kernel launch failed");
    }

    cudaError_t err = s_cuda_sync();
    if (err != 0) {
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return alloc_result;
}

/* ========================================================================== */
/* NIF Load — resolve all symbols                                             */
/* ========================================================================== */

static int nif_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)load_info;

    /* Register the GPU buffer resource type with a destructor */
    s_gpu_buffer_type = enif_open_resource_type(
        env, NULL, "gpu_buffer",
        gpu_buffer_dtor,
        ERL_NIF_RT_CREATE, NULL);

    if (!s_gpu_buffer_type) {
        return -1;  /* fatal — can't register resource type */
    }

    char kernels_path[512];

    /* Try to load libcudart first — may already be in LD_LIBRARY_PATH */
    s_cudart_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (!s_cudart_handle) {
        /* Try versioned name */
        s_cudart_handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!s_cudart_handle) {
        /* CUDA not available — NIF loads but functions return errors */
        return 0;
    }

    s_cuda_malloc = (cuda_malloc_fn)dlsym(s_cudart_handle, "cudaMalloc");
    s_cuda_free   = (cuda_free_fn)dlsym(s_cudart_handle, "cudaFree");
    s_cuda_sync   = (cuda_device_synchronize_fn)dlsym(s_cudart_handle, "cudaDeviceSynchronize");

    if (!s_cuda_malloc || !s_cuda_free || !s_cuda_sync) {
        return 0;
    }

    /* First try: relative to CWD (mix project root) */
    s_kernels_handle = dlopen("priv/cuda/libedifice_cuda_kernels.so", RTLD_NOW);
    if (!s_kernels_handle) {
        /* Second try: find via Dl_info on our own symbol */
        Dl_info info;
        if (dladdr((void*)nif_load, &info) && info.dli_fname) {
            const char* fname = info.dli_fname;
            const char* last_slash = strrchr(fname, '/');
            if (last_slash) {
                size_t dir_len = (size_t)(last_slash - fname);
                if (dir_len + 40 < sizeof(kernels_path)) {
                    memcpy(kernels_path, fname, dir_len);
                    strcpy(kernels_path + dir_len, "/cuda/libedifice_cuda_kernels.so");
                    s_kernels_handle = dlopen(kernels_path, RTLD_NOW);
                }
            }
        }
    }

    if (!s_kernels_handle) {
        return 0;
    }

    s_mingru_launch = (mingru_launch_fn)dlsym(
        s_kernels_handle, "fused_mingru_scan_launch");
    s_minlstm_launch = (minlstm_launch_fn)dlsym(
        s_kernels_handle, "fused_minlstm_scan_launch");
    s_elu_gru_launch = (scan_2input_launch_fn)dlsym(
        s_kernels_handle, "fused_elu_gru_scan_launch");
    s_real_gru_launch = (scan_2input_launch_fn)dlsym(
        s_kernels_handle, "fused_real_gru_scan_launch");
    s_diag_linear_launch = (scan_2input_launch_fn)dlsym(
        s_kernels_handle, "fused_diag_linear_scan_launch");
    s_liquid_launch = (scan_2input_launch_fn)dlsym(
        s_kernels_handle, "fused_liquid_scan_launch");
    s_linear_launch = (scan_2input_launch_fn)dlsym(
        s_kernels_handle, "fused_linear_scan_launch");
    s_delta_net_launch = (delta_net_launch_fn)dlsym(
        s_kernels_handle, "fused_delta_net_scan_launch");
    s_gated_delta_net_launch = (gated_delta_net_launch_fn)dlsym(
        s_kernels_handle, "fused_gated_delta_net_scan_launch");

    return 0;
}

static void nif_unload(ErlNifEnv* env, void* priv_data) {
    (void)env;
    (void)priv_data;

    if (s_kernels_handle) {
        dlclose(s_kernels_handle);
        s_kernels_handle = NULL;
    }
    if (s_cudart_handle) {
        dlclose(s_cudart_handle);
        s_cudart_handle = NULL;
    }
    s_mingru_launch          = NULL;
    s_minlstm_launch         = NULL;
    s_elu_gru_launch         = NULL;
    s_real_gru_launch        = NULL;
    s_diag_linear_launch     = NULL;
    s_liquid_launch          = NULL;
    s_linear_launch          = NULL;
    s_delta_net_launch       = NULL;
    s_gated_delta_net_launch = NULL;
    s_cuda_malloc            = NULL;
    s_cuda_free          = NULL;
    s_cuda_sync          = NULL;
}

/* ========================================================================== */
/* NIF Registration                                                           */
/* ========================================================================== */

static ErlNifFunc nif_funcs[] = {
    {"fused_mingru_scan",       6, nif_fused_mingru_scan,       ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_minlstm_scan",      7, nif_fused_minlstm_scan,      ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_elu_gru_scan",      6, nif_fused_elu_gru_scan,      ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_real_gru_scan",     6, nif_fused_real_gru_scan,     ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_diag_linear_scan",  6, nif_fused_diag_linear_scan,  ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_liquid_scan",       6, nif_fused_liquid_scan,       ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_linear_scan",           6, nif_fused_linear_scan,           ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_delta_net_scan",        8, nif_fused_delta_net_scan,        ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_gated_delta_net_scan",  9, nif_fused_gated_delta_net_scan,  ERL_NIF_DIRTY_JOB_IO_BOUND}
};

ERL_NIF_INIT(Elixir.Edifice.CUDA.NIF, nif_funcs, nif_load, NULL, NULL, nif_unload)
