#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
// v2_splitkv: the sequence is cut into num_splits segments. A partial kernel
// (grid (Hq, B, num_splits)) runs the v1 warp-per-token scan on its segment
// and writes one fp32 (m, l, acc) state per segment to a global workspace;
// a merge kernel (grid (Hq, B)) combines the segments with the online-softmax
// merge identity and writes the normalized bf16 output.
//
// Workspace layout:
//   ws_acc [B, Hq, num_splits, 128] fp32
//   ws_m, ws_l [B, Hq, num_splits] fp32

struct __align__(8) bf16x4_v2 {
    __nv_bfloat16 v[4];
};

__global__ void gqa_decode_v2_partial_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ block_table, const int* __restrict__ seq_lens, float* __restrict__ ws_acc, float* __restrict__ ws_m, float* __restrict__ ws_l, int Hq, int Hkv, int block_size, int max_blocks, int num_splits, float scale)
{
    constexpr int D = 128;
    constexpr int NW = 4;   // warps per CTA
    const int h = blockIdx.x, b = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int hkv = h / (Hq / Hkv);   // GQA consecutive grouping (HF repeat_kv)
    const int S = seq_lens[b];

    const size_t ws_state = (size_t)(b*Hq + h)*num_splits + split;
    const size_t ws_row = ws_state * D;

    const int chunk = (S + num_splits - 1) / num_splits;
    const int seg_start = split * chunk;
    const int seg_end = min(S, (split + 1) * chunk);
    if (seg_start >= S) {
        // Empty segment: neutral state so the merge kernel's factor is 0.
        if (w == 0) {
            if (lane == 0) { ws_m[ws_state] = -INFINITY; ws_l[ws_state] = 0.f; }
            for (int i = 0; i < 4; ++i) ws_acc[ws_row + 4*lane + i] = 0.f;
        }
        return;
    }

    float qv[4];
    {
        bf16x4_v2 qb = *reinterpret_cast<const bf16x4_v2*>(&q[(size_t)(b*Hq + h)*D + 4*lane]);
        for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(qb.v[i]);
    }

    float m = -INFINITY, l = 0.f;
    float accv[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = seg_start + w; t < seg_end; t += NW) {
        int lb = t / block_size; int s = t % block_size; int p = block_table[b*max_blocks + lb];
        size_t base = ((size_t)(p*block_size + s) * Hkv + hkv) * D;   // cast BEFORE multiplying

        bf16x4_v2 kb = *reinterpret_cast<const bf16x4_v2*>(&k_cache[base + 4*lane]);
        float dot = 0.f;
        for (int i = 0; i < 4; ++i) dot += qv[i] * __bfloat162float(kb.v[i]);
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);

        float sc = dot * scale;
        float m_new = fmaxf(m, sc); float alpha = expf(m - m_new); float pw = expf(sc - m_new);
        l = l*alpha + pw;
        bf16x4_v2 vb = *reinterpret_cast<const bf16x4_v2*>(&v_cache[base + 4*lane]);
        for (int i = 0; i < 4; ++i) accv[i] = accv[i]*alpha + pw*__bfloat162float(vb.v[i]);
        m = m_new;
    }

    // 4-warp merge as in v1, but the segment state is written unnormalized to
    // the workspace instead of being divided by l.
    __shared__ float sm_m[NW], sm_l[NW], sm_acc[NW][D];
    if (lane == 0) { sm_m[w] = m; sm_l[w] = l; }
    for (int i = 0; i < 4; ++i) sm_acc[w][4*lane + i] = accv[i];
    __syncthreads();
    if (w != 0) return;

    float M = -INFINITY;
    for (int i = 0; i < NW; ++i) M = fmaxf(M, sm_m[i]);
    float factor[NW];
    float L = 0.f;
    for (int i = 0; i < NW; ++i) {
        factor[i] = expf(sm_m[i] - M);   // an idle warp has m = -inf -> factor 0
        L += sm_l[i] * factor[i];
    }
    if (lane == 0) { ws_m[ws_state] = M; ws_l[ws_state] = L; }
    for (int i = 0; i < 4; ++i) {
        int d = 4*lane + i;
        float val = 0.f;
        for (int k = 0; k < NW; ++k) val += sm_acc[k][d] * factor[k];
        ws_acc[ws_row + d] = val;
    }
}

__global__ void gqa_decode_v2_merge_kernel(const float* __restrict__ ws_acc, const float* __restrict__ ws_m, const float* __restrict__ ws_l, __nv_bfloat16* __restrict__ out, int Hq, int num_splits)
{
    constexpr int D = 128;
    const int h = blockIdx.x, b = blockIdx.y, j = threadIdx.x;

    const size_t state_base = (size_t)(b*Hq + h)*num_splits;
    float M = -INFINITY;
    for (int i = 0; i < num_splits; ++i) M = fmaxf(M, ws_m[state_base + i]);

    float L = 0.f, acc = 0.f;
    for (int i = 0; i < num_splits; ++i) {
        float factor = expf(ws_m[state_base + i] - M);   // empty segment: -inf -> 0
        L += ws_l[state_base + i] * factor;
        acc += ws_acc[(state_base + i)*D + j] * factor;
    }
    out[(size_t)(b*Hq + h)*D + j] = __float2bfloat16(acc / L);
}

void gqa_decode_v2_launch(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                          const __nv_bfloat16* v_cache, const int* block_table,
                          const int* seq_lens, __nv_bfloat16* out, int B, int Hq,
                          int Hkv, int block_size, int max_blocks, float scale,
                          int num_splits, float* ws_acc, float* ws_m, float* ws_l,
                          cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid_partial(Hq, B, num_splits);
    gqa_decode_v2_partial_kernel<<<grid_partial, block, 0, stream>>>(
        q, k_cache, v_cache, block_table, seq_lens, ws_acc, ws_m, ws_l,
        Hq, Hkv, block_size, max_blocks, num_splits, scale);
    dim3 grid_merge(Hq, B);
    gqa_decode_v2_merge_kernel<<<grid_merge, block, 0, stream>>>(
        ws_acc, ws_m, ws_l, out, Hq, num_splits);
}
