#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
// v1_vectorized: one CTA per (batch b, q-head h), 128 threads = 4 warps.
// Warp w scans tokens t = w, w+4, w+8, ... with 8 B vectorized K/V loads
// (lane owns dims [4*lane, 4*lane+4)) and per-warp fp32 online softmax.
// Warp states are merged once at the end via the split-softmax formula;
// the loop body has no __syncthreads at all.

struct __align__(8) bf16x4 {
    __nv_bfloat16 v[4];
};

__global__ void gqa_decode_v1_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ block_table, const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out, int Hq, int Hkv, int block_size, int max_blocks, float scale)
{
    constexpr int D = 128;
    constexpr int NW = 4;   // warps per CTA
    const int h = blockIdx.x, b = blockIdx.y;
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int hkv = h / (Hq / Hkv);   // GQA consecutive grouping (HF repeat_kv)
    const int S = seq_lens[b];

    float qv[4];
    {
        bf16x4 qb = *reinterpret_cast<const bf16x4*>(&q[(size_t)(b*Hq + h)*D + 4*lane]);
        for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(qb.v[i]);
    }

    float m = -INFINITY, l = 0.f;
    float accv[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = w; t < S; t += NW) {
        int lb = t / block_size; int s = t % block_size; int p = block_table[b*max_blocks + lb];
        size_t base = ((size_t)(p*block_size + s) * Hkv + hkv) * D;   // cast BEFORE multiplying

        bf16x4 kb = *reinterpret_cast<const bf16x4*>(&k_cache[base + 4*lane]);
        float dot = 0.f;
        for (int i = 0; i < 4; ++i) dot += qv[i] * __bfloat162float(kb.v[i]);
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);

        float sc = dot * scale;
        float m_new = fmaxf(m, sc); float alpha = expf(m - m_new); float pw = expf(sc - m_new);
        l = l*alpha + pw;
        bf16x4 vb = *reinterpret_cast<const bf16x4*>(&v_cache[base + 4*lane]);
        for (int i = 0; i < 4; ++i) accv[i] = accv[i]*alpha + pw*__bfloat162float(vb.v[i]);
        m = m_new;
    }

    // Merge the 4 per-warp states. Raw (m, l, acc) are staged in shared
    // memory; the merge factors are computed on the read side so the global
    // max M is derived only once, by the merging warp.
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
    for (int i = 0; i < 4; ++i) {
        int d = 4*lane + i;
        float val = 0.f;
        for (int k = 0; k < NW; ++k) val += sm_acc[k][d] * factor[k];
        out[(size_t)(b*Hq + h)*D + d] = __float2bfloat16(val / L);
    }
}

void gqa_decode_v1_launch(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                          const __nv_bfloat16* v_cache, const int* block_table,
                          const int* seq_lens, __nv_bfloat16* out, int B, int Hq,
                          int Hkv, int block_size, int max_blocks, float scale,
                          cudaStream_t stream)
{
    dim3 grid(Hq, B);
    dim3 block(128);
    gqa_decode_v1_kernel<<<grid, block, 0, stream>>>(q, k_cache, v_cache, block_table, seq_lens, out, Hq, Hkv, block_size, max_blocks, scale);
}
