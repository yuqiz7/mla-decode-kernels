#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
// v0_naive: one CTA per (batch b, q-head h), 128 threads, thread j owns dim j.
// Sequential scan over the paged KV of request b with fp32 online softmax.
__global__ void gqa_decode_v0_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ block_table, const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out, int Hq, int Hkv, int block_size, int max_blocks, float scale)
{
    constexpr int D = 128;
    const int h = blockIdx.x, b = blockIdx.y, j = threadIdx.x;
    const int hkv = h / (Hq / Hkv);   // GQA consecutive grouping (HF repeat_kv)
    const int S = seq_lens[b];
    __shared__ float red[D];
    float qj = __bfloat162float(q[(size_t)(b*Hq + h)*D + j]);
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int t = 0; t < S; ++t) {
        int lb = t / block_size; int s = t % block_size; int p = block_table[b*max_blocks + lb];
        size_t base = ((size_t)(p*block_size + s) * Hkv + hkv) * D;   // cast BEFORE multiplying
        red[j] = qj * __bfloat162float(k_cache[base + j]); __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) { if (j < stride) red[j] += red[j + stride]; __syncthreads(); }   // __syncthreads OUTSIDE the if
        float sc = red[0] * scale; __syncthreads();   // second sync: protect red[0] from next iteration's overwrite
        float m_new = fmaxf(m, sc); float alpha = expf(m - m_new); float w = expf(sc - m_new);
        l = l*alpha + w; float vj = __bfloat162float(v_cache[base + j]); acc = acc*alpha + w*vj; m = m_new;
    }
    out[(size_t)(b*Hq + h)*D + j] = __float2bfloat16(acc / l);
}

void gqa_decode_v0_launch(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                          const __nv_bfloat16* v_cache, const int* block_table,
                          const int* seq_lens, __nv_bfloat16* out, int B, int Hq,
                          int Hkv, int block_size, int max_blocks, float scale,
                          cudaStream_t stream)
{
    dim3 grid(Hq, B);
    dim3 block(128);
    gqa_decode_v0_kernel<<<grid, block, 0, stream>>>(q, k_cache, v_cache, block_table, seq_lens, out, Hq, Hkv, block_size, max_blocks, scale);
}
