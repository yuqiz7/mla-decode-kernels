#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

void gqa_decode_v0_launch(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                          const __nv_bfloat16* v_cache, const int* block_table,
                          const int* seq_lens, __nv_bfloat16* out, int B, int Hq,
                          int Hkv, int block_size, int max_blocks, float scale,
                          cudaStream_t stream);

void gqa_decode_v1_launch(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                          const __nv_bfloat16* v_cache, const int* block_table,
                          const int* seq_lens, __nv_bfloat16* out, int B, int Hq,
                          int Hkv, int block_size, int max_blocks, float scale,
                          cudaStream_t stream);

void gqa_decode_v2_launch(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                          const __nv_bfloat16* v_cache, const int* block_table,
                          const int* seq_lens, __nv_bfloat16* out, int B, int Hq,
                          int Hkv, int block_size, int max_blocks, float scale,
                          int num_splits, float* ws_acc, float* ws_m, float* ws_l,
                          cudaStream_t stream);
