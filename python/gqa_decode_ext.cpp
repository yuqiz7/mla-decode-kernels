#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "gqa_decode.h"

torch::Tensor gqa_decode_v0(torch::Tensor q, torch::Tensor k_cache,
                            torch::Tensor v_cache, torch::Tensor block_table,
                            torch::Tensor seq_lens, double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 128, "q must be [B, Hq, 128]");

  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k_cache/v_cache must be CUDA tensors");
  TORCH_CHECK(k_cache.scalar_type() == torch::kBFloat16 &&
                  v_cache.scalar_type() == torch::kBFloat16,
              "k_cache/v_cache must be bf16");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),
              "k_cache/v_cache must be contiguous");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == 128,
              "k_cache must be [num_blocks, block_size, Hkv, 128]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache and v_cache must have the same shape");

  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, max_blocks]");

  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [B]");

  const int64_t B = q.size(0);
  const int64_t Hq = q.size(1);
  const int64_t Hkv = k_cache.size(2);
  const int64_t block_size = k_cache.size(1);
  const int64_t max_blocks = block_table.size(1);

  TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(block_table.size(0) == B, "block_table batch dim must match q");
  TORCH_CHECK(seq_lens.size(0) == B, "seq_lens batch dim must match q");
  TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device() &&
                  q.device() == block_table.device() &&
                  q.device() == seq_lens.device(),
              "all tensors must be on the same device");

  auto out = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  gqa_decode_v0_launch(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)B, (int)Hq,
      (int)Hkv, (int)block_size, (int)max_blocks, (float)scale, stream);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gqa_decode_v0", &gqa_decode_v0,
        "GQA decode v0 naive kernel (paged KV, bf16)");
}
