#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "comm/common.h"

#include "kernels/chunk_prefill/chunk_prefill_runner.hpp"
#include "kernels/flash_attention/xe_fmha_fwd_docode_runner.hpp"
using namespace cute;

std::vector<at::Tensor> mha_fwd(
    at::Tensor& q,        // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor& k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size,
                          // h_k, d) if there is page_table.
    const at::Tensor& v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages,
                          // page_size, h_k, dv) if there is page_table.
    std::optional<const at::Tensor>& q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    const at::Tensor& cu_seqlens_q,         // b+1
    const at::Tensor& cu_seqlens_k,         // b+1
    int max_seqlen_q,
    const at::Tensor& page_table,                      // (b_k, max_num_pages_per_seq)
    std::optional<const at::Tensor>& kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // b
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& seqlens_rotary_,  // b
    std::optional<at::Tensor>& q_descale_,             // (b, h_k), not (b, h)
    std::optional<at::Tensor>& k_descale_,             // (b, h_k)
    std::optional<at::Tensor>& v_descale_,             // (b, h_k)
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor>& scheduler_metadata_,  // (b + 1)
    int num_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin) {
  // TODO: check GPU support
  // auto dprops = at::cuda::getCurrentDeviceProperties();
  // TORCH_CHECK(drops->name.find("B580") != std::string::npos, "sgl_kernel_xpu only supports BMG+");

  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "SGL Kernel XPU only supports fp16 and bf16 type");

  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  CHECK_DEVICE(page_table);
  TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
  TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");

  TORCH_CHECK(q.dim() == 3, "query must be in ragged format");
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");

  CHECK_DEVICE(cu_seqlens_k);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");

  auto const sizes = q.sizes();
  const int batch_size = cu_seqlens_q.size(0) - 1;
  int seqlen_q = max_seqlen_q;
  int total_q = q.size(0);
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const max_num_pages_per_seq = page_table.size(1);
  int const num_pages = k.size(0);
  int const page_size = k.size(1);
  int const seqlen_k = max_num_pages_per_seq * page_size;
  int const total_k = num_pages * page_size;
  int const num_heads_k = k.size(-2);
  int const batch_size_k = page_table.size(0);
  float softmax_scale = softmax_scale_;

  if (seqlen_q == 1 || ((total_q == batch_size * max_seqlen_q) && (max_seqlen_q <= 16))) {
    return decode::mha_fwd(
        q,
        k,
        v,
        q_v_,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        page_table,
        kv_batch_idx_,
        leftpad_k_,
        rotary_cos_,
        rotary_sin_,
        seqlens_rotary_,
        q_descale_,
        k_descale_,
        v_descale_,
        softmax_scale_,
        sinks_,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        scheduler_metadata_,
        num_splits,
        pack_gqa_,
        sm_margin);
  } else {
    return chunkprefill::mha_fwd(
        q,
        k,
        v,
        q_v_,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        page_table,
        kv_batch_idx_,
        leftpad_k_,
        rotary_cos_,
        rotary_sin_,
        seqlens_rotary_,
        q_descale_,
        k_descale_,
        v_descale_,
        softmax_scale_,
        sinks_,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        scheduler_metadata_,
        num_splits,
        pack_gqa_,
        sm_margin);
  }
}
