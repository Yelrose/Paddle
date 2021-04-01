
#include <cuda_runtime.h>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;



template <typename T, typename IndexT = int>
__global__ void SBMMCUDAKernel(const T* q,
                               const T* k, 
                               const IndexT*  block_row,
                               const IndexT*  block_col,
                               const IndexT*  block_head,
                               T* output,
                               int block_size,
                               int dim_size, 
                               int non_zeros,
                               int seqlen,
                               int head_size,
                               int batch_size){
  int batch_slice = block_size * block_size * non_zeros;
  int non_zero_slice = block_size * block_size;

  CUDA_KERNEL_LOOP(i, batch_size * non_zeros * block_size * block_size) {
      int batch_id = i / batch_slice;
      int batch_shift = i - batch_id * batch_slice;
      int non_zero_id = batch_shift / non_zero_slice;
      int non_zero_shift = batch_shift - non_zero_id * non_zero_slice;
      int row_id = non_zero_shift / block_size;
      int row_base = block_row[non_zero_id] * block_size + row_id;
      int col_id = non_zero_shift - row_id * block_size;
      int col_base = block_col[non_zero_id] * block_size + col_id;
      int head = block_head[non_zero_id];
      if ( (row_base < seqlen) && (col_base < seqlen)) {
          for(int j = 0; j < dim_size; j ++) {
              int qid = ((batch_id * head_size + head) * seqlen + row_base) * dim_size;
              int kid = ((batch_id * head_size + head) * seqlen + col_base) * dim_size;
              output[i] += q[qid + j] * k[kid + j];
          }
      }
  }
}


template <typename T, typename IndexT = int>
__global__ void SBMMBackwardCUDAKernel(const T* q,
                               const T* k, 
                               const IndexT*  block_row,
                               const IndexT*  block_col,
                               const IndexT*  block_head,
                               const T* output_grad,
                               T* q_grad,
                               T* k_grad,
                               int block_size,
                               int dim_size, 
                               int non_zeros,
                               int seqlen,
                               int head_size,
                               int batch_size){
  int batch_slice = block_size * block_size * non_zeros;
  int non_zero_slice = block_size * block_size;

  CUDA_KERNEL_LOOP(i, batch_size * non_zeros * block_size * block_size) {
      int batch_id = i / batch_slice;
      int batch_shift = i - batch_id * batch_slice;
      int non_zero_id = batch_shift / non_zero_slice;
      int non_zero_shift = batch_shift - non_zero_id * non_zero_slice;
      int row_id = non_zero_shift / block_size;
      int row_base = block_row[non_zero_id] * block_size + row_id;
      int col_id = non_zero_shift - row_id * block_size;
      int col_base = block_col[non_zero_id] * block_size + col_id;
      int head = block_head[non_zero_id];
      if ( (row_base < seqlen) && (col_base < seqlen)) {
          for(int j = 0; j < dim_size; j ++) {
              int qid = ((batch_id * head_size + head) * seqlen + row_base) * dim_size;
              int kid = ((batch_id * head_size + head) * seqlen + col_base) * dim_size;
              paddle::platform::CudaAtomicAdd(q_grad + qid + j, output_grad[i] * k[kid + j]);
              paddle::platform::CudaAtomicAdd(k_grad + kid + j, output_grad[i] * q[qid + j]);
          }
      }
  }
}

// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class SparseBlockMatmulCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

    auto* Q = ctx.Input<Tensor>("Q");
    auto* K = ctx.Input<Tensor>("K");
    auto* BlockRow = ctx.Input<Tensor>("BlockRow");
    auto* BlockCol = ctx.Input<Tensor>("BlockCol");
    auto* BlockHead = ctx.Input<Tensor>("BlockHead");

    auto* Out = ctx.Output<Tensor>("Out");
    int block_size = ctx.Attr<int>("block_size");


    T* p_output = Out->mutable_data<T>(ctx.GetPlace());

    size_t batch_size = Q->dims()[0];
    size_t head_size = Q->dims()[1];
    size_t seqlen = Q->dims()[2];
    size_t dim_size = Q->dims()[3];
    size_t non_zeros = BlockRow->dims()[0];
    size_t memset_size = batch_size * non_zeros * block_size * block_size;
    const size_t& memset_bytes = memset_size * sizeof(T);
    const int32_t* block_row = BlockRow->data<int32_t>(); 
    const int32_t* block_col = BlockCol->data<int32_t>(); 
    const int32_t* block_head = BlockHead->data<int32_t>(); 
    const T* q_data = Q->data<T>();
    const T* k_data = K->data<T>();

    cudaMemset(p_output, 0, memset_bytes);

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int n = batch_size * non_zeros * block_size * block_size;
    int block = 512;
    int grid = (n + block - 1) / block;
    SBMMCUDAKernel<T, int><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
      q_data, k_data, block_row, block_col, block_head, p_output, block_size, dim_size, non_zeros, seqlen, head_size, batch_size);

  }
};


// 反向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class SparseBlockMatmulGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* Out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* Q_grad = ctx.Output<Tensor>(framework::GradVarName("Q"));
    auto* K_grad = ctx.Output<Tensor>(framework::GradVarName("K"));

    auto* Q = ctx.Input<Tensor>("Q");
    auto* K = ctx.Input<Tensor>("K");
    auto* BlockRow = ctx.Input<Tensor>("BlockRow");
    auto* BlockCol = ctx.Input<Tensor>("BlockCol");
    auto* BlockHead = ctx.Input<Tensor>("BlockHead");

    // Compute Y Shape
    auto src_dims = Q -> dims();
    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);

    T* q_output = Q_grad -> mutable_data<T>(ctx.GetPlace());
    T* k_output = K_grad -> mutable_data<T>(ctx.GetPlace());

    int block_size = ctx.Attr<int>("block_size");

    // Initialize out
    // Compute Y Shape
    size_t batch_size = Q->dims()[0];
    size_t head_size = Q->dims()[1];
    size_t seqlen = Q->dims()[2];
    size_t dim_size = Q->dims()[3];
    size_t non_zeros = BlockRow->dims()[0];
    const int32_t* block_row = BlockRow->data<int32_t>(); 
    const int32_t* block_col = BlockCol->data<int32_t>(); 
    const int32_t* block_head = BlockHead->data<int32_t>(); 
    const T* q_data = Q->data<T>();
    const T* k_data = K->data<T>();
    const T* out_grad = Out_grad->data<T>();

    cudaMemset(q_output, 0, memset_bytes);
    cudaMemset(k_output, 0, memset_bytes);

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int n = batch_size * non_zeros * block_size * block_size;
    int block = 512;
    int grid = (n + block - 1) / block;
    SBMMBackwardCUDAKernel<T, int><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
        q_data,
        k_data,
        block_row,
        block_col,
        block_head,
        out_grad,
        q_output,
        k_output,
        block_size,
        dim_size,
        non_zeros,
        seqlen,
        head_size,
        batch_size);

  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(sparse_block_matmul,
                        paddle::operators::SparseBlockMatmulCUDAKernel<CUDA, float>,
                        paddle::operators::SparseBlockMatmulCUDAKernel<CUDA, double>);

REGISTER_OP_CUDA_KERNEL(sparse_block_matmul_grad,
                        paddle::operators::SparseBlockMatmulGradCUDAKernel<CUDA, float>,
                        paddle::operators::SparseBlockMatmulGradCUDAKernel<CUDA, double>);

