
#include <cuda_runtime.h>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"
#include <iostream>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;



template <typename T>
__global__ void DSMMCUDAKernel(const int* col_index, 
                               const int* row_index,
                               const int* head_index,
                               T* output,
                               const T* v,
                               const T* attn,
                               int hidden_size,
                               int seqlen,
                               int num_heads,
                               int non_zeros,
                               int batch_size,
                               int block_size) {
                                      
  
  CUDA_KERNEL_LOOP(all_ind, batch_size * non_zeros * block_size * block_size) {
    int seqlen_slice = hidden_size;
    int head_slice = seqlen_slice * seqlen;
    int batch_slice = head_slice * num_heads; 

    int cum_attn_dims_3 = block_size;
    int cum_attn_dims_2 = cum_attn_dims_3 * block_size;
    int cum_attn_dims_1 = cum_attn_dims_2 * non_zeros;
    
    int b = all_ind / (non_zeros * block_size * block_size);
    int ind = all_ind - b * non_zeros * block_size * block_size;
    int i = ind / (block_size * block_size); 
    int block_i = (ind - i * block_size * block_size) / block_size;
    int block_j = ind % block_size;
    int head_id = head_index[i];

    for(size_t d = 0; d < hidden_size; d ++) {
        int64_t scatter_index = b * batch_slice + head_id * head_slice + (row_index[i] * block_size + block_i) * seqlen_slice + d; 
        int64_t gather_index = b * batch_slice + head_id * head_slice + (col_index[i] * block_size + block_j) * seqlen_slice + d; 
        int64_t attn_index = b * cum_attn_dims_1 + i * cum_attn_dims_2 + block_i * cum_attn_dims_3 + block_j; 
        paddle::platform::CudaAtomicAdd(output + scatter_index, attn[attn_index] * v[gather_index]);
    }
  }
}

template <typename T>
__global__ void DSMMGradCUDAKernel(const int* col_index, 
                               const int* row_index,
                               const int* head_index,
                               const T* output,
                               const T* v,
                               const T* attn,
                               T* grad_v,
                               T* grad_attn,
                               int hidden_size,
                               int seqlen,
                               int num_heads,
                               int non_zeros,
                               int batch_size,
                               int block_size) {
                                      
  
  CUDA_KERNEL_LOOP(all_ind, batch_size * non_zeros * block_size * block_size) {
    int seqlen_slice = hidden_size;
    int head_slice = seqlen_slice * seqlen;
    int batch_slice = head_slice * num_heads; 

    int cum_attn_dims_3 = block_size;
    int cum_attn_dims_2 = cum_attn_dims_3 * block_size;
    int cum_attn_dims_1 = cum_attn_dims_2 * non_zeros;
    
    int b = all_ind / (non_zeros * block_size * block_size);
    int ind = all_ind - b * non_zeros * block_size * block_size;
    int i = ind / (block_size * block_size); 
    int block_i = (ind - i * block_size * block_size) / block_size;
    int block_j = ind % block_size;
    int head_id = head_index[i];

    for(size_t d = 0; d < hidden_size; d ++) {
        int64_t scatter_index = b * batch_slice + head_id * head_slice + (row_index[i] * block_size + block_i) * seqlen_slice + d; 
        int64_t gather_index = b * batch_slice + head_id * head_slice + (col_index[i] * block_size + block_j) * seqlen_slice + d; 
        int64_t attn_index = b * cum_attn_dims_1 + i * cum_attn_dims_2 + block_i * cum_attn_dims_3 + block_j; 
        //paddle::platform::CudaAtomicAdd(output + scatter_index, attn[attn_index] * v[gather_index]);
        paddle::platform::CudaAtomicAdd(grad_attn + attn_index, output[scatter_index] * v[gather_index]);
        paddle::platform::CudaAtomicAdd(grad_v + gather_index, output[scatter_index] * attn[attn_index]);
    }
  }
}



// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class DenseSparseBlockMatmulCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* V = ctx.Input<Tensor>("V"); // [B, H, L, D]
    auto* BlockCol = ctx.Input<Tensor>("BlockCol"); // 
    auto* BlockRow = ctx.Input<Tensor>("BlockRow");
    auto* BlockHead = ctx.Input<Tensor>("BlockHead");
    auto* Attn = ctx.Input<Tensor>("Attn"); // [B, *, block_size, block_size]
    auto* Out = ctx.Output<Tensor>("Out");
    int block_size = ctx.Attr<int>("block_size");

    T* p_output = Out -> mutable_data<T>(ctx.GetPlace());
    const T* p_input = V -> data<T>();
    const T* attn_input = Attn -> data<T>();

    const int* col_index = BlockCol -> data<int>();
    const int* row_index = BlockRow -> data<int>();
    const int* head_index = BlockHead -> data<int>();
    size_t non_zeros = BlockCol -> dims()[0];
    size_t batch_size = V -> dims()[0];
    size_t num_heads = V -> dims()[1];
    size_t seqlen = V -> dims()[2];
    size_t hidden_size = V -> dims()[3];

    auto src_dims = V -> dims();
    std::vector<int64_t> cum_src_dims = {src_dims[0], src_dims[1], src_dims[2], src_dims[3]};
    for(int i = src_dims.size() - 2; i >= 0;i --) {
        cum_src_dims[i] = cum_src_dims[i] * cum_src_dims[i + 1];
    }
    cudaMemset(p_output, 0, cum_src_dims[0] * sizeof(T));

    auto attn_dims = Attn -> dims();
    std::vector<int64_t> cum_attn_dims = {attn_dims[0], attn_dims[1], attn_dims[2], attn_dims[3]};

    for(int i = attn_dims.size() - 2; i >= 0;i --) {
        cum_attn_dims[i] = cum_attn_dims[i] * cum_attn_dims[i + 1];
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
 
    int block = 512;
    int n = batch_size * non_zeros * block_size * block_size;
    //std::cerr << " blocks  " << n << std::endl;
    int grid = (n + block - 1) / block;
    DSMMCUDAKernel<T><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
      col_index, row_index, head_index, p_output, p_input, attn_input, hidden_size,
      seqlen, num_heads, non_zeros, batch_size, block_size);

  }

};


// 反向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class DenseSparseBlockMatmulGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* V = ctx.Input<Tensor>("V"); // [B, H, L, D]
    auto* BlockCol = ctx.Input<Tensor>("BlockCol"); // 
    auto* BlockRow = ctx.Input<Tensor>("BlockRow");
    auto* BlockHead = ctx.Input<Tensor>("BlockHead");
    auto* Attn = ctx.Input<Tensor>("Attn"); // [B, *, block_size, block_size]
    auto* Out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int block_size = ctx.Attr<int>("block_size");

    //Init Output 
    auto* grad_Attn = ctx.Output<Tensor>(framework::GradVarName("Attn"));
    auto* grad_V = ctx.Output<Tensor>(framework::GradVarName("V"));

    const T* p_output = Out->data<T>();
    const T* p_input = V->data<T>();
    const T* attn_input = Attn->data<T>();

    T* grad_p_input = grad_V -> mutable_data<T>(ctx.GetPlace());
    T* grad_attn_input = grad_Attn -> mutable_data<T>(ctx.GetPlace());
    // Compute Out Shape


    const int* col_index = BlockCol -> data<int>();
    const int* row_index = BlockRow -> data<int>();
    const int* head_index = BlockHead -> data<int>();
    size_t non_zeros = BlockCol -> dims()[0];
    size_t batch_size = V -> dims()[0];
    size_t num_heads = V -> dims()[1];
    size_t seqlen = V -> dims()[2];
    size_t hidden_size = V -> dims()[3];

    auto src_dims = V -> dims();
    std::vector<int64_t> cum_src_dims = {src_dims[0], src_dims[1], src_dims[2], src_dims[3]};
    for(int i = src_dims.size() - 2; i >= 0;i --) {
        cum_src_dims[i] = cum_src_dims[i] * cum_src_dims[i + 1];
    }
    cudaMemset(grad_p_input, 0, cum_src_dims[0] * sizeof(T));

    auto attn_dims = Attn -> dims();
    std::vector<int64_t> cum_attn_dims = {attn_dims[0], attn_dims[1], attn_dims[2], attn_dims[3]};

    for(int i = attn_dims.size() - 2; i >= 0;i --) {
        cum_attn_dims[i] = cum_attn_dims[i] * cum_attn_dims[i + 1];
    }
    cudaMemset(grad_attn_input, 0, cum_attn_dims[0] * sizeof(T));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
 
    int block = 512;
    int n = batch_size * non_zeros * block_size * block_size;
    //std::cerr << " blocks  " << n << std::endl;
    int grid = (n + block - 1) / block;
    DSMMGradCUDAKernel<T><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
      col_index, row_index, head_index, p_output, p_input, attn_input, grad_p_input, grad_attn_input, hidden_size,
      seqlen, num_heads, non_zeros, batch_size, block_size);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(dense_sparse_block_matmul,
                        paddle::operators::DenseSparseBlockMatmulCUDAKernel<CUDA, float>,
                        paddle::operators::DenseSparseBlockMatmulCUDAKernel<CUDA, double>);

REGISTER_OP_CUDA_KERNEL(dense_sparse_block_matmul_grad,
                        paddle::operators::DenseSparseBlockMatmulGradCUDAKernel<CUDA, float>,
                        paddle::operators::DenseSparseBlockMatmulGradCUDAKernel<CUDA, double>);
