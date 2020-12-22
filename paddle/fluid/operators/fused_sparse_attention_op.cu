
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
__global__ void SparseDotProductCUDAKernel(const T* q, const T* k,
                                  const IndexT* q_index_data, const IndexT* k_index_data,
                                  size_t num_heads, size_t hidden_size,
                                  T* out, size_t index_size) {
  CUDA_KERNEL_LOOP(i, index_size * num_heads) {
    int indices_i = i / num_heads;
    size_t h = i - indices_i * num_heads;
    IndexT q_index = *(q_index_data + indices_i);
    IndexT k_index = *(k_index_data + indices_i);
    size_t qi = (q_index * num_heads + h) * hidden_size;
    size_t ki = (k_index * num_heads + h) * hidden_size;
    for(size_t j = 0; j < hidden_size;j ++ ) {
       *(out + indices_i * num_heads + h) += (*(q + qi + j)) *  (*(k + ki + j));
    }
  }
}

template <typename DeviceContext, typename T, typename IndexT = int>
void SparseDotProduct(const framework::ExecutionContext& context, const T* q_data, const T* k_data,
                     const Tensor * Q_index, const Tensor * K_index,
                                  size_t num_heads, size_t hidden_size,
                                  T* p_output, size_t index_size) {

    int block = 512;
    int n = index_size * num_heads;
    int grid = (n + block - 1) / block;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    const IndexT * q_index = Q_index -> data<IndexT >();
    const IndexT * k_index = K_index -> data<IndexT >();

    SparseDotProductCUDAKernel<T, IndexT><<<
        grid, block, 0,
        reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
        q_data, k_data, q_index, k_index, num_heads, hidden_size, p_output, index_size);
}

template <typename T, typename IndexT = int>
__global__ void SparseDotProductGradCUDAKernel(const T* q, T* k,
                                  const IndexT* q_index_data, const IndexT* k_index_data,
                                  size_t num_heads, size_t hidden_size,
                                  const T* out, size_t index_size) {
  CUDA_KERNEL_LOOP(i, index_size * num_heads) {
    int indices_i = i / num_heads;
    size_t h = i - indices_i * num_heads;
    IndexT q_index = *(q_index_data + indices_i);
    IndexT k_index = *(k_index_data + indices_i);
    size_t qi = (q_index * num_heads + h) * hidden_size;
    size_t ki = (k_index * num_heads + h) * hidden_size;
    for(size_t j = 0; j < hidden_size;j ++ ) {
        T value =  (*(out + indices_i * num_heads + h)) *  (*(q + qi + j));
        paddle::platform::CudaAtomicAdd(k + ki +j, value);
    }
  }
}


template <typename DeviceContext, typename T, typename IndexT = int>
void SparseDotProductGrad(const framework::ExecutionContext& context, const T* Q_data, T * grad_K_data,
                     const Tensor * Q_index, const Tensor * K_index,
                                  size_t num_heads, size_t hidden_size,
                                  const T* grad_Y_data, size_t index_size) {

    int block = 512;
    int n = index_size * num_heads;
    int grid = (n + block - 1) / block;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    const IndexT * q_index = Q_index -> data<IndexT >();
    const IndexT * k_index = K_index -> data<IndexT >();

    SparseDotProductGradCUDAKernel<T, IndexT><<<
        grid, block, 0,
        reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
        Q_data, grad_K_data, q_index, k_index, num_heads, hidden_size, grad_Y_data, index_size);
}




// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class FusedSparseAttentionCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* Q = ctx.Input<Tensor>("Q");
    auto* Q_index = ctx.Input<Tensor>("Q_index");
    auto* K = ctx.Input<Tensor>("K");
    auto* K_index = ctx.Input<Tensor>("K_index");
    int index_size = Q_index -> dims()[0];

    auto* Y = ctx.Output<Tensor>("Y");
    T* p_output = Y -> mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // Compute Y Shape

    auto src_dims = Q -> dims();
    size_t memset_size = index_size * src_dims[1];
    const size_t& memset_bytes = memset_size * sizeof(T);
    cudaMemset(p_output, 0, memset_bytes);

    // Initialize Y Shape
    auto * q_data = Q -> data<T >();
    auto * k_data = K -> data<T >();


    const auto &index_type = Q_index -> type();
    if (index_type == framework::proto::VarType::INT32) {
        SparseDotProduct<DeviceContext, T, int32_t>(ctx, q_data, k_data, Q_index, K_index, src_dims[1], src_dims[2], p_output, index_size);
    } else {
        SparseDotProduct<DeviceContext, T, int64_t>(ctx, q_data, k_data, Q_index, K_index, src_dims[1], src_dims[2], p_output, index_size);
    }
    

  }
};


// 反向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class FusedSparseAttentionGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* Q = ctx.Input<Tensor>("Q");
    auto* K = ctx.Input<Tensor>("K");

    auto* Q_index = ctx.Input<Tensor>("Q_index");
    auto* K_index = ctx.Input<Tensor>("K_index");

    auto* grad_Y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* grad_Q = ctx.Output<Tensor>(framework::GradVarName("Q"));
    auto* grad_K = ctx.Output<Tensor>(framework::GradVarName("K"));

    auto src_dims = Q -> dims();

    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) {
        memset_size *= src_dims[i];
    }
    const size_t& memset_bytes = memset_size * sizeof(T);

    // grad data
    T * grad_Q_data = grad_Q -> mutable_data<T>(ctx.GetPlace());
    T * grad_K_data = grad_K -> mutable_data<T>(ctx.GetPlace());

    const T * Q_data = Q -> data<T>();
    const T * K_data = K -> data<T>();
    const T * grad_Y_data = grad_Y -> data<T>();


    cudaMemset(grad_Q_data, 0, memset_bytes);
    cudaMemset(grad_K_data, 0, memset_bytes);

    int index_size = Q_index -> dims()[0];

    const auto &index_type = Q_index -> type();
    
    if (index_type == framework::proto::VarType::INT32) {
        SparseDotProductGrad<DeviceContext, T, int32_t>(ctx,
        Q_data, grad_K_data, Q_index, K_index, src_dims[1], src_dims[2], grad_Y_data, index_size);

        SparseDotProductGrad<DeviceContext, T, int32_t>(ctx,
        K_data, grad_Q_data, K_index, Q_index, src_dims[1], src_dims[2], grad_Y_data, index_size);
    } else {
        SparseDotProductGrad<DeviceContext, T, int64_t>(ctx,
        Q_data, grad_K_data, Q_index, K_index, src_dims[1], src_dims[2], grad_Y_data, index_size);

        SparseDotProductGrad<DeviceContext, T, int64_t>(ctx,
        K_data, grad_Q_data, K_index, Q_index, src_dims[1], src_dims[2], grad_Y_data, index_size);

    }
 
    


  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(fused_sparse_attention,
                        paddle::operators::FusedSparseAttentionCUDAKernel<CUDA, float>,
                        paddle::operators::FusedSparseAttentionCUDAKernel<CUDA, double>);

REGISTER_OP_CUDA_KERNEL(fused_sparse_attention_grad,
                        paddle::operators::FusedSparseAttentionGradCUDAKernel<CUDA, float>,
                        paddle::operators::FusedSparseAttentionGradCUDAKernel<CUDA, double>);
