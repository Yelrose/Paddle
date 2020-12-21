
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
__global__ void GatherScatterCUDAKernel(const T* params, const IndexT* gather_indices, const IndexT*  scatter_indices,
                                  T* output, size_t index_size,
                                  size_t slice_size) {
  CUDA_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = gather_indices[indices_i];
    IndexT scatter_i = scatter_indices[indices_i];
    IndexT int_i = gather_i * slice_size + slice_i;

    IndexT out_i = scatter_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(output + out_i, *(params + int_i));
  }
}


// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class FusedGatherScatterCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

    auto* X = ctx.Input<Tensor>("X");
    auto* Gather_index = ctx.Input<Tensor>("Gather_index");
    auto* Scatter_index = ctx.Input<Tensor>("Scatter_index");
    auto* Y = ctx.Output<Tensor>("Y");

    int index_size = Gather_index -> dims()[0];
    T* p_output = Y -> mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto src_dims = X -> dims();

    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);

    cudaMemset(p_output, 0, memset_bytes);

    size_t slice_size = 1;
    for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];
    const size_t& slice_bytes = slice_size * sizeof(T);



    const T* p_src = X -> data<T>(); 
    const int * g_index = Gather_index -> data<int>();
    const int * s_index = Scatter_index -> data<int>();

    int block = 512;
    int n = slice_size * index_size;
    int grid = (n + block - 1) / block;
    GatherScatterCUDAKernel<T, int><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
      p_src, g_index, s_index, p_output, index_size, slice_size);
    

  }
};


// 反向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class FusedGatherScatterGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* Gather_index = ctx.Input<Tensor>("Scatter_index");
    auto* Scatter_index = ctx.Input<Tensor>("Gather_index");
    auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));



    int index_size = Gather_index -> dims()[0];
    T* p_output = Y -> mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto src_dims = X -> dims();

    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);

    cudaMemset(p_output, 0, memset_bytes);

    size_t slice_size = 1;
    for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];
    const size_t& slice_bytes = slice_size * sizeof(T);



    const T* p_src = X -> data<T>(); 
    const int * g_index = Gather_index -> data<int>();
    const int * s_index = Scatter_index -> data<int>();

    int block = 512;
    int n = slice_size * index_size;
    int grid = (n + block - 1) / block;
    GatherScatterCUDAKernel<T, int><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx).stream()>>>(
      p_src, g_index, s_index, p_output, index_size, slice_size);
    


  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(fused_gather_scatter,
                        paddle::operators::FusedGatherScatterCUDAKernel<CUDA, float>,
                        paddle::operators::FusedGatherScatterCUDAKernel<CUDA, double>);

REGISTER_OP_CUDA_KERNEL(fused_gather_scatter_grad,
                        paddle::operators::FusedGatherScatterGradCUDAKernel<CUDA, float>,
                        paddle::operators::FusedGatherScatterGradCUDAKernel<CUDA, double>);
