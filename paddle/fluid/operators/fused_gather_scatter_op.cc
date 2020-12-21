#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"
#include "unordered_set"

namespace paddle {
namespace operators {

class FusedGatherScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Gather_index", "The gather index tensor.");
    AddInput("Scatter_index", "The scatter index tensor.");
    AddOutput("Y", "Output of fused_gather_scatter");
    AddComment(R"DOC(
Relu Operator.
Y = scatter(gather(X, gather_index), scatter_index)
)DOC");
  }
};

class FusedGatherScatterOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Gather_index"), "Input(gather_index) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Scatter_index"), "Input(scatter_index) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Y", in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

};

using Tensor = framework::Tensor;

template <typename T, typename IndexT = int>
void elementwise_inner_add(const framework::ExecutionContext& ctx,
                      const Tensor& src,
                      Tensor* dist, const int& src_index,
                      const IndexT& dist_index) {
  auto src_slice = src.Slice(src_index, src_index + 1);
  auto dist_slice = dist->Slice(dist_index, dist_index + 1);

  auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
  auto eigen_dist = framework::EigenVector<T>::Flatten(dist_slice);

  eigen_dist += eigen_src;
}


template <typename DeviceContext, typename T>
class FusedGatherScatterKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

    auto* X = ctx.Input<Tensor>("X");
    auto* Gather_index = ctx.Input<Tensor>("Gather_index");
    auto* Scatter_index = ctx.Input<Tensor>("Scatter_index");
    auto* Y = ctx.Output<Tensor>("Y");
    int index_size = Gather_index -> dims()[0];
    T* p_output = Y -> mutable_data<T>(ctx.GetPlace());

    // Compute Y Shape
    auto src_dims = X -> dims();
    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);

    memset(p_output, 0, memset_bytes);

    // Initialize Y Shape
    const int* gather_index = Gather_index -> data<int>();
    const int* scatter_index = Scatter_index -> data<int>();

    for (int i= 0; i < index_size; i ++) {
        int src_ptr = gather_index[i]; 
        int dst_ptr = scatter_index[i]; 
        elementwise_inner_add<T, int>(ctx, *X, Y, src_ptr, dst_ptr);
    }

  }
};

// 定义反向OP的输入Y和dY、输出dX、属性:
template <typename T>
class FusedGatherScatterGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_gather_scatter_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput("Gather_index", this->Input("Gather_index"));
    op->SetInput("Scatter_index", this->Input("Scatter_index"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

// 定义反向OP和InferShape实现,设置dX的shape
class FusedGatherScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }

};

template <typename DeviceContext, typename T>
class FusedGatherScatterGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* Gather_index = ctx.Input<Tensor>("Scatter_index");
    auto* Scatter_index = ctx.Input<Tensor>("Gather_index");
    auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));
    int index_size = Gather_index -> dims()[0];
    T* p_output = Y -> mutable_data<T>(ctx.GetPlace());

    // Compute Y Shape
    auto src_dims = X -> dims();
    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);

    memset(p_output, 0, memset_bytes);

    // Initialize Y Shape
    const int* gather_index = Gather_index -> data<int>();
    const int* scatter_index = Scatter_index -> data<int>();

    for (int i= 0; i < index_size; i ++) {
        int src_ptr = gather_index[i]; 
        int dst_ptr = scatter_index[i]; 
        elementwise_inner_add<T, int>(ctx, *X, Y, src_ptr, dst_ptr);
    }


  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(fused_gather_scatter,
                  ops::FusedGatherScatterOp,
                  ops::FusedGatherScatterOpMaker,
                  ops::FusedGatherScatterGradMaker<paddle::framework::OpDesc>,
                  ops::FusedGatherScatterGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(fused_gather_scatter_grad, ops::FusedGatherScatterGradOp);

REGISTER_OP_CPU_KERNEL(fused_gather_scatter,
                       ops::FusedGatherScatterKernel<CPU, float>);
                       //ops::FusedGatherScatterKernel<CPU, double>);

REGISTER_OP_CPU_KERNEL(fused_gather_scatter_grad,
                       ops::FusedGatherScatterGradKernel<CPU, float>);
                       //ops::FusedGatherScatterGradKernel<CPU, double>);
