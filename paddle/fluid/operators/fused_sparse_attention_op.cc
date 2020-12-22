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

class FusedSparseAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "The input q tensor.");
    AddInput("K", "The input k tensor.");
    AddInput("Q_index", "The q index tensor.");
    AddInput("K_index", "The k index tensor.");
    AddOutput("Y", "Output of fused_sparse_attention");
    AddComment(R"DOC(
Relu Operator.
Y = scatter(gather(X, gather_index), scatter_index)
)DOC");
  }
};

class FusedSparseAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Q"), "Input(Q) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Q_index"), "Input(Q_index) should be not null.");

    PADDLE_ENFORCE(ctx->HasInput("K"), "Input(K) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("K_index"), "Input(K_index) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto in_dims = ctx->GetInputDim("Q");
    auto index_dims = ctx->GetInputDim("Q_index");

    auto dims_vector = vectorize(in_dims);
    const int kDelFlag = -2;
    dims_vector[2] = kDelFlag;
    dims_vector[0] = index_dims[0];
    dims_vector.erase(
        remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
        dims_vector.end());
    auto out_dims = framework::make_ddim(dims_vector);
    ctx->SetOutputDim("Y", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Q"),
        ctx.device_context());
  }

};

using Tensor = framework::Tensor;

template <typename T, typename IndexT = int>

void elementwise_inner_dot_product(const framework::ExecutionContext& ctx,
                      const T * q,
                      const T * k,
                      const int& q_index,
                      const int& k_index,
                      const size_t num_heads,
                      const size_t hidden_size,
                      T* out, 
                      const int& out_index) {

  for(size_t i = 0; i < num_heads; i ++) {
      size_t qi = (q_index * num_heads + i) * hidden_size;
      size_t ki = (k_index * num_heads + i) * hidden_size;
      for(size_t j = 0; j < hidden_size;j ++ ) {
          *(out + out_index * num_heads + i) += (*(q + qi + j)) *  (*(k + ki + j));
      }
  }
}

template <typename T, typename IndexT = int>
void all_elementwise_inner_dot_product(const framework::ExecutionContext& ctx,
                      const T * q_data,
                      const T * k_data,
                      const Tensor *  Q_index,
                      const Tensor *  K_index,
                      const size_t num_heads,
                      const size_t hidden_size,
                      T* p_output, 
                      const int& index_size) {

    const IndexT * q_index = Q_index -> data<IndexT>();
    const IndexT * k_index = K_index -> data<IndexT>();
    for (int i= 0; i < index_size; i ++) {
        int src_ptr = q_index[i]; 
        int dst_ptr = k_index[i]; 
        elementwise_inner_dot_product<T, IndexT>(ctx, q_data, k_data, 
           src_ptr, dst_ptr, num_heads, hidden_size, p_output, i);
    }
}




template <typename T, typename IndexT = int>
void elementwise_inner_dot_product_grad(const framework::ExecutionContext& ctx,
                      const T * q,
                      T * k,
                      const IndexT & q_index,
                      const IndexT & k_index,
                      const size_t num_heads,
                      const size_t hidden_size,
                      const T * out, 
                      const int& out_index) {
  for(size_t i = 0; i < num_heads; i ++) {
      size_t qi = (q_index * num_heads + i) * hidden_size;
      size_t ki = (k_index * num_heads + i) * hidden_size;
      for(size_t j = 0; j < hidden_size;j ++ ) {
          *(k + ki + j) += (*(out + out_index * num_heads + i)) *  (*(q + qi + j));
      }
  }
}

template <typename T, typename IndexT = int>
void all_elementwise_inner_dot_product_grad(const framework::ExecutionContext& ctx,
                      const T * Q_data,
                      T * grad_K_data,
                      const Tensor * Q_index,
                      const Tensor * K_index,
                      const size_t num_heads,
                      const size_t hidden_size,
                      const T * grad_Y_data, 
                      const int& index_size) {
    const int* q_index = Q_index -> data<int>();
    const int* k_index = K_index -> data<int>();
    for (int i= 0; i < index_size; i ++) {
        int src_ptr = q_index[i]; 
        int dst_ptr = k_index[i]; 
        elementwise_inner_dot_product_grad<T, IndexT>(ctx, Q_data, grad_K_data,
                 src_ptr, dst_ptr, num_heads, hidden_size, grad_Y_data, i);
    }
}



template <typename DeviceContext, typename T>
class FusedSparseAttentionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

    auto* Q = ctx.Input<Tensor>("Q");
    auto* Q_index = ctx.Input<Tensor>("Q_index");
    auto* K = ctx.Input<Tensor>("K");
    auto* K_index = ctx.Input<Tensor>("K_index");
    int index_size = Q_index -> dims()[0];

    auto* Y = ctx.Output<Tensor>("Y");
    T* p_output = Y -> mutable_data<T>(ctx.GetPlace());

    // Compute Y Shape

    auto src_dims = Q -> dims();
    size_t memset_size = index_size * src_dims[1];
    const size_t& memset_bytes = memset_size * sizeof(T);

    memset(p_output, 0, memset_bytes);

    // Initialize Y Shape
    auto * q_data = Q -> data<T >();
    auto * k_data = K -> data<T >();

    const auto &index_type = Q_index -> type();
    if (index_type == framework::proto::VarType::INT32) {
        all_elementwise_inner_dot_product<T, int32_t>(ctx, q_data, k_data, 
            Q_index, K_index, src_dims[1], src_dims[2], p_output, index_size);
    } else {
        all_elementwise_inner_dot_product<T, int64_t>(ctx, q_data, k_data, 
            Q_index, K_index, src_dims[1], src_dims[2], p_output, index_size);
    }

  }
};

// 定义反向OP的输入Y和dY、输出dX、属性:
template <typename T>
class FusedSparseAttentionGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_sparse_attention_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput("Q_index", this->Input("Q_index"));
    op->SetInput("K_index", this->Input("K_index"));
    op->SetInput("Q", this->Input("Q"));
    op->SetInput("K", this->Input("K"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("Q"), this->InputGrad("Q"));
    op->SetOutput(framework::GradVarName("K"), this->InputGrad("K"));
  }
};

// 定义反向OP和InferShape实现,设置dX的shape
class FusedSparseAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("Q");
    ctx->SetOutputDim(framework::GradVarName("Q"), in_dims);
    ctx->SetOutputDim(framework::GradVarName("K"), in_dims);
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
class FusedSparseAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

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

    memset(grad_Q_data, 0, memset_bytes);
    memset(grad_K_data, 0, memset_bytes);
    int index_size = Q_index -> dims()[0];

    const auto &index_type = Q_index -> type();

    if (index_type == framework::proto::VarType::INT32) {
        all_elementwise_inner_dot_product_grad<T, int32_t>(ctx, Q_data, grad_K_data, Q_index, K_index,
                         src_dims[1], src_dims[2], grad_Y_data, index_size);
        all_elementwise_inner_dot_product_grad<T, int32_t>(ctx, K_data, grad_Q_data, K_index, Q_index,
                         src_dims[1], src_dims[2], grad_Y_data, index_size);
    }
    else {
        all_elementwise_inner_dot_product_grad<T, int64_t>(ctx, Q_data, grad_K_data, Q_index, K_index,
                         src_dims[1], src_dims[2], grad_Y_data, index_size);
        all_elementwise_inner_dot_product_grad<T, int64_t>(ctx, K_data, grad_Q_data, K_index, Q_index,
                         src_dims[1], src_dims[2], grad_Y_data, index_size);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(fused_sparse_attention,
                  ops::FusedSparseAttentionOp,
                  ops::FusedSparseAttentionOpMaker,
                  ops::FusedSparseAttentionGradMaker<paddle::framework::OpDesc>,
                  ops::FusedSparseAttentionGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(fused_sparse_attention_grad, ops::FusedSparseAttentionGradOp);

REGISTER_OP_CPU_KERNEL(fused_sparse_attention,
                       ops::FusedSparseAttentionKernel<CPU, float>,
                       ops::FusedSparseAttentionKernel<CPU, double>);

REGISTER_OP_CPU_KERNEL(fused_sparse_attention_grad,
                       ops::FusedSparseAttentionGradKernel<CPU, float>,
                       ops::FusedSparseAttentionGradKernel<CPU, double>);
