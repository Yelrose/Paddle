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

class SparseBlockMatmulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "The Q tensor with shape [B, H, L, D]");
    AddInput("K", "The K tensor with shape [B, H, L, D]");
    AddInput("BlockRow", "BlockRow.");
    AddInput("BlockCol", "BlockCol.");
    AddInput("BlockHead", "BlockHead.");
    AddAttr<int>(
        "block_size",
        R"DOC((int, default 16), 
               block_size
        )DOC")
        .SetDefault(16)
        .EqualGreaterThan(1);
    AddOutput("Out", "Output of sparse_block_matmul [B, *, block_size, block_size]");
    AddComment(R"DOC(
Relu Operator.
      sparse(Q).matmul(sparse(K.T))
)DOC");
  }
};

class SparseBlockMatmulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Q"), "Input(Q) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("K"), "Input(K) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("BlockRow"), "Input(BlockRow) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("BlockCol"), "Input(BlockCol) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("BlockHead"), "Input(BlockHead) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Y) should be not null.");
    int block_size = ctx->Attrs().Get<int>("block_size");
    std::vector<int64_t> output_dims;

    int batch_size = ctx->GetInputDim("Q")[0];
    int non_zeros = ctx->GetInputDim("BlockRow")[0];
    output_dims.push_back(batch_size);
    output_dims.push_back(non_zeros);
    output_dims.push_back(block_size);
    output_dims.push_back(block_size);
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
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
class SparseBlockMatmulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

    auto* Q = ctx.Input<Tensor>("Q");
    auto* K = ctx.Input<Tensor>("K");
    auto* BlockRow = ctx.Input<Tensor>("BlockRow");
    auto* BlockCol = ctx.Input<Tensor>("BlockCol");
    auto* BlockHead = ctx.Input<Tensor>("BlockHead");

    auto* Out = ctx.Output<Tensor>("Out");
    int block_size = ctx.Attr<int>("block_size");

    // Initialize out
    T* p_output = Out->mutable_data<T>(ctx.GetPlace());
    // Out size [Batch_size, len(BlockCol), block_size, block_size]

    // Compute Y Shape
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

    memset(p_output, 0, memset_bytes);
    // foolish implement
    for(size_t index = 0; index < non_zeros; index ++) {
        for(int i = 0; i < block_size; i ++) {
            for(int j = 0; j < block_size;j ++) {
                for(size_t b = 0; b < batch_size; b ++) {
                    for(size_t d = 0; d < dim_size; d ++) {
                        int out_index = (b * non_zeros + index) * block_size * block_size  + i * block_size + j;
                        p_output[out_index] += q_data[b * head_size* seqlen * dim_size + block_head[index]*seqlen*dim_size+ (block_row[index] * block_size + i) * dim_size +d] * k_data[b * head_size* seqlen * dim_size + block_head[index]*seqlen*dim_size+ (block_col[index] * block_size + j) * dim_size +d];
                    }
                }
            }
        }
    }

  }
};

// 定义反向OP的输入Y和dY、输出dX、属性:
template <typename T>
class SparseBlockMatmulGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sparse_block_matmul_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("BlockRow", this->Input("BlockRow"));
    op->SetInput("BlockCol", this->Input("BlockCol"));
    op->SetInput("BlockHead", this->Input("BlockHead"));
    op->SetInput("Q", this->Input("Q"));
    op->SetInput("K", this->Input("K"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("Q"), this->InputGrad("Q"));
    op->SetOutput(framework::GradVarName("K"), this->InputGrad("K"));
  }
};

// 定义反向OP和InferShape实现,设置dX的shape
class SparseBlockMatmulGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto q_dims = ctx->GetInputDim(this->Input("Q"));
    auto k_dims = ctx->GetInputDim(this->Input("K"));
    ctx->SetOutputDim(framework::GradVarName("Q"), q_dims);
    ctx->SetOutputDim(framework::GradVarName("K"), k_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }

};

template <typename DeviceContext, typename T>
class SparseBlockMatmulGradKernel : public framework::OpKernel<T> {
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
    memset(q_output, 0, memset_bytes);
    memset(k_output, 0, memset_bytes);

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

    // foolish implement
    for(size_t index = 0; index < non_zeros; index ++) {
        for(int i = 0; i < block_size; i ++) {
            for(int j = 0; j < block_size;j ++) {
                for(size_t b = 0; b < batch_size; b ++) {
                    for(size_t d = 0; d < dim_size; d ++) {
                        int out_index = (b * non_zeros + index) * block_size * block_size  + i * block_size + j;
                        k_output[b * head_size* seqlen * dim_size + block_head[index]*seqlen*dim_size+ (block_col[index] * block_size + j) * dim_size +d] += out_grad[out_index] * q_data[b * head_size* seqlen * dim_size + block_head[index]*seqlen*dim_size+ (block_row[index] * block_size + i) * dim_size +d];
                        q_output[b * head_size* seqlen * dim_size + block_head[index]*seqlen*dim_size+ (block_row[index] * block_size + i) * dim_size +d] += out_grad[out_index] * k_data[b * head_size* seqlen * dim_size + block_head[index]*seqlen*dim_size+ (block_col[index] * block_size + j) * dim_size +d];
                    }
                }
            }
        }
    }

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(sparse_block_matmul,
                  ops::SparseBlockMatmulOp,
                  ops::SparseBlockMatmulOpMaker,
                  ops::SparseBlockMatmulGradMaker<paddle::framework::OpDesc>,
                  ops::SparseBlockMatmulGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(sparse_block_matmul_grad, ops::SparseBlockMatmulGradOp);

REGISTER_OP_CPU_KERNEL(sparse_block_matmul,
                       ops::SparseBlockMatmulKernel<CPU, float>,
                       ops::SparseBlockMatmulKernel<CPU, double>);

REGISTER_OP_CPU_KERNEL(sparse_block_matmul_grad,
                       ops::SparseBlockMatmulGradKernel<CPU, float>,
                       ops::SparseBlockMatmulGradKernel<CPU, double>);
