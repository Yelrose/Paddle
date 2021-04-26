#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"
#include "unordered_set"
#include <iostream>

namespace paddle {
namespace operators {

class DenseSparseBlockMatmulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Attn", "The Attention matrix with shape [B, *, block_size, block_size]");
    AddInput("BlockRow", "BlockRow.");
    AddInput("BlockCol", "BlockCol.");
    AddInput("BlockHead", "BlockHead.");
    AddInput("V", "The V tensor with shape [B, H, L, D]");
    AddOutput("Out", "Output of shape [B, H, L, D]");
    AddAttr<int>(
        "block_size",
        R"DOC((int, default 16), 
               block_size
        )DOC")
        .SetDefault(16)
        .EqualGreaterThan(1);
    AddComment(R"DOC(
    V = sparse_mamtul(attn, V)
)DOC");
  }
};

class DenseSparseBlockMatmulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Attn"), "Shape ");
    PADDLE_ENFORCE(ctx->HasInput("BlockRow"), "BlockRow");
    PADDLE_ENFORCE(ctx->HasInput("BlockCol"), "BlockCol");
    PADDLE_ENFORCE(ctx->HasInput("BlockHead"), "BlockHead");
    PADDLE_ENFORCE(ctx->HasInput("V"), "The V tensor with shape [B, H, L, D]");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Shape ");
    auto in_dims = ctx->GetInputDim("V");
    ctx->SetOutputDim("Out", in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "V"),
        ctx.device_context());
  }

};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class DenseSparseBlockMatmulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Take Col mul Y to Row 
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

    // Compute Out Shape
    auto src_dims = V -> dims();
    size_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);

    memset(p_output, 0, memset_bytes);

    const int* col_index = BlockCol -> data<int>();
    const int* row_index = BlockRow -> data<int>();
    const int* head_index = BlockHead -> data<int>();
    size_t non_zeros = BlockCol -> dims()[0];
    size_t batch_size = V -> dims()[0];
    size_t hidden_size = V -> dims()[3];

    std::vector<int64_t> cum_src_dims = {src_dims[0], src_dims[1], src_dims[2], src_dims[3]};
    for(int i = src_dims.size() - 2; i >= 0;i --) {
        cum_src_dims[i] = cum_src_dims[i] * cum_src_dims[i + 1];
    }

    auto attn_dims = Attn -> dims();
    std::vector<int64_t> cum_attn_dims = {attn_dims[0], attn_dims[1], attn_dims[2], attn_dims[3]};

    for(int i = attn_dims.size() - 2; i >= 0;i --) {
        cum_attn_dims[i] = cum_attn_dims[i] * cum_attn_dims[i + 1];
    }
     
    for (size_t i = 0; i <  non_zeros; i ++) {
        // Take V[B, head_id, col_index:col_index+block_size, D]
        // Mul  Attn[B, i, block_size, block_size]
        // Set V[B, head_id, row_index: rol_index +block_size, D]
        int head_id = head_index[i]; 
        for(size_t b = 0; b < batch_size; b ++ ) {
            for(int block_i = 0; block_i < block_size; block_i ++ ){
                for(size_t d = 0; d < hidden_size; d ++) {
                    int64_t scatter_index = b * cum_src_dims[1] + head_id * cum_src_dims[2] + (row_index[i] * block_size + block_i) * cum_src_dims[3] + d; 
                    T sum = 0;
                    for(int block_j = 0; block_j < block_size; block_j ++ ) {
                    // Out[b, head_id, row_index[i] * block_size + block_i, D] =  Attn[b, i, block_i, block_j] * V[b, head_id, col_index[i] * block_size + block_j, D]
                        int64_t gather_index = b * cum_src_dims[1] + head_id * cum_src_dims[2] + (col_index[i] * block_size + block_j) * cum_src_dims[3] + d; 
                        int64_t attn_index = b * cum_attn_dims[1] + i * cum_attn_dims[2] + block_i * cum_attn_dims[3] + block_j; 
                        //std::cout << " attn_input " << " i " << block_i << " " << block_j << " " << attn_input[attn_index] << std::endl; 
                        sum += attn_input[attn_index] * p_input[gather_index];
                    }
                    p_output[scatter_index] += sum;
                }
            }
        }
    }

  }
};

// 定义反向OP的输入Y和dY、输出dX、属性:
template <typename T>
class DenseSparseBlockMatmulGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("dense_sparse_block_matmul_grad");
    op->SetInput("Attn", this->Input("Attn"));
    op->SetInput("BlockRow", this->Input("BlockRow"));
    op->SetInput("BlockCol", this->Input("BlockCol"));
    op->SetInput("BlockHead", this->Input("BlockHead"));
    op->SetInput("V", this->Input("V"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("Attn"), this->InputGrad("Attn"));
    op->SetOutput(framework::GradVarName("V"), this->InputGrad("V"));
  }
};

// 定义反向OP和InferShape实现,设置dX的shape
class DenseSparseBlockMatmulGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("V");
    auto attn_dims = ctx->GetInputDim("Attn");

    ctx->SetOutputDim(framework::GradVarName("V"), in_dims);
    ctx->SetOutputDim(framework::GradVarName("Attn"), attn_dims);
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
class DenseSparseBlockMatmulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Init Input
    
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


    const int* col_index = BlockCol->data<int>();
    const int* row_index = BlockRow->data<int>();
    const int* head_index = BlockHead->data<int>();
    size_t non_zeros = BlockCol -> dims()[0];
    size_t batch_size = V->dims()[0];
    size_t hidden_size = V->dims()[3];

    auto src_dims = V -> dims();
    std::vector<int64_t> cum_src_dims = {src_dims[0], src_dims[1], src_dims[2], src_dims[3]};
    for(int i = src_dims.size() - 2; i >= 0;i --) {
        cum_src_dims[i] = cum_src_dims[i] * cum_src_dims[i + 1];
    }
    memset(grad_p_input, 0, cum_src_dims[0] * sizeof(T));

    auto attn_dims = Attn -> dims();
    std::vector<int64_t> cum_attn_dims = {attn_dims[0], attn_dims[1], attn_dims[2], attn_dims[3]};

    for(int i = attn_dims.size() - 2; i >= 0;i --) {
        cum_attn_dims[i] = cum_attn_dims[i] * cum_attn_dims[i + 1];
    }
    memset(grad_attn_input, 0, cum_attn_dims[0] * sizeof(T));
     

    for (size_t i = 0; i <  non_zeros; i ++) {
        // Take V[B, head_id, col_index:col_index+block_size, D]
        // Mul  Attn[B, i, block_size, block_size]
        // Set V[B, head_id, row_index: rol_index +block_size, D]
        int head_id = head_index[i]; 
        for(size_t b = 0; b < batch_size; b ++ ) {
            for(int block_i = 0; block_i < block_size; block_i ++ ){
                for(size_t d = 0; d < hidden_size; d ++) {
                    for(int block_j = 0; block_j < block_size; block_j ++ ) {
                    // Out[b, head_id, row_index[i] * block_size + block_i, D] =  Attn[b, i, block_i, block_j] * V[b, head_id, col_index[i] * block_size + block_j, D]
                        int64_t scatter_index = b * cum_src_dims[1] + head_id * cum_src_dims[2] + (row_index[i] * block_size + block_i) * cum_src_dims[3] + d; 
                        int64_t gather_index = b * cum_src_dims[1] + head_id * cum_src_dims[2] + (col_index[i] * block_size + block_j) * cum_src_dims[3] + d; 
                        int64_t attn_index = b * cum_attn_dims[1] + i * cum_attn_dims[2] + block_i * cum_attn_dims[3] + block_j; 
                        //std::cout << " attn_input " << " i " << block_i << " " << block_j << " " << attn_input[attn_index] << std::endl; 
                        // p_output[scatter_index] += attn_input[attn_index] * p_input[gather_index];
                        grad_attn_input[attn_index] += p_input[gather_index] * p_output[scatter_index];
                        grad_p_input[gather_index] += attn_input[attn_index] * p_output[scatter_index];
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

REGISTER_OPERATOR(dense_sparse_block_matmul, ops::DenseSparseBlockMatmulOp,
                  ops::DenseSparseBlockMatmulOpMaker,
                  ops::DenseSparseBlockMatmulGradMaker<paddle::framework::OpDesc>,
                  ops::DenseSparseBlockMatmulGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(dense_sparse_block_matmul_grad, ops::DenseSparseBlockMatmulGradOp);

REGISTER_OP_CPU_KERNEL(dense_sparse_block_matmul,
                       ops::DenseSparseBlockMatmulKernel<CPU, float>,
                       ops::DenseSparseBlockMatmulKernel<CPU, double>);

REGISTER_OP_CPU_KERNEL(dense_sparse_block_matmul_grad,
                       ops::DenseSparseBlockMatmulGradKernel<CPU, float>,
                       ops::DenseSparseBlockMatmulGradKernel<CPU, double>);
