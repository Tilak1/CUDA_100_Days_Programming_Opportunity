Kernel Fusion
By using CUTLASS’s Epilogue Visitor Tree we’re literally fusing multiple operations (Conv → GroupNorm → Activation) into one single GPU kernel launch. That collapses three separate dispatches (convolution, normalization, nonlinear) into one, eliminating extra global-memory writes/reads and host–device launch overhead.


the key “cues” from the CUTLASS 3.x slides that we can leverage when designing our custom Conv2D→GN→Activation kernel:

Hierarchical, Tile-Based Programming Model

CUTLASS decomposes a GEMM/Conv into multiple layers—Atom (single MMA), Spatial Micro-kernel (tiled MMA + copy), Temporal Micro-kernel (collective mainloop + epilogue), Kernel (grid planning/thread marshalling), and Device (host interface).

We can mirror this by structuring our Conv2D kernel as:

A small WMMA MMA for each (Kₜile, Mₜile, Nₜile) using WMMA/PTX mma_sync

A micro-kernel that loops over K-tiles, accumulating into registers

An epilogue stage that applies GroupNorm (scale + shift) and activation via an Epilogue Visitor Tree
Speaking_Tensor_Cores_C…

Epilogue Visitor Tree (EVT) for Fusion

CUTLASS’s EVT lets you declaratively compose per-element ops (bias add, activation, normalization) into the GEMM epilogue.

We can define an EVT that takes the accumulator output, applies the GroupNorm per-row/column broadcast node, then a Mish/ReLU node, and finally writes back—eliminating separate GN and activation kernels.
Speaking_Tensor_Cores_C…

Asynchronous Data-Movement with TMA & cp.async

Hopper’s TMA and Ampere’s cp.async primitives allow bulk L2-bypassing copies from global→shared mem without stalling.

In our Conv2D, we can prefetch the next input and weight tiles into shared memory using cuda::memcpy_async (or Triton’s tl.copyp async) to overlap memory with computation.
Speaking_Tensor_Cores_C…

Tile-Scheduler & Persistent/Ping-Pong Kernels

CUTLASS supports persistent kernels (ping-pong or cooperative) to hide prologue/epilogue overhead by alternating mainloop and epilogue among warp-groups.

For larger H×W output tiles, a persistent ping-pong schedule can keep Tensor Cores busy—critical if our fused kernel has a heavier epilogue (GN + activation).
Speaking_Tensor_Cores_C…

Autotunable Shapes & Dispatch Policies

The CollectiveBuilder API auto-selects the best mainloop, stage count, and schedule for a given tile shape and architecture.

We should prototype with a small set of tile configurations (e.g. <M=128,N=128,K=32>), then sweep BLOCK_M/N/K to find the sweet spot on our SM8.x hardware.
Speaking_Tensor_Cores_C…

By adopting CUTLASS’s multi-layer structure, using an Epilogue Visitor Tree to fuse GN+activation, and leveraging asynchronous TMA/cp.async plus persistent scheduling, our custom Conv2D kernel can maximize Tensor-Core utilization and eliminate the separate layout-transform, GroupNorm, and activation launches.


C++ Skeleton code for it:

Below is a sketch of how you might declare a CUTLASS 3.x Conv2D kernel with an Epilogue Visitor Tree that fuses GroupNorm and a Mish activation right into the GEMM epilogue. This is illustrative—some of the node/functor types (e.g. GroupNormVisitor, MishFunctor) you’d implement yourself, but it follows the CUTLASS layering from the PPT Speaking_Tensor_Cores_C….

cpp
Copy
Edit
#include “cutlass/cutlass.h”
#include “cutlass/conv/device/implicit_gemm.h”
#include “cutlass/epilogue/thread/epilogue_visitor_tree.h”
#include “cutlass/epilogue/thread/bias_add.h”
#include “cutlass/epilogue/thread/activation.h”
#include “cutlass/epilogue/thread/normalization.h”

////////////////////////////////////////////////////////////////////////////////
// 1) Define your custom functors / visitor nodes
////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Accumulator>
struct GroupNormVisitor {
  // Parameters for GN (γ, β, ε, channel count, etc.) would be passed here
  CUTLASS_HOST_DEVICE
  Accumulator operator()(Element const &acc, int channel_id, int row, int col) const {
    // load mean/var for this channel, apply: (acc - μ)/√(σ²+ε)*γ + β
    // This runs per-accumulator element in the epilogue
    // …your implementation…
  }
};

template <typename Element>
struct MishFunctor {
  CUTLASS_HOST_DEVICE
  Element operator()(Element x) const {
    // Mish: x * tanh(log(1 + exp(x)))
    return x * cutlass::fast_tanh(cutlass::fast_log(cutlass::fast_exp(x) + Element(1)));
  }
};

////////////////////////////////////////////////////////////////////////////////
// 2) Compose the EpilogueVisitorTree
////////////////////////////////////////////////////////////////////////////////

using Element    = float;            // output accumulator type
using Layout     = cutlass::layout::TensorNHWC;  // or NCHW
using Epilogue   = cutlass::epilogue::thread::EpilogueVisitorTree<

  // First, optional bias-add node (if you have a conv bias)
  cutlass::epilogue::thread::BiasAddNode<Element, Layout>,

  // Then GroupNorm over channels
  GroupNormVisitor<
    Element,                    // element type
    Element                     // accumulator type
  >,

  // Finally a Mish activation
  cutlass::epilogue::thread::ActivationNode<
    Element,
    MishFunctor<Element>
  >
>;

////////////////////////////////////////////////////////////////////////////////
// 3) Instantiate the Conv2D device kernel
////////////////////////////////////////////////////////////////////////////////

using Conv2dFpropKernel = typename cutlass::conv::device::ImplicitGemmConvolution<
  // Data types
  cutlass::half_t,          // ElementA (input)
  Layout,                   // LayoutA
  cutlass::half_t,          // ElementB (weight)
  Layout,                   // LayoutB
  Element,                  // ElementC (output)
  Layout,                   // LayoutC
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,

  // Threadblock tile sizes
  cutlass::gemm::GemmShape<128, 128, 32>,  // M × N × K

  // Warp-level tile sizes
  cutlass::gemm::GemmShape<64, 64, 32>,

  // Instruction-level tile sizes
  cutlass::gemm::GemmShape<16, 8, 16>,

  // Epilogue visitor tree
  Epilogue,

  // Number of pipeline stages
  3
>::Kernel;

////////////////////////////////////////////////////////////////////////////////
// 4) Launch your kernel in host code (simplified)
////////////////////////////////////////////////////////////////////////////////
void launch_conv2d_fprop(
  cutlass::half_t const *A,  // input N×Cin×H×W
  cutlass::half_t const *B,  // weights Cout×Cin×KH×KW
  float               *C,    // output N×Cout×H_out×W_out
  int                  N, Cin, H, W,
  int                  Cout, KH, KW,
  cudaStream_t         stream
) {
  typename Conv2dFpropKernel::Arguments args{
    {N, Cin, H, W},                      // input size
    {Cout, Cin, KH, KW},                 // filter size
    {N, Cout, H - KH + 1, W - KW + 1},   // output size (no pad, unit stride example)
    A, B, C,                              // pointers
    {1, 1},                               // strides
    {0, 0},                               // padding
    {1, 1},                               // dilation
    cutlass::conv::Mode::kCrossCorrelation,
    1.0f, 0.0f                            // alpha, beta
  };

  Conv2dFpropKernel kernel_op;
  cutlass::Status status = kernel_op(args, nullptr, stream);
  assert(status == cutlass::Status::kSuccess);
}
Notes:

We’ve built an EpilogueVisitorTree that first applies a bias add (if you want it), then your GroupNormVisitor, then the MishFunctor.

The ImplicitGemmConvolution template hooks that epilogue into the tiled MMA mainloop for SM 8.x Tensor Cores.

You’ll need to implement the details of GroupNormVisitor (loading per-channel stats, computing mean/variance, etc.) and tune your GemmShape<> (BLOCK_M/N/K) for your heatmap sizes.

This pattern directly mirrors CUTLASS’s layered design—Atom → Microkernel → Kernel → Device—and uses the Epilogue Visitor Tree to fuse normalization + activation into one launch Speaking_Tensor_Cores_C….

Ref: 

* PDF - Speaking_Tensor_Cores_CUTLASS_2024 (attached in repo)


