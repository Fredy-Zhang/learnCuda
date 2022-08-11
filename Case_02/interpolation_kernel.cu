#include <torch/extension.h>

// fw: forward: put input to get the output.
// bw: backward: output to update the parameters.
// cu: cuda
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor point
){
    // generate the output tensor
    const int N = feats.size[0], F = feat.size(2);

    // should put into same cuda device.
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());
    
    // set the data type of variable, and put this variable into same device
    //  torch::zeeros({N, F}, torch::dtype(torch::kInt32).device(feats.device));
    
    // every thread operates one point calculates. 

    // At first, defining the number of Blocks in Grids, and need how many threads.
    // 1. the N points can parallel computes;
    // 2. the F features can parallel computes;

    // two parts can be paralleled.
    // taking the two threads. 16*16 = 256
    const dim3 threads(16, 16); // max thread 256
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    // throw to kernel.
    // AT_DISPATCH_FLOATING_TYPES_HALF 16bits calculated

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_cu<scalar_t><<<blocks, threads>>>(
        feats.packed_accessor<scalar_t, 3, torch::RestrictTraits, size_t>(),
        points.packed_accessor<scalar_t, 2, torch::RestrictTraits, size_t>()
        feat_interp.packed_accessor<scalar_t, 2, torch::RestrictTraits, size_t>(),
        // gates.data<scalar_t>(),
        // old_cell.data<scalar_t>(),
        // new_h.data<scalar_t>(),
        // new_cell.data<scalar_t>(),
        // input_gate.data<scalar_t>(),
        // output_gate.data<scalar_t>(),
        // candidate_cell.data<scalar_t>(),
        // state_size);
  }));

    // // if only one part need parallel.
    // const int threads = 256;
    // const dim3 threads(256);



}