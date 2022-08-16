#include <torch/extension.h>

/*
__host__ : call CPU and execute in CPU
__device__: call GPU and execute in GPU

__global__: call in CPU and execute in GPU.
*/

/*
In cuda coding: only have the void type for function.
*/

template <typename scalar_t>
__global__ void trilinear_fw_kernel (
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictTraits, size_t> feat_interp
){
    /*
      1. Getting the thread ID;
      2. filter the useless threads.
    */
    // getting the thread ID.
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= feats.size(0) || f >= feats.size(2)) return;
    
    // point (-1~1), need to normalize
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;

    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] + 
                                b*feats[n][1][f] +
                                c*feats[n][2][f] +
                                d*feats[n][3][f]) + 
                        u*(a*feats[n][4][f] + 
                            b*feats[n][5][f] +
                            c*feats[n][6][f] + 
                            d*feats[n][7][f]);
}






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

    /* 
    scalar_t: place_holder <if don't know the exact data type.>
    <<<blocks, threads>>>: number of blocks and number of threads
    Input and Ouputs: 
            feats.packed_accessor (packed_accessor: covert the tensor type to others)
            points.packed_accessor
            feat_interp.packed_accessor
        because cuda do not return the result.
    <scalar_t, 3, torch::RestrictTraits, size_t>: 
        <data_type of input, dimension of input, Individual storage, shape>
    */
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] {
        // if adding a Int variable, just add in it, no need to packed_accessor
        // a,
        trilinear_fw_cu<scalar_t><<<blocks, threads>>>(
        feats.packed_accessor<scalar_t, 3, torch::RestrictTraits, size_t>(),
        points.packed_accessor<scalar_t, 2, torch::RestrictTraits, size_t>()
        feat_interp.packed_accessor<scalar_t, 2, torch::RestrictTraits, size_t>(),
        
        /* If already know the data type
        trilinear_fw_cu<<<blocks, threads>>>(
        feats.packed_accessor<float, 3, torch::RestrictTraits, size_t>(),
        points.packed_accessor<float, 2, torch::RestrictTraits, size_t>()
        feat_interp.packed_accessor<float, 2, torch::RestrictTraits, size_t>(),

        */
        
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