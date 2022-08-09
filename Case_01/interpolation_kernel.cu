#include <torch/extension.h>

// fw: forward: put input to get the output.
// bw: backward: output to update the parameters.
// cu: cuda
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor point
){
    return feats;
}