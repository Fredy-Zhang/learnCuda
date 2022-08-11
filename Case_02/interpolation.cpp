#include <torch/extension.h>

// define the header for 
#include "utils.h"

// as a bridge to CUDA, running code in CUDA in GPU
torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor point
){
    return trilinear_fw_cu(feats, points);
}

// Using python to call the C++
//
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}

