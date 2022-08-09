#include <torch/extension.h>


// as a bridge to CUDA, running code in CUDA in GPU
torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor point
){
    return feats;
}

// Using python to call the C++
//
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}

