# learnCuda
The simple codes are used to write the basic frameworks for CUDA in Pytorch Extension.

## How python calling cuda code?
It will use the C++ code as the bridge, in short, python calls C++ code, then C++ code calls the Cuda code. In addition, Cuda code will be excuted in GPU directly.

## Which cases are suitable to use the CUDA?
1. For the case where there are a large number of non-parallel operations.
2. A large number of serial calculations.

![cuda structure](https://user-images.githubusercontent.com/62839136/183607470-b2315047-9e20-4112-b090-c67fd565fd21.png)
