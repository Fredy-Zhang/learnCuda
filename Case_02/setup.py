from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
import glob
import os.path as osp

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')

setup(
        name="cppcuda",
        version="1.0",
        author="fredy",
        author_email="",
        description="",
        long_description="",
        ext_modules=[
            CUDAExtension(
                name="cppcuda",
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args={"css": ['-02'],
                                    "nvcc": ['-02']}
                )
            ],
        cmdclass={
            'build_ext': BuildExtension
            }
)

