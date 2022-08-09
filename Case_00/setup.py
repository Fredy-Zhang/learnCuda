from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

setup(
        name="cppcuda",
        version="1.0",
        author="fredy",
        author_email="",
        description="",
        long_description="",
        ext_modules=[
            CppExtension(
                name="cppcuda",
                sources=['interpolation.cpp'])
            ],
        cmdclass={
            'build_ext': BuildExtension
            }
)

