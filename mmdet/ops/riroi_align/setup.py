from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='riroi_align_cuda',
    ext_modules=[
        CUDAExtension('riroi_align_cuda', [
            'src/riroi_align_cuda.cpp',
            'src/riroi_align_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
