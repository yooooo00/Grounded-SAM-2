from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ms_deform_attn',
    ext_modules=[
        CUDAExtension('ms_deform_attn', [
            'ms_deform_attn.cpp',
            'ms_deform_attn_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
