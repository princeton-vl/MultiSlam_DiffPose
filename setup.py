import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['multi_slam/dpvo/altcorr/correlation.cpp', 'multi_slam/dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['multi_slam/dpvo/fastba/ba.cpp', 'multi_slam/dpvo/fastba/ba_cuda.cu', 'multi_slam/dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')]
            ),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'multi_slam/dpvo/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'multi_slam/dpvo/lietorch/src/lietorch.cpp', 
                'multi_slam/dpvo/lietorch/src/lietorch_gpu.cu',
                'multi_slam/dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

