from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

source_dir = '/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_SoftLabelLoader_openWebText'

setup(
    name='Trie_module',
    ext_modules=[
        CppExtension(
            name='Trie_module',
            sources=[os.path.join(source_dir, 'Trie_module.cpp')],
            include_dirs=[source_dir],
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3', '-fopenmp', '-fPIC'],
            },
            extra_link_args=['-fopenmp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)