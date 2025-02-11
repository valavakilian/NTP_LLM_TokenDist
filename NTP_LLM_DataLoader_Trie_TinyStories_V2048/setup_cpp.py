from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

# Get the absolute path to your source directory
source_dir = '/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_DataLoader_Trie_TinyStories_V2048'

setup(
    name='Trie_dataloader',
    ext_modules=[
        CppExtension(
            name='Trie_dataloader',
            sources=[os.path.join(source_dir, 'Trie_dataloader.cpp')],
            include_dirs=[source_dir],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3', 
                    '-fopenmp',
                    '-fPIC',
                    '-DTORCH_API_INCLUDE_EXTENSION_H',
                    '-DTORCH_EXTENSION_NAME=Trie_dataloader',
                ],
            },
            extra_link_args=['-fopenmp', '-fPIC']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)