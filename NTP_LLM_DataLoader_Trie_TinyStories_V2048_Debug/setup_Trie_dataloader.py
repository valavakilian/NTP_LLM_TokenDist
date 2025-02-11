from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='Trie_dataloader',
    ext_modules=[
        CppExtension(
            name='Trie_dataloader',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_SoftLabelLoader_openWebText/Trie_dataloader.cpp'],
            extra_compile_args=['-std=c++17', '-O3', '-fopenmp'],
            extra_link_args=['-fopenmp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)