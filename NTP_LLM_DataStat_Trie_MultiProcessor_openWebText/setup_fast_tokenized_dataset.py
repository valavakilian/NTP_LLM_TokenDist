from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='fast_tokenized_dataset',
    ext_modules=[
        CppExtension(
            name='fast_tokenized_dataset',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_DataStat_Trie_MultiProcessor_openWebText/fast_tokenized_dataset.cpp'],
            extra_compile_args=['-std=c++17', '-O3', '-fopenmp'],
            extra_link_args=['-fopenmp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)