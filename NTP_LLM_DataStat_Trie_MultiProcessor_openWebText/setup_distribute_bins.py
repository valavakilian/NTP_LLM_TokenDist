from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='distribute_bins_cpp',
    ext_modules=[
        CppExtension(
            name='distribute_bins_cpp',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_DataStat_Trie_MultiProcessor_openWebText/distribute_bins.cpp'],
            extra_compile_args=['-std=c++17', '-O3']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)