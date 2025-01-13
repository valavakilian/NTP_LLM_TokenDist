from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='root_context_dict',
    ext_modules=[
        CppExtension(
            name='root_context_dict',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_DataStat_Trie_MultiProcessor_openWebText/root_context_dict.cpp'],
            extra_compile_args=['-std=c++17', '-O3'],
            include_dirs=['pybind11/include']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['pybind11>=2.6.0']
)