
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='trie_module_memap_sorted',
    ext_modules=[
        CppExtension(
            name='trie_module_memap_sorted',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_TokenDist_Wiki_childDict/trie_module_memap_sorted.cpp'],  # Correct file name
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
