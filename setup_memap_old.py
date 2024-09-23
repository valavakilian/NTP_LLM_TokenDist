# from setuptools import setup, Extension
# from torch.utils.cpp_extension import BuildExtension, CppExtension

# setup(
#     name='trie_module',
#     ext_modules=[
#         CppExtension('trie_module', ['trie_module.cpp']),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     },
#     install_requires=['torch>=1.0'],
# )


from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='trie_module_memap_old',
    ext_modules=[
        CppExtension(
            name='trie_module_memap_old',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_TokenDist_Wiki_childDict/trie_module_memap_old.cpp'],  # Correct file name
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
