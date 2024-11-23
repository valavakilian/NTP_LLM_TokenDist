
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='trie_module_protV1_lib',
    ext_modules=[
        CppExtension(
            name='trie_module_protV1_lib',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_Trie_MultithreadDebug/trie_module_protV1_singleThread.cpp'],  # Correct file name
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
