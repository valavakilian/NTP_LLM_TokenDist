
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='trie_module_protV1_lib_multithreadedRAM',
    ext_modules=[
        CppExtension(
            name='trie_module_protV1_lib_multithreadedRAM',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Debug_Folder_RAM/trie_module_protV1_multiThread_RAM.cpp'],  # Correct file name
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
