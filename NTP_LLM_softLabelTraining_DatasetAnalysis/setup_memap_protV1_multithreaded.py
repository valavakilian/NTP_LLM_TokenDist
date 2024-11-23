
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='trie_module_protV1_lib_multithreaded',
    ext_modules=[
        CppExtension(
            name='trie_module_protV1_lib_multithreaded',
            sources=['/arc/project/st-cthrampo-1/vala/NTP_LLM_tokenDist_Wiki/NTP_LLM_softLabelTraining_DatasetAnalysis/trie_module_protV1_multiThread.cpp'],  # Correct file name
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
