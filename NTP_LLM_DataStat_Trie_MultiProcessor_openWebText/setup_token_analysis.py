# setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "token_analysis",
        ["token_analysis.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
    ),
]

setup(
    name="token_analysis",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4.3'],
    setup_requires=['pybind11>=2.4.3'],
)