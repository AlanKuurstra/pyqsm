from distutils.core import setup, Extension

import numpy
import os

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

print("""
In setup.py
""")



extensions = []

includeDirs = numpy.get_include()
libDirs = "" #giving a blank lib list causes errors!
libs = ""

extensions.append(
    Extension('calculateReliability',
              sources=['calculateReliabilityInterface.cpp'],
              include_dirs=[includeDirs],
              #library_dirs=[libDirs], giving a blank lib list causes errors!
              #libraries=libs, giving a blank lib list causes errors!
              #extra_compile_args=[]
              )
    )

setup(
    name="calculateReliabilityInterface",
    version="0.1",
    description = "Python interface to calculateReliability",
    #packages=['PfileInterface'],
    ext_modules=extensions)
