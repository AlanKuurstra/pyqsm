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
    Extension('unwrap3d',
              sources=['unwrap3dInterface.cpp'],
              include_dirs=[includeDirs],
              #library_dirs=[libDirs], giving a blank lib list causes errors!
              #libraries=libs, giving a blank lib list causes errors!
              #extra_compile_args=[]
              )
    )

setup(
    name="unwrap3dInterface",
    version="0.1",
    description = "Python interface to unwrap3d",
    #packages=['PfileInterface'],
    ext_modules=extensions)
