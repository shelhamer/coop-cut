from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

maxflow_dir = 'maxflow'
maxflow_files = ['graph.cpp', 'maxflow.cpp']

build_files = [os.path.join(maxflow_dir, f) for f in maxflow_files]
build_files.insert(0, 'pymaxflow.pyx')

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('pymaxflow', build_files, language='c++',
                             include_dirs=[maxflow_dir, numpy.get_include()],
                             library_dirs=[maxflow_dir],
                             extra_compile_args=['-DNDEBUG', '-fpermissive'])
                  ])
