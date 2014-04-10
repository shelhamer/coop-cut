from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os, sys
import numpy

os.environ['CC'] = 'g++'
if sys.platform == 'darwin':
    os.environ['CC'] = 'clang++'  # maxflow must be compiled by clang++ on OSX

coop_dir = 'coop-cut'
coop_files = ['coop_cut.cpp', 'image_graph.cpp', 'costfun.cpp', 'itBM.cpp',
              'maxflow/graph.cpp', 'maxflow/maxflow.cpp']

build_files = [os.path.join(coop_dir, f) for f in coop_files]
build_files.insert(0, 'pycoop.pyx')

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('pycoop', build_files, language='c++',
                             include_dirs=[coop_dir, coop_dir + '/maxflow',
                                           numpy.get_include()],
                             library_dirs=[coop_dir, coop_dir + '/maxflow'],
                             extra_compile_args=['-DNDEBUG', '-fpermissive'])
                  ])
