#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pkgconfig

if ((sys.version_info[0] == 2 and sys.version_info[1] < 7) or
   (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
    sys.stderr.write("Error in setup script for HTSeq:\n")
    sys.stderr.write("HTSeq support Python 2.7 or 3.4+.")
    sys.exit(1)


# Setuptools but not distutils support build/runtime/optional dependencies
# NOTE: old setuptools < 18.0 has issues with extras
kwargs = dict(
    setup_requires=[
        'pybind11>=2.2',
        'numpy',
        'pkgconfig',
    ],
    install_requires=[
        'numpy',
    ],
    extras_require={
    },
  )


try:
    import numpy
except ImportError:
    sys.stderr.write("Setup script: Failed to import 'numpy'.\n")
    sys.stderr.write("Please install numpy and then try again.\n")
    sys.exit(1)

numpy_include_dir = os.path.join(os.path.dirname(numpy.__file__),
                                 'core', 'include')

# Update version from VERSION file into module
with open('VERSION') as fversion:
    __version__ = fversion.readline().rstrip()
with open('lshknn/_version.py', 'wt') as fversion:
    fversion.write('__version__ = "'+__version__+'"')


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-msse4.2'],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(name='lshknn',
      version=__version__,
      author='Fabio Zanini, Paolo Carnevali',
      author_email='fabio.zanini@stanford.edu',
      maintainer='Fabio Zanini',
      maintainer_email='fabio.zanini@stanford.edu',
      url='https://github.com/iosonofabio/lshknn',
      description="k nearest neighbor (KNN) graphs via Pearson correlation distance and local sensitive hashing (LSH).",
      long_description="""
k nearest neighbor (KNN) graphs via Pearson correlation distance and local sensitive hashing (LSH).

- **Development**: https://github.com/iosonofabio/lshknn
- **Authors**: Fabio Zanini and Paolo Carnevali
- **License**: MIT
- **Copyright**: Fabio Zanini and Chan Zuckerberg Initiative

.. code-block:: python

    import numpy as np
    import lshknn

    # Make mock data
    # 2 features (rows), 4 samples (columns)
    data = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1]],
            dtype=np.float64)

    # Instantiate class
    c = lshknn.Lshknn(
            data=data,
            k=1,
            threshold=0.2,
            m=10,
            slice_length=4)

    # Call subroutine
    knn, similarity, n_neighbors = c()

    # Check result
    assert (knn == [[2], [3], [0], [1]]).all()

      """,
      license='MIT',
      classifiers=[
         'Topic :: Scientific/Engineering :: Bio-Informatics',
         'Intended Audience :: Developers',
         'Intended Audience :: Science/Research',
         'License :: OSI Approved :: MIT License',
         'Operating System :: POSIX',
         'Programming Language :: Python'
      ],
      ext_modules=[
         Extension(
             'lshknn._lshknn',
             ['lshknn/lshknn.cpp'],
             include_dirs=[
                 numpy_include_dir,
                 get_pybind_include(),
                 get_pybind_include(user=True)] +
                 pkgconfig.parse("eigen3")["include_dirs"],
             language='c++',
             ),
      ],
      py_modules=[
          'lshknn.__init__'
      ],
      cmdclass={'build_ext': BuildExt},
      zip_safe=False,
      **kwargs
      )
