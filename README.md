[Cooperative Cut](coopcut.berkeleyvision.org)

This is the reference implementation of the Cooperative Cut inference algorithm for Markov Random Field (MRF) models and its application to segmentation.

This implementation relies on [Kolmogorov's](http://pub.ist.ac.at/~vnk/software.html) [maxflow](http://pub.ist.ac.at/~vnk/software/maxflow-v3.04.src.zip).

The package includes:

- the reference implementation of cooperative cut (C++)
- a Python wrapper for cooperative cut
- a Python wrapper for graph cut / maxflow
- unary learning and extraction
- edge group learning and clustering

## License

Cooperative Cut is BSD 2-Clause licensed (refer to the [LICENSE](http://coopcut.berkeleyvision.org/license.html) for details).

## Citing

**A tech report on this implementation and extensions to the model are coming soon!**

In the meantime, the approach is described in the paper

  S. Jegelka and J. Bilmes. "Submodularity beyond submodular energies: coupling edges in graph cuts". IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.

Cite this paper if you use this code.

## Installation

**Dependencies:**

Cooperative Cut depends on the Kolmogorov maxflow C++ library (bundled) and Python with numpy (matrix computation), scikit-image (image IO) scikit-learn (mixture modeling and clustering), and cython (wrapper interface).

The easiest way to satisfy the requirements is to install [Anaconda Python](http://continuum.io/downloads). It is an excellent package for scientific computing and research coding with Python. Note that once installed, Anaconda Python must be included in your `PATH` environment variable.

To double-check your installation, open a terminal and run `python`. The output should resemble

    Python 2.7.7 |Anaconda 1.9.2 (x86_64)| (default, Jun 19 2014, 13:38:03)
    [GCC 4.0.1 (Apple Inc. build 5493)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    Anaconda is brought to you by Continuum Analytics.
    Please check out: http://continuum.io/thanks and https://binstar.org

**Python wrapper:** To compile the Python wrapper module, you need to first add the libraries included with Anaconda to your `LD_LIBRARY_PATH` by `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/anaconda/lib` or the like. With that done,

    cd pycoop
    make

will build the module `pycoop.so`. *Note* that `setup.py` configures the build in case you need to customize anything for your environment.

In order to import the module into Python, it needs to be added to your `PYTHONPATH` environment variable. (Note that this step isn't need to run the example! It's enough to be in the same directory as the module in this case.) You can either `export PYTHONPATH=$PYTHONPATH:/path/to/coop-cut` to add the repository directory; however the preferred method is to

    mkdir ~/python
    ln -s /path/to/pycoop ~/python/pycoop
    export PYTHONPATH=$PYTHONPATH:$HOME/python

for convenience. This way you can link new modules without constantly messing with your environment variables.

To check the installation, open `python` and run `import pycoop`. The command should execute without error and return to you to the prompt. Run `exit()` to close Python.

On to the example!

## Example

Start the example by `./example.sh` or

    ipython notebook --pylab=inline coopcut.ipynb

for our tutorial exposition of the code and an example segmentation with `pycoop`, the Python interface.

## Data

[Data](http://melodi.ee.washington.edu/~jegelka/cc/index.html) for the experiments in the paper includes images with shading and fine structures.
