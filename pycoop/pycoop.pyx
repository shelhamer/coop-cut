#cython: boundscheck=False
#cython: wraparound=False
"""
TODO
- safety checks on numpy inputs
"""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool

np.import_array()  # initialize numpy


cdef extern from 'coop_cut.hpp':
    double coop_cut(vector[unsigned int]& cut, vector[bool]& node_labels,
                    ImageGraph* im_graph, double lambda_, double theta,
                    int max_iter)


cdef extern from 'image_graph.hpp':
    cdef cppclass ImageGraph:
        ImageGraph()
        void setDims(int height, int width, int channels)
        void setUnaries(double* source_unaries, double* sink_unaries)
        void setEdgeWeights(double* weights)
        void setClasses(int* classes, int num_classes)
        void setClasses(int* classes, int num_classes, double modthresh)
        void extractEdges(double* im_arr)
        unsigned int n, m, h, w, chn
        double* edge_gradients  # TODO shouldn't be public


cpdef segment(InputGraph im_graph, double lambda_, double theta, int max_iter):
    cdef vector[unsigned int] cut
    cdef vector[bool] node_labels
    cdef double cost
    cost = coop_cut(cut, node_labels, im_graph._g, lambda_, theta, max_iter)
    h, w, _ = im_graph.dims
    label_im = np.reshape(np.array(node_labels), (h, w))
    return label_im, cost, cut


cdef class InputGraph:
    cdef ImageGraph* _g


    def __cinit__(self, np.ndarray[double, ndim=3] im_arr):
        """
        Create image graph input representation.

        Take
            im_arr: input image ndarray, [H x W x C], 0-255, double.
        """
        cdef int h, w, k
        self._g = new ImageGraph()
        h = im_arr.shape[0]
        w = im_arr.shape[1]
        k = im_arr.shape[2]
        self._g.setDims(h, w, k)
        self._g.extractEdges(&im_arr[0,0,0])


    def __dealloc__(self):
        """
        Free memory for input representation.
        """
        del self._g


    @property
    def n(self):
        """
        Number of nodes/pixels.
        """
        return self._g.n


    @property
    def m(self):
        """
        Number of (undirected) edges.
        """
        return self._g.m


    @property
    def dims(self):
        """
        Dimensions of image input.
        """
        return self._g.h, self._g.w, self._g.chn


    @property
    def edges(self):
        """
        Directed edge gradients. ndarray, [2*m x chn], int

        TODO should generate once at construction
        """
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> 2*self.m
        shape[1] = <np.npy_intp> self._g.chn
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE,
                                           self._g.edge_gradients);
        return arr


    def setUnaries(self, np.ndarray[double, ndim=1] source,
                   np.ndarray[double, ndim=1] sink):
        """
        Set unary potentials.

        Take
            source: ndarray of length n
            sink: ...
        """
        self._g.setUnaries(&source[0], &sink[0])


    def setEdgeWeights(self, np.ndarray[double, ndim=1] weights):
        """
        Set custom edge weights.

        Take
            weights: flat ndarray of length m
        """
        self._g.setEdgeWeights(&weights[0])


    def setClasses(self, np.ndarray[int, ndim=1] classes, int num_classes,
                   modthresh = None):
        """
        Set edge classes, with or without modular thresholding.

        Take
            classes: flat ndarray of length 2*m
            num_classes: number of edge classes
            modthresh: modular threshold after which discount begins.
        """
        if not modthresh:
            self._g.setClasses(&classes[0], num_classes)
        else:
            self._g.setClasses(&classes[0], num_classes, modthresh)

