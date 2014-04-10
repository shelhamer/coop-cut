#cython: boundscheck=False
#cython: wraparound=False
"""
TODO
- combine set_nodes() and set_edges + call to reset(), since set_* are unlikely
  to be called in isolation
- add safety checks for graph structure, potentials
- fast ndarray io
- hook in error function callback for raising exceptions
- wrap dynamic graphcut
"""

import numpy as np
cimport numpy as np
ctypedef double NPCapacity  # TODO use numpy float dtype


cdef extern from 'pymaxflow.h':
    ctypedef float Capacity
    # TODO I'm suspicious this is the proper way to expose the enum to both
    # python and cython.
    ctypedef enum Terminal:
        tSOURCE
        tSINK
    ctypedef int NodeID
    cdef const Terminal kSOURCE
    cdef const Terminal kSINK

SOURCE = kSOURCE
SINK = kSINK


cdef extern from 'graph.h':
    # G = Graph<Capacity, Capacity, Capacity>
    cdef cppclass G:
        G(int node_num_max, int edge_num_max)  # TODO add error function
        NodeID add_node(int num)
        void add_edge(NodeID i, NodeID j, Capacity cap, Capacity rev_cap)
        void add_tweights(NodeID i, Capacity cap_source, Capacity cap_sink)
        Capacity maxflow()  # TODO wrap args for dynamic graph cuts
        Terminal what_segment(NodeID i)
        void reset()
        NodeID get_node_num()


cdef class Graph:
    cdef G* _g

    def __cinit__(self, node_num, edge_num):
        """
        Create maxflow graph.

        Take
            V: number of nodes...
            E: number of edges...
               ...to allocate. Including more triggers costly re-allocation.
        """
        self._g = new G(node_num, edge_num)


    def __dealloc__(self):
        """
        Free memory for maxflow graph.
        (This should be automatic, as far as I know, but explicit is best.)
        """
        del self._g


    def add_node(self, num = None):
        """
        Add node(s) to graph.

        Take
            num: number of nodes to add.
        """
        if not num:
            num = 1
        self._g.add_node(num)


    def add_edge(self, i, j, cap, rev_cap):
        """
        Add edge to graph.

        Take
            i, j: nodes spanned by edge
            cap: i -> j capacity of the edge
            rev_cap: j -> i capacity of the edge
        """
        self._g.add_edge(i, j, cap, rev_cap)


    def set_unary(self, i, cap_source, cap_sink):
        """
        Set unary potential for source/sink at a node.

        Take
            i: node index
            cap_source, cap_sink: source/sink capacities
                                  (lower cap = higher affinity).
        """
        self._g.add_tweights(i, cap_source, cap_sink)


    def set_nodes(self, np.ndarray[NPCapacity, ndim=2] nodes):
        """
        Mass instantiate nodes and unaries.

        Take
            nodes: Capacity ndarray, N x 2, with rows [SOURCE, SINK] of
                   SOURCE and SINK potentials for the N nodes.
        """
        if np.any(nodes < 0):
            raise ValueError("Unaries must be nonnegative.")
        cdef int n = nodes.shape[0]
        cdef unsigned int i
        for i in range(0, n):
            ix = self._g.add_node(1)
            self._g.add_tweights(ix, nodes[i, 0], nodes[i, 1])


    def set_edges(self, np.ndarray[NPCapacity, ndim=2] edges):
        """
        Mass instantiate edges and pairwise potentials.

        Take
            edges: ndarray, N x 4, of [i, j, cap, rev_cap] for node connections
                   and potentials. (Note dtype must be Capacity.)
        """
        if np.any(edges < 0):
            raise ValueError("Pairwise potentials must be nonnegative.")
        cdef int n = edges.shape[0]
        cdef unsigned int i
        for i in range(0, n):
            self._g.add_edge(<NodeID>edges[i, 0], <NodeID>edges[i, 1],
                            edges[i, 2], edges[i, 3])


    def maxflow(self):
        """
        Compute the flow and unpack the cut.

        Give
            flow: energy value after optimization.
            cut: the sink/source assignment of nodes, list.
        """
        flow = self._g.maxflow()
        cut = [self._g.what_segment(i)
                    for i in range(self._g.get_node_num())]
        return (flow, cut)


    def reset(self):
        """
        Reset the graph, clearing all nodes and edges.
        """
        self._g.reset()
