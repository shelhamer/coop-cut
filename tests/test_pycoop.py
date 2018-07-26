from skimage import io
import numpy as np
from pycoop import pycoop
from pycoop import potentials

im = io.imread('data/bee.jpg').astype(np.float64)

def test_input_graph():
    ig = pycoop.InputGraph(im)
    assert(isinstance(ig, pycoop.InputGraph))

def test_cluster_edges():
    ig = pycoop.InputGraph(im)
    edge_cluster_classes, edge_centroids = potentials.cluster_edges(ig.edges, k=8)
    assert(isinstance(edge_centroids, np.ndarray))
    assert(len(edge_centroids) > 0)