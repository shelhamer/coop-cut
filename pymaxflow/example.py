#!/usr/bin/env python
import pymaxflow
import numpy as np

def readme_example():
  """
  The Boykov-Kolmogorov maxflow README example.
  """
  g = pymaxflow.Graph(2,1)
  g.add_node()
  g.add_node()
  g.set_unary(0, 1, 5)
  g.set_unary(1, 2, 6)
  g.add_edge(0, 1, 3, 4)
  flow, labels = g.maxflow()
  print "Flow: {}".format(flow)
  for i, l in enumerate(labels):
      print "{} in {}".format(i, 'source' if l == pymaxflow.SOURCE else 'sink')


def random_example():
  """
  Simple randomized example.

  TODO: fix to avoid duplicate edges, increase size, display input graph and
  output labels.
  """
  g = pymaxflow.Graph(100,20)
  unary = np.random.random((100,2))
  pairwise = np.random.random((20,2))
  connections = np.round(np.random.random((20,2))*20)
  edges = np.concatenate([connections, pairwise], axis=1)
  g.set_nodes(unary) ; g.set_edges(edges)
  flow, labels = g.maxflow()
  print "Flow: {}".format(flow)
  for i, l in enumerate(labels):
      print "{} in {}".format(i, 'source' if l == pymaxflow.SOURCE else 'sink')


if __name__ == "__main__":
  readme_example()
