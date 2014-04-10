// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#ifndef COOP_CUT_IMAGE_GRAPH_H_
#define COOP_CUT_IMAGE_GRAPH_H_

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// Collect image and edge information for the cooperative random field:
// 0. Collect unary potentials and image dimensions.
// 1. Make the 8-neighbor graph over image pixels.
// 2. Weight edges by the image gradient for pairwise potentials.
// 3. Collect edge classes for the high-order cooperative potential.

class ImageGraph {
 public:
  ImageGraph() : h(0), w(0), chn(0), n(0), m(0), num_classes(0),
                 unaries(NULL), edge_weights(NULL), edge_classes(NULL) {}
  ~ImageGraph() {
    delete[] unaries;
    delete[] edge_weights;
    delete[] edge_classes;
    delete[] edge_gradients;
  }

  // Image dimensions.
  unsigned int h, w, chn;

  // n: number of nodes
  unsigned int n;

  // m: number of edges
  // num_classses: number of edge classes
  unsigned int m, num_classes;

  // Unary potentials for source / sink:
  // row-major order of nodes, 2*n in length, with source then sink.
  double* unaries;

  // Edge weights and classes.
  // weights has length m, and classes has length 2*m, since edges
  // (i,j) and (j,i) have the same weight, but can have different
  // classes since the gradient between neighbors is signed.
  //
  // Edges are in the following order from the 8-neighbor graph:
  // 1. vertical up-down
  // 2. horizontal left-right
  // 3. diagonal, top-left to bottom-right and reverse
  // 4. diagonal, bottom-left to top-right and reverse
  // where edges within a set are in row-major order of the nodes.
  //
  // Note that "sister" edges immediately follow one another in the order.
  double* edge_weights;
  int* edge_classes;

  // Signed gradients of the directed edges in the image graph,
  // with sister edges following one another as described above for
  // edge weights and classes. (2*m)*3 in total since because
  // this is the signed/directed gradient over three channels.
  double* edge_gradients;

  // Set image dimensions
  void setDims(unsigned int h_in, unsigned int w_in, unsigned int chn_in);

  // Set unaries by array.
  void setUnaries(double* source_unaries, double* sink_unaries);

  // Set custom edge weights.
  void setEdgeWeights(double* edge_weights_in);

  // Set classes by array.
  void setClasses(int* classes_in, int num_classes_in,
                   double modthresh = 0.0000000001);

  // Extract edge weights and directed edge gradients from an image array.
  void extractEdges(double* im_arr);
};

#endif  // COOP_CUT_IMAGE_GRAPH_H__
