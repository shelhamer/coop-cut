// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "image_graph.hpp"

// TODO combine edge weights and classes data like in comm-cut?

// Set dimensions.
void ImageGraph::setDims(unsigned int h_in, unsigned int w_in,
                         unsigned int chn_in) {
  h = h_in;
  w = w_in;
  chn = chn_in;
  // A node for every pixel.
  n = h * w;
  // Number of (undirected) edges:
  // 8-neighbors except for the missing boundary edges and double-counting.
  // Drop 6 left/right edges for rows and 6 top/bottom edges for columns, and
  // adjust for the double-counting of the corners.
  m = 4*h*w - 3*h - 3*w + 2;

  // Handle memory.
  delete[] unaries;
  delete[] edge_weights;
  delete[] edge_classes;
  unaries = new double[2*n];
  edge_weights = new double[m];
  edge_classes = new int[2*m];
  edge_gradients = new double[6*m];
}


// Set unaries from source and sink arrays.
void ImageGraph::setUnaries(double* source_unaries, double* sink_unaries) {
  std::copy(source_unaries, source_unaries + n, unaries);
  std::copy(sink_unaries, sink_unaries + n, unaries + n);
}


// Set custom edge weights.
void ImageGraph::setEdgeWeights(double* edge_weights_in) {
  std::copy(edge_weights_in, edge_weights_in + m, edge_weights);
}


// Set classes by array.
// If you do not want modular treatment, then set modthresh < 0.
// class_in must have 2*m elements in edge list order.
void ImageGraph::setClasses(int* classes_in, int num_classes_in,
    double modthresh) {
  std::copy(classes_in, classes_in + 2*m, edge_classes);
  num_classes = num_classes_in;

  if (modthresh >= 0) {
    // threshold for weights from when we treat it as modular
    double wthresh = 0.05 + 0.95*exp(-modthresh);

    int ct = 0;
    for (unsigned int i = 0; i < m; i++) {
      if (edge_weights[i] > wthresh) {
        edge_classes[2*i] = num_classes;
        edge_classes[2*i+1] = num_classes;
        ++ct;
      }
    }
    printf("%d of %d not submod: %1.2e\n", ct, m, ((double)ct/(double)m)*100);
  }
}


// Extract undirected edge weights and directed edge gradients
// from an input image array.
// Weights of sister edges aren't recorded, since they are redundant,
// although sister edge gradients are, since they are signed.
void ImageGraph::extractEdges(double* im_arr) {
  // Mean square of the edge gradients
  double sigma = 0.0;

  // Edge traversal bookkeeping
  int i, j, k;
  int step = w*chn;  // row-major contiguous array
  double diff = 0.0;
  unsigned int e_ix = 0;

  // Vertical
  for (j = 0; (int) j < w; j++) {
    for (i = 0; (int) i < (h-1); i++) {
      edge_weights[e_ix] = 0.0;
      for (k = 0; (int) k < chn; k++) {
        diff = im_arr[(i+1)*step + j*chn + k]
               - im_arr[i*step + j*chn + k];
        edge_weights[e_ix] += diff * diff;
        edge_gradients[e_ix*6 + k] = diff;
        edge_gradients[e_ix*6 + chn + k] = -diff;
      }
      sigma += edge_weights[e_ix] / m;
      e_ix++;
    }
  }

  // Horizontal
  for (j = 0; (int) j < (w-1); j++) {
    for (i = 0; (int) i < h; i++) {
      edge_weights[e_ix] = 0.0;
      for (k = 0; (int) k < chn; k++) {
        diff = im_arr[i*step + (j+1)*chn + k]
             - im_arr[i*step + j*chn + k];
        edge_weights[e_ix] += diff * diff;
        edge_gradients[e_ix*6 + k] = diff;
        edge_gradients[e_ix*6 + chn + k] = -diff;
      }
      sigma += edge_weights[e_ix] / m;
      e_ix++;
    }
  }

  // Diagonal right/down
  for (j = 0; (int) j < (w-1); j++) {
    for (i = 0; (int) i < h-1; i++) {
      edge_weights[e_ix] = 0.0;
      for (k = 0; (int) k < chn; k++) {
        diff = im_arr[(i+1)*step + (j+1)*chn + k]
             - im_arr[i*step + j*chn + k];
        edge_weights[e_ix] += diff * diff;
        edge_gradients[e_ix*6 + k] = diff;
        edge_gradients[e_ix*6 + chn + k] = -diff;
      }
      sigma += edge_weights[e_ix] / m;
      e_ix++;
    }
  }

  // Diagonal left/down
  for (j = 0; (int) j < w-1; j++) {
    for (i = 0; (int) i < h-1; i++) {
      edge_weights[e_ix] = 0.0;
      for (k = 0; (int) k < chn; k++) {
        diff = im_arr[(i+1)*step + j*chn + k]
             - im_arr[i*step + (j+1)*chn + k];
        edge_weights[e_ix] += diff * diff;
        edge_gradients[e_ix*6 + k] = diff;
        edge_gradients[e_ix*6 + chn + k] = -diff;
      }
      sigma += edge_weights[e_ix] / m;
      e_ix++;
    }
  }

  // Set weights by exponential dropoff by gradient magnitude, normalized by
  // average edge deviation over image.
  for (i = 0; i < (int)m; i++) {
    edge_weights[i] = 0.05 + 0.95*exp(-edge_weights[i]/(2*sigma));
  }

  printf("sigma = %1.3e\n", sigma);
}
