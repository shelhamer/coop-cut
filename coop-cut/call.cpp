// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cstdio>
#include "coop_cut.hpp"
#include "costfun.hpp"
#include "image_graph.hpp"
#include "itBM.hpp"
#include "util.hpp"

//  This example computes a cooperative cut segmentation for an image with
//  predefined unaries and edge classes. For the theory and algorithm, see
//
//  S. Jegelka and J. Bilmes. "Submodularity beyond submodular energies:
//  coupling edges in graph cuts". IEEE Conference on Computer Vision and
//  Pattern Recognition (CVPR), 2011.
//
//  Please cite this paper if you use this library in any form.
//
//  For implementation details, see coop_cut.cpp.
//
//  In lieu of OpenCV, one can create the ImageGraph holding the input data
//  through setDims(), setUnaries(), setEdgeWeights(), and setEdgeClasses()
//
//  Edge classes are assumed to be in a different binary file that uses the
//  following format:
//  number_of_classes
//  class of edge 1
//  class of edge 2 .....
//  ... class of edge m
//
//  - edges are the 8-neighbors of every pixel.
//  - order: vertical up-down, horizontal left-right,
//    diagonal down left and reverse, diagonal up right and reverse
//
//  The cut cost is
//  cost(C)= w(C \cap E_{terminal}) + lambda * f(C \cap E_{interpixel})
//  where f(C) = \sum_i f_i(C \cap S_i)
//  and f_i(A) = w(A \cap S_i)  if w(A \cap S_i) <= theta*w(S_i);
//  f_i(A) = theta*w(S_i) + sqrt(1+ w(A \cap S_i) - theta*w(S_i)) -1
//  otherwise. Edges between nodes of identical color are treated in a separate
//  group without any discounts.
//
//  To use something else than sqrt, replacing the function 'sqrt2'
//  in costfun.cpp should work (but no guarantees).


int main(int argc, char** argv) {
  if (argc < 5) {
    printf("usage: coopcut image_file edge_class_file unary_file "
           "lambda theta\n");
    return 1;
  }

  // parameters:
  double lambda = atof(argv[4]);  // regularization coefficient
  double theta = atof(argv[5]);  // threshold for edge potential discounting
  char* classfile = argv[2];  // file that has the edge classes in binary format
  char* imname = argv[1];     // name of the image
  char* unaryFile = argv[3];  // name of the file with the terminal weights

  // Cap iterations of the approximate optimization.
  int max_iter = 12;

  // Load input image.
  cv::Mat img = cv::imread(imname, CV_LOAD_IMAGE_COLOR);
  if (!img.data) { printf("Could not load image file!\n"); }

  // 1. Create an image graph structure to hold the image dimensions, unaries,
  //    and edge weights and classes.
  ImageGraph* im_graph = new ImageGraph();
  im_graph->setDims(img.rows, img.cols, img.channels());

  // Role of the modthresh parameter: In the CVPR paper's experiments, we kept
  // one special class of edges that do not enjoy any discount. This class is
  // for edges between very similar nodes, where the distance between their
  // feature vectors is smaller than modthresh.
  double modthresh = 0.0000000001;

  // 2a. Read the unaries into the image graph structure. 2*n total
  // for source and sink, in row-major order.
  coop_read_unaries(im_graph, unaryFile);

  // 2b. Create edge weights from the image.
  coop_read_weights(im_graph, imname);
  // 2c. Read in the classes, the order of the classes must follow the order of
  // the edges outlined above. For each edge described above, the following edge
  // is the reverse edge, meaning if edge 2*i is (u,v), then edge 2*i+1 is
  // (v,u). modthresh determines when two nodes are similar enough s.t. the edge
  // between them is heavy enough to not get discounts.
  coop_read_classes(im_graph, classfile, modthresh);

  // Do the rest through the coop_cut() helper:
  // 3. Create the optimization algorithm. This requires a cost function,
  //    and the maximum number of iterations.
  // 4. Call the minimizer. "cut" will in the end contain the set of cut edges.
  // 5. Get the node labels.
  std::vector<unsigned int> cut;
  std::vector<bool> node_labels;
  double cost = coop_cut(cut, node_labels, im_graph, lambda, theta, max_iter);

  printf("cut has %lu edges\n", cut.size());
  printf("cut has total cost %f\n", cost);

  // Show the image.
  uchar* data = reinterpret_cast<uchar*>(img.data);
  int channels = img.channels(), step = img.step;
  int nodeid;

  for (int y = 0; y < im_graph->h; ++y) {
    for (int x = 0; x < im_graph->w; ++x) {
      nodeid = y*im_graph->w + x;
      if (!node_labels[nodeid]) {
        for (int k = 0; k < 3; ++k)
          data[y*step + x*channels + k] = 255;
      }
    }
  }
  cv::namedWindow("mainWin", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("mainWin", 100, 100);
  cv::imshow("mainWin", img);
  cv::waitKey(0);

  // Cleanup
  delete im_graph;

  return 0;
}
