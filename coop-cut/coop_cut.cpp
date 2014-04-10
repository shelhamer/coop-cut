// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#include <vector>
#include "coop_cut.hpp"
#include "costfun.hpp"
#include "image_graph.hpp"
#include "itBM.hpp"

// This is the reference implementation of cooperative cut from
//
// S. Jegelka and J. Bilmes. "Submodularity beyond submodular energies:
// coupling edges in graph cuts". IEEE Conference on Computer Vision and
// Pattern Recognition (CVPR), 2011.
//
// Please cite this paper if you use this library in any form.

// Infer cooperative cut segmentation, return cost, assign nodes by
// the minimum energy solution.
double coop_cut(std::vector<unsigned int>& cut, std::vector<bool>& node_labels,
    ImageGraph* im_graph, double lambda, double theta, int max_iter) {
  // Create a cost function and set the edge regularization weight lambda.
  // Set the potentials from the unaries, edge weights, and edge classes.
  Costfun* cost = new Costfun();
  cost->setLambda(lambda);
  cost->setPotentials(im_graph, theta);

  // Configure the optimization, minimize.
  ItBM* algo = new ItBM(cost, max_iter);
  double cut_cost = algo->minimize(cut);
  node_labels = algo->get_node_labels();

  delete algo;
  delete cost;
  return cut_cost;
}
