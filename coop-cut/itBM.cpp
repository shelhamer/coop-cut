// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.

#include <sys/stat.h>
#include <cstdio>
#include <iostream>  // NOLINT(readability/streams)
#include <stdexcept>
#include <vector>
#include "itBM.hpp"
#include "costfun.hpp"

// Minimize the cooperative cost function
// and list the resulting cut edges in  `cut` according to the edge class order.
// (If sort_like_input == true, the edge list refers to the input order instead
// of the edge class order.)
double ItBM::minimize(std::vector<unsigned int>& cut, bool sort_like_input) {
  double submod_min, new_min;
  int iter = 0;

  // Do maxflow for initial solution, then set the submodular cost higher
  // to kickstart the iterative optimization.
  new_min = F->segment();
  submod_min = new_min + 1;

  // Re-establish the approximation at the current minimum and re-optimize
  // until energy doesn't improve or max iterations are reached.
  while ((new_min < submod_min) && (iter < max_iter)) {
    submod_min = new_min;
    printf("current min: %1.3e ", submod_min);

    F->update_wts();
    new_min = F->segment();

    printf("new_min = %1.3e\n", new_min);
    iter++;
  }

  F->get_cut(cut);
  if (sort_like_input) {
    F->get_original_indices(cut);
  }
  return submod_min;
}


std::vector<bool> ItBM::get_node_labels() {
  return F->get_node_labels(FlowGraph::SOURCE);
}
