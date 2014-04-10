// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#ifndef COOP_CUT_ITBM_H_
#define COOP_CUT_ITBM_H_

#include <vector>
#include "costfun.hpp"

// Iterative minimization of the cooperative cost function.
class ItBM {
 public:
  ItBM(Costfun* Fin, int max_iter_) : F(Fin), max_iter(max_iter_) {}

  // minimization
  double minimize(std::vector<unsigned int>& cut, bool sort_like_input = false);

  std::vector<bool> get_node_labels();

 private:
  Costfun* F;
  int max_iter;
};

#endif  // COOP_CUT_ITBM_H_
