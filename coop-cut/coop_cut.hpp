// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
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
//

// TODO make cost function and update instead of re-creating it each call.
// TODO recombine ImageGraph, Costfun, ItBM
// TODO consider precision, and where float or int might be better
// TODO consider image input range (uint 0-255 or float 0-1?)

// This interface infers segmentations by cooperative cut.
//
// 1. prepare graph and potentials by ImageGraph
// 2. configure cooperative cost function by Costfun
// 3. infer segmentation by minimizing cooperative energy by iterative opt.

// Configure cooperative cost and solve approximate optimization to
// infer the segmentation (steps 2 & 3).
//
// take: prepared image graph, regularization weight lambda,
//       discount threshold theta, iteraction cap max_iter
// give: return cut cost, set cut and segmentation labels
double coop_cut(std::vector<unsigned int>& cut, std::vector<bool>& node_labels,
    ImageGraph* im_graph, double lambda, double theta, int max_iter = 12);
