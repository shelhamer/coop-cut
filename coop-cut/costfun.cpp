// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.

#include <math.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <iostream>  // NOLINT(readability/streams)
#include <numeric>
#include "maxflow/graph.h"
#include "costfun.hpp"
#include "image_graph.hpp"

// TODO replace maxflow/find_cut_set/etc. with segment() alone
// TODO perhaps drop the uptodate caching; it can be done outside
// TODO inheritance for graph/coop/comm cut variations

// sqrt for concave, nondecreasing sum-of-edges function
// in the paper: g()
double gsqrt(double x) {
  // TODO double-check this is never given negative input
  return (sqrt(x + 1) - 1);
}


// Compare edges by class for sorting.
bool comp(const Costfun::edgeInf& a, const Costfun::edgeInf& b) {
  return a.classno < b.classno;
}


double Costfun::segment(){
  run_maxflow();
  find_cut_set();
  return submod_cut_cost();
}


double Costfun::run_maxflow() {
  s_uptodate = false;
  cost_uptodate = false;
  return myGraph->maxflow();
}


void Costfun::find_cut_set() {
  if (s_uptodate) {
    return;
  }

  current_S = std::vector<unsigned int>(0);
  unsigned int i;
  double rcap = 0;
  FlowGraph::node_id head, tail;

  for (i = 0; i < elist.size(); i++) {
    rcap = myGraph->get_rcap(elist[i].edgep);
    if (rcap == 0) {
      // get node capacities
      myGraph->get_arc_ends(elist[i].edgep, head, tail);
      if (myGraph->what_segment(head, FlowGraph::SOURCE) == FlowGraph::SOURCE &&
          myGraph->what_segment(tail, FlowGraph::SOURCE) == FlowGraph::SINK) {
        current_S.push_back(i);
      }
    }
  }
  s_uptodate = true;
}


void Costfun::update_wts(const std::vector<unsigned int>& S) {
  // relies on a call to submod_cut_cost() first for class weight sums
  // assume S is ordered as elist, and entries in S are indices in elist
  unsigned int i;
  int cl;
  double tmp;
  unsigned int lastone = S.size();

  // Marginal gains for submodular bound (excluding the modular n+1th class).
  for (i = 0; i < S.size(); i++) {
    cl = elist[S[i]].classno;
    if (cl >= lim) {
      lastone = i;
      break;
    }

    // set the rhoe if that hasn't happened yet
    if (rhoEe[S[i]] < 0) {
      tmp = (class_sums[cl] - elist[S[i]].weight);  // weight of the other edges
      if (tmp < threshs[cl]) {
        rhoEe[S[i]] = class_sums_discounted[cl] - tmp;
      } else {
        rhoEe[S[i]] = class_sums_discounted[cl] - (threshs[cl] +
                                        gsqrt(tmp - threshs[cl]));
      }
    }
  }

  // now we know class weights
  // so go through elist and re-set weights
  unsigned int counter = 0;
  for (unsigned int iC = 0; iC < (csizes.size()-1); iC++) {
    // is class saturated?
    if (cut_class_sums[iC] >= threshs[iC]) {
      double base1 = gsqrt(cut_class_sums[iC] - threshs[iC]);
      double base2 = cut_class_sums[iC] - threshs[iC];
      for (i = 0; i < csizes[iC]; i++) {
        tmp = gsqrt(elist[counter + i].weight + base2) - base1;
        myGraph->set_rcap(elist[i + counter].edgep, tmp);
      }
    } else {
      double miss = threshs[iC] - cut_class_sums[iC];
      for (i = 0; i < csizes[iC]; i++) {
        if (elist[counter + i].weight <= miss) {
          tmp = elist[counter + i].weight;
        } else {
          tmp = miss + gsqrt(elist[counter + i].weight - miss);
        }
        myGraph->set_rcap(elist[i + counter].edgep, tmp);
      }
    }
    counter += csizes[iC];
  }

  for (i = 0; i < lastone; i++) {
    myGraph->set_rcap(elist[S[i]].edgep, rhoEe[S[i]]);
  }

  // re-set modular and terminal edges
  for (i = submodlim; i < elist.size(); i++) {
    myGraph->set_rcap(elist[i].edgep, elist[i].weight);
  }
  double tg = 1/lambda;
  for (i = 0; i < n; i++) {
    myGraph->set_trcap(i, tg*termweights[i]);
  }
}


void Costfun::update_wts() {
  update_wts(current_S);
}


void Costfun::update_wts(std::vector<double> cut_sums,
    std::vector<double> full_sums, std::vector<double> full_sums_discounted,
    std::vector<double> full_threshs) {
  // Adopt global cut sums, class sums, and thresholds for communal potentials.
  cut_class_sums = cut_sums;
  class_sums = full_sums;
  class_sums_discounted = full_sums_discounted;
  threshs = full_threshs;
  // Update weights according to communal potentials
  // w.r.t. current reference set in cut.
  update_wts(current_S);
}


double Costfun::submod_cut_cost() {
  if (cost_uptodate) { return scost; }

  if (~s_uptodate) {
    find_cut_set();
  }

  // Sum class weights of the current cut.
  std::fill(cut_class_sums.begin(), cut_class_sums.end(), 0.0);
  for (unsigned int i = 0; i < current_S.size(); i++) {
    cut_class_sums[elist[current_S[i]].classno] += elist[current_S[i]].weight;
  }

  scost = 0;
  for (int i = 0; i < num_classes; i++) {
    if (cut_class_sums[i] < threshs[i]) {
      scost += cut_class_sums[i];
    } else {
      scost += threshs[i] + gsqrt(cut_class_sums[i] - threshs[i]);
    }
  }
  scost += cut_class_sums[num_classes];

  // Sum current unary cost.
  unary_cost = 0.0;
  double tg = 1 / lambda;
  for (unsigned int i = 0; i < n; i++) {
    if (myGraph->what_segment(i, FlowGraph::SOURCE) == FlowGraph::SOURCE) {
      // source, cut sink
      unary_cost += ((termweights[i] > 0) ? 0.0 : (-tg*termweights[i]));
    } else {
      // sink, cut source
      unary_cost += ((termweights[i] > 0) ? tg*termweights[i] : 0.0);
    }
  }
  scost += unary_cost;

  cost_uptodate = true;
  return scost;
}


void Costfun::get_cut(std::vector<unsigned int>& cut) {
  if (~s_uptodate) {
    find_cut_set();
  }
  cut = current_S;
}


// get indices like in the input heads/tails,
// and not like the sorted edge list
void Costfun::get_original_indices(std::vector<unsigned int>& cut) {
  for (unsigned int i = 0; i < cut.size(); i++) {
    cut[i] = (elist[i]).eindex;
  }
}


std::vector<bool> Costfun::get_node_labels(FlowGraph::termtype default_label) {
  std::vector<bool> node_labels(n);
  for (unsigned int i = 0; i < n; i++) {
    node_labels[i] = (myGraph->what_segment(i, default_label)
                      == FlowGraph::SOURCE);
  }
  return node_labels;
}


double Costfun::get_class_total(int class_ix) {
  return class_sums[class_ix];
}


double Costfun::get_class_cut(int class_ix) {
  return cut_class_sums[class_ix];
}


double Costfun::get_unary_cost() {
  return unary_cost;
}


void Costfun::setPotentials(ImageGraph* im_graph, double discount_thresh) {
  n = im_graph->n;
  m = im_graph->m;
  num_classes = im_graph->num_classes;

  csizes = std::vector<unsigned int>(num_classes + 1, 0);
  threshs = std::vector<double>(num_classes, 0.0);
  class_sums = std::vector<double>(num_classes, 0.0);
  class_sums_discounted = std::vector<double>(num_classes, 0.0);
  cut_class_sums = std::vector<double>(num_classes + 1, 0.0);
  lim = num_classes;

  // add terminal edges
  myGraph = new FlowGraph(n+1, 2*m+1);
  myGraph->add_node(n);
  elist = std::vector<edgeInf>(2*m);

  sortindex = std::vector<unsigned int>(2*m);
  cost_uptodate = false;

  // put the weights into the termweights vector
  unsigned int i, j;
  double tg = 1/lambda;
  termweights = std::vector<double>(n);
  for (i = 0; i < n; i++) {
    termweights[i] = (im_graph->unaries[i] - im_graph->unaries[n+i]);
    if (termweights[i] > 0) {
      myGraph->add_tweights(i, tg*termweights[i], 0);
    } else {
      myGraph->add_tweights(i, 0, -tg*termweights[i]);
    }
  }

  // add interpixel edges
  std::vector<double> cmaxs(num_classes, 0);
  FlowGraph::arc_id farc = myGraph->get_first_arc();

  // add edges to graph, row-major node numbering
  // where (i, j) is i*w+j

  // horizontal
  unsigned int ct = 0;
  for (j = 0; (int)j < im_graph->w; j++) {
    for (i = 0; (int)i < (im_graph->h)-1; i++) {
      edgeInf nedge;
      myGraph->add_edge(i * im_graph->w + j, (i + 1)*im_graph->w + j,
          50.0 * im_graph->edge_weights[ct], 50.0 * im_graph->edge_weights[ct]);
      nedge.edgep = farc++;
      nedge.eindex = 2*ct;
      nedge.weight = 50.0*im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2*ct];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct] = nedge;
      csizes[nedge.classno]++;

      // sister edge
      nedge.edgep = farc++;
      nedge.eindex = 2*ct+1;
      nedge.weight = 50.0*im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2*ct + 1];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct+1] = nedge;
      csizes[nedge.classno]++;

      ct++;
    }
  }


  // vertical
  for (j = 0; (int)j < (im_graph->w)-1; j++) {
    for (i = 0; (int)i < (im_graph->h); i++) {
      edgeInf nedge;
      myGraph->add_edge(i * im_graph->w + j + 1, i * im_graph->w + j,
          50.0 * im_graph->edge_weights[ct], 50.0 * im_graph->edge_weights[ct]);
      nedge.edgep = farc++;
      nedge.eindex = 2 * ct;
      nedge.weight = 50.0 * im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2 * ct];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct] = nedge;
      csizes[nedge.classno]++;

      // sister edge
      nedge.edgep = farc++;
      nedge.eindex = 2*ct+1;
      nedge.weight = 50.0*im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2*ct + 1];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct+1] = nedge;
      csizes[nedge.classno]++;

      ct++;
    }
  }


  // right diag down
  for (j = 0; (int)j < (im_graph->w)-1; j++) {
    for (i = 0; (int)i < (im_graph->h)-1; i++) {
      edgeInf nedge;
      myGraph->add_edge(i * im_graph->w + j, (i + 1) * im_graph->w + j + 1,
          50.0 * im_graph->edge_weights[ct], 50.0 * im_graph->edge_weights[ct]);
      nedge.edgep = farc++;
      nedge.eindex = 2*ct;
      nedge.weight = 50.0*im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2*ct];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct] = nedge;
      csizes[nedge.classno]++;

      // sister edge
      nedge.edgep = farc++;
      nedge.eindex = 2*ct+1;
      nedge.weight = 50.0*im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2*ct + 1];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct+1] = nedge;
      csizes[nedge.classno]++;

      ct++;
    }
  }


  // left diag down
  for (j = 0; (int)j < (im_graph->w)-1; j++) {
    for (i = 0; (int)i < (im_graph->h)-1; i++) {
      edgeInf nedge;
      myGraph->add_edge((i + 1) * im_graph->w + j, i*im_graph->w + j + 1,
          50.0 * im_graph->edge_weights[ct], 50.0 * im_graph->edge_weights[ct]);
      nedge.edgep = farc++;
      nedge.eindex = 2 * ct;
      nedge.weight = 50.0 * im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2 * ct];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2 * ct] = nedge;
      csizes[nedge.classno]++;

      // sister edge
      nedge.edgep = farc++;
      nedge.eindex = 2*ct + 1;
      nedge.weight = 50.0 * im_graph->edge_weights[ct];
      nedge.classno = im_graph->edge_classes[2*ct + 1];
      if (nedge.classno >= lim) {
        nedge.classno = lim;
      } else {
        class_sums[nedge.classno] += nedge.weight;
        if (nedge.weight > cmaxs[nedge.classno]) {
          cmaxs[nedge.classno] = nedge.weight;
        }
      }
      elist[2*ct + 1] = nedge;
      csizes[nedge.classno]++;

      ct++;
    }
  }

  submodlim = 2*m - csizes[csizes.size() - 1];
  std::stable_sort(elist.begin(), elist.end(), ::comp);
  for (i = 0; i < 2*m; i++) {
    sortindex[elist[i].eindex] = i;
  }

  double tmp;
  for (i = 0 ; (int)i < num_classes; i++) {
    // threshold starts at discount_thresh fraction of class weight
    tmp = discount_thresh * class_sums[i];

    if (tmp >= cmaxs[i]) {
      threshs[i] = tmp;
    } else {
      threshs[i] = cmaxs[i];
    }
    class_sums_discounted[i] = threshs[i] + gsqrt(class_sums[i] - threshs[i]);
  }
  rhoEe = std::vector<double>(2*m, -1);
}


void Costfun::setLambda(double lambdaIn) {
  lambda = lambdaIn;
  if (myGraph != NULL && n > 0) {
    // re-set terminal weights
    double tg = 1/lambda;
    for (unsigned int i = 0; i < n; i++) {
      myGraph->set_trcap(i, tg*termweights[i]);
    }
    cost_uptodate = false;
  }
}
