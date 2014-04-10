// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#ifndef COOP_CUT_COSTFUN_H_
#define COOP_CUT_COSTFUN_H_

#include <vector>
#include <list>
#include "maxflow/graph.h"
#include "image_graph.hpp"

// Coop. Cut energies are double precision throughout.
typedef Graph<double, double, double> FlowGraph;

// Pick square root for our concave, nondecreasing sum-of-edges function
// in the paper: g()
double gsqrt(double x);

// Hold the random field graph, update edge weights according to the bound, and
// compute the cost of the cut.
class Costfun {
 public:
  Costfun(): n(0), m(0), num_classes(0), myGraph(NULL), lim(0), submodlim(0),
             lambda(1.0), scost(0), unary_cost(0), s_uptodate(false),
             cost_uptodate(false) {}
  ~Costfun() {
    delete myGraph;
  }

  // Segment: does maxflow, finds cut set, and determines submodular cost.
  double segment();

  // the algorithm will iteratively run maxflow and use S as new basis set
  double run_maxflow();

  // identifies the currently cut edges from maxflow results
  void find_cut_set();

  // Submodular cost of the current cut current_S
  // (also computes current class weight sums and unary cost)
  double submod_cut_cost();

  // Computes sum for each class and whether the bound is satisfied,
  // then updates edge weights accordingly to rho_e(E-e) or rho_e(S).
  // Edge indices must refer to internal sorted list elist (sorted by class)
  void update_wts(const std::vector<unsigned int>& S);
  // Reset weights (as above) w.r.t. current reference set
  void update_wts();
  // Reset weights (as above) w.r.t. global communal potentials
  void update_wts(std::vector<double> cut_sums, std::vector<double> full_sums,
      std::vector<double> full_sums_discounted,
      std::vector<double> full_threshs);

  void get_cut(std::vector<unsigned int>& cut);
  void get_original_indices(std::vector<unsigned int>& cut);
  std::vector<bool> get_node_labels(FlowGraph::termtype default_label);

  // Getters for aggregation in communal potentials across local MRFs:
  // Total class sums, and class sums and unary cost of the current cut.
  double get_class_total(int class_ix);
  double get_class_cut(int class_ix);
  double get_unary_cost();

  // Set the potentials / create the graph for the random field.
  // discount_thresh: threshold on proportion of weight at which submodular
  // discounting kicks in. \vartheta "script theta" in paper.
  void setPotentials(ImageGraph* im_graph, double discount_thresh);

  // Set the regularization (edge energy) weight.
  void setLambda(double lambdaIn);

 private:
  unsigned int n, m;
  int num_classes;
  struct edgeInf;
  FlowGraph* myGraph;
  std::vector<edgeInf> elist;
  std::vector<unsigned int> csizes;  // class sizes
  int lim;
  unsigned int submodlim;  // pointer to first modular edge (class > lim)

  // Regularization (edge energy) weight
  // (in practice, applied as 1/lambda to terminal weights instead)
  double lambda;

  // Marginal costs and thresholds for discounting.
  // in the paper: \rho(E \setminus e) and \theta_s
  std::vector<double> rhoEe;
  std::vector<double> threshs;

  // Current cut set and total submodular cost.
  std::vector<unsigned int> current_S;
  double scost;

  // Current class sums along cut and current unary cost.
  std::vector<double> cut_class_sums;
  double unary_cost;

  // unary source/sink weights (for resetting capacities across iterations)
  std::vector<double> termweights;

  // The total weights and total post-threshold discounted weights for classes.
  // in the paper: f_S(\Gamma)
  std::vector<double> class_sums;
  std::vector<double> class_sums_discounted;

  // bookkeeping
  std::vector<unsigned int> sortindex;
  bool s_uptodate, cost_uptodate;

  struct edgeInf {
    FlowGraph::arc_id edgep;
    unsigned int eindex;
    double weight;
    int classno;
  };

  // Compare edges by class for sorting.
  friend bool comp(const Costfun::edgeInf&, const Costfun::edgeInf&);
};


#endif  // COOP_CUT_COSTFUN_H_
