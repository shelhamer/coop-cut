#include "graph.h"

// Handle capacities/energies as floats throughout in the wrapper,
// even though the Boykov-Kolmogorov code is more general by templating.
typedef float Capacity;
typedef Graph<Capacity, Capacity, Capacity> G;

// Redeclare nested types for Cython, with shorthand for terminal types.
typedef G::node_id NodeID;
typedef G::termtype Terminal;
const Terminal kSOURCE = G::SOURCE;
const Terminal kSINK = G::SINK;
