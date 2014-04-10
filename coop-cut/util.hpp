// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#ifndef COOP_CUT_UTIL_H_
#define COOP_CUT_UTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include "image_graph.hpp"

// Read unaries from a binary file.
void coop_read_unaries(ImageGraph* im_graph, const char* unaryfile);

// Read edge weights from an image file.
void coop_read_weights(ImageGraph* im_graph, const char* imagename);

// Read the edge classes from a binary file.
void coop_read_classes(ImageGraph* im_graph, const char* classfile,
    double modthresh = 0.0000000001);

#endif  // COOP_CUT_UTIL_H_
