// Copyright 2014 Stefanie Jegelka and Evan Shelhamer.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include "util.hpp"
#include "image_graph.hpp"

// Read unaries from file with both source and sink (in that order).
void coop_read_unaries(ImageGraph* im_graph, const char* unaryfile) {
  FILE* fp = fopen(unaryfile, "r");
  double* source_unaries = new double[im_graph->n];
  double* sink_unaries = new double[im_graph->n];
  fread(source_unaries, sizeof(double), im_graph->n, fp);
  fread(sink_unaries, sizeof(double), im_graph->n, fp);
  fclose(fp);
  im_graph->setUnaries(source_unaries, sink_unaries);
  delete[] source_unaries;
  delete[] sink_unaries;
}


// Create edge weights from an image file.
void coop_read_weights(ImageGraph* im_graph, const char* imagename) {
  // Load image in opencv, then convert to double array.
  // (I suspect there is a much better way to do this.)
  cv::Mat img = cv::imread(imagename, CV_LOAD_IMAGE_COLOR);
  if (!img.data) { printf("Could not load image file!\n"); }

  double* arr = new double[img.rows * img.cols * img.channels()];
  int i, j, ix;
  for (i = 0; i < img.rows; i++) {
    for (j = 0; j < img.cols; j++) {
      ix = i*img.step + j*img.channels();
      // RGB order from opencv's native BGR
      arr[ix] = img.data[ix + 2];
      arr[ix + 1] = img.data[ix + 1];
      arr[ix + 2] = img.data[ix];
    }
  }

  im_graph->extractEdges(arr);
}


// Read in the edge classes from a binary file
// if you do not want modular treatment, then set modthresh < 0
// classfile should be of the form:
// num_classes
// int array of edge classes, length 2*m
void coop_read_classes(ImageGraph* im_graph, const char* classfile,
    double modthresh) {
  FILE* fp = fopen(classfile, "r");
  int num_classes;
  int* edge_classes = new int[2*im_graph->m];
  fread(&num_classes, sizeof(int), 1, fp);
  fread(edge_classes, sizeof(int), 2*im_graph->m, fp);
  fclose(fp);
  im_graph->setClasses(edge_classes, num_classes, modthresh);
  delete[] edge_classes;
}
