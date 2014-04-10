CC := g++
CFLAGS := -Wno-write-strings -Wall
LIBS := $(shell pkg-config --cflags --libs opencv)
MAXFLOW_DIR := maxflow

# on OSX the bundled g++ is confused by the maxflow templating;
# compile with clang++ instead
ifeq ($(shell uname -s), Darwin)
	CC := clang++
endif

# Debugging
#CFLAGS += -DDEBUG -g -O0

.PHONY: clean lint

all: coopcut

# TODO build the library and the example separately
coopcut: call.cpp coop_cut.cpp image_graph.cpp costfun.cpp util.cpp
	$(CC) call.cpp coop_cut.cpp image_graph.cpp costfun.cpp util.cpp \
		$(MAXFLOW_DIR)/graph.cpp $(MAXFLOW_DIR)/maxflow.cpp itBM.cpp -o coopcut \
		$(LIBS) $(CFLAGS)

clean:
	@- $(RM) -f coopcut

lint:
	@../scripts/cpp_lint.py *.cpp *.hpp
