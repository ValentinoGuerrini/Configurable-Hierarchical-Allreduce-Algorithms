# Configurable parameters
RANKS_PER_NODE ?= 3
NUM_NODES      ?= 3
DEBUG          ?= 0
TOTAL_PROCS    = $(shell expr $(RANKS_PER_NODE) \* $(NUM_NODES))

# Use MPI compiler wrappers
CC   = mpicc
CXX  = mpicxx

# Basic flags
CFLAGS   = -Wall -O2
CXXFLAGS = -Wall -O2 -std=c++11

# Debug flags (for the full benchmark build)
ifeq ($(DEBUG), 1)
	CFLAGS   += -g -DDEBUG_MODE
	CXXFLAGS += -g -DDEBUG_MODE
endif

# Always pass RANKS_PER_NODE into every compile
CFLAGS   += -DRANKS_PER_NODE=$(RANKS_PER_NODE)
CXXFLAGS += -DRANKS_PER_NODE=$(RANKS_PER_NODE)

# Implementation sources
IMPL_SRCS = allreduce_k_reduce_scatter_allgather.cpp \
            allreduce_recursive_doubling.cpp          \
            allreduce_recursive_multiplying.cpp       \
            allreduce_reduce_scatter_allgather.cpp    \
            allreduce_ring.cpp                        \
            allreduce_inter_reduce_exchange_bcast.cpp \
            allreduce_recexch.cpp

IMPL_OBJS = $(IMPL_SRCS:.cpp=.o)

# Main benchmark executable
MAIN_EXEC = benchmark

# -------------------------------------------------------------------
# Default target: build the benchmark
# -------------------------------------------------------------------
.PHONY: all
all: $(MAIN_EXEC)

# Compile each implementation/source file into an object
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the benchmark executable
$(MAIN_EXEC): main.cpp $(IMPL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# -------------------------------------------------------------------
# Run the full benchmark
# -------------------------------------------------------------------
.PHONY: run-benchmark
run-benchmark: $(MAIN_EXEC)
	mpirun -np $(TOTAL_PROCS) ./$(MAIN_EXEC) 10

# -------------------------------------------------------------------
# Debug a single implementation (build+run with its own main)
#
# Usage:
#   make debug-<impl> [RANKS_PER_NODE=X] [NUM_NODES=Y]
# Example:
#   make debug-allreduce_ring RANKS_PER_NODE=2 NUM_NODES=1
# -------------------------------------------------------------------
.PHONY: debug-%
debug-%: %.cpp
	$(CXX) $(CXXFLAGS) -g -DDEBUG_MODE -DRANKS_PER_NODE=$(RANKS_PER_NODE) \
	        $< -o $@
	mpirun -np $(TOTAL_PROCS) ./\$@

# -------------------------------------------------------------------
# Clean up all generated files
# -------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f $(MAIN_EXEC) debug-* *.o results.csv

# -------------------------------------------------------------------
# Help message
# -------------------------------------------------------------------
.PHONY: help
.PHONY: help
help:
	@echo "MPI Implementations Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [RANKS_PER_NODE=X] [NUM_NODES=Y] [DEBUG=0|1]"
	@echo ""
	@echo "Targets:"
	@echo "  all                 - Build the full benchmark executable (default)"
	@echo "  run-benchmark       - Run the full benchmark across all implementations"
	@echo "  debug-<impl>        - Build & run a single implementation in debug mode"
	@echo "                         (e.g. make debug-allreduce_ring)"
	@echo "  clean               - Remove executables, objects, and results"
	@echo "  help                - Display this help message"
	@echo ""
	@echo "Available implementations for debug:"
	@echo "  allreduce_k_reduce_scatter_allgather"
	@echo "  allreduce_recursive_doubling"
	@echo "  allreduce_recursive_multiplying"
	@echo "  allreduce_reduce_scatter_allgather"
	@echo "  allreduce_ring"
	@echo "  allreduce_inter_reduce_exchange_bcast"
	@echo "  allreduce_recexch"
	@echo ""
	@echo "Parameters:"
	@echo "  RANKS_PER_NODE      - Number of ranks per node (default: 3)"
	@echo "  NUM_NODES           - Number of nodes (default: 3)"
	@echo "  DEBUG               - Enable debug mode for full build (0=off,1=on)"
	@echo ""
	@echo "Examples:"
	@echo "  make run-benchmark RANKS_PER_NODE=4 NUM_NODES=2"
	@echo "  make debug-allreduce_ring RANKS_PER_NODE=2 NUM_NODES=1"
	@echo "  make debug-allreduce_recursive_doubling"
	@echo "  make RANKS_PER_NODE=4 NUM_NODES=2 DEBUG=1"
	@echo "  make clean"