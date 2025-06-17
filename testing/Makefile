# Configurable parameters
RANKS_PER_NODE ?= 9
NUM_NODES      ?= 1
DEBUG          ?= 0
OVERWRITE      ?= 0       # 0 = append to results.csv, 1 = overwrite
N_ITER         ?= 10      # Number of iterations (powers of two)
TOTAL_PROCS    = $(shell expr $(RANKS_PER_NODE) \* $(NUM_NODES))


ifndef SYSTEM_ENVIRONMENT
  $(error "SYSTEM_ENVIRONMENT is undefined.  Run `source setup.sh` first.")
endif

ENV := $(shell echo "$(SYSTEM_ENVIRONMENT)" | tr -d '[:space:]' | tr a-z A-Z)

ifeq ($(ENV),POLARIS)

	PBS_SCRIPT       := runner_polaris.sh

else ifeq ($(ENV),AURORA)

	PBS_SCRIPT       := runner_aurora.sh
endif

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
IMPL_SRCS = mpich_implementations/allreduce_k_reduce_scatter_allgather.cpp \
            mpich_implementations/allreduce_recursive_doubling.cpp          \
            mpich_implementations/allreduce_recursive_multiplying.cpp       \
            mpich_implementations/allreduce_reduce_scatter_allgather.cpp    \
            mpich_implementations/allreduce_ring.cpp                        \
            mpich_implementations/allreduce_inter_reduce_exchange_bcast.cpp \
            mpich_implementations/allreduce_recexch.cpp

IMPL_OBJS = $(IMPL_SRCS:.cpp=.o)

# Main benchmark executable
MAIN_EXEC = benchmark

# -------------------------------------------------------------------
# Default target: build the benchmark
# -------------------------------------------------------------------
.PHONY: all
all: $(MAIN_EXEC)

# Compile each implementation/source file into an object
%.o: mpich_implementations/%.cpp
	$(CXX) $(CXXFLAGS) -I mpich_implementations -c $< -o $@

# Link the benchmark executable
$(MAIN_EXEC): main.cpp $(IMPL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -I mpich_implementations -o $@

# -------------------------------------------------------------------
# Compute the overwrite flag at parse time (no leading tab!)
# -------------------------------------------------------------------
ifeq ($(OVERWRITE),1)
OVERWRITE_FLAG := --overwrite
else
OVERWRITE_FLAG :=
endif

# -------------------------------------------------------------------
# Run the full benchmark
# -------------------------------------------------------------------
.PHONY: run-benchmark
run-benchmark: $(MAIN_EXEC)
ifeq ($(ENV),LOCAL)
	@echo "Running on LOCAL: mpirun -np $(TOTAL_PROCS) ./$(MAIN_EXEC) $(N_ITER) $(OVERWRITE_FLAG)"
	mpirun -np $(TOTAL_PROCS) ./$(MAIN_EXEC) $(N_ITER) $(OVERWRITE_FLAG)

else ifeq ($(ENV),POLARIS)
	@echo "Submitting to POLARIS via PBS:"
	@echo "  Nodes:          $(NUM_NODES)"
	@echo "  Ranks per node: $(RANKS_PER_NODE)"
	@echo "  Total ranks:    $(TOTAL_PROCS)"
	@echo "  Executable:     $(MAIN_EXEC)"
	@echo

	# Export so runner_polaris.sh can read them
	@export EXECUTABLE=$(MAIN_EXEC); \
	 export RPN=$(RANKS_PER_NODE); \
	 export NN=$(NUM_NODES); \
	 export NI=$(N_ITER); \
	 export OF=$(OVERWRITE_FLAG); \
	 echo "qsub $(PBS_SCRIPT)"; \
	qsub -l select=$(NUM_NODES):system=polaris $(PBS_SCRIPT)

else
	$(error Unsupported SYSTEM_ENVIRONMENT='$(SYSTEM_ENVIRONMENT)' (must be LOCAL or POLARIS))
endif

# -------------------------------------------------------------------
# Debug a single implementation (build+run with its own main)
# -------------------------------------------------------------------
.PHONY: debug-%
debug-%: mpich_implementations/%.cpp
	$(CXX) $(CXXFLAGS) -g -DDEBUG_MODE -DRANKS_PER_NODE=$(RANKS_PER_NODE) \
	        $< -I mpich_implementations -o $@
	mpirun -np $(TOTAL_PROCS) ./\$@

# -------------------------------------------------------------------
# Clean up all generated files
# -------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f $(MAIN_EXEC) debug-* $(IMPL_OBJS) results.csv

# -------------------------------------------------------------------
# Help message
# -------------------------------------------------------------------
.PHONY: help
help:
	@echo "MPI Implementations Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [RANKS_PER_NODE=X] [NUM_NODES=Y] [DEBUG=0|1] [OVERWRITE=0|1] [N_ITER=K]"
	@echo ""
	@echo "Targets:"
	@echo "  all                 - Build the full benchmark executable (default)"
	@echo "  run-benchmark       - Run the full benchmark across all implementations"
	@echo "                         (appends by default; use OVERWRITE=1 to rewrite results.csv)"
	@echo "  debug-<impl>        - Build & run a single implementation in debug mode"
	@echo "                         (e.g. make debug-allreduce_ring)"
	@echo "  clean               - Remove executables, objects, and results"
	@echo "  help                - Display this help message"
	@echo ""
	@echo "Implementations available for debug:"
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
	@echo "  OVERWRITE           - Overwrite results.csv (0=append,1=overwrite)"
	@echo "  N_ITER              - Number of iterations (powers of two, default: 10)"
	@echo ""
	@echo "Examples:"
	@echo "  make run-benchmark RANKS_PER_NODE=4 NUM_NODES=2 N_ITER=15"
	@echo "  make run-benchmark OVERWRITE=1"
	@echo "  make debug-allreduce_ring RANKS_PER_NODE=2 NUM_NODES=1"
	@echo "  make DEBUG=1"