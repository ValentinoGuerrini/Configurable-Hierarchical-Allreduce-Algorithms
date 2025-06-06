# Makefile for building, running, and visualizing “0reduce scatter_beta1.cpp”
#
# Usage:
#   make             # Builds the MPI executable “reduce_scatter”
#   make run         # Runs MPI with default NP=4, R=2, B=7
#   make visualize   # Launches the Python visualization on “all_buffers.txt”
#   make full        # Builds → Runs → Visualizes in one step
#   make clean       # Removes the executable and the output text file
#   make help        # Shows this help message
#
# You can override the default variables NP, R, and B on the command line. For example:
#   make run NP=8 R=3 B=5
#   make full NP=8 R=3 B=5

#===========================
# USER-EDITABLE DEFINITIONS
#===========================

# MPI C++ compiler
CXX       := mpicxx
CXXFLAGS  := -std=c++11 -O3 -Wno-vla-cxx-extension

# Source file (note: escape the space with a backslash)
SRC       := reduce_scatter_beta1.cpp

# Name of the final executable
TARGET    := reduce_scatter

# Default values (can be overridden on the `make` command line)
NP  ?= 4       # Number of MPI processes
R   ?= 2       # First argument to reduce_scatter (r)
B   ?= 1       # Second argument to reduce_scatter (b)

# Python interpreter and visualization script
PY         := python3
PY_SCRIPT  := visualize_buffers.py

# The output file that contains all send + recv buffers
TXT        := all_buffers.txt

#===========================
# PHONY TARGETS
#===========================
.PHONY: all run visualize full clean help

#===========================
# Default target: build only
#===========================
all: $(TARGET)

#===========================
# Compile the MPI program
#===========================
# Note: Each command here must be preceded by a real tab character.
$(TARGET): $(SRC)
	@echo "Compiling $(SRC) → $(TARGET)"
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)
	@echo "Compilation complete."

#===========================
# Run the MPI program
#===========================
# Produces “all_buffers.txt” in the current directory.
# Override NP, R, B as needed:
#   make run NP=8 R=3 B=5
run: $(TARGET)
	@echo "Running with NP=$(NP), R=$(R), B=$(B) → generating $(TXT)"
	mpirun -np $(NP) ./$(TARGET) $(R) $(B)
	@echo "Run complete. Output in $(TXT)."

#===========================
# Visualize the output
#===========================
# Assumes $(TXT) already exists.
visualize:
	@echo "Launching visualization on $(TXT)..."
	@$(PY) $(PY_SCRIPT) $(TXT)

#===========================
# Full pipeline: build → run → visualize
#===========================
full: all run visualize

#===========================
# Clean up generated files
#===========================
clean:
	@echo "Cleaning up: removing $(TARGET) and $(TXT)"
	@rm -f $(TARGET) $(TXT)

#===========================
# Help message
#===========================
help:
	@echo ""
	@echo "=========================================="
	@echo "  Makefile help for reduce_scatter project"
	@echo "=========================================="
	@echo ""
	@echo "Targets:"
	@echo "  all          Build the MPI executable ($(TARGET))"
	@echo "  run          Run the executable via mpirun"
	@echo "  visualize    Launch the Python visualization"
	@echo "  full         Build → Run → Visualize (all-in-one)"
	@echo "  clean        Remove executable and output file ($(TXT))"
	@echo "  help         Show this message"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  NP           Number of MPI processes (default: $(NP))"
	@echo "  R            First argument to reduce_scatter (default: $(R))"
	@echo "  B            Second argument to reduce_scatter (default: $(B))"
	@echo ""
	@echo "Examples:"
	@echo "  make                         # Build the executable"
	@echo "  make run                     # Run with NP=4, R=2, B=7"
	@echo "  make run NP=6 R=3 B=5        # Run with 6 procs, r=3, b=5"
	@echo "  make visualize               # Visualize existing all_buffers.txt"
	@echo "  make full NP=8 R=3 B=5       # Build, run (8 procs, r=3, b=5), and visualize"
	@echo "  make clean                   # Remove binaries and output files"
	@echo ""
	@echo "=========================================="
	@echo ""