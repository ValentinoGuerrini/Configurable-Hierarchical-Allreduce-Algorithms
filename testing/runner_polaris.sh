#!/bin/bash -l
#PBS -V
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A dist_relational_alg

cd ${PBS_O_WORKDIR}

# Compute total ranks
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=${RPN}            # ranks per node
NDEPTH=1                 # threads-per-rank depth
NTHREADS=1               # OMP threads per rank

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

# Run the executable passed via EXECUTABLE
mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth \
        -env OMP_NUM_THREADS=${NTHREADS} ./${EXECUTABLE} ${NI} ${OF}