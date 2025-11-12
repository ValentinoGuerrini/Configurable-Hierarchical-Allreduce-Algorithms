#!/bin/bash -l
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home_fs
#PBS -q debug
#PBS -A UIC-HPC
cd ${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=64 # Number of MPI ranks to spawn per node
NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

# Change the directory to work directory, which is the directory you submit the job.
cd $PBS_O_WORKDIR
mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -env OMP_NUM_THREADS=${NTHREADS} ./out 13 --overwrite b=8 base=1 num_nodes=${NNODES} radix_increment=1