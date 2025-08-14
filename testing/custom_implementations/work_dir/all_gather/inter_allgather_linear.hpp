#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

//#define DEBUG


int inter_allgather_linear(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int b);
