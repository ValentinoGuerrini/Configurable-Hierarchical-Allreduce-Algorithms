#include <mpi.h>
#include <iostream>
#include <cstring>   // for memcpy, memmove, etc.
#include <vector>
#include <fstream>
#include <cstdlib>
#include <climits>

//#define DEBUG_MODE

int inter_reduce_linear(const void *sendbuf, void *recvbuf,
                        MPI_Aint recvcount, MPI_Datatype datatype,
                        MPI_Op op, MPI_Comm comm, int b)
{
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Basic sanity
    if (nranks % b != 0) return MPI_ERR_OTHER;
    int nnodes = nranks / b;
    int node_id   = rank / b;    // 0..nnodes-1
    int node_rank = rank % b;    // 0..b-1

    int type_size = 0;
    MPI_Type_size(datatype, &type_size);

    // Each (i) is a chunk reduced across nodes within a lane (node_rank)
    // Total elements per chunk = intra_recvcount = recvcount * b
    // NOTE: many MPI impls require 'int' count; check for overflow
    MPI_Aint intra_recvcount_aint = recvcount * (MPI_Aint)b;

    if (intra_recvcount_aint > INT_MAX) return MPI_ERR_COUNT;
    int intra_recvcount = (int)intra_recvcount_aint;

    size_t chunk_bytes = (size_t)intra_recvcount * (size_t)type_size;

    // temp buffer used by roots to receive one message at a time
    void *tmp = malloc(chunk_bytes);
    if (!tmp) return MPI_ERR_NO_MEM;

    int nIters = (nnodes % b == 0) ? (nnodes / b) : 1 + (nnodes / b); // number of rotating roots (i)
    for (int i = 0; i < nIters; ++i) {
        int root_node = i * b + node_rank;        // 0..nnodes-1 (lane root for this i)
        if (root_node >= nnodes) continue;        // guard (shouldn’t happen if nnodes % b == 0)
        int root_rank = root_node * b + node_rank;

        const char *my_chunk = (const char*)sendbuf + (size_t)i * chunk_bytes;

        if (node_id == root_node) {
            // Initialize accumulator with local contribution
            memcpy(recvbuf, my_chunk, chunk_bytes);

            // Receive from every other node in my lane and accumulate
            for (int j = 0; j < nnodes; ++j) {
                if (j == node_id) continue;
                int src_rank = j * b + node_rank;
                MPI_Recv(tmp, intra_recvcount, datatype, src_rank, i, comm, MPI_STATUS_IGNORE);
                // tmp (inbuf) reduced into recvbuf (inoutbuf)
                MPI_Reduce_local(tmp, recvbuf, intra_recvcount, datatype, op);
            }
        } else {
            // Non-root sends its lane-chunk to the root of this iteration
            MPI_Send(my_chunk, intra_recvcount, datatype, root_rank, i, comm);
        }
    }

    free(tmp);
    return MPI_SUCCESS;
}



#ifdef DEBUG_MODE
static void print_buf(const char *label, int rank, const int *buf,
                      int total_elems, int intra_recvcount, int nIters)
{
    printf("[rank %d] %s\n", rank, label);
    for (int i = 0; i < nIters; ++i) {
        printf("  chunk i=%d: ", i);
        for (int j = 0; j < intra_recvcount; ++j) {
            int v = buf[i * intra_recvcount + j];
            printf("%d%s", v, (j + 1 == intra_recvcount ? "" : " "));
        }
        printf("\n");
    }
    fflush(stdout);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = -1, nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Params: b (#ranks per node lane) and recvcount (per-rank contribution)
    int b = (argc >= 2) ? atoi(argv[1]) : 2;
    int recvcount = (argc >= 3) ? atoi(argv[2]) : 1;  // per the function signature semantics

    if (b <= 0 || recvcount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <P> %s <b> <recvcount>\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 8 %s 2 3\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (nranks % b != 0) {
        if (rank == 0) {
            fprintf(stderr, "nranks (%d) must be a multiple of b (%d)\n", nranks, b);
        }
        MPI_Finalize();
        return 1;
    }

    const int nnodes = nranks / b;            // nodes
    const int node_id = rank / b;             // 0..nnodes-1
    const int node_rank = rank % b;           // 0..b-1
    const int nIters = (nnodes % b == 0) ? (nnodes / b) : 1 + (nnodes / b); // conservative guard
    const int intra_recvcount = recvcount * b;
    const int total_elems = nIters * intra_recvcount;

    // Allocate send/recv buffers
    int *sendbuf = (int*)malloc((size_t)total_elems * sizeof(int));
    int *recvbuf = (int*)malloc((size_t)total_elems * sizeof(int));
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "[rank %d] allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fill sendbuf with recognizable pattern:
    // value = 10000*rank + 100*i (chunk) + j (elem within chunk)
    for (int i = 0; i < nIters; ++i) {
        for (int j = 0; j < intra_recvcount; ++j) {
            sendbuf[i * intra_recvcount + j] = 10000 * rank + 100 * i + j;
        }
    }

    // Initialize recvbuf to a sentinel
    for (int t = 0; t < total_elems; ++t) recvbuf[t] = -777777;

    // Print BEFORE (rank-ordered)
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) {
            printf("== BEFORE rank %d (nranks=%d, b=%d, nnodes=%d, nIters=%d, intra_recvcount=%d) ==\n",
                   rank, nranks, b, nnodes, nIters, intra_recvcount);
            print_buf("sendbuf", rank, sendbuf, total_elems, intra_recvcount, nIters);
            print_buf("recvbuf", rank, recvbuf, total_elems, intra_recvcount, nIters);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Call your function (using int datatype and MPI_SUM as example op)
    int rc = inter_reduce_linear((const void*)sendbuf, (void*)recvbuf,
                                 (MPI_Aint)recvcount, MPI_INT, MPI_SUM,
                                 MPI_COMM_WORLD, b);

    MPI_Barrier(MPI_COMM_WORLD);

    // Print AFTER (rank-ordered)
    for (int p = 0; p < nranks; ++p) {
        if (p == rank) {
            printf("== AFTER  rank %d (rc=%d) ==\n", rank, rc);
            print_buf("recvbuf", rank, recvbuf, total_elems, intra_recvcount, nIters);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}

#endif