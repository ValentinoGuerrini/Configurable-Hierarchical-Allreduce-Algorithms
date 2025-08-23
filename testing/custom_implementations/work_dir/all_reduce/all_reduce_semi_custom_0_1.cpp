#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstring>   // for std::memcpy / memcpy


int allgather_radix_batch(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b);




int all_reduce_semi_radix_batch(char *sendbuf, char *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                int k, int b)
{
    (void)k; (void)b;  // unused in pure-MPI version

    int rank, nprocs, typesize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typesize);

    if (count < 0) return MPI_ERR_COUNT;
    if (count == 0) return MPI_SUCCESS;

    // Require equal partition
    if (count % nprocs != 0) {
        return MPI_ERR_COUNT;  // or handle with MPI_Reduce_scatter + MPI_Allgatherv
    }

    const int chunk = count / nprocs;
    char *tmp = (char*)malloc((size_t)chunk * (size_t)typesize);
    if (!tmp) return MPI_ERR_NO_MEM;

    int rc;

    // 1) Reduce-Scatter (equal counts)
    rc = MPI_Reduce_scatter_block(sendbuf, tmp, chunk, datatype, op, comm);
    if (rc != MPI_SUCCESS) { free(tmp); return rc; }

    // 2) Allgather (equal counts) to assemble the full vector
    rc = allgather_radix_batch(tmp, chunk, datatype,recvbuf,comm,k,b);

    free(tmp);
    return rc;
}
#ifdef DEBUG
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Example: each process has an array of length count */
    int count = 400;  // must be divisible by nprocs
    int *sendbuf = (int*)malloc(count * sizeof(int));
    int *recvbuf_custom = (int*)malloc(count * sizeof(int));
    int *recvbuf_mpi = (int*)malloc(count * sizeof(int));

    // Initialize input: each element = rank
    for (int i = 0; i < count; i++) {
        sendbuf[i] = rank + 1; // so reduction result is predictable
    }

    // Run custom allreduce
    int rc1 = all_reduce_semi_radix_batch((char*)sendbuf, (char*)recvbuf_custom,
                                          count, MPI_INT, MPI_SUM,
                                          MPI_COMM_WORLD, 3, 4);

    // Run MPI_Allreduce for reference
    int rc2 = MPI_Allreduce(sendbuf, recvbuf_mpi, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Check correctness
    int ok = 1;
    if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
        ok = 0;
    } else {
        for (int i = 0; i < count; i++) {
            if (recvbuf_custom[i] != recvbuf_mpi[i]) {
                ok = 0;
                break;
            }
        }
    }

    if (rank == 0) {
        if (ok) {
            printf("✅ Custom all_reduce_semi_radix_batch matches MPI_Allreduce (k=2, b=3)\n");
        } else {
            printf("❌ Mismatch between custom allreduce and MPI_Allreduce!\n");
        }
    }

    free(sendbuf);
    free(recvbuf_custom);
    free(recvbuf_mpi);

    MPI_Finalize();
    return 0;
}

#endif