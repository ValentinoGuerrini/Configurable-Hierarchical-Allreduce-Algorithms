#include "common.hpp"



// WORKING
int MPICH_Setup_local_comm(MPI_Comm comm, MPI_Comm* local_comm, MPI_Comm* inter_comm) {
    int rank, size;
    int local_rank, local_size;
    int color, key;
    int mpi_errno = MPI_SUCCESS;
    
    // Get rank and size in the original communicator
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Calculate color (group ID) and key (rank within group)
    color = rank / RANKS_PER_NODE;
    key = rank % RANKS_PER_NODE;
    
    // Create local communicator by splitting the original communicator
    mpi_errno = MPI_Comm_split(comm, color, key, local_comm);
    if (mpi_errno != MPI_SUCCESS) {
        return mpi_errno;
    }
    
    // Get rank and size in the local communicator
    MPI_Comm_rank(*local_comm, &local_rank);
    MPI_Comm_size(*local_comm, &local_size);
    
    // Create inter-communicator containing the first process of each local group
    // Only processes with local_rank == 0 will be in this communicator
    color = (local_rank == 0) ? 1 : MPI_UNDEFINED;
    
    // Create a temporary communicator for the leaders
    MPI_Comm leaders_comm;
    mpi_errno = MPI_Comm_split(comm, color, rank, &leaders_comm);
    if (mpi_errno != MPI_SUCCESS) {
        MPI_Comm_free(local_comm);
        return mpi_errno;
    }
    
    // If this process is a leader (local_rank == 0), set the inter_comm
    if (local_rank == 0) {
        *inter_comm = leaders_comm;
    } else {
        // Non-leaders don't need the inter_comm, but we need to set it to something
        *inter_comm = MPI_COMM_NULL;
    }
    
    return MPI_SUCCESS;
}

//WORKING
int MPICH_Allreduce_inter_reduce_exchange_bcast(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    int typesize;
    char* tmp_buf = NULL;
    int rank, inter_rank;
    int mpi_errno = MPI_SUCCESS;

    MPI_Type_size(datatype, &typesize);

    MPI_Comm local_comm;
    MPI_Comm inter_comm;

    tmp_buf = (char*)malloc(count * typesize);
    if (tmp_buf == NULL) {
        return MPI_ERR_NO_MEM;
    }

    mpi_errno = MPICH_Setup_local_comm(comm, &local_comm, &inter_comm);
    if (mpi_errno != MPI_SUCCESS) {
        free(tmp_buf);
        return mpi_errno;
    }

    // Get rank in local communicator
    MPI_Comm_rank(local_comm, &rank);
    
    // Perform local reduction to rank 0 in each local communicator
    mpi_errno = MPI_Reduce(sendbuf, tmp_buf, count, datatype, op, 0, local_comm);
    if (mpi_errno != MPI_SUCCESS) {
        free(tmp_buf);
        if (inter_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&inter_comm);
        }
        MPI_Comm_free(&local_comm);
        return mpi_errno;
    }

    // Only leaders (rank 0 in local communicator) participate in inter-communicator reduction
    if (rank == 0) {
        // Get rank in inter-communicator
        MPI_Comm_rank(inter_comm, &inter_rank);
        
        // Perform reduction across leaders
        mpi_errno = MPI_Allreduce(tmp_buf, recvbuf, count, datatype, op, inter_comm);
        if (mpi_errno != MPI_SUCCESS) {
            free(tmp_buf);
            MPI_Comm_free(&inter_comm);
            MPI_Comm_free(&local_comm);
            return mpi_errno;
        }
        
        // Free the inter-communicator
        MPI_Comm_free(&inter_comm);
    }

    // Broadcast the result from rank 0 to all processes in local communicator
    mpi_errno = MPI_Bcast(recvbuf, count, datatype, 0, local_comm);
    
    // Free resources
    free(tmp_buf);
    MPI_Comm_free(&local_comm);

    return mpi_errno;
}


#ifdef DEBUG_MODE
int main(int argc, char** argv) {
    int rank, size;
    int count = 10; // Number of elements to reduce
    double *sendbuf, *recvbuf, *expected;
    int i, j;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Allocate buffers
    sendbuf = (double*)malloc(count * sizeof(double));
    recvbuf = (double*)malloc(count * sizeof(double));
    expected = (double*)malloc(count * sizeof(double));
    
    if (!sendbuf || !recvbuf || !expected) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize send buffer with rank-specific data
    for (i = 0; i < count; i++) {
        sendbuf[i] = rank + i * 0.1;
    }
    
    // Calculate expected results (sum of all ranks for each element)
    for (i = 0; i < count; i++) {
        expected[i] = 0;
        for (j = 0; j < size; j++) {
            expected[i] += j + i * 0.1;
        }
    }
    
    // Call our custom allreduce implementation
    MPICH_Allreduce_inter_reduce_exchange_bcast((char*)sendbuf, (char*)recvbuf, 
                                               count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Verify results
    int errors = 0;
    for (i = 0; i < count; i++) {
        if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
            errors++;
            if (rank == 0) {
                printf("Error at index %d: got %f, expected %f\n", 
                       i, recvbuf[i], expected[i]);
            }
        }
    }
    
    // Report results
    if (rank == 0) {
        if (errors == 0) {
            printf("Test PASSED: All %d values match expected results\n", count);
        } else {
            printf("Test FAILED: %d of %d values don't match expected results\n", 
                   errors, count);
        }
    }
    
    // Compare with standard MPI_Allreduce for validation
    double *mpi_result = (double*)malloc(count * sizeof(double));
    if (!mpi_result) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Allreduce(sendbuf, mpi_result, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    errors = 0;
    for (i = 0; i < count; i++) {
        if (fabs(recvbuf[i] - mpi_result[i]) > 1e-10) {
            errors++;
            if (rank == 0) {
                printf("Mismatch with MPI_Allreduce at index %d: custom %f, MPI %f\n", 
                       i, recvbuf[i], mpi_result[i]);
            }
        }
    }
    
    if (rank == 0) {
        if (errors == 0) {
            printf("Validation PASSED: Results match standard MPI_Allreduce\n");
        } else {
            printf("Validation FAILED: %d of %d values don't match MPI_Allreduce\n", 
                   errors, count);
        }
    }
    
    // Test with different data sizes
    int test_sizes[] = {1, 100, 1000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int test = 0; test < num_tests; test++) {
        int test_count = test_sizes[test];
        
        // Skip if already tested this size
        if (test_count == count) continue;
        
        // Reallocate buffers for new size
        double *test_sendbuf = (double*)malloc(test_count * sizeof(double));
        double *test_recvbuf = (double*)malloc(test_count * sizeof(double));
        double *test_mpi_result = (double*)malloc(test_count * sizeof(double));
        
        if (!test_sendbuf || !test_recvbuf || !test_mpi_result) {
            fprintf(stderr, "Memory allocation failed for test size %d\n", test_count);
            continue;
        }
        
        // Initialize test data
        for (i = 0; i < test_count; i++) {
            test_sendbuf[i] = rank + i * 0.1;
        }
        
        // Run custom implementation
        MPICH_Allreduce_inter_reduce_exchange_bcast((char*)test_sendbuf, (char*)test_recvbuf, 
                                                   test_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Run standard MPI implementation
        MPI_Allreduce(test_sendbuf, test_mpi_result, test_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Compare results
        errors = 0;
        for (i = 0; i < test_count; i++) {
            if (fabs(test_recvbuf[i] - test_mpi_result[i]) > 1e-10) {
                errors++;
            }
        }
        
        if (rank == 0) {
            if (errors == 0) {
                printf("Size %d test PASSED\n", test_count);
            } else {
                printf("Size %d test FAILED: %d of %d values don't match\n", 
                       test_count, errors, test_count);
            }
        }
        
        // Free test buffers
        free(test_sendbuf);
        free(test_recvbuf);
        free(test_mpi_result);
    }
    
    // Free memory
    free(sendbuf);
    free(recvbuf);
    free(expected);
    free(mpi_result);
    
    MPI_Finalize();
    return errors > 0 ? 1 : 0;
}
#endif // DEBUG_MODE


