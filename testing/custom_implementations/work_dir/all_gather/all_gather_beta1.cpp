#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

#define CHECK_ALLOCATIONS(...) \
    if ((__VA_ARGS__)) { \
        std::cerr << "Memory allocation failed!" << std::endl; \
        return 1; \
    }

#define CHECK_MPI_ERROR(...) \
    if ((__VA_ARGS__)) { \
        std::cerr << "MPI error " << std::endl; \
        return 1; \
    }


int ipow(int base, int exp) {
    int result = 1;
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}


int intra_gather_k_nomial(char* sendbuf,
                          int   count,
                          MPI_Datatype datatype,
                          char* recvbuf,
                          MPI_Comm comm,
                          int   k,
                          int   b)
{
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);

    int node_id    = rank / b;   // which node
    int local_rank = rank % b;   // which slot on that node

    // — pick the “home” root on each node: the slot equal to node_id
    int root_local = node_id;

    // — normalize so that root_local→0, others shift down mod b
    int shift = (local_rank - root_local + b) % b;

    // — compute #phases = ceil(log_k(b))
    int max_v = b - 1, nphases = 0;
    while (max_v > 0) { ++nphases; max_v /= k; }
    bool exact_pow = (ipow(k, nphases) == b);

    // — everyone allocates the same temp buffer of b*count elements
    char* tmp = (char*)malloc(b * count * typeSize);
    if (!tmp) return MPI_ERR_NO_MEM;

    // copy our sendbuf into our normalized slot “shift”
    std::memcpy(tmp + shift * count * typeSize,
                sendbuf,
                count * typeSize);

    // scratch array for up to (k-1) outstanding Irecvs
    MPI_Request* reqs = (MPI_Request*)malloc((k - 1) * sizeof(*reqs));
    if (!reqs) { free(tmp); return MPI_ERR_NO_MEM; }

    // — k-nomial gather in the [0..b-1] “normalized” domain
    for (int phase = 0, delta = 1; phase < nphases; ++phase, delta *= k) {
        int groupSize  = delta * k;
        int gstart     = (shift / groupSize) * groupSize;
        int offset     = shift - gstart;

        // real‐world rank of the normalized 0-root:
        int real_root_local = (gstart + root_local) % b;
        int real_root_glob  = node_id * b + real_root_local;

        if (offset == 0) {
            // I am this subtree’s root → post up to (k−1) Irecvs
            int nr = 0;
            for (int j = 1; j < k; ++j) {
                int childNorm = gstart + j*delta;
                if (childNorm >= b) break;
                int subtree = delta;
                if (phase == nphases - 1 && !exact_pow)
                    subtree = std::min(subtree, b - childNorm);

                // recv into the *normalized* slot “childNorm”
                int src_local = (childNorm + root_local) % b;
                int src_glob  = node_id * b + src_local;
                MPI_Irecv(tmp + childNorm * count * typeSize,
                          subtree * count, datatype,
                          src_glob, phase,
                          comm, &reqs[nr++]);
            }
            MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
        }
        else if ((offset % delta) == 0 && offset < groupSize) {
            // I am a child‐subtree root → send my normalized block up
            int subtree = delta;
            if (phase == nphases - 1 && !exact_pow)
                subtree = std::min(subtree, b - shift);

            MPI_Request sreq;
            MPI_Isend(tmp + shift * count * typeSize,
                      subtree * count, datatype,
                      real_root_glob, phase,
                      comm, &sreq);
            MPI_Wait(&sreq, MPI_STATUS_IGNORE);
            break;
        }
        // all other shifts sit idle
    }

    free(reqs);

    // — now only the node‐root must unshift *once* into recvbuf
    if (local_rank == root_local) {
        for (int real_slot = 0; real_slot < b; ++real_slot) {
            // find which normalized index holds that real_slot
            int norm_i = (real_slot - root_local + b) % b;
            std::memcpy(recvbuf + real_slot * count * typeSize,
                        tmp      + norm_i    * count * typeSize,
                        count * typeSize);
        }
    }

    free(tmp);
    return MPI_SUCCESS;
}


int intra_allgather_k_brucks(char* sendbuf, int count, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b) {
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);

    int b_rank = rank % b;  // Rank within the block
    int group = std::ceil(rank / b);

    int num_reqs = 0;
    int max = b - 1;
    int nphases = 0;

    // Calculate number of phases
    while(max) {
        nphases++;
        max /= k;  // Fixed missing semicolon
    }

    // Allocate request array
    MPI_Request* reqs = (MPI_Request*) malloc(2 * (k-1) * sizeof(MPI_Request));
    if (reqs == NULL) {
        return MPI_ERR_NO_MEM;
    }

    int dst, src, p_of_k = 0;

    if(ipow(k, nphases) == b) {
        p_of_k = 1;
    }

    char* tmp_recv_buffer;

    if(b_rank == 0) {
        tmp_recv_buffer = recvbuf;
    } else {
        tmp_recv_buffer = (char*) malloc(b * count * typeSize);  // Fixed 'size' to 'nprocs'
        if (tmp_recv_buffer == NULL) {
            free(reqs);
            return MPI_ERR_NO_MEM;
        }
    }

    // Copy local data to receive buffer
    memcpy(tmp_recv_buffer, sendbuf, count * typeSize);

    int i, j;
    int delta = 1;

    for(i = 0; i < nphases; i++) {
        num_reqs = 0;
        for(j = 1; j < k; j++) {
            if(delta * j >= b)  // Fixed 'size' to 'nprocs'
                break;
                
            dst = (b + (b_rank - delta * j)) % b + group*b;
            src = (b_rank + delta * j) % b + group*b;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

            int tmp_count;
            if ((i == nphases - 1) && (!p_of_k)) {
                tmp_count = count * delta;
                int left_count = count * (b - delta * j);
                if (j == k - 1) {
                    tmp_count = left_count;
                } else {
                    tmp_count = min(tmp_count, left_count);
                }
            } else {
                tmp_count = count * delta;
            }

            MPI_Irecv(tmp_recv_buffer + j * count * delta * typeSize, 
                        tmp_count, datatype, src, 1, comm, &reqs[num_reqs++]);

            MPI_Isend(tmp_recv_buffer, tmp_count, datatype, 
                        dst, 1, comm, &reqs[num_reqs++]);
        }
        
        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
        delta *= k;
    }

    if(b_rank != 0) {
        // Rearrange data in the correct order
        memcpy(recvbuf, 
                tmp_recv_buffer + (b - b_rank) * count * typeSize, 
                b_rank * count * typeSize);

        memcpy(recvbuf + b_rank * count * typeSize, 
                tmp_recv_buffer, 
                (b - b_rank) * count * typeSize);
                
        free(tmp_recv_buffer);
    }

    free(reqs);
    return MPI_SUCCESS;
}


int inter_allgather_linear(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int b){
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);
    int num_reqs = 0;
    int count= sendcount * b;

    int node_id = rank/b;
    int node_rank = rank%b;

    

    MPI_Request* reqs = (MPI_Request*) malloc((b-1)*(b-1) * sizeof(MPI_Request));


    if(b==nprocs/b){
        if(node_id!=node_rank){
            MPI_Irecv(recvbuf, count, datatype, node_rank*b + node_rank, 1 , comm, &reqs[num_reqs++]);

        }else{
            memcpy(recvbuf, sendbuf, count * typeSize);
            for(int i=0; i<b; i++){
                if(i==node_id)
                    continue;
                int dst = i*b + node_rank;
                MPI_Isend(sendbuf, count, datatype, dst, 1, comm, &reqs[num_reqs++]);
            }
            
        }

        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    }


    free(reqs);
    return MPI_SUCCESS;
}

int allgather_radix_batch(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b){

    int result;
    result = intra_gather_k_nomial(sendbuf, sendcount, datatype, recvbuf, comm, k, b);
    if(result != MPI_SUCCESS){
        return result;
    }
    result = inter_allgather_linear(recvbuf, sendcount, datatype, recvbuf, comm, b);
    if(result != MPI_SUCCESS){
        return result;
    }
    result = intra_allgather_k_brucks(recvbuf, sendcount*b, datatype, recvbuf, comm, k, b);
    if(result!= MPI_SUCCESS){
        return result;
    }
    return MPI_SUCCESS;
    
}



int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* default radix and batch */
    int k = 2, b = 3;
    if (argc >= 3) {
        k = atoi(argv[1]);
        b = atoi(argv[2]);
    }

    if (rank == 0) {
        printf("# allgather_radix_batch vs MPI_Allgather\n");
        printf("# k=%d, b=%d\n", k, b);
        printf("# sendcount   custom_time(s)   std_time(s)\n");
    }

    for (int i = 0; i < 10; ++i) {
        int sendcount = 4 << i;  /* 4,8,16,...,2048 */

        /* allocate buffers as ints; cast to char* for the call */
        int *sendbuf     =(int*)malloc(sendcount * sizeof(int));
        int *recv_custom = (int*)malloc(sendcount * size * sizeof(int));
        int *recv_std    = (int*)malloc(sendcount * size * sizeof(int));
        if (!sendbuf || !recv_custom || !recv_std) {
            fprintf(stderr, "Rank %d: allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* init sendbuf to unique values so we can check correctness */
        for (int j = 0; j < sendcount; ++j) {
            sendbuf[j] = rank * sendcount + j;
        }

        /* synchronize and time custom allgather */
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        allgather_radix_batch((char*)sendbuf,
                              sendcount,
                              MPI_INT,
                              (char*)recv_custom,
                              MPI_COMM_WORLD,
                              k,
                              b);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        /* synchronize and time standard MPI_Allgather */
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        MPI_Allgather(sendbuf,
                      sendcount,
                      MPI_INT,
                      recv_std,
                      sendcount,
                      MPI_INT,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime();

        /* correctness check */
        int errors = 0;
        for (int idx = 0; idx < sendcount * size; ++idx) {
            if (recv_custom[idx] != recv_std[idx]) {
                if (errors < 5 && rank == 0) {
                    int from_rank = idx / sendcount;
                    int offset   = idx % sendcount;
                    fprintf(stderr,
                            "Mismatch at global idx %d (src %d, off %d): custom=%d std=%d\n",
                            idx,
                            from_rank,
                            offset,
                            recv_custom[idx],
                            recv_std[idx]);
                }
                errors++;
            }
        }
        if (errors > 0 && rank == 0) {
            fprintf(stderr, "Total mismatches for sendcount=%d: %d\n",
                    sendcount, errors);
        }

        /* only rank 0 prints the timing results */
        if (rank == 0) {
            printf("%10d   %12.6f   %10.6f\n",
                   sendcount,
                   t1 - t0,
                   t3 - t2);
            fflush(stdout);
        }

        free(sendbuf);
        free(recv_custom);
        free(recv_std);
    }

    MPI_Finalize();
    return 0;
}