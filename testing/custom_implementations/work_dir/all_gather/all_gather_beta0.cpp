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


int gather_k_nomial(char* sendbuf, int count, MPI_Datatype datatype,
                    char* recvbuf, MPI_Comm comm, int k, int root = 0) {
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);

    // compute number of phases = ceil(log_k(nprocs))
    int max = nprocs - 1, nphases = 0;
    while (max) { nphases++; max /= k; }
    bool p_of_k = (ipow(k, nphases) == nprocs);

    // allocate request array for nonblocking ops
    MPI_Request* reqs = (MPI_Request*)malloc((k - 1) * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;

    // tmp buffer: root uses recvbuf, others allocate nprocs*count*typeSize
    char* tmpbuf;
    if (rank == root) {
        tmpbuf = recvbuf;
    } else {
        tmpbuf = (char*)malloc(nprocs * count * typeSize);
        if (!tmpbuf) { free(reqs); return MPI_ERR_NO_MEM; }
    }

    // copy own data into tmpbuf at index 'rank'
    std::memcpy(tmpbuf + rank * count * typeSize,
                sendbuf,
                count * typeSize);

    for (int phase = 0, delta = 1; phase < nphases; phase++, delta *= k) {
        int groupSize = delta * k;
        int groupStart = ((rank - root) / groupSize) * groupSize + root;
        int offset = rank - groupStart;

        // Receiver: group root
        if (offset == 0) {
            int req_count = 0;
            for (int j = 1; j < k; j++) {
                int child = groupStart + j * delta;
                if (child >= root + nprocs) break;
                int subtree_size = delta;
                if (phase == nphases - 1 && !p_of_k) {
                    int remaining = (root + nprocs) - child;
                    subtree_size = std::min(delta, remaining);
                }
                MPI_Irecv(tmpbuf + child * count * typeSize,
                          subtree_size * count, datatype,
                          child, phase, comm,
                          &reqs[req_count++]);
            }
            MPI_Waitall(req_count, reqs, MPI_STATUS_IGNORE);
        }
        // Sender: subtree roots (excluding group root)
        else if (offset % delta == 0 && offset < groupSize) {
            int j = offset / delta;
            int subtree_root = rank;
            int dest = groupStart;
            int subtree_size = delta;
            if (phase == nphases - 1 && !p_of_k) {
                int remaining = (root + nprocs) - subtree_root;
                subtree_size = std::min(delta, remaining);
            }
            MPI_Request req;
            MPI_Isend(tmpbuf + subtree_root * count * typeSize,
                      subtree_size * count, datatype,
                      dest, phase, comm,
                      &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            break; // done for this rank
        }
        // others idle
    }

    if (rank != root) free(tmpbuf);
    free(reqs);
    return MPI_SUCCESS;
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
    int root_local = node_id%b;

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

int allgather_k_brucks(char* sendbuf, int count, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k) {
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);

    int num_reqs = 0;
    int max = nprocs - 1;
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

    if(ipow(k, nphases) == nprocs) {
        p_of_k = 1;
    }

    char* tmp_recv_buffer;

    if(rank == 0) {
        tmp_recv_buffer = recvbuf;
    } else {
        tmp_recv_buffer = (char*) malloc(nprocs * count * typeSize);  // Fixed 'size' to 'nprocs'
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
            if(delta * j >= nprocs)  // Fixed 'size' to 'nprocs'
                break;
                
            dst = (nprocs + (rank - delta * j)) % nprocs;
            src = (rank + delta * j) % nprocs;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

            int tmp_count;
            if ((i == nphases - 1) && (!p_of_k)) {
                tmp_count = count * delta;
                int left_count = count * (nprocs - delta * j);
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

    if(rank != 0) {
        // Rearrange data in the correct order
        memcpy(recvbuf, 
               tmp_recv_buffer + (nprocs - rank) * count * typeSize, 
               rank * count * typeSize);

        memcpy(recvbuf + rank * count * typeSize, 
               tmp_recv_buffer, 
               (nprocs - rank) * count * typeSize);
               
        free(tmp_recv_buffer);
    }

    free(reqs);
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

    int y = nprocs/b;
    bool highn = b<y;



    
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

    if(highn){
        // allocate new tmp buffer
        tmp_recv_buffer = (char*) malloc(nprocs * count * typeSize);  // Fixed'size' to 'nprocs'

        // copy recv_buf to tmp_recv_buffer
        memcpy(tmp_recv_buffer, recvbuf, nprocs * count * typeSize);

        int tmp_count = count / (y/b);

        for(int i = 0, j = 0; i < y; i++, j+=(y/b)){
            memcpy(recvbuf + i*tmp_count*typeSize, tmp_recv_buffer + (j%(y) + j/(y))*tmp_count*typeSize, tmp_count*typeSize);
        }
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

    

    MPI_Request* reqs = (MPI_Request*) malloc(((nprocs/b)*(nprocs/(b*b)) * sizeof(MPI_Request)));

    bool highn = b<(nprocs/b);



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
    }else if(highn){
        ///DOES NOT WORK IF NN!=K*B
        for(int i=0; i<nprocs/(b); i+=b){
            if((node_id - i) != node_rank){
                MPI_Irecv(recvbuf+count*(i/b)*typeSize, count, datatype, (i+node_rank)*b + node_rank, 1, comm, &reqs[num_reqs++]);
            }else{
                memcpy(recvbuf+count*(i/b)*typeSize, sendbuf, count * typeSize);
                for(int j=0; j<nprocs/b; j++){
                    if(j==node_id )
                        continue;
                    int dst = j*b + node_rank;
                    MPI_Isend(sendbuf, count, datatype, dst, 1, comm, &reqs[num_reqs++]);
                }
            }
        }
        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    }


    free(reqs);
    return MPI_SUCCESS;
}





int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Set parameters
    int k = 2;  // Radix parameter
    int b = 4;  // Block size parameter
    int count = 4;  // Number of elements per process
    
    // Check if we have command line arguments
    if (argc > 1) count = atoi(argv[1]);
    if (argc > 2) k = atoi(argv[2]);
    if (argc > 3) b = atoi(argv[3]);
    
    // Ensure b is valid (not larger than nprocs)
    b = std::min(b, nprocs);
    
    // Create data buffers
    std::vector<float> sendbuf(count);
    std::vector<float> recvbuf(count * b);
    std::vector<float> expected_result(count * b);
    
    // Initialize send buffer with rank-specific data
    for (int i = 0; i < count; i++) {
        sendbuf[i] = rank * 100.0f + i;
    }
    
    // Clear receive buffer
    std::fill(recvbuf.begin(), recvbuf.end(), 0.0f);
    
    // Print initial values from rank 0
    if (rank == 0) {
        std::cout << "\nTesting intra_allgather_k_brucks with:" << std::endl;
        std::cout << "  k = " << k << std::endl;
        std::cout << "  b = " << b << std::endl;
        std::cout << "  count = " << count << std::endl;
        std::cout << "  nprocs = " << nprocs << std::endl;
        
        std::cout << "\nInitial values:" << std::endl;
        for (int r = 0; r < std::min(nprocs, 8); r++) {
            if (r == 0) {
                std::cout << "  Rank " << r << ": ";
                for (int i = 0; i < std::min(count, 8); i++) {
                    std::cout << sendbuf[i] << " ";
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (r == rank && r != 0) {
                std::cout << "  Rank " << r << ": ";
                for (int i = 0; i < std::min(count, 8); i++) {
                    std::cout << sendbuf[i] << " ";
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    } else {
        for (int r = 0; r < std::min(nprocs, 8); r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank && r != 0) {
                std::cout << "  Rank " << r << ": ";
                for (int i = 0; i < std::min(count, 8); i++) {
                    std::cout << sendbuf[i] << " ";
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    // Synchronize before starting the algorithm
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Measure time
    double start_time = MPI_Wtime();
    
    // Call the intra_allgather_k_brucks function
    int result = intra_gather_k_nomial(
        (char*)sendbuf.data(), count, MPI_FLOAT,
        (char*)recvbuf.data(), MPI_COMM_WORLD, k, b
    );
    // int result = allgather_k_brucks(
    //     (char*)sendbuf.data(), count, MPI_FLOAT,
    //     (char*)recvbuf.data(), MPI_COMM_WORLD, k
    // );
    // Measure end time
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    // Gather timing results
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Verify results using MPI_Allgather for comparison
    // We'll use a separate buffer for the expected result
    // MPI_Allgather(sendbuf.data(), count, MPI_FLOAT, 
    //              expected_result.data(), count, MPI_FLOAT, MPI_COMM_WORLD);
    
    // Check if our implementation matches the expected result
    // bool correct = true;
    // for (int i = 0; i < std::min(b, nprocs) * count; i++) {
    //     if (std::abs(recvbuf[i] - expected_result[i]) > 1e-5) {
    //         correct = false;
    //         if (rank == 0) {
    //             std::cout << "Mismatch at index " << i << ": "
    //                       << recvbuf[i] << " vs expected " << expected_result[i] << std::endl;
    //         }
    //         break;
    //     }
    // }
    
    // Print results from rank 0
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        std::cout << "\nExecution time: " << max_time * 1000 << " ms" << std::endl;
        std::cout << "\nResults after intra_gather_k_nomial:" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank%b == (rank/b)%b) {
        



        std::cout << "  Rank " << rank << ": ";
        for (int i = 0; i < count * b; i++) {
            std::cout << recvbuf[i] << " ";

        }
        std::cout << std::endl;
    }

    int mult = nprocs/(b*b);




    std::vector<float> recvbuf2(count * b*mult);

    MPI_Barrier(MPI_COMM_WORLD);

    int result2 = inter_allgather_linear(
        (char*)recvbuf.data(), count, MPI_FLOAT,
        (char*)recvbuf2.data(), MPI_COMM_WORLD, b
    );

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nResults after inter_allgather:" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "  Rank " << rank << ": ";
        for (int i = 0; i < count * b*mult; i++) {
            std::cout << recvbuf2[i] << " ";

        }
        std::cout << std::endl;

    std::vector<float> recvbuf3(count * nprocs*mult);

    int result3 = intra_allgather_k_brucks(
        (char*)recvbuf2.data(), count*b*mult, MPI_FLOAT,
        (char*)recvbuf3.data(), MPI_COMM_WORLD, k, b
    );

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\nResults after intra_allgather_k_brucks:" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i = 0; i < nprocs; i++){
        if(rank == i){
            std::cout << "  Rank " << rank << ": ";
            for (int j = 0; j < count * nprocs; j++) {
                std::cout << recvbuf3[j] << " ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    //free all buffers
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}