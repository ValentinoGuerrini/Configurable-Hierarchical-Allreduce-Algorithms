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


int intra_allgather_k_brucks(char* sendbuf, int count, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b) {
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);

    int i, j, s;
    int delta = 1;


    int b_rank = rank % b;  // Rank within the block
    int group = rank / b;

    int num_reqs = 0;
    int max = b - 1;
    int nphases = 0;
    const int nnodes = nprocs / b;

    const int nstages = nnodes/b;
    char* tmp_recv_buffer;



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

   

    if(b_rank == 0) {
        tmp_recv_buffer = recvbuf;
    } else {
        tmp_recv_buffer = (char*) malloc((nstages*b + nnodes%b) * count * typeSize);  // Fixed 'size' to 'nprocs'
        if (tmp_recv_buffer == NULL) {
            free(reqs);
            return MPI_ERR_NO_MEM;
        }
    }



    for(s = 0; s < nstages; s++){
            // Copy local data to receive buffer
        memcpy(tmp_recv_buffer, sendbuf, count * typeSize);
    

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
                    
            //free(tmp_recv_buffer);
        }
        delta = 1;

        tmp_recv_buffer += b * count * typeSize;
        recvbuf += b*count*typeSize;
        sendbuf += count*typeSize;
    }

    int nu_count = nnodes%b;

    if(nu_count != 0) {

        if(b_rank < nu_count) {
            memcpy(tmp_recv_buffer, sendbuf, count * typeSize);
        } else {
            free(tmp_recv_buffer-nstages*b*count*typeSize);
            tmp_recv_buffer = recvbuf;
        }

        int active[b];
        int send_sizes[nphases+1][b];
        memset(send_sizes, 0, (nphases+1)*b*sizeof(int));
        //sizes array

        for(i =0; i< b; i++){
            active[i] = i < nu_count ? 0 : -1;
            send_sizes[0][i] = i < nu_count ? count : 0;

            //initialize sizes to 1 block if i < nnodes%b  nproc/b 0 otherwise 
        }



        int isrc,idst;
        int received;

        for(i = 0; i < nphases; i++) {
            num_reqs = 0;

            received = send_sizes[i][b_rank];
            
            for(j = 1; j < k; j++) {
                if(delta * j >= b){

                    for(int l = 0; l < b; l++){
                        if(active[l] == i){
                                active[l] = i+1;

                        }
                        send_sizes[i+1][l] += send_sizes[i][l];
                    }


                    break;

                }  

                

                

                isrc = (b_rank + delta * j) % b;
                idst = (b + (b_rank - delta * j)) % b;
                    
                dst = idst + group*b;
                src = isrc + group*b;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

                
                int ssssize;
                ssssize = min(send_sizes[i][isrc], (nu_count) * count - received);
                if(active[isrc] == i && ssssize > 0){
                    MPI_Irecv(tmp_recv_buffer + received * typeSize, 
                            ssssize, datatype, src, 1, comm, &reqs[num_reqs++]);
                    
                    received += ssssize;
                }
                ssssize = min(send_sizes[i][b_rank], (nu_count) * count - (send_sizes[i][idst] + send_sizes[i+1][idst]));

                if(active[b_rank] == i && ssssize > 0){
                    MPI_Isend(tmp_recv_buffer, ssssize , datatype, 
                            dst, 1, comm, &reqs[num_reqs++]);
                            
                }

                for(int l = 0; l < b; l++){
                    if(active[l] == i){
                        active[( b+ (l - delta * j)) % b] = active[(b+ (l - delta * j)) % b] != i ? i+1 : i;

                        
                        send_sizes[i+1][(b + (l - delta * j)) % b] += send_sizes[i][l];
                        
                        if(j == k-1){
                            active[l] = i+1;
                        }
                    }
                    if(j == k-1){
                        send_sizes[i+1][l] += send_sizes[i][l];
                    }

                }

            }
            MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
            delta *= k;
            
        }

        if(b_rank != 0 && b_rank < nu_count) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    tmp_recv_buffer + ((nu_count) - b_rank) * count * typeSize, 
                    b_rank * count * typeSize);

            memcpy(recvbuf + b_rank * count * typeSize, 
                    tmp_recv_buffer, 
                    ((nu_count) - b_rank) * count * typeSize);

            // memcpy(recvbuf, tmp_recv_buffer, count* (nnodes%b) * typeSize);
            free(tmp_recv_buffer-nstages*b*count*typeSize);
        }

    }else if(b_rank != 0){
        free(tmp_recv_buffer-nstages*b*count*typeSize);
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
    int nnodes = nprocs/b;

    

    MPI_Request* reqs = (MPI_Request*) malloc(((nprocs/b)*(nprocs/(b*b)) * sizeof(MPI_Request)));

    bool highn = b<nnodes;

    int i,j;
    bool rootCond;


    for(i = 0; i<nnodes; i+=b){
        rootCond = ((node_id - i) == node_rank);
        if(!rootCond && (i+node_rank < nnodes)){
            MPI_Irecv(recvbuf+count*(i/b)*typeSize, count, datatype, (i+node_rank)*b + node_rank, 1, comm, &reqs[num_reqs++]);
        }else if(rootCond){
            memcpy(recvbuf+count*(i/b)*typeSize, sendbuf, count * typeSize);
                for(j=0; j<nnodes; j++){
                    if(j==node_id )
                        continue;
                    int dst = j*b + node_rank;
                    MPI_Isend(sendbuf, count, datatype, dst, 1, comm, &reqs[num_reqs++]);
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
    int b = 3;  // Block size parameter
    int count = 4;  // Number of elements per process
    
    // Check if we have command line arguments
    if (argc > 1) count = atoi(argv[1]);
    if (argc > 2) k = atoi(argv[2]);
    if (argc > 3) b = atoi(argv[3]);
    
    // Ensure b is valid (not larger than nprocs)
    b = std::min(b, nprocs);
    
    // Create data buffers
    std::vector<float> sendbuf(count);
    std::vector<float> recvbuf(count * nprocs);
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




    std::vector<float> recvbuf2(count * nprocs);

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

    // std::vector<float> recvbuf3(count * nprocs*mult);
    std::vector<float> recvbuf3(count * nprocs);

    int result3 = intra_allgather_k_brucks(
        (char*)recvbuf2.data(), count*b, MPI_FLOAT,
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