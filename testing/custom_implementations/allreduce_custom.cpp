#include "common.hpp"

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

int calculate_max_datablocks_per_round(int r, int w) {
    int max_db_per_round = 1;
    for (int i = 0; i < w - 1; i++) {
        if (max_db_per_round > INT_MAX / r) {
            return -1;
        }
        max_db_per_round *= r;
    }
    return max_db_per_round;
}





int emma_algorithm (char *sendbuf, char *recvbuf, int totcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int r , int b) {







    int rank, nprocs, typesize;
    MPI_Comm_rank(comm, &rank); // get rank
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typesize);

    int count = totcount / nprocs;


    //General Checks
    if ( r < 2 ) { r = 2; }
    if ( r > nprocs - 1 ) { r = nprocs - 1; }
    if (b <= 0 || b > nprocs) b = nprocs;


    //for loop indexing
    int i, j, x, z, k, s;
    int mpi_errno = MPI_SUCCESS;
    int num_reqs;
    int rotate_index_array[nprocs];



    //const int w = calculate_w(r, nprocs - 1);  // number of bits required of r representation
    int w = 0;
    int max_rank = nprocs - 1;

    while (max_rank > 0) {
        max_rank /= r;
        w++;
    }

    const int max_db_per_round = calculate_max_datablocks_per_round(r,w);   // maximum send number of elements

    if (max_db_per_round == -1) {
        std::cerr << "Overflow detected in calculate_max_datablocks_per_round" << std::endl;
        return 1; // Exit program with error
    }



    const int highest_digit = (max_db_per_round*r - nprocs) / max_db_per_round; // calculate the number of highest digits

    const int K = w * (r - 1) - highest_digit; // the total number of communication rounds

    const int comm_round = K + 1;
    const int rem2 = r + 1;

    const int size = count * typesize;
    


    char *extra_buffer, *temp_recv_buffer, *temp_send_buffer;
    int extra_ids[nprocs - rem2];
    memset(extra_ids, -1, sizeof(extra_ids));
    int spoint, distance=1, next_distance = distance*r, di = 0;

    if (K < nprocs - 1) {
        // 1. Find max send count


        // 2. create local index array after rotation
        for (i = 0; i < nprocs; i++) {
            rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;
        }

        // 3. exchange data with log(P) steps
        extra_buffer = (char*) malloc(size * (nprocs - comm_round));
        temp_recv_buffer = (char*) malloc(size * nprocs);///this can be modified for reduce
        temp_send_buffer = (char*) malloc(size * max_db_per_round);


        CHECK_ALLOCATIONS(extra_buffer == nullptr || temp_recv_buffer == nullptr || temp_send_buffer == nullptr);

        for (x = 0; x < w; x++) {
            for (z = 1; z < r; z ++) {
                spoint = z * distance;

                if (spoint > nprocs) { break; }

                int end = spoint + distance;

                if (spoint + distance > nprocs) {
                    end =nprocs;
                }

                for (i = spoint + 1; i < end; i++) {
                    extra_ids[i-rem2] = di++;
                }
            }
            distance *= r;
        }


    }

    // copy data that need to be sent to each rank itself

    char * reducebuf = (char*) malloc(size);
    
    memcpy(reducebuf, &sendbuf[rank*size], size);
    

    

    int sent_blocks[r-1][max_db_per_round];

    int nc, rem, ns, ze, ss;
    spoint = 1, distance =1;

    MPI_Request* reqs = (MPI_Request *) malloc(2 * r * sizeof(MPI_Request));


    MPI_Status* stats = (MPI_Status *) malloc(2 * r * sizeof(MPI_Status));


    CHECK_ALLOCATIONS(stats == nullptr || reqs == nullptr);

    char* temp = (char*) malloc(size*max_db_per_round);
    CHECK_ALLOCATIONS(temp == nullptr);
    int count_tmp=0;


    for (x = 0; x < w; x++) {
        ze = (x == w - 1)? r - highest_digit: r;


        int zoffset = 0, zc = ze-1;
        int zns[zc];

        for (k = 1; k < ze; k += b) {
            ss = ze - k < b ? ze - k : b;
            num_reqs = 0;

            

            for (s = 0; s < ss; s++) { // s = 0, s = 1


                z = k + s;



                spoint = z * distance;
                nc = nprocs / next_distance * distance;
                rem = nprocs % next_distance - spoint;



                if (rem < 0) {
                    rem = 0;
                }


                ns = (rem > distance)? (nc + distance) : (nc + rem);

                zns[z-1] = ns;

                int recvrank = (rank + spoint) % nprocs;

                int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process


                if (ns == 1) {

                    

                    mpi_errno = MPI_Irecv(&temp[count_tmp++], size, MPI_CHAR, recvrank, 1, comm, &reqs[num_reqs++]);

                    CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
                    
                    mpi_errno = MPI_Isend(&sendbuf[count*sendrank*typesize], size, MPI_CHAR, sendrank, 1, comm, &reqs[num_reqs++]);

                    CHECK_MPI_ERROR(mpi_errno!= MPI_SUCCESS);
  
                }else {
                    di = 0;
                    for (i = spoint; i < nprocs; i += next_distance) {
                        int j_end = (i+distance > nprocs)? nprocs: i+distance;
                        for (j = i; j < j_end; j++) {
                            int id = (j + rank) % nprocs;
                            sent_blocks[z-1][di++] = id;
                        }
                    }



                    int sendCount = count * di; 


                    // prepare send data

                    for (i = 0; i < di; i++) {
                        int send_index = rotate_index_array[sent_blocks[z-1][i]];
                        int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;

                        
                        if (i % distance == 0) {
                            memcpy(&temp_send_buffer[i*size], &sendbuf[count*send_index*typesize], size);
                        }
                        else {
                            memcpy(&temp_send_buffer[i*size], &extra_buffer[extra_ids[o]*size], size);
                        }

                    }

                    mpi_errno = MPI_Irecv(&temp_recv_buffer[zoffset], sendCount*typesize, MPI_CHAR, recvrank, recvrank+z, comm, &reqs[num_reqs++]);
                    CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
                    mpi_errno = MPI_Isend(temp_send_buffer, di*size, MPI_CHAR, sendrank, rank+z, comm, &reqs[num_reqs++]);
                    CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
                    // Print content of temp_recv_buffer for debugging
                    zoffset += sendCount*typesize;
                }

            }

            mpi_errno = MPI_Waitall(num_reqs, reqs, stats);

            CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);

            for (i = 0; i <count_tmp; i++) {
    int rank, nprocs, typesize;
                mpi_errno = MPI_Reduce_local(&temp[i], reducebuf,count, datatype, op);
                CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
            }

        }

        if (K < nprocs - 1) {
            // replaces
            int offset = 0;
            for (i = 0; i < zc; i++) {
                for (j = 0; j < zns[i]; j++){

                    if (zns[i] > 1){
                        int o = (sent_blocks[i][j] - rank + nprocs) % nprocs - rem2;
                        if (j < distance) {
                            //memcpy(&recvbuf[count*sent_blocks[i][j]*typesize], &temp_recv_buffer[offset], size);
                            mpi_errno = MPI_Reduce_local(&temp_recv_buffer[offset], reducebuf,count, datatype, op);
                            CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
                        }
                        else {
                            memcpy(&extra_buffer[extra_ids[o]*size], &temp_recv_buffer[offset], size);
                        }
                        offset += size;
                    }
                }
            }
        }

        distance *= r;
        next_distance *= r;

    }

    
//ALLGATHER PHASE

    int p_of_k  = (max_db_per_round == nprocs)? 1: 0;
    distance = 1;
    
    if(rank == 0) {
        free(temp_recv_buffer);
        temp_recv_buffer = recvbuf;
    }
    memcpy(temp_recv_buffer, reducebuf, size);

    int sendrank, recvrank, left_count, distance_j;

    for(i = 0; i < w; i++) {
        num_reqs = 0;
        for(j = 0; j < r; j++) {
            distance_j = distance * j;
            if(distance_j >= nprocs)  // Fixed 'size' to 'nprocs'
                break;
                
            sendrank = (nprocs + (rank - distance_j)) % nprocs;
            recvrank = (rank + distance_j) % nprocs;  // Fixed 'size' to 'nprocs' and 'r' to 'j'


            if ((!p_of_k) && (i == w - 1)) {
                count_tmp = count * distance;
                left_count = count * (nprocs - distance_j);
                if (j == r - 1) {
                    count_tmp = left_count;
                } else {
                    count_tmp = count_tmp <= left_count ? count_tmp: left_count;
                }
            } else {
                count_tmp = count * distance;
            }
            mpi_errno = MPI_Irecv(temp_recv_buffer + size * distance_j, count_tmp, datatype, recvrank, recvrank + r, comm, &reqs[num_reqs++]);
            
            CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
            
            mpi_errno = MPI_Isend(temp_recv_buffer, count_tmp, datatype, sendrank, rank + r, comm, &reqs[num_reqs++]);

            CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
        }
        distance *= r;
        mpi_errno = MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);

        CHECK_MPI_ERROR(mpi_errno != MPI_SUCCESS);
        
    }

    if(rank != 0) {
        // Rearrange data in the correct order
        memcpy(recvbuf, 
               temp_recv_buffer + (nprocs - rank) * size, 
               rank * size);

        memcpy(recvbuf + rank * size, 
               temp_recv_buffer, 
               (nprocs - rank) * size);
               
        free(temp_recv_buffer);
    }
    if (K < nprocs - 1) {
        free(extra_buffer);
        free(temp_send_buffer);
    }
    free(stats);
    free(reqs);
    return MPI_SUCCESS;
}


#ifdef DEBUG_MODE 

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set parameters for emma_algorithm
    const int r = 3;
    const int w = 1;
    
    // Test with different counts
    const int test_counts[] = {10, 100, 1000};
    const int num_tests = sizeof(test_counts) / sizeof(test_counts[0]);
    
    for (int t = 0; t < num_tests; t++) {
        const int count = test_counts[t];
        const int totcount = count * size; // Total elements across all processes
        
        // Allocate and initialize buffers
        float *sendbuf = new float[totcount];
        float *recvbuf_emma = new float[totcount];
        float *recvbuf_mpi = new float[totcount];
        
        // Initialize send buffer with rank-specific data
        for (int i = 0; i < totcount; i++) {
            sendbuf[i] = 1000*rank + 1 * i;
        }
        
        // Clear receive buffers
        memset(recvbuf_emma, 0, totcount * sizeof(float));
        memset(recvbuf_mpi, 0, totcount * sizeof(float));
        
        // Barrier to synchronize before timing
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Test standard MPI_Allreduce
        double start_mpi = MPI_Wtime();
        MPI_Allreduce(sendbuf, recvbuf_mpi, totcount, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double end_mpi = MPI_Wtime();
        
        // Barrier to ensure fair timing comparison
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Test emma_algorithm
        double start_emma = MPI_Wtime();
        int result = emma_algorithm((char*)sendbuf, (char*)recvbuf_emma, totcount, MPI_FLOAT, 
                                   MPI_SUM, MPI_COMM_WORLD, r, w);
        double end_emma = MPI_Wtime();
        
        // Check for errors
        if (result != MPI_SUCCESS) {
            if (rank == 0) {
                std::cerr << "Error: emma_algorithm failed with count = " << count << std::endl;
            }
            continue;
        }
        
        // Verify results
        bool correct = true;
        float max_diff = 1e-10;
        for (int i = 0; i < totcount; i++) {
            float diff = std::abs(recvbuf_emma[i] - recvbuf_mpi[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > 1e-3) {
                correct = false;
                if (rank == 0) {
                    std::cerr << "Mismatch at index " << i << ": emma = " << recvbuf_emma[i] 
                              << ", mpi = " << recvbuf_mpi[i] << std::endl;
                }
                break;
            }
        }
        
        // Report results
        if (rank == 0) {
            std::cout << "Test with count = " << count << " (total elements = " << totcount << "):\n";
            std::cout << "  Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
            std::cout << "  Max difference: " << max_diff << std::endl;
            std::cout << "  Time (MPI): " << (end_mpi - start_mpi) * 1000 << " ms" << std::endl;
            std::cout << "  Time (EMMA): " << (end_emma - start_emma) * 1000 << " ms" << std::endl;
            std::cout << "  Speedup: " << (end_mpi - start_mpi) / (end_emma - start_emma) << "x" << std::endl;
            std::cout << std::endl;
        }
        
        // Clean up
        delete[] sendbuf;
        delete[] recvbuf_emma;
        delete[] recvbuf_mpi;
    }
    
    MPI_Finalize();
    return 0;
}


#endif