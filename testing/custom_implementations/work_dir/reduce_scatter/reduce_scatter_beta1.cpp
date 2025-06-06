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



static int calculate_max_datablocks_per_round(int r, int w) {
    int max_db_per_round = 1;
    for (int i = 0; i < w - 1; i++) {
        if (max_db_per_round > INT_MAX / r) {
            return -1;
        }
        max_db_per_round *= r;
    }
    return max_db_per_round;
}


int reduce_scatter_radix_block(char *sendbuf, char *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int r , int b) {


    int rank, nprocs, typesize;
    MPI_Comm_rank(comm, &rank); // get rank
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typesize);



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

    
    memcpy(recvbuf, &sendbuf[rank*size], size);
    

    

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
                mpi_errno = MPI_Reduce_local(&temp[i], recvbuf,count, datatype, op);
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
                            mpi_errno = MPI_Reduce_local(&temp_recv_buffer[offset], recvbuf,count, datatype, op);
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

    free(extra_buffer);
    free(temp_recv_buffer);

    free(temp_send_buffer);
    free(temp);
    free(stats);
    free(reqs);
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <r> <b>\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Parse r and b
    int r = std::atoi(argv[1]);
    int b = std::atoi(argv[2]);

    // Each rank will receive `count` ints.
    const int count = 8;
    // Total elements in sendbuf = count * size
    const int total_send_elems = count * size;

    // Allocate sendbuf and recvbuf with vectors
    std::vector<int> sendbuf(total_send_elems);
    std::vector<int> recvbuf(count);

    // Fill sendbuf so that element (j*count + k) = rank*1000 + (j*count + k)
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < count; ++k) {
            int idx = j * count + k;
            sendbuf[idx] = rank * 1000 + idx;
        }
    }

    //
    // === Gather all initial send-buffers at rank 0 ===
    //
    // We want to write each rank’s initial sendbuf into the file.
    std::vector<int> all_sendbuf; 
    if (rank == 0) {
        // root will hold size * total_send_elems ints
        all_sendbuf.resize(size * total_send_elems);
    }
    MPI_Gather(
        sendbuf.data(),                 // send buffer
        total_send_elems,               // sendcount
        MPI_INT,
        (rank == 0 ? all_sendbuf.data() : nullptr),
        total_send_elems,               // recvcount at root
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    //
    // === Now call your reduce_scatter_radix_block ===
    //
    int mpi_err = reduce_scatter_radix_block(
        reinterpret_cast<char*>(sendbuf.data()),
        reinterpret_cast<char*>(recvbuf.data()),
        count,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD,
        r, b
    );
    // Create recvcounts array - each process gets 'count' elements
    // int* recvcounts = new int[size];
    // for (int i = 0; i < size; i++) {
    //     recvcounts[i] = count;
    // }
    
    // int mpi_err = MPI_Reduce_scatter(sendbuf.data(), recvbuf.data(), recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // // Don't forget to free the memory
    // delete[] recvcounts;
    if (mpi_err != MPI_SUCCESS) {
        std::cerr << "Rank " << rank << ": reduce_scatter_radix_block failed\n";
        MPI_Abort(MPI_COMM_WORLD, mpi_err);
    }

    //
    // === Gather all final recv-buffers at rank 0 ===
    //
    std::vector<int> all_recvbuf;
    if (rank == 0) {
        all_recvbuf.resize(size * count);
    }
    MPI_Gather(
        recvbuf.data(),   // send buffer
        count,
        MPI_INT,
        (rank == 0 ? all_recvbuf.data() : nullptr),
        count,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    //
    // === Rank 0 writes everything to "all_buffers.txt" ===
    //
    if (rank == 0) {
        std::ofstream ofs("all_buffers.txt");
        if (!ofs.is_open()) {
            std::cerr << "Could not open all_buffers.txt for writing\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 1) Write initial send-buffers
        ofs << "## initial send buffers (r=" << r << ", b=" << b << ")\n\n";
        for (int src = 0; src < size; ++src) {
            ofs << "SendRank " 
                << (src < 10 ? " " : "") << src << ":";

            // For rank=src, its sendbuf occupies:
            //   all_sendbuf[ src * total_send_elems + 0 ... src * total_send_elems + (total_send_elems-1) ]
            int base = src * total_send_elems;
            for (int idx = 0; idx < total_send_elems; ++idx) {
                ofs << " " << all_sendbuf[base + idx];
            }
            ofs << "\n";
        }
        ofs << "\n";

        // 2) Write final recv-buffers
        ofs << "## reduce_scatter_radix_block results (r=" << r << ", b=" << b << ")\n"
            << "## Each row = MPI rank, columns = recvbuf[0.." << (count - 1) << "]\n\n";

        for (int src = 0; src < size; ++src) {
            ofs << "Rank " << (src < 10 ? " " : "") << src << ":";
            for (int k = 0; k < count; ++k) {
                int val = all_recvbuf[src * count + k];
                ofs << " " << val;
            }
            ofs << "\n";
        }

        ofs.close();
        std::cout << "All send & recv buffers written to all_buffers.txt\n";
    }

    MPI_Finalize();
    return 0;
}