#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstring>   // for std::memcpy / memcpy
#include <climits>   // for INT_MAX

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

int all_reduce_radix_batch(char *sendbuf, char *recvbuf, int sendcount,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                int k, int b)
{
    

    int rank, nprocs, typesize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typesize);

    if (sendcount < 0) return MPI_ERR_COUNT;
    if (sendcount == 0) return MPI_SUCCESS;

    // Require equal partition
    if (sendcount % nprocs != 0) {
        return MPI_ERR_COUNT;  // or handle with MPI_Reduce_scatter + MPI_Allgatherv
    }

    int count = sendcount / nprocs;

    //General Checks
    if ( k < 2 ) { k = 2; }
    if ( k > nprocs - 1 ) { k = nprocs - 1; }
    if (b <= 0 || b > nprocs) b = nprocs;


    //for loop indexing
    int i, j, x, z, r, s;
    int mpi_errno = MPI_SUCCESS;
    int num_reqs;
    int rotate_index_array[nprocs];



    //const int w = calculate_w(r, nprocs - 1);  // number of bits required of r representation
    int w = 0;
    int max_rank = nprocs - 1;

    while (max_rank > 0) {
        max_rank /= k;
        w++;
    }

    const int max_db_per_round = calculate_max_datablocks_per_round(k,w);   // maximum send number of elements

    if (max_db_per_round == -1) {
        std::cerr << "Overflow detected in calculate_max_datablocks_per_round" << std::endl;
        return 1; // Exit program with error
    }



    const int highest_digit = (max_db_per_round*k - nprocs) / max_db_per_round; // calculate the number of highest digits

    const int K = w * (k - 1) - highest_digit; // the total number of communication rounds

    const int comm_round = K + 1;
    const int rem2 = k + 1;

    const int size = count * typesize;
    


    char *extra_buffer, *temp_recv_buffer, *temp_send_buffer, *gather_buf;
    int extra_ids[nprocs - rem2];
    memset(extra_ids, -1, sizeof(extra_ids));
    int spoint, distance=1, next_distance = distance*k, di = 0;

    if (K < nprocs - 1) {
        // 1. Find max send count


        // 2. create local index array after rotation
        for (i = 0; i < nprocs; i++) {
            rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;
        }

        // 3. exchange data with log(P) steps
        extra_buffer = (char*) malloc(size * (nprocs - comm_round));
        temp_recv_buffer = (char*) malloc(size * nprocs);///this can be modified for reduce
        gather_buf = (char*) malloc(size * nprocs);
        temp_send_buffer = (char*) malloc(size * max_db_per_round);


        CHECK_ALLOCATIONS(extra_buffer == nullptr || temp_recv_buffer == nullptr || temp_send_buffer == nullptr);

        for (x = 0; x < w; x++) {
            for (z = 1; z < k; z ++) {
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
            distance *= k;
        }


    }

    // copy data that need to be sent to each rank itself

    
    memcpy(gather_buf, &sendbuf[rank*size], size);
    

    

    int sent_blocks[k-1][max_db_per_round];

    int nc, rem, ns, ze, ss;
    spoint = 1, distance =1;

    MPI_Request* reqs = (MPI_Request *) malloc(2 * k * sizeof(MPI_Request));


    MPI_Status* stats = (MPI_Status *) malloc(2 * k * sizeof(MPI_Status));


    CHECK_ALLOCATIONS(stats == nullptr || reqs == nullptr);

    char* temp = (char*) malloc(size*max_db_per_round);
    CHECK_ALLOCATIONS(temp == nullptr);
    int count_tmp=0;


    for (x = 0; x < w; x++) { // try to run the firsts steps and check if results are correct
        ze = (x == w - 1)? k - highest_digit: k;
        //to see when the code fails 
        //print at every iteration if things happends between iteration probably is a memory bug !

        int zoffset = 0, zc = ze-1;
        int zns[zc];

        for (r = 1; r < ze; r += b) {
            ss = ze - r < b ? ze - r : b;
            num_reqs = 0;

            

            for (s = 0; s < ss; s++) { // s = 0, s = 1


                z = r + s;



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
                mpi_errno = MPI_Reduce_local(&temp[i], gather_buf,count, datatype, op);
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
                            mpi_errno = MPI_Reduce_local(&temp_recv_buffer[offset], gather_buf,count, datatype, op);
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

        distance *= k;
        next_distance *= k;

    }

    free(extra_buffer);
    free(temp_recv_buffer);

    free(temp_send_buffer);
    free(temp);
    free(stats);
    free(reqs);





    num_reqs = 0;
    sendcount = count;
    count= count * b;

    int node_id = rank/b;
    int node_rank = rank%b;
    int nnodes = nprocs/b;
    const int nstages = nnodes/b;
    int group = rank / b;


    ////---------- PHASE 1 INTRA GATHER K-NOMIAL----------------
    int root_local = node_id%b;
    int shift = (node_rank - root_local + b) % b;
    int max = b - 1, nphases = 0;
    while (max > 0) { ++nphases; max /= k; }
    bool p_of_k = (ipow(k, nphases) == b);

    char* kgather_tmp_buf = (char*)malloc(b * sendcount * typesize);
    if (!kgather_tmp_buf) return MPI_ERR_NO_MEM;

    char* inter_tmp_buf = (char*)malloc(b * sendcount * typesize);
    if (!inter_tmp_buf) return MPI_ERR_NO_MEM;

    std::memcpy(kgather_tmp_buf + shift * sendcount * typesize,
            gather_buf,
            sendcount * typesize);

    // scratch array for up to (k-1) outstanding Irecvs
    reqs = (MPI_Request*)malloc(((nstages+1)*(1+nnodes)+2*(k - 1)) * sizeof(MPI_Request));

    if (!reqs) { free(kgather_tmp_buf); return MPI_ERR_NO_MEM; }

    int phase,delta;
    int src;

    for (phase = 0, delta = 1; phase < nphases; ++phase, delta *= k) {
        int groupSize  = delta * k;
        int gstart     = (shift / groupSize) * groupSize;
        int offset     = shift - gstart;

        // real‐world rank of the normalized 0-root:
        int real_root_local = (gstart + root_local) % b;
        int real_root_glob  = node_id * b + real_root_local;

        if (offset == 0) {
            // I am this subtree’s root → post up to (k−1) Irecvs
            num_reqs =0;
            for (j = 1; j < k; ++j) {
                int childNorm = gstart + j*delta;
                if (childNorm >= b) break;
                int subtree = delta;
                if (/*phase == nphases - 1 && */!p_of_k)
                    subtree = std::min(subtree, b - childNorm);

                // recv into the *normalized* slot “childNorm”
                int src_local = (childNorm + root_local) % b;
                src  = node_id * b + src_local;
                MPI_Irecv(kgather_tmp_buf + childNorm * sendcount * typesize,
                          subtree * sendcount, datatype,
                          src, phase,
                          comm, &reqs[num_reqs++]);
            }
            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        }
        else if ((offset % delta) == 0 && offset < groupSize) {
            // I am a child‐subtree root → send my normalized block up
            int subtree = delta;
            if (/*phase == nphases - 1 && */!p_of_k)
                subtree = std::min(subtree, b - shift);

            MPI_Request sreq;
            MPI_Isend(kgather_tmp_buf + shift * sendcount * typesize,
                      subtree * sendcount, datatype,
                      real_root_glob, phase,
                      comm, &sreq);
            MPI_Wait(&sreq, MPI_STATUS_IGNORE);
            break;
        }
    }

    if (node_rank == root_local) {
        for (int real_slot = 0; real_slot < b; ++real_slot) {
            // find which normalized index holds that real_slot
            int norm_i = (real_slot - root_local + b) % b;
            std::memcpy(inter_tmp_buf + real_slot * sendcount * typesize,
                        kgather_tmp_buf + norm_i    * sendcount * typesize,
                        sendcount * typesize);
        }
    }
    


    ////-------------------------------------------------------

    

    /// ------------- PHASE 2 INTER ALLGATHER LINEAR-----------

    bool rootCond;
    

    char* intra_tmp_buf = (char*)malloc((nstages + 1)*count*typesize);
    if (!intra_tmp_buf) return MPI_ERR_NO_MEM;


    for(i = 0; i<nnodes; i+=b){
        num_reqs = 0;
        rootCond = ((node_id - i) == node_rank);
        if(!rootCond && (i+node_rank < nnodes)){
            MPI_Irecv(intra_tmp_buf+count*(i/b)*typesize, count, datatype, (i+node_rank)*b + node_rank, 1, comm, &reqs[num_reqs++]);
        }else if(rootCond){
            memcpy(intra_tmp_buf+count*(i/b)*typesize, inter_tmp_buf, count * typesize);
                for(j=0; j<nnodes; j++){
                    if(j==node_id )
                        continue;
                    int dst = j*b + node_rank;
                    MPI_Isend(inter_tmp_buf, count, datatype, dst, 1, comm, &reqs[num_reqs++]);
                }
        }
        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    }




    ///--------------------------------------------------------
    

    ///-----------PHASE 3 INTRA ALLGATHER K BRUCKS-------------

    char* tmp_recv_buffer;
    

    if(node_rank == 0) {
        tmp_recv_buffer = recvbuf;
    } else {
        tmp_recv_buffer = (char*) malloc((nstages*b + nnodes%b) * count * typesize);  // Fixed 'size' to 'nprocs'
        if (tmp_recv_buffer == NULL) {
            free(reqs);
            return MPI_ERR_NO_MEM;
        }
    }

    int dst;
    delta = 1;

    for(s = 0; s < nstages; s++){

        memcpy(tmp_recv_buffer, intra_tmp_buf, count * typesize);
        
        for(i = 0; i < nphases; i++) {
            num_reqs = 0;
            for(j = 1; j < k; j++) {
                if(delta * j >= b)  // Fixed 'size' to 'nprocs'
                    break;
                    
                dst = (b + (node_rank - delta * j)) % b + group*b;
                src = (node_rank + delta * j) % b + group*b;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

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

                MPI_Irecv(tmp_recv_buffer + j * count * delta * typesize, 
                            tmp_count, datatype, src, 1, comm, &reqs[num_reqs++]);

                MPI_Isend(tmp_recv_buffer, tmp_count, datatype, 
                            dst, 1, comm, &reqs[num_reqs++]);
            }
            
            MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
            delta *= k;
        }

        if(node_rank != 0) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    tmp_recv_buffer + (b - node_rank) * count * typesize, 
                    node_rank * count * typesize);

            memcpy(recvbuf + node_rank * count * typesize, 
                    tmp_recv_buffer, 
                    (b - node_rank) * count * typesize);
                    
            //free(tmp_recv_buffer);
        }
        delta = 1;

        tmp_recv_buffer += b * count * typesize;
        recvbuf += b*count*typesize;
        intra_tmp_buf += count*typesize;
    }

    int nu_count = nnodes%b;

    if(nu_count != 0) {

        if(node_rank < nu_count) {
            memcpy(tmp_recv_buffer, intra_tmp_buf, count * typesize);
        } else {
            free(tmp_recv_buffer-nstages*b*count*typesize);
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

            received = send_sizes[i][node_rank];
            
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

                

                

                isrc = (node_rank + delta * j) % b;
                idst = (b + (node_rank - delta * j)) % b;
                    
                dst = idst + group*b;
                src = isrc + group*b;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

                
                int ssssize;
                ssssize = min(send_sizes[i][isrc], (nu_count) * count - received);
                if(active[isrc] == i && ssssize > 0){
                    MPI_Irecv(tmp_recv_buffer + received * typesize, 
                            ssssize, datatype, src, 1, comm, &reqs[num_reqs++]);
                    
                    received += ssssize;
                }
                ssssize = min(send_sizes[i][node_rank], (nu_count) * count - (send_sizes[i][idst] + send_sizes[i+1][idst]));

                if(active[node_rank] == i && ssssize > 0){
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

        if(node_rank != 0 && node_rank < nu_count) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    tmp_recv_buffer + ((nu_count) - node_rank) * count * typesize, 
                    node_rank * count * typesize);

            memcpy(recvbuf + node_rank * count * typesize, 
                    tmp_recv_buffer, 
                    ((nu_count) - node_rank) * count * typesize);

            // memcpy(recvbuf, tmp_recv_buffer, count* (nnodes%b) * typesize);
            free(tmp_recv_buffer-nstages*b*count*typesize);
        }

    }else if(node_rank != 0){
        free(tmp_recv_buffer-nstages*b*count*typesize);
    }


    free(intra_tmp_buf-nstages*count*typesize);

    free(kgather_tmp_buf);

    free(inter_tmp_buf);

    free(reqs);


    return MPI_SUCCESS;




}
