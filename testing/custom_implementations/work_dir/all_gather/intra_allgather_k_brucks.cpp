#include "intra_allgather_k_brucks.hpp"


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


int debug_prototype(char* sendbuf, int count, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b){
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


    if(b_rank < nnodes%b) {
        memcpy(tmp_recv_buffer, sendbuf, count * typeSize);
        //curr_cnt = count;
    } else {
        //curr_cnt = 0;
    }

    int active[b];
    int send_sizes[nphases+1][b];
    memset(send_sizes, 0, (nphases+1)*b*sizeof(int));
    //sizes array

    for(i =0; i< b; i++){
        active[i] = i < nnodes%b ? 0 : -1;
        send_sizes[0][i] = i < nnodes%b ? count : 0;

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
            ssssize = min(send_sizes[i][isrc], (nnodes%b) * count - received);
            if(active[isrc] == i && ssssize > 0){
                MPI_Irecv(tmp_recv_buffer + received * typeSize, 
                        ssssize, datatype, src, 1, comm, &reqs[num_reqs++]);
                
                received += ssssize;
            }
            ssssize = min(send_sizes[i][b_rank], (nnodes%b) * count - (send_sizes[i][idst] + send_sizes[i+1][idst]));

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

    if(b_rank != 0 && b_rank < nnodes%b) {
        // Rearrange data in the correct order
        memcpy(recvbuf, 
                tmp_recv_buffer + ((nnodes%b) - b_rank) * count * typeSize, 
                b_rank * count * typeSize);

        memcpy(recvbuf + b_rank * count * typeSize, 
                tmp_recv_buffer, 
                ((nnodes%b) - b_rank) * count * typeSize);

        // memcpy(recvbuf, tmp_recv_buffer, count* (nnodes%b) * typeSize);
        free(tmp_recv_buffer-nstages*b*count*typeSize);
    }else if (b_rank != 0) {
        memcpy(recvbuf, tmp_recv_buffer, count* (nnodes%b) * typeSize);
        free(tmp_recv_buffer-nstages*b*count*typeSize);
    }

            
    


    free(reqs);
    return MPI_SUCCESS;

}


#ifdef DEBUG

// int main(){
//     int rank, nprocs, nnodes;

//     MPI_Init(NULL, NULL);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

//     int k = 3;
//     int b = 5;
//     int per_process_count = 2;
//     nnodes = nprocs / b;

//     int sendsize = floor(nnodes/b)*per_process_count + 1*per_process_count;

//     //sendsize = rank%b < nnodes%b ? sendsize + 1*per_process_count : sendsize;


//     int* sendbuf = (int*) malloc(sendsize * sizeof(int));
//     int* recvbuf = (int*) malloc(nprocs * per_process_count * sizeof(int));

//     for(int i = 0; i < sendsize; i++) {
//         if(rank%b < nnodes%b)
//             sendbuf[i] = rank%b + 1 + i*10000;
//         else
//             sendbuf[i] = -1;
//     }

//     //print sendbuf
//     std::cout << "rank: " << rank << " sendbuf: ";
//     for(int i=0; i<sendsize; i++){
//         std::cout << sendbuf[i] << " ";
//     }
//     std::cout << std::endl;



//     MPI_Barrier(MPI_COMM_WORLD);

//     intra_allgather_k_brucks((char*)sendbuf, per_process_count, MPI_INT, (char*)recvbuf, MPI_COMM_WORLD, k, b);

//     MPI_Barrier(MPI_COMM_WORLD);

//     // intra_allgather_k_brucks((char*)sendbuf, per_process_count, MPI_INT, (char*)recvbuf, MPI_COMM_WORLD, k, b);

//     //print recvbuf
//      std::cout << "rank: " << rank << " recvbuf: ";
//     for(int i=0; i<sendsize*nnodes; i++){
//         std::cout << recvbuf[i] << " ";
//     }
//     std::cout << std::endl;

//     MPI_Finalize();
//     return 0;
// }






int main(){
    int rank, nprocs, nnodes;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int k = 2;
    int b = 5;
    int per_process_count = 2;
    nnodes = nprocs / b;

    int sendsize = floor(nnodes/b)*per_process_count;

    sendsize = rank%b < nnodes%b ? sendsize + 1*per_process_count : sendsize;


    int* sendbuf = (int*) malloc(sendsize * sizeof(int));
    int* recvbuf = (int*) malloc(nprocs * per_process_count * sizeof(int));

    for(int i = 0; i < sendsize; i++) {
        sendbuf[i] = rank%b + i*10000;
    }

    //print sendbuf
    std::cout << "rank: " << rank << " sendbuf: ";
    for(int i=0; i<sendsize; i++){
        std::cout << sendbuf[i] << " ";
    }
    std::cout << std::endl;



    MPI_Barrier(MPI_COMM_WORLD);
    

    intra_allgather_k_brucks((char*)sendbuf, per_process_count, MPI_INT, (char*)recvbuf, MPI_COMM_WORLD, k, b);

    //print recvbuf
     std::cout << "rank: " << rank << " recvbuf: ";
    for(int i=0; i<(nnodes) * per_process_count; i++){
        std::cout << recvbuf[i] << " ";
    }
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}


#endif