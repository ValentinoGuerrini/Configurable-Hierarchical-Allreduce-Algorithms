#include "inter_allgather_linear.hpp"


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

#ifdef DEBUG


int main(){
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int b = 3;
    int sendcount = b*3; //every process have 3 data_blocks
    int recv_size = std::floor(nprocs/(b*b));
    recv_size = rank%b < (nprocs/b)%b ? recv_size + 1 : recv_size;
    int* sendbuf = (int*) malloc(sendcount * sizeof(int));
    int* recvbuf = (int*) malloc(recv_size * sendcount * sizeof(int));

    for(int i=0; i<sendcount; i++){
        sendbuf[i] = rank + i*10000;
    }

    if(rank % b == (rank/b)%b){
        //print sendbuf
        std::cout << "rank: " << rank << " sendbuf: ";
        for(int i=0; i<sendcount; i++){
            std::cout << sendbuf[i] << " ";
        }
        std::cout << std::endl;
    }

    inter_allgather_linear((char*)sendbuf, 3, MPI_INT, (char*)recvbuf, MPI_COMM_WORLD, b);

    //every rank print his buffer
    std::cout << "rank: " << rank << " recvbuf: ";
    for(int i=0; i<sendcount*recv_size; i++){
        // if(i < sendcount || rank%b < (nprocs/b)%b)
        std::cout << recvbuf[i] << " ";
    }
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}

#endif
