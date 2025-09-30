#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstring>   // for std::memcpy / memcpy


int allgather_radix_batch(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b);



int MPICH_reduce_scatter(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    int rank, comm_size, i;
    int extent;
    int *disps;
    void *tmp_recvbuf, *tmp_results;
    int mpi_errno = MPI_SUCCESS;
    int total_count, dst;
    int mask;
    int rem, newdst, send_idx, recv_idx, last_idx, send_cnt, recv_cnt;
    int pof2, old_i, newrank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Type_size(datatype, &extent);

    disps = (int*) malloc(sizeof(int) * comm_size);

    total_count = 0;

    for (i = 0; i < comm_size; i++) {
        disps[i] = total_count;
        total_count += count;
    }


    tmp_recvbuf = (char*) malloc(extent * total_count);
    tmp_results = (char*) malloc(extent * total_count);

    if(sendbuf != MPI_IN_PLACE) {
        memcpy((char*)tmp_results, sendbuf, extent * total_count);
    } else {
        memcpy((char*)tmp_results, recvbuf, extent * total_count);
    }

    pof2 = 1;
    while (pof2 <= comm_size) {
        pof2 *= 2;
    
    }

    pof2 = pof2 == comm_size ? pof2 : pof2 / 2;

    rem = comm_size - pof2;

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { 
            mpi_errno = MPI_Send(tmp_results, total_count, datatype, rank + 1, 0, comm);
            newrank = -1;

        }else{
            mpi_errno = MPI_Recv(tmp_recvbuf, total_count, datatype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            

            mpi_errno = MPI_Reduce_local(tmp_recvbuf, tmp_results, total_count, datatype, op);
            newrank = rank / 2;

        }

    }else{
        newrank = rank - rem;
    }

    if (newrank != -1){
        int *newcnts, *newdisps;
        newcnts = (int*) malloc(sizeof(int) * pof2);
        newdisps = (int*) malloc(sizeof(int) * pof2);

        for (i = 0; i < pof2; i++) {
            old_i = (i < rem) ? i * 2 + 1 : i + rem;
            if (old_i < 2 * rem) {
                newcnts[i] = count * 2;
            } else {
                newcnts[i] = count;
            }
        }
        newdisps[0] = 0;
        for (i = 1; i < pof2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }
        mask = pof2 >> 1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + mask;
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
            } else {
                recv_idx = send_idx + mask;
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            if((send_cnt != 0) && (recv_cnt != 0)){
                mpi_errno = MPI_Sendrecv((char*)tmp_results + newdisps[send_idx] * extent, send_cnt, datatype, dst, 0,
                                    (char*)tmp_recvbuf + newdisps[recv_idx] * extent, recv_cnt, datatype, dst, 0, comm, MPI_STATUS_IGNORE);
            }else if ((send_cnt == 0) && (recv_cnt != 0)){
                mpi_errno = MPI_Recv((char*)tmp_recvbuf + newdisps[recv_idx] * extent, recv_cnt, datatype, dst, 0, comm, MPI_STATUS_IGNORE);
            }else if ((send_cnt != 0) && (recv_cnt == 0)){
                mpi_errno = MPI_Send((char*)tmp_results + newdisps[send_idx] * extent, send_cnt, datatype, dst, 0, comm);
            }

            if(recv_cnt){
                mpi_errno = MPI_Reduce_local((char*)tmp_recvbuf + newdisps[recv_idx] * extent,
                                        (char*)tmp_results + newdisps[recv_idx] * extent, recv_cnt, datatype, op);
            }

            send_idx = recv_idx;
            last_idx = recv_idx + mask;
            mask >>=1;
        }
        memcpy(recvbuf, (char*)tmp_results + disps[rank] * extent, count * extent);

        free(newcnts);
        free(newdisps);

    }

    if(rank < 2 * rem){
        if (rank % 2){
            mpi_errno = MPI_Send((char *)tmp_results + disps[rank-1] * extent, count, datatype, rank - 1, 0, comm);
        }else{
            mpi_errno = MPI_Recv(recvbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    free(tmp_recvbuf);
    free(tmp_results);
    free(disps);

    return mpi_errno;



}

int all_reduce_semi_radix_batch(char *sendbuf, char *recvbuf, int count,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                int k, int b)
{
    

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
    rc = MPICH_reduce_scatter(sendbuf, tmp, chunk, datatype, op, comm);
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