#include "reduce_scatter.hpp"

#define MAX_RADIX 8


static int MPICH_Recexchalgo_step2rank_to_origrank(int rank, int rem, int T, int k)
{
    int orig_rank;

    orig_rank = (rank < rem / (k - 1)) ? (rank * k) + (k - 1) : rank + rem;

    return orig_rank;
}

static int MPICH_Recexchalgo_origrank_to_step2rank(int rank, int rem, int T, int k)
{
    int step2rank;

    step2rank = (rank < T) ? rank / k : rank - rem;

    return step2rank;
}



static int MPICH_Recexchalgo_get_count_and_offset(int rank, int phase, int k, int nranks, int *count,
                                          int *offset)
{
    int mpi_errno = MPI_SUCCESS;
    int step2rank, min, max, orig_max, orig_min;
    int k_power_phase = 1;
    int p_of_k = 1, rem, T;


    /* p_of_k is the largest power of k that is less than nranks */
    while (p_of_k <= nranks) {
        p_of_k *= k;
    }
    p_of_k /= k;

    rem = nranks - p_of_k;
    T = (rem * k) / (k - 1);

    /* k_power_phase is k^phase */
    while (phase > 0) {
        k_power_phase *= k;
        phase--;
    }
    /* Calculate rank in step2 */
    step2rank = MPICH_Recexchalgo_origrank_to_step2rank(rank, rem, T, k);
    /* Calculate min and max ranks of the range of ranks that 'rank'
     * represents in phase 'phase' */
    min = ((step2rank / k_power_phase) * k_power_phase) - 1;
    max = min + k_power_phase;
    /* convert (min,max] to their original ranks */
    orig_min = (min >= 0) ? MPICH_Recexchalgo_step2rank_to_origrank(min, rem, T, k) : min;
    orig_max = MPICH_Recexchalgo_step2rank_to_origrank(max, rem, T, k);
    *count = orig_max - orig_min;
    *offset = orig_min + 1;



    return mpi_errno;
}

static int MPICH_Recexchalgo_reverse_digits_step2(int rank, int comm_size, int k)
{
    int i, T, rem, power, step2rank, step2_reverse_rank = 0;
    int pofk = 1, log_pofk = 0;
    int *digit, *digit_reverse;


    while (pofk <= comm_size) {
        pofk *= k;
        log_pofk++;
    }
    if(!(log_pofk > 0)) {
        std::cout<<"Error in calculating log_pofk"<<std::endl;
        return MPI_ERR_OTHER;
    }
    pofk /= k;
    log_pofk--;

    rem = comm_size - pofk;
    T = (rem * k) / (k - 1);

    /* step2rank is the rank in the particiapting ranks group of recursive exchange step2 */
    step2rank = MPICH_Recexchalgo_origrank_to_step2rank(rank, rem, T, k);

    /* calculate the digits in base k representation of step2rank */
    digit = (int*) malloc(sizeof(int) * log_pofk);
    digit_reverse = (int*) malloc( sizeof(int) * log_pofk);

    for (i = 0; i < log_pofk; i++)
        digit[i] = 0;

    int remainder, i_digit = 0;
    while (step2rank != 0) {
        remainder = step2rank % k;
        step2rank = step2rank / k;
        digit[i_digit] = remainder;
        i_digit++;
    }

    /* reverse the number in base k representation to get the step2_reverse_rank
     * which is the reversed rank in the participating ranks group of recursive exchange step2
     */
    for (i = 0; i < log_pofk; i++)
        digit_reverse[i] = digit[log_pofk - 1 - i];
    /*calculate the base 10 value of the reverse rank */
    step2_reverse_rank = 0;
    power = 1;
    for (i = 0; i < log_pofk; i++) {
        step2_reverse_rank += digit_reverse[i] * power;
        power *= k;
    }

    /* calculate the actual rank from logical rank */
    step2_reverse_rank = MPICH_Recexchalgo_step2rank_to_origrank(step2_reverse_rank, rem, T, k);

    free(digit);
    free(digit_reverse);
    return step2_reverse_rank;
}

static int MPICH_Recexchalgo_get_neighbors(int rank, int nranks, int *k_, int *step1_sendto, int **step1_recvfrom_, int *step1_nrecvs, int ***step2_nbrs_, int *step2_nphases, int *p_of_k_, int *T_){
    int mpi_errno = MPI_SUCCESS;
    int i, j, k;
    int p_of_k = 1, log_p_of_k = 0, rem, T, newrank;
    int **step2_nbrs;
    int *step1_recvfrom;

    k = *k_;
    if (nranks < k)     /* If size of the communicator is less than k, reduce the value of k */
        k = (nranks > 2) ? nranks : 2;
    *k_ = k;

    while (p_of_k <= nranks) {
        p_of_k *= k;
        log_p_of_k++;
    }
    p_of_k /= k;
    log_p_of_k--;

    step1_recvfrom = *step1_recvfrom_ = (int *) malloc(sizeof(int) * (k - 1));
    step2_nbrs = *step2_nbrs_ = (int **) malloc(sizeof(int *) * log_p_of_k);



    if(step1_recvfrom == NULL || step2_nbrs == NULL) {
        std::cout<<"Error in malloc"<<std::endl;
        return MPI_ERR_OTHER;
    }

    for (i = 0; i < log_p_of_k; i++) {
        (*step2_nbrs_)[i] = (int *) malloc(sizeof(int) * (k - 1));
    }


    *step2_nphases = log_p_of_k;

    rem = nranks - p_of_k;

    T = (rem * k) / (k - 1);
    *T_ = T;
    *p_of_k_ = p_of_k;

    *step1_nrecvs = 0;
    *step1_sendto = -1;

    if (rank < T) {
        if (rank % k != (k - 1)) {      /* I am a non-participating rank */
            *step1_sendto = rank + (k - 1 - rank % k);  /* partipating rank to send the data to */
            /* if the corresponding participating rank is not in T,
             * then send to the Tth rank to preserve non-commutativity */
            if (*step1_sendto > T - 1)
                *step1_sendto = T;
            newrank = -1;       /* tag this rank as non-participating */
        } else {        /* participating rank */
            for (i = 0; i < k - 1; i++) {
                step1_recvfrom[i] = rank - i - 1;
            }
            *step1_nrecvs = k - 1;
            newrank = rank / k; /* this is the new rank amongst the set of participating ranks */
        }
    } else {    /* rank >= T */
        newrank = rank - rem;

        if (rank == T && (T - 1) % k != k - 1 && T >= 1) {
            int nsenders = (T - 1) % k + 1;     /* number of ranks sending their data to me in Step 1 */

            for (j = nsenders - 1; j >= 0; j--) {
                step1_recvfrom[nsenders - 1 - j] = T - nsenders + j;
            }
            *step1_nrecvs = nsenders;
        }
    }

    if (*step1_sendto == -1) {  /* calculate step2_nbrs only for participating ranks */
        int *digit = (int *) malloc(sizeof(int) * log_p_of_k);
        if(digit == NULL) {
            std::cout<<"Error in malloc"<<std::endl;
            return MPI_ERR_OTHER;
        }
        int temprank = newrank;
        int mask = 0x1;
        int phase = 0, cbit, cnt, nbr, power;

        /* calculate the digits in base k representation of newrank */
        for (i = 0; i < log_p_of_k; i++)
            digit[i] = 0;

        int remainder, i_digit = 0;
        while (temprank != 0) {
            remainder = temprank % k;
            temprank = temprank / k;
            digit[i_digit] = remainder;
            i_digit++;
        }

        while (mask < p_of_k) {
            cbit = digit[phase];        /* phase_th digit changes in this phase, obtain its original value */
            cnt = 0;
            for (i = 0; i < k; i++) {   /* there are k-1 neighbors */
                if (i != cbit) {        /* do not generate yourself as your nieighbor */
                    digit[phase] = i;   /* this gets us the base k representation of the neighbor */

                    /* calculate the base 10 value of the neighbor rank */
                    nbr = 0;
                    power = 1;
                    for (j = 0; j < log_p_of_k; j++) {
                        nbr += digit[j] * power;
                        power *= k;
                    }

                    /* calculate its real rank and store it */
                    step2_nbrs[phase][cnt] =
                        (nbr < rem / (k - 1)) ? (nbr * k) + (k - 1) : nbr + rem;
                    cnt++;
                }
            }

            digit[phase] = cbit;        /* reset the digit to original value */
            phase++;
            mask *= k;
        }

        free(digit);
    }

    return mpi_errno;


}

int MPICH_reduce_scatter_rec_halving(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

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



#ifdef DEBUG_MODE

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
    const int count =4;
    // Total elements in sendbuf = count * size
    const int total_send_elems = count * size;

    // Allocate sendbuf and recvbuf with vectors
    std::vector<int> sendbuf(total_send_elems);
    std::vector<int> recvbuf(total_send_elems);

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
    int mpi_err = MPICH_reduce_scatter_rec_halving(
        reinterpret_cast<char*>(sendbuf.data()),
        reinterpret_cast<char*>(recvbuf.data()),
        count,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
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
        all_recvbuf.resize(size * total_send_elems);
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

#endif