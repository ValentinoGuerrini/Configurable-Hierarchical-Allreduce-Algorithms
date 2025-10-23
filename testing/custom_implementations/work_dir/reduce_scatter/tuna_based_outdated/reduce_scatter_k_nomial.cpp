// reduce_scatter_k_nomial.cpp
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

/* ---------- helpers ---------- */

static inline int sum_range_int(const int *a, int L, int R) {
    int s = 0;
    for (int i = L; i < R; ++i) s += a[i];
    return s;
}

static inline int map_newidx_to_oldrank(int newidx, int rem, int k) {
    // First 'rem' indices correspond to proxies of the first 'rem' k-groups.
    // Each proxy maps to old rank proxy = g*k. Others are compacted with an offset.
    return (newidx < rem) ? (newidx * k) : (newidx + rem * (k - 1));
}

/* ---------- the k-nomial reduce-scatter (block) ---------- */

int MPICH_reduce_scatter_k_nomial(const char* sendbuf, char* recvbuf,
                                  int count, MPI_Datatype datatype, MPI_Op op,
                                  MPI_Comm comm, int k)
{
    // ---- Declarations up front (avoid goto over inits) ----
    int rank = 0, comm_size = 0, i = 0;
    int extent = 0;
    int *disps = nullptr;
    void *tmp_recvbuf = nullptr, *tmp_results = nullptr;
    int mpi_errno = MPI_SUCCESS;
    int total_count = 0;
    int pofk = 1;
    int rem = 0;
    int newrank = -1;
    int *newcnts = nullptr;
    int *newdisps = nullptr;

    if (k < 2) return MPI_ERR_ARG;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Type_size(datatype, &extent);

    disps = (int*) std::malloc(sizeof(int) * (size_t)comm_size);
    if (!disps) { mpi_errno = MPI_ERR_NO_MEM; goto cleanup_all; }

    // Build original displacements for uniform 'count'
    total_count = 0;
    for (i = 0; i < comm_size; i++) {
        disps[i] = total_count;
        total_count += count;
    }

    tmp_recvbuf  = std::malloc((size_t)extent * (size_t)total_count);
    tmp_results  = std::malloc((size_t)extent * (size_t)total_count);
    if (!tmp_recvbuf || !tmp_results) { mpi_errno = MPI_ERR_NO_MEM; goto cleanup_all; }

    if (sendbuf != (const char*) MPI_IN_PLACE) {
        std::memcpy(tmp_results, sendbuf, (size_t)extent * (size_t)total_count);
    } else {
        std::memcpy(tmp_results, recvbuf, (size_t)extent * (size_t)total_count);
    }

    // Compute largest power of k not exceeding comm_size
    pofk = 1;
    while ((long long)pofk * (long long)k <= comm_size) pofk *= k;
    rem = comm_size - pofk;

    // ---- pre-compression: fold first rem groups of size k onto their proxy (pos==0) ----
    if (rank < rem * k) {
        int g = rank / k;      // group id [0..rem-1]
        int pos = rank % k;    // position in group
        int proxy = g * k;

        if (pos == 0) {
            // proxy collects and reduces k-1 full buffers
            for (int t = 1; t < k; ++t) {
                mpi_errno = MPI_Recv(tmp_recvbuf, total_count, datatype,
                                     proxy + t, 0, comm, MPI_STATUS_IGNORE);
                if (mpi_errno) goto cleanup_all;

                mpi_errno = MPI_Reduce_local(tmp_recvbuf, tmp_results,
                                             total_count, datatype, op);
                if (mpi_errno) goto cleanup_all;
            }
            newrank = g; // compressed indices 0..rem-1 are proxies
        } else {
            // non-proxy sends its full buffer and becomes inactive
            mpi_errno = MPI_Send(tmp_results, total_count, datatype, proxy, 0, comm);
            if (mpi_errno) goto cleanup_all;
            newrank = -1;
        }
    } else {
        // ranks beyond front (rem*k) are compacted
        newrank = rank - rem * (k - 1);
    }

    // ---- k-nomial RS core over the pofk active indices ----
    if (newrank != -1) {
        newcnts  = (int*) std::malloc(sizeof(int) * (size_t)pofk);
        newdisps = (int*) std::malloc(sizeof(int) * (size_t)pofk);
        if (!newcnts || !newdisps) { mpi_errno = MPI_ERR_NO_MEM; goto cleanup_all; }

        for (i = 0; i < pofk; ++i) newcnts[i] = (i < rem) ? (k * count) : count;
        newdisps[0] = 0;
        for (i = 1; i < pofk; ++i) newdisps[i] = newdisps[i - 1] + newcnts[i];

        // group_len starts as pofk and shrinks by factor k each round
        int group_len = pofk;
        while (group_len > 1) {
            int part_len = group_len / k;                 // each of k parts
            int base     = (newrank / group_len) * group_len;
            int d        = (newrank - base) / part_len;   // our digit 0..k-1
            int offset   = newrank - (base + d * part_len);

            // We will keep part 'd'; exchange the other (k-1) parts
            for (int t = 0; t < k; ++t) {
                if (t == d) continue;

                int newdst_idx = base + t * part_len + offset;
                int real_dst   = map_newidx_to_oldrank(newdst_idx, rem, k);

                // Send: indices of part t; Recv: indices of part d
                int send_jb = base + t * part_len;
                int send_je = send_jb + part_len;
                int recv_jb = base + d * part_len;
                int recv_je = recv_jb + part_len;

                int send_cnt = sum_range_int(newcnts, send_jb, send_je);
                int recv_cnt = sum_range_int(newcnts, recv_jb, recv_je);

                char* send_ptr = (char*)tmp_results + (size_t)extent * (size_t)newdisps[send_jb];
                char* recv_ptr = (char*)tmp_recvbuf + (size_t)extent * (size_t)newdisps[recv_jb];

                if (send_cnt && recv_cnt) {
                    mpi_errno = MPI_Sendrecv(send_ptr, send_cnt, datatype, real_dst, 0,
                                             recv_ptr, recv_cnt, datatype, real_dst, 0,
                                             comm, MPI_STATUS_IGNORE);
                    if (mpi_errno) goto cleanup_all;
                } else if (!send_cnt && recv_cnt) {
                    mpi_errno = MPI_Recv(recv_ptr, recv_cnt, datatype, real_dst, 0, comm, MPI_STATUS_IGNORE);
                    if (mpi_errno) goto cleanup_all;
                } else if (send_cnt && !recv_cnt) {
                    mpi_errno = MPI_Send(send_ptr, send_cnt, datatype, real_dst, 0, comm);
                    if (mpi_errno) goto cleanup_all;
                }

                if (recv_cnt) {
                    mpi_errno = MPI_Reduce_local(
                        (char*)tmp_recvbuf + (size_t)extent * (size_t)newdisps[recv_jb],
                        (char*)tmp_results + (size_t)extent * (size_t)newdisps[recv_jb],
                        recv_cnt, datatype, op);
                    if (mpi_errno) goto cleanup_all;
                }
            }

            group_len = part_len; // shrink
        }

        // At this point, the reduced value for each k-nomial index i resides at
        // tmp_results[newdisps[i] .. newdisps[i] + newcnts[i]).
        // Copy *our* original-rank piece into recvbuf, using (i_new, tslot).
        int i_new = 0, tslot = 0;
        if (rank < rem * k) {      // in proxy region
            i_new  = rank / k;     // proxy index 0..rem-1
            tslot  = rank % k;     // slot within proxy block
        } else {
            i_new  = rank - rem * (k - 1);
            tslot  = 0;
        }
        size_t src_off_elems = (size_t)newdisps[i_new] + (size_t)tslot * (size_t)count;
        std::memcpy(recvbuf,
                    (char*)tmp_results + (size_t)extent * src_off_elems,
                    (size_t)extent * (size_t)count);
    }

    // ---- post-expansion: proxies send to dropped-out partners ----
    if (rank < rem * k) {
        int g = rank / k;
        int pos = rank % k;
        int proxy = g * k;

        if (pos == 0) {
            // Rebuild minimal newdisps/newcnts (cheap; ensures availability here)
            int *nc = (int*) std::malloc(sizeof(int) * (size_t)pofk);
            int *nd = (int*) std::malloc(sizeof(int) * (size_t)pofk);
            if (!nc || !nd) { if (nc) std::free(nc); if (nd) std::free(nd); mpi_errno = MPI_ERR_NO_MEM; goto cleanup_all; }
            for (i = 0; i < pofk; ++i) nc[i] = (i < rem) ? (k * count) : count;
            nd[0] = 0; for (i = 1; i < pofk; ++i) nd[i] = nd[i - 1] + nc[i];

            int i_new = g; // proxy index in 0..rem-1
            for (int t = 1; t < k; ++t) {
                size_t src_off_elems = (size_t)nd[i_new] + (size_t)t * (size_t)count;
                mpi_errno = MPI_Send((char*)tmp_results + (size_t)extent * src_off_elems,
                                     count, datatype, proxy + t, 0, comm);
                if (mpi_errno) { std::free(nc); std::free(nd); goto cleanup_all; }
            }
            // Proxy's own piece (t=0) was already copied above.
            std::free(nc); std::free(nd);
        } else {
            mpi_errno = MPI_Recv(recvbuf, count, datatype, proxy, 0, comm, MPI_STATUS_IGNORE);
            if (mpi_errno) goto cleanup_all;
        }
    }

cleanup_all:
    if (newcnts)  std::free(newcnts);
    if (newdisps) std::free(newdisps);
    if (tmp_recvbuf) std::free(tmp_recvbuf);
    if (tmp_results) std::free(tmp_results);
    if (disps) std::free(disps);
    return mpi_errno;
}

/* ---------- test harness main ---------- */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::printf("Usage: mpirun -np <procs> ./out <k> [count]\n");
            std::printf("  <k>     : base for k-nomial (>=2)\n");
            std::printf("  [count] : elems per rank (default 4)\n");
        }
        MPI_Finalize();
        return 0;
    }

    int k = std::atoi(argv[1]);
    int count = (argc >= 3) ? std::atoi(argv[2]) : 4;
    if (k < 2) {
        if (rank == 0) std::fprintf(stderr, "k must be >= 2\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<double> sendbuf((size_t)count * (size_t)size);
    std::vector<double> recvbuf(count, 0.0);
    std::vector<double> refbuf(count, 0.0);

    // Fill: each rank has its own block disps[rank].. of size 'count'
    for (int i = 0; i < count * size; ++i)
        sendbuf[i] = (double)rank + 0.1 * (double)i;

    // Reference reduce_scatter via Allreduce + Scatter
    std::vector<double> tmp((size_t)count * (size_t)size, 0.0);
    MPI_Allreduce(sendbuf.data(), tmp.data(), count * size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scatter(tmp.data(), count, MPI_DOUBLE, refbuf.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Our implementation
    int rc = MPICH_reduce_scatter_k_nomial((const char*)sendbuf.data(),
                                           (char*)recvbuf.data(),
                                           count, MPI_DOUBLE, MPI_SUM,
                                           MPI_COMM_WORLD, k);
    if (rc != MPI_SUCCESS) {
        std::printf("Rank %d: function returned error %d\n", rank, rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Compare
    int errors = 0;
    for (int i = 0; i < count; ++i) {
        double diff = std::fabs(recvbuf[i] - refbuf[i]);
        if (diff > 1e-9) errors++;
    }

    if (errors == 0) {
        std::printf("Rank %d: PASS (k=%d, count=%d)\n", rank, k, count);
    } else {
        std::printf("Rank %d: FAIL (k=%d, count=%d) mismatches=%d\n", rank, k, count, errors);
        std::printf("  expected:");
        for (int i = 0; i < count; ++i) std::printf(" %.2f", refbuf[i]);
        std::printf("\n  got     :");
        for (int i = 0; i < count; ++i) std::printf(" %.2f", recvbuf[i]);
        std::printf("\n");
    }

    MPI_Finalize();
    return 0;
}