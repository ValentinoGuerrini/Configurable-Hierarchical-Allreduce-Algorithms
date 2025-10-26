
#include <mpi.h>
#include <iostream>
#include <cstring>   // for memcpy, memmove, etc.
#include <vector>
#include <fstream>
#include <cstdlib>

//#define DEBUG_MODE

int intra_scatter_radix_batch(char* sendbuf,
                              int   recvcount,
                              MPI_Datatype datatype,
                              char* recvbuf,
                              MPI_Comm comm,
                              int k,
                              int b)
{
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);

    if (k < 2 || b <= 0) return MPI_SUCCESS;

    const int node_id    = rank / b;        // which node
    const int local_rank = rank %  b;       // slot within node
    const int root_local = node_id % b;     // node root (matches your gather)
    const int shift      = (local_rank - root_local + b) % b; // normalized index

    // #phases = ceil_log_k(b) without FP
    int nphases = 0, tmpb = b - 1;
    while (tmpb > 0) { ++nphases; tmpb /= k; }

    // tmp holds normalized slots [0..b-1] locally
    const size_t bytes_per_block = (size_t)recvcount * (size_t)typeSize;
    char* tmp = (char*)std::malloc((size_t)b * bytes_per_block);
    if (!tmp) return MPI_ERR_NO_MEM;

    // Only the node-root seeds the normalized layout
    if (local_rank == root_local) {
        for (int real_slot = 0; real_slot < b; ++real_slot) {
            const int norm_i = (real_slot - root_local + b) % b;
            std::memcpy(tmp + (size_t)norm_i * bytes_per_block,
                        sendbuf + (size_t)real_slot * bytes_per_block,
                        bytes_per_block);
        }
    }

    MPI_Request* reqs = (MPI_Request*)std::malloc((k - 1) * sizeof(*reqs));
    if (!reqs) { std::free(tmp); return MPI_ERR_NO_MEM; }

    // Scatter: go from largest delta down to 1
    // initial delta = k^(nphases-1)
    int delta = 1;
    for (int i = 1; i < nphases; ++i) delta *= k;

    for (int phase = nphases - 1; phase >= 0; --phase) {
        const int groupSize = delta * k;                          // size of this k-ary group
        const int gstart    = (shift / groupSize) * groupSize;    // leader of my group
        const int blockEnd  = std::min(gstart + groupSize, b);    // cap at b
        const int offset    = shift - gstart;                      // position inside group

        if (offset == 0) {
            // I am the leader of this group → send to child-group leaders
            int ns = 0;
            for (int j = 1; j < k; ++j) {
                const int child = gstart + j * delta;             // leader of child subgroup
                if (child >= blockEnd) break;

                const int subtree = std::min(delta, blockEnd - child);

                const int dst_local = (child + root_local) % b;   // map back to real local slot
                const int dst_glob  = node_id * b + dst_local;

                MPI_Isend(tmp + (size_t)child * bytes_per_block,
                          subtree * recvcount, datatype,
                          dst_glob, /*tag=*/phase, comm, &reqs[ns++]);
            }
            if (ns) MPI_Waitall(ns, reqs, MPI_STATUSES_IGNORE);
        }
        else if ((offset % delta) == 0 && offset < groupSize) {
            // I am a child-subgroup leader → receive my subtree block from my parent (gstart)
            const int parent    = gstart;                          // leader of parent group
            const int src_local = (parent + root_local) % b;
            const int src_glob  = node_id * b + src_local;

            const int subtree = std::min(delta, blockEnd - shift);

            MPI_Request rreq;
            MPI_Irecv(tmp + (size_t)shift * bytes_per_block,
                      subtree * recvcount, datatype,
                      src_glob, /*tag=*/phase, comm, &rreq);
            MPI_Wait(&rreq, MPI_STATUS_IGNORE);
            // After this phase, I'll be a leader for my smaller subgroups in later (smaller delta) phases.
        }

        // next phase uses smaller delta
        delta = (phase > 0 ? delta / k : delta);
    }

    // Each rank extracts its own block from normalized index = shift
    std::memcpy(recvbuf,
                tmp + (size_t)shift * bytes_per_block,
                bytes_per_block);

    std::free(reqs);
    std::free(tmp);
    return MPI_SUCCESS;
}


#ifdef DEBUG_MODE

static void print_blocks_int(const int* buf, int b, int recvcount,
                             int rank, int node_id, int local_rank,
                             const char* title)
{
    std::fflush(stdout);
    std::printf("%s | rank %d (node %d, local %d)\n", title, rank, node_id, local_rank);
    for (int s = 0; s < b; ++s) {
        std::printf("  B%d: [", s);
        for (int j = 0; j < recvcount; ++j) {
            if (j) std::printf(", ");
            std::printf("%d", buf[s * recvcount + j]);
        }
        std::printf("]\n");
    }
    std::fflush(stdout);
}

static void print_vec_int(const int* buf, int n,
                          int rank, int node_id, int local_rank,
                          const char* title)
{
    std::fflush(stdout);
    std::printf("%s | rank %d (node %d, local %d): [", title, rank, node_id, local_rank);
    for (int i = 0; i < n; ++i) {
        if (i) std::printf(", ");
        std::printf("%d", buf[i]);
    }
    std::printf("]\n");
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Args: k b recvcount
    int k = 7;
    int b = 9;
    int recvcount = 7;
    if (argc >= 2) k = std::atoi(argv[1]);
    if (argc >= 3) b = std::atoi(argv[2]);
    if (argc >= 4) recvcount = std::atoi(argv[3]);
    if (k < 2) k = 2;
    if (b < 1) b = 1;
    if (recvcount < 1) recvcount = 1;

    if (nranks % b != 0) {
        if (rank == 0)
            std::fprintf(stderr, "Error: nranks (%d) must be a multiple of b (%d)\n", nranks, b);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int node_id    = rank / b;
    const int local_rank = rank %  b;
    const int root_local = node_id % b;
    const int root_rank  = node_id * b + root_local;

    // Buffers
    std::vector<int> recvbuf(recvcount, -777);
    std::vector<int> sendbuf;  // allocated only on node root

    if (rank == root_rank) {
        sendbuf.resize(b * recvcount);
        // Pattern: for node n, block s, element j:
        // value = 100000*n + 1000*s + j
        for (int s = 0; s < b; ++s) {
            for (int j = 0; j < recvcount; ++j) {
                sendbuf[s * recvcount + j] = 100000 * node_id + 1000 * s + j;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Pretty, ordered "BEFORE": only node-roots print, in global rank order
    if (rank == 0) std::printf("=== BEFORE (node roots) ===\n");
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < nranks; ++r) {
        if (rank == r && rank == root_rank) {
            print_blocks_int(sendbuf.data(), b, recvcount, rank, node_id, local_rank, "sendbuf");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Call the scatter (non-roots may pass nullptr; function only reads sendbuf at root)
    char* sendptr = (rank == root_rank) ? reinterpret_cast<char*>(sendbuf.data()) : nullptr;
    int err = intra_scatter_radix_batch(sendptr, recvcount, MPI_INT,
                                        reinterpret_cast<char*>(recvbuf.data()),
                                        MPI_COMM_WORLD, k, b);
    if (err != MPI_SUCCESS) {
        std::fprintf(stderr, "rank %d: intra_scatter_radix_batch returned error %d\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Ordered "AFTER": everyone prints their recvbuf in global rank order
    if (rank == 0) std::printf("\n=== AFTER (each rank's recvbuf) ===\n");
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < nranks; ++r) {
        if (rank == r) {
            print_vec_int(recvbuf.data(), recvcount, rank, node_id, local_rank, "recvbuf");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Correctness check
    bool ok = true;
    for (int j = 0; j < recvcount; ++j) {
        int expect = 100000 * node_id + 1000 * local_rank + j;
        if (recvbuf[j] != expect) {
            ok = false;
            break;
        }
    }

    int local = ok ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global == 1) std::printf("\nRESULT: PASS ✅\n");
        else             std::printf("\nRESULT: FAIL ❌\n");
    }

    MPI_Finalize();
    return 0;
}
#endif