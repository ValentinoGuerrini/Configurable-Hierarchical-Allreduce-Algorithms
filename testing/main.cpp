// test_allreduce.cpp
// C++ MPI test harness for custom Allreduce implementations

#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>

    int MPICH_Allreduce_k_reduce_scatter_allgather(const char* sendbuf, char* recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        int k, int single_phase_recv);

    int MPICH_Allreduce_recursive_multiplying(const char* sendbuf, char* recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
        int k);

    int MPICH_Allreduce_reduce_scatter_allgather(const char* sendbuf, char* recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    int MPICH_Allreduce_ring(const char* sendbuf, char* recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    int MPICH_Allreduce_recursive_doubling(const char* sendbuf, char* recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    int MPICH_Allreduce_recursive_exchange(const char* sendbuf, char* recvbuf, 
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int k, int single_phase_recv); 


// Compare output buffer to reference buffer within tolerance
bool check_correctness(const std::vector<double>& buf,
                       const std::vector<double>& ref,
                       double eps = 1e-9) {
    for (size_t i = 0; i < buf.size(); ++i) {
        if (std::fabs(buf[i] - ref[i]) > eps)
            return false;
    }
    return true;
}

// Run variants with two 'k' parameters (k + single_phase_recv)
template<typename Func>
void run_k2(const std::string& name, int k, int count, Func func,
            MPI_Comm comm, std::ofstream& csv, int rank, int nprocs) {
    std::vector<double> sendbuf(count), refbuf(count), recvbuf(count);
    // Unique initialization
    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<double>(count) + i;

    // Reference result via MPI_Allreduce
    MPI_Allreduce(sendbuf.data(), refbuf.data(), count,
                  MPI_DOUBLE, MPI_SUM, comm);

    for (int rep = 0; rep < 50; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0.0);
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<const char*>(sendbuf.data()),
                       reinterpret_cast<char*>(recvbuf.data()),
                       count, MPI_DOUBLE, MPI_SUM, comm,
                       k, /*single_phase_recv=*/0);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS)
                       && check_correctness(recvbuf, refbuf);
        if (rank == 0) {
            csv << name << "," << k << ","<<nprocs<<"," << count << ","
                << (t1 - t0) << "," << (correct?1:0) << "\n";
            csv.flush();
        }
    }
}

// Run variants with a single 'k' parameter
template<typename Func>
void run_k1(const std::string& name, int k, int count, Func func,
            MPI_Comm comm, std::ofstream& csv, int rank, int nprocs) {
    std::vector<double> sendbuf(count), refbuf(count), recvbuf(count);
    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<double>(count) + i;
    MPI_Allreduce(sendbuf.data(), refbuf.data(), count,
                  MPI_DOUBLE, MPI_SUM, comm);

    for (int rep = 0; rep < 50; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0.0);
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<const char*>(sendbuf.data()),
                       reinterpret_cast<char*>(recvbuf.data()),
                       count, MPI_DOUBLE, MPI_SUM, comm,
                       k);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS)
                       && check_correctness(recvbuf, refbuf);
        if (rank == 0) {
            csv << name << "," << k << "," <<nprocs<<","<< count << ","
                << (t1 - t0) << "," << (correct?1:0) << "\n";
            csv.flush();
        }
    }
}

// Run variants without any 'k' parameter
template<typename Func>
void run_no_k(const std::string& name, int count, Func func,
              MPI_Comm comm, std::ofstream& csv, int rank, int nprocs) {
    std::vector<double> sendbuf(count), refbuf(count), recvbuf(count);
    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<double>(count) + i;
    MPI_Allreduce(sendbuf.data(), refbuf.data(), count,
                  MPI_DOUBLE, MPI_SUM, comm);

    for (int rep = 0; rep < 50; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0.0);
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<const char*>(sendbuf.data()),
                       reinterpret_cast<char*>(recvbuf.data()),
                       count, MPI_DOUBLE, MPI_SUM, comm);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS)
                       && check_correctness(recvbuf, refbuf);
        if (rank == 0) {
            csv << name << ",0," <<nprocs<<","<< count << ","
                << (t1 - t0) << "," << (correct?1:0) << "\n";
            csv.flush();
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //
    // Parse arguments:  program <n_iter> [--overwrite]
    //
    if (argc < 2 || argc > 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <n_iter> [--overwrite]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int  n_iter    = std::atoi(argv[1]);
    bool overwrite = (argc == 3 && std::strcmp(argv[2], "--overwrite") == 0);

    //
    // Only rank‐0 manages the CSV file.
    //
    std::ofstream csv;
    if (rank == 0) {
        // check whether results.csv already exists
        bool exists = std::ifstream("results.csv").good();

        if (overwrite || !exists) {
            // truncate (or create) + write header
            csv.open("results.csv", std::ios::out | std::ios::trunc);
            csv << "algorithm_name,k,nprocs,send_count,time,is_correct\n";
        }
        else {
            // append, no header
            csv.open("results.csv", std::ios::out | std::ios::app);
        }
    }

    const int base = 8;
    for (int i = 0; i < n_iter; ++i) {
        int count = base << i;

        // Algorithms with k + single_phase_recv
        for (int k = 2; k < nprocs; ++k) {
            run_k2("reduce_scatter_allgather_k", k, count,
                   MPICH_Allreduce_k_reduce_scatter_allgather,
                   MPI_COMM_WORLD, csv, rank, nprocs);
            run_k2("recursive_exchange", k, count,
                   MPICH_Allreduce_recursive_exchange,
                   MPI_COMM_WORLD, csv, rank, nprocs);

            run_k1("recursive_multiplying", k, count,
                   MPICH_Allreduce_recursive_multiplying,
                   MPI_COMM_WORLD, csv, rank, nprocs);
        }

        // Algorithms without k
        run_no_k("reduce_scatter_allgather", count,
                 MPICH_Allreduce_reduce_scatter_allgather,
                 MPI_COMM_WORLD, csv, rank, nprocs);

        run_no_k("ring", count,
                 MPICH_Allreduce_ring,
                 MPI_COMM_WORLD, csv, rank, nprocs);

        run_no_k("recursive_doubling", count,
                 MPICH_Allreduce_recursive_doubling,
                 MPI_COMM_WORLD, csv, rank, nprocs);
    }

    if (rank == 0) csv.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}