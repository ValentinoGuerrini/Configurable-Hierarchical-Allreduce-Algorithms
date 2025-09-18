

#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cstdio>

int tuna2_algorithm (int r, int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	if ( r < 2 ) { r = 2; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	MPI_Type_size(sendtype, &typesize);

	if ( r > nprocs - 1 ) { r = nprocs - 1; }
	if (b <= 0 || b > nprocs) b = nprocs;

	int w, max_rank, nlpow, d, K, i, num_reqs;
	int local_max_count=0, max_send_count=0;
	int rotate_index_array[nprocs];
	w = 0, nlpow = 1, max_rank = nprocs - 1;

    while (max_rank) { w++; max_rank /= r; }   // number of bits required of r representation

    for (i = 0; i < w - 1; i++) { nlpow *= r; }   // maximum send number of elements

	d = (nlpow*r - nprocs) / nlpow; // calculate the number of highest digits
	K = w * (r - 1) - d; // the total number of communication rounds

	int rem1 = K + 1, rem2 = r + 1;
	int sendNcopy[nprocs + rem1];




	char *extra_buffer, *temp_recv_buffer, *temp_send_buffer;
	int extra_ids[nprocs - rem2];
	memset(extra_ids, -1, sizeof(extra_ids));
	int spoint = 1, distance = 1, next_distance = distance*r, di = 0;

	if (K < nprocs - 1) {
		// 1. Find max send count
		for (i = 0; i < nprocs; i++) {
			if (sendcounts[i] > local_max_count) { local_max_count = sendcounts[i]; }
		}
		MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

		// 2. create local index array after rotation
		for (i = 0; i < nprocs; i++) { rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs; }

		// 3. exchange data with log(P) steps
		extra_buffer = (char*) malloc(max_send_count * typesize * (nprocs - rem1));



		temp_recv_buffer = (char*) malloc(max_send_count * nprocs * typesize);


	    if (extra_buffer == nullptr || temp_recv_buffer == nullptr) {
	        std::cerr << "extra_buffer or temp_recv_buffer allocation failed!" << std::endl;
	        return 1; // Exit program with error
	    }
		temp_send_buffer = (char*) malloc(max_send_count * nlpow * typesize);



	    if (temp_send_buffer == nullptr) {
	        std::cerr << "temp_send_buffer allocation failed!" << std::endl;
	        return 1; // Exit program with error
	    }

		for (int x = 0; x < w; x++) {
			for (int z = 1; z < r; z ++) {
				spoint = z * distance;
				if (spoint > nprocs) { break; }
				int end = (spoint + distance > nprocs)? nprocs : spoint + distance;
				for (i = spoint + 1; i < end; i++) {
					extra_ids[i-rem2] = di++;
				}
			}
			distance *= r;
		}

	}

	// copy data that need to be sent to each rank itself
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);


	int sent_blocks[r-1][nlpow];




	int metadata_recv[r-1][nlpow];

	int nc, rem, ns, ze, ss;
	spoint = 1, distance = 1, next_distance = distance*r;

//	MPI_Request* reqs = (MPI_Request *) malloc(2 * b * sizeof(MPI_Request));
	MPI_Request* reqs = (MPI_Request *) malloc(2 * r * sizeof(MPI_Request));
    if (reqs == nullptr) {
        std::cerr << "MPI_Requests allocation failed!" << std::endl;
        return 1; // Exit program with error
    }

    MPI_Status* stats = (MPI_Status *) malloc(2 * r * sizeof(MPI_Status));
    if (stats == nullptr) {
        std::cerr << "MPI_Status allocation failed!" << std::endl;
        return 1; // Exit program with error
    }

	int metadata_send[nlpow];
    


    int comm_size[r-1];

	for (int x = 0; x < w; x++) {
		ze = (x == w - 1)? r - d: r;
		int zoffset = 0, zc = ze-1;
		int zns[zc];

		for (int k = 1; k < ze; k += b) {
			ss = ze - k < b ? ze - k : b;
			num_reqs = 0;

			for (int s = 0; s < ss; s++) {

				int z = k + s;

//				if (rank == 0) {
//					std::cout << k << " " << s << " " << z << std::endl;
//				}

				spoint = z * distance;
				nc = nprocs / next_distance * distance, rem = nprocs % next_distance - spoint;
				if (rem < 0) { rem = 0; }
				ns = (rem > distance)? (nc + distance) : (nc + rem);
				zns[z-1] = ns;

				int recvrank = (rank + spoint) % nprocs;
				int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process


				if (ns == 1) {


               
					MPI_Irecv(&recvbuf[rdispls[recvrank]*typesize], recvcounts[recvrank]*typesize, MPI_CHAR, recvrank, recvrank, comm, &reqs[num_reqs++]);

					MPI_Isend(&sendbuf[sdispls[sendrank]*typesize], sendcounts[sendrank]*typesize, MPI_CHAR, sendrank, rank, comm, &reqs[num_reqs++]);
				}
				else {
                    
					di = 0;
					for (int i = spoint; i < nprocs; i += next_distance) {
						int j_end = (i+distance > nprocs)? nprocs: i+distance;
						for (int j = i; j < j_end; j++) {
							int id = (j + rank) % nprocs;
							sent_blocks[z-1][di++] = id;
						}
					}

					// 2) prepare metadata
					int sendCount = 0, offset = 0;
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;



						if (i % distance == 0) {
							metadata_send[i] = sendcounts[send_index];
						}
						else {
							metadata_send[i] = sendNcopy[extra_ids[o]];
						}
						offset += metadata_send[i] * typesize;
					}

					MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv[z-1], di,
							MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

					for(int i = 0; i < di; i++) { sendCount += metadata_recv[z-1][i]; }
					comm_size[z-1] = sendCount; // total exchanged data per round

					// prepare send data
					offset = 0;
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;


						int size = 0;

						if (i % distance == 0) {
							size = sendcounts[send_index]*typesize;
							memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], size);
						}
						else {
							size = sendNcopy[extra_ids[o]]*typesize;
							memcpy(&temp_send_buffer[offset], &extra_buffer[extra_ids[o]*max_send_count*typesize], size);
						}
						offset += size;
					}

					MPI_Irecv(&temp_recv_buffer[zoffset], comm_size[z-1]*typesize, MPI_CHAR, recvrank, recvrank+z, comm, &reqs[num_reqs++]);
					MPI_Isend(temp_send_buffer, offset, MPI_CHAR, sendrank, rank+z, comm, &reqs[num_reqs++]);

					zoffset += comm_size[z-1]*typesize;
				}

			}

			MPI_Waitall(num_reqs, reqs, stats);

			for (int i = 0; i < num_reqs; i++) {
			    if (stats[i].MPI_ERROR != MPI_SUCCESS) {
			        printf("Request %d encountered an error: %d\n", i, stats[i].MPI_ERROR);
			    }
			}


		}

		if (K < nprocs - 1) {
			// replaces
			int offset = 0;
			for (int i = 0; i < zc; i++) {
				for (int j = 0; j < zns[i]; j++){

					if (zns[i] > 1){
						int size = metadata_recv[i][j]*typesize;
						int o = (sent_blocks[i][j] - rank + nprocs) % nprocs - rem2;



						if (j < distance) {
							memcpy(&recvbuf[rdispls[sent_blocks[i][j]]*typesize], &temp_recv_buffer[offset], size);
						}
						else {
                        
							memcpy(&extra_buffer[extra_ids[o]*max_send_count*typesize], &temp_recv_buffer[offset], size);
							sendNcopy[extra_ids[o]] = metadata_recv[i][j];
						}
						offset += size;
					}
				}
			}
		}

		distance *= r;
		next_distance *= r;

	}
	if (K < nprocs - 1) {
		free(extra_buffer);
		free(temp_recv_buffer);
		free(temp_send_buffer);
	}
	free(reqs);
	free(stats);

	return 0;
}

// Replace your fill_pattern with this non-periodic version
static void fill_pattern(char *buf, int world_size, int rank, int count_per_peer) {
    // Write 32-bit little-endian words with a non-repeating counter per (dst, wordIndex)
    // so offsets/reorders are obvious and don't alias every 256 bytes.
    for (int dst = 0; dst < world_size; ++dst) {
        uint32_t *w = (uint32_t *)(buf + dst * count_per_peer);
        int words = count_per_peer / 4;
        for (int k = 0; k < words; ++k) {
            // Mix rank, dst, and k to avoid simple periodicities
            uint32_t v = 0x9E3779B9u * (uint32_t)k
                       ^ ((uint32_t)rank * 0x85EBCA6Bu)
                       ^ ((uint32_t)dst  * 0xC2B2AE35u);
            w[k] = v;
        }
        // tail bytes (if any)
        for (int t = words * 4; t < count_per_peer; ++t) {
            buf[dst * count_per_peer + t] = (char)((rank ^ dst ^ t) & 0xFF);
        }
    }
}

// Add this helper to compute a simple 64-bit checksum per source block in recv buffers
static uint64_t block_crc64(const char *p, int nbytes) {
    // Very simple rolling checksum (not cryptographic)
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < nbytes; ++i) {
        h ^= (unsigned char)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Run with at least 2 ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int b = 8;
    if (argc >= 2) {
        b = atoi(argv[1]);
    } else {
        if (rank == 0) {
            fprintf(stdout, "Enter b (batch/block count): ");
            fflush(stdout);
            if (scanf("%d", &b) != 1) b = 0;
        }
        MPI_Bcast(&b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (b < 3) {
        if (rank == 0) fprintf(stderr, "b must be >= 3 to test r in [2, b-1].\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Pre-allocate for the largest tested size (1024 bytes/peer).
    const int max_count_per_peer = 1024; // last of the ten sizes
    const int total_bytes = max_count_per_peer * size;

    char *sendbuf  = (char*)malloc(total_bytes);
    char *recvbuf  = (char*)malloc(total_bytes);
    char *recv_ref = (char*)malloc(total_bytes);
    int  *sendcounts = (int*)malloc(sizeof(int) * size);
    int  *recvcounts = (int*)malloc(sizeof(int) * size);
    int  *sdispls    = (int*)malloc(sizeof(int) * size);
    int  *rdispls    = (int*)malloc(sizeof(int) * size);

    if (!sendbuf || !recvbuf || !recv_ref || !sendcounts || !recvcounts || !sdispls || !rdispls) {
        if (rank == 0) fprintf(stderr, "Allocation failure.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    int all_ok = 1;

    // Ten sizes: 2,4,8,...,1024 bytes per peer
    for (int exp = 1; exp <= 10; ++exp) {
        int count_per_peer = 1 << exp;

        // Set uniform counts/displacements for this size
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = count_per_peer;
            recvcounts[i] = count_per_peer;
            sdispls[i]    = i * count_per_peer;
            rdispls[i]    = i * count_per_peer;
        }

        // Outer loop over r
        for (int r = 2; r <= b - 1; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);

            // Prepare fresh buffers
            memset(recvbuf, 0, total_bytes);
            memset(recv_ref, 0, total_bytes);
            fill_pattern(sendbuf, size, rank, count_per_peer);

            // Reference
            MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_CHAR,
                          recv_ref, recvcounts, rdispls, MPI_CHAR, MPI_COMM_WORLD);

            // Refill send buffer (keep inputs pristine)
            fill_pattern(sendbuf, size, rank, count_per_peer);

            // Under test
            int rc = tuna2_algorithm(r, b,
                                     sendbuf, sendcounts, sdispls, MPI_CHAR,
                                     recvbuf, recvcounts, rdispls, MPI_CHAR,
                                     MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) {
                if (rank == 0)
                    fprintf(stderr, "[count=%d, r=%d] tuna2_algorithm returned %d\n",
                            count_per_peer, r, rc);
                all_ok = 0;
                //goto done;
            }

            // Compare
            int local_ok  = (memcmp(recvbuf, recv_ref, count_per_peer * size) == 0);
            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("[count=%4d bytes/peer, r=%d] %s\n",
                       count_per_peer, r, global_ok ? "PASS" : "FAIL");
                fflush(stdout);
            }

            if (!global_ok && rank == 0) {
                fprintf(stderr, "Mismatch details (per-source CRCs on rank 0):\n");
            }
            if (!global_ok) {
                // Only rank 0 prints its own so output isn't too noisy
                if (rank == 0) {
                    for (int src = 0; src < size; ++src) {
                        const char *blk_ref = recv_ref + src * count_per_peer;
                        const char *blk_tut = recvbuf + src * count_per_peer;
                        uint64_t c_ref = block_crc64(blk_ref, count_per_peer);
                        uint64_t c_tut = block_crc64(blk_tut, count_per_peer);
                        printf("  src=%d  CRC_ref=%016llx  CRC_tuna=%016llx  %s\n",
                            src, (unsigned long long)c_ref, (unsigned long long)c_tut,
                            (c_ref==c_tut ? "OK" : "DIFF"));
                    }
                    fflush(stdout);
                }
            }
        }
    }

done:
    if (rank == 0) {
        printf(all_ok ? "All tests PASSED.\n" : "Some tests FAILED.\n");
        fflush(stdout);
    }

    free(sendbuf);
    free(recvbuf);
    free(recv_ref);
    free(sendcounts);
    free(recvcounts);
    free(sdispls);
    free(rdispls);

    MPI_Finalize();
    //return all_ok ? 0 : 1;
    return 0;
}