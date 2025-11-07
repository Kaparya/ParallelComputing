#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>

void initialize_matrix(double *matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)(rand() % 100);
    }
}

void matrix_multiply_block(double *A, double *B, double *C, int block_sz) {
    for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
            for (int k = 0; k < block_sz; k++) {
                C[i * block_sz + j] += 
                    A[i * block_sz + k] * B[k * block_sz + j];
            }
        }
    }
}

void cannon_algorithm(double *A, double *B, double *C, int N, 
                      int rank, int size) {
    int shift = (int)sqrt(size);
    if (shift * shift != size) {
        if (rank == 0) {
            fprintf(stderr, "error: number of processes must be "
                    "perfect square\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (N % shift != 0) {
        if (rank == 0) {
            fprintf(stderr, "error: matrix size must be divisible by "
                    "grid dimension\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int block_sz = N / shift;
    int block_elements = block_sz * block_sz;

    MPI_Comm cart_comm;
    int shifts[2] = {shift, shift};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, shifts, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    double *local_A = (double*)malloc(block_elements * sizeof(double));
    double *local_B = (double*)malloc(block_elements * sizeof(double));
    double *local_C = (double*)calloc(block_elements, sizeof(double));

    if (!local_A || !local_B || !local_C) {
        fprintf(stderr, "error: memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        for (int proc_row = 0; proc_row < shift; proc_row++) {
            for (int proc_col = 0; proc_col < shift; proc_col++) {
                double *block_A = (double*)malloc(block_elements * 
                                                   sizeof(double));
                double *block_B = (double*)malloc(block_elements * 
                                                   sizeof(double));
                for (int i = 0; i < block_sz; i++) {
                    for (int j = 0; j < block_sz; j++) {
                        int global_row = proc_row * block_sz + i;
                        int global_col = proc_col * block_sz + j;
                        block_A[i * block_sz + j] = 
                            A[global_row * N + global_col];
                    }
                }
                for (int i = 0; i < block_sz; i++) {
                    for (int j = 0; j < block_sz; j++) {
                        int global_row = proc_row * block_sz + i;
                        int global_col = proc_col * block_sz + j;
                        block_B[i * block_sz + j] = 
                            B[global_row * N + global_col];
                    }
                }
                if (proc_row == 0 && proc_col == 0) {
                    memcpy(local_A, block_A, block_elements * sizeof(double));
                    memcpy(local_B, block_B, block_elements * sizeof(double));
                } else {
                    int dest_rank;
                    int dest_coords[2] = {proc_row, proc_col};
                    MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
                    MPI_Send(block_A, block_elements, MPI_DOUBLE, 
                             dest_rank, 0, cart_comm);
                    MPI_Send(block_B, block_elements, MPI_DOUBLE, 
                             dest_rank, 1, cart_comm);
                }

                free(block_A);
                free(block_B);
            }
        }
    } else {
        MPI_Recv(local_A, block_elements, MPI_DOUBLE, 0, 0, 
                 cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, block_elements, MPI_DOUBLE, 0, 1, 
                 cart_comm, MPI_STATUS_IGNORE);
    }

    int left_rank, right_rank;
    MPI_Cart_shift(cart_comm, 1, -row, &right_rank, &left_rank);
    MPI_Sendrecv_replace(local_A, block_elements, MPI_DOUBLE, 
                         left_rank, 0, right_rank, 0, 
                         cart_comm, MPI_STATUS_IGNORE);

    int up_rank, down_rank;
    MPI_Cart_shift(cart_comm, 0, -col, &down_rank, &up_rank);
    MPI_Sendrecv_replace(local_B, block_elements, MPI_DOUBLE, 
                         up_rank, 0, down_rank, 0, 
                         cart_comm, MPI_STATUS_IGNORE);

    for (int step = 0; step < shift; step++) {
        matrix_multiply_block(local_A, local_B, local_C, block_sz);

        MPI_Cart_shift(cart_comm, 1, -1, &right_rank, &left_rank);
        MPI_Sendrecv_replace(local_A, block_elements, MPI_DOUBLE, 
                             left_rank, 0, right_rank, 0, 
                             cart_comm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(cart_comm, 0, -1, &down_rank, &up_rank);
        MPI_Sendrecv_replace(local_B, block_elements, MPI_DOUBLE, 
                             up_rank, 0, down_rank, 0, 
                             cart_comm, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        for (int i = 0; i < block_sz; i++) {
            for (int j = 0; j < block_sz; j++) {
                C[i * N + j] = local_C[i * block_sz + j];
            }
        }
        for (int proc_row = 0; proc_row < shift; proc_row++) {
            for (int proc_col = 0; proc_col < shift; proc_col++) {
                if (proc_row == 0 && proc_col == 0) continue;

                int src_coords[2] = {proc_row, proc_col};
                int src_rank;
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);

                double *recv_block = (double*)malloc(block_elements * 
                                                      sizeof(double));
                MPI_Recv(recv_block, block_elements, MPI_DOUBLE, 
                         src_rank, 2, cart_comm, MPI_STATUS_IGNORE);

                for (int i = 0; i < block_sz; i++) {
                    for (int j = 0; j < block_sz; j++) {
                        int global_row = proc_row * block_sz + i;
                        int global_col = proc_col * block_sz + j;
                        C[global_row * N + global_col] = 
                            recv_block[i * block_sz + j];
                    }
                }

                free(recv_block);
            }
        }
    } else {
        MPI_Send(local_C, block_elements, MPI_DOUBLE, 0, 2, cart_comm);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Comm_free(&cart_comm);
}

int main(int argc, char *argv[]) {
    int comm_sz;
    int my_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int N = 8;
    
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            if (my_rank == 0) {
                fprintf(stderr, "error: invalid matrix size\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    int grid_dim = (int)sqrt(comm_sz);
    if (grid_dim * grid_dim != comm_sz) {
        if (my_rank == 0) {
            fprintf(stderr, "error: number of processes must be "
                    "perfect square\n");
            fprintf(stderr, "current processes: %d\n", comm_sz);
            fprintf(stderr, "valid options: 1, 4, 9, 16, 25, 36, 49, 64...\n");
        }
        MPI_Finalize();
        return 1;
    }

    double *A = NULL, *B = NULL, *C = NULL;

    if (my_rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)calloc(N * N, sizeof(double));

        if (!A || !B || !C) {
            fprintf(stderr, "error: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand(time(NULL));
        initialize_matrix(A, N, rand());
        initialize_matrix(B, N, rand());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    cannon_algorithm(A, B, C, N, my_rank, comm_sz);

    double elapsed = MPI_Wtime() - start_time;

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, 
               MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("|%d,%d,%f|\n", N, comm_sz, max_elapsed);

        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}