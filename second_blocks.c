#include "clock.h"

#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void FillVector(int *vector, int size, int my_rank)
{
    if (my_rank == 0)
    {
        for (int i = 0; i < size; ++i)
        {
            vector[i] = i % 5 + 1;
        }
    }
    MPI_Bcast(vector, size, MPI_INT, 0, MPI_COMM_WORLD);
}

void MultiplyByBlock(int *matrix, int *vector, int *result, int block_rows, int block_cols, int my_rank, int coordI, int coordJ)
{
    for (int i = 0; i < block_rows; ++i)
    {
        for (int j = 0; j < block_cols; ++j)
        {
            int curI = i + block_rows * coordI;
            int curJ = j + block_cols * coordJ;
            result[curI] += matrix[i * block_cols + j] * vector[curJ];
        }
    }
}

int main(int argc, char **argv)
{
    int row_size = 1000, column_size = 1000;
    if (argc > 2)
    {
        row_size = atoll(argv[1]);
        column_size = atoll(argv[2]);
    }

    int comm_sz;
    int my_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ((row_size * column_size) % comm_sz != 0)
    {
        if (my_rank == 0)
        {
            printf("Incorrect processes number (rows * columns should be divisible by processes)");
        }
        MPI_Finalize();
        return 0;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(comm_sz, 2, dims);

    int p_row = dims[0];
    int p_col = dims[1];
    int block_rows = row_size / p_row;
    int block_cols = column_size / p_col;

    int periods[2] = {0, 0};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, my_rank, 2, coords);

    int *matrix = NULL;
    int *vector = calloc(column_size, sizeof(int));
    int *local_matrix = calloc(block_rows * block_cols, sizeof(int));

    if (my_rank == 0)
    {
        matrix = calloc(row_size * column_size, sizeof(int));
        for (int i = 0; i < row_size * column_size; ++i)
        {
            matrix[i] = i % 5 + 1;
        }
    }
    FillVector(vector, column_size, my_rank);

    int *sendcounts = NULL;
    int *displs = NULL;

    if (my_rank == 0)
    {
        sendcounts = malloc(comm_sz * sizeof(int));
        displs = malloc(comm_sz * sizeof(int));
        for (int rank = 0; rank < comm_sz; rank++)
        {
            sendcounts[rank] = block_rows * block_cols;
            displs[rank] = rank * block_rows * block_cols;
        }
        MPI_Scatterv(matrix, sendcounts, displs, MPI_INT, local_matrix, block_rows * block_cols, MPI_INT, 0, MPI_COMM_WORLD);
        free(sendcounts);
        free(displs);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, local_matrix, block_rows * block_cols, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int *result = calloc(row_size, sizeof(int));
    int *total = calloc(row_size, sizeof(int));

    struct MyClock clock;
    clock_start(&clock);

    MultiplyByBlock(local_matrix, vector, result, block_rows, block_cols, my_rank, coords[0], coords[1]);
    MPI_Reduce(result, total, row_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    clock_stop(&clock);

    // Time measurement
    double elapsed = clock_elapsed(&clock);
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        long long totalSum = 0;
        for (int i = 0; i < row_size; ++i)
        {
            totalSum += total[i];
        }
        printf("|%lld,%d,%d,%f|\n", totalSum, row_size, column_size, max_elapsed);
    }

    if (my_rank == 0)
    {
        free(matrix);
    }
    free(local_matrix);
    free(result);
    free(total);
    free(vector);

    MPI_Finalize();
    return 0;
}