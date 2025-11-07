#include "clock.h"

#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void BuildSize(int row_size, int column_size, int comm_sz, int *sizes)
{
    for (int i = 0; i < comm_sz; i++)
    {
        sizes[i] = column_size / comm_sz;
        if (i < column_size % comm_sz)
            sizes[i]++;
    }
}

void BuildDisplacements(int comm_sz, int *displs, int *sizes)
{
    displs[0] = 0;
    for (int i = 1; i < comm_sz; i++)
    {
        displs[i] = displs[i - 1] + sizes[i - 1];
    }
}

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

void FillMatrix(int row_size, int column_size, int *local_matrix, int my_rank, int *sizes, int *displs, MPI_Datatype column_type)
{
    int *temp = NULL;
    if (my_rank == 0)
    {
        temp = calloc(row_size * column_size, sizeof(int));
        for (int i = 0; i < row_size * column_size; i++)
        {
            temp[i] = i % 5 + 1;
        }
    }
    MPI_Scatterv(temp, sizes, displs, column_type,
                 local_matrix, row_size * sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        free(temp);
    }
}

void MultiplyByColumn(int *matrix, int *vector, int *result, int *sizes_mat, int* displacements_mat, int my_rank, int row_size, int column_size)
{
    for (int i = 0; i < row_size * sizes_mat[my_rank]; i++)
    {
        int curI = i % row_size;
        int curJ = i / row_size + displacements_mat[my_rank];
        result[curI] += matrix[i] * vector[curJ];
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

    MPI_Datatype column_type, col_resized;
    MPI_Type_vector(row_size, 1, column_size, MPI_INT, &column_type);
    MPI_Type_create_resized(column_type, 0, sizeof(int), &col_resized);
    MPI_Type_commit(&col_resized);
    MPI_Type_free(&column_type);

    int *sizes_mat = calloc(comm_sz, sizeof(int));
    int *displacements_mat = calloc(comm_sz, sizeof(int));
    int *sizes_vec = calloc(comm_sz, sizeof(int));
    int *displacements_vec = calloc(comm_sz, sizeof(int));

    BuildSize(row_size, column_size, comm_sz, sizes_mat);
    BuildDisplacements(comm_sz, displacements_mat, sizes_mat);
    BuildSize(1, column_size, comm_sz, sizes_vec);
    BuildDisplacements(comm_sz, displacements_vec, sizes_vec);

    int *local_matrix = calloc(row_size * sizes_mat[my_rank], sizeof(int));
    int *vector = calloc(column_size, sizeof(int));

    FillMatrix(row_size, column_size, local_matrix, my_rank, sizes_mat, displacements_mat, col_resized);

    FillVector(vector, column_size, my_rank);

    int *result = calloc(row_size, sizeof(int));
    int *total = calloc(row_size, sizeof(int));

    struct MyClock clock;
    clock_start(&clock);

    MultiplyByColumn(local_matrix, vector, result, sizes_mat, displacements_mat, my_rank, row_size, column_size);
    MPI_Reduce(result, total, row_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    clock_stop(&clock);

    // Time measurement
    double elapsed = clock_elapsed(&clock);
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        long long totalSum = 0;
        for (int i = 0; i < row_size; ++i) {
            totalSum += total[i];
        }
        printf("|%lld,%d,%d,%f|\n", totalSum, row_size, column_size, max_elapsed);
    }

    MPI_Type_free(&col_resized);
    MPI_Finalize();
    return 0;
}