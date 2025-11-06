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
        sizes[i] = row_size / comm_sz;
        if (i < row_size % comm_sz)
            sizes[i]++;
        sizes[i] *= column_size;
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

void FillMatrix(int row_size, int column_size, int *matrix, int my_rank, int *sizes, int *displs)
{
    if (my_rank == 0)
    {
        int *temp = calloc(row_size * column_size, sizeof(int));
        for (int i = 0; i < row_size * column_size; i++)
        {
            temp[i] = i % 5 + 1;
        }
        MPI_Scatterv(temp, sizes, displs, MPI_INT,
                     matrix, sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
        free(temp);
    }
    else
    {
        MPI_Scatterv(NULL, sizes, displs, MPI_INT,
                     matrix, sizes[my_rank], MPI_INT, 0, MPI_COMM_WORLD);
    }
}

long long GetDistributedVectorSum(int *v, int n, int my_rank, int *sizes, int *displs)
{
    long long sum = -1;
    int *temp = calloc(n, sizeof(int));
    MPI_Gatherv(v, sizes[my_rank], MPI_INT, temp, sizes, displs,
                MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += temp[i];
        }
    }
    free(temp);
    return sum;
}

void PrintDistributedMatrix(int row_size, int column_size, int *local_matrix,
                            int my_rank, int *sizes, int *displs)
{
    if (my_rank == 0)
    {
        int *temp = calloc(row_size * column_size, sizeof(int));
        MPI_Gatherv(local_matrix, sizes[my_rank], MPI_INT, temp, sizes,
                    displs, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < row_size; i++)
        {
            for (int j = 0; j < column_size; j++)
                printf("%d ", temp[i * column_size + j]);
            printf("\n");
        }
        free(temp);
        printf("------------\n");
    }
    else
    {
        MPI_Gatherv(local_matrix, sizes[my_rank], MPI_INT, NULL, sizes,
                    displs, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void MultiplyByRow(int *matrix, int *vector, int *result, int local_row, int column_size)
{
    for (int i = 0; i < local_row; i++)
    {
        result[i] = 0;
        for (int j = 0; j < column_size; j++)
        {
            result[i] += matrix[i * column_size + j] * vector[j];
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

    int *sizes_mat = calloc(comm_sz, sizeof(int));
    int *displacements_mat = calloc(comm_sz, sizeof(int));
    int *sizes_vec = calloc(comm_sz, sizeof(int));
    int *displacements_vec = calloc(comm_sz, sizeof(int));

    BuildSize(row_size, column_size, comm_sz, sizes_mat);
    BuildDisplacements(comm_sz, displacements_mat, sizes_mat);
    BuildSize(row_size, 1, comm_sz, sizes_vec);
    BuildDisplacements(comm_sz, displacements_vec, sizes_vec);

    int *matrix = calloc(sizes_mat[my_rank], sizeof(int));
    int *vector = calloc(column_size, sizeof(int));

    FillMatrix(row_size, column_size, matrix, my_rank, sizes_mat, displacements_mat);
    FillVector(vector, column_size, my_rank);

    int *result = calloc(row_size, sizeof(int));

    struct MyClock clock;
    clock_start(&clock);

    int local_row = sizes_mat[my_rank] / column_size;
    MultiplyByRow(matrix, vector, result, local_row, column_size);

    clock_stop(&clock);

    // Time measurement
    double elapsed = clock_elapsed(&clock);
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long totalSum = GetDistributedVectorSum(result, row_size, my_rank,
                                                    sizes_vec, displacements_vec);
    if (my_rank == 0)
    {
        printf("|%lld,%d,%d,%f|\n", totalSum, row_size, column_size, max_elapsed);
    }

    MPI_Finalize();
    return 0;
}