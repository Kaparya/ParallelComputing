#include "clock.h"

#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long countIns(long long pointsNumber) {
    long long inCircle = 0;
    for (long long i = 0; i < pointsNumber; ++i) {
        long double x = (long double)rand() / RAND_MAX * 2.0 - 1.0;
        long double y = (long double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1) {
            ++inCircle;
        }
    }
    return inCircle;
}

int main(int argc, char** argv)
{
    long long POINTS_NUMBER = 1000;
    if (argc > 1) {
        POINTS_NUMBER = atoll(argv[1]);
    }

    int comm_sz;
    int my_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    srand(time(NULL) * 10 + my_rank % 10); // Make unique seed for srand

    struct MyClock clock;
    clock_start(&clock);

    long long currentSize = POINTS_NUMBER / comm_sz;
    if (my_rank == comm_sz - 1) {
        currentSize += POINTS_NUMBER % currentSize;
    }
    long long localIns = countIns(currentSize);
    long long totalIns = 0;

    MPI_Reduce(&localIns , &totalIns , 1, MPI_LONG_LONG,
        MPI_SUM, 0, MPI_COMM_WORLD);

    clock_stop(&clock);

    // Time measurement
    double elapsed = clock_elapsed(&clock);
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        long double pi = (long double)totalIns * 4.0 / POINTS_NUMBER;
        printf("|%Lf,%lld,%f|\n", pi, POINTS_NUMBER, max_elapsed);
    }

    MPI_Finalize();
    return 0;
}