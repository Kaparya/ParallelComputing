#pragma once

#include <mpi.h>

struct MyClock {
    double startTime;
    double endTime;
};

static inline void clock_start(struct MyClock* clock) {
    clock->startTime = MPI_Wtime();
}

static inline void clock_stop(struct MyClock* clock) {
    clock->endTime = MPI_Wtime();
}

static inline double clock_elapsed(const struct MyClock* clock) {
    return clock->endTime - clock->startTime;
}
