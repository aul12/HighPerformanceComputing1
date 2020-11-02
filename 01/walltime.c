/**
 * @file walltime.c
 * @author paul
 * @date 02.11.20
 * Description here TODO
 */
#include "walltime.h"

#include <sys/times.h>
#include <unistd.h>

double walltime() {
    static clock_t ticks_per_second = 0;
    if (!ticks_per_second) {
        ticks_per_second = sysconf(_SC_CLK_TCK);
    }
    struct tms timebuf;
    /* times returns the number of real time ticks passed since start */
    return (double) times(&timebuf) / ticks_per_second;
}
