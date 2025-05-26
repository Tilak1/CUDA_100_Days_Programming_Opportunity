#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> Timer;

#define RED   "\033[1;31m"
#define GREEN "\033[1;32m"
#define CYAN  "\033[1;36m"
#define YELLOW "\033[1;33m"
#define RESET "\033[0m"

// Start the timer
inline void startTime(Timer* timer) {
    *timer = Clock::now();
}

// Stop the timer and print elapsed time in ms
inline void stopTime(Timer* timer) {
    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - *timer).count();
    printf("Elapsed time: %ld ms\n", duration);
}

// Optional color-coded timing print
inline void printElapsedTime(Timer timer, const char* label, const char* color = CYAN) {
    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - timer).count();
    printf("%s%s: %ld ms%s\n", color, label, duration, RESET);
}

#endif
