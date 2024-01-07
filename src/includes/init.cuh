#ifndef _H_INIT
#define _H_INIT

#include "vector.cuh"
#include "codes.cuh"
#include "defines.cuh"

// Get random double in range [min, max]
double random_double(double min, double max);

// Initialize every point randomly
int random_init(vector* points);

// Randomize a single point and then
// place all other points randomly next to it
int area_init(vector* points);

#endif
