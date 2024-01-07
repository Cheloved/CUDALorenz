#ifndef _H_VECTOR
#define _H_VECTOR

// CUDA libs
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// For return codes
#include "codes.cuh"

// Default vector structure
typedef struct 
{
    double x;
    double y;
    double z;
} vector;

// Basic operations
__device__ int vector_copy(vector* dest, vector* source);

__device__ int vector_add(vector* a, vector* b, vector* result);
__device__ int vector_sub(vector* a, vector* b, vector* result);

__device__ int vector_mul(vector* v, double s, vector* result);
__device__ int vector_div(vector* v, double s, vector* result);

__device__ int vector_dot(vector* a, vector* b, double* result);

__device__ int vector_len(vector* v, double* result);

__device__ int vector_normalize(vector* v, vector* result);

// Converts spherical coordinates into cartesian
__device__ int vector_spherical_to_cartesian(double theta, double phi, double r,
                                             vector* result);

// Calculates determinant of a 3x3 matrix
__device__ int vector_det3x3(double m[3][3], double* result);

#endif
