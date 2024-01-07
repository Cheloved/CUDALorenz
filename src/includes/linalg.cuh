#ifndef _H_LINALG
#define _H_LINALG

#include "vector.cuh"
#include "codes.cuh"
#include "defines.cuh"

// Returns coordinates of projection of point onto the given plane,
// described by normal vector and origin point
__device__ int get_projection(vector* point, vector* norm, vector* orig, vector* result);

// Returns basis (e1, e2, n),
// where e1, e2 belong to the plane;
// e1, e1 perpendicular to each other
// n is perpendicular to them
__device__ int get_basis(vector* norm, vector* orig, vector* result);

// Solves the system of linear equations to
// decompose point by given basis using Cramer's rule
__device__ int decompose_by_basis(vector* p, vector* b, vector* result);

// Get point in the middle
__device__ int get_center(vector* points, vector* result);

#endif
