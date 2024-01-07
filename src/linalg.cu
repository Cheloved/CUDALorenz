#include "includes/linalg.cuh"
#include <stdio.h>

__device__ int get_projection(vector* point, vector* norm, vector* orig, vector* result)
{
    if ( point == NULL || norm == NULL || orig == NULL || result == NULL )
        return CODE_NULLPTR;

	// Assert |norm| = 1

	// Given plane equation: a(x-x0) + b(y-y0) + c(z-z0) + offset = 0
    double offset; vector_dot(norm, orig, &offset);
    offset = -offset;

	// Calculate distance between point and plane
	double distance; vector_dot(norm, point, &distance);
    distance = fabs(distance + offset);

	// Calculate coordinates
    // result = point - norm * distance - orig
    vector_mul(norm, distance, result);
    vector_sub(point, result, result);
    vector_sub(result, orig, result);

    return CODE_SUCCESS;
}

__device__ int get_basis(vector* norm, vector* orig, vector* result)
{
    if ( norm == NULL || orig == NULL || result == NULL )
        return CODE_NULLPTR;

	// Select any non-zero scalar y0
	double y0 = 1.0;
	double a = norm->x;
	double b = norm->y;
	double c = norm->z;

	// Calculate first vector, assuming x = 0, y = y0
    vector e1 = { .x = 0,
                  .y = y0,
                  .z = -b*y0 / c};
    vector_normalize(&e1, &e1);
    vector_copy(&result[0], &e1);

	// Calculate second vector as cross multiplication of e1 and n
	vector e2 = { .x = -(b * b * y0 / c) - (y0 * c),
                  .y = a * b * y0 / c,
                  .z = a * y0};
    vector_normalize(&e2, &e2);
    vector_copy(&result[1], &e2);

	// Last vector is just n
	vector e3 = { .x = a,
                  .y = b,
                  .z = c};
    vector_normalize(&e3, &e3);
    vector_copy(&result[2], &e3);

    return CODE_SUCCESS;
}

__device__ int decompose_by_basis(vector* p, vector* b, vector* result)
{
    if ( p == NULL || b == NULL || result == NULL )
        return CODE_NULLPTR;

	double det_idx[][3] = {  {b[0].x, b[1].x, b[2].x},
							 {b[0].y, b[1].y, b[2].y},
							 {b[0].z, b[1].z, b[2].z} };

    double det = 0; vector_det3x3(det_idx, &det);
	if (det == 0)
		return CODE_ZERO_DET;

	double detX_idx[3][3] = { {p->x, b[1].x, b[2].x},
						      {p->y, b[1].y, b[2].y},
						      {p->z, b[1].z, b[2].z} };
    double detX = 0; vector_det3x3(detX_idx, &detX);

	double detY_idx[3][3] = { {b[0].x, p->x, b[2].x},
							  {b[0].y, p->y, b[2].y},
							  {b[0].z, p->z, b[2].z} };
    double detY = 0; vector_det3x3(detY_idx, &detY);

	double detZ_idx[3][3] = { {b[0].x, b[1].x, p->x},
							  {b[0].y, b[1].y, p->y},
							  {b[0].z, b[1].z, p->z} };
    double detZ = 0; vector_det3x3(detZ_idx, &detZ);


    *result = { .x = detX / det,
                .y = detY / det,
                .z = detZ / det};

    return CODE_SUCCESS;
}

__device__ int get_center(vector* points, vector* result)
{
    if ( result == NULL )
        return CODE_NULLPTR;

    *result = { 0, 0, 0 };
	for (int i = 0; i < POINTS_COUNT; i++)
        vector_add(result, &points[i], result);

    vector_div(result, POINTS_COUNT, result);

	return CODE_SUCCESS;
}
