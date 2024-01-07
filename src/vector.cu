#include "includes/vector.cuh"
#include <stdio.h>

__device__ int vector_copy(vector* dest, vector* source)
{
    if ( dest == NULL || source == NULL )
        return CODE_NULLPTR;

    dest->x = source->x;
    dest->y = source->y;
    dest->z = source->z;

    return CODE_SUCCESS;
}

__device__ int vector_add(vector* a, vector* b, vector* result)
{
    if ( a == NULL || b == NULL || result == NULL )
        return CODE_NULLPTR;

    result->x = a->x + b->x;
    result->y = a->y + b->y;
    result->z = a->z + b->z;

    return CODE_SUCCESS;
}

__device__ int vector_sub(vector* a, vector* b, vector* result)
{
    if ( a == NULL || b == NULL || result == NULL )
        return CODE_NULLPTR;

    result->x = a->x - b->x;
    result->y = a->y - b->y;
    result->z = a->z - b->z;

    return CODE_SUCCESS;
}

__device__ int vector_mul(vector* v, double s, vector* result)
{
    if ( v == NULL || result == NULL )
        return CODE_NULLPTR;

    result->x = v->x * s;
    result->y = v->y * s;
    result->z = v->z * s;

    return CODE_SUCCESS;
}

__device__ int vector_div(vector* v, double s, vector* result)
{
    if ( v == NULL || result == NULL )
        return CODE_NULLPTR;

    if ( s == 0 )
        return CODE_ZERO_DIV;

    result->x = v->x / s;
    result->y = v->y / s;
    result->z = v->z / s;

    return CODE_SUCCESS;
}

__device__ int vector_dot(vector* a, vector* b, double* result)
{
    if ( a == NULL || b == NULL || result == NULL )
        return CODE_NULLPTR;

    *result = a->x*b->x + a->y*b->y + a->z*b->z;

    return CODE_SUCCESS;
}

__device__ int vector_len(vector* v, double* result)
{
    if ( v == NULL || result == NULL )
        return CODE_NULLPTR;

    *result = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);

    return CODE_SUCCESS;
}

__device__ int vector_normalize(vector* v, vector* result)
{
    double length;
    int ret = vector_len(v, &length);
    
    if ( ret != CODE_SUCCESS )
        return ret;

    ret = vector_div(v, length, result);
    if ( ret != CODE_SUCCESS )
        return ret;

    return CODE_SUCCESS;
}

__device__ int vector_spherical_to_cartesian(double theta, double phi, double r,
                                             vector* result)
{
    if ( result == NULL )
        return CODE_NULLPTR;

	result->x = r * cos(theta) * sin(phi);
	result->y = r * sin(theta) * sin(phi);
	result->z = r * cos(phi);

    return CODE_SUCCESS;
}

__device__ int vector_det3x3(double m[3][3], double* result)
{
    if ( m == NULL || result == NULL )
        return CODE_NULLPTR;

	*result = m[0][0] * m[1][1] * m[2][2] +
	          m[0][1] * m[1][2] * m[2][0] +
		      m[1][0] * m[2][1] * m[0][2] -
			  m[0][2] * m[1][1] * m[2][0] -
			  m[0][1] * m[1][0] * m[2][2] -
			  m[1][2] * m[2][1] * m[0][0];

    return CODE_SUCCESS;
}
