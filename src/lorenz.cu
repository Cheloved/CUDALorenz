#include "includes/defines.cuh"
#include "includes/lorenz.cuh"
#include <stdio.h>
#include <device_launch_parameters.h>


__global__ void update_points(vector* c_points)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= POINTS_COUNT )
        return;

    vector p; vector_copy(&p, &c_points[i]);
    vector velocity = { .x = 10.0 * (p.y - p.x),
                        .y = p.x * (28.0 - p.z) - p.y,
                        .z = p.x * p.y - (8.0 / 3.0) * p.z};

    vector_normalize(&velocity, &velocity);
    vector_mul(&velocity, TIME_SCALE, &velocity);

    vector_add(&c_points[i], &velocity, &c_points[i]);

    return;
}

__global__ void points_to_pixel_space(vector* c_points,
                                      vector* c_normal, vector* c_orig, vector* c_center,
                                      vector* c_basis, s_settings* c_settings, vector* c_pixels)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= POINTS_COUNT )
        return;

    // Get projection of point onto the plane
    //printf("c_points[i]: (%lf, %lf, %lf)\n", c_points[i].x, c_points[i].y, c_points[i].z);
    //printf("c_normal: (%lf, %lf, %lf)\n", c_normal->x, c_normal->y, c_normal->z);
    //printf("c_orig: (%lf, %lf, %lf)\n", c_orig->x, c_orig->y, c_orig->z);
    vector coords; get_projection(&c_points[i], c_normal, c_orig, &coords);
    //printf("After get_projection()\n");
    //printf("coords: (%lf, %lf, %lf)\n", coords.x, coords.y, coords.z);

    if ( c_settings->angleChanged )
    {
        vector_spherical_to_cartesian(c_settings->theta, c_settings->phi, 1.0, c_normal);
        //printf("theta, phi: (%lf, %lf)\n", c_settings->theta, c_settings->phi);
        vector_normalize(c_normal, c_normal);
        //printf("c_normal: (%lf, %lf, %lf)\n", c_normal->x, c_normal->y, c_normal->z);

		// Find the center of the plane
        vector_sub(c_center, c_normal, c_orig);
        //printf("c_orig: (%lf, %lf, %lf)\n", c_orig->x, c_orig->y, c_orig->z);

		// Create a basis of the plane
        get_basis(c_normal, c_orig, c_basis);
        //printf("Basis:\ne1 (%lf, %lf, %lf)\ne2 (%lf, %lf, %lf)\ne3 (%lf, %lf, %lf)\n",
        //        c_basis[0].x, c_basis[0].y, c_basis[0].z,
        //        c_basis[1].x, c_basis[1].y, c_basis[1].z,
        //        c_basis[2].x, c_basis[2].y, c_basis[2].z);

        c_settings->angleChanged = 0;
    }
    //printf("After condition\n");
    //printf("Basis:\ne1 (%lf, %lf, %lf)\ne2 (%lf, %lf, %lf)\ne3 (%lf, %lf, %lf)\n",
    //        c_basis[0].x, c_basis[0].y, c_basis[0].z,
    //        c_basis[1].x, c_basis[1].y, c_basis[1].z,
    //        c_basis[2].x, c_basis[2].y, c_basis[2].z);
    //printf("coords: (%lf, %lf, %lf)\n", coords.x, coords.y, coords.z);

    // Decompose coordinates by plane basis
    // so that z coordinate is equal to zero
    decompose_by_basis(&coords, c_basis, &c_pixels[i]);
    //printf("Result: (%lf, %lf, %lf)\n", c_pixels[i].x, c_pixels[i].y, c_pixels[i].z);
    return;
}

__global__ void update_pixels(vector* c_pixels, s_settings* c_settings, float* c_pixel_buffer)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= POINTS_COUNT )
        return;

    int x = (c_pixels[i].x + c_settings->offsetX) / c_settings->scale * SCREEN_WIDTH;
    int y = (c_pixels[i].y + c_settings->offsetY) / c_settings->scale * SCREEN_HEIGHT;

    int base_idx = (y*SCREEN_WIDTH + x) * 3;

    if ( base_idx < 0 || base_idx > SCREEN_HEIGHT*SCREEN_WIDTH * 3 )
        return;

    c_pixel_buffer[ base_idx + 0] = 1;
    c_pixel_buffer[ base_idx + 1] = 1;
    c_pixel_buffer[ base_idx + 2] = 1;

    return;
}

__global__ void clear_buffer(float* c_pixel_buffer)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i >= SCREEN_WIDTH * SCREEN_HEIGHT )
        return;

    c_pixel_buffer[i*3 + 0] = 0;
    c_pixel_buffer[i*3 + 1] = 0;
    c_pixel_buffer[i*3 + 2] = 0;

    return;
}
