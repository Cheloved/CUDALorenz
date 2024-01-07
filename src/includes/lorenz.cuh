#ifndef _H_LORENZ
#define _H_LORENZ

#include "vector.cuh"
#include "defines.cuh"
#include "linalg.cuh"


// Recalculate velocities and move points
__global__ void update_points(vector* c_points);

// Convert 3d points' coordinates to pixel space
// by projecting them onto camera plane
__global__ void points_to_pixel_space(vector* c_points,
                                      vector* c_normal, vector* c_orig, vector* c_center,
                                      vector* c_basis, s_settings* c_settings, vector* c_pixels);

__global__ void update_pixels(vector* c_pixels, s_settings* c_settings, float* c_pixel_buffer);

__global__ void clear_buffer(float* c_pixel_buffer);

#endif
