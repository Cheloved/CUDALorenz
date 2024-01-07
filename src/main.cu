#include <GL/freeglut_std.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

// Lib for graphics
#include <GL/glut.h>

// CUDA libs
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "includes/defines.cuh"
#include "includes/vector.cuh"
#include "includes/init.cuh"
#include "includes/linalg.cuh"
#include "includes/lorenz.cuh"

// === Global Variables === //
// Array of pixels to draw
float* pixel_buffer;
float* c_pixel_buffer;

// Array of points's vectors
vector* points;
vector* c_points = 0;

// Array of points's pixel coordinates
vector* pixel_coords;
vector* c_pixel_coords;

vector normal; // Normal of camera plane
vector* c_normal;

vector orig;   // Origin of camera plane
vector* c_orig;

vector center;   // Center of points
vector* c_center;

vector* basis; // 3 basis vectors of camera plane
vector* c_basis;

s_settings settings = 
{
    .theta = 1.5,
    .phi = 4.8,

    .degree = PI / 180,

    .angleChanged = 1,

    .offsetX = 40.0,
    .offsetY = 40.0,
    .scale = 80.0
};

s_settings* c_settings;

int threads = 1024;
int blocks = ceil(POINTS_COUNT / threads);

int threads_screen = 1024;
int blocks_screen = ceil(SCREEN_WIDTH*SCREEN_HEIGHT / threads_screen);

void check_cuda_error(char* text)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr, text, cudaGetErrorString(error) );
        exit(-1);
    }
}

void display (void) 
{
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    clear_buffer <<< blocks_screen, threads_screen >>> (c_pixel_buffer);
    check_cuda_error((char*)"Error after clear_buffer()(): %s\n");

    update_points <<< blocks, threads >>> (c_points);
    check_cuda_error((char*)"Error after update_points(): %s\n");

    points_to_pixel_space <<< blocks, threads >>> (c_points, c_normal, c_orig, c_center, c_basis, c_settings, c_pixel_coords);
    check_cuda_error((char*)"Error after points_to_pixel_space(): %s\n");

    update_pixels <<< blocks, threads >>> (c_pixel_coords, c_settings, c_pixel_buffer);
    check_cuda_error((char*)"Error after update_pixels(): %s\n");

    cudaMemcpy(pixel_buffer, c_pixel_buffer, SCREEN_WIDTH*SCREEN_HEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error((char*)"Error after memcpy: %s\n");
    glDrawPixels(SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_FLOAT, pixel_buffer);
    glutSwapBuffers();
    glutPostRedisplay();

    //glFlush();
}

void copy_settings_to_device()
{
    cudaMemcpy(c_normal, &normal, sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(c_orig, &orig, sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(c_center, &center, sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(c_settings, &settings, sizeof(s_settings), cudaMemcpyHostToDevice);
}


int main(int argc, char** argv)
{
    // Initialize and allocate memory
    pixel_buffer = (float*)calloc(SCREEN_WIDTH*SCREEN_HEIGHT*3, sizeof(float));
    cudaMalloc((void**)&c_pixel_buffer, SCREEN_WIDTH*SCREEN_HEIGHT*3*sizeof(float));

    points = (vector*)calloc(POINTS_COUNT, sizeof(vector));
    cudaMalloc((void**)&c_points, POINTS_COUNT*sizeof(vector));

    cudaMalloc((void**)&c_pixel_coords, POINTS_COUNT*sizeof(vector));

	normal = { -0.284223, 0.460413, 0.840974 };
    cudaMalloc((void**)&c_normal, sizeof(vector));

	orig = { -40.0, -40.0, -40.0 };
    cudaMalloc((void**)&c_orig, sizeof(vector));

	center = { 0.0, 0.0, 27.0 };
    cudaMalloc((void**)&c_center, sizeof(vector));

    cudaMalloc((void**)&c_basis, sizeof(vector)*3);

    cudaMalloc((void**)&c_settings, sizeof(s_settings));

    copy_settings_to_device();

    // Initialize points
    random_init(points);
    cudaMemcpy(c_points, points, POINTS_COUNT*sizeof(vector), cudaMemcpyHostToDevice);

    // Initialize OpenGL 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_MULTISAMPLE);

    // Enable transparency
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable( GL_BLEND );

    glShadeModel(GL_SMOOTH);

    // Set window size and position
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutInitWindowPosition(0, 0);

    // Set window name
    glutCreateWindow("CUDALorenz");

    // Set background color
    glClearColor(0.0, 0.0, 0.0, 1.0);
      
    // Set size of a drawing point
    glPointSize(1.0);

    // Prepare matrix
    glMatrixMode(GL_PROJECTION); 
    glLoadIdentity();
      
    // Center the coordinates
    gluOrtho2D(-SCREEN_WIDTH/2, SCREEN_WIDTH/2, -SCREEN_HEIGHT/2, SCREEN_HEIGHT/2);

    glutDisplayFunc(display);
    //glutKeyboardFunc(processNormalKeys);
    glutMainLoop();

    return 0;
}
