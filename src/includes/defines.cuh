#ifndef _H_DEFINES
#define _H_DEFINES

#include "vector.cuh"

// Pi for calculations
#define PI 3.14159265359

// Amount of points to calculate
#define POINTS_COUNT 1024.0 * 1000

// Coefficient of point movement. Better not to change
#define TIME_SCALE 0.5

// Create points' tails of not
#define LEAVE_TRACES 1

// Window size
#define SCREEN_WIDTH 1920.0
#define SCREEN_HEIGHT 1080.0
//#define SCREEN_WIDTH 600.0
//#define SCREEN_HEIGHT 400.0

// Degrees per single button press
#define ROTATION_SPEED 1.0

typedef struct
{
    // Rotation variables
    double theta = 1.5; // Angle between point and X axis
    double phi = 4.8;   // Angle between point and Z axis

    // Precalculated 1 degree
    double degree = PI / 180;

    // To handle input
    char angleChanged = 0;

    // Variables to move and scale image
    double offsetX = 40.0;
    double offsetY = 40.0;
    double scale = 80.0;
} s_settings;

#endif
