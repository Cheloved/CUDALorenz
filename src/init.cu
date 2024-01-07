#include "includes/init.cuh"

double random_double(double min, double max)
{
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}

int random_init(vector* points)
{
    if ( points == NULL )
        return CODE_NULLPTR;

    for ( int i = 0; i < POINTS_COUNT; i++ )
        points[i] = (vector){ .x = random_double(-40, 40),
                              .y = random_double(-40, 40),
                              .z = random_double(-40, 40)};

    return CODE_SUCCESS;
}

int area_init(vector* points)
{
    if ( points == NULL )
        return CODE_NULLPTR;

	double x = random_double(-20, 20);
	double y = random_double(-20, 20);
	double z = random_double(0, 40);
	double d = 0.01;

	for (int i = 0; i < POINTS_COUNT; i++)
	{
		double dx = random_double(-d, d);
		double dy = random_double(-d, d);
		double dz = random_double(-d, d);

        points[i] = (vector){ .x = x + dx,
                              .y = y + dy,
                              .z = z + dz};
	}

    return CODE_SUCCESS;
}
