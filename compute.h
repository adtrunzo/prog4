#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "vector.h"

__global__ void compute(vector3 *d_hVel, vector3 *d_hPos, double *mass);

#endif

