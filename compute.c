#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

__global__ void computePairwiseAccelerations(vector3 *d_hVel, vector3 *d_hPos, double *d_mass, vector3 *d_accels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < NUMENTITIES; ++j) {
        if (i != j) {
            vector3 distance;
            for (int k = 0; k < 3; ++k) {
                distance[k] = d_hPos[i][k] - d_hPos[j][k];
            }

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -GRAV_CONSTANT * d_mass[j] / magnitude_sq;

            d_accels[i][0] += accelmag * distance[0] / magnitude;
            d_accels[i][1] += accelmag * distance[1] / magnitude;
            d_accels[i][2] += accelmag * distance[2] / magnitude;
        }
    }
}

__global__ void sumRows(vector3 *d_accels, vector3 *d_accel_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    vector3 accel_sum = {0, 0, 0};
    for (int j = 0; j < NUMENTITIES; ++j) {
        accel_sum[0] += d_accels[i * NUMENTITIES + j][0];
        accel_sum[1] += d_accels[i * NUMENTITIES + j][1];
        accel_sum[2] += d_accels[i * NUMENTITIES + j][2];
    }

    d_accel_sum[i] = accel_sum;
}

__global__ void updateVelocitiesAndPositions(vector3 *d_hVel, vector3 *d_hPos, vector3 *d_accel_sum, double INTERVAL) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < 3; ++k) {
        d_hVel[i][k] += d_accel_sum[i][k] * INTERVAL;
        d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
    }
}

void compute(vector3 *d_hVel, vector3 *d_hPos, double *d_mass) {
    dim3 blocks(1, 1, 1);
    dim3 threads(NUMENTITIES, 1, 1);

    vector3 *d_accels, *d_accel_sum;

    cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc((void**)&d_accel_sum, sizeof(vector3) * NUMENTITIES);

    cudaMemset(d_accels, 0, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    computePairwiseAccelerations<<<blocks, threads>>>(d_hVel, d_hPos, d_mass, d_accels);
    cudaDeviceSynchronize();  

    sumRows<<<blocks, threads>>>(d_accels, d_accel_sum);
    cudaDeviceSynchronize();

    updateVelocitiesAndPositions<<<blocks, threads>>>(d_hVel, d_hPos, d_accel_sum, INTERVAL);
    cudaDeviceSynchronize();

    cudaFree(d_accels);
    cudaFree(d_accel_sum);
}

