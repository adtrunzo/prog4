#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>  // Include CUDA headers
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// Represents the objects in the system. Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass;

// Initialize GPU kernel launch configuration
dim3 blocks;
dim3 threads;

//init host memory
void initHostMemory(int numObjects) {
    hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
    hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
    mass = (double *)malloc(sizeof(double) * numObjects);

    for (int i = 0; i < numObjects; ++i) {
        hVel[i] = {0, 0, 0};  // Initialize each vector in hVel
        hPos[i] = {0, 0, 0};  // Initialize each vector in hPos
    }
}


// Free host memory
void freeHostMemory() {
    free(hVel);
    free(hPos);
    free(mass);
}

// Fill the first NUMPLANETS+1 entries of the entity arrays
void planetFill() {
    int i, j;
    double data[][7] = {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE};
    for (i = 0; i <= NUMPLANETS; i++) {
        for (j = 0; j < 3; j++) {
            hPos[i][j] = data[i][j];
            hVel[i][j] = data[i][j + 3];
        }
        mass[i] = data[i][6];
    }
}

// Fill the rest of the objects in the system randomly
void randomFill(int start, int count) {
    int i, j, c = start;
    for (i = start; i < start + count; i++) {
        for (j = 0; j < 3; j++) {
            hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
            hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
            mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
        }
    }
}

// Print the entire system to the supplied file
void printSystem(FILE* handle) {
    int i, j;
    for (i = 0; i < NUMENTITIES; i++) {
        fprintf(handle, "pos=(");
        for (j = 0; j < 3; j++) {
            fprintf(handle, "%lf,", hPos[i][j]);
        }
        printf("),v=(");
        for (j = 0; j < 3; j++) {
            fprintf(handle, "%lf,", hVel[i][j]);
        }
        fprintf(handle, "),m=%lf\n", mass[i]);
    }
}

int main(int argc, char **argv) {
    clock_t t0 = clock();
    int t_now;

    srand(1234);
    initHostMemory(NUMENTITIES);
    planetFill();
    randomFill(NUMPLANETS + 1, NUMASTEROIDS);

    #ifdef DEBUG
    printSystem(stdout);
    #endif

    // Allocate GPU memory
    cudaMalloc((void**)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&mass, sizeof(double) * NUMENTITIES);

    // Copy data from host to GPU
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    blocks.x = 1;
    blocks.y = 1;
    blocks.z = 1;
    threads.x = NUMENTITIES;
    threads.y = 1;
    threads.z = 1;

    for (t_now = 0; t_now < DURATION; t_now += INTERVAL) {
        compute(d_hVel, d_hPos, mass);  // Corrected function call
        cudaDeviceSynchronize();  // Ensure the kernel is finished before proceeding
    }
    
    // Copy data back from GPU to host
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

    clock_t t1 = clock() - t0;

    #ifdef DEBUG
    printSystem(stdout);
    #endif

    printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);

    // Free GPU memory
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(mass);

    freeHostMemory();
}

