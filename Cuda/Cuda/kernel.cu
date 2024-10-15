#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MATRIX_SIZE 2048

using namespace std;

__global__ void matrixMultiply(double* matA, double* matB, double* resultMat) {
    double sum;
    int rowStart = blockIdx.x, blockCount = gridDim.x;
    int colStart = threadIdx.x, threadCount = blockDim.x;

    for (int i = rowStart; i < MATRIX_SIZE; i += blockCount)
        for (int j = colStart; j < MATRIX_SIZE; j += threadCount) {
            sum = 0;
            for (int k = 0; k < MATRIX_SIZE; ++k)
                sum += matA[i * MATRIX_SIZE + k] * matB[k * MATRIX_SIZE + j];
            resultMat[i * MATRIX_SIZE + j] = sum;
        }
}

int main() {
    double* matA, * matB, * resultMat;
    int memSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);
    cudaEvent_t eventStart, eventStop;
    float elapsedTime;

    cudaMallocManaged(&matA, memSize);
    cudaMallocManaged(&matB, memSize);
    cudaMallocManaged(&resultMat, memSize);

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        matA[i] = matB[i] = 2;

    int blockCount = 32, threadCount = 1024;
    dim3 threads(threadCount);
    dim3 blocks(blockCount);

    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventStop);
    cudaEventRecord(eventStart, 0);

    matrixMultiply << <blocks, threads >> > (matA, matB, resultMat);
    cudaDeviceSynchronize();

    cudaEventRecord(eventStop, 0);
    cudaEventSynchronize(eventStop);
    cudaEventElapsedTime(&elapsedTime, eventStart, eventStop);
    printf("Block count = %i, Thread count = %i, Time = %f ms\n", blockCount, threadCount, elapsedTime);

    cudaEventDestroy(eventStart);
    cudaEventDestroy(eventStop);
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(resultMat);

    return 0;
}