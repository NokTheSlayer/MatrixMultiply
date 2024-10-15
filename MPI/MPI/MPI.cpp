#include <iostream>
#include <mpi.h>

#define MATRIX_SIZE 2048

using namespace std;

double* matrixMultiply(double* matA, double* matB, int processID, int totalProcesses) {
    double* resultMatrix = new double[MATRIX_SIZE * MATRIX_SIZE];

    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = processID; col < MATRIX_SIZE; col += totalProcesses) {
            resultMatrix[row * MATRIX_SIZE + col] = 0.0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                resultMatrix[row * MATRIX_SIZE + col] += matA[row * MATRIX_SIZE + k] * matB[k * MATRIX_SIZE + col];
            }
        }
    }

    return resultMatrix;
}

int main(int argc, char* argv[]) {
    double* matA, * matB, * localResult, * globalResult;
    int processID, totalProcesses;
    double startTime;

    matA = new double[MATRIX_SIZE * MATRIX_SIZE];
    matB = new double[MATRIX_SIZE * MATRIX_SIZE];
    globalResult = new double[MATRIX_SIZE * MATRIX_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processID);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);

    if (processID == 0) {
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                matA[i * MATRIX_SIZE + j] = matB[i * MATRIX_SIZE + j] = 2;
            }
        }
        startTime = MPI_Wtime();
        for (int proc = 1; proc < totalProcesses; proc++) {
            MPI_Send(matA, MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
            MPI_Send(matB, MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(matA, MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(matB, MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    localResult = matrixMultiply(matA, matB, processID, totalProcesses);
    MPI_Reduce(localResult, globalResult, MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (processID == 0) {
        cout << "Total processes = " << totalProcesses << ", time = " << MPI_Wtime() - startTime << "s\n";
    }

    MPI_Finalize();

    delete[] matA;
    delete[] matB;
    delete[] localResult;
    delete[] globalResult;

    return 0;
}