#include <gputk.h>

// Compute C = A * B
// Sgemm stands for single precision general matrix-matrix multiply
__global__ void sgemm(float *A, float *B, float *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numARows && col < numBColumns) {
    float sum = 0;
    for (int ii = 0; ii < numAColumns; ii++) {
      sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numARows * numBColumns * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  numCRows    = numARows;
  numCColumns = numBColumns;

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  gpuTKLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  gpuTKCheck(cudaMalloc((void **)&deviceA,
                     numARows * numAColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceB,
                     numBRows * numBColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceC,
                     numARows * numBColumns * sizeof(float)));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  gpuTKCheck(cudaMemcpy(deviceA, hostA,
                     numARows * numAColumns * sizeof(float),
                     cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(deviceB, hostB,
                     numBRows * numBColumns * sizeof(float),
                     cudaMemcpyHostToDevice));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(16, 16);
// changed to BColumns and ARows from Acolumns and BRows
  dim3 gridDim(ceil(((float)numBColumns) / blockDim.x),
               ceil(((float)numARows) / blockDim.y));

  gpuTKLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
  gpuTKLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  gpuTKCheck(cudaMemset(deviceC, 0, numARows * numBColumns * sizeof(float)));
  sgemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows,
                               numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  gpuTKCheck(cudaMemcpy(hostC, deviceC,
                     numARows * numBColumns * sizeof(float),
                     cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numARows, numBColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
