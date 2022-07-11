// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}

#include <gputk.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void fixup(float *input, float *aux, int len) {
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
  if (blockIdx.x) {
    if (start + t < len)
      input[start + t] += aux[blockIdx.x - 1];
    if (start + BLOCK_SIZE + t < len)
      input[start + BLOCK_SIZE + t] += aux[blockIdx.x - 1];
  }
}

__global__ void scan(float *input, float *output, float *aux, int len) {
  // Load a segment of the input vector into shared memory
  __shared__ float scan_array[BLOCK_SIZE << 1];
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
  if (start + t < len)
    scan_array[t] = input[start + t];
  else
    scan_array[t] = 0;
  if (start + BLOCK_SIZE + t < len)
    scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
  else
    scan_array[BLOCK_SIZE + t] = 0;
  __syncthreads();

  // Reduction
  int stride;
  for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
    int index = (t + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE)
      scan_array[index] += scan_array[index - stride];
    __syncthreads();
  }

  // Post reduction
  for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE)
      scan_array[index + stride] += scan_array[index];
    __syncthreads();
  }

  if (start + t < len)
    output[start + t] = scan_array[t];
  if (start + BLOCK_SIZE + t < len)
    output[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

  if (aux && t == 0)
    aux[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the list

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numElements);
  cudaHostAlloc(&hostOutput, numElements * sizeof(float),
                cudaHostAllocDefault);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The number of input elements in the input is ",
        numElements);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  gpuTKCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  // XXX the size is fixed for ease of implementation.
  cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(float));
  cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(float));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Clearing output memory.");
  gpuTKCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Clearing output memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  gpuTKCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil((float)numElements / (BLOCK_SIZE << 1));
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  gpuTKLog(TRACE, "The number of blocks is ", numBlocks);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAuxArray,
                              numElements);
  cudaDeviceSynchronize();
  scan<<<dim3(1, 1, 1), dimBlock>>>(deviceAuxArray, deviceAuxScannedArray,
                                    NULL, BLOCK_SIZE << 1);
  cudaDeviceSynchronize();
  fixup<<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray,
                               numElements);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  gpuTKCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostOutput, numElements);

  free(hostInput);
  cudaFreeHost(hostOutput);

  return 0;
}
