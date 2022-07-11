#include <gputk.h>

#define NUM_BINS 128

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram_kernel(const char *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Privatized bins
  extern __shared__ unsigned int bins_s[];
  for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
       binIdx += blockDim.x) {
    bins_s[binIdx] = 0;
  }
  __syncthreads();

  // Histogram
  for (unsigned int i = tid; i < num_elements;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&(bins_s[(unsigned int)input[i]]), 1);
  }
  __syncthreads();

  // Commit to global memory
  for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
       binIdx += blockDim.x) {
    atomicAdd(&(bins[binIdx]), bins_s[binIdx]);
  }
}

void histogram(const char *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins) {

  // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(256), gridDim(30);
    histogram_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength;
  char *hostInput;
  unsigned int *hostBins;
  char *deviceInput;
  unsigned int *deviceBins;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (char *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength, "Text");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);
  gpuTKLog(TRACE, "The number of bins is ", NUM_BINS);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK(cudaMalloc((void **)&deviceInput, inputLength));
  CUDA_CHECK(
      cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, inputLength,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  gpuTKLog(TRACE, "Launching kernel");
  gpuTKTime_start(Compute, "Performing CUDA computation");
  // @@ Insert code here
  histogram(deviceInput, deviceBins, inputLength, NUM_BINS);
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceBins));
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  gpuTKSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
