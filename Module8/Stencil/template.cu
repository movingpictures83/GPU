#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void stencil(float *output, float *input, int width, int height,
                        int depth) {
  //@@ INSERT CODE HERE
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  //@@ INSERT CODE HERE
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  gpuTKImage_t input;
  gpuTKImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = gpuTKArg_read(argc, argv);

  inputFile = gpuTKArg_getInputFile(arg, 0);

  input = gpuTKImport(inputFile);

  width  = gpuTKImage_getWidth(input);
  height = gpuTKImage_getHeight(input);
  depth  = gpuTKImage_getChannels(input);

  output = gpuTKImage_new(width, height, depth);

  hostInputData  = gpuTKImage_getData(input);
  hostOutputData = gpuTKImage_getData(output);

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData,
             width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData,
             width * height * depth * sizeof(float));
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData,
             width * height * depth * sizeof(float),
             cudaMemcpyHostToDevice);
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData,
             width * height * depth * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  gpuTKImage_delete(output);
  gpuTKImage_delete(input);

  return 0;
}