#include <gputk.h>

//@@ Part B: Comment out the below line for Part B lab setup
//#define PINNED 1

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(GPU, "Allocating Pinned memory.");

#ifdef PINNED
  //@@ Part B: Allocate GPU memory here using pinned memory here
#endif

#ifndef PINNED
  //@@ Part A: Allocate GPU memory here using cudaMalloc here - this is
  //@@ non pinned version.
#endif 

  gpuTKTime_stop(GPU, "Allocating Pinned memory.");

#ifdef PINNED
  //@@ Part B: GPUTK artificat to make the lab compatible for pinned memory
#endif

#ifndef PINNED
  //@@ Part A: Setup streams for non pinned version. Here in this example,
  //@@ we have 32 streams.

  //@@ Part A: Create segments
 
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Part A: perform parallel vector addition with different streams. 
  for (unsigned int s =0; s<numSegs; s++){
          //@@ Part A: Asynchronous copy data to the device memory in segments 

          //@@ Part A: Invoke CUDA Kernel
          
          //@@ Part A: Asynchronous copy data from the device memory in segments 
  }
  //@@ Part A: Synchronize
  gpuTKTime_stop(Compute, "Performing CUDA computation");
#endif

#ifdef PINNED
  //@@ Part B: Initialize the grid and block dimensions here

  gpuTKLog(TRACE, "Block dimension is ", blockDim.x);
  gpuTKLog(TRACE, "Grid dimension is ", gridDim.x);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Part B: Launch the GPU Kernel here
  
  
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  //@@ Part B: GPUTK artificat to make the lab compatible
  memcpy(hostOutput, deviceOutput, inputLength * sizeof(float));
#endif

  gpuTKTime_start(GPU, "Freeing Pinned Memory");
#ifndef PINNED 
  //@@ Destory cudaStream

  //@@ Part A: Free the GPU memory here

#endif 
#ifdef PINNED
  //@@ Part B: Free the GPU memory here


#endif 
  gpuTKTime_stop(GPU, "Freeing Pinned Memory");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
