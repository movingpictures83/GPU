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
  cudaMallocHost((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMallocHost((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMallocHost((void **)&deviceOutput, inputLength * sizeof(float));
#endif

#ifndef PINNED
  //@@ Part A: Allocate GPU memory here using cudaMalloc here - this is
  //@@ non pinned version.
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));
#endif 

  gpuTKTime_stop(GPU, "Allocating Pinned memory.");

#ifdef PINNED
  //@@ Part B: GPUTK artificat to make the lab compatible for pinned memory
  memcpy(deviceInput1, hostInput1, inputLength * sizeof(float));
  memcpy(deviceInput2, hostInput2, inputLength * sizeof(float));
#endif

#ifndef PINNED
  //@@ Part A: Setup streams for non pinned version. Here in this example,
  //@@ we have 32 streams.
  unsigned int numStreams = 32; 
  cudaStream_t stream[numStreams];
  for(unsigned int i=0; i < numStreams; i++)
          cudaStreamCreate(&stream[i]);

  //@@ Part A: Create segments
  unsigned int numSegs = numStreams; 
  unsigned int segSize = (inputLength + numSegs -1 )/numSegs; 
 
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Part A: perform parallel vector addition with different streams. 
  for (unsigned int s =0; s<numSegs; s++){
          unsigned int start = s*segSize; 
          unsigned int end   = (start + segSize < (unsigned int) inputLength)? \
          start+segSize : inputLength;
          unsigned int Nseg  = end - start; 
          //@@ Part A: Copy data to the device memory in segments asynchronously
          cudaMemcpyAsync(&deviceInput1[start], &hostInput1[start], \
          Nseg*sizeof(float), cudaMemcpyHostToDevice, stream[s]);
          cudaMemcpyAsync(&deviceInput2[start], &hostInput2[start], \
          Nseg*sizeof(float), cudaMemcpyHostToDevice, stream[s]);
          const unsigned int numThreads = 32;
          const unsigned int numBlocks = (Nseg+numThreads-1)/numThreads; 

          //@@ Part A: Invoke CUDA Kernel
          vecAdd<<<numBlocks, numThreads, 0, stream[s]>>>(&deviceInput1[start], \
          &deviceInput2[start], &deviceOutput[start], Nseg);

          cudaMemcpyAsync(&hostOutput[start], &deviceOutput[start], \
          Nseg*sizeof(float), cudaMemcpyDeviceToHost, stream[s]);
  }
  //@@ Part A: Synchronize
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");
#endif

#ifdef PINNED
  //@@ Part B: Initialize the grid and block dimensions here
  dim3 blockDim(32);
  dim3 gridDim(ceil(((float)inputLength) / ((float)blockDim.x)));

  gpuTKLog(TRACE, "Block dimension is ", blockDim.x);
  gpuTKLog(TRACE, "Grid dimension is ", gridDim.x);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Part B: Launch the GPU Kernel here
  vecAdd<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput,
                                inputLength);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  //@@ Part B: GPUTK artificat to make the lab compatible
  memcpy(hostOutput, deviceOutput, inputLength * sizeof(float));
#endif

  gpuTKTime_start(GPU, "Freeing Pinned Memory");
#ifndef PINNED 
  //@@ Destory cudaStream
  for(unsigned int i=0; i < numStreams; i++)
          cudaStreamDestroy(stream[0]);

  //@@ Part A: Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
#endif 
#ifdef PINNED
  //@@ Part B: Free the GPU memory here
  cudaFreeHost(deviceInput1);
  cudaFreeHost(deviceInput2);
  cudaFreeHost(deviceOutput);
#endif 
  gpuTKTime_stop(GPU, "Freeing Pinned Memory");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
