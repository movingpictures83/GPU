#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <gputk.h>

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  int inputLength;

  args = gpuTKArg_read(argc, argv); /* parse the input arguments */

  // Import host input data
  gpuTKTime_start(Generic, "Importing data to host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  gpuTKTime_stop(Generic, "Importing data to host");

  // Declare and allocate host output
  //@@ Insert code here
  hostOutput = (float *)malloc(sizeof(float) * inputLength);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  // Declare and allocate thrust device input and output vectors
  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ Insert code here
  thrust::device_vector<float> deviceInput1(inputLength);
  thrust::device_vector<float> deviceInput2(inputLength);
  thrust::device_vector<float> deviceOutput(inputLength);
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  // Copy to device
  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ Insert code here
  thrust::copy(hostInput1, hostInput1 + inputLength, deviceInput1.begin());
  thrust::copy(hostInput2, hostInput2 + inputLength, deviceInput2.begin());
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  // Execute vector addition
  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ Insert Code here
  thrust::transform(deviceInput1.begin(), deviceInput1.end(),
                    deviceInput2.begin(), deviceOutput.begin(),
                    thrust::plus<float>());
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");
  /////////////////////////////////////////////////////////

  // Copy data back to host
  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ Insert code here
  thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  return 0;
}
