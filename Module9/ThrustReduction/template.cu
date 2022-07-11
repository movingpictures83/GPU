#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <gputk.h>

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  float *hostInput;
  float hostOutput;
  int inputLength;

  args = gpuTKArg_read(argc, argv); /* parse the input arguments */

  // Import host input data
  gpuTKTime_start(Generic, "Importing data to host");
  hostInput = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  gpuTKTime_stop(Generic, "Importing data to host");

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");
  // Declare and allocate thrust device input and output vectors
  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ Insert code here
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  // Copy to device
  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ Insert code here
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  // Execute vector addition
  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ Insert Code here
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");
  /////////////////////////////////////////////////////////

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(args, &hostOutput, 1);

  free(hostInput);
  return 0;
}
