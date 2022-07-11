#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <gputk.h>

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostInput, *hostOutput; // The input 1D list
  int num_elements;              // number of elements in the input list

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &num_elements);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The number of input elements in the input is ",
        num_elements);

  // Declare and allocate the host output array
  //@@ Insert code here

  // Declare and allocate thrust device input and output vectors
  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Insert code here
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  // Execute vector addition
  gpuTKTime_start(
      Compute,
      "Doing the computation on the GPU and copying data back to host");
  //@@ Insert Code here
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKSolution(args, hostOutput, num_elements);

  // Free Host Memory
  free(hostInput);
  //@@ Insert code here

  return 0;
}
