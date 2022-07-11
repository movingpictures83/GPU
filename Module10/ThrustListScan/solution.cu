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
  hostOutput = (float *)malloc(num_elements * sizeof(float));

  // Declare and allocate thrust device input and output vectors
  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Insert code here
  thrust::device_vector<float> deviceInput(num_elements);
  thrust::device_vector<float> deviceOutput(num_elements);
  thrust::copy(hostInput, hostInput + num_elements, deviceInput.begin());
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  // Execute vector addition
  gpuTKTime_start(
      Compute,
      "Doing the computation on the GPU and copying data back to host");
  //@@ Insert Code here
  thrust::inclusive_scan(deviceInput.begin(), deviceInput.end(),
                         deviceOutput.begin());
  thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput);

  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKSolution(args, hostOutput, num_elements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
