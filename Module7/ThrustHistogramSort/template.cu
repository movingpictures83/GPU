#include <gputk.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput, *hostBins;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)gpuTKImport(gpuTKArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  // Copy the input to the GPU
  gpuTKTime_start(GPU, "Allocating GPU memory");
  //@@ Insert code here
  gpuTKTime_stop(GPU, "Allocating GPU memory");

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  num_bins = deviceInput.back() + 1;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here

  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.

  // Copy the histogram to the host
  //@@ insert code here

  // Check the solution is correct
  gpuTKSolution(args, hostBins, num_bins);

  // Free space on the host
  //@@ insert code here
  free(hostBins);

  return 0;
}
