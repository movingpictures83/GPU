#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gputk.h>

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  float *__restrict__ input1;
  float *__restrict__ input2;
  float *__restrict__ output;
  int inputLength;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  input1 = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  input2 = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  output = (float *)malloc(inputLength * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  //@@ Insert vector addition code here.
  gpuTKTime_start(GPU, "Copy to GPU, compute, and copy back to host.");
  gpuTKTime_stop(GPU, "Copy to GPU, compute, and copy back to host.");

  gpuTKSolution(args, output, inputLength);

  // Release memory
  free(input1);
  free(input2);
  free(output);

  return 0;
}
