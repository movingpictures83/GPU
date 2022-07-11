// Histogram Equalization

#include <gputk.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

int main(int argc, char **argv) {
  gpuTKArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(args, 0);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  inputImage    = gpuTKImport(inputImageFile);
  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);
  outputImage   = gpuTKImage_new(imageWidth, imageHeight, imageChannels);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here

  gpuTKSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
