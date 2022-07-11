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

#define BLUR_SIZE 5

//@@ INSERT CODE HERE
#define TILE_WIDTH 16
__global__ void blurKernel(float *out, float *in, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float pixVal = 0;
    float pixels = 0;

    // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
    for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow) {
      for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1; ++blurcol) {

        int currow = row + blurrow;
        int curcol = col + blurcol;
        // Verify we have a valid image pixel
        if (currow > -1 && currow < height && curcol > -1 &&
            curcol < width) {
          pixVal += in[currow * width + curcol];
          pixels++; // Keep track of number of pixels in the avg
        }
      }
    }

    // Write our new pixel value out
    out[row * width + col] = (pixVal / pixels);
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(args, 0);

  inputImage = gpuTKImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = gpuTKImage_getWidth(inputImage);
  imageHeight = gpuTKImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = gpuTKImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH),
               ceil((float)imageHeight / TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                    deviceInputImageData, imageWidth,
                                    imageHeight);
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  gpuTKTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
