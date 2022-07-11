#include <gputk.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
// OpenMP header
#include <omp.h>

#define gpuTKCheck(stmt)                                                  \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                      \
      break;                                                         \
    }                                                                     \
  } while (0)

// Initialize the grid boundary from corner values and linearly interpolating
// the remainder of the boundary
void boundaryInit(float* temp, int dimX, int dimY) {
	float temp_bl = 200.0f;
	float temp_br = 300.0f;
	float temp_tl = 200.0f;
	float temp_tr = 300.0f;
	memset(temp, 0, sizeof(float) * dimX * dimY);
	int offset = dimX * (dimY - 1);
	// Initialize the bottom and top boundary edges
	for (int x = 0; x < dimX; ++x) {
		float t = (float) x / (float) (dimX - 1);
		temp[x] = temp_bl * (1. - t) + temp_br * t;
		temp[x + offset] = temp_tl * (1. - t) + temp_tr * t;
	}
	offset = dimX - 1;
	// Initialize the left and right boundary edges
	for (int y = 0; y < dimY; ++y) {
		float t = (float) y / (float) (dimY - 1);
		temp[y * dimX] = temp_bl * (1. - t) + temp_tl * t;
		temp[y * dimX + offset] = temp_br * (1. - t) + temp_tr * t;
	}
}

// Compute a time step for a grid
__global__ void step_kernel(int dimX, int dimY, float alpha, float* temp,
		float* temp_next) {
	float dTx, dTy;
	//@@ Insert the code to implement the time step of the heat equation
	for (int y = blockIdx.y + 1; y < dimY - 1; y += gridDim.y) {
		for (int x = threadIdx.x + blockIdx.x * blockDim.x + 1; x < dimX - 1;
				x += blockDim.x * gridDim.x) {
			int center = x + dimX * y;
			// Compute d^2 T / d x^2
			dTx = temp[x - 1 + dimX * y] - 2 * temp[center]
					+ temp[x + 1 + dimX * y];
			// Compute d^2 T / d y^2
			dTy = temp[x + dimX * (y - 1)] - 2 * temp[center]
					+ temp[x + dimX * (y + 1)];
			// Compute step T + alpha * ( d^2 T / d x^2 + d^2 T / d y^2 )
			temp_next[center] = temp[center] + alpha * (dTx + dTy);
		}
	}
}

int main(int argc, char **argv) {
	gpuTKArg_t args;
	int devNum;          // Number of GPU's to use
	int dimX;            // X dimension of the grid
	int dimY;            // Y dimension of the grid
	int nsteps;          // Number of time steps to perform
	float alpha = 0.2;  // Diffusion coefficient
	float* temp;        // Array to store the final time step

	int nitX = 2; // Number of iterations the kernel will perform in the X direction
	int nitY = 4; // Number of iterations the kernel will perform in the Y direction

	// Read the arguments from the command line
	args = gpuTKArg_read(argc, argv);
	devNum = atoi(gpuTKArg_getInputFile(args, 0));
	dimX = atoi(gpuTKArg_getInputFile(args, 1));
	dimY = atoi(gpuTKArg_getInputFile(args, 2));
	nsteps = atoi(gpuTKArg_getInputFile(args, 3));

	std::vector<float*> temperaturesD(devNum); // Vector storing device temperatures
	std::vector<float*> temp_current(devNum); // Vector storing current device temperatures
	std::vector<float*> temp_next(devNum); // Vector storing next device temperatures

	gpuTKLog(TRACE, "The number of devices to use is ", devNum);
	gpuTKLog(TRACE, "The X dimension of the grid is ", dimX);
	gpuTKLog(TRACE, "The Y dimension of the grid is ", dimY);
	gpuTKLog(TRACE, "The number of time steps to perform is ", nsteps);

	//@@ Insert the code to allocate the temp array
	gpuTKTime_start(GPU, "Allocating CPU Memory");
	gpuTKCheck(cudaMallocHost((void** ) &temp, sizeof(float) * dimX * dimY));
	gpuTKTime_stop(GPU, "Allocating CPU Memory");
	//Initiliaze the boundary conditions for the heat equation
	gpuTKTime_start(Generic, "Initializing memory on host");
	boundaryInit(temp, dimX, dimY);
	gpuTKTime_stop(Generic, "Initializing memory on host");

	//@@ Insert the code to compute the number of rows to process in each GPU
	int ldimy = (dimY - 2) / devNum;
	//@@ Initialize the block and grid dimensions here
	dim3 blockDim = dim3(128);
	dim3 gridDim = dim3((dimX + (blockDim.x * nitX) - 1) / (blockDim.x * nitX),
			(ldimy + nitY - 1) / nitY);

	// OpenMP section to launch the step kernel
#pragma omp parallel num_threads(devNum)
	{
		// Get the number of threads in the OpenMP parallel region
		int tnum = omp_get_num_threads();
		// Get the id of the OpenMP calling thread
		int tid = omp_get_thread_num();
		// Display OpenMP information
		gpuTKLog(TRACE, "OpenMP information, thread count: ", tnum,
				", thread id: ", tid);
		//@@ Insert the code to set the device here
		gpuTKCheck(cudaSetDevice(tid));
		//@@ Insert the code to allocate the device temp arrays for each GPU
		gpuTKTime_start(GPU, "Allocating Device Memory");
		gpuTKCheck(
				cudaMalloc((void** ) &temperaturesD[tid],
						2 * (ldimy + 2) * dimX * sizeof(float)));
		gpuTKTime_stop(GPU, "Allocating Device Memory");
		//@@ Insert the code setting the temp_current and temp_next pointer for each device
		temp_current[tid] = temperaturesD[tid];
		temp_next[tid] = temperaturesD[tid] + (ldimy + 2) * dimX;
		//@@ Insert the directive to synchronize the temp_current temp_next shared variables
		//@@ across devices
#pragma omp barrier
		// Private CUDA stream to be used by a GPU
		cudaStream_t stream;
		//@@ Insert the code to create the stream to be used by a GPU
		gpuTKCheck(cudaStreamCreate(&stream));
		//@@ Insert the code to copy the respective initial time step to each GPU
		gpuTKTime_start(CopyAsync, "Copying memory to device");
		gpuTKCheck(
				cudaMemcpyAsync(temp_current[tid], temp + ldimy * dimX * tid,
						(ldimy + 2) * dimX * sizeof(float),
						cudaMemcpyHostToDevice, stream));
		gpuTKCheck(
				cudaMemcpyAsync(temp_next[tid], temp_current[tid],
						(ldimy + 2) * dimX * sizeof(float), cudaMemcpyDefault,
						stream));
		gpuTKTime_stop(CopyAsync, "Copying memory to device");
		gpuTKTime_start(Compute, "Performing CUDA computation");
		for (int i = 0; i < nsteps; ++i) {
			//@@ Launch the GPU Kernel here
			step_kernel<<<gridDim, blockDim, 0, stream>>>(dimX, ldimy + 2,
					alpha, temp_current[tid], temp_next[tid]);
			if (tid < (tnum - 1)) {
				//@@ Insert the code to copy the upper halo to the appropiate device & position
				gpuTKCheck(
						cudaMemcpyAsync(temp_next[tid + 1],
								temp_next[tid] + ldimy * dimX,
								dimX * sizeof(float), cudaMemcpyDeviceToDevice,
								stream));
			}
			if (tid > 0) {
				//@@ Insert the code to copy the bottom halo to the appropiate device & position
				gpuTKCheck(
						cudaMemcpyAsync(temp_next[tid - 1] + (ldimy + 1) * dimX,
								temp_next[tid] + dimX, dimX * sizeof(float),
								cudaMemcpyDeviceToDevice, stream));
			}
			//@@ Insert the code to synchronize work across devices here
			gpuTKCheck(cudaStreamSynchronize(stream));
#pragma omp barrier
			// Swap the pointers temp_current temp_next before the next iteration
			float* tmp = temp_current[tid];
			temp_current[tid] = temp_next[tid];
			temp_next[tid] = tmp;
		}
		gpuTKTime_stop(Compute, "Performing CUDA computation");
		//@@ Insert code to syncronize and copy back data to the host
#pragma omp barrier
		gpuTKTime_start(CopyAsync, "Copying memory to the host");
		gpuTKCheck(
				cudaMemcpyAsync(temp + ldimy * dimX * tid + dimX,
						temp_current[tid] + dimX, ldimy * dimX * sizeof(float),
						cudaMemcpyDeviceToHost, stream));
		gpuTKTime_stop(CopyAsync, "Copying memory to the host");
		//@@ Insert code to destroy the cuda stream
		gpuTKCheck(cudaStreamDestroy(stream));
	}
	// Code to output the computed solution to a file (.raw)
	// In the cmd line use ./HeatEquation -o <file name>.raw [other options]
	char* outputFileName = gpuTKArg_getOutputFile(args);
	if (outputFileName != nullptr) {
		gpuTKExport(outputFileName, gpuTKExportKind_raw, temp, dimX, dimY,
				gpuTKType_float);
	}
	// Code to compare the computed solution to the expected one
	// In the cmd line use ./HeatEquation -e <file name>.raw [other options]
	char* expectedFileName = gpuTKArg_getExpectedOutputFile(args);
	if (expectedFileName != nullptr) {
		int expectedRows, expectedCols;
		// Import the matrix using GPUTK
		float* expectedMatrix = (float*) gpuTKImport(expectedFileName,
				&expectedRows, &expectedCols);
		if (expectedRows == dimX && expectedCols == dimY) {
			// Compute the absolute error between the solutions
			float err = -1.;
			for (int i = 0; i < expectedRows; ++i) {
				for (int j = 0; j < expectedCols; ++j) {
					err = max(abs(expectedMatrix[i * expectedCols + j]
											- temp[i * expectedCols + j]), err);
				}
			}
			// Output the error, it should be a small number ( < 1e-2)
			gpuTKLog(TRACE, "Numeric error between the expected solution and the computed one: ", err);
		} else {
			gpuTKLog(ERROR, "Error the expected matrix doesn't have the correct sizes.");
		}
	}
	//@@ Free the GPU memory here
	// Freeing memory
	gpuTKTime_start(GPU, "Freeing Memory");
	for (auto ptr : temperaturesD)
		gpuTKCheck(cudaFree(ptr));
	gpuTKCheck(cudaFreeHost(temp));
	gpuTKTime_stop(GPU, "Freeing Memory");
	return 0;
}
