#include <gputk.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Data type to use for the matrices
typedef double real_t;

// Initialize a matrix A with random values between min and max
void matrix_init(real_t *A, int rows, int cols, real_t min=0., real_t max=1.) {
	for(int i = 0; i < (rows * cols); ++i)
		A[i] = (max - min) * (rand() % 51 ) / 50. + min;
}

// Verify the result of C = A * B
double verify(real_t *C, real_t *A, real_t *B, int CRows, int CCols, int ACols) {
	double error = 0;
#pragma omp parallel for collapse(2)
	for(int i = 0; i < CRows; ++i) {
		for(int j = 0; j < CCols; ++j) {
			real_t Cij = 0;
			for(int k = 0; k < ACols; ++k)
				Cij += A[i * ACols + k] * B[k * CCols + j];
			error += abs(Cij - C[i * CCols + j]);
		}
	}
	return error;
}

// Compute C = A * B
template <int TILE_SIZE=32> __global__ void matrixMultiply(real_t *C,
		real_t *A, real_t *B, int CRows, int CCols, int ACols) {
	//@@ Insert code to implement matrix multiplication here
}

int main(int argc, char **argv) {
	constexpr int TILE_SIZE=32;
	gpuTKArg_t args;
	real_t *A; // The A matrix
	real_t *B; // The B matrix
	real_t *C; // The output matrix
	int ARows; // number of rows in the matrix A
	int ACols; // number of cols in the matrix A
	int BRows; // number of rows in the matrix B
	int BCols; // number of cols in the matrix B
	int CRows; // number of rows in the matrix C
	int CCols; // number of cols in the matrix C
	int concurrentAccessQ = 0;
	int device; // current device
	cudaStream_t stream;  // stream to run the computations

	// Get current device
	gpuTKCheck(cudaGetDevice(&device));
	// Get cudaDevAttrConcurrentManagedAccess device property
	gpuTKCheck(cudaDeviceGetAttribute(&concurrentAccessQ,
			cudaDevAttrConcurrentManagedAccess,device));
	// Create stream
	gpuTKCheck(cudaStreamCreate(&stream));

	args = gpuTKArg_read(argc, argv);
	// Read matrices input sizes
	ARows = atoi(gpuTKArg_getInputFile(args, 0));
	ACols = atoi(gpuTKArg_getInputFile(args, 1));
	BCols = atoi(gpuTKArg_getInputFile(args, 2));

	// Set CRows, CCols and BRows
	CRows = ARows;
	CCols = BCols;
	BRows = ACols;

	gpuTKLog(TRACE, "The dimensions of A are ", ARows, " x ", ACols);
	gpuTKLog(TRACE, "The dimensions of B are ", BRows, " x ", BCols);
	gpuTKLog(TRACE, "The dimensions of C are ", CRows, " x ", CCols);

	gpuTKTime_start(GPU, "Allocating Managed Memory");
	//@@ Insert code to allocate magaed memory here
	gpuTKTime_start(GPU, "Allocating Managed Memory");

	gpuTKTime_start(GPU, "Prefetching and advising Managed Memory");
	//@@ Insert code to prefetch data and set advises here
	if(concurrentAccessQ != 0) {
		//@@ Insert code to prefetch data here
	}
	gpuTKTime_stop(GPU, "Prefetching and advising Managed Memory");

	gpuTKTime_start(Generic, "Initializing memory on host");
	// Initialize matrices A and B with random numbers
	matrix_init(A, CRows, ACols);
	matrix_init(B, ACols, CCols);
	gpuTKTime_stop(Generic, "Initializing memory on host");

	gpuTKTime_start(CopyAsync, "Prefetching GPU memory to device");
	if(concurrentAccessQ!=0) {
		//@@ Insert code to prefetch data here
	}
	gpuTKTime_stop(CopyAsync, "Prefetching GPU memory to device");

	gpuTKTime_start(Compute, "Performing CUDA computation");
	//@@ Initialize the grid and block dimensions here
	//@@ Launch the GPU Kernel here
	gpuTKCheck(cudaDeviceSynchronize());
	gpuTKTime_stop(Compute, "Performing CUDA computation");

	gpuTKTime_start(CopyAsync, "Prefetching GPU memory to device");
	if(concurrentAccessQ!=0) {
		//@@ Insert code to prefetch data here
	}
	gpuTKTime_stop(CopyAsync, "Prefetching GPU memory to host");

	gpuTKTime_start(Generic, "Verifying matrix multiplication result");
	// Computing numeric error of the computation
	// The error should be somewhat less than 10E-6
	gpuTKLog(TRACE, "Numeric error: ", verify(C, A, B, CRows, CCols, ACols));
	gpuTKTime_stop(Generic, "Verifying matrix multiplication result");

	gpuTKTime_start(GPU, "Freeing Managed Memory");
	//@@ Free the GPU memory here
	gpuTKTime_stop(GPU, "Freeing Managed Memory");
	return 0;
}
