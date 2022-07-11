
#include <stdio.h>
#include <gputk.h>

#define BLOCK_SIZE 512
// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Global queuing stub
__global__ void gpu_global_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  //@@ Insert Global Queuing Code Here

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the global queue (already marked in the exchange)
}

// Block queuing stub
__global__ void gpu_block_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  //@@ Insert Block Queuing Code Here

  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the block queue
  // If full, add it to the global queue

  // Allocate space for block queue to go into global queue

  // Store block queue in global queue
}

// Host function for global queuing invocation
void gpu_global_queuing(int *nodePtrs, int *nodeNeighbors,
                        int *nodeVisited, int *currLevelNodes,
                        int *nextLevelNodes,
                        unsigned int numCurrLevelNodes,
                        int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

// Host function for block queuing invocation
void gpu_block_queuing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
                       int *currLevelNodes, int *nextLevelNodes,
                       unsigned int numCurrLevelNodes,
                       int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

int main(int argc, char *argv[]) {
  // Variables
  int numNodes;
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int numTotalNeighbors_h;
  int *currLevelNodes_h;
  int *nextLevelNodes_h;
  int numCurrLevelNodes;
  int numNextLevelNodes_h;
  int *nodePtrs_d;
  int *nodeNeighbors_d;
  int *nodeVisited_d;
  int *currLevelNodes_d;
  int *nextLevelNodes_d;
  int *numNextLevelNodes_d;

  enum Mode { GPU_GLOBAL_QUEUE = 2, GPU_BLOCK_QUEUE };

  gpuTKArg_t args = gpuTKArg_read(argc, argv);
  Mode mode    = (Mode)gpuTKImport_flag(gpuTKArg_getInputFile(args, 0));

  nodePtrs_h =
      (int *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numNodes, "Integer");
  nodeNeighbors_h = (int *)gpuTKImport(gpuTKArg_getInputFile(args, 2),
                                    &numTotalNeighbors_h, "Integer");

  nodeVisited_h =
      (int *)gpuTKImport(gpuTKArg_getInputFile(args, 3), &numNodes, "Integer");
  currLevelNodes_h = (int *)gpuTKImport(gpuTKArg_getInputFile(args, 4),
                                     &numCurrLevelNodes, "Integer");

  // (do not modify) Datasets should be consistent
  if (nodePtrs_h[numNodes] != numTotalNeighbors_h) {
    gpuTKLog(ERROR, "Datasets are inconsistent! Please report this.");
  }

  // (do not modify) Prepare next level containers (i.e. output variables)
  numNextLevelNodes_h = 0;
  nextLevelNodes_h    = (int *)malloc((numNodes) * sizeof(int));

  gpuTKLog(TRACE, "# Modes = ", mode);
  gpuTKLog(TRACE, "# Nodes = ", numNodes);
  gpuTKLog(TRACE, "# Total Neighbors = ", numTotalNeighbors_h);
  gpuTKLog(TRACE, "# Current Level Nodes = ", numCurrLevelNodes);

  // (do not modify) Allocate device variables --------------------------

  gpuTKLog(TRACE, "Allocating device variables...");

  gpuTKCheck(cudaMalloc((void **)&nodePtrs_d, (numNodes + 1) * sizeof(int)));
  gpuTKCheck(cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(int)));
  gpuTKCheck(cudaMalloc((void **)&nodeNeighbors_d,
                     nodePtrs_h[numNodes] * sizeof(int)));
  gpuTKCheck(cudaMalloc((void **)&currLevelNodes_d,
                     numCurrLevelNodes * sizeof(int)));
  gpuTKCheck(cudaMalloc((void **)&numNextLevelNodes_d, sizeof(int)));
  gpuTKCheck(
      cudaMalloc((void **)&nextLevelNodes_d, (numNodes) * sizeof(int)));
  gpuTKCheck(cudaDeviceSynchronize());

  // (do not modify) Copy host variables to device --------------------

  gpuTKLog(TRACE, "Copying data from host to device...");

  gpuTKCheck(cudaMemcpy(nodePtrs_d, nodePtrs_h, (numNodes + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(int),
                     cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h,
                     nodePtrs_h[numNodes] * sizeof(int),
                     cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
                     numCurrLevelNodes * sizeof(int),
                     cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemset(numNextLevelNodes_d, 0, sizeof(int)));
  gpuTKCheck(cudaDeviceSynchronize());

  // (do not modify) Launch kernel ----------------------------------------

  printf("Launching kernel ");

  if (mode == GPU_GLOBAL_QUEUE) {
    gpuTKLog(INFO, "(GPU with global queuing)...");
    gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                       currLevelNodes_d, nextLevelNodes_d,
                       numCurrLevelNodes, numNextLevelNodes_d);
    gpuTKCheck(cudaDeviceSynchronize());
  } else if (mode == GPU_BLOCK_QUEUE) {
    gpuTKLog(INFO, "(GPU with block and global queuing)...");
    gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                      currLevelNodes_d, nextLevelNodes_d,
                      numCurrLevelNodes, numNextLevelNodes_d);
    gpuTKCheck(cudaDeviceSynchronize());
  } else {
    gpuTKLog(ERROR, "Invalid mode!\n");
    exit(0);
  }

  // (do not modify) Copy device variables from host ----------------------

  gpuTKLog(INFO, "Copying data from device to host...");

  gpuTKCheck(cudaMemcpy(&numNextLevelNodes_h, numNextLevelNodes_d,
                     sizeof(int), cudaMemcpyDeviceToHost));
  gpuTKCheck(cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d,
                     numNodes * sizeof(int), cudaMemcpyDeviceToHost));
  gpuTKCheck(cudaMemcpy(nodeVisited_h, nodeVisited_d, numNodes * sizeof(int),
                     cudaMemcpyDeviceToHost));
  gpuTKCheck(cudaDeviceSynchronize());

  // (do not modify) Verify correctness
  // -------------------------------------
  // Only check that the visited nodes match the reference implementation

  gpuTKSolution(args, nodeVisited_h, numNodes);

  // (do not modify) Free memory
  // ------------------------------------------------------------
  free(nodePtrs_h);
  free(nodeVisited_h);
  free(nodeNeighbors_h);
  free(currLevelNodes_h);
  free(nextLevelNodes_h);
  gpuTKCheck(cudaFree(nodePtrs_d));
  gpuTKCheck(cudaFree(nodeVisited_d));
  gpuTKCheck(cudaFree(nodeNeighbors_d));
  gpuTKCheck(cudaFree(currLevelNodes_d));
  gpuTKCheck(cudaFree(numNextLevelNodes_d));
  gpuTKCheck(cudaFree(nextLevelNodes_d));

  return 0;
}
