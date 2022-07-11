#include "graph.h"
#include "compare.h"
#include <gputk.h>


#define gpuTKCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(1);
    }
}


__global__ void bfs_neighbors_kernel(CSRGraph csrGraph, unsigned int* nodeLevel, \
    unsigned int* currFrontier, unsigned int* numCurrFrontier,  unsigned int level, \
    unsigned int start, unsigned int numNeighbors) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numNeighbors) {
        unsigned int edge = start + i;
        unsigned int neighbor = csrGraph.neighbors[edge];
        if(atomicCAS(&nodeLevel[neighbor], UINT_MAX, level) == UINT_MAX) { // Node not
        //previously visited
            unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
            currFrontier[currFrontierIdx] = neighbor;
        }
    }
}

__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* nodeLevel, unsigned int* \
    prevFrontier, unsigned int* currFrontier, unsigned int numPrevFrontier, unsigned \
    int* numCurrFrontier,  unsigned int level) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPrevFrontier) {
        unsigned int node = prevFrontier[i];
        unsigned int start = csrGraph.nodePtrs[node];
        unsigned int numNeighbors = csrGraph.nodePtrs[node + 1] - start;
        unsigned int numThreadsPerBlock = 1024;
        unsigned int numBlocks = (numNeighbors + numThreadsPerBlock - 1) \
        /numThreadsPerBlock;
        bfs_neighbors_kernel <<< numBlocks, numThreadsPerBlock >>> (csrGraph, \
        nodeLevel, currFrontier, numCurrFrontier, level, start, numNeighbors);
    }
}

void bfs_levels(CSRGraph csrGraph, unsigned int* nodeLevel, unsigned int* prevFrontier, \
unsigned int* currFrontier, unsigned int* numCurrFrontier) {
    gpuTKCheck(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, \
    csrGraph.numNodes));
    unsigned int numPrevFrontier = 1;
    unsigned int numThreadsPerBlock = 256;
    //gpuTKTime_start(GPU, "Allocating GPU memory.");
    for(unsigned int level = 1; numPrevFrontier > 0; ++level) {
        // Visit nodes in previous frontier
        gpuTKCheck(cudaMemset(numCurrFrontier, 0, sizeof(unsigned int)));
        unsigned int numBlocks = (numPrevFrontier + numThreadsPerBlock - 1) \
        /numThreadsPerBlock;
        bfs_kernel <<< numBlocks, numThreadsPerBlock >>> (csrGraph, nodeLevel, \
        prevFrontier, currFrontier, numPrevFrontier, numCurrFrontier, level);

        // Swap buffers
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
        gpuTKCheck(cudaMemcpy(&numPrevFrontier, numCurrFrontier, sizeof(unsigned int), \
        cudaMemcpyDeviceToHost));

    }
    gpuTKCheck(cudaDeviceSynchronize());
}

void bfs_gpu(CSRGraph csrGraph, unsigned int srcNode, unsigned int* nodeLevel) {

    //@@ Allocate GPU memory
    gpuTKTime_start(GPU, "Allocating GPU memory.");
    CSRGraph csrGraph_d;
    csrGraph_d.numNodes = csrGraph.numNodes;
    csrGraph_d.numEdges = csrGraph.numEdges;
    gpuTKCheck(cudaMalloc((void**) &csrGraph_d.nodePtrs, (csrGraph_d.numNodes + 1) \
    *sizeof(unsigned int)));
    gpuTKCheck(cudaMalloc((void**) &csrGraph_d.neighbors, csrGraph_d.numEdges*sizeof \
    (unsigned int)));
    unsigned int* nodeLevel_d;
    gpuTKCheck(cudaMalloc((void**) &nodeLevel_d, csrGraph_d.numNodes*sizeof \
    (unsigned int)));
    unsigned int* buffer1_d;
    gpuTKCheck(cudaMalloc((void**) &buffer1_d, csrGraph_d.numNodes*sizeof( \
    unsigned int)));
    unsigned int* buffer2_d;
    gpuTKCheck(cudaMalloc((void**) &buffer2_d, csrGraph_d.numNodes*sizeof( \
    unsigned int)));
    unsigned int* numCurrFrontier_d;
    gpuTKCheck(cudaMalloc((void**) &numCurrFrontier_d, sizeof(unsigned int)));
    unsigned int* prevFrontier_d = buffer1_d;
    unsigned int* currFrontier_d = buffer2_d;
    gpuTKCheck(cudaDeviceSynchronize());
    gpuTKTime_stop(GPU, "Allocating GPU memory.");

    // Copy data to GPU
    gpuTKTime_start(GPU, "Copying Data to GPU memory.");
    gpuTKCheck(cudaMemcpy(csrGraph_d.nodePtrs, csrGraph.nodePtrs, ( \
    csrGraph_d.numNodes + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(csrGraph_d.neighbors, csrGraph.neighbors, \
    csrGraph_d.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice));
    nodeLevel[srcNode] = 0;
    gpuTKCheck(cudaMemcpy(nodeLevel_d, nodeLevel, csrGraph_d.numNodes*sizeof( \
    unsigned int), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(prevFrontier_d, &srcNode, sizeof(unsigned int), \
    cudaMemcpyHostToDevice));
    gpuTKCheck(cudaDeviceSynchronize());
    gpuTKTime_stop(GPU, "Copying Data to GPU memory.");

    // Call kernel
    gpuTKTime_start(GPU, "BFS Kernel Time");
    bfs_levels(csrGraph_d, nodeLevel_d, prevFrontier_d, currFrontier_d, \
    numCurrFrontier_d);
    gpuTKTime_stop(GPU, "BFS Kernel Time");

    // Copy data from GPU
    gpuTKTime_start(GPU, "Copying data from GPU memory");
    gpuTKCheck(cudaMemcpy(nodeLevel, nodeLevel_d, csrGraph_d.numNodes*sizeof( \
    unsigned int), cudaMemcpyDeviceToHost));
    gpuTKCheck(cudaDeviceSynchronize());
    gpuTKTime_stop(GPU, "Copying data from GPU memory");

    // Free GPU memory
    gpuTKTime_start(GPU, "Free GPU memory");
    cudaFree(csrGraph_d.nodePtrs);
    gpuTKCheck(cudaFree(csrGraph_d.neighbors));
    gpuTKCheck(cudaFree(nodeLevel_d));
    gpuTKCheck(cudaFree(buffer1_d));
    gpuTKCheck(cudaFree(buffer2_d));
    gpuTKCheck(cudaFree(numCurrFrontier_d));
    gpuTKCheck(cudaDeviceSynchronize());
    gpuTKTime_stop(GPU, "Free GPU memory");
}


int main(int argc, char**argv) {
    
    //@@ Wake up device
    gpuTKCheck(cudaDeviceSynchronize());

    const char* fileName = (argc > 1)?argv[1]:"data/loc-gowalla_edges.txt";

    //@@ Allocate memory and read graph in COO and then convert to CSR format. 
    //@@ COO and CSR funcstions are provided for future use. 
    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    COOGraph cooGraph = readCOOGraph(fileName);
    //COOGraph cooGraph = readCOOGraph_old(fileName);
    CSRGraph csrGraph = coo2csr(cooGraph);
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");
    
    gpuTKLog(TRACE, "Number of Nodes is ", csrGraph.numNodes);
    gpuTKLog(TRACE, "Number of Edges is ", csrGraph.numEdges);


    //@@ Allocate host output memory 
    unsigned int* nodeLevel_gpu = (unsigned int*) malloc(csrGraph.numNodes*sizeof( \
    unsigned int));
    
    //@@ Declare source node 
    unsigned int srcNode = 0;
    //@@ Initialize outputs
    for(unsigned int i = 0; i < csrGraph.numNodes; ++i) {
        nodeLevel_gpu[i] = UINT_MAX;
    }

    //@@ Call function that perform BFS GPU Computation
    bfs_gpu(csrGraph, srcNode, nodeLevel_gpu);

    //@@ Compare with the solution.
    compare(csrGraph, srcNode, nodeLevel_gpu); 

    //@@ Free memory
    freeCOOGraph(cooGraph);
    freeCSRGraph(csrGraph);
    free(nodeLevel_gpu);

    return 0;

}

