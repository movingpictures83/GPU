#ifndef __COMPARE_H__
#define __COMPARE_H__

#include "graph.h"

void bfs_cpu(CSRGraph csrGraph, unsigned int srcNode, unsigned int* nodeLevel) {

    // Initialize frontier double buffers
    unsigned int* buffer1 = (unsigned int*) malloc(csrGraph.numNodes*sizeof(unsigned int));
    unsigned int* buffer2 = (unsigned int*) malloc(csrGraph.numNodes*sizeof(unsigned int));
    unsigned int* prevFrontier = buffer1;
    unsigned int* currFrontier = buffer2;

    // Run BFS
    nodeLevel[srcNode] = 0;
    prevFrontier[0] = srcNode;
    unsigned int numPrevFrontier = 1;
    for(unsigned int level = 1; numPrevFrontier > 0; ++level) { 

        // Visit nodes in the previous frontier
        unsigned int numCurrFrontier = 0;
        for(unsigned int i = 0; i < numPrevFrontier; ++i) {
            unsigned int node = prevFrontier[i];
            if(nodeLevel[node] == level - 1) {
                for(unsigned int edge = csrGraph.nodePtrs[node]; edge < csrGraph.nodePtrs[node + 1]; ++edge) {
                    unsigned int neighbor = csrGraph.neighbors[edge];
                    if(nodeLevel[neighbor] == UINT_MAX) { // Node not previously visited
                        nodeLevel[neighbor] = level;
                        currFrontier[numCurrFrontier] = neighbor;
                        ++numCurrFrontier;
                    }
                }
            }
        }

        // Swap buffers
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
        numPrevFrontier = numCurrFrontier;

    }

    // Free double buffers
    free(buffer1);
    free(buffer2);

}


void compare(CSRGraph csrGraph, unsigned int srcNode, unsigned int* nodeLevel_gpu){

    unsigned int* nodeLevel_cpu = (unsigned int*) malloc(csrGraph.numNodes*sizeof(unsigned int));
    for(unsigned int i = 0; i < csrGraph.numNodes; ++i) {
        nodeLevel_cpu[i] = UINT_MAX;
    }

    // Run CPU implementation
    bfs_cpu(csrGraph, srcNode, nodeLevel_cpu);
        
    // Verify result
    for(unsigned int i = 0; i < csrGraph.numNodes; ++i) {
        if(nodeLevel_cpu[i] != nodeLevel_gpu[i]) {
            printf("Mismatch detected at node %u (CPU result = %u, GPU result = %u)\n", i, nodeLevel_cpu[i], nodeLevel_gpu[i]);
            exit(0);
        }
    }
    printf("Solution is correct\n");
        
        
    free(nodeLevel_cpu);
}

#endif //__COMPARE_H__
