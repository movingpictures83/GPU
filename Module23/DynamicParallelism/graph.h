#ifndef __GRAPH_H__
#define __GRAPH_H__
// This will only have effect on Windows with MSVC
#ifdef _MSC_VER
    #define _CRT_SECURE_NO_WARNINGS 1
    #define restrict __restrict
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#define EDGE_BLOCK_SIZE 4194304 


struct CSRGraph {
    unsigned int numNodes;
    unsigned int numEdges;
    unsigned int* nodePtrs;
    unsigned int* neighbors;
};

struct COOGraph {
    unsigned int numNodes;
    unsigned int numEdges;
    unsigned int* nodes;
    unsigned int* neighbors;
};

struct COOGraph readCOOGraph_old(const char* fileName) {

    struct COOGraph cooGraph;
    // Initialize fields
    FILE* fp = fopen(fileName, "r");
    int x = 1;
    x |= fscanf(fp, "%u", &cooGraph.numNodes);
    x |= fscanf(fp, "%u", &cooGraph.numEdges);
    cooGraph.nodes = (unsigned int*) malloc(cooGraph.numEdges*sizeof(unsigned int));
    cooGraph.neighbors = (unsigned int*) malloc(cooGraph.numEdges*sizeof(unsigned int));

    // Read the nonzeros
    for(unsigned int i = 0; i < cooGraph.numEdges; ++i) {
        x |= fscanf(fp, "%u", &cooGraph.nodes[i]);
        x |= fscanf(fp, "%u", &cooGraph.neighbors[i]);
    }

    return cooGraph;

}


/*
Taken and modified from: https://solarianprogrammer.com/2019/04/03/c-programming-read-file-lines-fgets-getline-implement-portable-getline/ 
POSIX getline replacement for non-POSIX systems (like Windows)
Differences:
    - the function returns int64_t instead of ssize_t
    - does not accept NUL characters in the input file
Warnings:
    - the function sets EINVAL, ENOMEM, EOVERFLOW in case of errors. The above are not defined by ISO C17,
    but are supported by other C compilers like MSVC
*/
int64_t modgetline(char **line, 
			size_t *len, 
			FILE *fp){
    // Check if either line, len or fp are NULL pointers
    if(line == NULL || len == NULL || fp == NULL) {
        errno = EINVAL;
        return -1;
    }
    
    // Use a chunk array of 128 bytes as parameter for fgets
    char chunk[128];

    // Allocate a block of memory for *line if it is NULL or smaller than the chunk array
    if(*line == NULL || *len < sizeof(chunk)) {
        *len = sizeof(chunk);
        if((*line = (char*) malloc(*len)) == NULL) {
            errno = ENOMEM;
            return -1;
        }
    }

    // "Empty" the string
    (*line)[0] = '\0';

    while(fgets(chunk, sizeof(chunk), fp) != NULL) {
        // Resize the line buffer if necessary
        size_t len_used = strlen(*line);
        size_t chunk_used = strlen(chunk);

        if(*len - len_used < chunk_used) {
            // Check for overflow
            if(*len > SIZE_MAX / 2) {
                errno = EOVERFLOW;
                return -1;
            } else {
                *len *= 2;
            }
            
            if((*line = (char*) realloc(*line, *len)) == NULL) {
                errno = ENOMEM;
                return -1;
            }
        }

        // Copy the chunk to the end of the line buffer
        memcpy(*line + len_used, chunk, chunk_used);
        len_used += chunk_used;
        (*line)[len_used] = '\0';

        // Check if *line contains '\n', if yes, return the *line length
        if((*line)[len_used - 1] == '\n') {
            return len_used;
        }
    }

    return -1;
}


struct COOGraph readCOOGraph(const char* fileName){

    struct COOGraph cooGraph;

    FILE* fp = fopen(fileName, "r"); 
    unsigned int largest = 0; 
    unsigned int linecnt = 0; 
    unsigned int tmp1=0;
    unsigned int tmp2=0; 

    //We dont know how much memory we need to allocate. So allocate in block size.
    //This limitation is coming because of C99 standard. Chaging to C++ would address this issue. 
    cooGraph.nodes = (unsigned int*) malloc(EDGE_BLOCK_SIZE*sizeof(unsigned int));
    cooGraph.neighbors = (unsigned int*) malloc(EDGE_BLOCK_SIZE*sizeof(unsigned int));

    unsigned int max_edge_index = EDGE_BLOCK_SIZE - 1;
    int err =1;
 
	char *line = NULL; 
	size_t line_size = 0; 
	fpos_t prev_pos; 
    fgetpos(fp, &prev_pos); 
	while(modgetline(&line, &line_size, fp) != -1){
	    if(line[0] != '#'){
			fsetpos(fp, &prev_pos); 
			break; 	
		} else { 
			fgetpos(fp, &prev_pos); 
		}
	}	
 
    while(!feof(fp)){
       err = fscanf(fp, "%u\t%u", &tmp1, &tmp2); 
       cooGraph.nodes[linecnt]     = tmp1;
       cooGraph.neighbors[linecnt] = tmp2;
    //   printf("lineno: %d, %d %d\n", linecnt, tmp1, tmp2);
       linecnt ++;
       
       if(linecnt > max_edge_index){
               cooGraph.nodes = (unsigned int*) realloc(cooGraph.nodes, ((max_edge_index+EDGE_BLOCK_SIZE)*sizeof(unsigned int)));
               cooGraph.neighbors = (unsigned int*) realloc(cooGraph.neighbors, ((max_edge_index+EDGE_BLOCK_SIZE)*sizeof(unsigned int)));
               if((cooGraph.neighbors == NULL) || (cooGraph.nodes == NULL)){
                       printf("Insufficient Memory\n");
                       exit(0); 
               }
               max_edge_index += EDGE_BLOCK_SIZE; 
       }
       
       if(tmp1 > largest)
               largest = tmp1;
       if(tmp2 >largest)
               largest = tmp2;
    }

    cooGraph.numEdges = linecnt - 1; 
    cooGraph.numNodes = largest + 1; 
    
    cooGraph.nodes = (unsigned int*) realloc(cooGraph.nodes, (cooGraph.numEdges)*sizeof(unsigned int));
    cooGraph.neighbors = (unsigned int*) realloc(cooGraph.neighbors, (cooGraph.numEdges)*sizeof(unsigned int));
        
    return cooGraph;        
}


struct CSRGraph coo2csr(struct COOGraph cooGraph) {

    struct CSRGraph csrGraph;

    // Initialize fields
    csrGraph.numNodes = cooGraph.numNodes;
    csrGraph.numEdges = cooGraph.numEdges;
    csrGraph.nodePtrs = (unsigned int*) malloc((csrGraph.numNodes + 1)*sizeof(unsigned int));
    csrGraph.neighbors = (unsigned int*) malloc(csrGraph.numEdges*sizeof(unsigned int));

    // Histogram nodes
    memset(csrGraph.nodePtrs, 0, (csrGraph.numNodes + 1)*sizeof(unsigned int));
    for(unsigned int i = 0; i < cooGraph.numEdges; ++i) {
        unsigned int node = cooGraph.nodes[i];
        csrGraph.nodePtrs[node]++;
    }

    // Prefix sum nodePtrs
    unsigned int sumBeforeNextRow = 0;
    for(unsigned int node = 0; node < csrGraph.numNodes; ++node) {
        unsigned int sumBeforeRow = sumBeforeNextRow;
        sumBeforeNextRow += csrGraph.nodePtrs[node];
        csrGraph.nodePtrs[node] = sumBeforeRow;
    }
    csrGraph.nodePtrs[csrGraph.numNodes] = sumBeforeNextRow;

    // Bin the nonzeros
    for(unsigned int i = 0; i < cooGraph.numEdges; ++i) {
        unsigned int node = cooGraph.nodes[i];
        unsigned int edgeIdx = csrGraph.nodePtrs[node]++;
        csrGraph.neighbors[edgeIdx] = cooGraph.neighbors[i];
    }

    // Restore nodePtrs
    for(unsigned int node = csrGraph.numNodes - 1; node > 0; --node) {
        csrGraph.nodePtrs[node] = csrGraph.nodePtrs[node - 1];
    }
    csrGraph.nodePtrs[0] = 0;
    
    //printf("NodePtr\n");
    //for(int i=0; i< csrGraph.numNodes+1; i++)
    //        printf("%d  ",csrGraph.nodePtrs[i]);
    //printf("\nEdges\n");
    //for(int i =0; i<csrGraph.numEdges; i++)
    //        printf("%d  ", csrGraph.neighbors[i]);
    //printf("\n");

    return csrGraph;

}
void freeCOOGraph(struct COOGraph cooGraph) {
    free(cooGraph.nodes);
    free(cooGraph.neighbors);
}

void freeCSRGraph(struct CSRGraph csrGraph) {
    free(csrGraph.nodePtrs);
    free(csrGraph.neighbors);
}


#endif // __GRAPH_H__
