---
title: CUDA Dynamic Parallelism on BFS Kernel
author: GPU Teaching Kit -- Accelerated Computing
module: 23
---

# Objective

The purpose of this lab is to introduce the student to the Dynamic Parallelism using BFS kernel. The student will implement BFS kernel level and then convert to Dynamic Parallelism implementation. Interested students can attempt to optimize the Dynamic Parallelism for performance as discussed in the slides. 

# Prerequisites

Before starting this lab, make sure that:

- You have completed all modules before "Dynamic Parallelism"
- You have completed the lecture videos of Dynamic Parallelism
- You have played with BFS example provided in Module 11. 

# Instructions

Edit the code in the code tab to perform the following:

- Implement BFS kernel without using Dynamic Parallelism 
- Implement BFS kernel with Dynamic Parallelism
- Implement host side code for calling the BFS Kernel

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```
./DynamicParallelism_DatasetGenerator    #this will download graphs from /
  SNAP website. If you are WINDOWS uses, you need to manually download the graphs /
  provided in dataset_generator.cpp file.  
./DynamicParallelism_Template graphs/<anygraph.txt>
```

where `<anygraph.txt>` is the real world graph download from the SNAP website. 
