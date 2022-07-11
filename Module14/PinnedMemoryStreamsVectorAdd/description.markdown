---
title: Pinned Memory Vector Add
author: GPU Teaching Kit -- Accelerated Computing
module: 14
---

# Objective

The purpose of this lab is to introduce the student to the Pinned Memory APIs along with CUDA Streams by implementing vector addition. The student will implement pinned memory allocation and experience the usage of CUDA Streams by writing the CUDA Calls in the host code.

# Prerequisites

Before starting this lab, make sure that:

- You have completed CUDA Streams, Unified Memory and Pinned Memory Modules in the teaching kit
- You have completed the "Device Query" and the "VectorAdd" lab

# Instructions

The lab is broken down into two parts. Part A covers cudaStreams while part B coverts Pinned memory on vectorAdd. To controll if you want to work on part A or part B, use the `#define PINNED 1` in template.cu. By default we will cover the cudaStream lab. 

Part A lab setup. 

Edit the code in the code tab to perform the following:

- Setup and Initialize CudaStreams using cudaStreamCreate.
- Find the segSize each stream should transfer asynchronously. 
- In the for loop, generate start and end indexes of the transfer and invoke asynchronous copy. 
- Declare number of threads and blocks.
- Invoke CUDA kernel
- Copy back the computed data to host asynchronously. 
- Synchornize out the for loop
- Free device memory

Part B lab setup.

- Insert `#define PINNED 1` 
- Call pinned memory allocators. 
- Initialize grid and block dimnensions 
- Call the cuda kernel
- Synchornize 
- Free device memory

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```
./PinnedMemoryStreamVectorAdd_Template -e <expected.raw> -i <intput1.raw>, \
  <input2.raw> -o <output.raw> -t vector
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.
