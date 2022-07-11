---
title: CUDA Unified Memory Matrix Multiplication
author: GPU Teaching Kit -- Accelerated Computing
module: 14
---

# Objective

Implement a tiled dense matrix multiplication routine while using efficient Unified Memory practices.

# Prerequisites

Before starting this lab, make sure that:

- You have completed "Tiled Matrix Multiplication" Lab
- You have completed the required module lectures

# Instructions

Edit the code in the code tab to perform the following:

- allocate unified memory memory
- prefecth memory to the device and to the host
- set memory advices to the unified memory driver
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- deallocate unified memory memory
- implement the matrix-matrix multiplication routine

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./UMMatrixMultiplication\_Template -i <A rows>,<A cols>,<B cols>
```

where `<A rows>,<A cols>` is the number of rows and columns for the matrix `A` and `<B cols>`is the number of columns for the matrix `B`.
