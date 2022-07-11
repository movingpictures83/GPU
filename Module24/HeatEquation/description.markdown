---
title: Multi-GPU Heat Equation
author: GPU Teaching Kit -- Accelerated Computing
module: 24
---

# Objective

Implement a Multi-GPU version of an explicit finite difference recurrence relation for the Heat equation in two dimensions.

# Prerequisites

Before starting this lab, make sure that:

- You have completed "Stencil" Lab from Module 8
- You have completed the required module lectures, in particular lecture 24.5 Heat Equation

# Instructions

Edit the code in the code tab to perform the following:

- allocate host memory
- set the appropiate device to use
- allocate device memory in several devices
- copy memory to multiple devices, to the host and between devices
- create cuda streams
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- deallocate host and device memory
- implement the finite difference recurrence relation for the Heat equation routine

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](https://bitbucket.org/hwuligans/gputeachingkit-labs/src). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](https://bitbucket.org/hwuligans/gputeachingkit-labs/src/master/README.md) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./HeatEquation\_Template -i <dev num>,<grid dim X>,<grid dim Y>,<num steps>
```

where `<dev num>` is the number of devices to use, `<grid dim X>,<grid dim Y>` the dimensions of the grid and `<num steps>` the number of time steps to perform. 

To verify the correctness of your solution for the lab use:

```{.bash}
# Generate the expected solution:
./HeatEquation\_Solution -i 1,<grid dim X>,<grid dim Y>,<num steps> -o heat_solution.raw
# Compare and verify your solution: 
./HeatEquation\_Template -i <dev num>,<grid dim X>,<grid dim Y>,<num steps> \
                         -e heat_solution.raw
```

In order to the solution to be correct the numerical error should be close to zero.
