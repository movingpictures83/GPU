---
title: Reduction - Thrust
author: GPU Teaching Kit -- Accelerated Computing
module: 9
---

# Objective

Use the Thrust template library to perform a reduction of a 1D list of 32-bit floats. The reduction should give the sum of the list. Your code should be able to handle lists of arbitrary length, but for brevity assume that all data will fit within the GPU global memory.

# Prerequisites

Before starting this lab, make sure that:

- You have completed week 4 lecture videos

# Instruction

Edit the code in the code tab to perform the following:

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./ThrustReduction_Template -e <expected.raw> \
  -i <input.raw> -o <output.raw> -t integral_vector
```

where `<expected.raw>` is the expected output, `<input.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.