---
title: Heat Equation with NVIDIA libraries
author: GPU Teaching Kit -- Accelerated Computing
module: 25
---

# Objective

Implement an explicit finite difference iterative method for the Heat Equation in a one dimensional metal rod using cuSPARSE, cuBLAS and Unified Memory.

# Prerequisites

Before starting this lab, make sure that:

- You have completed the module lectures of modules 2 and 14 on Unified Memory
- You have completed the "Heat Equation" Lab from Module 24
- You have completed the module lectures of module 25
- You have read about the [Compressed Storage Row (CSR)](https://docs.nvidia.com/cuda/cusparse/index.html#csr-format) sparse matrix format
- You have read the introduction of the [cuSPARSE documentation](https://docs.nvidia.com/cuda/cusparse/index.html)
- You have read about the functions to initialize ([cusparseCreate()](https://docs.nvidia.com/cuda/cusparse/index.html#cusparseCreate)) and destroy ([cusparseDestroy()](https://docs.nvidia.com/cuda/cusparse/index.html#cusparseDestroy)) the cuSPARSE environment
- You have read about the struct [cusparseMatDescr_t](https://docs.nvidia.com/cuda/cusparse/index.html#cusparseMatDescr_t) and how to initialize and destroy it with [cusparseCreateMatDescr()](https://docs.nvidia.com/cuda/cusparse/index.html#cusparseCreateMatDescr), [cusparseDestroyMatDescr()](https://docs.nvidia.com/cuda/cusparse/index.html#cusparseDestroyMatDescr) respectively
- You have read about the routine performing Sparse Matrix Per Vector (SMPV) [cusparseCsrmvEx()](https://docs.nvidia.com/cuda/cusparse/index.html#csrmvEx) and the function to get the size of the buffer needed by the routine [cusparseCsrmvEx_bufferSize()](https://docs.nvidia.com/cuda/cusparse/index.html#csrmvEx)
- You have read about the cuBLAS axpy routine [cublasDaxpy()](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy)
- You have read about the cuBLAS norm routine [cublasDnrm2()](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-nrm2)

# Instructions

Edit the code in the code tab to perform the following:
- Allocate unified memory needed by the sparse matrix, the temperature array and a temporal array.
- Prefetch data to the appropriate locations.
- Add the necessary cuBLAS, cuSPARSE API functions to create and destroy the environments.
- Add the necessary cuBLAS, cuSPARSE routines to perform the algorithm.
- Add the necessary cuBLAS routines to perform the relative error.
- Deallocate memory.
- Answer the questions found in the questions tab.

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Problem description & Algorithm
In this lab the idea is to model a metal rod with constant heat sources at the boundaries and simulate what is happening in the interior of the rod after a long period of time. The heat equation in 1D simulates this problem and is given by the differential equation:

$$ \dfrac{\partial T}{\partial t} = \alpha \dfrac{\partial^2 T}{\partial x^2} $$

Using the Finite Differences Method the Heat Equation can be discretized into:

$$ T^{n+1}_{i} = T^{n}_{i} + \alpha (T^{n}_{i-1} - 2 T^{n}_{i}) + T^{n}_{i+1} $$

Given the boundary conditions and the discretization we obtain the algorithm:

```{.ruby}
for i from 1 to nsteps:
  tmp = A * temp
  temp = alpha * tmp + temp
  norm = vectorNorm(tmp)
  if tmp < 1e-4 :
    break
end
```
where nsteps is the number of time steps to perform and the sparse matrix A of dimX rows and dimX columns is given by the tridiagonal matrix:

$$ \begin{pmatrix} 0 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0  \\
1 & -2 & 1 & 0 & \cdots &  0 & 0 & 0 & 0 \\
0 & 1 & -2 & 1 & \cdots &  0 & 0 & 0 & 0 \\
0 & 0 & 1 & -2 & \cdots &  0 & 0 & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \cdots &  \vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & 1 & -2 & 1 & 0  \\
0 & 0 & 0 & 0 & \cdots & 0 & 1 & -2 & 1  \\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0
\end{pmatrix} $$

In addition the lab requires the student to compute the relative error, which is given by:

$$ error = \dfrac{\Vert temp - exact \Vert}{\Vert exact \Vert}, $$

where temp is the computed approximation and exact is the exact solution of the problem. The exact solution is already given in the template code.

## Remark

The algorithm needs a large number of iterations to reach steady state, for example for 128 grid points the algorithm requires around 35 000 nsteps.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](https://bitbucket.org/hwuligans/gputeachingkit-labs/src). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](https://bitbucket.org/hwuligans/gputeachingkit-labs/src/master/README.md) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./HeatEquation\_Template -i <grid dim X>,<num steps>
```

where <grid dim X> is the dimension of the metal rod and <num steps> is the number of time steps to perform.
