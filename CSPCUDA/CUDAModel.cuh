#pragma once
// System includes
#include <stdlib.h>
#include <stdio.h>

//Project includes
#include "XMLModel.h"
//CUDA runtimes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>

//typedef __host__ __device__ struct CUDADomain
//{
//	int id;
//	int size;
//	thrust::device_vector<int> values;
//};

//typedef __host__ __device__ struct CUDAModel
//{
//	thrust::device_vector<CUDADomain> domains;
//};
//»¡
typedef int4 arc;
//µã
typedef int2 node;
//Óò¸öÊý
//typedef ushort2 var_size;

