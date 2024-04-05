#include "tricount.h"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cub/cub.cuh>
#include <cstring>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <fstream>
#include <random>
#include <chrono>
#include <pthread.h>
#include <thread>
using namespace std;


#ifdef DEBUG_DF

TimeInterval kernelTime;
TimeInterval subTime;
TimeInterval totalTime;

TimeInterval allTime;
TimeInterval preProcessTime;
TimeInterval tmpTime;
#endif





#define BOUND 128

#define WARP_SIZE 32
#define BLOCK_SHD_SIZE 4102
#define BLOCK_BLOCK_NUM 1024
#define BLOCK_BLOCK_SIZE 1024
#define WARP_BLOCK_NUM 1024
#define WARP_BLOCK_SIZE 256

#define HASH_SIZE_large 1021
#define HASH_SIZE_small 511

#define warp_size   BOUND
#define large_size  BOUND


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major){
    case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
    case 3: // Kepler
      cores = mp * 192;
      break;
    case 5: // Maxwell
      cores = mp * 128;
      break;
    case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}

void initGPU(const Offset_type edge_num, const Vertex_type N, int gpuid){
    // cudaDeviceProp deviceProp;
    // gpuErrchk( cudaGetDeviceProperties(&deviceProp, 0) );

    // int deviceCount = 0;
    // gpuErrchk(cudaGetDeviceCount(&deviceCount));
    // printf("Found %d devices\n", deviceCount);
    // // Another way to get the # of cores: #include <helper_cuda.h> in this link:
    // // https://github.com/NVIDIA/cuda-samples/blob/6be514679b201c8a0f0cda050bc7c01c8cda32ec/Common/helper_cuda.h
    // //int CUDACores = _ConvertSMVer2Cores(props.major, props.minor) * props.multiProcessorCount;
    // for (int device = 0; device < deviceCount; device++) {
    //   cudaDeviceProp prop;
    //   gpuErrchk(cudaSetDevice(device));
    //   gpuErrchk(cudaGetDeviceProperties(&prop, device));
    //   printf("  Device[%d]: %s\n", device, prop.name);
    //   if (device == 0) {
    //     printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    //     printf("  Warp size: %d\n", prop.warpSize);
    //     printf("  Total # SM: %d\n", prop.multiProcessorCount);
    //     printf("  Total # CUDA cores: %d\n", getSPcores(prop));
    //     printf("  Total amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    //     printf("  Total amount of shared memory per block: %lu uint32\n", prop.sharedMemPerBlock/4);
    //     printf("  Total # registers per block: %d\n", prop.regsPerBlock);
    //     printf("  Total amount of constant memory: %lu bytes\n", prop.totalConstMem);
    //     printf("  Total global memory: %.1f GB\n", float(prop.totalGlobalMem)/float(1024*1024*1024));
    //     printf("  Memory Clock Rate: %.2f GHz\n", float(prop.memoryClockRate)/float(1024*1024));
    //     printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    //     printf("  Memory Bus Width: %d bytes\n", prop.memoryBusWidth/8);
    //     //printf("  Maximum memory pitch: %u\n", prop.memPitch);
    //     printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    //   }
    // }

    gpuErrchk(cudaSetDevice(gpuid));
    // cout << "cudaSetDevice "<<gpuid<<" \n";
}

inline void enableP2P(int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        for (int j = 0; j < ngpus; j++)
        {
            if (i == j)
                continue;
            
            int peer_access_available = 0;
 
            cudaDeviceCanAccessPeer(&peer_access_available, i, j);
 
            if (peer_access_available)
            {
                cudaDeviceEnablePeerAccess(j, i);
                printf(" > GP%d enbled direct access to GPU%d\n", i, j);
            }
            else
                printf("(%d, %d)\n", i, j);
        }
    }
}

__global__ void __launch_bounds__(WARP_BLOCK_SIZE)
StepTC_block_vertex_hash(
  Vertex_type min_, Vertex_type max_, 
  Vertex_type bucket_num, Vertex_type bucket_size, Vertex_type* hash_bucket, 
  Vertex_type vertex_len_less_128, Vertex_type *vertex_less_128,
  Offset_type* dev_block_vertex_offset, Vertex_type* dev_block_vertex_value, 
  Offset_type* beg_pos, Vertex_type* adj_list,Vertex_type device_i, Vertex_type all_device, unsigned long long *total, unsigned int *dev_nowNode)//, unsigned long long dev_sum,unsigned int dev_nowNode)
{
  
  __shared__ Vertex_type bucket_count[HASH_SIZE_large+5];
  extern __shared__ Vertex_type shared_bucket[];

  Vertex_type * hash_bucket_cta;
  bucket_num = HASH_SIZE_large;
  hash_bucket_cta = hash_bucket + blockIdx.x * bucket_num * bucket_size;
	unsigned long long __shared__ G_counter;
	if (threadIdx.x==0) G_counter=0;

	unsigned long long P_counter=0;

  Vertex_type now_vertex_idx;
  __shared__ Vertex_type ver;

  if (threadIdx.x==0)
  {
    ver= atomicAdd(dev_nowNode, all_device);
  }
  __syncthreads();
  now_vertex_idx=ver;
  __syncthreads();

	while (now_vertex_idx < vertex_len_less_128)
	{

		Vertex_type now_vertex_value = vertex_less_128[now_vertex_idx];
		Offset_type degree=beg_pos[now_vertex_value+1]-beg_pos[now_vertex_value];
    if(degree > min_ && degree <= max_)
    {
      Offset_type start=beg_pos[now_vertex_value];
      Offset_type end  =beg_pos[now_vertex_value+1];
      Offset_type now  =start + threadIdx.x;

      for (Vertex_type i=threadIdx.x;i<bucket_num;i+=blockDim.x)
        bucket_count[i]=0;
      __syncthreads();

      while(now<end)
      {
        Vertex_type target= adj_list[now];
        Vertex_type bin=target % bucket_num; 
        Vertex_type index=atomicAdd(&bucket_count[bin], 1);
        hash_bucket_cta[bin*bucket_size + index]=target;
        now+=blockDim.x;
      }
      __syncthreads();

      Vertex_type thid = threadIdx.x;
      Vertex_type bdim = blockDim.x;
      Vertex_type pout = 0, pin = 1;

      for (Vertex_type i = thid; i < bucket_num; i += bdim) {
          shared_bucket[i] = (i > 0) ? bucket_count[i - 1] : 0;
      }
      __syncthreads();

      for (Vertex_type offset = 1; offset < bucket_num; offset *= 2) {
          pout = 1 - pout;
          pin = 1 - pout;
          for (Vertex_type i = thid; i < bucket_num; i += bdim) {
              if (i >= offset) {
                  shared_bucket[pout * bucket_num + i] = shared_bucket[pin * bucket_num + i - offset] + shared_bucket[pin * bucket_num + i];
              } else {
                  shared_bucket[pout * bucket_num + i] = shared_bucket[pin * bucket_num + i];
              }
          }
          __syncthreads();
      }

      __syncthreads();

      for (Vertex_type i = thid; i < bucket_num; i += bdim) {
          bucket_count[i] = (bucket_count[i] & 0xFFFF) | ((shared_bucket[pout * bucket_num + i] & 0xFFFF) << 16);
      }

      for (Vertex_type i = threadIdx.x; i < bucket_num; i += bdim) {
          Vertex_type i_beg = bucket_count[i] >> 16;
          Vertex_type i_num = bucket_count[i] & 0xFFFF;
          for (Vertex_type index = 0; index < i_num; index ++){
            shared_bucket[i_beg + index] = hash_bucket_cta[i*bucket_size + index];
          }
      }

      __syncthreads();

      now=dev_block_vertex_offset[now_vertex_idx];
      end=dev_block_vertex_offset[now_vertex_idx+1];

      Vertex_type LargeWarp = 32;
      Vertex_type oneTimenbr = blockDim.x / LargeWarp;
      Vertex_type superwarp_ID=threadIdx.x/LargeWarp;
      Vertex_type superwarp_TID=threadIdx.x%LargeWarp;
      Vertex_type workid=superwarp_TID;
      now=now+superwarp_ID;
      
      Vertex_type neighbor;
      Offset_type neighbor_start;
      Offset_type neighbor_degree;
      if(now<end) {
        neighbor=dev_block_vertex_value[now];
        neighbor_start=beg_pos[neighbor];
        neighbor_degree=beg_pos[neighbor+1]-neighbor_start;
      }

      while (now<end)
      {
        while (now<end && workid>=neighbor_degree)
        {
          now+=oneTimenbr;
          workid-=neighbor_degree; 
          neighbor=dev_block_vertex_value[now]; 
          neighbor_start=beg_pos[neighbor]; 
          neighbor_degree=beg_pos[neighbor+1]-neighbor_start; 
        }
        if (now<end) 
        {
          Vertex_type target=adj_list[neighbor_start+workid];
          Vertex_type bin=target % bucket_num; 

          Vertex_type now_bucket_idx = bucket_count[bin] >> 16;
          Vertex_type now_len = 0;
          Vertex_type all_len = bucket_count[bin] & 0xFFFF;

          while(now_len < all_len)
          {
            if(shared_bucket[now_bucket_idx]==target)
            {
              P_counter++;
              now_len = all_len;
            }
            now_bucket_idx++;
            now_len++;
          }

        }
        workid+=LargeWarp;
      }
    }

		__syncthreads();
    if (threadIdx.x==0)
    {
      ver= atomicAdd(dev_nowNode, all_device);
    }
    __syncthreads();
    now_vertex_idx=ver;
	}

	atomicAdd(&G_counter,P_counter);
	__syncthreads();
	if(threadIdx.x==0) atomicAdd(total, G_counter);
}



__global__ void __launch_bounds__(WARP_BLOCK_SIZE)
StepTC_block_vertex_hash_1024(
  Vertex_type min_, Vertex_type max_, 
  Vertex_type bucket_num, Vertex_type bucket_size, Vertex_type* hash_bucket, 
  Vertex_type vertex_len_less_128, Vertex_type *vertex_less_128,
  Offset_type* dev_block_vertex_offset, Vertex_type* dev_block_vertex_value, 
  Offset_type* beg_pos, Vertex_type* adj_list,Vertex_type device_i, Vertex_type all_device, unsigned long long *total, unsigned int *dev_nowNode)//, unsigned long long dev_sum,unsigned int dev_nowNode)
{

  __shared__ Vertex_type bucket_count[HASH_SIZE_small+5];
  extern __shared__ Vertex_type shared_bucket[];

  Vertex_type * hash_bucket_cta;
  bucket_num = HASH_SIZE_small;
  hash_bucket_cta = hash_bucket + blockIdx.x * bucket_num * bucket_size;
	unsigned long long __shared__ G_counter;
	if (threadIdx.x==0) G_counter=0;

	unsigned long long P_counter=0;

  Vertex_type now_vertex_idx;
  __shared__ Vertex_type ver;

  if (threadIdx.x==0)
  {
    ver= atomicAdd(dev_nowNode, all_device);
  }
  __syncthreads();
  now_vertex_idx=ver;
  __syncthreads();

	while (now_vertex_idx < vertex_len_less_128)
	{
    
		Vertex_type now_vertex_value = vertex_less_128[now_vertex_idx];
		Offset_type degree=beg_pos[now_vertex_value+1]-beg_pos[now_vertex_value];
    if(degree > min_ && degree <= max_)
    {
      Offset_type start=beg_pos[now_vertex_value];
      Offset_type end  =beg_pos[now_vertex_value+1];
      Offset_type now  =start + threadIdx.x;

      for (Vertex_type i=threadIdx.x;i<bucket_num;i+=blockDim.x)
        bucket_count[i]=0;
      __syncthreads();

      while(now<end)
      {
        
        Vertex_type target= adj_list[now];
        Vertex_type bin=target % bucket_num;
        Vertex_type index=atomicAdd(&bucket_count[bin], 1);
        hash_bucket_cta[bin*bucket_size + index]=target;
        now+=blockDim.x;
      }
      __syncthreads();

      Vertex_type thid = threadIdx.x;
      Vertex_type bdim = blockDim.x;
      Vertex_type pout = 0, pin = 1;

      for (Vertex_type i = thid; i < bucket_num; i += bdim) {
          shared_bucket[i] = (i > 0) ? bucket_count[i - 1] : 0;
      }
      __syncthreads();

      for (Vertex_type offset = 1; offset < bucket_num; offset *= 2) {
          pout = 1 - pout;
          pin = 1 - pout;
          for (Vertex_type i = thid; i < bucket_num; i += bdim) {
              if (i >= offset) {
                  shared_bucket[pout * bucket_num + i] = shared_bucket[pin * bucket_num + i - offset] + shared_bucket[pin * bucket_num + i];
              } else {
                  shared_bucket[pout * bucket_num + i] = shared_bucket[pin * bucket_num + i];
              }
          }
          __syncthreads();
      }

      __syncthreads();

      for (Vertex_type i = thid; i < bucket_num; i += bdim) {
          bucket_count[i] = (bucket_count[i] & 0xFFFF) | ((shared_bucket[pout * bucket_num + i] & 0xFFFF) << 16);
      }

      for (Vertex_type i = threadIdx.x; i < bucket_num; i += bdim) {
          Vertex_type i_beg = bucket_count[i] >> 16;
          Vertex_type i_num = bucket_count[i] & 0xFFFF;
          for (Vertex_type index = 0; index < i_num; index ++){
            shared_bucket[i_beg + index] = hash_bucket_cta[i*bucket_size + index];
          }
      }

      __syncthreads();

      now=dev_block_vertex_offset[now_vertex_idx];
      end=dev_block_vertex_offset[now_vertex_idx+1];

      Vertex_type LargeWarp = 32;
      Vertex_type oneTimenbr = blockDim.x / LargeWarp;
      Vertex_type superwarp_ID=threadIdx.x/LargeWarp;
      Vertex_type superwarp_TID=threadIdx.x%LargeWarp;
      Vertex_type workid=superwarp_TID;
      now=now+superwarp_ID;
      
      Vertex_type neighbor;
      Offset_type neighbor_start;
      Offset_type neighbor_degree;
      if(now<end) {
        neighbor=dev_block_vertex_value[now];
        neighbor_start=beg_pos[neighbor];
        neighbor_degree=beg_pos[neighbor+1]-neighbor_start;
      }

      while (now<end)
      {
        while (now<end && workid>=neighbor_degree)
        {
          now+=oneTimenbr;
          workid-=neighbor_degree; 
          neighbor=dev_block_vertex_value[now]; 
          neighbor_start=beg_pos[neighbor]; 
          neighbor_degree=beg_pos[neighbor+1]-neighbor_start; 
        }

        if (now<end) 
        {
          Vertex_type target=adj_list[neighbor_start+workid];
          Vertex_type bin=target % bucket_num; //& MOD;


          Vertex_type now_bucket_idx = bucket_count[bin] >> 16;
          Vertex_type now_len = 0;
          Vertex_type all_len = bucket_count[bin] & 0xFFFF;

          while(now_len < all_len)
          {
            if(shared_bucket[now_bucket_idx]==target)
            {
              P_counter++;
              now_len = all_len;
            }
            now_bucket_idx++;
            now_len++;
          }
        }
        workid+=LargeWarp;
      }
    }

		__syncthreads();
    if (threadIdx.x==0)
    {
      ver= atomicAdd(dev_nowNode, all_device);
    }
    __syncthreads();
    now_vertex_idx=ver;
	}

  atomicAdd(&G_counter,P_counter);
  __syncthreads();
  if(threadIdx.x==0)
  {
    atomicAdd(total, G_counter);
  }
}


__global__ void __launch_bounds__(WARP_BLOCK_SIZE)
StepTC_warp_vertex(
  Vertex_type vertex_len_less_128, Vertex_type* vertex_less_128, Offset_type* vertex_less_128_offset, 
  Vertex_type* dev_task2_edge_end, Offset_type* beg_pos, Vertex_type* adj_list,Vertex_type device_i, Vertex_type all_device, 
  unsigned long long *total, unsigned int *dev_nowNode)//, unsigned long long dev_sum,unsigned int dev_nowNode)
{
	__shared__ Vertex_type shd_src[WARP_BLOCK_SIZE/WARP_SIZE*BOUND];
	unsigned long long __shared__ G_counter;
	Vertex_type WARPSIZE=32;
	if (threadIdx.x==0)
	{
		G_counter=0;
	}
	unsigned long long P_counter=0;

	Vertex_type WARPID = threadIdx.x/WARPSIZE;
	Offset_type WARP_TID=threadIdx.x%WARPSIZE;
	Vertex_type now_vertex_idx;

  Vertex_type *p_shd=shd_src+WARPID*BOUND;

  __syncwarp();
  if (WARP_TID==0)
  {
    now_vertex_idx= atomicAdd(dev_nowNode, all_device);
  }
  __syncwarp();
  now_vertex_idx= __shfl_sync(0xffffffff, now_vertex_idx, 0);		

	while (now_vertex_idx < vertex_len_less_128)
	{


    Vertex_type now_vertex_value = vertex_less_128[now_vertex_idx];
    Offset_type degree=beg_pos[now_vertex_value+1]-beg_pos[now_vertex_value];
    Offset_type start=beg_pos[now_vertex_value];
    Offset_type end=beg_pos[now_vertex_value+1];
    Offset_type now=WARP_TID + start;

    while(now<end && degree <= BOUND)
    {
      Vertex_type temp= adj_list[now];
      Offset_type iddx = now - beg_pos[now_vertex_value];
      p_shd[iddx] = temp;

      now+=WARPSIZE;
    }
    __syncwarp();

    now=vertex_less_128_offset[now_vertex_idx];
    end=vertex_less_128_offset[now_vertex_idx+1];

    Offset_type workid=WARP_TID;
    Offset_type l = 0;
    Offset_type r = degree-1;

    while (now<end)
    {
      Vertex_type neighbor=dev_task2_edge_end[now];
      Offset_type neighbor_start=beg_pos[neighbor];
      Offset_type neighbor_degree=beg_pos[neighbor+1]-neighbor_start;

      while (now<end && workid>=neighbor_degree)
      {
        now++;
        workid-=neighbor_degree;
        neighbor=dev_task2_edge_end[now];
        neighbor_start=beg_pos[neighbor];
        neighbor_degree=beg_pos[neighbor+1]-neighbor_start;
        l = 0;
      }
      
      r = degree-1;
      if (now<end)
      {

        Vertex_type temp=adj_list[neighbor_start+workid];
        Vertex_type * srcList = p_shd;
        Offset_type mid = 0;
        
        while(l <= r){
          mid = (l + r) >> 1;
          if(mid >= degree) break;
          if(srcList[mid] == temp){
            P_counter++;
            l = mid+1;
            break;
          }
          else if(srcList[mid] > temp){
            r = mid-1;
          }
          else if(srcList[mid] < temp){
            l = mid+1;
          }
        }

      }
      __syncwarp();

      l=__shfl_sync(0xffffffff,l,WARPSIZE-1);
      now=__shfl_sync(0xffffffff,now,WARPSIZE-1);
      workid=__shfl_sync(0xffffffff,workid,WARPSIZE-1);
      
      workid+=WARP_TID+1;
    }

		__syncwarp();

    if (WARP_TID==0)
    {
      now_vertex_idx= atomicAdd(dev_nowNode, all_device);
    }
    __syncwarp();
    now_vertex_idx= __shfl_sync(0xffffffff, now_vertex_idx, 0);		
	}

	atomicAdd(&G_counter,P_counter);
	__syncthreads();
	if(threadIdx.x==0)
	{
		atomicAdd(total, G_counter);
	}
}



__global__ void dev_calc_csr_degree(
  Vertex_type N, Offset_type* New_dev_nodeIndex, Vertex_type* New_dev_adjList, Offset_type* New_dev_offsets_arr)
{
    Vertex_type src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src < N) {
        Offset_type beg = New_dev_offsets_arr[src];
        Offset_type end = New_dev_offsets_arr[src + 1];
        New_dev_nodeIndex[src] = end - beg;
    }
}


__global__ void dev_calc_re_degree(
  Vertex_type N, Offset_type* dev_degree, Vertex_type* New_dev_adjList, Offset_type* New_dev_offset)
{
    Vertex_type src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src < N) {
        dev_degree[src] = 0;
        Offset_type beg = New_dev_offset[src];
        Offset_type end = New_dev_offset[src + 1];
        Offset_type src_degree = end - beg;

        for (Offset_type idx = beg; idx < end; idx++) {
            uint32_t dst = New_dev_adjList[idx];
            Offset_type degree_arr_dst = New_dev_offset[dst + 1] - New_dev_offset[dst];
            Offset_type degree_arr_src = src_degree;

            if (degree_arr_dst > degree_arr_src || 
                (degree_arr_dst == degree_arr_src && dst > src)) {
                dev_degree[src]++;
            }
        }
    }
}


__global__ void dev_calc_re_adjlist(
  Vertex_type N, Offset_type* New_dev_ori_offset, Vertex_type* New_dev_ori_adjList, 
  Offset_type* New_dev_offsets, Vertex_type* New_dev_half_adjList)
{
    Vertex_type src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src < N) {
        Offset_type beg = New_dev_ori_offset[src];
        Offset_type end = New_dev_ori_offset[src + 1];
        Offset_type new_beg = New_dev_offsets[src];
        Offset_type offset = 0;
        for (Offset_type idx = beg; idx < end; idx++) {
            Vertex_type dst = New_dev_ori_adjList[idx];
            Offset_type degree_arr_dst = New_dev_ori_offset[dst + 1] - New_dev_ori_offset[dst];
            Offset_type degree_arr_src = end - beg;
            if (degree_arr_dst > degree_arr_src ||
                (degree_arr_dst == degree_arr_src && dst > src)) {
                New_dev_half_adjList[new_beg+offset] = dst;
                offset ++;
            }
        }
    }
}


__global__ void dev_calc_task_degree(
  Vertex_type N, Offset_type* Dev_CSR_degree_large, Offset_type* Dev_CSR_degree_small, 
  Offset_type* Dev_re_CSR_degree, 
  Vertex_type* Dev_CSR_adjlist, Offset_type* Dev_CSR_offset)
{
    Vertex_type src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src < N) {
        Dev_CSR_degree_large[src] = 0;
        Dev_CSR_degree_small[src] = 0;
        Offset_type beg                 = Dev_CSR_offset[src];
        Offset_type end                 = Dev_CSR_offset[src + 1];
        Offset_type src_degree          = end - beg;
        Offset_type src_re_degree = Dev_re_CSR_degree[src];
        for (Offset_type idx = beg; idx < end; idx++) {
            Vertex_type dst           = Dev_CSR_adjlist[idx];
            Offset_type dst_re_degree = Dev_re_CSR_degree[dst];
            Offset_type end_degree    = Dev_CSR_offset[dst + 1] - Dev_CSR_offset[dst];
            if(min(src_re_degree, dst_re_degree) < 1) continue;
            if ( max(src_re_degree, dst_re_degree) <= warp_size) 
            {
                if (end_degree > src_degree || 
                    (end_degree == src_degree && dst > src))
                  {
                    Dev_CSR_degree_small[src]++;
                  }
            }
            else if ( src_re_degree >= large_size || dst_re_degree >= large_size ) 
            {
                if(src_re_degree > dst_re_degree || 
                    (src_re_degree == dst_re_degree && src > dst ) )
                    {
                  Dev_CSR_degree_large[src]++;
                }
            }
        }
    }
}

__global__ void dev_calc_task_adjlist(
  Vertex_type N, Offset_type* Dev_CSR_offset, Vertex_type* Dev_CSR_adjlist, 
  Offset_type* Dev_task_CSR_offset_large, Offset_type* Dev_task_CSR_offset_small, 
  Offset_type* Dev_re_CSR_degree, Vertex_type* Dev_task_CSR_adjlist_small, 
  Vertex_type* Dev_task_CSR_adjlist_large)
{
  Vertex_type src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src < N) {
    Offset_type new_beg_large     = Dev_task_CSR_offset_large[src];
    Offset_type new_beg_small     = Dev_task_CSR_offset_small[src];
    Offset_type offset_large      = 0;
    Offset_type offset_small      = 0;
    Offset_type beg               = Dev_CSR_offset[src];
    Offset_type end               = Dev_CSR_offset[src + 1];
    Offset_type src_degree        = end - beg;
    Offset_type src_re_degree     = Dev_re_CSR_degree[src];
    for (Offset_type idx = beg; idx < end; idx++) {
      Vertex_type dst           = Dev_CSR_adjlist[idx];
      Offset_type dst_re_degree = Dev_re_CSR_degree[dst];
      Offset_type end_degree    = Dev_CSR_offset[dst + 1] - Dev_CSR_offset[dst];
      if(min(src_re_degree, dst_re_degree) < 1) continue;
      if ( max(src_re_degree, dst_re_degree) <= warp_size) 
      {
        if (end_degree > src_degree || 
            (end_degree == src_degree && dst > src))
        {
          Dev_task_CSR_adjlist_small[new_beg_small+offset_small] = dst;
          offset_small++;
        }
      }
      else if ( src_re_degree >= large_size || dst_re_degree >= large_size ) 
      {
        if(src_re_degree > dst_re_degree || 
          (src_re_degree == dst_re_degree && src > dst ) )
          {
            Dev_task_CSR_adjlist_large[new_beg_large+offset_large] = dst;
            offset_large++;
          }
      }
    }
  }
}

__global__ void dev_calc_task_vertex(Vertex_type N, Offset_type *Dev_CSR_degree, char *Dev_CSR_flag)
{
    Vertex_type src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src < N) {
        if(Dev_CSR_degree[src] > 0)
        {
            Dev_CSR_flag[src]   = true;
            Dev_CSR_degree[src] = src;
        }
    }
}




unsigned long long tricount(Vertex_type gpu_num, Offset_type edge_num, Vertex_type N, Vertex_type * adjList, Offset_type * offSet){

    Offset_type * CPU_CSR_offset = offSet;
    Vertex_type * CPU_CSR_adjlist = adjList;

    Offset_type* Dev_CSR_offset;
    Vertex_type* Dev_CSR_adjlist;
    gpuErrchk( cudaMalloc((void**)&Dev_CSR_adjlist , sizeof(Vertex_type) * edge_num));   // sleep(30);
    gpuErrchk( cudaMalloc((void**)&Dev_CSR_offset  , sizeof(Offset_type) * (N + 1)));
    
    gpuErrchk( cudaMemcpy(Dev_CSR_offset  , CPU_CSR_offset  , sizeof(Offset_type) * (N + 1)   , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(Dev_CSR_adjlist , CPU_CSR_adjlist , sizeof(Vertex_type) * (edge_num), cudaMemcpyHostToDevice) );

    Offset_type* Dev_re_CSR_degree;
    Offset_type* Dev_re_CSR_offset;
    Vertex_type* Dev_re_CSR_adjlist;
    gpuErrchk( cudaMalloc((void**)&Dev_re_CSR_degree  , sizeof(Offset_type) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_re_CSR_offset  , sizeof(Offset_type) * (N + 1)));
    

    dev_calc_re_degree<<<(N + 1023) / 1024, 1024>>>(
      N, Dev_re_CSR_degree, Dev_CSR_adjlist, Dev_CSR_offset);
    gpuErrchk( cudaDeviceSynchronize() ); 

    void* dev_temp_storage = nullptr; size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(dev_temp_storage, temp_storage_bytes, 
    Dev_re_CSR_degree, Dev_re_CSR_offset, N+1);
    cudaMalloc(&dev_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(dev_temp_storage, temp_storage_bytes, 
    Dev_re_CSR_degree, Dev_re_CSR_offset, N+1);
    cudaFree(dev_temp_storage);
    gpuErrchk( cudaDeviceSynchronize() ); 


    char    * Dev_task_CSR_flag_large;
    Vertex_type* Dev_task_CSR_vertex_large;
    Offset_type* Dev_task_CSR_degree_large;
    Offset_type* Dev_task_CSR_offset_large;
    Vertex_type* Dev_task_CSR_adjlist_large;

    char    * Dev_task_CSR_flag_small;
    Vertex_type* Dev_task_CSR_vertex_small;
    Offset_type* Dev_task_CSR_degree_small;
    Offset_type* Dev_task_CSR_offset_small;
    Vertex_type* Dev_task_CSR_adjlist_small;
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_flag_large    , sizeof(char) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_flag_small    , sizeof(char) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_vertex_large  , sizeof(Vertex_type) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_vertex_small  , sizeof(Vertex_type) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_degree_large  , sizeof(Offset_type) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_degree_small  , sizeof(Offset_type) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_offset_large  , sizeof(Offset_type) * (N + 1)));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_offset_small  , sizeof(Offset_type) * (N + 1)));

    dev_calc_task_degree<<<(N + 1023) / 1024, 1024>>>(
      N, Dev_task_CSR_degree_large, Dev_task_CSR_degree_small, 
      Dev_re_CSR_degree, Dev_CSR_adjlist, Dev_CSR_offset);
    gpuErrchk( cudaDeviceSynchronize() ); 

    void* dev_temp_storage_off = nullptr; 
    size_t temp_storage_bytes_off = 0;
    cub::DeviceScan::ExclusiveSum(dev_temp_storage_off, temp_storage_bytes_off, 
      Dev_task_CSR_degree_large, Dev_task_CSR_offset_large, N+1);
    cudaMalloc(&dev_temp_storage_off, temp_storage_bytes_off);
    cub::DeviceScan::ExclusiveSum(dev_temp_storage_off, temp_storage_bytes_off, 
      Dev_task_CSR_degree_large, Dev_task_CSR_offset_large, N+1);
    cudaFree(dev_temp_storage_off);

    void* dev_temp_storage_off1 = nullptr; 
    size_t temp_storage_bytes_off1 = 0;
    cub::DeviceScan::ExclusiveSum(dev_temp_storage_off1, temp_storage_bytes_off1, 
      Dev_task_CSR_degree_small, Dev_task_CSR_offset_small, N+1);
    cudaMalloc(&dev_temp_storage_off1, temp_storage_bytes_off1);
    cub::DeviceScan::ExclusiveSum(dev_temp_storage_off1, temp_storage_bytes_off1, 
      Dev_task_CSR_degree_small, Dev_task_CSR_offset_small, N+1);
    cudaFree(dev_temp_storage_off1);
    gpuErrchk( cudaDeviceSynchronize() ); 

    Offset_type large_nonzero = 0;
    Offset_type small_nonzero = 0;
    gpuErrchk( cudaMemcpy(&large_nonzero, Dev_task_CSR_offset_large + N, sizeof(Offset_type), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&small_nonzero, Dev_task_CSR_offset_small + N, sizeof(Offset_type), cudaMemcpyDeviceToHost) );

    dev_calc_task_vertex<<<(N + 1023) / 1024, 1024>>>(
      N, Dev_task_CSR_degree_large, Dev_task_CSR_flag_large);
    gpuErrchk( cudaDeviceSynchronize() ); 

    Offset_type     *d_num_selected_out   = NULL;
    cudaMalloc((void**)&d_num_selected_out, sizeof(Offset_type));
    void* dev_temp_storage_non = nullptr; 
    size_t temp_storage_bytes_non = 0;
    cub::DeviceSelect::Flagged(
      dev_temp_storage_non, temp_storage_bytes_non, 
      Dev_task_CSR_degree_large, Dev_task_CSR_flag_large, Dev_task_CSR_vertex_large, 
      d_num_selected_out, N);
    cudaMalloc(&dev_temp_storage_non, temp_storage_bytes_non);
    cub::DeviceSelect::Flagged(
      dev_temp_storage_non, temp_storage_bytes_non, 
      Dev_task_CSR_degree_large, Dev_task_CSR_flag_large, Dev_task_CSR_vertex_large, 
      d_num_selected_out, N);
    cudaFree(dev_temp_storage_non);
    Offset_type NewN_large = 0;
    cudaMemcpy(&NewN_large, d_num_selected_out, sizeof(Offset_type), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaDeviceSynchronize() ); 

    cudaFree(Dev_task_CSR_degree_large);
    cudaFree(Dev_task_CSR_flag_large);

    dev_calc_task_vertex<<<(N + 1023) / 1024, 1024>>>(
      N, Dev_task_CSR_degree_small, Dev_task_CSR_flag_small);
    gpuErrchk( cudaDeviceSynchronize() ); 

    Offset_type     *d_num_selected_out_small   = NULL;
    cudaMalloc((void**)&d_num_selected_out_small, sizeof(Offset_type));
    void* dev_temp_storage_non_small = nullptr; 
    size_t temp_storage_bytes_non_small = 0;
    cub::DeviceSelect::Flagged(
      dev_temp_storage_non_small, temp_storage_bytes_non_small, 
      Dev_task_CSR_degree_small, Dev_task_CSR_flag_small, Dev_task_CSR_vertex_small, 
      d_num_selected_out_small, N);
    cudaMalloc(&dev_temp_storage_non_small, temp_storage_bytes_non_small);
    cub::DeviceSelect::Flagged(
      dev_temp_storage_non_small, temp_storage_bytes_non_small, 
      Dev_task_CSR_degree_small, Dev_task_CSR_flag_small, Dev_task_CSR_vertex_small, 
      d_num_selected_out_small, N);
    cudaFree(dev_temp_storage_non_small);
    Offset_type NewN_small = 0;
    cudaMemcpy(&NewN_small, d_num_selected_out_small, sizeof(Offset_type), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaDeviceSynchronize() ); 

    cudaFree(Dev_task_CSR_degree_small);
    cudaFree(Dev_task_CSR_flag_small);

    gpuErrchk( cudaMalloc((void**)&Dev_re_CSR_adjlist , sizeof(Vertex_type) * edge_num/2));
    dev_calc_re_adjlist<<<(N + 1023) / 1024, 1024>>>(
      N, Dev_CSR_offset, Dev_CSR_adjlist, Dev_re_CSR_offset, Dev_re_CSR_adjlist);
    gpuErrchk( cudaDeviceSynchronize() ); 

    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_adjlist_large , sizeof(Vertex_type) * large_nonzero));
    gpuErrchk( cudaMalloc((void**)&Dev_task_CSR_adjlist_small , sizeof(Vertex_type) * small_nonzero));
    dev_calc_task_adjlist<<<(N + 1023) / 1024, 1024>>>(
      N, Dev_CSR_offset, Dev_CSR_adjlist,
      Dev_task_CSR_offset_large, Dev_task_CSR_offset_small, 
      Dev_re_CSR_degree, 
      Dev_task_CSR_adjlist_small, Dev_task_CSR_adjlist_large);
    gpuErrchk( cudaDeviceSynchronize() ); 

    void* dev_temp_storage_compact = nullptr;
    size_t temp_storage_bytes_compact = 0;
    cub::DeviceSelect::Unique(
      dev_temp_storage_compact, temp_storage_bytes_compact, 
      Dev_task_CSR_offset_large, Dev_task_CSR_offset_large, d_num_selected_out, N+1);
    cudaMalloc(&dev_temp_storage_compact, temp_storage_bytes_compact);
    cub::DeviceSelect::Unique(
      dev_temp_storage_compact, temp_storage_bytes_compact, 
      Dev_task_CSR_offset_large, Dev_task_CSR_offset_large, d_num_selected_out, N+1);
    cudaFree(dev_temp_storage_compact);
    gpuErrchk( cudaDeviceSynchronize() ); 

    void* dev_temp_storage_compact_small = nullptr;
    size_t temp_storage_bytes_compact_small = 0;
    cub::DeviceSelect::Unique(
      dev_temp_storage_compact_small, temp_storage_bytes_compact_small, 
      Dev_task_CSR_offset_small, Dev_task_CSR_offset_small, d_num_selected_out_small, N+1);
    cudaMalloc(&dev_temp_storage_compact_small, temp_storage_bytes_compact_small);
    cub::DeviceSelect::Unique(
      dev_temp_storage_compact_small, temp_storage_bytes_compact_small, 
      Dev_task_CSR_offset_small, Dev_task_CSR_offset_small, d_num_selected_out_small, N+1);
    cudaFree(dev_temp_storage_compact_small);
    gpuErrchk( cudaDeviceSynchronize() ); 

    Offset_type maxDegree;
    size_t temp_storage_bytes_max = 0;
    void* d_temp_storage_max = nullptr;
    Offset_type     *d_num_selected_out_max   = NULL;
    cudaMalloc((void**)&d_num_selected_out_max, sizeof(Offset_type));

    cub::DeviceReduce::Max(d_temp_storage_max, temp_storage_bytes_max, Dev_re_CSR_degree, d_num_selected_out_max, N);
    cudaMalloc(&d_temp_storage_max, temp_storage_bytes_max);
    cub::DeviceReduce::Max(d_temp_storage_max, temp_storage_bytes_max, Dev_re_CSR_degree, d_num_selected_out_max, N, cudaStreamDefault);
    cudaMemcpy(&maxDegree, d_num_selected_out_max, sizeof(Offset_type), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage_max);
    gpuErrchk( cudaDeviceSynchronize() ); 
    cudaFree(Dev_re_CSR_degree);

    unsigned long long sum = 0;

    Vertex_type bucket_size = 64;
    Offset_type bucket_num = HASH_SIZE_large;
    Offset_type bucket_num_small = HASH_SIZE_small;

    int warp_block_num  = WARP_BLOCK_NUM;
    int warp_thread_num = WARP_BLOCK_SIZE;
    int warp_nowNode = warp_block_num*warp_thread_num/WARP_SIZE; 


    std::vector<unsigned long long>   h_counts(gpu_num, 0);
    std::vector<unsigned long long *> d_count(gpu_num);
    for (Vertex_type i = 0; i < gpu_num; i++) {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMalloc(&d_count[i], sizeof(unsigned long long)));
    }

    std::vector<unsigned int> h_nowNode(gpu_num, WARP_BLOCK_NUM);
    std::vector<unsigned int> h_nowNodeWarp(gpu_num, warp_nowNode);
    
    std::vector<unsigned int *> d_nowNode(gpu_num);
    for (Vertex_type i = 0; i < gpu_num; i++) {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMalloc(&d_nowNode[i], sizeof(unsigned int)));
    }

    Offset_type* Inner_Dev_re_CSR_offset[gpu_num];
    Vertex_type* Inner_Dev_re_CSR_adjlist[gpu_num]; 

    Offset_type  Inner_NewN_large[gpu_num];
      Vertex_type* Inner_Dev_task_CSR_vertex_large[gpu_num];
      Offset_type* Inner_Dev_task_CSR_offset_large[gpu_num];
      Vertex_type* Inner_Dev_task_CSR_adjlist_large[gpu_num]; 

    Offset_type Inner_NewN_small[gpu_num];
      Vertex_type* Inner_Dev_task_CSR_vertex_small[gpu_num]; 
      Offset_type* Inner_Dev_task_CSR_offset_small[gpu_num];
      Vertex_type* Inner_Dev_task_CSR_adjlist_small[gpu_num]; 

    
    enableP2P(gpu_num);

    gpuErrchk(cudaSetDevice(0));
    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(gpu_num * sizeof(
                                cudaStream_t));

    int block_block_num  = WARP_BLOCK_NUM;
    int block_thread_num = WARP_BLOCK_SIZE;

    Vertex_type * hash_bucket[gpu_num];

    
    TimeInterval mulitygpuallTime;
    {
        std::vector<std::thread> threads;

        for (Vertex_type gpuid = 0; gpuid < gpu_num; gpuid++) {
          threads.push_back(std::thread([&,gpuid]() {
            gpuErrchk(cudaSetDevice(gpuid));
            gpuErrchk( cudaMalloc((void**)&hash_bucket[gpuid], sizeof(Vertex_type)*block_block_num* (bucket_num+5) * bucket_size ) );

            gpuErrchk(cudaStreamCreate(&(streams[gpuid])));

            gpuErrchk(cudaMemcpy(d_count[gpuid], &h_counts[gpuid], sizeof(unsigned long long), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNode[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));

            if(gpuid==0){

              Inner_Dev_re_CSR_offset[gpuid]  = Dev_re_CSR_offset;
              Inner_Dev_re_CSR_adjlist[gpuid] = Dev_re_CSR_adjlist;
              
              Inner_NewN_large[gpuid]                 = NewN_large;
              Inner_Dev_task_CSR_vertex_large[gpuid]  = Dev_task_CSR_vertex_large;
              Inner_Dev_task_CSR_offset_large[gpuid]  = Dev_task_CSR_offset_large;
              Inner_Dev_task_CSR_adjlist_large[gpuid] = Dev_task_CSR_adjlist_large; 


              Inner_NewN_small[gpuid]                 = NewN_small;
              Inner_Dev_task_CSR_vertex_small[gpuid]  = Dev_task_CSR_vertex_small; 
              Inner_Dev_task_CSR_offset_small[gpuid]  = Dev_task_CSR_offset_small;
              Inner_Dev_task_CSR_adjlist_small[gpuid] = Dev_task_CSR_adjlist_small;
            }
            else{

              gpuErrchk( cudaMalloc((void**)&Inner_Dev_re_CSR_offset[gpuid]  , sizeof(Offset_type) * (N + 1)));
              gpuErrchk( cudaMalloc((void**)&Inner_Dev_re_CSR_adjlist[gpuid] , sizeof(Vertex_type) * edge_num/2));
              Inner_NewN_large[gpuid]                 = NewN_large;
              Inner_NewN_small[gpuid]                 = NewN_small;

              cudaMemcpyPeerAsync(Inner_Dev_re_CSR_offset[gpuid], gpuid,  Dev_re_CSR_offset,  0, sizeof(Offset_type) * (N + 1));
              cudaMemcpyPeerAsync(Inner_Dev_re_CSR_adjlist[gpuid], gpuid, Dev_re_CSR_adjlist, 0, sizeof(Vertex_type) * edge_num/2);


              gpuErrchk( cudaMalloc((void**)&Inner_Dev_task_CSR_vertex_large[gpuid]  , sizeof(Vertex_type) * (N + 1)));
              gpuErrchk( cudaMalloc((void**)&Inner_Dev_task_CSR_offset_large[gpuid]  , sizeof(Offset_type) * (N + 1)));
              gpuErrchk( cudaMalloc((void**)&Inner_Dev_task_CSR_adjlist_large[gpuid] , sizeof(Vertex_type) * large_nonzero));

              cudaMemcpyPeerAsync(Inner_Dev_task_CSR_vertex_large[gpuid]  , gpuid,  Dev_task_CSR_vertex_large,  0, sizeof(Vertex_type) * (N + 1));
              cudaMemcpyPeerAsync(Inner_Dev_task_CSR_offset_large[gpuid]  , gpuid,  Dev_task_CSR_offset_large,  0, sizeof(Offset_type) * (N + 1));
              cudaMemcpyPeerAsync(Inner_Dev_task_CSR_adjlist_large[gpuid] , gpuid,  Dev_task_CSR_adjlist_large, 0, sizeof(Vertex_type) * large_nonzero);
              

              gpuErrchk( cudaMalloc((void**)&Inner_Dev_task_CSR_vertex_small[gpuid]  , sizeof(Vertex_type) * (N + 1)));    
              gpuErrchk( cudaMalloc((void**)&Inner_Dev_task_CSR_offset_small[gpuid]  , sizeof(Offset_type) * (N + 1)));
              gpuErrchk( cudaMalloc((void**)&Inner_Dev_task_CSR_adjlist_small[gpuid] , sizeof(Vertex_type) * small_nonzero));

              cudaMemcpyPeerAsync(Inner_Dev_task_CSR_vertex_small[gpuid]  , gpuid,  Dev_task_CSR_vertex_small , 0, sizeof(Vertex_type) * (N + 1));    
              cudaMemcpyPeerAsync(Inner_Dev_task_CSR_offset_small[gpuid]  , gpuid,  Dev_task_CSR_offset_small , 0, sizeof(Offset_type) * (N + 1));
              cudaMemcpyPeerAsync(Inner_Dev_task_CSR_adjlist_small[gpuid] , gpuid,  Dev_task_CSR_adjlist_small, 0, sizeof(Vertex_type) * small_nonzero);

              
            }
            cudaDeviceSynchronize();

          }));
        }
        for (auto &thread: threads) thread.join();
    }



    gpuErrchk(cudaSetDevice(0));
    
    mulitygpuallTime.check();
    {
        std::vector<std::thread> threads;
        for (Vertex_type gpuid = 0; gpuid < gpu_num; gpuid++) {
          threads.push_back(std::thread([&,gpuid]() {
            gpuErrchk(cudaSetDevice(gpuid));

            h_nowNode[gpuid] = gpuid;
            Vertex_type end  = Inner_NewN_large[gpuid];

            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNode[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));
            StepTC_block_vertex_hash<<<
            block_block_num,block_thread_num, max(Offset_type(maxDegree+128),bucket_num*2)*sizeof(Vertex_type),streams[gpuid]>>>(
              4096, 100000, bucket_num, bucket_size, hash_bucket[gpuid], 
              end                                   , Inner_Dev_task_CSR_vertex_large[gpuid],
              Inner_Dev_task_CSR_offset_large[gpuid], Inner_Dev_task_CSR_adjlist_large[gpuid],
              Inner_Dev_re_CSR_offset[gpuid]        , Inner_Dev_re_CSR_adjlist[gpuid], 
              gpuid,gpu_num, d_count[gpuid], d_nowNode[gpuid]);
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNode[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            if(maxDegree > 3072) 
            StepTC_block_vertex_hash<<<
            block_block_num,block_thread_num, max(Offset_type(4096),bucket_num*2)*sizeof(Vertex_type),streams[gpuid]>>>(
              3072, 4096, bucket_num, bucket_size, hash_bucket[gpuid], 
              end                                   , Inner_Dev_task_CSR_vertex_large[gpuid],
              Inner_Dev_task_CSR_offset_large[gpuid], Inner_Dev_task_CSR_adjlist_large[gpuid], 
              Inner_Dev_re_CSR_offset[gpuid]        , Inner_Dev_re_CSR_adjlist[gpuid], 
              gpuid,gpu_num,  d_count[gpuid], d_nowNode[gpuid]);
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));


            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNode[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            if(maxDegree > 2048) 
            StepTC_block_vertex_hash<<<
            block_block_num,block_thread_num, max(Offset_type(3072),bucket_num*2)*sizeof(Vertex_type),streams[gpuid]>>>(
              2048, 3072, bucket_num, bucket_size, hash_bucket[gpuid], 
              end                                   , Inner_Dev_task_CSR_vertex_large[gpuid],
              Inner_Dev_task_CSR_offset_large[gpuid], Inner_Dev_task_CSR_adjlist_large[gpuid], 
              Inner_Dev_re_CSR_offset[gpuid]        , Inner_Dev_re_CSR_adjlist[gpuid],  
              gpuid,gpu_num,  d_count[gpuid], d_nowNode[gpuid]);
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNode[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            if(maxDegree > 1024) 
            StepTC_block_vertex_hash<<<
            block_block_num,block_thread_num, max(Offset_type(2048),bucket_num*2)*sizeof(Vertex_type),streams[gpuid]>>>(
              1024, 2048, bucket_num, bucket_size, hash_bucket[gpuid], 
              end               , Inner_Dev_task_CSR_vertex_large[gpuid],
              Inner_Dev_task_CSR_offset_large[gpuid], Inner_Dev_task_CSR_adjlist_large[gpuid], 
              Inner_Dev_re_CSR_offset[gpuid]        , Inner_Dev_re_CSR_adjlist[gpuid], 
              gpuid,gpu_num,  d_count[gpuid], d_nowNode[gpuid]);
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));
          
            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNode[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            if(maxDegree > large_size) 
            StepTC_block_vertex_hash_1024<<<
            block_block_num,block_thread_num, max(Offset_type(1024+128),bucket_num_small*2)*sizeof(Vertex_type),streams[gpuid]>>>(
              0, 1024, bucket_num_small, bucket_size, hash_bucket[gpuid], 
              end                                   , Inner_Dev_task_CSR_vertex_large[gpuid],
              Inner_Dev_task_CSR_offset_large[gpuid], Inner_Dev_task_CSR_adjlist_large[gpuid], 
              Inner_Dev_re_CSR_offset[gpuid]        , Inner_Dev_re_CSR_adjlist[gpuid], 
              gpuid,gpu_num,  d_count[gpuid], d_nowNode[gpuid]);
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            h_nowNodeWarp[gpuid] = gpuid;
            Vertex_type warp_end  = Inner_NewN_small[gpuid];

            gpuErrchk(cudaMemcpy(d_nowNode[gpuid], &h_nowNodeWarp[gpuid], sizeof(unsigned int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));

            StepTC_warp_vertex<<<warp_block_num,warp_thread_num,0,streams[gpuid]>>>(
              warp_end                              , Inner_Dev_task_CSR_vertex_small[gpuid], 
              Inner_Dev_task_CSR_offset_small[gpuid], Inner_Dev_task_CSR_adjlist_small[gpuid], 
              Inner_Dev_re_CSR_offset[gpuid]        ,  Inner_Dev_re_CSR_adjlist[gpuid], 
              gpuid,gpu_num,  d_count[gpuid]        , d_nowNode[gpuid]);
            gpuErrchk(cudaStreamSynchronize(streams[gpuid]));
          }));
        }
        for (auto &thread: threads) thread.join();
    }

    mulitygpuallTime.print("mulity gpu all Time Cost");


    gpuErrchk(cudaSetDevice(0));
    gpuErrchk( cudaPeekAtLastError() );
    for(Vertex_type gpuid = 0; gpuid < gpu_num; gpuid++){
        gpuErrchk(cudaStreamSynchronize(streams[gpuid]));
        gpuErrchk(cudaMemcpy(&h_counts[gpuid], d_count[gpuid], sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaStreamSynchronize(streams[gpuid]));
        gpuErrchk( cudaFree(hash_bucket[gpuid]));
        sum+=h_counts[gpuid];
        //cout << h_counts[gpuid] << endl;
    }
    
    return sum;
}

