#include <stdio.h>
#include <assert.h>
#include <math.h>

extern "C" {
#include "my_library.h"
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif
  
__global__ void transposeNoBankConflicts(DATA_TYPE *odata, const DATA_TYPE *idata)
{
  __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeNoBankConflictsUnrolled(DATA_TYPE *odata, const DATA_TYPE *idata)
{
  __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;


  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


// Function pointer type for the kernels
template <typename KernelFunc>
void runKernelAndMeasure(const char* kernelName, KernelFunc kernel, dim3 dimGrid, dim3 dimBlock, 
                         DATA_TYPE* d_cdata, const DATA_TYPE* d_idata, DATA_TYPE* h_cdata, 
                         size_t memory_size, size_t size, int numberOfTests, cudaEvent_t startEvent, 
                         cudaEvent_t stopEvent) 
{
    float ms;
    printf("%25s", kernelName);
    checkCuda(cudaMemset(d_cdata, 0, memory_size));

    // Warm up
    kernel<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < numberOfTests; i++)
        kernel<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, memory_size, cudaMemcpyDeviceToHost));
    calculate_effective_bandwidth(size * size, numberOfTests, ms);
    printf("%25s %f ms\n", "Time:", ms);
}



int main(int argc, char **argv)
{
    int matrixSize = -1;
    int numberOfTests = -1;

    if (argc < 2)
    {
        printf("No matrix size or number of tries was provided. Defaulting to 10. \n");
        matrixSize = 10;
        numberOfTests = 100;
    }
    else
    {
        matrixSize = atoi(argv[1]);
        numberOfTests = atoi(argv[2]);
    }

    int size = pow(2, matrixSize);
    printf("The size of the matrix is %dx%d \n",size,size );


    if (size % BLOCK_ROWS != 0) 
    {
        printf("Block size must be a multiple of the matrix size.\n");
        exit(1);
        
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        exit(1);
    }

    const int memory_size = size*size*sizeof(DATA_TYPE);

    dim3 dimGrid(size/TILE_DIM, size/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    printf("Block size: %d %d, Tile size: %d %d\n", TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
            dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    DATA_TYPE *h_idata = (DATA_TYPE*)malloc(memory_size);
    DATA_TYPE *h_cdata = (DATA_TYPE*)malloc(memory_size);
    DATA_TYPE *d_idata, *d_cdata;

    cudaEvent_t startEvent, stopEvent;

    checkCuda( cudaMalloc(&d_idata, memory_size) );
    checkCuda( cudaMalloc(&d_cdata, memory_size) );

        
    initializeMatrixValues(h_idata,size);
    
    // events for timing
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    // device
    checkCuda( cudaMemcpy(d_idata, h_idata, memory_size, cudaMemcpyHostToDevice) );
    
    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    // Run the kernels using the template function
    runKernelAndMeasure("transposeNoBankConflicts", transposeNoBankConflicts, dimGrid, dimBlock, 
                        d_cdata, d_idata, h_cdata, gold, memory_size, size, numberOfTests, startEvent, stopEvent);

    runKernelAndMeasure("Unroll", transposeNoBankConflictsUnrolled, dimGrid, dimBlock, 
                        d_cdata, d_idata, h_cdata, gold, memory_size, size, numberOfTests, startEvent, stopEvent);

    // cleanup
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
    checkCuda( cudaFree(d_cdata) );
    checkCuda( cudaFree(d_idata) );
    free(h_idata);
    free(h_cdata);;
}