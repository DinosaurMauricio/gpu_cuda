#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "./include/helper_cuda.h"

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

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;



__global__ void copy(DATA_TYPE *odata, const DATA_TYPE *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

int main(int argc, char *argv[])
{

    int matrixSize = -1;
    //int numberOfTests = -1;
    int blockSize = -1;

    if (argc < 2)
    {
        printf("No matrix size or number of tries was provided. Defaulting to 1. \n");
        matrixSize = 1;
        //numberOfTests = 1;
    }
    else
    {
        matrixSize = atoi(argv[1]);
        //numberOfTests = atoi(argv[2]);
        if (argc >= 4) 
        {
            if (strcmp(argv[argc - 1], "--valgrind") != 0)
            {
                blockSize = atoi(argv[3]);
            }
        }
    }
    int size = pow(2, matrixSize);

    printf("The size of the matrix is %dx%d \n",size,size );

    if(blockSize == -1)
    {
        printf("No block size was provided. Will work with full matrix \n");
    }
    else 
    {
        printf("Block size is %d \n", blockSize);
    }

    if (size % blockSize != 0) 
    {
        printf("Block size must be a multiple of the matrix size.\n");
        exit(1);
        
    }
    const int mem_size = size*sizeof(DATA_TYPE);

    dim3 dimGrid(matrixSize/TILE_DIM, matrixSize/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         size, size, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
            dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    DATA_TYPE *h_idata = (DATA_TYPE*)malloc(mem_size);
    DATA_TYPE *h_cdata = (DATA_TYPE*)malloc(mem_size);
    DATA_TYPE *h_tdata = (DATA_TYPE*)malloc(mem_size);
    DATA_TYPE *gold    = (DATA_TYPE*)malloc(mem_size);


    DATA_TYPE *d_idata, *d_cdata, *d_tdata;
    checkCuda( cudaMalloc(&d_idata, mem_size) );
    checkCuda( cudaMalloc(&d_cdata, mem_size) );
    checkCuda( cudaMalloc(&d_tdata, mem_size) );

    for (int j = 0; j < matrixSize; j++)
        for (int i = 0; i < matrixSize; i++)
        h_idata[j*matrixSize + i] = j*matrixSize + i;

    // correct result for error checking
    for (int j = 0; j < matrixSize; j++)
        for (int i = 0; i < matrixSize; i++)
        gold[j*matrixSize + i] = h_idata[i*matrixSize + j];

    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    float ms;

    checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    printf("%25s", "copy");
    checkCuda( cudaMemset(d_cdata, 0, mem_size) );
    // warm up
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda( cudaEventRecord(startEvent, 0) );
    for (int i = 0; i < NUM_REPS; i++)
        copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    checkCuda( cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost) );
    postprocess(h_idata, h_cdata, size, ms);

    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
    checkCuda( cudaFree(d_tdata) );
    checkCuda( cudaFree(d_cdata) );
    checkCuda( cudaFree(d_idata) );
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(gold);
}