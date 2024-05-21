#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "my_library.h"

void printMatrix(DATA_TYPE **array, int size, const char *message) 
{
    if(size < 10)
    {
        printf(message);

        for (int i = 0; i < size; i++) 
        {
            for (int j = 0; j < size; j++) 
            {
                printf(FORMAT_SPECIFIER" ", array[i][j]);
            }
            printf("\n");
        }
    }
    else
    {
        printf("Matrix size is to big, skipping print \n");
    }
}

DATA_TYPE** createMatrix(int size){
    DATA_TYPE **matrix = (DATA_TYPE **)malloc(size * sizeof(DATA_TYPE *));
    
    // Allocate memory for each 2D array
    for (int i = 0; i < size; i++) {
        // Allocate memory for rows
        matrix[i] = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    }
    
    return matrix;
}

void initializeMatrixValues(DATA_TYPE **matrix, int size)
{
    if (matrix == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            matrix[i][j] = rand() % 10 + 1;
        }
    }
}


void flush_cache() {
    const int CACHE_SIZE = 39424 * 1024;  // Kilobytes to bytes
    char *cache = (char *)malloc(CACHE_SIZE);
    if (cache == NULL) {
        fprintf(stderr, "Failed to allocate memory for cache flushing\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < CACHE_SIZE; i += 4096) {
        cache[i] = 0; // Accessing each page to ensure it's loaded into cache
    }
    free(cache);
}


double calculate_effective_bandwidth(int size, double time)
{
    const int GB_SIZE = 1073741824;

    // bytes/second
    double effective_bandwidth = (calculateWork(size) * sizeof(DATA_TYPE))/ (time);
    double effectve_bandwidth_gb_per_second = effective_bandwidth/GB_SIZE;

    // return the effective bandwidth in gb/s
    return effectve_bandwidth_gb_per_second;
}

// Check errors and print GB/s
void calculate_effective_bandwidth(const DATA_TYPE *ref, const DATA_TYPE *res, int size, float time)
{
  bool passed = true;
  int NUM_REPTS = 1; // TODO: is the number of repetitions we define outside
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
  {
    printf("%20.2f\n", 2 * size * sizeof(DATA_TYPE) * 1e-6 * NUM_REPS / time );
    const int GB_SIZE = 1073741824;

    // bytes/second
    double effective_bandwidth = (calculateWork(size) * sizeof(DATA_TYPE))/ (time);
    double effectve_bandwidth_gb_per_second = effective_bandwidth/GB_SIZE;

    printf("Mine %20.2f\n", effectve_bandwidth_gb_per_second);
  }
}