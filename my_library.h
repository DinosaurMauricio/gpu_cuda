#include <stdio.h>
#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

#ifndef FORMAT_SPECIFIER
#define FORMAT_SPECIFIER "%d"
#endif

void host_transpose(DATA_TYPE *matrix, DATA_TYPE * transposed_matrix, int size);
void printMatrix(DATA_TYPE *array, int size, const char *message);
void initializeMatrixValues(DATA_TYPE *matrix, int size);
void calculate_effective_bandwidth(const DATA_TYPE *ref, const DATA_TYPE *res, int size, int number_of_repetitions,float time);