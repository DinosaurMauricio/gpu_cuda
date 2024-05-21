#include <stdio.h>
#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

#ifndef FORMAT_SPECIFIER
#define FORMAT_SPECIFIER "\t%d"
#endif

void printMatrix(DATA_TYPE **array, int size, const char *message);
void initializeMatrixValues(DATA_TYPE **matrix, int size);
DATA_TYPE** createMatrix(int size);
void calculate_effective_bandwidth(const DATA_TYPE *ref, const DATA_TYPE *res, int size, float time);

