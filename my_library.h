#include <stdio.h>
#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

#ifndef FORMAT_SPECIFIER
#define FORMAT_SPECIFIER "\t%d"
#endif

void flush_cache();
void printMatrix(DATA_TYPE **array, int size, const char *message);
void initializeMatrixValues(DATA_TYPE **matrix, int size);
DATA_TYPE** createMatrix(int size);
double calculate_effective_bandwidth(int size, double time);

