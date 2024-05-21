# GPU Computing: Homework 1

This repository contains the code and results for Homework 1 of the GPU Computing course. The homework focuses on analyzing the performance of a matrix transposition algorithm using different optimization levels and cache behavior analysis.

## Makefile 
Use the Makefile to compile and run the program with specific options. Here's how you can use it

```bash
make DATA_TYPE=double BANDWIDTH_PERFORMANCE=O3
```

Replace the options DATA_TYPE and BANDWIDTH_PERFORMANCE as needed. In the example above, DATA_TYPE specifies the data type (in this case, double) and BANDWIDTH_PERFORMANCE specifies the optimization level (in this case, O3).

If DATA_TYPE or BANDWIDTH_PERFORMANCE are not provided they will default to int and -O0 respectively.



## Usage

```bash
./gpu_transpose <matrix_size> <number_of_runs> [<block_size>] [--valgrind]

<matrix_size>: Specify the size of the square matrices to be transposed. It will be the power of two, e.g. 3 will be a matrix of 8x8
<number_of_runs>: Specify the number of runs the program should execute.
<block_size> (optional): If provided, the transpose operation will be done by blocks, the block size must be a multiple of the matrix size.
--valgrind (optional): If provided, runs the program with Valgrind for memory checkin
```
## Example

To run the program with a matrix size of 16x16 and execute 10 runs:

```bash
./gpu_transpose 4 10
```

To run the program with Valgrind for memory checking:

```bash
./gpu_transpose 5 10 --valgrind
```

To run the program with a block size:

```bash
./gpu_transpose 12 10 1024
```

To run the program with a block size and with Valgrind for memory checking:

```bash
./gpu_transpose 12 10 1024 --valgrind
```

## Example to use bash
You can also use a bash script to run the program. The paramters are the same.
```
sbatch batch.sh <matrix_size> <number_of_runs> [<block_size>] [--valgrind]

<matrix_size>: Specify the size of the square matrices to be transposed. It will be the power of two, e.g. 3 will be a matrix of 8x8
<number_of_runs>: Specify the number of runs the program should execute.
<block_size> (optional): If provided, the transpose operation will be done by blocks, the block size must be a multiple of the matrix size.
--valgrind (optional): If provided, runs the program with Valgrind for memory checkin
```



To run the program with a matrix size of 16x16 and execute 10 runs:

```bash
sbatch batch.sh  4 10
```

To run the program with Valgrind for memory checking:

```bash
sbatch batch.sh  5 10 --valgrind
```

To run the program with a block size:

```bash
sbatch batch.sh  12 10 1024
```

To run the program with a block size and with Valgrind for memory checking:

```bash
sbatch batch.sh  12 10 1024 --valgrind
```


