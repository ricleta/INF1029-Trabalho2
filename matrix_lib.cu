// Ricardo Bastos Leta Vieira 2110526

#include <stdio.h>
#include <stdlib.h> 
#include <cuda_runtime.h>
#include "matrix_lib.h"

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS_PER_GRID 4096

// Kernel function for matrix-matrix multiplication
__global__ void aux_matrix_matrix_mult(int a_height, int b_height, int c_width, float *a_rows_d, float *b_rows_d, float *c_rows_d)
{
    // Calculate the thread ID and the total number of threads
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = blockDim.x * gridDim.x;

    // Calculate the number of rows each thread will process
    int qtd = (a_height + n_threads - 1) / n_threads;
    int n_linhas = qtd * thread_id;

    // Iterate over the assigned rows of matrix A
    for (int row = 0; row < qtd; row++)
    {
        // Calculate the indices for accessing the elements of matrices A and C
        int a_index = (n_linhas + row) * b_height;
        int c_index = (n_linhas + row) * c_width;

        // Check if the indices are within the valid range
        if (a_index <= a_height * b_height && c_index <= a_height * c_width)
        {
            // Get the pointers to the current elements of matrices A and C
            float *iterA = a_rows_d + a_index;
            float *iterC = c_rows_d + c_index;

            // Iterate over matrix B
            for (int j = 0; j < b_height; j++)
            {
                // Calculate the index for accessing the elements of matrix B
                int b_index = j * c_width;

                // Check if the index is within the valid range
                if (b_index <= b_height * c_width)
                {
                    // Get the pointers to the current elements of matrices B and A_j
                    float *iterB = b_rows_d + b_index;
                    float *iterA_j = iterA + j;

                    // Perform the matrix multiplication for the current row and column
                    for (int k = 0; k < c_width; k++)
                    {
                        // Check if the indices are within the valid range
                        if (a_index + j <= a_height * b_height && c_index + k <= a_height * c_width)
                        {
                            *iterC++ += *iterB++ * *iterA_j;
                        }
                    }
                    iterC -= c_width; // Reset iterC to the beginning of the current row in C
                }
            }
        }
    }
}

// Function for matrix-matrix multiplication
int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c)
{
    if (a->width != b->height)
    {
        fprintf(stderr, "Impossivel multplicar matrizes de tamanhos imcompativeis\n");
        exit(1);
    }

    float *a_rows_d;
    float *b_rows_d;
    float *c_rows_d;

    int a_size = a->height * a->width;
    int b_size = b->height * b->width;
    int c_size = c->height * c->width;
    
    cudaError_t cudaError;

    // Allocate device memory for matrices A, B, and C
    cudaError = cudaMalloc(&a_rows_d, a_size * sizeof(float));
    cudaError = cudaMalloc(&b_rows_d, b_size * sizeof(float));  
    cudaError = cudaMalloc(&c_rows_d, c_size * sizeof(float));

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc returned error %s (code %d) \n", cudaGetErrorString(cudaError), cudaError);
        exit(2);
    }

    // Copy matrices A, B, and C from host to device
    cudaError = cudaMemcpy(a_rows_d, a->rows, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaError = cudaMemcpy(b_rows_d, b->rows, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaError = cudaMemcpy(c_rows_d, c->rows, c_size * sizeof(float), cudaMemcpyHostToDevice);
    
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMemCpy host -> device returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        
        cudaFree(a_rows_d);
        cudaFree(b_rows_d);
        cudaFree(c_rows_d);
        
        exit(3);
    }

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (c_size + blockSize - 1) / blockSize;

    if (numBlocks > MAX_BLOCKS_PER_GRID)
    {
        numBlocks = MAX_BLOCKS_PER_GRID;
    }
    
    // Launch the kernel for matrix-matrix multiplication
    aux_matrix_matrix_mult<<<numBlocks, blockSize>>>(a->height, b->height, c->width, a_rows_d, b_rows_d, c_rows_d);

    cudaDeviceSynchronize();
    
    // Copy matrix C from device to host
    cudaError = cudaMemcpy(c->rows, c_rows_d,  c_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMemCpy device -> host returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        
        cudaFree(a_rows_d);
        cudaFree(b_rows_d);
        cudaFree(c_rows_d);
        
        exit(4);
    }

    cudaFree(a_rows_d);
    cudaFree(b_rows_d);
    cudaFree(c_rows_d);
        
    return 1;
}

// Kernel function for scalar-matrix multiplication
__global__ void aux_scalar_mult(float *rows, int size, float scalar)
{
    int n_threads = gridDim.x * blockDim.x;                // number of threads created
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // number of thread

    int qtd = (size + n_threads - 1) / n_threads;
    int ini = qtd * thread_id;

    // Iterator depends on the thread running, adding ini indicates the appropriate start for the thread
    float *p = rows + ini;

    // there might be more threads than needed, this ensures no illegal memory access
    if (thread_id < size)
    {
        // Perform scalar multiplication on the assigned portion of the matrix
        for (int i = 0; i < qtd; i++)
        {
            *p++ *= scalar;        
        }
    }
}

// Function for scalar-matrix multiplication
int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
    float *device_temp;
    int matrix_size = matrix->height * matrix->width;
    cudaError_t cudaError;

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (matrix_size + blockSize - 1) / blockSize;

    if (numBlocks > MAX_BLOCKS_PER_GRID)
    {
        numBlocks = MAX_BLOCKS_PER_GRID;
    }

    // Allocate device memory for temporary matrix
    cudaError = cudaMalloc(&device_temp, matrix_size * sizeof(float));

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc returned error %s (code %d) \n", cudaGetErrorString(cudaError), cudaError);
        exit(2);
    }

    // Copy matrix from host to device
    cudaError = cudaMemcpy(device_temp, matrix->rows, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
        
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMemCpy matrix->rows -> device_temp returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        cudaFree(device_temp);
        exit(3);
    }

    // Launch the kernel for scalar-matrix multiplication
    aux_scalar_mult<<<numBlocks, blockSize>>>(device_temp, matrix_size, scalar_value);
    cudaDeviceSynchronize();

    // Copy matrix from device to host
    cudaError = cudaMemcpy(matrix->rows, device_temp, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMemCpy device_temp -> matrix->rows returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        cudaFree(device_temp);
        exit(4);
    }

    cudaFree(device_temp);

    return 1;
}