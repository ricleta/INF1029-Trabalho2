// Ricardo Bastos Leta Vieira 2110526
// #---------------TODO-----------------#
// Doesn't handle cases where MAX_BLOCKS_PER_GRID < numBlocks for the matrix size
// I.e. using A = matrix_400x800 causes no errors; using B = matrix_1600x800 causes ilegal access error

#include <stdio.h>
#include <stdlib.h> 
#include <cuda_runtime.h>
#include "matrix_lib.h"

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS_PER_GRID 4096

__global__ void aux_matrix_matrix_mult(int a_height, int b_height, int c_width, float *a_rows_d, float *b_rows_d, float *c_rows_d)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = blockDim.x * gridDim.x;

    int qtd = (a_height + n_threads - 1) / n_threads;
    int n_linhas = qtd * thread_id;

    for (int row = 0; row < qtd; row++)
    {
        int a_index = (n_linhas + row) * b_height;
        int c_index = (n_linhas + row) * c_width;

        if (a_index <= a_height * b_height || c_index <= a_height * c_width)
        {
            float *iterA = a_rows_d + a_index;
            float *iterC = c_rows_d + c_index;

            for (int j = 0; j < b_height; j++)
            {
                int b_index = j * c_width;

                if (b_index <= b_height * c_width)
                {
                    float *iterB = b_rows_d + b_index;

                    for (int k = 0; k < c_width; k++)
                    {
                        if (a_index + j <= a_height * b_height || c_index + k <= a_height * c_width)
                        {
                            iterC[k] += iterB[k] * iterA[j];
                        }
                    }
                }
            }
        }
    }
}

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

    cudaError = cudaMalloc(&a_rows_d, a_size * sizeof(float));
    cudaError = cudaMalloc(&b_rows_d, b_size * sizeof(float));  
    cudaError = cudaMalloc(&c_rows_d, c_size * sizeof(float));

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc returned error %s (code %d) \n", cudaGetErrorString(cudaError), cudaError);
        exit(2);
    }

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

    // printf("Num blocks matrix_matrix_mult = %d\n", numBlocks);
    if (numBlocks > MAX_BLOCKS_PER_GRID)
    {
        printf("Num blocks = %d\n", MAX_BLOCKS_PER_GRID);
        numBlocks = MAX_BLOCKS_PER_GRID;
    }

    // int blockSize = 256;
    // int numBlocks = 4096;
    
    printf("Num blocks matrix_matrix_mult = %d\n", numBlocks);
    
    aux_matrix_matrix_mult<<<numBlocks, blockSize>>>(a->height, b->height, c->width, a_rows_d, b_rows_d, c_rows_d);

    cudaDeviceSynchronize();
    
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

// Funcao auxiliar para multiplicacao de matriz por escalar chamada pelos threads no caso do tamanho da matrix ser menor que on numero de threads
__global__ void aux_scalar_mult(float *rows, int size, float scalar)
{
    int n_threads = gridDim.x * blockDim.x;                // number of threads created
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // number of thread

    int qtd = (size + n_threads - 1) / n_threads;
    int ini = qtd * thread_id;

    // printf("n_threads = %d; qtd = %d; size = %d\n", n_threads, qtd, size);

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

int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
    float *device_temp;
    int matrix_size = matrix->height * matrix->width;
    cudaError_t cudaError;

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (matrix_size + blockSize - 1) / blockSize;

    printf("Num blocks = %d\n", numBlocks);
    if (numBlocks > MAX_BLOCKS_PER_GRID)
    {
        numBlocks = MAX_BLOCKS_PER_GRID;
        printf("Num blocks = %d\n", numBlocks);
    }

    if (numBlocks * blockSize < matrix_size)
    {
        printf("numBlocks * blockSize = %d\n", numBlocks * blockSize);

        int part_size = matrix->width;
        int n_parts = matrix->height;
        cudaError = cudaMalloc(&device_temp, part_size * sizeof(float));

        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc returned error %s (code %d) \n", cudaGetErrorString(cudaError), cudaError);
            exit(2);
        }
        
        float *current = matrix->rows;
        for (int i = 0; i < n_parts; i++, current+=part_size)
        {
            // printf("i = %d; n_parts = %d; part_size = %d; i * part_size = %d\n", i, n_parts, part_size, i * part_size);
            cudaError = cudaMemcpy(device_temp, current, part_size * sizeof(float), cudaMemcpyHostToDevice);
        
            if (cudaError != cudaSuccess)
            {
                fprintf(stderr, "cudaMemCpy matrix->rows -> device_temp returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
                cudaFree(device_temp);
                exit(3);
            }

            aux_scalar_mult<<<numBlocks, blockSize>>>(device_temp, part_size, scalar_value);
            cudaDeviceSynchronize();

            cudaError = cudaMemcpy(current, device_temp, part_size * sizeof(float), cudaMemcpyDeviceToHost);

            if (cudaError != cudaSuccess)
            {
                fprintf(stderr, "cudaMemCpy device_temp -> matrix->rows returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
                cudaFree(device_temp);
                exit(4);
            }
        }

        cudaFree(device_temp);
    }
    else
    {
        cudaError = cudaMalloc(&device_temp, matrix_size * sizeof(float));

        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc returned error %s (code %d) \n", cudaGetErrorString(cudaError), cudaError);
            exit(2);
        }

        cudaError = cudaMemcpy(device_temp, matrix->rows, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
        
        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "cudaMemCpy matrix->rows -> device_temp returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
            cudaFree(device_temp);
            exit(3);
        }

        printf("numBlocks * blockSize = %d\n", numBlocks * blockSize);
        aux_scalar_mult<<<numBlocks, blockSize>>>(device_temp, matrix_size, scalar_value);
        cudaDeviceSynchronize();

        cudaError = cudaMemcpy(matrix->rows, device_temp, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

        if (cudaError != cudaSuccess)
        {
            fprintf(stderr, "cudaMemCpy device_temp -> matrix->rows returned error %s (code %d) line %d\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
            cudaFree(device_temp);
            exit(4);
        }

        cudaFree(device_temp);
    }

    return 1;
}
