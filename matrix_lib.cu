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

__global__ void aux_matrix_matrix_mult(int a_height, int b_height, int b_width, int c_width, float *a_rows_d, float *b_rows_d, float *c_rows_d)
{
    // int n_threads = gridDim.x * blockDim.x;                // number of threads created
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // number of thread
    // int threads_per_block = blockDim.x;
    // int total_threads = gridDim.x * threads_per_block;
    // int elements_per_thread = (a_height * b_width) / total_threads; // Adjust based on workload preferences

    int iniA = a_height * thread_id;

    float *iniB = b_rows_d;
    float *iniC = c_rows_d;
    float *iterA = a_rows_d + iniA;
    float *iterB = b_rows_d;
    float *iterC = c_rows_d;

    // int i = thread_id * elements_per_thread; i < min((thread_id + 1) * elements_per_thread, a_height); i++
    for (int i = iniA; i < a_height; i++)
    {
        iterB = iniB;

        for (int j = 0; j < b_height; j++)
        {
            iterC = iniC + i * c_width;

            for (int k = 0; k < c_width; k++)
            {
                *iterC++ += *iterB++ * *iterA;
            }

            iterA++;
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
    
    printf("a_size = %d; b_size = %d; c_size = %d\n", a_size, b_size, c_size);
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

    printf("Num blocks matrix_matrix_mult = %d\n", numBlocks);
    if (numBlocks > MAX_BLOCKS_PER_GRID)
    {
        printf("Num blocks = %d\n", MAX_BLOCKS_PER_GRID);
        numBlocks = MAX_BLOCKS_PER_GRID;
    }

    printf("Num blocks matrix_matrix_mult = %d\n", numBlocks);
    
    aux_matrix_matrix_mult<<<numBlocks, blockSize>>>(a->height, b->height, b->width, c->width, a_rows_d, b_rows_d, c_rows_d);

    cudaDeviceSynchronize();
    
    cudaError = cudaMemcpy(c->rows, c_rows_d,  c_size * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < c_size; i++) {
    //     if (c->rows[i] != 0) {
    //         int row = i / c->width;
    //         int col = i % c->width;
    //         printf("Element at index (%d, %d) is %f\n", row, col, c->rows[i]);
    //     }
    // }
    
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

// Funcao auxiliar para multiplicacao de matriz por escalar chamada pelas threads
__global__ void aux_scalar_mult(float *rows, int height, int width, float scalar)
{
    int n_threads = gridDim.x * blockDim.x;                // number of threads created
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // number of thread
    int matrix_size = height * width;

    int qtd = (matrix_size + n_threads - 1) / n_threads;
    int ini = qtd * thread_id;

    // printf("n_threads = %d; qtd = %d; matrix_size = %d\n", n_threads, qtd, matrix_size);

    // Iterator depends on the thread running, adding ini indicates the appropriate start for the thread
    float *p = rows + ini;

    // Perform scalar multiplication on the assigned portion of the matrix
    for (int i = 0; i < qtd; i++)
    {
        *p++ *= scalar;        
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
    float *device_temp;
    int matrix_size = matrix->height * matrix->width;
    cudaError_t cudaError;

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

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (matrix_size + blockSize - 1) / blockSize;

    printf("Num blocks = %d\n", numBlocks);
    if (numBlocks > MAX_BLOCKS_PER_GRID)
    {
        printf("Num blocks = %d\n", MAX_BLOCKS_PER_GRID);
        numBlocks = MAX_BLOCKS_PER_GRID;
    }

    printf("Num blocks = %d\n", numBlocks);
    
    aux_scalar_mult<<<numBlocks, blockSize>>>(device_temp, matrix->height, matrix->width, scalar_value);

    cudaDeviceSynchronize();
    
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