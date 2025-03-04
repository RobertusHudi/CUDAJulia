#include <stdio.h>
#include <cuda_runtime.h>

#define N 16384
//#define ll long long 

__global__ void matrix_mul(int *a,int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    if (i < n && j < n) {
        for (int k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}

int main() {
    int n = N;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * n * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
            if(i==j)b[i * n + j] = 1;
        }

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++)
    //         printf("%d ", b[i * n + j]);
    //     printf("\n\n");
    // }

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++)
    //         printf("%d ", a[i * n + j]);
    //     printf("\n\n");
    // }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int N_THREADS = 32;
    int N_BLOCKS = (n + N_THREADS - 1) / N_THREADS;

    dim3 threads(N_THREADS, N_THREADS);
    dim3 blocks(N_BLOCKS, N_BLOCKS);


    // dim3 blockSize(N, N);
    // dim3 gridSize((n + N - 1) / N, (n + N - 1) / N);
    matrix_mul<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++)
    //         printf("%d ", c[i * n + j]);
    //     printf("\n");
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
