#include <cuda.h>
#include <iostream>
#include <random>
#include <stdexcept>

__global__ void matrix_multiplication(int *a, int *b, int *c, int n, int chunk_size, int start_row, int start_col) {
  int x_index = blockDim.x * blockIdx.x + threadIdx.x;
  int y_index = blockDim.y * blockIdx.y + threadIdx.y;

  if (x_index < chunk_size && y_index < chunk_size) {
    int sum = 0;
    int global_x_index = start_col + x_index;
    int global_y_index = start_row + y_index;

    if (global_x_index < n && global_y_index < n) {
      for (int i = 0; i < n; i++) {
        sum += a[global_y_index * n + i] * b[i * n + global_x_index];
      }
      c[global_y_index * n + global_x_index] = sum;
    }
  }
}

int main() {
  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_int_distribution<> distribution(-1000, 1000);

  int n = 65535; // Large value
  int chunk_size = 1024; // Process in smaller chunks
  int size = n * n;

  // Use unified memory
  int *a, *b, *c;

  cudaMallocManaged(&a, size * sizeof(int));
  cudaMallocManaged(&b, size * sizeof(int));
  cudaMallocManaged(&c, size * sizeof(int));

  for (int i = 0; i < size; i++) {
    a[i] = distribution(engine);
    b[i] = distribution(engine);
  }

  int N_THREADS = 32;
  int N_BLOCKS = (chunk_size + N_THREADS - 1) / N_THREADS;

  dim3 threads(N_THREADS, N_THREADS);
  dim3 blocks(N_BLOCKS, N_BLOCKS);

  // Process matrix in chunks
  for (int i = 0; i < n; i += chunk_size) {
    for (int j = 0; j < n; j += chunk_size) {
      matrix_multiplication<<<blocks, threads>>>(a, b, c, n, chunk_size, i, j);
      cudaDeviceSynchronize(); // Wait for the kernel to complete
    }
  }

  // Validate the result
  bool is_ok = true;

  int *test = new int[size]{0}; // Initialize to zero

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        test[i * n + j] += a[n * i + k] * b[k * n + j];
      }
    }
  }

  for (int i = 0; i < size; i++) {
    if (c[i] != test[i]) {
      is_ok = false;
      break;
    }
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  delete[] test;

  if (is_ok) {
    std::cout << "OK" << std::endl;
  }

  return 0;
}
