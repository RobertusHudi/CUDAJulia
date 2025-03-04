#include <cuda.h>
#include <iostream>
#include <random>
#include <stdexcept>

__global__ void matrix_multiplication(int *a, int *b, int *c, int n) {
  int x_index = blockDim.x * blockIdx.x + threadIdx.x;
  int y_index = blockDim.y * blockIdx.y + threadIdx.y;

  int sum = 0;

  if (x_index < n && y_index < n) {
    for (int i = 0; i < n; i++) {
      sum += a[y_index * n + i] * b[i * n + x_index];
    }
    c[y_index * n + x_index] = sum;
  }
}

int main() {

  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_int_distribution<> distribution(0, 1000);

  int n = 4096; // Note: Large value
  int size = n * n;

  // Check for large size allocation
  try {
    // int *host_a = new int[size];
    // int *host_b = new int[size];
    // int *host_c = new int[size];
    // int *test = new int[size]{0}; // Initialize to zero

    int *host_a = (int*) malloc(size * sizeof(int));
    int *host_b = (int*) malloc(size * sizeof(int));
    int *host_c = (int*) malloc(size * sizeof(int));
    int *test = (int*) malloc(size * sizeof(int)); // Initialize to zero

    for (int i = 0; i < size; i++) {
      host_a[i] = distribution(engine);
      host_b[i] = distribution(engine);
    }

    int *device_a, *device_b, *device_c;

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    //cudaThreadSynchronize();

    cudaError_t err = cudaMalloc(&device_a, sizeof(int) * size);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory for device_a");
    }

    err = cudaMalloc(&device_b, sizeof(int) * size);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory for device_b");
    }

    err = cudaMalloc(&device_c, sizeof(int) * size);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory for device_c");
    }

    cudaMemcpy(device_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

    int N_THREADS = 32;
    int N_BLOCKS = (n + N_THREADS - 1) / N_THREADS;

    dim3 threads(N_THREADS, N_THREADS);
    dim3 blocks(N_BLOCKS, N_BLOCKS);

    matrix_multiplication<<<blocks, threads>>>(device_a, device_b, device_c, n);

    cudaMemcpy(host_c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    bool is_ok = true;

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          test[i * n + j] += host_a[n * i + k] * host_b[k * n + j];
        }
      }
    }

    for (int i = 0; i < size; i++) {
      if (host_c[i] != test[i]) {
        is_ok = false;
        break;
      }
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    delete[] test;

    if (is_ok) {
      std::cout << "OK" << std::endl;
    }

  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    return -1;
  } catch (const std::runtime_error &e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
