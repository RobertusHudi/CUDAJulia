#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace std::chrono;

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
  auto start = high_resolution_clock::now();

  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_int_distribution<> distribution(0, 1000);

  int n = 32768;
  int size = n * n;

  int *host_a = new int[size];
  int *host_b = new int[size];
  int *host_c = new int[size];

  for (int i = 0; i < size; i++) {
    host_a[i] = distribution(engine);
    host_b[i] = distribution(engine);
  }

  int *device_a, *device_b, *device_c;

  cudaMalloc(&device_a, sizeof(int) * size);
  cudaMalloc(&device_b, sizeof(int) * size);
  cudaMalloc(&device_c, sizeof(int) * size);

  cudaMemcpy(device_a, host_a, size * sizeof(int),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  cudaMemcpy(device_b, host_b, size * sizeof(int),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  int N_THREADS = 32;
  int N_BLOCKS = (n + N_THREADS - 1) / N_THREADS;

  dim3 threads(N_THREADS, N_THREADS);
  dim3 blocks(N_BLOCKS, N_BLOCKS);

  matrix_multiplication<<<blocks, threads>>>(device_a, device_b, device_c, n);

  cudaMemcpy(host_c, device_c, size * sizeof(int),
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  bool is_ok = true;

  int *test = (int*) malloc(size * sizeof(int)); 

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

  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
 // auto  timeInSecond = duration.count(); 

    std::cout << "Total time taken : "
         << duration.count()  << " microseconds" << std::endl;
  return 0;
}
