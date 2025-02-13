#include <stdio.h>
#include <random>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vectorAdd(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int n = 1e6;
    size_t size = n * sizeof(float);

    // float *h_a, *h_b, *h_c; //host
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    for (int i = 0; i < n; i++){
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
    }

    float *d_a, *d_b, *d_c; //device

    // should define error checking here in case OOM
    // cudaMalloc(&d_a, size); cudaMalloc(&d_b, size); cudaMalloc(&d_c, size);
    CUDA_CHECK(cudaMalloc(&d_a, size));CUDA_CHECK(cudaMalloc(&d_b, size));CUDA_CHECK(cudaMalloc(&d_c, size));

    // copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //lauch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock >>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++){
        printf("%.2f + %.2f = %.2f \n", h_a[i], h_b[i], h_c[i]);
    }

    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b);cudaFree(d_c);

    return 0;
}