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

__global__ void matrixAdd(float* A, float* B, float* C, int width, int height){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width){
        C[row * width + col] = A[row * width + col] + B[row * width + col];
    }
}

void print_matrix(float* A, int width, int height){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            printf("%.2f", A[i*width+j]);
        }
        printf("\n");
    }
}

int main(){
    int height = 5;
    int width = 6;

    float* h_A = new float[height*width];
    float* h_B = new float[height*width];
    float* h_C = new float[height*width];

    //  initialization

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int idx = i * width + j;
            h_A[idx] = 1.0f;
            h_B[idx] = 2.0f;
        }
    }
    printf("Matrix A\n");
    print_matrix(h_A, width, height);
    printf("Matrix B\n");
    print_matrix(h_B, width, height);


    float *d_A, *d_B, *d_C; //device
    size_t size = width * height * sizeof(float);

    // should define error checking here in case OOM
    CUDA_CHECK(cudaMalloc(&d_A, size));CUDA_CHECK(cudaMalloc(&d_B, size));CUDA_CHECK(cudaMalloc(&d_C, size));

    // copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //lauch kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
        (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y
    );
    matrixAdd<<<numBlocks, numThreadsPerBlock >>>(d_A, d_B, d_C, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Matrix C\n");
    print_matrix(h_C, width, height);

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B);cudaFree(d_C);

    return 0;
}