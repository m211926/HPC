#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

// Defines
#define GridWidth 60
#define BlockWidth 128

// Variables for host and device vectors.

__global__ void AddVectors(float* A, float* B, float *C, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<=N)
	   C[idx] = A[idx] + B[idx];
}
// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
	float* h_A; 
	float* h_B; 
	float* h_C; 
	float* d_A; 
	float* d_B; 
	float* d_C;

	
    int N= GridWidth * BlockWidth;    
	printf("N=%d\n",N);
    size_t size = N * sizeof(float);
    
    dim3 dimGrid(GridWidth);                    
    dim3 dimBlock(BlockWidth);                 

    h_A = (float*)malloc(size);    
    h_B = (float*)malloc(size);   
    h_C = (float*)malloc(size);    

    cudaMalloc((void**)&d_A, size);    
    cudaMalloc((void**)&d_B, size);    
    cudaMalloc((void**)&d_C, size);
    // Initialize host vectors h_A and h_B
    
    for(int i=0; i<N; ++i)
    {
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }

    // Copy host vectors h_A and h_B to device vectores d_A and d_B
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);    
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
    AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,N);    
    cudaThreadSynchronize();   
     
    // Copy result from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    
    
    
    
    for(int i=0; i<N; ++i)
    {
     printf("%f\n",h_C[i]);   
    }					     
    // Free device vectors    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory    
    free(h_A);
    free(h_B);
    free(h_C);
}
