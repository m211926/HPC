#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void setup_kernel(curandState * state){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1234, idx, 0, &state[idx];
}

__global__ void update(short * inGrid[], short * outGrid[], int N, curandState * state, ){
	int y = blockIdx;
	int x = threadIdx.x;
	int index = 0;
	int state = inGrid[x][y];

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	//random number generation
	float randf = curand_uniform(state+idx);
	randf *= (

	//calculate index
	index += inGrid[(x-1) % N][(y-1) % N];
	index += inGrid[x][(y-1) % N];
	index += inGrid[(x-1) % N][y];
	index += inGrid[(x+1) % N][(y+1) % N];
	index += inGrid[(x+1) % N][(y-1) % N];
	index += inGrid[(x-1) % N][(y+1) % N];
	index += inGrid[(x+1) % N][y];
	index += inGrid[x][(y+1) % N];

	if(state == 0){
		
	}
}


void printGrid(short* grid[]) {
	grid

}

int main(int argc, char * argv[]){
	
	//arguments(board and GPU dimensions)
	const int N = atoi(argv[1]);
	const int t = atoi(argv[2]);	
	const int num_blocks = atoi(argv[3]);
	const int bw = atoi(argv[4]);

	//grids (host)
	short * evenGrid[];
	short * oddGrid[];

	//memory allocation (host)
	evenGrid = calloc(N, sizeof(short*));
	oddGrid = calloc(N, sizeof(short*));
	for(int i = 0; i < N, i++){
		evenGrid[i] = calloc(N, sizeof(short));
		oddGrid[i] = calloc(N, sizeof(short));
	}
	
	//grids (device)
	short * evenGrid_d[];
	short * oddGrid_d[];

	//memory allocation (device)
	cudaMalloc((void***) &evenGrid, N*sizeof(short*));
	cudaMalloc((void***) &oddGrid, N*sizeof*(short*));
	for(int i = 0; i < N; i++){
		cudaMalloc((void**) &evenGrid[i], N*sizeof(short));
		cudaMalloc((void**) &oddGrid[i], N*sizeof(short));
	}

	//transfer CPU contents to GPU (all zeroed out)
	cudaMemcpy(evenGrid_d, evenGrid, N*sizeof(short*), cudaMemcpyHostToDevice);
	cudaMemcpy(oddGrid_d, oddGrid, N*sizeof(short*), cudaMemcpyHostToDevice);
	for(int i = 0; i < N; i++){
		cudaMemcpy(evenGrid_d[i], evenGrid[i], N*sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(oddGrid_d[i], oddGrid[i], N*sizeof(short), cudaMemcpyHostToDevice);
	}

	//declare dimentions of blocks and block arrangement
	dim3 BLOCK_ARRANGEMENT(1,N,1);
	dim3 BLOCK_SHAPE(N,1,1);
	
	//call updates for each timestep
	for(int i = 1; i < t+1; i++){
		if(i%2==0){
			update<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(evenGrid_d, oddGrid_d);
		}
		else{
			update<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(oddGrid_d, evenGrid_d);
		}
	}
}
