#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define MAX 501

__global__ void setup_kernel(curandState ** state){
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int randid = x + (blockDim.x * gridDim.x * y);

	curand_init(420+69, randid, 0, &state[x][y]);
}

__global__ void update(short * inGrid[], short * outGrid[], int N, curandState ** rand_state){
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	//generate random number
	unsigned int randres = curand(rand_state[x][y]);
	randres = ((int) randres) % 500;

	int index = 0;
	short state = inGrid[x][y];
	//calculate index
	index += inGrid[(x-1) % N][(y-1) % N];
	index += inGrid[x][(y-1) % N];
	index += inGrid[(x-1) % N][y];
	index += inGrid[(x+1) % N][(y+1) % N];
	index += inGrid[(x+1) % N][(y-1) % N];
	index += inGrid[(x-1) % N][(y+1) % N];
	index += inGrid[(x+1) % N][y];
	index += inGrid[x][(y+1) % N];

	//find new state
	if(state == 0){
		if(randres == 0) state = 2;
		else if(index < 7) state = 0;	
		else if(index < 17) state = 1;
		else state = 3;
	}
	else if(state == 1){
		if(randres == 0) state = 3;
		else if(index < 1) state = 0;
		else state = 1;
	}
	else if(state == 2){
		if(randres % 5 < 2) state = 2;
		else if(randres % 5 < 4) state = 1;
		else state = 0;
	}
	else{
		if index < 10 state = 1;
		else state = 3;
	}
	
	//update relevant array
	outGrid[x][y] = state;
}


void print_grid(short* grid[], short* source_d, int N) {
	for(int y = 0; y < N; i++){
		for(int x = 0; x < N; x++){
			cudaMemcpy(grid[x][y], source_d[x][y], sizeof(short), cudaMemcpyDeviceToHost)
		}
	}
	
	for(int y = 0; y < N; y++){
		for(int x = 0; x < N; x++){
			printf("%d ", grid[x][y]);
		}
		printf("/n");
	}
}

int main(int argc, char * argv[]){
	
	if(argc < 4){
		printf("usage: ./simCuda <sidelength> <timesteps> <block divisor for each dimention>");
		exit(1);
	}
	
	//arguments(board and GPU dimensions)
	const int N = atoi(argv[1]);
	const int t = atoi(argv[2]);	
	const int common_divisor = atoi(argv[3]);

	if(N*N % 32){
		printf("Choose multiple of 32 threads for best results");
	}
	
	if(N % common_divisor){
		printf("Try again with divisor of sidelength");
		exit(1);
	} 	
	
	//*******************************************************************
	//****************************GRID SETUP*****************************
	//*******************************************************************
	int blockw = N/common_divisor;

	//blank grid (host)
	short * blankGrid[];

	//grid memory allocation (host)
	blankGrid = calloc(N, sizeof(short*));
	for(int i = 0; i < N, i++){
		blankGrid[i] = calloc(N, sizeof(short));
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
	for(int y = 0; y < N; y++){
		for(int x = 0; x < N; x++){
			cudaMemcpy(evenGrid_d[x][y], blankGrid[x][y], sizeof(short), cudaMemcpyHostToDevice);
			cudaMemcpy(oddGrid_d[x][y], blankGrid[x][y], sizeof(short), cudaMemcpyHostToDevice);
		}
	}



	//********************************************************************
	//***************************KERNEL CALLS*****************************
	//********************************************************************
	//declare dimentions of blocks and block arrangement
	dim3 BLOCK_ARRANGEMENT(common_divisor,common_divisor,1);
	dim3 BLOCK_SHAPE(blockw,blockw,1);

	//random number stuff (it's a headache)
	curandState * states_d[];
	cudaMalloc((void***) &states_d, N * sizeof(curandState*));
	for(int i = 0; i < N; i++){
		cudaMalloc((void**) &states_d[i], N * sizeof(curandState));
	}
	setup_kernel<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(states_d);


	//call updates for each timestep
	for(int i = 1; i < t+1; i++){
		if(i%2==0){
			update<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(evenGrid_d, oddGrid_d, N, states_d);
			print_grid(blankGrid, oddGrid_d, N);
		}
		else{
			update<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(oddGrid_d, evenGrid_d, N, states_d);
			print_grid(blankGrid, evenGrid_d, N);
		}
		sleep(1)
	}

	//cleanup
	for(int i = 0; i < N; i++){
		cudaFree(evenGrid_d[i]);
		cudaFree(oddGrid_d[i]);
		free(blankGrid[i]);
		cudaFree(states_d[i]);
	}
	cudaFree(states_d);
        cudaFree(evenGrid_d);
	cudaFree(oddGrid_d);
	free(blankGrid);
}
