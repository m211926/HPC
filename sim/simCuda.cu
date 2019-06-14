#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


/**
 * This kernel essentially serves as a "srand(seed)" on the GPU
 */
__global__ void setup_kernel(curandState * state){
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int id = x + (blockDim.x * gridDim.x * y);

	curand_init(420+69, id, 0, &state[id]);
}

/**
 * Executes one timestep on a thread
 */
__global__ void update(short * inGrid, short * outGrid, curandState * rand_state){
	short y = (blockIdx.y * blockDim.y) + threadIdx.y;
	short x = (blockIdx.x * blockDim.x) + threadIdx.x;
	short N = blockDim.x * gridDim.x;
	short id = x + (N * y);

	//generate random number
	unsigned int randres = curand(&rand_state[id]);
	randres = (short) (((int) randres) % 300);

	short index = 0;
	short state = inGrid[id];
	
	//calculate index
	index += inGrid[((x-1) % N) + N*y];
	index += inGrid[((x+1) % N) + N*y];
	index += inGrid[x + N*((y-1) % N)];
	index += inGrid[x - N*((y+1) % N)];
	index += inGrid[((x+1) % N) + N*((y-1) % N)];
	index += inGrid[((x-1) % N) + N*((y-1) % N)];
	index += inGrid[((x+1) % N) + N*((y+1) % N)];
	index += inGrid[((x-1) % N) + N*((y+1) % N)];
	
	//find new state
	if(state == 0){
		if(randres == 0) state = 2;
		else if(index < 7) state = 0;	
		else if(index < 17) state = 1;
		else state = 3;
	}
	else if(state == 1){
		if(randres == 0 || index > 16) state = 3;
		else if(index < 1) state = 0;
		else state = 1;
	}
	else if(state == 2){
		if(randres % 5 < 2) state = 2;
		else if(randres % 5 < 4) state = 1;
		else state = 0;
	}
	else if(state == 3){
		if(index > 9) state = 3;
		else state = 1;
	}
	
	//update relevant array
	outGrid[id] = state;
}


void print_grid(short * grid, short* source_d, int N) {
	cudaMemcpy(grid, source_d, N*N*sizeof(short), cudaMemcpyDeviceToHost);
	
	for(int y = 0; y < N; y++){
		for(int x = 0; x < N; x++){
			printf("%d ", grid[x + N*y]);
		}
		printf("\n");
	}
	
	return;
}

int main(int argc, char * argv[]){
	
	if(argc < 4){
		printf("usage: ./simCuda <sidelength> <timesteps> <block divisor for each dimention>\n");
		exit(1);
	}
	
	//arguments(board and GPU dimensions)
	const int N = atoi(argv[1]);
	const int t = atoi(argv[2]);	
	const int common_divisor = atoi(argv[3]);

	if(N*N % 32){
		printf("Choose multiple of 32 on sides for best results\n");
	}
	
	if(N % common_divisor){
		printf("Try again with divisor of sidelength\n");
		exit(1);
	} 	
	
	printf("begin grid setup\n");	
	//*******************************************************************
	//****************************GRID SETUP*****************************
	//*******************************************************************
	int blockw = N/common_divisor;

	//blank grid (host)
	short * blankGrid;

	//grid memory allocation (host)
	blankGrid = (short*) calloc(N*N, sizeof(short));
	printf("grid host allocation complete\n");

	//grids (device)
	short * evenGrid_d;
	short * oddGrid_d;

	//memory allocation (device)
	cudaMalloc((void**) &evenGrid_d, N*N*sizeof(short));
	cudaMalloc((void**) &oddGrid_d, N*N*sizeof(short));

	printf("even/odd grid device allocation complete\n");
	
	//transfer CPU contents to GPU (all zeroed out)
	for(int y = 0; y < N; y++){
		for(int x = 0; x < N; x++){
			cudaMemcpy(evenGrid_d, blankGrid, N*N*sizeof(short), cudaMemcpyHostToDevice);
			cudaMemcpy(oddGrid_d, blankGrid, N*N*sizeof(short), cudaMemcpyHostToDevice);
		}
	}
	
	printf("device grid initialization complete\n");


	//*******************************************************************
	//***************************KERNEL CALLS****************************
	//*******************************************************************
		
	printf("begin kernel calls\n");
	//declare dimentions of blocks and block arrangement
	dim3 BLOCK_ARRANGEMENT(common_divisor,common_divisor,1);
	dim3 BLOCK_SHAPE(blockw,blockw,1);

	//random number stuff (it's a headache)
	
	curandState * states_d;
	cudaMalloc((void**) &states_d, N*N*sizeof(curandState));
	printf("device random memory allocation complete\n");

	setup_kernel<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(states_d);
	printf("device random initialization complete, begin sim\n");	

	//call updates for each timestep
	for(int i = 1; i < t+1; i++){
		if(i%2==0){
			update<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(evenGrid_d, oddGrid_d, states_d);
			print_grid(blankGrid, oddGrid_d, N);
			for(int i = 0; i < N; i++) printf("*");
			printf("\n");
		}
		else{
			update<<<BLOCK_ARRANGEMENT,BLOCK_SHAPE>>>(oddGrid_d, evenGrid_d, states_d);
			print_grid(blankGrid, evenGrid_d, N);
			for(int i = 0; i < N; i++) printf("*");
			printf("\n");
		}
		//usleep(300000);
	}

	//*******************************************************************
	//*****************************CLEANUP*******************************
	//*******************************************************************
	
	cudaFree(states_d);
        cudaFree(evenGrid_d);
	cudaFree(oddGrid_d);
	free(blankGrid);

	return 0;
}
