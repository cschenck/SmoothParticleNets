
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_BLOCKS 65535
#define SQRT3 1.73205f

//*
__global__
void kernel_countneighbors_n2conv(float3* clocs, int batch_size, int N, float radius, 
	float dilation, int num_blocks, uint32_t* cncount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N*batch_size*num_blocks; i += stride)
    {
    	int b = i/(N*num_blocks);
    	int n = (i % (N*num_blocks))/num_blocks;
    	int block = i % num_blocks;
    	float3 r = clocs[b*N + n];
    	int block_size = ceilf(1.0f*N/num_blocks);
    	int start = b*N + block*block_size;
    	for(int j = start; j < start + block_size && j < (b + 1)*N; ++j)
    	{
    		float3 r2 = clocs[j];
    		float d = (r.x - r2.x)*(r.x - r2.x) +
					  (r.y - r2.y)*(r.y - r2.y) +
					  (r.z - r2.z)*(r.z - r2.z);
			if(d > (radius + dilation*SQRT3)*(radius + dilation*SQRT3))
				continue;
			for(int ii = -1; ii <= 1; ++ii)
			{
				for(int jj = -1; jj <= 1; ++jj)
				{
					for(int kk = -1; kk <= 1; ++kk)
					{
						d = (r.x + ii*dilation - r2.x)*(r.x + ii*dilation - r2.x) +
							(r.y + jj*dilation - r2.y)*(r.y + jj*dilation - r2.y) +
							(r.z + kk*dilation - r2.z)*(r.z + kk*dilation - r2.z);
						if(d < radius*radius)
							atomicAdd(cncount + (b*N + n)*27 + (ii + 1)*9 + (jj + 1)*3 + (kk + 1), 1);
					}
				}
			}
    	}
    }
}
/*/
__global__
void kernel_countneighbors_n2conv(float3* clocs, int batch_size, int N, float radius, 
	float dilation, int block_size, uint32_t* cncount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N*batch_size; i += stride)
    {
    	int b = i/N;
    	float3 r = clocs[i];
    	for(int j = b*N; j < (b + 1)*N; ++j)
    	{
    		float3 r2 = clocs[j];
    		float d = (r.x - r2.x)*(r.x - r2.x) +
					  (r.y - r2.y)*(r.y - r2.y) +
					  (r.z - r2.z)*(r.z - r2.z);
			if(d > (radius + dilation*SQRT3)*(radius + dilation*SQRT3))
				continue;
			for(int ii = -1; ii <= 1; ++ii)
			{
				for(int jj = -1; jj <= 1; ++jj)
				{
					for(int kk = -1; kk <= 1; ++kk)
					{
						d = (r.x + ii*dilation - r2.x)*(r.x + ii*dilation - r2.x) +
							(r.y + jj*dilation - r2.y)*(r.y + jj*dilation - r2.y) +
							(r.z + kk*dilation - r2.z)*(r.z + kk*dilation - r2.z);
						if(d < radius*radius)
							atomicAdd(cncount + i*27 + (ii + 1)*9 + (jj + 1)*3 + (kk + 1), 1);
					}
				}
			}
    	}
    }
}
//*/


__global__
void kernel_countneighbors_n2(float3* clocs, int batch_size, int N, float radius, 
	uint32_t* cncount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N*batch_size; i += stride)
    {
    	int b = i/N;
    	float3 r = clocs[i];
    	for(int j = b*N; j < (b + 1)*N; ++j)
    	{
    		float3 r2 = clocs[j];
    		float d = (r.x - r2.x)*(r.x - r2.x) +
					  (r.y - r2.y)*(r.y - r2.y) +
					  (r.z - r2.z)*(r.z - r2.z);
		    if(d < radius*radius){
				atomicAdd(cncount + i, 1);
		    }
    	}
    }
}

int test_n2(void)
{
	printf("\n=============== TEST N2 ===========\n");
	// constants
	const int N = 8000;
	const int BATCH_SIZE = 32;
	const int NUM_BLOCKS = 4;
	const float RADIUS = 0.1f;
	const float DILATION = 0.05f;

	// init random
	srand(time(NULL));

	// setup the particle locations
	float* locs = new float[BATCH_SIZE*N*3];
	for(int i = 0; i < BATCH_SIZE*N*3; ++i)
		locs[i] = 1.0f*rand()/RAND_MAX;
	float3* clocs;
	cudaMalloc((void**)&clocs, sizeof(float3)*BATCH_SIZE*N);
	cudaMemcpy(clocs, locs, sizeof(float)*BATCH_SIZE*N*3, cudaMemcpyHostToDevice); 
	

	

	/******** PERFORMING NEIGHBOR COUNT N2 *********/
	// setup memory for neighbor count
	uint32_t* cncount;
	cudaMalloc((void**)&cncount, sizeof(uint32_t)*BATCH_SIZE*N);
	cudaMemset(cncount, 0, sizeof(uint32_t)*BATCH_SIZE*N);

	// Compute the bloc and thread counts
	int nops = BATCH_SIZE*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    // setup the timer
    clock_t start, stop;
    start = clock();

    // Run the kernel
	kernel_countneighbors_n2<<<blocks, threads>>>(clocs, BATCH_SIZE, N, RADIUS, cncount);
	cudaDeviceSynchronize();

	// Time it
	stop = clock();
	printf("Time to do neighbor lookup N2: %7.2fms\n", 1000.0f*(stop - start)/CLOCKS_PER_SEC);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in kernel_countneighbors_n2: %s\n", cudaGetErrorString(err));
		return 1;
	}

	// Verify the counds
	uint32_t* ncount = new uint32_t[BATCH_SIZE*N];
	cudaMemcpy(ncount, cncount, sizeof(uint32_t)*BATCH_SIZE*N, cudaMemcpyDeviceToHost); 
	double sum = 0.0f;
	for(int b = 0; b < BATCH_SIZE; ++b)
	{
		for(uint32_t n = 0; n < N; ++n)
		{
			sum += ncount[b*N + n];
		}
	}
	printf("Average neighbor count: %lf\n", sum/(BATCH_SIZE*N));


	/******** PERFORMING NEIGHBOR COUNT CONV *********/
	// setup memory for neighbor count
	uint32_t* cccount;
	cudaMalloc((void**)&cccount, sizeof(uint32_t)*BATCH_SIZE*N*27);
	cudaMemset(cccount, 0, sizeof(uint32_t)*BATCH_SIZE*N*27);

	// Compute the bloc and thread counts
	nops = BATCH_SIZE*N*NUM_BLOCKS;
    numBlocks = min((int)MAX_BLOCKS, (int)ceil(nops * (1.0/256)));
    blocks = dim3(numBlocks);
    threads = dim3(256);

    // setup the timer
    start = clock();

    // Run the kernel
	kernel_countneighbors_n2conv<<<blocks, threads>>>(clocs, BATCH_SIZE, N, RADIUS, 
		DILATION, NUM_BLOCKS, cncount);
	cudaDeviceSynchronize();

	// Time it
	stop = clock();
	printf("Time to do neighbor lookup conv: %7.2fms\n", 1000.0f*(stop - start)/CLOCKS_PER_SEC);

	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in kernel_countneighbors_n2conv: %s\n", cudaGetErrorString(err));
		return 1;
	}

	// Verify the counds
	uint32_t* nccount = new uint32_t[BATCH_SIZE*N*27];
	cudaMemcpy(nccount, cccount, sizeof(uint32_t)*BATCH_SIZE*N*27, cudaMemcpyDeviceToHost); 
	sum = 0.0f;
	for(int b = 0; b < BATCH_SIZE; ++b)
	{
		for(uint32_t n = 0; n < N; ++n)
		{
			for(int i = 0; i < 27; ++i)
				sum += nccount[b*N*27 + n*27 + i];
		}
	}
	printf("Average neighbor count: %lf\n", sum/(BATCH_SIZE*N*27));


	return 0;

    
}


int main(int argc, char** argv)
{
	// return test_hashgrid();
	return test_n2();
}






























__device__ 
int atomicListAdd(uint32_t *list, uint32_t value, int max_length)
{
    uint32_t old;
    int i = 0;
    do
    {
    	while(true)
    	{
    		if(i >= max_length)
	    		return -1;
	    	else if(i - 1 >= 0 && list[i-1] == UINT32_MAX)
	    		--i;
	    	else if(list[i] != UINT32_MAX)
	    		++i;
	    	else
	    		break;
    	}
		old = atomicCAS(list + i, UINT32_MAX, value);
	} while(old != UINT32_MAX);
	return i;
}

__global__
void kernel_assignhash(float3* clocs, uint32_t* chash, int N, int batch_size, int xdim,
	int ydim, int zdim, int max_neighbors, uint8_t* failed_flag, float radius)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N*batch_size; i += stride)
    {
    	int b = i/N;
    	int n = i % N;
    	float3 r = clocs[i];
		int xmin = max((int)0, (int)((r.x - radius)*xdim));
		int ymin = max((int)0, (int)((r.y - radius)*ydim));
		int zmin = max((int)0, (int)((r.z - radius)*zdim));
		int xmax = min(xdim, (int)((r.x + radius)*xdim) + 1);
		int ymax = min(ydim, (int)((r.y + radius)*ydim) + 1);
		int zmax = min(zdim, (int)((r.z + radius)*zdim) + 1);
		for(int ii = xmin; ii < xmax; ++ii)
		{
			for(int jj = ymin; jj < ymax; ++jj)
			{
				for(int kk = zmin; kk < zmax; ++kk)
				{
					int l = atomicListAdd(chash + b*xdim*ydim*zdim*max_neighbors
    								+ ii*ydim*zdim*max_neighbors
    								+ jj*zdim*max_neighbors
    								+ kk*max_neighbors,
    								(uint32_t)n, max_neighbors);
			    	if(l < 0)
			    		*failed_flag = 1;
				}
			}
		}
    }
}

__global__
void kernel_countneighbors(float3* clocs, int batch_size, int N, uint32_t* chash, int3 hdim, 
	int max_neighbors, uint32_t*cblock, int3 bdim, int max_block_neighbors, 
	float radius, uint32_t* cncount)
{
	int b = blockIdx.x/bdim.x;
	int bi = blockIdx.x % bdim.x;
	int bj = blockIdx.y;
	int bk = blockIdx.z;
    for(int i = threadIdx.x; i < max_block_neighbors; i += blockDim.x)
    {
    	uint32_t n = cblock[b*bdim.x*bdim.y*bdim.z*max_block_neighbors +
    						bi*bdim.y*bdim.z*max_block_neighbors +
    						bj*bdim.z*max_block_neighbors +
    						bk*max_block_neighbors +
    						i];
    	if(n == UINT32_MAX)
    		continue;
		float3 r = clocs[b*N + n];

    	int3 low = make_int3(max((int)(1.0f*(r.x - radius)*hdim.x), (int)bi*hdim.x/bdim.x), 
    						 max((int)(1.0f*(r.y - radius)*hdim.y), (int)bj*hdim.y/bdim.y), 
    						 max((int)(1.0f*(r.z - radius)*hdim.z), (int)bk*hdim.z/bdim.z));
    	int3 high = make_int3(min((int)(1.0f*(r.x + radius)*hdim.x) + 1, (int)(bi + 1)*hdim.x/bdim.x), 
    		                  min((int)(1.0f*(r.y + radius)*hdim.y) + 1, (int)(bj + 1)*hdim.y/bdim.y), 
    						  min((int)(1.0f*(r.z + radius)*hdim.z) + 1, (int)(bk + 1)*hdim.z/bdim.z));	
    	// int3 low = make_int3(max((int)(1.0f*(r.x - radius)*hdim.x), (int)0), 
    	// 					 max((int)(1.0f*(r.y - radius)*hdim.y), (int)0), 
    	// 					 max((int)(1.0f*(r.z - radius)*hdim.z), (int)0));
    	// int3 high = make_int3(min((int)hdim.x, (int)(bi + 1)*hdim.x/bdim.x), 
    	// 	                  min((int)hdim.y, (int)(bj + 1)*hdim.y/bdim.y), 
    	// 					  min((int)hdim.z, (int)(bk + 1)*hdim.z/bdim.z));	
    	for(int ii = low.x; ii < high.x; ++ii)
    	{
    		for(int jj = low.y; jj < high.y; ++jj)
    		{
    			for(int kk = low.z; kk < high.z; ++kk)
    			{
    				uint32_t* h = chash + b*hdim.x*hdim.y*hdim.z*max_neighbors
    								+ ii*hdim.y*hdim.z*max_neighbors
    								+ jj*hdim.z*max_neighbors
    								+ kk*max_neighbors;
					for(int j = 0; j < max_neighbors && h[j] != UINT32_MAX; ++j)
					{
						uint32_t n2 = h[j];
						float3 r2 = clocs[b*N + n2];
						float d = (r.x - r2.x)*(r.x - r2.x) +
								  (r.y - r2.y)*(r.y - r2.y) +
								  (r.z - r2.z)*(r.z - r2.z);
					    if(d < radius*radius){
							atomicAdd(cncount + b*N + n, 1);
					    }
					}
    			}
    		}
    	}
    }
}




int test_hashgrid(void)
{
	printf("\n=============== TEST HASHGRID ===========\n");
	// constants
	const int GRID_DIM = 50;
	const int N = 8000;
	const int BATCH_SIZE = 32;
	const int MAX_NEIGHBORS = max(128, N/(GRID_DIM*GRID_DIM*GRID_DIM)+1);
	const float RADIUS = 0.1f;
	const int BLOCK_DIM = 10;
	const int MAX_BLOCK_NEIGHBORS = max(401, N/(BLOCK_DIM*BLOCK_DIM*BLOCK_DIM)+1);

	if(GRID_DIM % BLOCK_DIM != 0)
	{
		printf("BLOCK_DIM (%d) does not divide evenly into GRID_DIM (%d)!\n", BLOCK_DIM, GRID_DIM);
		printf("Rerun with different values for those variables.\n");
		return 1;
	}

	// init random
	srand(time(NULL));

	// setup the particle locations
	float* locs = new float[BATCH_SIZE*N*3];
	for(int i = 0; i < BATCH_SIZE*N*3; ++i)
		locs[i] = 1.0f*rand()/RAND_MAX;
	float3* clocs;
	cudaMalloc((void**)&clocs, sizeof(float3)*BATCH_SIZE*N);
	cudaMemcpy(clocs, locs, sizeof(float)*BATCH_SIZE*N*3, cudaMemcpyHostToDevice); 

	/******** SETTING UP THE HASH GRID *********/
	// setup the hash grid
	uint32_t* chash;
	cudaMalloc((void**)&chash, 
		sizeof(uint32_t)*BATCH_SIZE*GRID_DIM*GRID_DIM*GRID_DIM*MAX_NEIGHBORS);
	cudaMemset(chash, 255, sizeof(uint32_t)*BATCH_SIZE*GRID_DIM*GRID_DIM*GRID_DIM*MAX_NEIGHBORS);

	// setup the failure flag
	uint8_t* cfailed_flag;
	cudaMalloc((void**)&cfailed_flag, sizeof(uint8_t));
	cudaMemset(cfailed_flag, 0, sizeof(uint8_t));

	// Compute the bloc and thread counts
	int nops = BATCH_SIZE*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    // setup the timer
    clock_t start, stop;
    start = clock();

    // Run the kernel
	kernel_assignhash<<<blocks, threads>>>(clocs, chash, N, BATCH_SIZE, GRID_DIM, GRID_DIM,
		GRID_DIM, MAX_NEIGHBORS, cfailed_flag, 0.0f);
	cudaDeviceSynchronize();

	// Time it
	stop = clock();
	printf("Time to construct hash: %9.5fms\n", 1000.0f*(stop - start)/CLOCKS_PER_SEC);
	uint8_t failed_flag;
	cudaMemcpy(&failed_flag, cfailed_flag, sizeof(uint8_t), cudaMemcpyDeviceToHost); 
	if(failed_flag)
		printf("Failure flag tripped when constructing hash grid\n");

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in kernel_assignhash: %s\n", cudaGetErrorString(err));
		return 1;
	}

	// Verify the hashgrid
	uint32_t* hash = new uint32_t[BATCH_SIZE*GRID_DIM*GRID_DIM*GRID_DIM*MAX_NEIGHBORS];
	cudaMemcpy(hash, chash, sizeof(uint32_t)*BATCH_SIZE*GRID_DIM*GRID_DIM*GRID_DIM*MAX_NEIGHBORS, cudaMemcpyDeviceToHost); 
	int not_found = 0;
	int maxi = 0;
	for(int b = 0; b < BATCH_SIZE; ++b)
	{
		for(uint32_t n = 0; n < N; ++n)
		{
			int3 idx = make_int3(locs[b*N*3 + n*3 + 0]*GRID_DIM, 
								 locs[b*N*3 + n*3 + 1]*GRID_DIM,
								 locs[b*N*3 + n*3 + 2]*GRID_DIM);
			uint32_t* h;
			h = hash + b*GRID_DIM*GRID_DIM*GRID_DIM*MAX_NEIGHBORS
					 + idx.x*GRID_DIM*GRID_DIM*MAX_NEIGHBORS
					 + idx.y*GRID_DIM*MAX_NEIGHBORS
					 + idx.z*MAX_NEIGHBORS;
			int i;
			for(i = 0; i < MAX_NEIGHBORS && h[i] != n; ++i) {}
			if(i >= MAX_NEIGHBORS)
				not_found++;
			maxi = max(maxi, i);
		}
	}
	printf("%d/%d were not found (%f%% failure rate).\n", not_found, 
		N*BATCH_SIZE, 100.0*not_found/(N*BATCH_SIZE));
	printf("Max list size: %d\n", maxi);

	/******** SETTING UP THE BLOCK GRID *********/
	uint32_t* cblock;
	cudaMalloc((void**)&cblock, 
		sizeof(uint32_t)*BATCH_SIZE*BLOCK_DIM*BLOCK_DIM*BLOCK_DIM*MAX_BLOCK_NEIGHBORS);
	cudaMemset(cblock, 255, sizeof(uint32_t)*BATCH_SIZE*BLOCK_DIM*BLOCK_DIM*
		BLOCK_DIM*MAX_BLOCK_NEIGHBORS);

	// reset the failure flag
	cudaMemset(cfailed_flag, 0, sizeof(uint8_t));

	// Compute the bloc and thread counts
	nops = BATCH_SIZE*N;
    numBlocks = ceil(nops * (1.0/256));
    blocks = dim3(numBlocks);
    threads = dim3(256);

    // start the timer
    start = clock();

    // Run the kernel
	kernel_assignhash<<<blocks, threads>>>(clocs, cblock, N, BATCH_SIZE, BLOCK_DIM, BLOCK_DIM,
		BLOCK_DIM, MAX_BLOCK_NEIGHBORS, cfailed_flag, RADIUS);
	cudaDeviceSynchronize();

	// Time it
	stop = clock();
	printf("Time to construct block: %9.5fms\n", 1000.0f*(stop - start)/CLOCKS_PER_SEC);
	cudaMemcpy(&failed_flag, cfailed_flag, sizeof(uint8_t), cudaMemcpyDeviceToHost); 
	if(failed_flag)
		printf("Failure flag tripped when constructing block grid\n");

	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in kernel_assignhash: %s\n", cudaGetErrorString(err));
		return 1;
	}

	// Verify the block grid
	uint32_t* block = new uint32_t[BATCH_SIZE*BLOCK_DIM*BLOCK_DIM*BLOCK_DIM*MAX_BLOCK_NEIGHBORS];
	cudaMemcpy(block, cblock, 
		sizeof(uint32_t)*BATCH_SIZE*BLOCK_DIM*BLOCK_DIM*BLOCK_DIM*MAX_BLOCK_NEIGHBORS, 
		cudaMemcpyDeviceToHost); 
	not_found = 0;
	maxi = 0;
	for(int b = 0; b < BATCH_SIZE; ++b)
	{
		for(uint32_t n = 0; n < N; ++n)
		{
			int3 idx = make_int3(locs[b*N*3 + n*3 + 0]*BLOCK_DIM, 
								 locs[b*N*3 + n*3 + 1]*BLOCK_DIM,
								 locs[b*N*3 + n*3 + 2]*BLOCK_DIM);
			uint32_t* h;
			h = block + b*BLOCK_DIM*BLOCK_DIM*BLOCK_DIM*MAX_BLOCK_NEIGHBORS
					 + idx.x*BLOCK_DIM*BLOCK_DIM*MAX_BLOCK_NEIGHBORS
					 + idx.y*BLOCK_DIM*MAX_BLOCK_NEIGHBORS
					 + idx.z*MAX_BLOCK_NEIGHBORS;
			int i;
			for(i = 0; i < MAX_BLOCK_NEIGHBORS && h[i] != n; ++i) {}
			if(i >= MAX_BLOCK_NEIGHBORS)
				not_found++;
			maxi = max(maxi, i);
		}
	}
	printf("%d/%d were not found (%f%% failure rate).\n", not_found, 
		N*BATCH_SIZE, 100.0*not_found/(N*BATCH_SIZE));
	printf("Max list size: %d\n", maxi);

	/******** PERFORMING NEIGHBOR COUNT *********/
	// setup memory for neighbor count
	uint32_t* cncount;
	cudaMalloc((void**)&cncount, sizeof(uint32_t)*BATCH_SIZE*N);
	cudaMemset(cncount, 0, sizeof(uint32_t)*BATCH_SIZE*N);

	// Compute the bloc and thread counts
    blocks = dim3(BLOCK_DIM*BATCH_SIZE, BLOCK_DIM, BLOCK_DIM);
    threads = dim3(256);

    // setup the timer
    start = clock();

    // Run the kernel
	kernel_countneighbors<<<blocks, threads>>>(clocs, BATCH_SIZE, N, chash, 
		make_int3(GRID_DIM, GRID_DIM, GRID_DIM), MAX_NEIGHBORS, cblock, 
		make_int3(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), MAX_BLOCK_NEIGHBORS, RADIUS, cncount);
	cudaDeviceSynchronize();

	// Time it
	stop = clock();
	printf("Time to do neighbor lookup: %fms\n", 1000.0f*(stop - start)/CLOCKS_PER_SEC);

	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in kernel_countneighbors: %s\n", cudaGetErrorString(err));
		return 1;
	}

	// Verify the counds
	uint32_t* ncount = new uint32_t[BATCH_SIZE*N];
	cudaMemcpy(ncount, cncount, sizeof(uint32_t)*BATCH_SIZE*N, cudaMemcpyDeviceToHost); 
	double sum = 0.0f;
	int ii = 0;
	for(int b = 0; b < BATCH_SIZE; ++b)
	{
		for(uint32_t n = 0; n < N; ++n)
		{
			sum += ncount[b*N + n];
			++ii;
		}
	}
	printf("Average neighbor count: %lf\n", sum/(BATCH_SIZE*N));


	/******** PERFORMING NEIGHBOR COUNT N2 *********/
	printf("Redoing neighbor count with N2 algorithm.\n");
	// setup memory for neighbor count
	cudaMemset(cncount, 0, sizeof(uint32_t)*BATCH_SIZE*N);

	// Compute the bloc and thread counts
	nops = BATCH_SIZE*N;
    numBlocks = ceil(nops * (1.0/256));
    blocks = dim3(numBlocks);
    threads = dim3(256);

    // setup the timer
    start = clock();

    // Run the kernel
	kernel_countneighbors_n2<<<blocks, threads>>>(clocs, BATCH_SIZE, N, RADIUS, cncount);
	cudaDeviceSynchronize();

	// Time it
	stop = clock();
	printf("Time to do neighbor lookup N2: %fms\n", 1000.0f*(stop - start)/CLOCKS_PER_SEC);

	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in kernel_countneighbors_n2: %s\n", cudaGetErrorString(err));
		return 1;
	}

	// Verify the counds
	cudaMemcpy(ncount, cncount, sizeof(uint32_t)*BATCH_SIZE*N, cudaMemcpyDeviceToHost); 
	sum = 0.0f;
	ii = 0;
	for(int b = 0; b < BATCH_SIZE; ++b)
	{
		for(uint32_t n = 0; n < N; ++n)
		{
			sum += ncount[b*N + n];
			++ii;
		}
	}
	printf("Average neighbor count: %lf\n", sum/(BATCH_SIZE*N));


	return 0;

    
}