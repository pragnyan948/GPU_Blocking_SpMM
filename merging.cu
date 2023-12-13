#include <stdio.h>
#include "merging.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
extern const int TILE_SIZE = 256;
extern const int BLOCK_SIZE = 256;

#define NUM_ELEMENTS 256
#define HISTO_WIDTH 256
void blockSums_fn(unsigned int* outArray, unsigned int* inArray,unsigned int* blockSums, unsigned int* blockSums_out, int numElements, int gridsize, dim3 gridDim, dim3 blockDim);


__global__ void similarity(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height)
{
    __shared__ float V_s[NUM_ELEMENTS];
    __syncthreads();
    int t = threadIdx.x; 
    int start_csr, end_csr, start_point, end_point;
    float num_tau[NUM_ELEMENTS];
    float den_tau[NUM_ELEMENTS];
    float tau_idx[NUM_ELEMENTS];
    int compare_factor[NUM_ELEMENTS];
    float limit_factor;
    int lambda0;
    int NNZPattern=0;
    unsigned int start = blockDim.x*blockIdx.x;
    unsigned int end = blocks_across_width*height;
    if(t<height){
        for( int k = 0; k <blocks_across_width ; k++){
            V_s[t*height+k] = in.inVector[start+t*blocks_across_width+k]; 
            //out.outPattern[start+t*blocks_across_width+k]=in.inVector[start+t*blocks_across_width+k]; 
        }
        __syncthreads();
        for( int k = 0; k <blocks_across_width ; k++){
            V_s[t+end]+=V_s[t*height+k];
        }
        __syncthreads();
        start_csr=blocks_across_width-V_s[t+end]-1;
        if(start_csr<0){start_csr=0;}
        end_csr=blocks_across_width-V_s[t+end]+2;
        if(end_csr>blocks_across_width){end_csr=blocks_across_width;}
        start_point=in.group_indices_csr[start_csr];
        end_point=in.group_indices_csr[end_csr];
        __syncthreads();
        limit_factor=V_s[t+end]/(1-tau/2);
        for (int jj = start_point; jj <end_point ; jj++){
            for (int k = 0; k <blocks_across_width ; k++)
                {
                    num_tau[jj]+= ((in.group[in.group_indices_data[jj]]==-1)&&(V_s[t*height+k] == in.inVector[in.group_indices_data[jj]*blocks_across_width+k]))?in.inVector[in.group_indices_data[jj]*blocks_across_width+k]:0;
                    den_tau[jj]+= ((in.group[in.group_indices_data[jj]]==-1)&&((V_s[t*height+k] + in.inVector[in.group_indices_data[jj]*blocks_across_width+k])>=1))?1:0;
                }
            tau_idx[jj]=num_tau[jj]/(den_tau[jj]);
            compare_factor[jj]=(tau_idx[jj]>tau)?1:0;
            //Compute NNZpattern
            (compare_factor[jj])?atomicCAS(&in.group[in.group_indices_data[jj]],-1, t):0;
            __syncthreads();
            NNZPattern=(in.group[in.group_indices_data[jj]]!=-1)?0:NNZPattern;
            for (int k = 0; k <blocks_across_width ; k++){
                (compare_factor[jj]&&(in.group[in.group_indices_data[jj]]!=-1))?atomicMax(&out.outPattern[in.group[in.group_indices_data[jj]]*blocks_across_width+k],in.inVector[in.group_indices_data[jj]*blocks_across_width+k]):0;
                NNZPattern+=(in.group[in.group_indices_data[jj]]!=-1)?out.outPattern[in.group[in.group_indices_data[jj]]*blocks_across_width+k]:0;
            }
            __syncthreads();
            //(compare_factor[jj] &&(NNZPattern[group[jj]]<limit_factor[group[jj]])))?Atomic_Compare (group[jj],,-1, t):
        }
        in.group[t]+=1;
    __syncthreads();
    }
}

__global__ void block_height(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height){
    //histogram code
    __shared__ unsigned int histo_group[HISTO_WIDTH];
   if (threadIdx.x < HISTO_WIDTH){
        histo_group[threadIdx.x] = 0;
    }
   __syncthreads();
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;
    while (k< height) {
        atomicAdd( &(histo_group[in.group[k]]), 1);
        k += stride;
    }
    __syncthreads();
    if (threadIdx.x < HISTO_WIDTH){
        atomicAdd( &(out.blocks_height[threadIdx.x]),histo_group[threadIdx.x] );
    }
    __syncthreads();
}

__global__ void reordering(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height,unsigned int* blocksums){
    int t = threadIdx.x;
    int block_test[NUM_ELEMENTS];
    for(int i=0; i<NUM_ELEMENTS; i++){
        block_test[i]=0;
    }
    if(t<height){
        if(t>0){
            int jj=0;
            while(jj< (out.blocks_height[t]-out.blocks_height[t-1])){
                for(int k = 0; k <height; k++){
                    for(int i = 0; i <blocks_across_width; i++){
                        //out.outVector[(out.blocks_height[t]-1+jj)*blocks_across_width+i]=(in.group[k]==t)?in.inVector[k*blocks_across_width+i]:0;
                        block_test[(out.blocks_height[t-1]+jj)*blocks_across_width+i]=(in.group[k]==t)?in.inVector[k*blocks_across_width+i]:0;
                    }
                    jj+=(in.group[k]==t)?1:0;
                }
            }
        }
        else if(t==0){
            int jj=0;
            while(jj< out.blocks_height[0]){
                for(int k = 0; k <height; k++){
                    for(int i = 0; i <blocks_across_width; i++){
                        //out.outVector[(out.blocks_height[t]+jj)*blocks_across_width+i]=(in.group[k]==t)?in.inVector[k*blocks_across_width+i]:0;
                        block_test[(jj)*blocks_across_width+i]=(in.group[k]==0)?in.inVector[k*blocks_across_width+i]:0;
                    }
                    jj+=(in.group[k]==t)?1:0;
                }                
            }
        }
    }
}

__global__ void reduction(unsigned int *in_data, unsigned int *out_data, int stride, int num_elements)
{
    __shared__ float scan_array[2*BLOCK_SIZE]; 
	unsigned int t = threadIdx.x; 
    //unsigned int start = blockDim.x*blockIdx.x; 
    unsigned int start = blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y; 
	int Row = start + t;
    if(Row<num_elements){
        scan_array[blockDim.x+t] = in_data[start+ blockDim.x+t]; 
        scan_array[t] = in_data[start + t]; 
    }
    __syncthreads();
    //printf("%d\n", scan_array[6]);
    int index =  (threadIdx.x+1)*stride*2 - 1;
    if(index <  2*BLOCK_SIZE)
        scan_array[index] += scan_array[index-stride];
     __syncthreads();
    if(Row<num_elements){
        out_data[Row] = scan_array[t];
    }
     __syncthreads();
}

__global__ void distribution(unsigned int *in_data, unsigned int *out_data, int stride, int num_elements)
{
    __shared__ float scan_array[2*BLOCK_SIZE]; 
	unsigned int t = threadIdx.x; 
    //unsigned int start = blockDim.x*blockIdx.x; 
    unsigned int start = blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y; 
	int Row = start + t;
    if(Row<num_elements){
        scan_array[blockDim.x+t] = in_data[start+ blockDim.x+t]; 
        scan_array[t] = in_data[start + t]; 
    }
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < 2*BLOCK_SIZE)
    {
        scan_array[index+stride] += scan_array[index];
    }
     __syncthreads();
    if(Row<num_elements){
        out_data[Row] = scan_array[t];
    }
     __syncthreads();
}

__global__ void blockSumsCompute(unsigned int *in_data, unsigned int *out_data, int num_elements)
{
     __shared__ float scan_array[BLOCK_SIZE]; 
	unsigned int t = threadIdx.x; 
    //unsigned int start = blockDim.x*blockIdx.x; 
    unsigned int start = blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y; 
	int Row = start+ t;
    if(Row<num_elements){
        scan_array[t]=in_data[(start+t+1)*blockDim.x-1];
    __syncthreads();   
    }
    if(Row<num_elements){
         out_data[Row] = scan_array[t];
    }
    __syncthreads(); 
}

__global__ void shift(unsigned int *in_data, unsigned int *out_data, int num_elements)
{
     __shared__ float scan_array[BLOCK_SIZE]; 
	unsigned int t = threadIdx.x; 
    //unsigned int start = blockDim.x*blockIdx.x; 
    unsigned int start = blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y; 
	int Row =start + t;
    if (start<1){
        scan_array[0] = 0;
        if (t>0){
             scan_array[t]=in_data[start+t-1];
        }
    }
    else{
        if(Row<num_elements){
            scan_array[t]=in_data[start+t-1];
        }
    }
    __syncthreads();
    if(Row<num_elements){
        out_data[Row] = scan_array[t];
    }
     //printf("%d\n",out_data[7]);
    __syncthreads(); 
}

__global__ void VectorAddition(unsigned int *in_data,unsigned int *in_data_1, unsigned int *out_data, int num_elements){
    __shared__ float scan_array[BLOCK_SIZE+1]; 
	unsigned int t = threadIdx.x; 
    //unsigned int start = blockDim.x*blockIdx.x; 
    unsigned int start = blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y; 
	int Row = start + t;
    if(Row<num_elements){
        scan_array[t] = in_data[start + t]; 
        scan_array[blockDim.x] = in_data_1[blockIdx.x];
    }
    __syncthreads(); 
    if(Row<num_elements){
        scan_array[t]+= scan_array[blockDim.x];
    }
    __syncthreads(); 
    if(Row<num_elements){
        out_data[Row] = scan_array[t];
    }
    __syncthreads(); 
}

void inclusive_scan(unsigned int* outArray, unsigned int* inArray, unsigned int* blockSums, unsigned int* blockSums_out, int numElements, int gridsize, dim3 gridDim, dim3 blockDim){
    int stride = 1;
    while(stride <= BLOCK_SIZE)
    {
        reduction<<<gridDim,blockDim>>>(inArray, outArray, stride, numElements);
        inArray=outArray;
        stride = stride*2;
    }
    //distribution
    stride = BLOCK_SIZE/2;
    while(stride > 0)
    {
        distribution<<<gridDim,blockDim>>>(inArray, outArray, stride, numElements);
        inArray=outArray;
        stride = stride/2;
    }
    
    if (gridsize>=1){
        blockSums_fn(outArray, inArray, blockSums, blockSums_out, numElements, gridsize, gridDim, blockDim);
        blockSums=blockSums_out;  
        VectorAddition<<<gridDim,blockDim>>>(inArray, blockSums, outArray, numElements);
        inArray=outArray;
    }
}

void blockSums_fn(unsigned int* outArray, unsigned int* inArray,unsigned int* blockSums, unsigned int* blockSums_out, int numElements, int gridsize, dim3 gridDim, dim3 blockDim){
    blockSumsCompute<<<gridDim,blockDim>>>(inArray, blockSums, gridsize);
    inArray=outArray;
    int gridsize2=0;
    //if (gridsize>TILE_SIZE){
        //gridsize2=gridsize/TILE_SIZE;
    //}
    //printf("gridsize:%d\n", gridsize);
    //printf("gridsize2:%d\n", gridsize2);
    inclusive_scan(blockSums_out, blockSums, blockSums, blockSums_out, gridsize, gridsize2, gridDim, blockDim);
    blockSums=blockSums_out;
    shift<<<gridDim,blockDim>>>(blockSums, blockSums_out,gridsize);
    blockSums=blockSums_out;
}


void MergeOnDevice(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height)
{
    unsigned int* outArray;
	cudaMalloc((void**)&outArray, height*sizeof(unsigned int));
    unsigned int* blockSums;
	cudaMalloc((void**)&blockSums, height*sizeof(unsigned int));
    unsigned int* blockSums_out;
	cudaMalloc((void**)&blockSums_out, height*sizeof(unsigned int));
    dim3 blockDim(BLOCK_SIZE,1,1);
    int gridsize = 1;
    int gridsize_x;
    int gridsize_y;
    if (gridsize > TILE_SIZE){
        gridsize_x = TILE_SIZE;
        gridsize_y = gridsize/TILE_SIZE+1;
    }
    else{
        gridsize_x=gridsize;
        gridsize_y=1;
    }
   	dim3 gridDim(gridsize_x, gridsize_y, 1);
    similarity<<<gridDim,blockDim>>>(out, in, tau, blocks_across_width, height);
    block_height<<<gridDim,blockDim>>>(out, in, tau, blocks_across_width, height);
    inclusive_scan(outArray, out.blocks_height, blockSums, blockSums_out, height, gridsize, gridDim, blockDim);
    out.blocks_height=outArray;
    reordering<<<gridDim,blockDim>>>(out, in, tau, blocks_across_width, height, blockSums_out);
}