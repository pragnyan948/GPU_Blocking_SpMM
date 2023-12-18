#include <stdio.h>
#include "merging.h"

extern const int TILE_SIZE = 1024;
__constant__ unsigned int group_indices_c[BLOCK_SIZE];
__constant__ unsigned int group_csr_c[BLOCK_SIZE];
#define HISTO_WIDTH 1024
void blockSums_fn(unsigned int* outArray, unsigned int* inArray,unsigned int* blockSums, unsigned int* blockSums_out, int numElements, int gridsize, dim3 gridDim, dim3 blockDim);


__global__ void similarity(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height)
{
    __shared__ float V_s[BLOCK_SIZE];
    __shared__ float num_tau[BLOCK_SIZE];
    __shared__ float den_tau[BLOCK_SIZE];
    __shared__ float tau_idx[BLOCK_SIZE];
    __shared__ int group_indices_s[BLOCK_SIZE];
    __shared__ int pattern[BLOCK_SIZE];
    __shared__ int pattern_NNZ[BLOCK_SIZE];
    __shared__ int compare_factor[BLOCK_SIZE];
    __syncthreads();
    int start_csr, end_csr, start_point, end_point;
    float limit_factor;
    unsigned int start =  blockDim.x*blockIdx.x+blockDim.x*gridDim.x*blockIdx.y;
    unsigned int end = blockDim.x - blocks_across_width;
	int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    if((tx<blocks_across_width)&&(bx<height)){
        pattern[tx]=in.inVector[bx*blocks_across_width+tx]; 
        pattern_NNZ[0]=0;
        group_indices_s[tx] = in.group_indices_data[tx];
        __syncthreads();
        for( int k = 0; k <blocks_across_width ; k++){
            V_s[tx]+=pattern[k];
        }
        __syncthreads();
        limit_factor=V_s[tx]/(1-tau/2);
        start_csr=blocks_across_width-1-V_s[tx]-limit_factor;
        if(start_csr<0){start_csr=0;}
        end_csr=blocks_across_width-1-V_s[tx]+limit_factor;
        if(end_csr>blocks_across_width-1){end_csr=blocks_across_width-1;}
        start_point=in.group_indices_csr[start_csr];
        end_point=in.group_indices_csr[end_csr];
        __syncthreads();
        for (int k = group_indices_s[end_point]; k <group_indices_s[start_point]+1 ; k++){
            num_tau[tx]= ((in.group[k]==-1)&&(pattern[tx] == in.inVector[k*blocks_across_width+tx]))?in.inVector[k*blocks_across_width+tx]:0;
            den_tau[tx]= ((in.group[k]==-1)&&((pattern[tx] + in.inVector[k*blocks_across_width+tx])>=1))?1:0;
            __syncthreads();
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                int index = 2 * stride * tx;
                if (index < blockDim.x) {
                    num_tau[index] += num_tau[index + stride];
                    den_tau[index] += den_tau[index + stride];
                }
                __syncthreads();
            }
            tau_idx[k]=(in.group[k]==-1)?num_tau[0]/(den_tau[0]):0;
            compare_factor[k]=(tau_idx[k]>tau)?1:0;
            ((tx==0) && compare_factor[k]&&(pattern_NNZ[0]<=limit_factor))?atomicCAS(&in.group[k],-1, bx):0;
            __syncthreads();
            (compare_factor[k]&&(in.group[k]!=-1))?atomicMax(&pattern[tx],in.inVector[k*blocks_across_width+tx]):0;
            __syncthreads(); 
            //Compute NNZpattern
            pattern_NNZ[tx] = pattern[tx];
            __syncthreads(); 
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                int index = 2 * stride * tx;
                if (index < blockDim.x) {
                    pattern_NNZ[index] += (in.group[k]!=-1)?pattern_NNZ[index + stride]:0;
                }
                __syncthreads();
            }
            //NNZPattern+=(in.group[group_indices_s[jj]]!=-1)?out.midPattern[in.group[group_indices_s[jj]]*blocks_across_width+k]:0;
            //(compare_factor[jj] &&(NNZPattern[group[jj]]<limit_factor[group[jj]])))?Atomic_Compare (in.group[jj],,-1, t):
        }
        in.group[bx]+=1;
        out.midPattern[bx*blocks_across_width+tx]=pattern[tx];
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
    __syncthreads();
    while (k< height) {
        atomicAdd( &(histo_group[in.group[k]]), 1);
        k += stride;
    }
    __syncthreads();
    if (threadIdx.x < HISTO_WIDTH){
        atomicAdd( &(out.blocks_height[threadIdx.x]),histo_group[threadIdx.x] );
    }
    __syncthreads();
    k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k< height){
        for(int i = 0; i<k; i++){
            out.pattern_indices[k]+=(out.blocks_height[i]!=0)?1:0;
        }
    }
    __syncthreads();
}

__global__ void reordering(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height,unsigned int* blocksums){
    int t = threadIdx.x;
    //int block_test[NUM_ELEMENTS];
    //for(int i=0; i<NUM_ELEMENTS; i++){
        //block_test[i]=0;
    //}
    if(t<height){
        if(t>0){
            int jj=0;
            while(jj< (out.blocks_height[t]-out.blocks_height[t-1])){
                for(int k = 0; k <height; k++){
                    for(int i = 0; i <blocks_across_width; i++){
                       out.outVector[(out.blocks_height[t-1]+jj)*blocks_across_width+i]=(in.group[k]==t)?in.inVector[k*blocks_across_width+i]: out.outVector[(out.blocks_height[t-1]+jj)*blocks_across_width+i];
                       if(jj==0){
                        out.outPattern[(out.pattern_indices[t])*blocks_across_width+i]=(in.group[k]==t)?out.midPattern[k*blocks_across_width+i]:out.outPattern[(out.pattern_indices[t])*blocks_across_width+i];
                       }
                    }
                    jj+=(in.group[k]==t)?1:0;
                }
            }
        }
        __syncthreads();
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


void MergeOnDevice(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height, unsigned int* group_indices_data, unsigned int* group_indices_csr)
{
    unsigned int* outArray;
	cudaMalloc((void**)&outArray, height*sizeof(unsigned int));
    unsigned int* blockSums;
	cudaMalloc((void**)&blockSums, height*sizeof(unsigned int));
    unsigned int* blockSums_out;
	cudaMalloc((void**)&blockSums_out, height*sizeof(unsigned int));
    //cudaMemcpyToSymbol(group_csr_c, group_indices_csr, (blocks_across_width+1)*sizeof(unsigned int));
    cudaMemcpyToSymbol(group_indices_c, group_indices_data, height*sizeof(unsigned int));
    dim3 blockDim(BLOCK_SIZE,1,1);
    int gridsize = height;
    int gridsize_x;
    int gridsize_y;
    if (gridsize > TILE_SIZE){
        gridsize_x = TILE_SIZE;
        gridsize_y = gridsize/TILE_SIZE;
        if(gridsize % TILE_SIZE){
            gridsize_y++;
        }
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
    //reordering<<<gridDim,blockDim>>>(out, in, tau, blocks_across_width, height, blockSums_out);
}