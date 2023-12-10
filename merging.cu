#include <stdio.h>
extern const int TILE_SIZE = 256;
extern const int BLOCK_SIZE = 256;
// includes, kernels
__constant__ int intersect[3]={1,0,1};
__constant__ int union_set[3]={1,2,1};
__constant__ int non_intersect[3]={0,1,0};

#define NUM_ELEMENTS 256

__global__ void similarity(unsigned int* inVector, unsigned int* group_indices_data, unsigned int* group_indices_csr, float tau, unsigned int blocks_across_width, unsigned int height)
{
    __shared__ int V_s[NUM_ELEMENTS];
    __syncthreads();
    int t = threadIdx.x; 
    int start_csr, end_csr, start_point, end_point;
    float num_tau[NUM_ELEMENTS];
    float den_tau[NUM_ELEMENTS];
    float tau_idx[NUM_ELEMENTS];
    int compare_factor[NUM_ELEMENTS];
    int lambda0;
    unsigned int start = blockDim.x*blockIdx.x;
    unsigned int end = blocks_across_width*height;
    if(t<height){
        for( int k = 0; k <blocks_across_width ; k++){
            V_s[t+k*height] = inVector[start+t*blocks_across_width+k]; 
        }
        __syncthreads();
        for( int k = 0; k <blocks_across_width ; k++){
            V_s[t+end]+=V_s[t+k*height];
        }
        start_csr=4-V_s[t+end]-1;
        if(start_csr<0){start_csr=0;}
        end_csr=4-V_s[t+end]+2;
        if(end_csr>4){end_csr=4;}
        start_point=group_indices_csr[start_csr];
        end_point=group_indices_csr[end_csr];
        __syncthreads();
        for (int jj = start_point; jj <end_point ; jj++){
            for (int k = 0; k <blocks_across_width ; k++)
                {
                    num_tau[jj]+=intersect[V_s[t+k*height]+V_s[group_indices_data[jj]+k*height]];
                    den_tau[jj]+=union_set[V_s[t+k*height]+V_s[group_indices_data[jj]+k*height]];
                }
            tau_idx[jj]=num_tau[jj]/den_tau[jj];
            compare_factor[jj]=tau_idx[jj]/tau;
        }       
    }
    __syncthreads();
}

void MergeOnDevice(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height, unsigned int grp_count)
{
    
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
    similarity<<<gridDim,blockDim>>>(inVector, group_indices_data, group_indices_csr, tau, blocks_across_width, height);
}