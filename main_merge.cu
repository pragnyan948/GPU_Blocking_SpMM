#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "merging.h"
#include <stdio.h>
// Write a vector to file
extern "C"
void computeGold(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height,int grp_count, unsigned int* order_ds, int* group,  unsigned int* NNZ_pattern);
void WriteFile(unsigned int* Vector, char* file_name, unsigned int width, int height);
void Initialize(unsigned int* Vector, unsigned int width, unsigned int height, unsigned int val);
void MergeOnDevice(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    
    float tau = atof(argv[1]);
    unsigned int blocks_across_width = atoi(argv[2]);
    unsigned int height = atoi(argv[3]);
    unsigned int* inVector = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int* outVector = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    int* group =(int*) malloc(height*sizeof(int));
    unsigned int* blocks_height =( unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* outPattern = (unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    for(unsigned int i = 0; i < height * blocks_across_width; i++)
	{
		//inVector[i]=1;
        inVector[i] = rand()%2;
        if(i<height){
            group[i]=-1;
        }
	}
    //inVector[i]=1;
    unsigned int group_indices_data[8]={0,2,5,7,1,3,6,4};
    unsigned int group_indices_csr[5]={0,0,4,6,7};
    int group_count=0;
    printf("tau, blocks_across_width, height, %f %d %d\n ", tau, blocks_across_width, height);
    char* file_name = argv[4];
    WriteFile(inVector, file_name, blocks_across_width, height);
    unsigned int* order_ds = (unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* NNZ_pattern = (unsigned int*) malloc(height*sizeof(unsigned int));
   
    Initialize(NNZ_pattern, 1, height, 0);
    Initialize(blocks_height, 1, height, 1);

     //GPU
    Out_1DSAAD out_d;
    In_1DSAAD in_d;
	cudaMalloc((void**)&in_d.group, height*sizeof(int));
	cudaMalloc((void**)&out_d.outVector, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&in_d.inVector, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&out_d.blocks_height, height*sizeof(unsigned int));
	cudaMalloc((void**)&out_d.outPattern, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&out_d.NNZPattern, height*sizeof(unsigned int));
	cudaMalloc((void**)&out_d.group_count, sizeof(unsigned int));
	cudaMalloc((void**)&in_d.group_indices_data, height*sizeof(unsigned int));
	cudaMalloc((void**)&in_d.group_indices_csr, 5*sizeof(unsigned int));
    
    cudaMemcpy(in_d.group, group, height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_d.inVector, inVector, height*blocks_across_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_d.group_indices_data, group_indices_data, height*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_d.group_indices_csr, group_indices_csr, 5*sizeof(unsigned int), cudaMemcpyHostToDevice);
   

    Timer timer;
    cudaError_t cuda_ret;
    startTime(&timer);
    computeGold(outVector, inVector, blocks_height, outPattern, group_indices_data, group_indices_csr, tau, blocks_across_width, height, group_count,order_ds, group, NNZ_pattern);
    stopTime(&timer); printf("CPU Time: %f s\n", elapsedTime(timer));
    char* file_name_out ="./Data/outVector.txt";
    WriteFile(outVector, file_name_out, blocks_across_width, height);

    free(inVector);
    
    startTime(&timer);
    MergeOnDevice(out_d, in_d, tau, blocks_across_width, height);
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("GPU Time: %f s\n", elapsedTime(timer));

    cudaFree(in_d.inVector);cudaFree(in_d.group_indices_data);cudaFree(in_d.group_indices_csr);
    unsigned int* outPattern_gpu = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int* outVector_gpu = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    int* group_gpu =(int*) malloc(height*sizeof(int));
    cuda_ret = cudaMemcpy(outVector_gpu, out_d.outVector, sizeof(unsigned int)*blocks_across_width*height, 
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
    cuda_ret = cudaMemcpy(outPattern_gpu, out_d.outPattern, sizeof(unsigned int)*blocks_across_width*height, 
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
    cuda_ret = cudaMemcpy(group_gpu, in_d.group, sizeof(int)*height, 
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");

    //compare outVector, outPattern, group;
    cudaFree(out_d.outVector); cudaFree(out_d.blocks_height); cudaFree(out_d.outPattern); cudaFree(out_d.NNZPattern);cudaFree(in_d.group);
    free(outVector), free(blocks_height); free(outPattern);
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

