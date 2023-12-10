#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
// Write a vector to file
extern "C" 
void computeGold(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height,unsigned int grp_count);
void WriteFile(unsigned int* Vector, char* file_name, unsigned int width, int height);
void MergeOnDevice(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height, unsigned int grp_count);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    
    float tau = atof(argv[1]);
    unsigned int blocks_across_width = atoi(argv[2]);
    unsigned int height = atoi(argv[3]);
    unsigned int* inVector = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int* outVector = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int* blocks_height =( unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* outPattern = (unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    for(unsigned int i = 0; i < height * blocks_across_width; i++)
	{
		//inVector[i]=1;
        inVector[i] = rand()%2;
	}
    //inVector[i]=1;
    unsigned int group_indices_data[8]={0,2,5,7,1,3,6,4};
    unsigned int group_indices_csr[5]={0,0,4,6,7};
    int group_count=0;
    printf("%f %d %d\n ", tau, blocks_across_width, height);
    char* file_name = argv[4];
    WriteFile(inVector, file_name, blocks_across_width, height);
    computeGold(outVector, inVector, blocks_height, outPattern, group_indices_data, group_indices_csr, tau, blocks_across_width, height, group_count);
    char* file_name_out ="./outVector.txt";
    WriteFile(outVector, file_name_out, blocks_across_width, height);

    //GPU
    unsigned int* inVector_d;
    unsigned int* outVector_d;
    unsigned int* blocks_height_d;
    unsigned int* outPattern_d;
    unsigned int* group_indices_data_d;
    unsigned int* group_indices_csr_d;
	cudaMalloc((void**)&outVector_d, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&inVector_d, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&blocks_height_d, height*sizeof(unsigned int));
	cudaMalloc((void**)&outPattern_d, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&group_indices_data_d, height*sizeof(unsigned int));
	cudaMalloc((void**)&group_indices_csr_d, 5*sizeof(unsigned int));

    cudaMemcpy(inVector_d, inVector, height*blocks_across_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_indices_data_d, group_indices_data, height*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_indices_csr_d, group_indices_csr, 5*sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(inVector);
    //MergeOnDevice(outVector_d, inVector_d, blocks_height_d, outPattern_d, group_indices_data_d, group_indices_csr_d, tau, blocks_across_width, height, group_count);
    cudaFree(inVector_d);cudaFree(group_indices_data_d);cudaFree(group_indices_csr_d);
    cudaFree(outVector_d); cudaFree(blocks_height_d); cudaFree(outPattern_d); 
    free(outVector), free(blocks_height); cudaFree(outPattern);
}

