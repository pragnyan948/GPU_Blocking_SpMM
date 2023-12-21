#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include "merging.h"

//column partitioning
void columnPartition(int *matrix, int numRows, int numCols, int partitionSize, int *result);
void columnPartitionCPU(int *matrix, int numRows, int numCols, int partitionSize, int *result);
void generateRandomSparseMatrix(int *matrix, int numRows, int numCols, double sparsity);
void printMatrix(int *matrix, int numRows, int numCols);
void writeMatrixToFile(int *matrix, int numRows, int numCols, const char *filename);
void writeResultMatrixToFile(int *resultCPU, int numRows, int quotientSize, const char *filename);
void writeResultMatrixToFileGPU(int *result, int numRows, int quotientSize, const char *filename);

//hash-based compression
void hash_fn(int *value_mat, int numRows, int numCols, int* row_idx , int* grp_ptr);
void host(int *value_mat, int numRows, int numCols);

//two-step merging
extern "C"
void computeGold(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height,int* grp_count, unsigned int* order_ds, int* group,  unsigned int* NNZ_pattern);
void WriteFile(unsigned int* Vector, char* file_name, unsigned int width, int height);
void WriteFile_2(int* Vector, char* file_name, unsigned int width, int height);
void Initialize(unsigned int* Vector, unsigned int width, unsigned int height, unsigned int val);
void MergeOnDevice(Out_1DSAAD out, In_1DSAAD in, float tau, unsigned int blocks_across_width, unsigned int height, unsigned int* group_indices_data, unsigned int* group_indices_csr);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
int ReadFile(unsigned int* Vector, char* file_name, unsigned int width, int height);


int main(int argc, char** argv) {

    Timer timer;

    // Example matrix dimensions
    int numRows = 5000;
    int numCols = 6000;

    // Initialize host variables ----------------------------------------------

        // Allocate memory for the matrix
    int *matrix = (int *)malloc(numRows * numCols * sizeof(int));
    
    // Generate a random matrix
    generateRandomSparseMatrix(matrix, numRows, numCols, 0.25); // 25% sparse matrix

    // Write the generated matrix to a text file
    writeMatrixToFile(matrix, numRows, numCols, "matrix.txt");


//column partitioning
    // Perform column partitioning on CPU
    int partitionSize = 2; // patrition size is 2, you can modify it as necessary
    int quotientSize = numCols / partitionSize;
    int *resultCPU = (int *)malloc(numRows * quotientSize * sizeof(int));
	int *resultGPU = (int *)malloc(numRows * quotientSize * sizeof(int));

	printf("\nColumn Partition Gold..."); fflush(stdout);
	startTime(&timer);
    columnPartitionCPU((int *)matrix, numRows, numCols, partitionSize, resultCPU);
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	//timer inside the function
	columnPartition((int *)matrix, numRows, numCols, partitionSize, resultGPU);

	bool res = CompareMatrices(resultCPU, resultGPU);
    printf("Test %s\n", (res) ? "PASSED" : "FAILED");

    // Write the result matrix to a text file
    writeResultMatrixToFile(resultCPU, numRows, quotientSize, "output_binary_matrix_after_column_partition.txt");

//hash-based compression
	int *row_idx =  (int *)malloc(numRows * sizeof(int));//stores grouped row indices 
    int *grp_ptr =  (int *)malloc(numCols * sizeof(int));//stores group pointer

	printf("\nhash-based compression Gold..."); fflush(stdout);
	startTime(&timer);
    hash_fn((int *)resultCPU, numRows, quotientSize,row_idx,grp_ptr);
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	host((int *)resultCPU, numRows, quotientSize);


//two-step merging

    float tau = atof(argv[1]);
	unsigned int blocks_across_width = quotientSize;
	unsigned int height = numRows;
    unsigned int* inVector = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int* outVector = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    int* group =(int*) malloc(height*sizeof(int));
    unsigned int* blocks_height =( unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* outPattern = (unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));

    for(unsigned int i = 0; i < height * blocks_across_width; i++)
	{
        inVector[i] = resultCPU[i];
        if(i<height){
            group[i]=-1;
        }
	}

    for(unsigned int i = 0; i < height ; i++){
        group[i]=-1;
    }
    unsigned int* group_indices_data =( unsigned int*) malloc(numRows*sizeof(unsigned int));;
    unsigned int* group_indices_csr =( unsigned int*) malloc(quotientSize*sizeof(unsigned int));;

	for (int i=0;i<numRows; i++) {
		group_indices_data[i] = row_idx[i];
	}
	for (int i=0;i<quotientSize; i++) {
		group_indices_csr[i] = grp_ptr[i];
	}
    int* group_count=(int*) malloc(2*sizeof(int));
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
	cudaMalloc((void**)&out_d.pattern_indices, height*sizeof(unsigned int));
	cudaMalloc((void**)&out_d.outPattern, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&out_d.midPattern, height*blocks_across_width*sizeof(unsigned int));
	cudaMalloc((void**)&in_d.group_indices_data, height*sizeof(unsigned int));
	cudaMalloc((void**)&in_d.group_indices_csr, (blocks_across_width+1)*sizeof(unsigned int));
    
    cudaMemcpy(in_d.group, group, height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_d.inVector, inVector, height*blocks_across_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_d.group_indices_data, group_indices_data, height*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_d.group_indices_csr, group_indices_csr, (blocks_across_width+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaError_t cuda_ret;
    startTime(&timer);
    computeGold(outVector, inVector, blocks_height, outPattern, group_indices_data, group_indices_csr, tau, blocks_across_width, height, group_count,order_ds, group, NNZ_pattern);
    stopTime(&timer); printf("CPU Time: %f s\n", elapsedTime(timer));
    char* file_name_param[4] ={"./Data/order_ds.txt", "./Data/group.txt","./Data/blocks_height.txt", "./Data/outPattern.txt"};
    WriteFile(order_ds, file_name_param[0], 1,group_count[1]);
    WriteFile_2(group, file_name_param[1], 1,height);
    WriteFile(blocks_height, file_name_param[2], 1,group_count[0]);
    WriteFile(outPattern, file_name_param[3], blocks_across_width, group_count[0]);
    char* file_name_out ="./Data/outVector.txt";
    WriteFile(outVector, file_name_out, blocks_across_width, group_count[1]);

    free(inVector);
    
    //startTime(&timer);
    //MergeOnDevice(out_d, in_d, tau, blocks_across_width, height, group_indices_data, group_indices_csr);
    //cuda_ret = cudaDeviceSynchronize();
	//if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    //stopTime(&timer); printf("GPU Time: %f s\n", elapsedTime(timer));

    cudaFree(in_d.inVector);cudaFree(in_d.group_indices_data);cudaFree(in_d.group_indices_csr);
    unsigned int* outPattern_gpu = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int* outVector_gpu = ( unsigned int*) malloc(height*blocks_across_width*sizeof(unsigned int));
    unsigned int group_count_gpu;
    int* group_gpu =(int*) malloc(height*sizeof(int));
   // cuda_ret = cudaMemcpy(outVector_gpu, out_d.outVector, sizeof(unsigned int)*blocks_across_width*height, 
   //     cudaMemcpyDeviceToHost);
   // if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
   // cuda_ret = cudaMemcpy(outPattern_gpu, out_d.outPattern, sizeof(unsigned int)*blocks_across_width*height, 
   //     cudaMemcpyDeviceToHost);
   // if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
   // cuda_ret = cudaMemcpy(group_gpu, in_d.group, sizeof(int)*height, 
   //     cudaMemcpyDeviceToHost);
   // if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
   // cuda_ret = cudaMemcpy(&group_count_gpu, out_d.pattern_indices+height-1, sizeof(unsigned int), 
   //     cudaMemcpyDeviceToHost);
   // if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
   //char* file_name_param_gpu[3] ={"./Data/outVector_gpu.txt", "./Data/group_gpu.txt", "./Data/outPattern_gpu.txt"};
   //WriteFile(outVector_gpu+blocks_across_width, file_name_param_gpu[0], blocks_across_width,height-1);
   //WriteFile(outPattern_gpu+blocks_across_width, file_name_param_gpu[2], blocks_across_width,group_count_gpu);
   //WriteFile_2(group_gpu, file_name_param_gpu[1], 1,height);

   // //compare outVector, outPattern, group;
   // cudaFree(out_d.outVector); cudaFree(out_d.blocks_height); cudaFree(out_d.outPattern);cudaFree(in_d.group);
    free(outVector), free(blocks_height); free(outPattern);free(outVector_gpu);free(outPattern_gpu);free(group_gpu);


	return 0;
}

