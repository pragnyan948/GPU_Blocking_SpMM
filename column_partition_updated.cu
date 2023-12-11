#include <stdio.h>

// CUDA kernel for column partitioning
__global__ void columnPartitionKernel(int *matrix, int numRows, int numCols, int partitionSize, int *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        int quotientSize = numCols / partitionSize;
        int *binaryVector = result + row * quotientSize;

        // Initialize the binary vector with zeros
        for (int j = 0; j < quotientSize; j++) {
            binaryVector[j] = 0;
        }

        // Check if there is a non-zero element in the current block in the partition
        for (int j = 0; j < numCols; j++) {
            int blockIndex = j / partitionSize; // Determine the block index

            // Check if there is a non-zero element in the current block in the partition
            if (matrix[row * numCols + j] != 0) {
                binaryVector[blockIndex] = 1;
            }
        }
    }
}

// Function to write a result matrix to a text file for GPU
void writeResultMatrixToFileGPU(int *result, int numRows, int quotientSize, const char *filename) {
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < quotientSize; j++) {
            fprintf(file, "%d ", result[i * quotientSize + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Function to perform column partitioning on GPU
void columnPartition(int *matrix, int numRows, int numCols, int partitionSize) {
    int quotientSize = numCols / partitionSize;

    // Allocate GPU memory for matrix and result
    int *d_matrix, *d_result;
    cudaMalloc((void **)&d_matrix, numRows * numCols * sizeof(int));
    cudaMemcpy(d_matrix, matrix, numRows * numCols * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_result, numRows * quotientSize * sizeof(int));

    // Launch the kernel with appropriate block and grid dimensions
    int blockSize = 256;
    int gridSize = (numRows + blockSize - 1) / blockSize;
    columnPartitionKernel<<<gridSize, blockSize>>>(d_matrix, numRows, numCols, partitionSize, d_result);

    // Copy the result back from GPU to CPU
    int *result = (int *)malloc(numRows * quotientSize * sizeof(int));
    cudaMemcpy(result, d_result, numRows * quotientSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the binary vectors for each row
    for (int i = 0; i < numRows; i++) {
        printf("Binary Vector for Row %d: ", i + 1);
        for (int j = 0; j < quotientSize; j++) {
            printf("%d ", result[i * quotientSize + j]);
        }
        printf("\n");
    }

    // Write the result matrix to a text file
    writeResultMatrixToFileGPU(result, numRows, quotientSize, "output_matrix_gpu.txt");

    // Free GPU memory
    cudaFree(d_matrix);
    cudaFree(d_result);
    free(result);
}

// Function to generate a random sparse matrix
void generateRandomSparseMatrix(int *matrix, int numRows, int numCols, double sparsity) {
    srand(42); // Seed the random number generator

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            // Generate a random number between 0 and 1
            double randomValue = (double)rand() / RAND_MAX;

            // Check if the random value is less than the desired sparsity
            if (randomValue < sparsity) {
                // Set a non-zero value
                matrix[i * numCols + j] = rand() % 10 + 1; // Values between 1 and 10
            } else {
                // Set a zero value
                matrix[i * numCols + j] = 0;
            }
        }
    }
}

int main() {
    // Example matrix dimensions
    int numRows = 100;
    int numCols = 100;
        // Allocate memory for the matrix
    int *matrix = (int *)malloc(numRows * numCols * sizeof(int));
    
    // Generate a random matrix
    generateRandomSparseMatrix(matrix, numRows, numCols, 0.25); // 25% sparse matrix

    // Perform column partitioning on GPU
    columnPartition((int *)matrix, numRows, numCols, 2);

    return 0;
}
