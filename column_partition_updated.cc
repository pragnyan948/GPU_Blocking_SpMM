#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to perform column partitioning on CPU
void columnPartitionCPU(int *matrix, int numRows, int numCols, int partitionSize, int *result) {
    // Assuming partitionSize is a multiple of numCols
    int quotientSize = numCols / partitionSize;

    // Iterate through each row
    for (int i = 0; i < numRows; i++) {
        // Create a K-dimensional binary vector for the row
        int *binaryVector = result + i * quotientSize;

        // Initialize the binary vector with zeros
        for (int j = 0; j < quotientSize; j++) {
            binaryVector[j] = 0;
        }

        // Check if there is a non-zero element in the current block in the partition
        for (int j = 0; j < numCols; j++) {
            int blockIndex = j / partitionSize; // Determine the block index

            // Check if there is a non-zero element in the current block in the partition
            if (matrix[i * numCols + j] != 0) {
                binaryVector[blockIndex] = 1;
            }
        }
    }
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
// Function to print a matrix
void printMatrix(int *matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%d ", matrix[i * numCols + j]);
        }
        printf("\n");
    }
}


// Function to write a matrix to a text file
void writeMatrixToFile(int *matrix, int numRows, int numCols, const char *filename) {
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            fprintf(file, "%d ", matrix[i * numCols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


// Function to write a result matrix to a text file
void writeResultMatrixToFile(int *resultCPU, int numRows, int quotientSize, const char *filename) {
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < quotientSize; j++) {
            fprintf(file, "%d ", resultCPU[i * quotientSize + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


int main() {
    // Example matrix dimensions
    int numRows = 100;
    int numCols = 100;
        // Allocate memory for the matrix
    int *matrix = (int *)malloc(numRows * numCols * sizeof(int));
    
    // Generate a random matrix
    generateRandomSparseMatrix(matrix, numRows, numCols, 0.25); // 25% sparse matrix

    // Write the generated matrix to a text file
    writeMatrixToFile(matrix, numRows, numCols, "matrix.txt");

    // Perform column partitioning on CPU
    int partitionSize = 2; // patrition size is 2, you can modify it as necessary
    int quotientSize = numCols / partitionSize;
    int *resultCPU = (int *)malloc(numRows * quotientSize * sizeof(int));
    columnPartitionCPU((int *)matrix, numRows, numCols, partitionSize, resultCPU);

    // Print the binary vectors for each row on CPU
    for (int i = 0; i < numRows; i++) {
        printf("Binary Vector for Row %d: ", i + 1);
        for (int j = 0; j < quotientSize; j++) {
            printf("%d ", resultCPU[i * quotientSize + j]);
        }
        printf("\n");
    }

     // Write the result matrix to a text file
    writeResultMatrixToFile(resultCPU, numRows, quotientSize, "output_binary_matrix_after_column_partition.txt");
}
