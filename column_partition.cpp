#include <stdio.h>
#include <stdlib.h>

// Function to perform column partitioning
void columnPartition(int *matrix, int numRows, int numCols, int partitionSize) {
    // Assuming partitionSize is a multiple of numCols
    int quotientSize = numCols / partitionSize;
    // Iterate through each row
    for (int i = 0; i < numRows; i++) {
        // Create a K-dimensional binary vector for the row
        int *binaryVector = (int *)malloc(quotientSize * sizeof(int));
        // Initialize the binary vector with zeros
        for (int j = 0; j < quotientSize; j++) {
            binaryVector[j] = 0;
        }
        for (int j = 0; j < numCols; j++) {
            int blockIndex = j / partitionSize; // Determine the block index
            // Check if there is a non-zero element in the current block in the partition
            if (matrix[i * numCols + j] != 0) {
                binaryVector[blockIndex] = 1;
            }
        }
        // Print the binary vector for the current row
        printf("Binary Vector for Row %d: ", i + 1);
        for (int j = 0; j < quotientSize; j++) {
            printf("%d ", binaryVector[j]);
        }
        printf("\n");
        // Free allocated memory for the binary vector
        free(binaryVector);
    }
}

int main() {
    // Example matrix dimensions
    int numRows = 3;
    int numCols = 6;
    // Example matrix
    int matrix[3][6] = {
        {1, 0, 2, 0, 0, 0},
        {0, 4, 0, 5, 0, 0},
        {7, 0, 0, 0, 9, 0}
    };
    // Perform column partitioning
    columnPartition((int *)matrix, numRows, numCols, 2);
    return 0;
}
