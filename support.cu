#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

//Compare matrix function to verify the GPU results with CPU results
bool CompareMatrices(int * reference, int * compare) {
    float errTol = 0.001f;

    unsigned int size = sizeof(reference);

    if (sizeof(reference) != sizeof(compare))
        return false;

    for (unsigned i = 0; i < size; i++)
    {
        float diff = abs(reference[i] - compare[i]);
        bool small= abs(reference[i]) < 1.0e-2f;

        if (small && diff > errTol)
            return false;
        else if (!small && abs(diff / reference[i]) > errTol)
            return false;
    }

    return true;
}


void verify(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float* csrData, float* inVector, float *outVector) {

    const float relativeTolerance = 1e-6;

    for(int row = 0; row < dim; ++row) {
        float result = 0.0f;
        unsigned int start = csrRowPtr[row];
        unsigned int end = csrRowPtr[row + 1];
        for(int elemIdx = start; elemIdx < end; ++elemIdx) {
            unsigned int colIdx = csrColIdx[elemIdx];
            result += csrData[elemIdx]*inVector[colIdx];
        }
        float relativeError = (result - outVector[row])/result;
        if (relativeError > relativeTolerance
                || relativeError < -relativeTolerance) {
            printf("TEST FAILED at row %d: CPU = %f, GPU = %f\n\n", 
                row, result, outVector[row]);
            exit(0);
        }
    }
    printf("TEST PASSED\n\n");

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

