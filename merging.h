#ifndef _MERGE_H_
#define _MERGE_H_

#include <sys/time.h>
#define BLOCK_SIZE 1024


typedef struct {
    unsigned int* outVector;
    unsigned int* blocks_height;
    unsigned int* outPattern;
    unsigned int* midPattern;
    unsigned int* pattern_indices;
} Out_1DSAAD;

typedef struct {
    unsigned int* inVector;
    int* group;
    unsigned int* group_indices_data;
    unsigned int* group_indices_csr;
} In_1DSAAD;

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#endif // _MERGE_H_

