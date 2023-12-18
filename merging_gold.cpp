#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cmath>

extern "C"
void computeGold(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height,int* grp_count, unsigned int* order_ds, int* group,  unsigned int* NNZ_pattern);
void WriteFile(unsigned int* Vector, char* file_name, unsigned int width, int height);
void Initialize(unsigned int* Vector, unsigned int width, unsigned int height, unsigned int val);
float similarity(unsigned int* Vector1, unsigned int* Vector2, unsigned int count);
void WriteFile_2(int* Vector, char* file_name, unsigned int width, int height);
void reorder(unsigned int* outVector,unsigned int* inVector, unsigned int* order, unsigned int* outPattern, unsigned int height, unsigned int blocks_across_width);
int ReadFile(unsigned int* Vector, char* file_name, unsigned int width, int height);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int ReadFile(unsigned int* Vector, char* file_name, unsigned int width, int height)
{
    unsigned int data_read = width * height;
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            fscanf(input, "%d", &(Vector[j+i*width]));
        }
     }
    fclose(input);
    return data_read;
}
void WriteFile(unsigned int* Vector, char* file_name, unsigned int width, int height)
{
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            fprintf(output,"%d ", Vector[j+i*width]);
        }
        fprintf(output,"\n");
    }
}
void WriteFile_2(int* Vector, char* file_name, unsigned int width, int height)
{
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            fprintf(output,"%d ", Vector[j+i*width]);
        }
        fprintf(output,"\n");
    }
}

void Initialize(unsigned int* Vector, unsigned int width, unsigned int height, unsigned int val)
{
    for(unsigned int i = 0; i < height * width; i++)
	{
		Vector[i]=val;
	}
}

float similarity(unsigned int* Vector1, unsigned int* Vector2, unsigned int count)
{
    float union_vec=0;
    float intersec_vec=0;
    for( int j = 0; j < count; ++j)
    {
        if(Vector1[j]+Vector2[j]>=1){
            union_vec+=1;
            if (Vector1[j]==Vector2[j]){
                intersec_vec+=1;
            }
        }
    }
    float jacc_simil=intersec_vec/union_vec;
    return jacc_simil;
}

void group_patt_up(unsigned int* outPattern,unsigned int* inVector,unsigned int NNZ_pattern, unsigned int width){
    NNZ_pattern=0;
    for( int j = 0; j < width; ++j)
    {
        if(outPattern[j]!=inVector[j]){
            outPattern[j]=1;
        }
        NNZ_pattern+=outPattern[j];
        //printf("%d", outPattern[j]);
    }
}
void reorder(unsigned int* outVector,unsigned int* inVector, unsigned int* order, unsigned int* outPattern, unsigned int height, unsigned int width){
    int k;
    for( int i = 0; i < height; i++){
        k=order[i];
        for( int j = 0; j < width; j++){
            outVector[j+i*width] = inVector[j+k*width];
        }
    }
}

void computeGold(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height,int* grp_count, unsigned int* order_ds, int* group,  unsigned int* NNZ_pattern)
{
    int NNZ_idx =0;
    float tau_idx;
    float limit_factor;
    int lambda0;
    grp_count[0]=0;

    for( int i = 0; i < height; i++) 
    {
        lambda0=0;
        //printf("flag[i]: %d\n", flag[i]);
        if(group[i]==-1){
            //Save initial pattern
            for( int j = 0; j < blocks_across_width; j++) 
            {
                outPattern[j+blocks_across_width*grp_count[0]] = inVector[j+i*blocks_across_width];
                group[i]=i;
                lambda0+=inVector[j+i*blocks_across_width];
                if(lambda0==0){
                    group[i]=-1;
                }
            }
            //printf("grp_count[0]: %d", grp_count[0]);
            if(lambda0!=0){
                order_ds[NNZ_idx]=i;
                NNZ_idx+=1;
                limit_factor=lambda0/(1-tau/2);
                //iterate over rows that not inserted into group and find similarity
                int start_csr=blocks_across_width-1-lambda0-limit_factor;
                if(start_csr<0){start_csr=0;}
                int end_csr=blocks_across_width-1-lambda0+limit_factor;
                if(end_csr>blocks_across_width-1){end_csr=blocks_across_width-1;}
                int start_index=group_indices_csr[start_csr];
                int end_index=group_indices_csr[end_csr];
                //printf("%d %d\n", start_index, end_index);
                for( int k = start_csr; k <end_index; k++){
                    int j=group_indices_data[k];
                    if((group[j]==-1)&&(i!=j)&&(NNZ_pattern[grp_count[0]]<=limit_factor)){
                        tau_idx = similarity( outPattern+grp_count[0]*blocks_across_width, inVector+j*blocks_across_width, blocks_across_width);
                        //printf("%f\n", tau_idx);
                        if(tau_idx>tau){
                            group_patt_up(outPattern+grp_count[0]*blocks_across_width, inVector+j*blocks_across_width, NNZ_pattern[grp_count[0]], blocks_across_width);
                            order_ds[NNZ_idx]=j;
                            group[j] =i;
                            blocks_height[grp_count[0]]+=1;
                            //printf("order_ds: %d\n", order_ds[NNZ_idx]);
                            NNZ_idx+=1;
                        }
                    }
                }
                grp_count[0]+=1;
            }
        }
    }
    for( int i = 0; i < height; i++) 
    {
        group[i]+=1;
    }
    reorder(outVector, inVector, order_ds, outPattern, height, blocks_across_width);
    grp_count[1]=NNZ_idx;
}

