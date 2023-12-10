#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
extern "C"
void computeGold(unsigned int* outVector, unsigned int* inVector,
                unsigned int* blocks_height, unsigned int* outPattern,
                unsigned int* group_indices_data, unsigned int* group_indices_csr,
                float tau, unsigned int blocks_across_width, unsigned int height,unsigned int grp_count);
void Initialize_zero(unsigned int* Vector, unsigned width, unsigned height);
float similarity(unsigned int* Vector1, unsigned int* Vector2, unsigned int count);
void WriteFile(unsigned int* Vector, char* file_name, unsigned int width, int height);
void reorder(unsigned int* outVector,unsigned int* inVector, unsigned int* order, unsigned int* outPattern, unsigned int height, unsigned int blocks_across_width);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

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
        union_vec+=1;
        if (Vector1[j] ==Vector2[j]){
            intersec_vec+=1;
        }
        else{
            union_vec+=1;
        }
    }
    float jacc_simil=intersec_vec/union_vec;
    return jacc_simil;
}

void group_patt_up(unsigned int* outPattern,unsigned int* inVector,unsigned int NNZ_pattern, unsigned int width){
     for( int j = 0; j < width; ++j)
    {
        if(outPattern[j]!=inVector[j]){
            outPattern[j]=1;
        }
        NNZ_pattern+=outPattern[j];
        printf("%d", outPattern[j]);
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
                float tau, unsigned int blocks_across_width, unsigned int height, unsigned int grp_count)
{
    int idx =0;
    float tau_idx, limit_factor;
    int lambda0;

    unsigned int* flag = (unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* order_ds = (unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* group = (unsigned int*) malloc(height*sizeof(unsigned int));
    unsigned int* NNZ_pattern = (unsigned int*) malloc(height*sizeof(unsigned int));
   
    Initialize(flag, 1, height, 0);
    Initialize(NNZ_pattern, 1, height, 0);
    Initialize(blocks_height, 1, height, 1);

    for( int i = 0; i < height; i++) 
    {
        lambda0=0;
        //printf("flag[i]: %d\n", flag[i]);
        if(flag[i]==0){
            //Save initial pattern
            for( int j = 0; j < blocks_across_width; j++) 
            {
                outPattern[j+blocks_across_width*grp_count] = inVector[j+i*blocks_across_width];
                group[i]=grp_count;
                lambda0+=outPattern[j+blocks_across_width*grp_count];
                flag[i]=1;
            }
            printf("grp_count: %d\n", grp_count);
            order_ds[idx]=i;
            idx+=1;
            if(lambda0!=0){
                limit_factor=lambda0/(1-tau/2);
                //iterate over rows that not inserted into group and find similarity
                int start_csr=4-lambda0-1;
                if(start_csr<0){start_csr=0;}
                int end_csr=4-lambda0+2;
                if(end_csr>4){end_csr=4;}
                int start_index=group_indices_csr[start_csr];
                int end_index=group_indices_csr[end_csr];
                for( int k = start_index; k < end_index; k++){
                    int j=group_indices_data[k];
                    printf("j:%d\n", j);
                    if((flag[j]==0) && (NNZ_pattern[grp_count]<limit_factor)&&(i!=j)){
                        tau_idx = similarity(inVector+i*blocks_across_width, inVector+j*blocks_across_width, blocks_across_width);
                        printf("%f\n", tau_idx);
                        if(tau_idx>tau){
                            group_patt_up(outPattern+grp_count*blocks_across_width, inVector+j*blocks_across_width, NNZ_pattern[grp_count], blocks_across_width);
                            order_ds[idx]=j;
                            group[j] =grp_count;
                            flag[j] =1;
                            blocks_height[grp_count]+=1;
                            //printf("order_ds: %d\n", order_ds[idx]);
                            idx+=1;
                        }
                    }
                }
            }
            grp_count+=1;
        }
    }
   
    reorder(outVector, inVector, order_ds, outPattern, height, blocks_across_width);

    char* file_name_param[4] ={"./order_ds.txt", "./group.txt","./blocks_height.txt", "./outPattern.txt"};
    WriteFile(order_ds, file_name_param[0], 1,idx);
    WriteFile(group, file_name_param[1], 1,idx);
    WriteFile(blocks_height, file_name_param[2], 1,grp_count);
    WriteFile(outPattern, file_name_param[3], blocks_across_width, grp_count);
}

