#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include<stdio.h>

#define ROWS 3


void hash_fn(int *value_mat, int numRows, int numCols, int* row_idx , int* grp_ptr)
{
    //allocation for hash array   
    int *hash =  (int *)malloc(ROWS * sizeof(int));
    
    
    for(int i= 0 ; i< numRows ; i++){
        hash[i] = 0;
    }
    

    //calculate hash
    for (int row= 0 ; row< numRows ; row++){

        for (int k = 0; k<numCols ; k++){
             if(value_mat[row*numCols + k] !=0)
                 hash[row] +=1; 
                 
        }

    }
    
    ///generating the groups based on hash. Similar to CSR format
    int ptr;
    int idx_ptr =0;
    grp_ptr[idx_ptr] = ptr;
    idx_ptr+=1;
    for (int j = numCols ; j >=0 ; j--){
       
        //printf("here3,%i,\n",j);
         
         
        for (int row= 0 ; row< numRows ; row++){
             //printf("here2,%i,%i,%i,\n",row,hash[row],j);
         if(hash[row] == j){
            
             row_idx[ptr] = row;       
             ptr++;
             //printf("here1,%i,%i,\n",row,ptr);
         }
     }
    grp_ptr[idx_ptr] = ptr;
    idx_ptr+=1;
     
    }
   

}




int main() {
    // Example matrix dimensions
    int numRows = 3;
    int numCols = 6;
    // Example matrix
   int matrix[3][6] = {
        {1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0}
    };
    
    
    int *row_idx =  (int *)malloc(numRows * sizeof(int));//stores grouped row indices 
    int *grp_ptr =  (int *)malloc(numCols * sizeof(int));//stores group pointer
    hash_fn((int *)matrix, numRows, numCols,row_idx,grp_ptr);

///print output
    for (int i = 0; i < numRows; i++) {
        printf("row_idx for Row %d: ", i );
       
        printf("%d ", row_idx[i]);
        
        printf("\n");
    }

    for (int i = 0; i < numCols; i++) {
        printf("grp_ptr for Row %d: ", i );
       
        printf("%d ", grp_ptr[i]);
        
        printf("\n");
    }
    return 0;
}