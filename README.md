1D Saad's Algorithm Implementation
----------------------------------

Owners: Pragnya Nalla, Sourav Kumar Ghosh, Gopikrishnan Ravindran Nair, Sreelakshmi Vinod
        University of Minnesota
Date: 20 Dec, 2023

Created for the course EE5351, Applied Parallel Programming

------------------------------------------------------------------------------------------

The code takes an input sparse matrix and takes it through a blocking algorithm. the output is reordered data.
Makefile given along with the code. 

To compile and run:
> make
> ./1dsa 0.5

The value of 0.5 is the similarity threshold whihc can be any value from 0 to 1. 
Need to create a folder "Data" inside the directory in whihc the code is run. 

Input: 
Currently we are generating random sparse matrix of given sparsity which can be changed in the main code.

Files:
main.cu: the main function which integrated all stages.
support.cu: Supporting function definitions.
support.h: supporting header files.
column_partition_updated.cpp: The CPU implementation for column partitioning.
column_partitioning_updated.cu: The GPU kernel implementation for column partitioning.
hash.cpp: CPU implementation for hash-based compression.
hash.cu: GPU implementation for hash-based compression.
merging_gold.cpp: CPU implementation for two-step merging.
merging.cu: GPU kernel for two-step merging.
merging.h: supporting header files for merging.

