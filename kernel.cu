/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

const unsigned int BLOCK_SIZE = 512;


__global__ void Histo_Kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements,
                             unsigned int num_bins) {


    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    //using shared memory
    extern __shared__ unsigned int histo_private[];
    for(int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        histo_private[i] = 0;
    }

    __syncthreads();


    while (index < num_elements) {
//        atomicAdd(&(histo_private[(input[index])]), 1);
        atomicAdd(&(histo_private[(input[index])]), 1);
        index += stride;
    }
    __syncthreads();

    //create final histogram using atomic add
    for(int j = threadIdx.x; j < num_bins; j += blockDim.x) {
        atomicAdd(&(bins[j]), histo_private[j]);
    }
    __syncthreads();

}


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    Histo_Kernel<<<ceil(num_elements/BLOCK_SIZE), BLOCK_SIZE,
            sizeof(unsigned int)*num_bins>>>(input, bins, num_elements, num_bins);

}


