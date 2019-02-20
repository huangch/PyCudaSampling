// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <math_constants.h>

// includes, common
#include "cudaCommon.h"

// includes headers
#include "cudaSampling.h"

// includes, kernels
#include "cudaSampling_kernel.cu"

#define MIN_EPSILON_ERROR 5e-3f

int initial(const int device) {
	CUDA_CALL(cudaSetDevice(device));

	// CUDA_CALL(cudaDeviceReset());

	return EXIT_SUCCESS;
}

int release() {

	CUDA_CALL(cudaThreadExit());

	return EXIT_SUCCESS;
}

int cudaSampling(
	float * const r_data,
	float const * const h_data, int const width, int const height,
	int N, 
	int const * const s_x, int const * const s_y, float const * const s_t, float const * const s_s, bool const * const s_f, int s_length
) {
    // allocate device memory for result
	unsigned int N2 = N*N;
	unsigned int r_size = N2*s_length*sizeof(float);
    float* d_data = NULL;
    
    CUDA_CALL( cudaMalloc( (void**) &d_data, r_size));

    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    CUDA_CALL( cudaMallocArray( &cu_array, &channelDesc, width, height )); 
    CUDA_CALL( cudaMemcpyToArray( cu_array, 0, 0, h_data, width * height*sizeof(float), cudaMemcpyHostToDevice));

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    CUDA_CALL( cudaBindTextureToArray( tex, cu_array, channelDesc));

	// Prepare the sampling data
    unsigned int *ds_x = NULL;
    unsigned int *ds_y = NULL;
    float *ds_t = NULL;
    float *ds_s = NULL;
    bool *ds_f = NULL;
    CUDA_CALL(cudaMalloc((void**)&ds_x, s_length*sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void**)&ds_y, s_length*sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void**)&ds_t, s_length*sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&ds_s, s_length*sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&ds_f, s_length*sizeof(bool)));
	CUDA_CALL(cudaMemcpy(ds_x, s_x, s_length*sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(ds_y, s_y, s_length*sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(ds_t, s_t, s_length*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(ds_s, s_s, s_length*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(ds_f, s_f, s_length*sizeof(bool), cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(((N2-1)/dimBlock.x)+1, ((s_length-1)/dimBlock.y)+1, 1);

    // execute the kernel
    samplingKernel<<< dimGrid, dimBlock, 0 >>>( d_data, width, height, N, ds_x, ds_y, ds_t, ds_s, ds_f, s_length);

    // check if kernel execution generated an error
    // cutilCheckMsg("Kernel execution failed");
    CUDA_GETLASTERROR();

    // copy result from device to host
    CUDA_CALL( cudaMemcpy( r_data, d_data, r_size, cudaMemcpyDeviceToHost) );

    CUDA_CALL(cudaFree(ds_x));
    CUDA_CALL(cudaFree(ds_y));
    CUDA_CALL(cudaFree(ds_t));
    CUDA_CALL(cudaFree(ds_s));
    CUDA_CALL(cudaFree(ds_f));
    CUDA_CALL(cudaFreeArray(cu_array));
    CUDA_CALL(cudaFree(d_data));
    
    return EXIT_SUCCESS;
}
