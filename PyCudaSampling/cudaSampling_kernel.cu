#ifndef _CUDARETINALSAMPLING_KERNEL_H_
#define _CUDARETINALSAMPLING_KERNEL_H_

// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;


__global__ void samplingKernel(
	float *g_odata, const unsigned int width, const unsigned int height,
	const unsigned int N, 
	const unsigned int *s_x, const unsigned int *s_y, const float *s_t, const float *s_s, const bool *s_f, const unsigned int s_length
) {
    // calculate normalized texture coordinates
	const unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int tidy = blockIdx.y*blockDim.y+threadIdx.y;
	const unsigned int N2 = (int)roundf((float)N/2.0F);
	const unsigned int NN = N*N;

	if((tidx < NN) && (tidy < s_length)) {
		const int s = tidy;
		const int cx = s_x[s];
		const int cy = s_y[s];
		const float t = s_t[s]*CUDART_PI_F/180.0F;
		const float scale = 1.0F/s_s[s];
		const int flip = s_f[s]? -1: 1;
		const int dx = (tidx%N)-(N2-1);
		const int dy = (tidx/N)-(N2-1);

		float u = scale*(float)dx;
		float v = scale*(float)dy;

		// sampling coordinates
		float tu = flip*(u*cosf(t)-v*sinf(t))+cx;
		float tv = v*cosf(t)+u*sinf(t)+cy;

		// read from texture and write to global memory
		g_odata[tidy*NN+tidx] = tex2D(tex, tu, tv);
	}

	__syncthreads();
}

#endif // #ifndef _CUDARETINALSAMPLING_KERNEL_H_
