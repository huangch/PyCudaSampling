#ifndef __cudaSampling_H__
#define __cudaSampling_H__

#ifdef __cplusplus
extern "C" {
#endif

int cudaSampling(
	float * const r_data,
	float const * const h_data, int const width, int const height,
	int N, 
	int const * const s_x, int const * const s_y, float const * const s_t, float const * const s_s, bool const * const s_f, int s_length
);

#ifdef __cplusplus
}
#endif

#endif // __cudaSampling_H__
