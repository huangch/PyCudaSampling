#ifndef __CUDACOMMON_CUH__
#define __CUDACOMMON_CUH__

#include <stdio.h>
#define CULA_USE_CUDA_COMPLEX
#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas.h>
#include <curand.h>
#include <cusparse.h>
#include <curand.h>

#define CUDA_MIN 1.17549e-38F
#define CUDA_MAX 1.70141e+38F
#define CUDA_EPSILON 1.19209e-07F
#define BLOCKSIZE 32
#define BLOCKSIZE2 (32*32)

#define PRINTF(str) fprintf(stderr, "%s", str);

#ifdef USE_MATLAB
	#include <mex.h>
	#undef PRINTF 
	#define PRINTF(str) {mexPrintf(str); mexEvalString("drawnow;");}
#endif



#ifdef USE_PYTHON
	#ifdef _DEBUG
		#define RESTORE_DEBUG
		#undef _DEBUG
	#endif

	#include <Python.h>

	#ifdef RESTORE_DEBUG
		#define _DEBUG
		#undef RESTORE_DEBUG
	#endif

	#undef PRINTF 
	#define PRINTF(str) PySys_WriteStdout("%s", str)

	#include <numpy/arrayobject.h>

	template<typename PyType, typename CType> struct PyObject2CArray {
		PyObject2CArray(PyArrayObject const * const _obj, int const axis=0) {

			if (NULL != _obj) {
				aryObj = _obj;
				nd = aryObj->nd;

				if (3 == nd) {
					aryNum = aryObj->dimensions[0];
					rowNum = aryObj->dimensions[1];
					colNum = aryObj->dimensions[2];
				}
				else if(2 == nd) {
					aryNum = 1;
					rowNum = aryObj->dimensions[0];
					colNum = aryObj->dimensions[1];
				}
				else if (1 == nd) {
					aryNum = 1;
					rowNum = aryObj->dimensions[0];
					colNum = 1;
				}

				size = aryNum*rowNum*colNum;
				// size = rowNum*colNum;

				data = (CType*)malloc(size*sizeof(CType));
				
				if (3 == nd) {
					if (axis == 0) {
						for (unsigned int i = 0; i < colNum; i++) {
							for (unsigned int j = 0; j < rowNum; j++) {
								for (unsigned int k = 0; k < aryNum; k++) {
									data[k*rowNum*colNum + i*rowNum + j] = (CType)*((PyType*)PyArray_GETPTR3(aryObj, k, j, i));
									// (CType)*((PyType*)aryObj->data + j*colNum + i);
								}
							}
						}
					}
					else if (axis == 1){
						for (unsigned int i = 0; i < colNum; i++) {
							for (unsigned int j = 0; j < rowNum; j++) {
								for (unsigned int k = 0; k < aryNum; k++) {
									data[k*rowNum*colNum + j*colNum + i] = (CType)*((PyType*)PyArray_GETPTR3(aryObj, k, j, i));
									// (CType)*((PyType*)aryObj->data + j*colNum + i);
								}
							}
						}
					}
				}
				else 
				
				if (2 == nd) {
					if (axis == 0) {
						for (unsigned int i = 0; i < colNum; i++) {
							for (unsigned int j = 0; j < rowNum; j++) {
								data[i*rowNum + j] = (CType)*((PyType*)PyArray_GETPTR2(aryObj, j, i));
								// (CType)*((PyType*)aryObj->data + j*colNum + i);
							}
						}
					}
					else if(axis == 1){
						for (unsigned int i = 0; i < colNum; i++) {
							for (unsigned int j = 0; j < rowNum; j++) {
								data[j*colNum + i] = (CType)*((PyType*)PyArray_GETPTR2(aryObj, j, i));
								// (CType)*((PyType*)aryObj->data + j*colNum + i);
							}
						}
					}
				}
				else if (1 == nd) {
					for (unsigned int i = 0; i < rowNum; i++) {
						data[i] = (CType)*((PyType*)PyArray_GETPTR1(aryObj, i));  // (CType)(*((PyType*)aryObj->data + i));
					}
				}
			}
		}

		virtual ~PyObject2CArray() {
			free(data);
		}

		size_t getAryNum() const { return aryNum; }
		size_t getColNum() const { return colNum; }
		size_t getRowNum() const { return rowNum; }
		size_t getSize() const { return size; }
		size_t getND() const { return nd; }
		CType const * getData() { return data; }

	private:
		PyArrayObject const * aryObj;
		size_t aryNum;
		size_t colNum;
		size_t rowNum;
		size_t size;
		size_t nd;
		CType * data;


	};

	template<typename CType, typename PyType> struct CArray2PyObject {
		
		CArray2PyObject(void const * const data, size_t const aryNum, size_t const rowNum, size_t const colNum, int const PyTypeId, int const axis = 0) {
			dims[0] = aryNum;
			dims[1] = rowNum;
			dims[2] = colNum;
			aryObj = (PyArrayObject *)PyArray_FromDims(3, dims, PyTypeId);

			if (axis == 0) {
				for (unsigned int i = 0; i < colNum; i++) {
					for (unsigned int j = 0; j < rowNum; j++) {
						for (unsigned int k = 0; k < aryNum; k++) {
							*((PyType*)PyArray_GETPTR3(aryObj, k, j, i)) = *((CType*)data + k*rowNum*colNum + i*rowNum + j);
						}
					}
				}
			}
			else if (axis == 1) {
				for (unsigned int i = 0; i < colNum; i++) {
					for (unsigned int j = 0; j < rowNum; j++) {
						for (unsigned int k = 0; k < aryNum; k++) {
							*((PyType*)PyArray_GETPTR3(aryObj, k, j, i)) = *((CType*)data + k*rowNum*colNum + j*colNum + i);
						}
					}
				}
			}
		}

		

		CArray2PyObject(void const * const data, size_t const rowNum, size_t const colNum, int const PyTypeId, int const axis = 0) {
			dims[0] = rowNum;
			dims[1] = colNum;
			aryObj = (PyArrayObject *)PyArray_FromDims(2, dims, PyTypeId);

			if (axis == 0) {
				for (unsigned int i = 0; i < colNum; i++) {
					for (unsigned int j = 0; j < rowNum; j++) {
						*((PyType*)PyArray_GETPTR2(aryObj, j, i)) = *((CType*)data + i*rowNum + j);
					}
				}
			}
			else if (axis == 1) {
				for (unsigned int i = 0; i < colNum; i++) {
					for (unsigned int j = 0; j < rowNum; j++) {
						*((PyType*)PyArray_GETPTR2(aryObj, j, i)) = *((CType*)data + j*colNum + i);
					}
				}
			}
		}

		CArray2PyObject(void const * const data, size_t const size, int const PyTypeId) {
			dims[0] = size;
			dims[1] = 0;
			aryObj = (PyArrayObject *)PyArray_FromDims(1, dims, PyTypeId);

			for (unsigned int i = 0; i < size; i++) {
				*((PyType*)PyArray_GETPTR1(aryObj, i)) = *((CType*)data + i);
			}
		}

		PyArrayObject * getPyArrayObject() const { return aryObj; }

		virtual ~CArray2PyObject() {
		}


	private:
		PyArrayObject * aryObj;
		int dims[3];

	};
#endif

/*
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#define PRINTF(str) 
*/
#define CALL(x) {\
	if((x) != EXIT_SUCCESS) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		throw EXIT_FAILURE;\
	}\
}

#define CUDA_CALL(x) {\
	if((x) != cudaSuccess) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUDA_CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		sprintf(DebugWriteLineBuf, "%s\n",cudaGetErrorString(cudaGetLastError()));\
		PRINTF(DebugWriteLineBuf);\
		throw EXIT_FAILURE;\
	}\
}

#define LOGGER(type, x, filename, m, n) { \
	type * logger_var_##x = (type *)malloc(m*n*sizeof(type)); \
	CUDA_CALL(cudaMemcpy(logger_var_##x, x, m*n*sizeof(type), cudaMemcpyDeviceToHost)); \
	FILE * logger_fp_##x = fopen(filename, "w"); \
	for (unsigned int i = 0; i < m; i++) { \
		for (unsigned int j = 0; j < n; j++) { \
			fprintf(logger_fp_##x, "%f", (float)logger_var_##x[j*m + i]); \
			fprintf(logger_fp_##x, "%s", (j == n-1)? "\n" : ", "); \
				} \
		} \
	fclose(logger_fp_##x); \
	free(logger_var_##x); \
}

#define LOGGER_CPU(type, x, filename, m, n) { \
	type * logger_var_##x = (type *)malloc(m*n*sizeof(type)); \
	CUDA_CALL(cudaMemcpy(logger_var_##x, x, m*n*sizeof(type), cudaMemcpyDeviceToHost)); \
	FILE * logger_fp_##x = fopen(filename, "w"); \
	for (unsigned int i = 0; i < m; i++) { \
		for (unsigned int j = 0; j < n; j++) { \
			fprintf(logger_fp_##x, "%f", (float)logger_var_##x[j*m + i]); \
			fprintf(logger_fp_##x, "%s", (j == n-1)? "\n" : ", "); \
						} \
			} \
	fclose(logger_fp_##x); \
	free(logger_var_##x); \
	}

/*
#define CUDA_FREE(x) {\
	if((cudaFree(x)) != cudaSuccess) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUDA_FREE Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		sprintf(DebugWriteLineBuf, "%s\n",cudaGetErrorString(cudaGetLastError()));\
		PRINTF(DebugWriteLineBuf);\
		cublasShutdown();\
		cudaDeviceReset();\
		return EXIT_FAILURE;\
	}\
}
*/


#define CULA_CALL(x) {\
	culaStatus CULA_CALL_culaStatus;\
	CULA_CALL_culaStatus = (x);\
	if(CULA_CALL_culaStatus) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CULA_CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		culaGetErrorInfoString(CULA_CALL_culaStatus, culaGetErrorInfo(), DebugWriteLineBuf, sizeof(DebugWriteLineBuf));\
		PRINTF(DebugWriteLineBuf);\
		PRINTF("\n");\
		culaShutdown();\
		cublasShutdown();\
		cudaDeviceReset();\
		return EXIT_FAILURE;\
	}\
}


/*
#define CULA_CALL(x) (x)
*/
/*
#define CURAND_CALL(x) {\
	if((x) != CURAND_STATUS_SUCCESS) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CURAND_CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		cublasShutdown();\
		cudaDeviceReset();\
		return EXIT_FAILURE;\
	}\
}
*/

/*
#define CUFFT_CALL(x) {\
	cufftResult CUFFT_CALL_cufftStatus = (x);\
	if(CUFFT_CALL_cufftStatus != CUFFT_SUCCESS) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUFFT_CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		switch (CUFFT_CALL_cufftStatus) {\
			case CUFFT_INVALID_PLAN:   sprintf(DebugWriteLineBuf, "CUFFT_INVALID_PLAN\n"); break;\
			case CUFFT_ALLOC_FAILED:   sprintf(DebugWriteLineBuf, "CUFFT_ALLOC_FAILED\n"); break;\
			case CUFFT_INVALID_TYPE:   sprintf(DebugWriteLineBuf, "CUFFT_INVALID_TYPE\n"); break;\
			case CUFFT_INVALID_VALUE:  sprintf(DebugWriteLineBuf, "CUFFT_INVALID_VALUE\n"); break;\
			case CUFFT_INTERNAL_ERROR: sprintf(DebugWriteLineBuf, "CUFFT_INTERNAL_ERROR\n"); break;\
			case CUFFT_EXEC_FAILED:    sprintf(DebugWriteLineBuf, "CUFFT_EXEC_FAILED\n"); break;\
			case CUFFT_SETUP_FAILED:   sprintf(DebugWriteLineBuf, "CUFFT_SETUP_FAILED\n"); break;\
			case CUFFT_INVALID_SIZE:   sprintf(DebugWriteLineBuf, "CUFFT_INVALID_SIZE\n"); break;\
			case CUFFT_UNALIGNED_DATA: sprintf(DebugWriteLineBuf, "CUFFT_UNALIGNED_DATA\n"); break;\
			default: sprintf(DebugWriteLineBuf, "CUFFT Unknown error code\n"); break;\
		}\
		PRINTF(DebugWriteLineBuf);\
		cublasShutdown();\
		cudaDeviceReset();\
		return EXIT_FAILURE;\
	}\
}
*/
/*
#define CUSPARSE_CALL(x) {\
	if((x) != CUSPARSE_STATUS_SUCCESS) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUSPARSE_CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		cublasShutdown();\
		cudaDeviceReset();\
		return EXIT_FAILURE;\
	}\
}
*/


#define CUBLAS_CALL(x) {\
	(x);\
	if(cublasGetError() != CUBLAS_STATUS_SUCCESS) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUBLAS_CALL Error at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		throw EXIT_FAILURE;\
	}\
}

#define CUDAMEMINFO(x) {\
	size_t free_byte;\
	size_t total_byte;\
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte );\
	if (cudaSuccess != cuda_status) {\
		PRINTF("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));\
		return EXIT_FAILURE;\
	}\
	double free_db = (double)free_byte;\
	double total_db = (double)total_byte;\
	double used_db = total_db - free_db;\
	PRINTF("(%s) GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n", x, used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);\
}

/*
#define ANTICUDAMEMFRAG(r) {\
	size_t free_byte;\
	size_t total_byte;\
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte );\
	if (cudaSuccess != cuda_status) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "cudaMemGetInfo fails at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		CUBLAS_CALL(cublasShutdown());\
		CUDA_CALL(cudaDeviceReset());\
		return EXIT_FAILURE;\
	}\
	float free_db = (float)free_byte;\
	float total_db = (float)total_byte;\
	float used_db = total_db - free_db;\
	if((free_db/total_db) < (float)r) CUDA_CALL(cudaDeviceReset());\
}
*/

#define CUDA_GETLASTERROR() {\
	cudaError_t err = cudaGetLastError();\
	if(cudaSuccess != err) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUDA_GETLASTERROR at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		sprintf(DebugWriteLineBuf, "%s\n",cudaGetErrorString(err));\
		PRINTF(DebugWriteLineBuf);\
		throw EXIT_FAILURE;\
	}\
}


#define CUDA_PEEKLASTERROR() {\
	cudaError_t err = cudaPeekAtLastError();\
	if(cudaSuccess != err) {\
		char DebugWriteLineBuf[256];\
		sprintf(DebugWriteLineBuf, "CUDA_PEEKLASTERROR at %s:%d\n",__FILE__,__LINE__);\
		PRINTF(DebugWriteLineBuf);\
		sprintf(DebugWriteLineBuf, "%s\n",cudaGetErrorString(err));\
		PRINTF(DebugWriteLineBuf);\
		throw EXIT_FAILURE;\
	}\
}

#endif // __CUDACOMMON_CUH__


