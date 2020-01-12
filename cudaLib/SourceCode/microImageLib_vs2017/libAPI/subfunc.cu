#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// Includes CUDA
//#include <cuda.h>
#include <cuda_runtime.h>
//
#include <memory.h>
#include "device_launch_parameters.h"
#include <cufft.h>
//#include <cufftw.h> // ** cuFFT also comes with CPU-version FFTW, but seems not to work when image size is large.
#include "fftw3.h"


#include "cukernel.cuh"
extern "C" {
#include "powell.h"
}
#include "apifunc_internal.h"

#define SMALLVALUE 0.01
#define NDIM 12
cudaError_t cudaStatus;
#define cudaCheckErrors(msg) \
    do { \
        cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(cudaStatus), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
								        } \
				    } while (0)

extern "C"
bool isPow2(int x)
{
	return ((x&(x - 1)) == 0);
};

//Round a / b to nearest higher integer value
inline long long int iDivUp(long long int a, long long int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline long long int iAlignUp(long long int a, long long int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}

int snapTransformSize(int dataSize)//
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 128)
	{
		return hiPOT;
	}
	else
	{
		return iAlignUp(dataSize, 64);
	}
}

//////////////// Basic math functions  /////////////////
// CPU functions
// sum
template <class T>
double sumcpu(T *h_idata, size_t totalSize) {
	double sumValue = 0;
	for (size_t i = 0; i < totalSize; i++) {
		sumValue += (double)h_idata[i];
	}
	return sumValue;
}
template double sumcpu<int>(int *h_idata, size_t totalSize);
template double sumcpu<float>(float *h_idata, size_t totalSize);
template double sumcpu<double>(double *h_idata, size_t totalSize);
// add
template <class T>
void addcpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize){
	for (size_t i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] + h_idata2[i];
}
template void addcpu<int>(int *h_odata, int *h_idata1, int *h_idata2, size_t totalSize);
template void addcpu<float>(float *h_odata, float *h_idata1, float *h_idata2, size_t totalSize);
template void addcpu<double>(double *h_odata, double *h_idata1, double *h_idata2, size_t totalSize);

template <class T>
void addvaluecpu(T *h_odata, T *h_idata1, T h_idata2, size_t totalSize){
	const T b = h_idata2;
	for (size_t i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] + b;
}
template void addvaluecpu<int>(int *h_odata, int *h_idata1, int h_idata2, size_t totalSize);
template void addvaluecpu<float>(float *h_odata, float *h_idata1, float h_idata2, size_t totalSize);
template void addvaluecpu<double>(double *h_odata, double *h_idata1, double h_idata2, size_t totalSize);
// subtract
template <class T>
void subcpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize){
	for (size_t i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] - h_idata2[i];
}
template void subcpu<int>(int *h_odata, int *h_idata1, int *h_idata2, size_t totalSize);
template void subcpu<float>(float *h_odata, float *h_idata1, float *h_idata2, size_t totalSize);
template void subcpu<double>(double *h_odata, double *h_idata1, double *h_idata2, size_t totalSize);
// multiply
template <class T>
void multicpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize){
	for (size_t i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] * h_idata2[i];
}
template void multicpu<int>(int *h_odata, int *h_idata1, int *h_idata2, size_t totalSize);
template void multicpu<float>(float *h_odata, float *h_idata1, float *h_idata2, size_t totalSize);
template void multicpu<double>(double *h_odata, double *h_idata1, double *h_idata2, size_t totalSize);
//divide
template <class T>
void divcpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize){
	for (size_t i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] / h_idata2[i];
}
template void divcpu<int>(int *h_odata, int *h_idata1, int *h_idata2, size_t totalSize);
template void divcpu<float>(float *h_odata, float *h_idata1, float *h_idata2, size_t totalSize);
template void divcpu<double>(double *h_odata, double *h_idata1, double *h_idata2, size_t totalSize);

template <class T>
void multivaluecpu(T *h_odata, T *h_idata1, T h_idata2, size_t totalSize){
	for (size_t i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] * h_idata2;
}
template void multivaluecpu<int>(int *h_odata, int *h_idata1, int h_idata2, size_t totalSize);
template void multivaluecpu<float>(float *h_odata, float *h_idata1, float h_idata2, size_t totalSize);
template void multivaluecpu<double>(double *h_odata, double *h_idata1, double h_idata2, size_t totalSize);

extern "C"
void multicomplexcpu(fComplex *h_odata, fComplex *h_idata1, fComplex *h_idata2, size_t totalSize){
	fComplex a;
	fComplex b;
	for (size_t i = 0; i < totalSize; i++){
		a = h_idata1[i];
		b = h_idata2[i];
		h_odata[i].x = a.x*b.x - a.y*b.y;
		h_odata[i].y = a.x*b.y + a.y*b.x;
	}		
}

// max3Dcpu: find max value and coordinates
template <class T>
T max3Dcpu(size_t *corXYZ, T *h_idata, size_t sx, size_t sy, size_t sz) {
	T peakValue = h_idata[0];
	T t;
	size_t sx0 = 0, sy0 = 0, sz0 = 0;
	for (size_t i = 0; i < sx; i++) {
		for (size_t j = 0; j < sy; j++) {
			for (size_t k = 0; k < sz; k++) {
				t = h_idata[i + j * sx + k * sx * sy];
				if (peakValue < t) {
					peakValue = t;
					sx0 = i;
					sy0 = j;
					sz0 = k;
				}
			}
		}
	}

	corXYZ[0] = sx0; corXYZ[1] = sy0; corXYZ[2] = sz0;
	return peakValue;
}
template int max3Dcpu<int>(size_t *corXYZ, int *h_idata, size_t sx, size_t sy, size_t sz);
template float max3Dcpu<float>(size_t *corXYZ, float *h_idata, size_t sx, size_t sy, size_t sz);
template double max3Dcpu<double>(size_t *corXYZ, double *h_idata, size_t sx, size_t sy, size_t sz);

// max with a single value
template <class T>
void maxvaluecpu(T *h_odata, T *h_idata1, T h_idata2, size_t totalSize) {
	T a;
	const T b = h_idata2;
	for (size_t i = 0; i < totalSize; i++) {
		a = h_idata1[i];
		h_odata[i] = (a > b) ? a : b;
	}
}
template void maxvaluecpu<int>(int *d_odata, int *d_idata1, int d_idata2, size_t totalSize);
template void maxvaluecpu<float>(float *d_odata, float *d_idata1, float d_idata2, size_t totalSize);
template void maxvaluecpu<double>(double *d_odata, double *d_idata1, double d_idata2, size_t totalSize);

template <class T>
void changestorageordercpu(T *h_odata, T *h_idata, size_t sx, size_t sy, size_t sz, int orderMode) {
	//orderMode
	// 1: change tiff storage order to C storage order
	//-1: change C storage order to tiff storage order
	if (orderMode == 1) {
		for (size_t i = 0; i < sx; i++) {
			for (size_t j = 0; j < sy; j++) {
				for (size_t k = 0; k < sz; k++) {
					h_odata[i*sy*sz + j*sz + k] = h_idata[k*sy*sx + j*sx + i];
				}
			}
		}
	}
	else if (orderMode == -1) {//change C storage order to tiff storage order:
		for (size_t i = 0; i < sx; i++) {
			for (size_t j = 0; j < sy; j++) {
				for (size_t k = 0; k < sz; k++) {
					h_odata[k*sy*sx + j*sx + i] = h_idata[i*sy*sz + j*sz + k];
				}
			}
		}
	}
}
template void changestorageordercpu<int>(int *h_odata, int *h_idata, size_t sx, size_t sy, size_t sz, int orderMode);
template void changestorageordercpu<float>(float *h_odata, float *h_idata, size_t sx, size_t sy, size_t sz, int orderMode);
template void changestorageordercpu<double>(double *h_odata, double *h_idata, size_t sx, size_t sy, size_t sz, int orderMode);


///// GPU functions
//add
template <class T>
void add3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	add3Dkernel<T> <<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void add3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, size_t sx, size_t sy, size_t sz);
template void add3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, size_t sx, size_t sy, size_t sz);
template void add3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, size_t sx, size_t sy, size_t sz);

// add with a single value
template <class T>
void addvaluegpu(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	addvaluekernel<T> <<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void addvaluegpu<int>(int *d_odata, int *d_idata1, int d_idata2, size_t sx, size_t sy, size_t sz);
template void addvaluegpu<float>(float *d_odata, float *d_idata1, float d_idata2, size_t sx, size_t sy, size_t sz);
template void addvaluegpu<double>(double *d_odata, double *d_idata1, double d_idata2, size_t sx, size_t sy, size_t sz);

//subtract
template <class T>
void sub3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	sub3Dkernel<T> <<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void sub3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, size_t sx, size_t sy, size_t sz);
template void sub3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, size_t sx, size_t sy, size_t sz);
template void sub3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, size_t sx, size_t sy, size_t sz);


//multiply
template <class T>
void multi3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multi3Dkernel<T> <<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void multi3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, size_t sx, size_t sy, size_t sz);
template void multi3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, size_t sx, size_t sy, size_t sz);
template void multi3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, size_t sx, size_t sy, size_t sz);

// multiply with a single value
template <class T>
void multivaluegpu(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multivaluekernel<T> <<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void multivaluegpu<int>(int *d_odata, int *d_idata1, int d_idata2, size_t sx, size_t sy, size_t sz);
template void multivaluegpu<float>(float *d_odata, float *d_idata1, float d_idata2, size_t sx, size_t sy, size_t sz);
template void multivaluegpu<double>(double *d_odata, double *d_idata1, double d_idata2, size_t sx, size_t sy, size_t sz);

//multiply float complex
extern "C"
void multicomplex3Dgpu(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multicomplex3Dkernel<<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}

//multiply float complex and do normalization
extern "C"
void multicomplexnorm3Dgpu(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multicomplexnorm3Dkernel <<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}


//multiply double complex
extern "C"
void multidcomplex3Dgpu(dComplex *d_odata, dComplex *d_idata1, dComplex *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multidcomplex3Dkernel<<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}

//divide
template <class T>
void div3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	div3Dkernel<T> <<<grids, threads>>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void div3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, size_t sx, size_t sy, size_t sz);
template void div3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, size_t sx, size_t sy, size_t sz);
template void div3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, size_t sx, size_t sy, size_t sz);

//conjugation of complex
extern "C"
void conj3Dgpu(fComplex *d_odata, fComplex *d_idata, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	conj3Dkernel <<<grids, threads>>>(d_odata, d_idata, sx, sy, sz);
	cudaThreadSynchronize();
}

// sumarization
// sumgpu 1: small data size
template <class T>
T sumgpu(T *d_idata, int totalSize){
	int gridSize = iDivUp(totalSize, blockSize);
	bool nIsPow2 = isPow2(totalSize);
	int smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(T) : blockSize * sizeof(T);
	T *h_temp = NULL, *d_temp = NULL;
	h_temp = (T *)malloc(gridSize * sizeof(T));
	cudaMalloc((void **)&d_temp, gridSize * sizeof(T));

	sumgpukernel<T><<<gridSize, blockSize, smemSize>>>(
		d_idata,
		d_temp,
		totalSize,
		nIsPow2
		);
	cudaThreadSynchronize();
	cudaMemcpy(h_temp, d_temp, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
	T sumValue = 0;
	for (int i = 0; i < gridSize; i++){
		sumValue += h_temp[i];
	}
	free(h_temp);
	cudaFree(d_temp);
	return sumValue;
}

template int sumgpu<int>(int *d_idata,  int totalSize);
template float sumgpu<float>(float *d_idata,  int totalSize);
template double sumgpu<double>(double *d_idata,  int totalSize);

// sumgpu 2: huge data size (3D data)
template <class T>
double sum3Dgpu(T *d_idata, size_t sx, size_t sy, size_t sz){
	size_t sxy = sx * sy;
	double *h_temp = NULL, *d_temp = NULL;
	h_temp = (double *)malloc(sxy * sizeof(double));
	cudaMalloc((void **)&d_temp, sxy * sizeof(double));
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	reduceZ<T> <<<grids, threads >>>(d_idata, d_temp, sx, sy, sz); 
	cudaThreadSynchronize();
	cudaMemcpy(h_temp, d_temp, sxy * sizeof(double), cudaMemcpyDeviceToHost); 
	double sumValue = 0; 
	for (size_t i = 0; i < sxy; i++)
		sumValue += h_temp[i];
	free(h_temp); 
	cudaFree(d_temp);
	return sumValue;
}

template double sum3Dgpu<int>(int *d_idata,  size_t sx, size_t sy, size_t sz);
template double sum3Dgpu<float>(float *d_idata,  size_t sx, size_t sy, size_t sz);
template double sum3Dgpu<double>(double *d_idata,  size_t sx, size_t sy, size_t sz);

// sumgpu 3: small data (1D data)
template <class T>
T sumgpu1D(T *d_idata,  size_t totalSize){
	const size_t r = 5; // a rough number, need further optimization
	size_t tempSize = r * blockSize;
	T *h_temp = NULL, *d_temp = NULL;
	h_temp = (T *)malloc(tempSize * sizeof(T));
	cudaMalloc((void **)&d_temp, tempSize * sizeof(T));
	sumgpu1Dkernel<T> <<<r, blockSize >>>(
		d_idata,
		d_temp,
		totalSize
		);
	cudaThreadSynchronize();
	cudaMemcpy(h_temp, d_temp, tempSize * sizeof(T), cudaMemcpyDeviceToHost);
	T sumValue = 0;
	for (int i = 0; i < tempSize; i++){
		sumValue += h_temp[i];
	}
	free(h_temp);
	cudaFree(d_temp);
	return sumValue;
}
template int sumgpu1D<int>(int *d_idata,  size_t totalSize);
template float sumgpu1D<float>(float *d_idata,  size_t totalSize);
template double sumgpu1D<double>(double *d_idata,  size_t totalSize);

// max3Dgpu: find max value and coordinates
template <class T>
T max3Dgpu(size_t *corXYZ, T *d_idata, size_t sx, size_t sy, size_t sz){
	size_t sx0 = 0, sy0 = 0, sz0 = 0;
	T *d_temp1 = NULL, *h_temp1 = NULL;
	size_t *d_temp2 = NULL, *h_temp2 = NULL;
	cudaMalloc((void **)&d_temp1, sx*sy *sizeof(T));
	cudaMalloc((void **)&d_temp2, sx*sy *sizeof(size_t));
	h_temp1 = (T *)malloc(sx*sy * sizeof(T));
	h_temp2 = (size_t *)malloc(sx*sy * sizeof(size_t));
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	maxZkernel<T> <<<grids, threads >>>(d_idata, d_temp1, d_temp2, sx, sy, sz);
	cudaThreadSynchronize();
	cudaMemcpy(h_temp1, d_temp1, sx*sy * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_temp2, d_temp2, sx*sy * sizeof(size_t), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	T peakValue = h_temp1[0];
	T t;
	for (size_t i = 0; i < sx; i++){
		for (size_t j = 0; j < sy; j++){
			t = h_temp1[i + j * sx];
			if (peakValue < t){
				peakValue = t;
				sx0 = i; 
				sy0 = j;
				sz0 = h_temp2[i + j * sx];
			}
		}
	}
	corXYZ[0] = sx0; corXYZ[1] = sy0; corXYZ[2] = sz0;
	free(h_temp1); free(h_temp2);
	cudaFree(d_temp1); cudaFree(d_temp2);
	return peakValue;
}
template int max3Dgpu<int>(size_t *corXYZ, int *d_idata, size_t sx, size_t sy, size_t sz);
template float max3Dgpu<float>(size_t *corXYZ, float *d_idata, size_t sx, size_t sy, size_t sz);
template double max3Dgpu<double>(size_t *corXYZ, double *d_idata, size_t sx, size_t sy, size_t sz);

// max with a single value
template <class T>
void maxvalue3Dgpu(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	maxvalue3Dgpukernel<T><<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void maxvalue3Dgpu<int>(int *d_odata, int *d_idata1, int d_idata2, size_t sx, size_t sy, size_t sz);
template void maxvalue3Dgpu<float>(float *d_odata, float *d_idata1, float d_idata2, size_t sx, size_t sy, size_t sz);
template void maxvalue3Dgpu<double>(double *d_odata, double *d_idata1, double d_idata2, size_t sx, size_t sy, size_t sz);


// maximum projection
template <class T>
void maxprojection(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int pDirection){
	size_t psx, psy, psz;
	if (pDirection == 1){
		psx = sx; psy = sy; psz = sz;
	}
	if (pDirection == 2){
		psx = sz; psy = sx; psz = sy;
	}
	if (pDirection == 3){
		psx = sy; psy = sz; psz = sx;
	}
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(psx, threads.x), iDivUp(psy, threads.y));
	maxprojectionkernel<T> <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz, psx, psy, psz, pDirection);
	cudaThreadSynchronize();
}

template void maxprojection<int>(int *d_odata, int *d_idata, size_t sx, size_t sy, size_t sz, int pDirection);
template void maxprojection<float>(float *d_odata, float *d_idata, size_t sx, size_t sy, size_t sz, int pDirection);
template void maxprojection<double>(double *d_odata, double *d_idata, size_t sx, size_t sy, size_t sz, int pDirection);
//Other functions
template <class T>
void changestorageordergpu(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int orderMode){
	//orderMode
	// 1: change tiff storage order to C storage order
	//-1: change C storage order to tiff storage order
	assert(d_odata != d_idata);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	changestorageordergpukernel<T><<<grids, threads>>>(d_odata, d_idata, sx, sy, sz, orderMode);
	cudaThreadSynchronize();
}
template void changestorageordergpu<int>(int *d_odata, int *d_idata, size_t sx, size_t sy, size_t sz, int orderMode);
template void changestorageordergpu<float>(float *d_odata, float *d_idata, size_t sx, size_t sy, size_t sz, int orderMode);
template void changestorageordergpu<double>(double *d_odata, double *d_idata, size_t sx, size_t sy, size_t sz, int orderMode);

// rotate 90/-90 degree by axis
template <class T>
void rotbyyaxis(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int rotDirection){
	//rot direction
	// 1: rotate 90 deg around Y axis
	//-1: rotate -90 deg around Y axis
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	rotbyyaxiskernel<T> <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz, rotDirection);
	cudaThreadSynchronize();
}
template void rotbyyaxis<int>(int *d_odata, int *d_idata, size_t sx, size_t sy, size_t sz, int rotDirection);
template void rotbyyaxis<float>(float *d_odata, float *d_idata, size_t sx, size_t sy, size_t sz, int rotDirection);
template void rotbyyaxis<double>(double *d_odata, double *d_idata, size_t sx, size_t sy, size_t sz, int rotDirection);

/*
// rotate any degree by y axis: matrix for affine transformation
void rot3Dbyyaxis(float *d_odata, float theta, int sx, int sz, int sx2, int sz2){
// Rotation matrix:translation (-sx2/2, -sz2/2) --> rotation--> translation back(sx/2,sy/2)
//	1	0	0	sx / 2			cos(theta)	0	sin(theta)	0		1	0	0	-sx2/2
//	0	1	0		0		*		0		1		0		0	*	0	1	0	0	
//	0	0	1	sz / 2			-sin(theta)	0	cos(theta)	0		0	0	1	-sz2/2
//	0	0	0		1				0		0		0		1		0	0	0	1
	d_odata[0] = cos(theta); d_odata[1] = 0; d_odata[2] = sin(theta);
	d_odata[3] = sx / 2 - sx2 / 2 * cos(theta) - sz2 / 2 * sin(theta);
	d_odata[4] = 0; d_odata[5] = 1; d_odata[6] = 0; d_odata[7] = 0;
	d_odata[8] = -sin(theta); d_odata[9] = 0; d_odata[10] = cos(theta);
	d_odata[11] = sz / 2 + sx2 / 2 * sin(theta) - sz2 / 2 * cos(theta);
}
*/

void p2matrix(float *m, float *x){

	m[0] = x[4], m[1] = x[5], m[2] = x[6], m[3] = x[1];
	m[4] = x[7], m[5] = x[8], m[6] = x[9], m[7] = x[2];
	m[8] = x[10], m[9] = x[11], m[10] = x[12], m[11] = x[3];

	/*
	m[0] = x[1], m[1] = x[2], m[2] = x[3], m[3] = x[4];
	m[4] = x[5], m[5] = x[6], m[6] = x[7], m[7] = x[8];
	m[8] = x[9], m[9] = x[10], m[10] = x[11], m[11] = x[12];
	*/
}
void matrix2p(float *m, float *x){
	x[0] = 0;

	x[1] = m[3], x[2] = m[7], x[3] = m[11], x[4] = m[0];
	x[5] = m[1], x[6] = m[2], x[7] = m[4], x[8] = m[5];
	x[9] = m[6], x[10] = m[8], x[11] = m[9], x[12] = m[10];

	/*
	x[1] = m[0], x[2] = m[1], x[3] = m[2], x[4] = m[3];
	x[5] = m[4], x[6] = m[5], x[7] = m[6], x[8] = m[7];
	x[9] = m[8], x[10] = m[9], x[11] = m[10], x[12] = m[11];
	*/
}


extern "C" void matrixmultiply(float * m, float *m1, float *m2){//for transformation matrix calcution only
	m[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8];
	m[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9];
	m[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10];
	m[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3];

	m[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8];
	m[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9];
	m[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10];
	m[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7];

	m[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8];
	m[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9];
	m[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10];
	m[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11];
	//**** 12 13 14 15 never change ****
	//no need to calculate m[12,13,14,15]:0 0 0 1

	/*
	m[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8] + m1[3] * m2[12];
	m[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9] + m1[3] * m2[13];
	m[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10] + m1[3] * m2[14];
	m[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3] * m2[15];

	m[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8] + m1[7] * m2[12];
	m[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9] + m1[7] * m2[13];
	m[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10] + m1[7] * m2[14];
	m[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7] * m2[15];

	m[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8] + m1[11] * m2[12];
	m[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9] + m1[11] * m2[13];
	m[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10] + m1[11] * m2[14];
	m[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11] * m2[15];

	m[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8] + m1[15] * m2[12];
	m[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9] + m1[15] * m2[13];
	m[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
	m[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
	*/
}


extern "C" void rot2matrix(float * p_out, float theta, long long int sx, long long int sy, long long int sz, int rotAxis){
	//p_out: 12 elements
	//theta: rotation angle
	//sx, sy, sz: images size
	////rotAxis
	// 1: rotate theta around X axis
	// 2: rotate theta around Y axis
	// 3: rotate theta around Z axis

	long long int sNew;
	float *p_temp, *p_temp1, *p_temp2, *p_temp3;
	p_temp = (float *)malloc(16 * sizeof(float));
	p_temp1 = (float *)malloc(16 * sizeof(float));
	p_temp2 = (float *)malloc(16 * sizeof(float));
	p_temp3 = (float *)malloc(16 * sizeof(float));
	for (int i = 0; i < 15; i++){
		p_temp[i] = p_temp1[i] = p_temp2[i] = p_temp3[i] = 0;
	}
	p_temp[15] = p_temp1[15] = p_temp2[15] = p_temp3[15] = 1; //**** 12 13 14 15 never change ****

	// matrix: p_temp1 * p_temp2 * p_temp3

	if (rotAxis == 1){//Rotate by x axis
		// Rotation matrix:translation (0, -sx2/2, -sz2/2) --> rotation--> translation back(0,sy/2,sz/2)
		//	1	0	0		0			1		0			0		0		1	0	0	0
		//	0	1	0	sx / 2		*	0	cos(theta)	sin(theta)	0	*	0	1	0	-sy2/2	
		//	0	0	1	sz / 2			0	-sin(theta)	cos(theta)	0		0	0	1	-sz2/2
		//	0	0	0		1			0		0			0		1		0	0	0	1
		p_temp1[0] = p_temp1[5] = p_temp1[10] = 1;
		p_temp1[7] = sy / 2; p_temp1[11] = sz / 2;

		p_temp2[0] = 1; p_temp2[1] = 0; p_temp2[2] = 0; p_temp2[3] = 0;
		p_temp2[4] = 0; p_temp2[5] = cos(theta); p_temp2[6] = sin(theta); p_temp2[7] = 0;
		p_temp2[8] = 0; p_temp2[9] = -sin(theta); p_temp2[10] = cos(theta); p_temp2[11] = 0;

		sNew = round(sqrt(sy * sy + sz*sz));
		p_temp3[0] = p_temp3[5] = p_temp3[10] = 1;
		p_temp3[7] = - sNew / 2; p_temp3[11] = - sNew / 2; 
	}

	if (rotAxis == 2){//Rotate by y axis

		// Rotation matrix:translation (-sx2/2, 0, -sz2/2) --> rotation--> translation back(sx/2,0,sz/2)
		//	1	0	0	sx / 2			cos(theta)	0	-sin(theta)	0		1	0	0	-sx2/2
		//	0	1	0		0		*		0		1		0		0	*	0	1	0	0	
		//	0	0	1	sz / 2			sin(theta)	0	cos(theta)	0		0	0	1	-sz2/2
		//	0	0	0		1				0		0		0		1		0	0	0	1

		p_temp1[0] = p_temp1[5] = p_temp1[10] = 1;
		p_temp1[3] = sx / 2; p_temp1[11] = sz / 2;

		p_temp2[0] = cos(theta); p_temp2[1] = 0; p_temp2[2] = -sin(theta); p_temp2[3] = 0;
		p_temp2[4] = 0; p_temp2[5] = 1; p_temp2[6] = 0; p_temp2[7] = 0;
		p_temp2[8] = sin(theta); p_temp2[9] = 0; p_temp2[10] = cos(theta); p_temp2[11] = 0;

		sNew = round(sqrt(sx * sx + sz*sz));
		p_temp3[0] = p_temp3[5] = p_temp3[10] = 1;
		p_temp3[3] = -sNew / 2; p_temp3[11] = -sNew / 2;
	}

	if (rotAxis == 3){//Rotate by z axis
		// Rotation matrix:translation (-sx2/2,-sy2/2, 0) --> rotation--> translation back(sx/2,sy/2,0)
		//	1	0	0	sx / 2			cos(theta)	sin(theta)	0	0		1	0	0	-sx2/2
		//	0	1	0	sy / 2		*	-sin(theta)	cos(theta)	0	0	*	0	1	0	-sy2/2	
		//	0	0	1		0				0			0		1	0		0	0	1	0
		//	0	0	0		1				0			0		0	1		0	0	0	1

		p_temp1[0] = p_temp1[5] = p_temp1[10] = 1;
		p_temp1[3] = sx / 2; p_temp1[7] = sy / 2;

		p_temp2[0] = cos(theta); p_temp2[1] = sin(theta); p_temp2[2] = 0; p_temp2[3] = 0;
		p_temp2[4] = -sin(theta); p_temp2[5] = cos(theta); p_temp2[6] = 0; p_temp2[7] = 0;
		p_temp2[8] = 0; p_temp2[9] = 0; p_temp2[10] = 1; p_temp2[11] = 0;

		sNew = round(sqrt(sx * sx + sy*sy));
		p_temp3[0] = p_temp3[5] = p_temp3[10] = 1;
		p_temp3[3] = -sNew / 2; p_temp3[7] = -sNew / 2;
	}


	matrixmultiply(p_temp, p_temp1, p_temp2);
	matrixmultiply(p_out, p_temp, p_temp3);

	free(p_temp);
	free(p_temp1);
	free(p_temp2);
	free(p_temp3);
}

extern "C" void dof9tomatrix(float * p_out, float *p_dof, int dofNum){
	//p_out: 12 elements
	//p_dof: 10 elements: 0 x y z alpha beta theda a b c 
	//dofNum: 3, 6, 7 or 9
	float *p_temp1, *p_temp2, *p_temp3;
	p_temp1 = (float *)malloc(16 * sizeof(float));
	p_temp2 = (float *)malloc(16 * sizeof(float));
	p_temp3 = (float *)malloc(16 * sizeof(float));
	for (int i = 0; i < 15; i++){
		p_temp1[i] = p_temp2[i] = p_temp3[i] = 0;
	}
	p_temp1[15] = p_temp2[15] = p_temp3[15] = 1; //**** 12 13 14 15 never change ****

	float x, y, z, alpha, beta, theta, a, b, c;
	if (dofNum == 3){//translation
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = 0;
		beta = 0;
		theta = 0;
		a = 1;
		b = 1;
		c = 1;
	}
	else if (dofNum == 6){//rigid body: translation, rotation
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = p_dof[4] / 57.3;
		beta = p_dof[5] / 57.3;
		theta = p_dof[6] / 57.3;
		a = 1;
		b = 1;
		c = 1;
	}
	else if (dofNum == 7){//translation,rotation, scale equelly in 3 dimemsions 
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = p_dof[4] / 57.3;
		beta = p_dof[5] / 57.3;
		theta = p_dof[6] / 57.3;
		a = p_dof[7];
		b = p_dof[7];
		c = p_dof[7];
	}
	else if (dofNum == 9){//translation,rotation,scale
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = p_dof[4] / 57.3;
		beta = p_dof[5] / 57.3;
		theta = p_dof[6] / 57.3;
		a = p_dof[7];
		b = p_dof[8];
		c = p_dof[9];
	}

	//translation
	// 1	0	0	x
	// 0	1	0	y
	// 0	0	1	z
	// 0	0	0	1
	p_temp2[3] = x;
	p_temp2[7] = y;
	p_temp2[11] = z;
	// scaling
	// a	0	0	0
	// 0	b	0	0
	// 0	0	c	0
	// 0	0	0	1
	p_temp2[0] = a;
	p_temp2[5] = b;
	p_temp2[10] = c;
	// rotating by Z axis
	// cos(alpha)	sin(alpha)	0	0
	// -sin(alpha)	cos(alpha)	0	0
	// 0			0			1	0
	// 0			0			0	1
	p_temp3[0] = cos(alpha); p_temp3[1] = sin(alpha); p_temp3[2] = 0; p_temp3[3] = 0;
	p_temp3[4] = -sin(alpha); p_temp3[5] = cos(alpha); p_temp3[6] = 0; p_temp3[7] = 0;
	p_temp3[8] = 0; p_temp3[9] = 0; p_temp3[10] = 1; p_temp3[11] = 0;
	//p_temp3[15] = 1;
	matrixmultiply(p_temp1, p_temp2, p_temp3);
	// rotating by X axis
	// 1	0			0			0
	// 0	cos(beta)	sin(beta)	0
	// 0	-sin(beta)	cos(beta)	0
	// 0	0			0			1
	p_temp3[0] = 1; p_temp3[1] = 0; p_temp3[2] = 0; p_temp3[3] = 0;
	p_temp3[4] = 0; p_temp3[5] = cos(beta); p_temp3[6] = sin(beta); p_temp3[7] = 0;
	p_temp3[8] = 0; p_temp3[9] = -sin(beta); p_temp3[10] = cos(beta); p_temp3[11] = 0;
	//p_temp3[15] = 1;
	matrixmultiply(p_temp2, p_temp1, p_temp3);
	// rotating by Y axis
	// cos(theta)	0	-sin(theta)		0
	// 0			1	0				0
	// sin(theta)	0	cos(theta)		0
	// 0			0	0				1
	p_temp3[0] = cos(theta); p_temp3[1] = 0; p_temp3[2] = -sin(theta); p_temp3[3] = 0;
	p_temp3[4] = 0; p_temp3[5] = 1; p_temp3[6] = 0; p_temp3[7] = 0;
	p_temp3[8] = sin(theta); p_temp3[9] = 0; p_temp3[10] = cos(theta); p_temp3[11] = 0;
	//p_temp3[15] = 1;
	matrixmultiply(p_out, p_temp2, p_temp3);

	free(p_temp1);
	free(p_temp2);
	free(p_temp3);
}

template <class T>
void circshiftgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz){
	assert(d_odata != d_idata);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	circshiftgpukernel<T> <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz, dx, dy, dz);
	cudaThreadSynchronize();
}
template void circshiftgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);
template void circshiftgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);
template void circshiftgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);
template <class T>
void imshiftgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz) {
	assert(d_odata != d_idata);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	imshiftgpukernel<T> << <grids, threads >> >(d_odata, d_idata, sx, sy, sz, dx, dy, dz);
	cudaThreadSynchronize();
}
template void imshiftgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);
template void imshiftgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);
template void imshiftgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);

extern "C" void CopyTranMatrix(float *x, int dataSize){
	cudaMemcpyToSymbol(d_aff, x, dataSize, 0, cudaMemcpyHostToDevice);
}


template <class T>
void cudacopyhosttoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, T *h_idata, size_t sx, size_t sy, size_t sz){
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)h_idata, sx*sizeof(T), sx, sy);
	copyParams.dstArray = d_Array;
	copyParams.extent = make_cudaExtent(sx, sy, sz);
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	cudaThreadSynchronize();
}
template void
cudacopyhosttoarray<unsigned short>(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, unsigned short *h_idata, size_t sx, size_t sy, size_t sz);
template void
cudacopyhosttoarray<float>(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *h_idata, size_t sx, size_t sy, size_t sz);

template <class T>
void cudacopydevicetoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, T *d_idata, size_t sx, size_t sy, size_t sz){
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)d_idata, sx*sizeof(T), sx, sy);
	copyParams.dstArray = d_Array;
	copyParams.extent = make_cudaExtent(sx, sy, sz);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	cudaThreadSynchronize();
}
template void
cudacopydevicetoarray<unsigned short>(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, unsigned short *d_idata, size_t sx, size_t sy, size_t sz);
template void
cudacopydevicetoarray<float>(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *d_idata, size_t sx, size_t sy, size_t sz);


extern "C" void BindTexture(cudaArray *d_Array, cudaChannelFormatDesc channelDesc){
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.addressMode[2] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; //NB coordinates in [0,1]
	// Bind the array to the texture
	cudaBindTextureToArray(tex, d_Array, channelDesc);
	cudaThreadSynchronize();
}

extern "C" void BindTexture2(cudaArray *d_Array, cudaChannelFormatDesc channelDesc) {
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.addressMode[2] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; //NB coordinates in [0,1]
							// Bind the array to the texture
	cudaBindTextureToArray(tex2, d_Array, channelDesc);
	cudaThreadSynchronize();
}

extern "C" void BindTexture16(cudaArray *d_Array, cudaChannelFormatDesc channelDesc){
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.addressMode[2] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; //NB coordinates in [0,1]
	// Bind the array to the texture
	cudaBindTextureToArray(tex16, d_Array, channelDesc);
	cudaThreadSynchronize();
}

extern "C" void UnbindTexture(){
	cudaUnbindTexture(tex);
	cudaThreadSynchronize();
}

extern "C" void UnbindTexture2() {
	cudaUnbindTexture(tex2);
	cudaThreadSynchronize();
}

extern "C" void UnbindTexture16(){
	cudaUnbindTexture(tex16);
	cudaThreadSynchronize();
}

extern "C" void AccessTexture(float x, float y,float z){
	dim3 threads(2, 2, 2);
	accesstexturekernel <<<1, threads >>>(x, y, z);
	cudaThreadSynchronize();
}

template <class T> 
void affineTransform(T *d_s, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grid(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	affinetransformkernel<T><<<grid, threads >>>(d_s, sx, sy, sz, sx2, sy2, sz2);
	cudaThreadSynchronize();
}
template void
affineTransform<unsigned short>(unsigned short *d_s, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2);
template void 
affineTransform<float>(float *d_s, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2);

float corrfunc(float *d_t, float sd_t, float *aff, long long int sx, 
	long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2){
	// temp bufs
	long long int sxy = sx * sy;
	double *d_temp1 = NULL, *d_temp2 = NULL;
	cudaMalloc((void **)&d_temp1, sxy * sizeof(double));
	cudaMalloc((void **)&d_temp2, sxy * sizeof(double));
	//copy aff to GPU const
	cudaMemcpyToSymbol(d_aff, aff, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	corrkernel<<<grids, threads>>>( d_t, // the source image is texture, trans matrix is const
		d_temp1, d_temp2, sx, sy, sz, sx2, sy2, sz2);
	cudaThreadSynchronize();
	double sqrSum = 0, corrSum = 0;
	if (sxy > 100000){ // if count more than 100000, use gpu to perform sum
		sqrSum = sumgpu1D(d_temp1,  sxy);
		corrSum = sumgpu1D(d_temp2, sxy);
	}
	else{
		double *h_temp = NULL;
		h_temp = (double *)malloc(sx*sy * sizeof(double));
		cudaMemcpy(h_temp, d_temp1, sxy * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < sxy; i++)
			sqrSum += h_temp[i];
		cudaMemcpy(h_temp, d_temp2, sxy * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < sxy; i++)
			corrSum += h_temp[i];
		free(h_temp);
	}
	cudaFree(d_temp1); 
	cudaFree(d_temp2); 
	if (sqrt(sqrSum) == 0) return -2.0;
	return (float)(corrSum / sqrt(sqrSum)) / sd_t;
}

extern "C" void BindTexture2D(cudaArray *d_Array, cudaChannelFormatDesc channelDesc){
	// set texture parameters
	tex2D1.addressMode[0] = cudaAddressModeWrap;
	tex2D1.addressMode[1] = cudaAddressModeWrap;
	tex2D1.filterMode = cudaFilterModeLinear;
	tex2D1.normalized = false;    // access with normalized texture coordinates

	// Bind the array to the texture
	cudaBindTextureToArray(tex2D1, d_Array, channelDesc);
}

extern "C" void UnbindTexture2D(
	){
	cudaUnbindTexture(tex2D1);
}

extern "C"
void affineTransform2D(float *d_t, int sx, int sy, int sx2, int sy2){
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	affineTransform2Dkernel <<<grids, threads >>>(d_t, sx, sy, sx2, sy2);
	cudaThreadSynchronize();
}

float corrfunc2D(float *d_t, float sd_t, float *aff, long long int sx, long long int sy, long long int sx2, long long int sy2){
	//copy aff to GPU const
	cudaMemcpyToSymbol(d_aff, aff, 6 * sizeof(float), 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	long long int totalSize = sx*sy;
	float *d_sqr = NULL, *d_corr = NULL, *h_temp = NULL;
	cudaMalloc((void **)&d_sqr, totalSize * sizeof(float));
	cudaMalloc((void **)&d_corr, totalSize * sizeof(float));
	h_temp = (float *)malloc(totalSize * sizeof(float));
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	corr2Dkernel <<<grids, threads >>>( // the other image is texture, trans matrix is const
		d_t, d_sqr, d_corr, sx, sy, sx2, sy2);
	cudaThreadSynchronize();
	cudaMemcpy(h_temp, d_corr, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	double corrSum = sumcpu(h_temp, totalSize);
	cudaMemcpy(h_temp, d_sqr, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	double sqrSum = sumcpu(h_temp, totalSize);
	cudaFree(d_sqr);
	cudaFree(d_corr);
	free(h_temp);
	if (sqrt(sqrSum) == 0) return -2.0;
	return float(corrSum / sqrt(sqrSum))/sd_t;
}

///// CPU interpolation
float lerp(float x, float x1, float x2, float q00, float q01) {
	return ((x2 - x) / (x2 - x1)) * q01 + ((x - x1) / (x2 - x1)) * q00;
}

float bilerp(float x, float y, float x1, float x2, float y1, float y2, float q11, float q12, float q21, float q22) {
	float r1 = lerp(x, x1, x2, q11, q12);
	float r2 = lerp(x, x1, x2, q21, q22);

	return lerp(y, y1, y2, r1, r2);
}

float trilerp(float x, float y, float z, float x1, float x2, float y1, float y2, float z1, float z2, 
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222) {
	float r1 = bilerp(x, y, x1, x2, y1, y2, q111, q112, q121, q122);
	float r2 = bilerp(x, y, x1, x2, y1, y2, q211, q212, q221, q222);
	return lerp(z, z1, z2, r1, r2);
}

float ilerp(float x, float x1, float x2, float q00, float q01) {
	return (x2 - x) * q00 + (x - x1) * q01;
}

float ibilerp(float x, float y, float x1, float x2, float y1, float y2, float q11, float q12, float q21, float q22) {
	float r1 = ilerp(x, x1, x2, q11, q12);
	float r2 = ilerp(x, x1, x2, q21, q22);

	return ilerp(y, y1, y2, r1, r2);
}

float itrilerp(float x, float y, float z, float x1, float x2, float y1, float y2, float z1, float z2,
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222) {
	float r1 = ibilerp(x, y, x1, x2, y1, y2, q111, q112, q121, q122);
	float r2 = ibilerp(x, y, x1, x2, y1, y2, q211, q212, q221, q222);
	return ilerp(z, z1, z2, r1, r2);
}

float ilerp2(float dx1, float dx2, float q00, float q01) {
	return dx2 * q00 + dx1 * q01;
}

float ibilerp2(float dx1, float dx2, float dy1, float dy2, float q11, float q12, float q21, float q22) {
	float r1 = ilerp2(dx1, dx2, q11, q12);
	float r2 = ilerp2(dx1, dx2, q21, q22);

	return ilerp2(dy1, dy2, r1, r2);
}

float itrilerp2(float dx1, float dx2, float dy1, float dy2, float dz1, float dz2,
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222) {
	float r1 = ibilerp2(dx1, dx2, dy1, dy2, q111, q112, q121, q122);
	float r2 = ibilerp2(dx1, dx2, dy1, dy2, q211, q212, q221, q222);
	return ilerp2(dz1, dz2, r1, r2);
}

//output[sz-k-1][j][i] = input[i][j][k]
//d_odata[(sz - k - 1)*sx*sy + j*sx + i] = d_idata[i*sy*sz + j*sz + k];
double corrfunccpu(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	double sqrSum = 0, corrSum = 0;
	int x1, y1, z1, x2, y2, z2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int sxy = sx*sy, sxy2 = sx2*sy2;
	for (int i = 0; i < sx; i++){
		for (int j = 0; j < sy; j++){
			for (int k = 0; k < sz; k++){
				float ix = (float)i;
				float iy = (float)j;
				float iz = (float)k;
				float tx = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
				float ty = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
				float tz = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
				x1 = floor(tx); y1 = floor(ty); z1 = floor(tz);
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;
				if ((x1 >= 0) && (y1 >= 0) && (z1 >= 0) && (x2 < sx2) && (y2 < sy2) && (z2 < sz2)){
					// [k*sy*sx + j*sx + i]
					q1 = h_t[z1*sxy2 + y1*sx2 + x1];
					q2 = h_t[z1*sxy2 + y1*sx2 + x2];
					q3 = h_t[z1*sxy2 + y2*sx2 + x1];
					q4 = h_t[z1*sxy2 + y2*sx2 + x2];
					q5 = h_t[z2*sxy2 + y1*sx2 + x1];
					q6 = h_t[z2*sxy2 + y1*sx2 + x2];
					q7 = h_t[z2*sxy2 + y2*sx2 + x1];
					q8 = h_t[z2*sxy2 + y2*sx2 + x2];
					t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
				}
				else
					t = 0;
				s = h_s[k*sxy + j*sx + i];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}


double corrfunccpu3(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	const float r0 = aff[0], r1 = aff[1], r2 = aff[2], r3 = aff[3], r4 = aff[4], r5= aff[5],
		r6 = aff[6], r7 = aff[7], r8 = aff[8], r9 = aff[9], r10 = aff[10], r11 = aff[11];

	double sqrSum = 0, corrSum = 0;
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int syz = sy*sz, syz2 = sy2*sz2, x1syz2, x2syz2, y1sz2, y2sz2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;
				
				tx = r0 * ix + r1 * iy + r2 * iz + r3;
				ty = r4 * ix + r5 * iy + r6 * iz + r7;
				tz = r8 * ix + r9 * iy + r10 * iz + r11;
				
				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					x1syz2 = x1*syz2;
					x2syz2 = x2*syz2;
					y1sz2 = y1*sz2;
					y2sz2 = y2*sz2;

					q1 = h_t[x1syz2 + y1sz2 + z1];
					q2 = h_t[x2syz2 + y1sz2 + z1];
					q3 = h_t[x1syz2 + y2sz2 + z1];
					q4 = h_t[x2syz2 + y2sz2 + z1];
					q5 = h_t[x1syz2 + y1sz2 + z2];
					q6 = h_t[x2syz2 + y1sz2 + z2];
					q7 = h_t[x1syz2 + y2sz2 + z2];
					q8 = h_t[x2syz2 + y2sz2 + z2];
					//t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
					t = dz2*(dy2*dx2*q1 + dy2*dx1*q2 + dy1*dx2*q3 + dy1*dx1*q4) + dz1*(dy2*dx2*q5 + dy2*dx1*q6 + dy1*dx2*q7 + dy1*dx1*q8);
					//t = 1;

				}
				else
					t = 0;
				s = h_s[i*syz + j*sz + k];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}
double corrfunccpu2_old(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	const float r0 = aff[0], r1 = aff[1], r2 = aff[2], r3 = aff[3], r4 = aff[4], r5 = aff[5],
		r6 = aff[6], r7 = aff[7], r8 = aff[8], r9 = aff[9], r10 = aff[10], r11 = aff[11];

	double sqrSum = 0, corrSum = 0;
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int sxy = sx*sy, sxy2 = sx2*sy2, z1sxy2, z2sxy2, y1sx2, y2sx2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;

				tx = r0 * ix + r1 * iy + r2 * iz + r3;
				ty = r4 * ix + r5 * iy + r6 * iz + r7;
				tz = r8 * ix + r9 * iy + r10 * iz + r11;

				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					z1sxy2 = z1*sxy2;
					z2sxy2 = z2*sxy2;
					y1sx2 = y1*sx2;
					y2sx2 = y2*sx2;

					q1 = h_t[z1sxy2 + y1sx2 + x1];
					q2 = h_t[z1sxy2 + y1sx2 + x2];
					q3 = h_t[z1sxy2 + y2sx2 + x1];
					q4 = h_t[z1sxy2 + y2sx2 + x2];
					q5 = h_t[z2sxy2 + y1sx2 + x1];
					q6 = h_t[z2sxy2 + y1sx2 + x2];
					q7 = h_t[z2sxy2 + y2sx2 + x1];
					q8 = h_t[z2sxy2 + y2sx2 + x2];
					//t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
					t = dz2*(dy2*dx2*q1 + dy2*dx1*q2 + dy1*dx2*q3 + dy1*dx1*q4) + dz1*(dy2*dx2*q5 + dy2*dx1*q6 + dy1*dx2*q7 + dy1*dx1*q8);
					//t = 1;

				}
				else
					t = 0;
				s = h_s[k*sxy + j*sx + i];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}

void affinetransformcpu_old(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float t;
	int sxy = sx*sy, sxy2 = sx2*sy2, z1sxy2, z2sxy2, y1sx2, y2sx2;
	int syz = sy*sz, syz2 = sy2*sz2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;
				tx = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
				ty = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
				tz = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					z1sxy2 = z1*sxy2;
					z2sxy2 = z2*sxy2;
					y1sx2 = y1*sx2;
					y2sx2 = y2*sx2;

					q1 = h_t[z1sxy2 + y1sx2 + x1];
					q2 = h_t[z1sxy2 + y1sx2 + x2];
					q3 = h_t[z1sxy2 + y2sx2 + x1];
					q4 = h_t[z1sxy2 + y2sx2 + x2];
					q5 = h_t[z2sxy2 + y1sx2 + x1];
					q6 = h_t[z2sxy2 + y1sx2 + x2];
					q7 = h_t[z2sxy2 + y2sx2 + x1];
					q8 = h_t[z2sxy2 + y2sx2 + x2];
					t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = dz2*(dy2*dx2*q1 + dy2*dx1*q2 + dy1*dx2*q3 + dy1*dx1*q4) + dz1*(dy2*dx2*q5 + dy2*dx1*q6 + dy1*dx2*q7 + dy1*dx1*q8);
				}
				else
					t = 0;
				h_s[k*sxy + j*sx + j] = t;
			}
		}
	}
}


double corrfunccpu2(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	const float r0 = aff[0], r1 = aff[1], r2 = aff[2], r3 = aff[3], r4 = aff[4], r5 = aff[5],
		r6 = aff[6], r7 = aff[7], r8 = aff[8], r9 = aff[9], r10 = aff[10], r11 = aff[11];

	double sqrSum = 0, corrSum = 0;
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int syz = sy*sz, syz2 = sy2*sz2, x1syz2, x2syz2, y1sz2, y2sz2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;

				tx = r0 * ix + r1 * iy + r2 * iz + r3;
				ty = r4 * ix + r5 * iy + r6 * iz + r7;
				tz = r8 * ix + r9 * iy + r10 * iz + r11;

				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					x1syz2 = x1*syz2;
					x2syz2 = x2*syz2;
					y1sz2 = y1*sz2;
					y2sz2 = y2*sz2;

					q1 = h_t[x1syz2 + y1sz2 + z1];
					q2 = h_t[x2syz2 + y1sz2 + z1];
					q3 = h_t[x1syz2 + y2sz2 + z1];
					q4 = h_t[x2syz2 + y2sz2 + z1];
					q5 = h_t[x1syz2 + y1sz2 + z2];
					q6 = h_t[x2syz2 + y1sz2 + z2];
					q7 = h_t[x1syz2 + y2sz2 + z2];
					q8 = h_t[x2syz2 + y2sz2 + z2];
					t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);

				}
				else
					t = 0;
				s = h_s[i*syz + j*sz + k];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}


void affinetransformcpu(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float t;
	int syz = sy*sz, syz2 = sy2*sz2, x1syz2, x2syz2, y1sz2, y2sz2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;
				tx = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
				ty = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
				tz = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					x1syz2 = x1*syz2;
					x2syz2 = x2*syz2;
					y1sz2 = y1*sz2;
					y2sz2 = y2*sz2;

					q1 = h_t[x1syz2 + y1sz2 + z1];
					q2 = h_t[x2syz2 + y1sz2 + z1];
					q3 = h_t[x1syz2 + y2sz2 + z1];
					q4 = h_t[x2syz2 + y2sz2 + z1];
					q5 = h_t[x1syz2 + y1sz2 + z2];
					q6 = h_t[x2syz2 + y1sz2 + z2];
					q7 = h_t[x1syz2 + y2sz2 + z2];
					q8 = h_t[x2syz2 + y2sz2 + z2];
					t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);

				}
				else
					t = 0;
				h_s[i*syz + j*sz + k] = t;
			}
		}
	}
}

// CPU
template <class T>
void flipcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz) {
	for (long long int i = 0; i < sx; i++) {
		for (long long int j = 0; j < sy; j++) {
			for (long long int k = 0; k < sz; k++) {
				//d_odata[k*sy*sx + j*sx + i] = d_idata[(sz - k - 1) *sy*sx + (sy - j - 1)*sx + (sx - i - 1)];
				h_odata[i*sy*sz + j*sz + k] = h_idata[(sx - i - 1) *sy*sz + (sy - j - 1)*sz + (sz - k - 1)];
			}
		}
	}
}
template void flipcpu<int>(int *h_odata, int *h_idata, long long int sx, long long int sy, long long int sz);
template void flipcpu<float>(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz);
template void flipcpu<double>(double *h_odata, double *h_idata, long long int sx, long long int sy, long long int sz);

template <class T>
void padPSFcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2, 
	long long int sy2, long long int sz2){
	long long int sox, soy, soz;
	sox = sx2 / 2; soy = sy2 / 2; soz = sz2 / 2;
	long long int dx, dy, dz;
	for (long long int x = 0; x < sx; x++) {
		for (long long int y = 0; y < sy; y++) {
			for (long long int z = 0; z < sz; z++) {
				dx = x - sox; dy = y - soy; dz = z - soz;
				if (dx < 0) dx += sx;
				if (dy < 0) dy += sy;
				if (dz < 0) dz += sz;
				//d_PaddedPSF[dz][dy][dx] = d_PSF[z][y][x]
				if (dx >= 0 && dx < sx && dy >= 0 && dy < sy && dz >= 0 && dz < sz) {
					//d_odata[dz*sy*sx + dy*sx + dx] = d_idata[z*sy2*sx2 + y*sx2 + x];
					h_odata[dx*sy*sz + dy*sz + dz] = h_idata[x*sy2*sz2 + y*sz2 + z];
				}
			}
		}
	}
}
template void
padPSFcpu<int>(int *h_odata, int *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padPSFcpu<float>(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padPSFcpu<double>(double *h_odata, double *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void padstackcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2){
	long long int sox, soy, soz;
	sox = (sx - sx2) / 2;
	soy = (sy - sy2) / 2;
	soz = (sz - sz2) / 2;
	long long int x, y, z;
	for (long long int dx = 0; dx < sx; dx++) {
		for (long long int dy = 0; dy < sy; dy++) {
			for (long long int dz = 0; dz < sz; dz++) {
				if (dx < sox) {
					x = 0;
				}
				if (dy < soy) {
					y = 0;
				}
				if (dz < soz) {
					z = 0;
				}
				if (dx >= sox && dx < (sox + sx2)) {
					x = dx - sox;
				}
				if (dy >= soy && dy < (soy + sy2)) {
					y = dy - soy;
				}
				if (dz >= soz && dz < (soz + sz2)) {
					z = dz - soz;
				}
				if (dx >= (sox + sx2)) {
					x = sx2 - 1;
				}
				if (dy >= (soy + sy2)) {
					y = sy2 - 1;
				}
				if (dz >= (soz + sz2)) {
					z = sz2 - 1;
				}
				//d_odata[dz*sy*sx + dy*sx + dx] = d_idata[z*sy2*sx2 + y*sx2 + x];
				h_odata[dx*sy*sz + dy*sz + dz] = h_idata[x*sy2*sz2 + y*sz2 + z];
			}
		}
	}
}
template void
padstackcpu<int>(int *h_odata, int *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padstackcpu<float>(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padstackcpu<double>(double *h_odata, double *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void cropcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2){
	long long int sox, soy, soz;
	sox = (sx2 - sx) / 2;
	soy = (sy2 - sy) / 2;
	soz = (sz2 - sz) / 2;
	long long int dx, dy, dz;
	for (long long int x = 0; x < sx; x++) {
		for (long long int y = 0; y < sy; y++) {
			for (long long int z = 0; z < sz; z++) {
				dx = sox + x; dy = soy + y; dz = soz + z;
				//d_odata[z*sy*sx + y*sx + x] = d_idata[dz*sy2*sx2 + dy*sx2 + dx];
				h_odata[x*sy*sz + y*sz + z] = h_idata[dx*sy2*sz2 + dy*sz2 + dz];
			}
		}
	}
}
template void
cropcpu<int>(int *h_odata, int *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
cropcpu<float>(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
cropcpu<double>(double *h_odata, double *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void cropcpu2(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz) {
	long long int dx, dy, dz;
	for (long long int x = 0; x < sx; x++) {
		for (long long int y = 0; y < sy; y++) {
			for (long long int z = 0; z < sz; z++) {
				dx = sox + x; dy = soy + y; dz = soz + z;
				h_odata[z*sy*sx + y*sx + x] = h_idata[dz*sy2*sx2 + dy*sx2 + dx];
				//h_odata[x*sy*sz + y*sz + z] = h_idata[dx*sy2*sz2 + dy*sz2 + dz];
			}
		}
	}
}
template void
cropcpu2<int>(int *h_odata, int *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);
template void
cropcpu2<float>(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);
template void
cropcpu2<double>(double *h_odata, double *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);

template <class T>
void alignsize3Dcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	long long int sox, soy, soz;
	sox = (sx - sx2) / 2;
	soy = (sy - sy2) / 2;
	soz = (sz - sz2) / 2;
	long long int x, y, z;
	for (long long int dx = 0; dx < sx; dx++) {
		for (long long int dy = 0; dy < sy; dy++) {
			for (long long int dz = 0; dz < sz; dz++) {
				x = dx - sox;
				y = dy - soy;
				z = dz - soz;
				if ((x < 0) || (y < 0) || (z < 0) || (x >= sx2) || (y >= sy2) || (z >= sz2))
					h_odata[dx*sy*sz + dy*sz + dz] = 0;
				else
					h_odata[dx*sy*sz + dy*sz + dz] = h_idata[x*sy2*sz2 + y*sz2 + z];
			}
		}
	}
}
template void alignsize3Dcpu<int>(int *h_odata, int *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void alignsize3Dcpu<float>(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void alignsize3Dcpu<double>(double *h_odata, double *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

extern "C"
void genOTFcpu(fftwf_complex *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, bool normFlag) {
	long long int totalSizeIn = sx2 * sy2 * sz2;
	long long int totalSizeOut = sx * sy * sz;
	long long int totalSizeMax = (totalSizeIn > totalSizeOut) ? totalSizeIn : totalSizeOut;
	float *h_temp = (float *)malloc(totalSizeMax * sizeof(float));
	if (normFlag) {
		double sumValue = sumcpu(h_idata, sx2 * sy2 * sz2);
		multivaluecpu(h_temp, h_idata, (float)(1 / sumValue), sx2 * sy2 * sz2);
	}
	else
		memcpy(h_temp, h_idata, totalSizeIn * sizeof(float));
	
	if((sx<sx2)||(sy<sy2)||(sz<sz2)){
		alignsize3Dcpu((float *)h_odata, h_temp, sx, sy, sz, sx2, sy2, sz2);
		padPSFcpu(h_temp, (float *)h_odata, sx, sy, sz, sx, sy, sz);
	}
	else {
		padPSFcpu((float *)h_odata, h_temp, sx, sy, sz, sx2, sy2, sz2);
		memcpy(h_temp, h_odata, totalSizeOut * sizeof(float));
	}
	fftwf_plan image2Spectrum = fftwf_plan_dft_r2c_3d(sx, sy, sz, h_temp, h_odata,  FFTW_MEASURE);
	fftwf_execute(image2Spectrum);
	free(h_temp);
	fftwf_destroy_plan(image2Spectrum);
}

// GPU
template <class T>
void flipgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz) {
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	flipgpukernel<T> << <grids, threads >> >(d_odata, d_idata, sx, sy, sz);
	cudaThreadSynchronize();
}
template void flipgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz);
template void flipgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz);
template void flipgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz);

template <class T>
void padPSFgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	assert(d_odata != d_idata);
	long long int sox, soy, soz;
	sox = sx2 / 2; soy = sy2 / 2; soz = sz2 / 2;
	cudaMemset(d_odata, 0, sx*sy*sz * sizeof(T));
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx2, threads.x), iDivUp(sy2, threads.y), iDivUp(sz2, threads.z));
	padPSFgpukernel<T> << <grids, threads >> >(d_odata, d_idata, sx, sy, sz, sx2, sy2, sz2, sox, soy, soz);
	cudaThreadSynchronize();
}
template void
padPSFgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padPSFgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padPSFgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void padstackgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	assert(d_odata != d_idata);
	long long int sox, soy, soz;
	sox = (sx - sx2) / 2;
	soy = (sy - sy2) / 2;
	soz = (sz - sz2) / 2;
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	padstackgpukernel<T> << < grids, threads >> > (d_odata, d_idata, sx, sy, sz, sx2, sy2, sz2, sox, soy, soz);
	cudaThreadSynchronize();
}
template void
padstackgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padstackgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
padstackgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void cropgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	assert(d_odata != d_idata);
	long long int sox, soy, soz;
	sox = (sx2 - sx) / 2;
	soy = (sy2 - sy) / 2;
	soz = (sz2 - sz) / 2;
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	cropgpukernel<T> <<< grids, threads >>> (d_odata, d_idata, sx, sy, sz, sx2, sy2, sz2, sox, soy, soz);
	cudaThreadSynchronize();
}
template void
cropgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
cropgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void
cropgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void cropgpu2(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz) {
	assert(d_odata != d_idata);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sz, threads.x), iDivUp(sy, threads.y), iDivUp(sx, threads.z));
	cropgpukernel<T> <<< grids, threads >>> (d_odata, d_idata, sz, sy, sx, sz2, sy2, sx2, soz, soy, sox);
	cudaThreadSynchronize();
}
template void
cropgpu2<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);
template void
cropgpu2<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);
template void
cropgpu2<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);


template <class T>
void alignsize3Dgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	assert(d_odata != d_idata);
	long long int sox, soy, soz;
	sox = (sx - sx2) / 2;
	soy = (sy - sy2) / 2;
	soz = (sz - sz2) / 2;
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	alignsize3Dgpukernel<T> << < grids, threads >> > (d_odata, d_idata, sx, sy, sz, sx2, sy2, sz2, sox, soy, soz);
	cudaThreadSynchronize();
}
template void alignsize3Dgpu<int>(int *d_odata, int *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void alignsize3Dgpu<float>(float *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template void alignsize3Dgpu<double>(double *d_odata, double *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

// Registration variables: 2D
static float *d_img2D = NULL;
static float *h_aff2D;
static long long int imx2D1, imy2D1, imx2D2, imy2D2;
static float valueStatic2D;
static int itNumStatic2D;

// Registration variables: 3D
static float *d_imgStatic = NULL;
static float valueStatic;
static long long int sxStatic1, syStatic1, szStatic1, sxStatic2, syStatic2, szStatic2;
static float *affCoef;
static int itNumStatic, dofNum;
static bool dof9Flag;
static float *h_s3D = NULL, *h_t3D = NULL;

float costfunc2D(float *x) {
	h_aff2D[0] = x[1], h_aff2D[1] = x[2], h_aff2D[2] = x[3];
	h_aff2D[3] = x[4], h_aff2D[4] = x[5], h_aff2D[5] = x[6];
	float costValue = corrfunc2D(d_img2D, valueStatic2D, h_aff2D, imx2D1, imy2D1, imx2D2, imy2D2);
	itNumStatic2D += 1;
	return -costValue;
}
extern "C"
int affinetrans2d0(float *h_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sx2, long long int sy2) {
	return 0;
}
extern "C"
// bug in affinetrans2d1 
int affinetrans2d1(float *h_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sx2, long long int sy2) {
	// total pixel count for each images
	long long int totalSize1 = sx*sy;
	long long int totalSize2 = sx2*sx2;

	float *d_imgTemp = NULL;
	cudaMalloc((void **)&d_imgTemp, totalSize1 * sizeof(float));
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");

	cudaChannelFormatDesc channelDesc2D =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *d_Array2D;
	cudaMallocArray(&d_Array2D, &channelDesc2D, sx2, sy2);
	cudaCheckErrors("****Memory array allocating fails... GPU out of memory !!!!*****\n");
	CopyTranMatrix(iTmx, 6 * sizeof(float));
	cudaMemcpyToArray(d_Array2D, 0, 0, h_idata, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
	BindTexture2D(d_Array2D, channelDesc2D);
	affineTransform2D(d_imgTemp, sx, sy, sx2, sy2);
	UnbindTexture2D;
	cudaMemcpy(h_odata, d_imgTemp, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_imgTemp);
	cudaFreeArray(d_Array2D);
	return 0;
}

extern "C"
int reg2d_phasor0(long long int *shiftXY, float *h_img1, float *h_img2, long long int sx, long long int sy) {
	return 0;
}
extern "C"
int reg2d_phasor1(long long int *shiftXY, float *d_img1, float *d_img2, long long int sx, long long int sy) {
	int totalSize = sx * sy;
	int totalSizeSpectrum = sy*(sx / 2 + 1); // in complex floating format
	fComplex *d_Spectrum1 = NULL, *d_Spectrum2 = NULL;
	cudaMalloc((void **)&d_Spectrum1, totalSizeSpectrum * sizeof(fComplex));
	cudaMalloc((void **)&d_Spectrum2, totalSizeSpectrum * sizeof(fComplex));
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	cufftPlan2d(&fftPlanFwd, sy, sx, CUFFT_R2C);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_img1, (cufftComplex *)d_Spectrum2);
	conj3Dgpu(d_Spectrum1, d_Spectrum2, sy, (sx / 2 + 1), 1);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_img2, (cufftComplex *)d_Spectrum2);
	// multiplication and normalization
	multicomplexnorm3Dgpu(d_Spectrum2, d_Spectrum1, d_Spectrum2, sy, (sx / 2 + 1), 1);
	cufftDestroy(fftPlanFwd);
	cufftPlan2d(&fftPlanInv, sy, sx, CUFFT_C2R);
	float *d_phasor1 = (float *)d_Spectrum1;
	cufftExecC2R(fftPlanInv, (cufftComplex *)d_Spectrum2, (cufftReal *)d_phasor1);
	cufftDestroy(fftPlanInv);
	size_t corXYZ[3];
	float *d_phasor2 = (float *)d_Spectrum2;
	circshiftgpu(d_phasor2, d_phasor1, sx, sy, 1, round(sx / 2), round(sy / 2), 0);
	float peakValue = max3Dgpu(&corXYZ[0], d_phasor2, sx, sy, 1);
	shiftXY[0] = long long int(corXYZ[0]) - sx / 2;
	shiftXY[1] = long long int(corXYZ[1]) - sy / 2;
	cudaFree(d_Spectrum1);
	cudaFree(d_Spectrum2);

	// compare 4 cases based on cross-correlation
	long long int shiftX = shiftXY[0];
	long long int shiftY = shiftXY[1];
	long long int xabs = abs(shiftX), yabs = abs(shiftY);
	long long int beta = 4; // threshold value: only if shift is more than 1/beta of the image size
	if ((xabs >(sx / beta)) || (yabs >(sy / beta))) {
		float *d_imgT = NULL, *d_crop1 = NULL, *d_crop2 = NULL;
		long long int sizex1, sizex2, sizey1, sizey2, sizez1, sizez2, sizex, sizey, sizez, sizeMaxCrop;
		sizeMaxCrop = totalSize;
		cudaMalloc((void **)&d_imgT, totalSize * sizeof(float));
		cudaMalloc((void **)&d_crop1, sizeMaxCrop * sizeof(float));
		cudaMalloc((void **)&d_crop2, sizeMaxCrop * sizeof(float));
		circshiftgpu(d_imgT, d_img2, sx, sy, 1, -shiftX, -shiftY, 0);
		// encode the 8 cases as for loop
		long long int imSizeCropx[2], imSizeCropy[2], imSizeCropz[2];
		long long int imox[2], imoy[2], imoz[2];
		// index 0 records original shifts, index 1 switches the shift to the opposite case.  
		imSizeCropx[0] = sx - xabs; imSizeCropx[1] = xabs;
		if (shiftX > 0) {
			imox[0] = 0; imox[1] = sx - xabs;
		}
		else {
			imox[0] = xabs; imox[1] = 0;
		}
		imSizeCropy[0] = sy - yabs; imSizeCropy[1] = yabs;
		if (shiftY > 0) {
			imoy[0] = 0; imoy[1] = sy - yabs;
		}
		else {
			imoy[0] = yabs; imoy[1] = 0;
		}
		int indx = 0, indy = 0;
		float ccMax = -3, ccNow = 0;
		for (int i = 0; i < 2; i++) {
			if (imSizeCropx[i] >(sx / beta)) {
				for (int j = 0; j < 2; j++) {
					if (imSizeCropy[j] >(sy / beta)) {
						cropgpu2(d_crop1, d_img1, imSizeCropx[i], imSizeCropy[j], 1, sx, sy, 1, imox[i], imoy[j], 0);
						cropgpu2(d_crop2, d_imgT, imSizeCropx[i], imSizeCropy[j], 1, sx, sy, 1, imox[i], imoy[j], 0);
						ccNow = zncc1(d_crop1, d_crop2, imSizeCropx[i], imSizeCropy[j], 1);
						if (ccMax < ccNow) {
							ccMax = ccNow;
							indx = i;
							indy = j;
						}
					}
				}
			}
		}
		// if ind ==1, flip the coordinates
		if (indx == 1) {
			if (shiftX > 0)
				shiftXY[0] = shiftX - sx;
			else
				shiftXY[0] = shiftX + sx;
		}
		if (indy == 1) {
			if (shiftY > 0)
				shiftXY[1] = shiftY - sy;
			else
				shiftXY[1] = shiftY + sy;
		}
		cudaFree(d_imgT);
		cudaFree(d_crop1);
		cudaFree(d_crop2);
	}
	return 0;
}

extern "C"
int reg2d_affine0(float *h_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx, long long int sy,
	long long int sx2, long long int sy2, int affMethod, bool flagTmx, float FTOL, int itLimit, float *regRecords) {
	// **** CPU affine registration for 2D images ***
	return 0;
}
extern "C"
int reg2d_affine1(float *h_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx, long long int sy, 
	long long int sx2, long long int sy2, int affMethod, bool flagTmx, float FTOL, int itLimit, float *records) {
	// **** GPU affine registration for 2D images ***
	/*
	*** flagTmx:
	true : use iTmx as input matrix;
	false: default;

	*** records: 8 element array
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	*/
	imx2D1 = sx; imy2D1 = sy;
	imx2D2 = sx2; imy2D2 = sy2;

	// total pixel count for each images
	long long int totalSize1 = imx2D1*imy2D1;
	long long int totalSize2 = imx2D2*imy2D2;
	long long int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;

	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t start, end, ctime1, ctime2, ctime3;
	start = clock();
	int iter;
	float fret;
	int DIM2D = 6;
	h_aff2D = (float *)malloc(DIM2D * sizeof(float));
	static float *p2D = (float *)malloc((DIM2D + 1) * sizeof(float));
	float **xi2D;
	xi2D = matrix(1, DIM2D, 1, DIM2D);

	float *h_imgT = (float *)malloc(totalSizeMax * sizeof(float));
	cudaMalloc((void **)&d_img2D, totalSize1 * sizeof(float));
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");

	cudaChannelFormatDesc channelDesc2D =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *d_Array2D;
	cudaMallocArray(&d_Array2D, &channelDesc2D, imx2D2, imy2D2);
	cudaCheckErrors("****Memory array allocating fails... GPU out of memory !!!!*****\n");

	if (flagTmx) {
		memcpy(h_aff2D, iTmx, DIM2D * sizeof(float));
	}
	else {
		h_aff2D[0] = 1, h_aff2D[1] = 0, h_aff2D[2] = (imx2D2 - imx2D1) / 2;
		h_aff2D[3] = 0, h_aff2D[4] = 1, h_aff2D[5] = (imy2D2 - imy2D1) / 2;
	}
	p2D[0] = 0;
	p2D[1] = h_aff2D[0], p2D[2] = h_aff2D[1], p2D[3] = h_aff2D[2];
	p2D[4] = h_aff2D[3], p2D[5] = h_aff2D[4], p2D[6] = h_aff2D[5];
	for (int i = 1; i <= DIM2D; i++)
		for (int j = 1; j <= DIM2D; j++)
			xi2D[i][j] = (i == j ? 1.0 : 0.0);

	float meanValue = (float)sumcpu(h_img1, totalSize1) / totalSize1;
	addvaluecpu(h_imgT, h_img1, -meanValue, totalSize1);
	multicpu(h_reg, h_imgT, h_imgT, totalSize1);
	double sumSqrA = sumcpu(h_reg, totalSize1);
	valueStatic2D = float(sqrt(sumSqrA));
	if (valueStatic2D == 0) {
		fprintf(stderr, "*** SD of image 1 is zero, empty image input **** \n");
		exit(1);
	}
	cudaMemcpy(d_img2D, h_imgT, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);

	meanValue = (float)sumcpu(h_img2, totalSize2) / totalSize2;
	addvaluecpu(h_imgT, h_img2, -meanValue, totalSize2);
	cudaMemcpyToArray(d_Array2D, 0, 0, h_imgT, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
	BindTexture2D(d_Array2D, channelDesc2D);
	cudaCheckErrors("****Fail to bind 2D texture!!!!*****\n");
	itNumStatic2D = 0;
	ctime1 = clock();
	records[1] = -costfunc2D(p2D);
	ctime2 = clock();
	if (affMethod > 0) {
		powell(p2D, xi2D, DIM2D, FTOL, &iter, &fret, costfunc2D, &itNumStatic2D, itLimit);
		memcpy(iTmx, h_aff2D, DIM2D * sizeof(float));
	}
	UnbindTexture2D;
	ctime3 = clock();

	cudaMemcpyToArray(d_Array2D, 0, 0, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
	BindTexture2D(d_Array2D, channelDesc2D);
	affineTransform2D(d_img2D, imx2D1, imy2D1, imx2D2, imy2D2);
	UnbindTexture2D;
	cudaMemcpy(h_reg, d_img2D, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);

	records[3] = -fret;
	records[4] = (float)(ctime2 - ctime1);
	records[5] = itNumStatic2D;
	records[6] = (float)(ctime3 - ctime2) / CLOCKS_PER_SEC;
	free(p2D);
	free(h_aff2D);
	free_matrix(xi2D, 1, DIM2D, 1, DIM2D);
	free(h_imgT);
	cudaFree(d_img2D);
	cudaFreeArray(d_Array2D);

	end = clock();
	records[7] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

extern "C"
int affinetrans3d0(float *h_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	// cpu
	return 0;
}
extern "C"
int affinetrans3d1(float *d_odata, float *iTmx, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_ArrayTemp;
	cudaMalloc3DArray(&d_ArrayTemp, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaThreadSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudacopydevicetoarray(d_ArrayTemp, channelDesc, d_idata, sx2, sy2, sz2);
	BindTexture(d_ArrayTemp, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_odata, sx, sy, sz, sx2, sy2, sz2);
	UnbindTexture();
	cudaFreeArray(d_ArrayTemp);
	return 0;
}
extern "C"
int affinetrans3d2(float *d_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_ArrayTemp;
	cudaMalloc3DArray(&d_ArrayTemp, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaThreadSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudacopyhosttoarray(d_ArrayTemp, channelDesc, h_idata, sx2, sy2, sz2);
	BindTexture(d_ArrayTemp, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_odata, sx, sy, sz, sx2, sy2, sz2);
	UnbindTexture();
	cudaFreeArray(d_ArrayTemp);
	return 0;
}

float costfunc(float *x) {
	if (dof9Flag) {
		dof9tomatrix(affCoef, x, dofNum);
	}
	else {
		p2matrix(affCoef, x);
	}
	float costValue = corrfunc(d_imgStatic, valueStatic, affCoef, sxStatic1, syStatic1, szStatic1, sxStatic2, syStatic2, szStatic2);

	itNumStatic += 1;
	return -costValue;
}

float costfunccpu(float *x) { // **** this function does not work correctly
	if (dof9Flag) {
		dof9tomatrix(affCoef, x, dofNum);
	}
	else {
		p2matrix(affCoef, x);
	}

	double costValue = corrfunccpu2(h_s3D, h_t3D, affCoef, sxStatic1, syStatic1, szStatic1, sxStatic2, syStatic2, szStatic2);

	itNumStatic += 1;
	return (float)(-costValue / valueStatic);
}

extern "C"
float zncc0(float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz) {
	return 0;
}
extern "C"
float zncc1(float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz) {
	// d_img1, d_img2 value change after calculation
	float znccValue = -2.0;
	long long int totalSize = sx*sy*sz;
	float *d_imgT = NULL;
	cudaMalloc((void **)&d_imgT, totalSize * sizeof(float));
	cudaCheckErrors("****GPU memory allocating fails... GPU out of memory !!!!*****\n");
	double sumImg1 = 0, sumImg2 = 0, sumST = 0, sumSS = 0, sumTT = 0;
	sumImg1 = sum3Dgpu(d_img1, sx, sy, sz);
	sumImg2 = sum3Dgpu(d_img2, sx, sy, sz);
	addvaluegpu(d_img1, d_img1, -float(sumImg1) / float(totalSize), sx, sy, sz);
	addvaluegpu(d_img2, d_img2, -float(sumImg2) / float(totalSize), sx, sy, sz);
	multi3Dgpu(d_imgT, d_img1, d_img2, sx, sy, sz);
	sumST = sum3Dgpu(d_imgT, sx, sy, sz);
	multi3Dgpu(d_imgT, d_img1, d_img1, sx, sy, sz);
	sumTT = sum3Dgpu(d_imgT, sx, sy, sz);
	multi3Dgpu(d_imgT, d_img2, d_img2, sx, sy, sz);
	sumSS = sum3Dgpu(d_imgT, sx, sy, sz);
	cudaFree(d_imgT);
	float b = float(sqrt(sumTT*sumSS));
	if (b != 0)
		znccValue = sumST / b;
	return znccValue;
}
extern "C"
float zncc2(float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz) {
	// d_img1, d_img2 value change after calculation
	float znccValue = -2.0;
	long long int totalSize = sx*sy*sz;
	double sumImg1 = 0, sumImg2 = 0, sumST = 0, sumSS = 0, sumTT = 0;
	float *h_img1 = (float *)malloc(totalSize * sizeof(float));
	sumImg1 = sum3Dgpu(d_img1, sx, sy, sz);
	sumImg2 = sum3Dgpu(d_img2, sx, sy, sz);
	addvaluegpu(d_img1, d_img1, -float(sumImg1) / float(totalSize), sx, sy, sz);
	addvaluegpu(d_img2, d_img2, -float(sumImg2) / float(totalSize), sx, sy, sz);
	cudaMemcpy(h_img1, d_img1, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

	multi3Dgpu(d_img1, d_img1, d_img1, sx, sy, sz);
	sumTT = sum3Dgpu(d_img1, sx, sy, sz);
	cudaMemcpy(d_img1, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	multi3Dgpu(d_img1, d_img1, d_img2, sx, sy, sz);
	sumST = sum3Dgpu(d_img1, sx, sy, sz);
	multi3Dgpu(d_img2, d_img2, d_img2, sx, sy, sz);
	sumSS = sum3Dgpu(d_img2, sx, sy, sz);
	free(h_img1);
	float b = float(sqrt(sumTT*sumSS));
	if (b != 0)
		znccValue = sumST / b;
	return znccValue;
}
extern "C"

extern "C"
int reg3d_phasor0(long long int *shiftXYZ, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz) {
	return 0;
}
extern "C"
int reg3d_phasor1(long long int *shiftXYZ, float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz) {
	int totalSize = sx * sy * sz;
	int totalSizeSpectrum = sz * sy*(sx / 2 + 1); // in complex floating format
	fComplex *d_Spectrum1 = NULL, *d_Spectrum2 = NULL;
	cudaMalloc((void **)&d_Spectrum1, totalSizeSpectrum * sizeof(fComplex));
	cudaMalloc((void **)&d_Spectrum2, totalSizeSpectrum * sizeof(fComplex));
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	cufftPlan3d(&fftPlanFwd, sz, sy, sx, CUFFT_R2C);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_img1, (cufftComplex *)d_Spectrum2);
	conj3Dgpu(d_Spectrum1, d_Spectrum2, sz, sy, (sx / 2 + 1));
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_img2, (cufftComplex *)d_Spectrum2);
	// multiplication and normalization
	multicomplexnorm3Dgpu(d_Spectrum2, d_Spectrum1, d_Spectrum2, sz, sy, (sx / 2 + 1));
	cufftDestroy(fftPlanFwd);
	cufftPlan3d(&fftPlanInv, sz, sy, sx, CUFFT_C2R);
	float *d_phasor1 = (float *)d_Spectrum1;
	cufftExecC2R(fftPlanInv, (cufftComplex *)d_Spectrum2, (cufftReal *)d_phasor1);
	cufftDestroy(fftPlanInv);
	size_t corXYZ[3];
	float *d_phasor2 = (float *)d_Spectrum2;
	circshiftgpu(d_phasor2, d_phasor1, sx, sy, sz, round(sx / 2), round(sy / 2), round(sz / 2));
	float peakValue = max3Dgpu(&corXYZ[0], d_phasor2, sx, sy, sz);
	shiftXYZ[0] = long long int(corXYZ[0]) - sx / 2;
	shiftXYZ[1] = long long int(corXYZ[1]) - sy / 2;
	shiftXYZ[2] = long long int(corXYZ[2]) - sz / 2;
	cudaFree(d_Spectrum1);
	cudaFree(d_Spectrum2);
	
	// compare 8 cases based on cross-correlation
	long long int shiftX = shiftXYZ[0];
	long long int shiftY = shiftXYZ[1];
	long long int shiftZ = shiftXYZ[2];
	long long int xabs = abs(shiftX), yabs = abs(shiftY), zabs = abs(shiftZ);
	long long int beta = 4; // threshold value: only if shift is more than 1/beta of the image size
	if ((xabs >(sx /beta)) ||( yabs >(sy / beta)) || (zabs >(sz / beta))) {
		float *d_imgT = NULL, *d_crop1 = NULL, *d_crop2 = NULL;
		long long int sizex1, sizex2, sizey1, sizey2, sizez1, sizez2, sizex, sizey, sizez, sizeMaxCrop;
		sizex1 = xabs * sy * sz; sizex2 = (sx - xabs) * sy * sz;
		sizey1 = sx *yabs * sz; sizey2 = sx * (sy - yabs) * sz;
		sizez1 = sx * sy * zabs; sizez2 = sx * sy * (sz - zabs);
		sizex = (sizex1 > sizex2) ? sizex1 : sizex2;
		sizey = (sizey1 > sizey2) ? sizey1 : sizey2;
		sizez = (sizez1 > sizez2) ? sizez1 : sizez2;
		sizeMaxCrop = (sizex > sizey) ? sizex : sizey;
		sizeMaxCrop = (sizeMaxCrop > sizez) ? sizeMaxCrop : sizez;
		cudaMalloc((void **)&d_imgT, totalSize * sizeof(float));
		cudaMalloc((void **)&d_crop1, sizeMaxCrop * sizeof(float));
		cudaMalloc((void **)&d_crop2, sizeMaxCrop * sizeof(float));
		circshiftgpu(d_imgT, d_img2, sx, sy, sz, -shiftX, -shiftY, -shiftZ);
		// encode the 8 cases as for loop
		long long int imSizeCropx[2], imSizeCropy[2], imSizeCropz[2];
		long long int imox[2], imoy[2], imoz[2];
		// index 0 records original shifts, index 1 switches the shift to the opposite case.  
		imSizeCropx[0] = sx - xabs; imSizeCropx[1] = xabs;
		if (shiftX > 0) {
			imox[0] = 0; imox[1] = sx - xabs;
		}
		else {
			imox[0] = xabs; imox[1] = 0;
		}
		imSizeCropy[0] = sy - yabs; imSizeCropy[1] = yabs;
		if (shiftY > 0) {
			imoy[0] = 0; imoy[1] = sy - yabs;
		}
		else {
			imoy[0] = yabs; imoy[1] = 0;
		}
		imSizeCropz[0] = sz - zabs; imSizeCropz[1] = zabs;
		if (shiftZ > 0) {
			imoz[0] = 0; imoz[1] = sz - zabs;
		}
		else {
			imoz[0] = zabs; imoz[1] = 0;
		}

		int indx = 0, indy = 0, indz = 0;
		float ccMax = -3, ccNow = 0;
		for (int i = 0; i < 2; i++) {
			if (imSizeCropx[i] > (sx / beta)) {
				for (int j = 0; j < 2; j++) {
					if (imSizeCropy[j] > (sy / beta)) {
						for (int k = 0; k < 2; k++) {
							if (imSizeCropz[k] > (sz / beta)) {
								cropgpu2(d_crop1, d_img1, imSizeCropx[i], imSizeCropy[j], imSizeCropz[k], sx, sy, sz, imox[i], imoy[j], imoz[k]);
								cropgpu2(d_crop2, d_imgT, imSizeCropx[i], imSizeCropy[j], imSizeCropz[k], sx, sy, sz, imox[i], imoy[j], imoz[k]);
								ccNow = zncc1(d_crop1, d_crop2, imSizeCropx[i], imSizeCropy[j], imSizeCropz[k]);
								if (ccMax < ccNow) {
									ccMax = ccNow;
									indx = i;
									indy = j;
									indz = k;
								}
							}
						}
					}
				}
			}
		}
		// if ind ==1, flip the coordinates
		if (indx == 1) {
			if (shiftX > 0)
				shiftXYZ[0] = shiftX - sx;
			else
				shiftXYZ[0] = shiftX + sx;
		}
		if (indy == 1) {
			if (shiftY > 0)
				shiftXYZ[1] = shiftY - sy;
			else
				shiftXYZ[1] = shiftY + sy;
		}
		if (indz == 1) {
			if (shiftZ > 0)
				shiftXYZ[2] = shiftZ - sz;
			else
				shiftXYZ[2] = shiftZ + sz;
		}
		cudaFree(d_imgT);
		cudaFree(d_crop1);
		cudaFree(d_crop2);
	}
	return 0;
}
extern "C"
int reg3d_phasor2(long long int *shiftXYZ, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz) {
	int totalSize = sx * sy * sz;
	int totalSizeSpectrum = sz * sy*(sx / 2 + 1); // in complex floating format
	fComplex *d_Spectrum1 = NULL, *d_Spectrum2 = NULL;
	cudaMalloc((void **)&d_Spectrum1, totalSizeSpectrum * sizeof(fComplex));
	cudaMalloc((void **)&d_Spectrum2, totalSizeSpectrum * sizeof(fComplex));
	float *d_img = (float *)d_Spectrum1;
	fComplex *h_Spectrum1 = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	cufftPlan3d(&fftPlanFwd, sz, sy, sx, CUFFT_R2C);
	cudaMemcpy(d_img, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_img, (cufftComplex *)d_Spectrum2);
	conj3Dgpu(d_Spectrum1, d_Spectrum2, sz, sy, (sx / 2 + 1));
	cudaMemcpy(h_Spectrum1, d_Spectrum1, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);

	cudaMemcpy(d_img, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_img, (cufftComplex *)d_Spectrum2);
	// multiplication and normalization
	cudaMemcpy(d_Spectrum1, h_Spectrum1, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
	multicomplexnorm3Dgpu(d_Spectrum2, d_Spectrum1, d_Spectrum2, sz, sy, (sx / 2 + 1));
	cufftDestroy(fftPlanFwd);
	cufftPlan3d(&fftPlanInv, sz, sy, sx, CUFFT_C2R);
	cufftExecC2R(fftPlanInv, (cufftComplex *)d_Spectrum2, (cufftReal *)d_img);
	cufftDestroy(fftPlanInv);
	size_t corXYZ[3];
	float *d_phasor2 = (float *)d_Spectrum2;
	circshiftgpu(d_phasor2, d_img, sx, sy, sz, round(sx / 2), round(sy / 2), round(sz / 2));
	float peakValue = max3Dgpu(&corXYZ[0], d_phasor2, sx, sy, sz);
	shiftXYZ[0] = long long int(corXYZ[0]) - sx / 2;
	shiftXYZ[1] = long long int(corXYZ[1]) - sy / 2;
	shiftXYZ[2] = long long int(corXYZ[2]) - sz / 2;
	cudaFree(d_Spectrum1);
	cudaFree(d_Spectrum2);

	// compare 8 cases based on cross-correlation
	long long int shiftX = shiftXYZ[0];
	long long int shiftY = shiftXYZ[1];
	long long int shiftZ = shiftXYZ[2];
	long long int xabs = abs(shiftX), yabs = abs(shiftY), zabs = abs(shiftZ);
	long long int beta = 4; // threshold value: only if shift is more than 1/beta of the image size
	if ((xabs >(sx / beta)) || (yabs >(sy / beta)) || (zabs >(sz / beta))) {
		float *d_img1 = NULL, *d_imgT = NULL, *d_crop1 = NULL, *d_crop2 = NULL;
		long long int sizex1, sizex2, sizey1, sizey2, sizez1, sizez2, sizex, sizey, sizez, sizeMaxCrop;
		sizex1 = xabs * sy * sz; sizex2 = (sx - xabs) * sy * sz;
		sizey1 = sx *yabs * sz; sizey2 = sx * (sy - yabs) * sz;
		sizez1 = sx * sy * zabs; sizez2 = sx * sy * (sz - zabs);
		sizex = (sizex1 > sizex2) ? sizex1 : sizex2;
		sizey = (sizey1 > sizey2) ? sizey1 : sizey2;
		sizez = (sizez1 > sizez2) ? sizez1 : sizez2;
		sizeMaxCrop = (sizex > sizey) ? sizex : sizey;
		sizeMaxCrop = (sizeMaxCrop > sizez) ? sizeMaxCrop : sizez;
		cudaMalloc((void **)&d_img1, totalSize * sizeof(float));
		cudaMalloc((void **)&d_imgT, totalSize * sizeof(float));
		cudaMalloc((void **)&d_crop1, sizeMaxCrop * sizeof(float));
		cudaMalloc((void **)&d_crop2, sizeMaxCrop * sizeof(float));
		cudaMemcpy(d_img1, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		circshiftgpu(d_imgT, d_img1, sx, sy, sz, -shiftX, -shiftY, -shiftZ);
		cudaMemcpy(d_img1, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		// encode the 8 cases as for loop
		long long int imSizeCropx[2], imSizeCropy[2], imSizeCropz[2];
		long long int imox[2], imoy[2], imoz[2];
		// index 0 records original shifts, index 1 switches the shift to the opposite case.  
		imSizeCropx[0] = sx - xabs; imSizeCropx[1] = xabs;
		if (shiftX > 0) {
			imox[0] = 0; imox[1] = sx - xabs;
		}
		else {
			imox[0] = xabs; imox[1] = 0;
		}
		imSizeCropy[0] = sy - yabs; imSizeCropy[1] = yabs;
		if (shiftY > 0) {
			imoy[0] = 0; imoy[1] = sy - yabs;
		}
		else {
			imoy[0] = yabs; imoy[1] = 0;
		}
		imSizeCropz[0] = sz - zabs; imSizeCropz[1] = zabs;
		if (shiftZ > 0) {
			imoz[0] = 0; imoz[1] = sz - zabs;
		}
		else {
			imoz[0] = zabs; imoz[1] = 0;
		}

		int indx = 0, indy = 0, indz = 0;
		float ccMax = -3, ccNow = 0;
		for (int i = 0; i < 2; i++) {
			if (imSizeCropx[i] >(sx / beta)) {
				for (int j = 0; j < 2; j++) {
					if (imSizeCropy[j] >(sy / beta)) {
						for (int k = 0; k < 2; k++) {
							if (imSizeCropz[k] >(sz / beta)) {
								cropgpu2(d_crop1, d_img1, imSizeCropx[i], imSizeCropy[j], imSizeCropz[k], sx, sy, sz, imox[i], imoy[j], imoz[k]);
								cropgpu2(d_crop2, d_imgT, imSizeCropx[i], imSizeCropy[j], imSizeCropz[k], sx, sy, sz, imox[i], imoy[j], imoz[k]);
								ccNow = zncc1(d_crop1, d_crop2, imSizeCropx[i], imSizeCropy[j], imSizeCropz[k]);
								if (ccMax < ccNow) {
									ccMax = ccNow;
									indx = i;
									indy = j;
									indz = k;
								}
							}
						}
					}
				}
			}
		}
		// if ind ==1, flip the coordinates
		if (indx == 1) {
			if (shiftX > 0)
				shiftXYZ[0] = shiftX - sx;
			else
				shiftXYZ[0] = shiftX + sx;
		}
		if (indy == 1) {
			if (shiftY > 0)
				shiftXYZ[1] = shiftY - sy;
			else
				shiftXYZ[1] = shiftY + sy;
		}
		if (indz == 1) {
			if (shiftZ > 0)
				shiftXYZ[2] = shiftZ - sz;
			else
				shiftXYZ[2] = shiftZ + sz;
		}
		cudaFree(d_img1);
		cudaFree(d_imgT);
		cudaFree(d_crop1);
		cudaFree(d_crop2);
	}
	return 0;
}

int reg3d_affine0(float *h_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz,
	int affMethod, bool flagTmx, float FTOL, int itLimit, bool verbose, float *records) {
	return 0;
}
extern "C"
int reg3d_affine1(float *d_reg, float *iTmx, float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz, 
	int affMethod, bool flagTmx, float FTOL, int itLimit, bool verbose, float *records) {
	// **** affine registration when GPU memory is sufficient: 3 images + 1 cuda array ***
	/*
	*** affine registration method: 
		0: no registration, transform d_img2 based on input matrix;
		1: translation only; 
		2: rigid body; 
		3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)  
		4: 9 degrees of freedom(translation, rotation, scaling); 
		5: 12 degrees of freedom; 
		6: rigid body first, then do 12 degrees of freedom; 
		7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*** flagTmx: 
		true: use iTmx as input matrix;
		false: default; 
		
	*** records: 8 element array
		[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
		[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	*/

	// ************get basic input images information ******************	
	// image size
	sxStatic1 = sx; syStatic1 = sy; szStatic1 = sz;
	sxStatic2 = sx; syStatic2 = sy; szStatic2 = sz;
	// total pixel count for each image
	long long int totalSize = sx*sy*sz;
	// ****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t ctime0, ctime1, ctime2, ctime3, ctime4;
	ctime0 = clock();

	// *** no registration
	if (affMethod == 0) {
		if (flagTmx)
			(void)affinetrans3d1(d_reg, iTmx, d_img2, sx, sy, sz, sx, sy, sz);
		else {
			cudaMemcpy(d_reg, d_img2, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
		}
		ctime4 = clock();
		records[7] = (float)(ctime4 - ctime0) / CLOCKS_PER_SEC;
		if (verbose) {
			printf("...no registration performed!\n");
		}
		return 0;
	}
	// *** registration
	// for powell searching
	affCoef = (float *)malloc((NDIM) * sizeof(float));
	float *affCoefInitial = (float *)malloc((NDIM) * sizeof(float));
	static float *p = (float *)malloc((NDIM + 1) * sizeof(float));
	int iter;
	float fret, **xi;
	xi = matrix(1, NDIM, 1, NDIM);
	for (int i = 1; i <= NDIM; i++)
		for (int j = 1; j <= NDIM; j++)
			xi[i][j] = (i == j ? 1.0 : 0.0);
	for (int j = 0; j < NDIM; j++) affCoefInitial[j] = 0;
	affCoefInitial[0] = 1;
	affCoefInitial[5] = 1;
	affCoefInitial[10] = 1;

	float *affCoefTemp = (float *)malloc((NDIM) * sizeof(float));
	float **xi_dof9;
	static float *p_dof9 = (float *)malloc((10) * sizeof(float));
	xi_dof9 = matrix(1, 9, 1, 9);

	// **** allocate memory for the images: 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;

	// *****************************************************
	// ************** Start processing ******************
	double
		sumImg1 = 0,
		sumImg2 = 0,
		sumSqr1 = 0;
	// ****** the definition of 12 DOF coefficients is totally diffrent with that of 3 DOF, 6 DOF, 7 DOF or 9 DOF;
	// if related to 3 DOF, 6 DOF, 7 DOF or 9 DOF (e.i. affMethod = 1, 2, 3, 4, 6, 7)
	// then perfrom initial affine transformation based on input matrix
	// *initialize transformation matrix
	if (flagTmx) {
		if (affMethod == 5) {
			// use input matrix as initialization if inputTmx is true
			memcpy(affCoefInitial, iTmx, NDIM * sizeof(float));
		}
		else {
			// make affine transformation
			(void)affinetrans3d1(d_reg, iTmx, d_img2, sx, sy, sz, sx, sy, sz);
		}
	}
	if(affMethod != 5) {
		xi_dof9 = matrix(1, 9, 1, 9);
		for (int i = 1; i <= 9; i++)
			for (int j = 1; j <= 9; j++)
				xi_dof9[i][j] = (i == j ? 1.0 : 0.0);
		p_dof9[0] = 0;
		p_dof9[1] = 0; p_dof9[2] = 0; p_dof9[3] = 0;
		p_dof9[4] = 0; p_dof9[5] = 0; p_dof9[6] = 0;
		p_dof9[7] = 1; p_dof9[8] = 1; p_dof9[9] = 1;
	}
	// *** preprocess source image
	if ((flagTmx)&&(affMethod != 5)) { // based on tranformed image
		sumImg2 = sum3Dgpu(d_reg, sx, sy, sz);
		addvaluegpu(d_reg, d_reg, -float(sumImg2) / float(totalSize), sx, sy, sz);
	}
	else {//based on input d_img2
		sumImg2 = sum3Dgpu(d_img2, sx, sy, sz);
		addvaluegpu(d_reg, d_img2, -float(sumImg2) / float(totalSize), sx, sy, sz);
	}
	// transfer source image into GPU array (later converted to texture memory)
	cudaMalloc3DArray(&d_Array, &channelDesc, make_cudaExtent(sx, sy, sz));
	cudaCheckErrors("****GPU memory allocating fails... GPU out of memory !!!!*****\n");
	cudacopydevicetoarray(d_Array, channelDesc, d_reg, sx, sy, sz);
	multi3Dgpu(d_reg, d_reg, d_reg, sx, sy, sz);
	sumSqr1 = sum3Dgpu(d_reg, sx, sy, sz);
	valueStatic = sqrt(sumSqr1);
	if (valueStatic == 0) {
		fprintf(stderr, "*** SD of image 2 is zero, empty image input or empty image after initial transformation **** \n");
		exit(1);
	}

	// *** preprocess target image
	sumImg1 = sum3Dgpu(d_img1, sx, sy, sz);
	addvaluegpu(d_reg, d_img1, -float(sumImg1) / float(totalSize), sx, sy, sz);
	multi3Dgpu(d_reg, d_reg, d_reg, sx, sy, sz);
	sumSqr1 = sum3Dgpu(d_reg, sx, sy, sz);
	valueStatic = sqrt(sumSqr1);
	if (valueStatic == 0) {
		fprintf(stderr, "*** SD of image 1 is zero, empty image input **** \n"); 
		exit(1);
	}
	addvaluegpu(d_reg, d_img1, -float(sumImg1) / float(totalSize), sx, sy, sz);
	cudaCheckErrors("****Image preprocessing fails...");

	// *** 3D registration begains
	// Create 3D texture for source image
	BindTexture(d_Array, channelDesc);
	// make target image as static
	d_imgStatic = d_reg;
	// calculate initial cost function value and time cost for each sub iteration
	ctime1 = clock();
	dof9Flag = false;
	matrix2p(affCoefInitial, p);
	ctime2 = clock();
	records[1] = -costfunc(p);
	records[4] = (float)(ctime2 - ctime1);
	if (verbose) {
		printf("...initial cross correlation value: %f;\n", records[1]);
		printf("...time cost for single sub iteration: %f ms;\n", records[4]);
	}

	itNumStatic = 0;
	switch (affMethod) {
	case 1:
		dof9Flag = true;
		dofNum = 3;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 2:
		dof9Flag = true;
		dofNum = 6;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 3:
		dof9Flag = true;
		dofNum = 7;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 4:
		dof9Flag = true;
		dofNum = 9;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 5:
		dof9Flag = false;
		dofNum = 12;
		powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 6:
		// do 6 DOF --> 12 DOF
		dof9Flag = true;
		dofNum = 6;
		powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &itNumStatic, itLimit);
		records[2] = -fret;
		if (verbose) {
			printf("...cross correlation value after 6 DOF: %f;\n", -fret);
		}
		// do DOF 12 registration
		dof9Flag = false;
		dofNum = 12;
		matrix2p(affCoef, p);
		powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 7:
		// do 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
		dof9Flag = true;
		dofNum = 3;
		powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &itNumStatic, itLimit);
		if (verbose) {
			printf("...cross correlation value after 3 DOF: %f;\n", -fret);
		}
		dofNum = 6;
		powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &itNumStatic, itLimit);
		if (verbose) {
			printf("...cross correlation value after 6 DOF: %f;\n", -fret);
		}
		dofNum = 9;
		powell(p_dof9, xi_dof9, dofNum, 0.005, &iter, &fret, costfunc, &itNumStatic, itLimit);
		records[2] = -fret;
		if (verbose) {
			printf("...cross correlation value after 9 DOF: %f;\n", -fret);
		}
		// do DOF 12 registration
		dof9Flag = false;
		dofNum = 12;
		matrix2p(affCoef, p);
		powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	default:
		printf("\n ****Wrong affine registration method is setup, no registraiton performed !!! **** \n");
	}
	if ((flagTmx) && (affMethod != 5)) {
		matrixmultiply(affCoefTemp, iTmx, affCoef); //final transformation matrix
		memcpy(affCoef, affCoefTemp, NDIM * sizeof(float));
	}
	UnbindTexture();
	memcpy(iTmx, affCoef, NDIM * sizeof(float));
	ctime3 = clock();
	records[3] = -fret; // negative of the mimized cost function value
	records[5] = (float)itNumStatic;
	records[6] = (float)(ctime3 - ctime2) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("...optimized cross correlation value: %f;\n", records[3]);
		printf("...total sub iteration number: %d;\n", int(records[5]));
		printf("...time cost for all iterations: %f s;\n", records[6]);
	}
	// ****Perform affine transformation with optimized coefficients****//
	cudacopydevicetoarray(d_Array, channelDesc, d_img2, sx, sy, sz);
	BindTexture(d_Array, channelDesc);
	CopyTranMatrix(affCoef, NDIM * sizeof(float));
	affineTransform(d_reg, sx, sy, sz, sx, sy, sz);
	UnbindTexture();

	free(affCoefTemp);
	free(p_dof9);
	free_matrix(xi_dof9, 1, 9, 1, 9);

	free(affCoef);
	free(affCoefInitial);
	free(p);
	free_matrix(xi, 1, NDIM, 1, NDIM);

	//free GPU variables
	cudaFreeArray(d_Array);
	ctime4 = clock();
	records[7] = (float)(ctime4 - ctime0) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("...time cost for registration: %f s;\n", records[7]);
	}
	return 0;
}
extern "C"
int reg3d_affine2(float *d_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz,
	int affMethod, bool flagTmx, float FTOL, int itLimit, bool verbose, float *records) {
	// **** affine registration when GPU memory is insufficient: 1 image + 1 cuda array ***
	/*
	*** affine registration method:
	0: no registration, transform d_img2 based on input matrix;
	1: translation only;
	2: rigid body;
	3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)
	4: 9 degrees of freedom(translation, rotation, scaling);
	5: 12 degrees of freedom;
	6: rigid body first, then do 12 degrees of freedom;
	7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*** flagTmx:
	true: use iTmx as input matrix;
	false: default;

	*** records: 8 element array
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	*/

	// ************get basic input images information ******************	
	// image size
	sxStatic1 = sx; syStatic1 = sy; szStatic1 = sz;
	sxStatic2 = sx; syStatic2 = sy; szStatic2 = sz;
	// total pixel count for each image
	long long int totalSize = sx*sy*sz;
	// ****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t ctime0, ctime1, ctime2, ctime3, ctime4;
	ctime0 = clock();

	// *** no registration
	if (affMethod == 0) {
		if (flagTmx)
			(void)affinetrans3d2(d_reg, iTmx, h_img2, sx, sy, sz, sx, sy, sz);
		else {
			cudaMemcpy(d_reg, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
		}
		ctime4 = clock();
		records[7] = (float)(ctime4 - ctime0) / CLOCKS_PER_SEC;
		if (verbose) {
			printf("...no registration performed!\n");
		}
		return 0;
	}
	// *** registration
	// for powell searching
	affCoef = (float *)malloc((NDIM) * sizeof(float));
	float *affCoefInitial = (float *)malloc((NDIM) * sizeof(float));
	static float *p = (float *)malloc((NDIM + 1) * sizeof(float));
	int iter;
	float fret, **xi;
	xi = matrix(1, NDIM, 1, NDIM);
	for (int i = 1; i <= NDIM; i++)
		for (int j = 1; j <= NDIM; j++)
			xi[i][j] = (i == j ? 1.0 : 0.0);
	for (int j = 0; j < NDIM; j++) affCoefInitial[j] = 0;
	affCoefInitial[0] = 1;
	affCoefInitial[5] = 1;
	affCoefInitial[10] = 1;

	float *affCoefTemp = (float *)malloc((NDIM) * sizeof(float));
	float **xi_dof9;
	static float *p_dof9 = (float *)malloc((10) * sizeof(float));
	xi_dof9 = matrix(1, 9, 1, 9);

	// **** allocate memory for the images: 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;
	float *h_imgTemp = (float *)malloc(totalSize * sizeof(float));

	// *****************************************************
	// ************** Start processing ******************
	double
		sumImg1 = 0,
		sumImg2 = 0,
		sumSqr1 = 0;
	// ****** the definition of 12 DOF coefficients is totally diffrent with that of 3 DOF, 6 DOF, 7 DOF or 9 DOF;
	// if related to 3 DOF, 6 DOF, 7 DOF or 9 DOF (e.i. affMethod = 1, 2, 3, 4, 6, 7)
	// then perfrom initial affine transformation based on input matrix
	// *initialize transformation matrix
	if (flagTmx) {
		if (affMethod == 5) {
			// use input matrix as initialization if inputTmx is true
			memcpy(affCoefInitial, iTmx, NDIM * sizeof(float));
		}
		else {
			// make affine transformation
			(void)affinetrans3d2(d_reg, iTmx, h_img2, sx, sy, sz, sx, sy, sz);
		}
	}
	if (affMethod != 5) {
		xi_dof9 = matrix(1, 9, 1, 9);
		for (int i = 1; i <= 9; i++)
			for (int j = 1; j <= 9; j++)
				xi_dof9[i][j] = (i == j ? 1.0 : 0.0);
		p_dof9[0] = 0;
		p_dof9[1] = 0; p_dof9[2] = 0; p_dof9[3] = 0;
		p_dof9[4] = 0; p_dof9[5] = 0; p_dof9[6] = 0;
		p_dof9[7] = 1; p_dof9[8] = 1; p_dof9[9] = 1;
	}
	// *** preprocess source image
	if ((flagTmx) && (affMethod != 5)) { // based on tranformed image
		sumImg2 = sum3Dgpu(d_reg, sx, sy, sz);
		addvaluegpu(d_reg, d_reg, -float(sumImg2) / float(totalSize), sx, sy, sz);
	}
	else {//based on input d_img2
		cudaMemcpy(d_reg, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		sumImg2 = sum3Dgpu(d_reg, sx, sy, sz);
		addvaluegpu(d_reg, d_reg, -float(sumImg2) / float(totalSize), sx, sy, sz);
	}
	// transfer source image into GPU array (later converted to texture memory)
	cudaMalloc3DArray(&d_Array, &channelDesc, make_cudaExtent(sx, sy, sz));
	cudaCheckErrors("****GPU memory allocating fails... GPU out of memory !!!!*****\n");
	cudacopydevicetoarray(d_Array, channelDesc, d_reg, sx, sy, sz);
	multi3Dgpu(d_reg, d_reg, d_reg, sx, sy, sz);
	sumSqr1 = sum3Dgpu(d_reg, sx, sy, sz);
	valueStatic = sqrt(sumSqr1);
	if (valueStatic == 0) {
		fprintf(stderr, "*** SD of image 2 is zero, empty image input or empty image after initial transformation **** \n");
		exit(1);
	}

	// *** preprocess target image
	cudaMemcpy(d_reg, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	sumImg1 = sum3Dgpu(d_reg, sx, sy, sz);
	addvaluegpu(d_reg, d_reg, -float(sumImg1) / float(totalSize), sx, sy, sz);
	cudaMemcpy(h_imgTemp, d_reg, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	multi3Dgpu(d_reg, d_reg, d_reg, sx, sy, sz);
	sumSqr1 = sum3Dgpu(d_reg, sx, sy, sz);
	valueStatic = sqrt(sumSqr1);
	if (valueStatic == 0) {
		fprintf(stderr, "*** SD of image 1 is zero, empty image input **** \n");
		exit(1);
	}
	cudaMemcpy(d_reg, h_imgTemp, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors("****Image preprocessing fails...");

	// *** 3D registration begains
	// Create 3D texture for source image
	BindTexture(d_Array, channelDesc);
	// make target image as static
	d_imgStatic = d_reg;
	// calculate initial cost function value and time cost for each sub iteration
	ctime1 = clock();
	dof9Flag = false;
	matrix2p(affCoefInitial, p);
	ctime2 = clock();
	records[1] = -costfunc(p);
	records[4] = (float)(ctime2 - ctime1);
	if (verbose) {
		printf("...initial cross correlation value: %f;\n", records[1]);
		printf("...time cost for single sub iteration: %f ms;\n", records[4]);
	}

	itNumStatic = 0;
	switch (affMethod) {
	case 1:
		dof9Flag = true;
		dofNum = 3;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 2:
		dof9Flag = true;
		dofNum = 6;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 3:
		dof9Flag = true;
		dofNum = 7;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 4:
		dof9Flag = true;
		dofNum = 9;
		powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 5:
		dof9Flag = false;
		dofNum = 12;
		powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 6:
		// do 6 DOF --> 12 DOF
		dof9Flag = true;
		dofNum = 6;
		powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &itNumStatic, itLimit);
		records[2] = -fret;
		if (verbose) {
			printf("...cross correlation value after 6 DOF: %f;\n", -fret);
		}
		// do DOF 12 registration
		dof9Flag = false;
		dofNum = 12;
		matrix2p(affCoef, p);
		powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	case 7:
		// do 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
		dof9Flag = true;
		dofNum = 3;
		powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &itNumStatic, itLimit);
		if (verbose) {
			printf("...cross correlation value after 3 DOF: %f;\n", -fret);
		}
		dofNum = 6;
		powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &itNumStatic, itLimit);
		if (verbose) {
			printf("...cross correlation value after 6 DOF: %f;\n", -fret);
		}
		dofNum = 9;
		powell(p_dof9, xi_dof9, dofNum, 0.005, &iter, &fret, costfunc, &itNumStatic, itLimit);
		records[2] = -fret;
		if (verbose) {
			printf("...cross correlation value after 9 DOF: %f;\n", -fret);
		}
		// do DOF 12 registration
		dof9Flag = false;
		dofNum = 12;
		matrix2p(affCoef, p);
		powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &itNumStatic, itLimit);
		break;
	default:
		printf("\n ****Wrong affine registration method is setup, no registraiton performed !!! **** \n");
	}
	if ((flagTmx) && (affMethod != 5)) {
		matrixmultiply(affCoefTemp, iTmx, affCoef); //final transformation matrix
		memcpy(affCoef, affCoefTemp, NDIM * sizeof(float));
	}
	UnbindTexture();
	memcpy(iTmx, affCoef, NDIM * sizeof(float));
	ctime3 = clock();
	records[3] = -fret; // negative of the mimized cost function value
	records[5] = (float)itNumStatic;
	records[6] = (float)(ctime3 - ctime2) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("...optimized cross correlation value: %f;\n", records[3]);
		printf("...total sub iteration number: %d;\n", int(records[5]));
		printf("...time cost for all iterations: %f s;\n", records[6]);
	}
	// ****Perform affine transformation with optimized coefficients****//
	cudacopyhosttoarray(d_Array, channelDesc, h_img2, sx, sy, sz);
	BindTexture(d_Array, channelDesc);
	CopyTranMatrix(affCoef, NDIM * sizeof(float));
	affineTransform(d_reg, sx, sy, sz, sx, sy, sz);
	UnbindTexture();
	
	free(h_imgTemp);

	free(affCoefTemp);
	free(p_dof9);
	free_matrix(xi_dof9, 1, 9, 1, 9);

	free(affCoef);
	free(affCoefInitial);
	free(p);
	free_matrix(xi, 1, NDIM, 1, NDIM);

	//free GPU variables
	cudaFreeArray(d_Array);
	ctime4 = clock();
	records[7] = (float)(ctime4 - ctime0) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("...time cost for registration: %f s;\n", records[7]);
	}
	return 0;
}

// Deconvolution
extern "C"
void genOTFgpu(fComplex *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, bool normFlag) {
	long long int totalSizeIn = sx2 * sy2 * sz2;
	long long int totalSizeOut = sx * sy * sz;
	long long int totalSizeMax = (totalSizeIn > totalSizeOut)?totalSizeIn:totalSizeOut;
	float *d_temp = NULL;
	cudaStatus = cudaMalloc((void **)&d_temp, totalSizeMax * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: GPU memory allocating error when calculating OTF \n");
		exit(1);
	}

	if (normFlag) {
		double sumValue = sum3Dgpu(d_idata, sx2, sy2, sz2);
		multivaluegpu(d_temp, d_idata, (float)(1 / sumValue), sx2, sy2, sz2);
	}
	else
		cudaMemcpy(d_temp, d_idata, totalSizeIn * sizeof(float), cudaMemcpyDeviceToDevice);
	if ((sx<sx2) || (sy<sy2) || (sz<sz2)) {
		alignsize3Dgpu((float *)d_odata, d_temp, sx, sy, sz, sx2, sy2, sz2);
		padPSFgpu(d_temp, (float *)d_odata, sx, sy, sz, sx, sy, sz);
	}
	else {
		padPSFgpu((float *)d_odata, d_temp, sx, sy, sz, sx2, sy2, sz2);
		cudaMemcpy(d_temp, d_odata, totalSizeOut * sizeof(float), cudaMemcpyDeviceToDevice);
	}	
	cufftHandle
		fftPlanFwd;
	cufftPlan3d(&fftPlanFwd, sx, sy, sz, CUFFT_R2C);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_temp, (cufftComplex *)d_odata);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: cufftPlan error when calculating OTF \n");
		exit(1);
	}
	cudaFree(d_temp);
	cufftDestroy(fftPlanFwd);
}

extern "C"
int decon_singleview_OTF0(float *h_decon, float *h_img, fftwf_complex *h_OTF, fftwf_complex *h_OTF_bp,
	long long int sx, long long int sy, long long int sz, int itNumForDecon, bool flagConstInitial) {
	// **** single view deconvolution with OTF interface on CPU ***
	// image size
	long long int totalSize = sx*sy*sz; // in floating format
	long long int totalSizeSpectrum = sx * sy*(sz / 2 + 1); // in complex floating format
	clock_t start, end;
	start = clock();
	float *h_StackA = h_img, *h_StackE = h_decon;
	float *h_StackT = (float *)malloc(totalSize * sizeof(float));
	fftwf_complex *h_StackESpectrum = (fftwf_complex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));

	// initialize estimation
	maxvaluecpu(h_StackA, h_StackA, (float)(SMALLVALUE), totalSize);
	// initialize estimation
	if (flagConstInitial) { // use constant mean value as initial
		float meanValue = (float)sumcpu(h_StackA, totalSize);
		memset(h_StackE, 0, totalSize * sizeof(float));
		addvaluecpu(h_StackE, h_StackE, meanValue, totalSize);
	}
	else { // use measured images as initial
		memcpy(h_StackE, h_StackA, totalSize * sizeof(float));
	}

	fftwf_plan stackE2Spectrum = fftwf_plan_dft_r2c_3d(sx, sy, sz, h_StackE, h_StackESpectrum, FFTW_MEASURE);
	fftwf_plan stackT2Spectrum = fftwf_plan_dft_r2c_3d(sx, sy, sz, h_StackT, h_StackESpectrum, FFTW_MEASURE);
	fftwf_plan spectrum2StackT = fftwf_plan_dft_c2r_3d(sx, sy, sz, h_StackESpectrum, h_StackT, FFTW_MEASURE);
	printf("...Start CPU Decon\n");
	for (int itNum = 1; itNum <= itNumForDecon; itNum++) {
		fftwf_execute(stackE2Spectrum);
		multicomplexcpu((fComplex *)h_StackESpectrum, (fComplex *)h_StackESpectrum, (fComplex *)h_OTF, sx * sy * (sz / 2 + 1));
		fftwf_execute(spectrum2StackT);

		divcpu(h_StackT, h_StackA, h_StackT, totalSize);

		fftwf_execute(stackT2Spectrum);
		multicomplexcpu((fComplex *)h_StackESpectrum, (fComplex *)h_StackESpectrum, (fComplex *)h_OTF_bp, sx * sy * (sz / 2 + 1));
		fftwf_execute(spectrum2StackT);
		multicpu(h_StackE, h_StackE, h_StackT, totalSize);//
	}
	free(h_StackT);
	free(h_StackESpectrum);
	fftwf_destroy_plan(stackE2Spectrum);
	fftwf_destroy_plan(stackT2Spectrum);
	fftwf_destroy_plan(spectrum2StackT);
	end = clock();
	printf("...Time cost for decon is %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}
extern "C"
int decon_singleview_OTF1(float *d_decon, float *d_img, fComplex *d_OTF, fComplex *d_OTF_bp, 
	long long int sx, long long int sy, long long int sz, int itNumForDecon, bool flagConstInitial) {
	// **** single view deconvolution with OTF interface when GPU memory is sufficient ***
	// image size
	long long int totalSize = sx*sy*sz; // in floating format
	long long int totalSizeSpectrum = sx * sy*(sz / 2 + 1); // in complex floating format
	size_t freeMem = 0, totalMem = 0;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	clock_t start, end;
	start = clock();
	float *d_StackA = d_img, *d_StackE = d_decon;
	float *d_StackT = NULL;
	fComplex *d_StackESpectrum = NULL;
	cudaMalloc((void **)&d_StackT, totalSize * sizeof(float));
	cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum * sizeof(fComplex));

	// initialize estimation
	maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), sx, sy, sz);
	if(flagConstInitial) {// use constant mean value as initial
		float meanValue = (float)sum3Dgpu(d_StackA, sx, sy, sz);
		cudaMemset(d_StackE, 0, totalSize * sizeof(float));
		addvaluegpu(d_StackE, d_StackE, meanValue, sx, sy, sz);
	}
	else { // use measured image as initial
		cudaMemcpy(d_StackE, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: initial image preparation failed \n");
		exit(1);
	}
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, sx, sy, sz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, sx, sy, sz, CUFFT_C2R);
	cudaMemGetInfo(&freeMem, &totalMem);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: cufftPlan error \n");
		exit(1);
	}
	printf("...GPU free memory (before decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	for (int itNum = 1; itNum <= itNumForDecon; itNum++) {
		// forward
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
		div3Dgpu(d_StackT, d_StackA, d_StackT, sx, sy, sz); 
		// backward
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF_bp, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);																		  
		multi3Dgpu(d_StackE, d_StackE, d_StackT, sx, sy, sz);
		maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), sx, sy, sz); // eliminate possible negative values
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: decon iterration error \n");
		exit(1);
	}
	cudaFree(d_StackT); cudaFree(d_StackESpectrum);
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory (after decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	end = clock();
	printf("...Time cost for decon is %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}
extern "C"
int decon_singleview_OTF2(float *d_decon, float *d_img, fComplex *h_OTF, fComplex *h_OTF_bp,
	long long int sx, long long int sy, long long int sz, int itNumForDecon, bool flagConstInitial) {
	// **** single view deconvolution with OTF interface when GPU memory is insufficient: 2 images + 2 fftPlans ***
	// **** d_decon and d_img should have total size: sx * sy*(sz / 2 + 1) * sizeof(float) to store image spectrum
	// image size
	long long int totalSize = sx*sy*sz; // in floating format
	long long int totalSizeSpectrum = sx * sy*(sz / 2 + 1); // in complex floating format
	// *****
	size_t freeMem = 0, totalMem = 0;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	clock_t start, end;
	start = clock();
	float *h_StackA = NULL, *h_StackE = NULL;
	h_StackA = (float *)malloc(totalSize * sizeof(float));
	h_StackE = (float *)malloc(totalSize * sizeof(float));


	float *d_StackA = d_img, *d_StackE = d_decon;
	fComplex  *d_OTF = NULL, *d_OTF_bp = NULL, *d_StackESpectrum = NULL;
	// initialize estimation
	maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), sx, sy, sz);
	cudaMemcpy(h_StackA, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	//if (initialFlag) // use measured image as initial
	if (flagConstInitial) { // use constant mean value as initial
		float meanValue = (float)sum3Dgpu(d_StackA, sx, sy, sz);	
		cudaMemset(d_StackA, 0, totalSize * sizeof(float));
		addvaluegpu(d_StackA, d_StackA, meanValue, sx, sy, sz);
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: initial image preparation failed \n");
		exit(1);
	}
	cudaMemcpy(h_StackE, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	d_OTF = (fComplex *)d_StackA; // share the same physic memory
	d_OTF_bp = (fComplex *)d_StackA; // share the same physic memory
	d_StackESpectrum = (fComplex *)d_StackE; // share the same physic memory

	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, sx, sy, sz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, sx, sy, sz, CUFFT_C2R);
	cudaMemGetInfo(&freeMem, &totalMem);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: cufftPlan error \n");
		exit(1);
	}
	printf("...GPU free memory (before decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);

	for (int itNum = 1; itNum <= itNumForDecon; itNum++) {
		// forward
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(d_OTF, h_OTF, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);
		cudaMemcpy(d_StackE, h_StackA, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		div3Dgpu(d_StackA, d_StackE, d_StackA, sx, sy, sz); 

		// backward
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(d_OTF_bp, h_OTF_bp, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF_bp, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);																			  
		cudaMemcpy(d_StackE, h_StackE, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		multi3Dgpu(d_StackA, d_StackE, d_StackA, sx, sy, sz);
		maxvalue3Dgpu(d_StackA, d_StackA, float(SMALLVALUE), sx, sy, sz); // eliminate possible negative values
		cudaMemcpy(h_StackE, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(d_StackE, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: decon iterration error \n");
		exit(1);
	}
	free(h_StackA); free(h_StackE);
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory (after decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	end = clock();
	printf("...Time cost for decon is %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}

extern "C"
int decon_dualview_OTF0(float *h_decon, float *h_img1, float *h_img2, fftwf_complex *h_OTF1, fftwf_complex *h_OTF2, fftwf_complex *h_OTF_bp1,
	fftwf_complex *h_OTF_bp2, long long int sx, long long int sy, long long int sz, int itNumForDecon, bool flagConstInitial) {
	// **** dual-view deconvolution with OTF interface on CPU ***
	// image size
	long long int totalSize = sx*sy*sz; // in floating format
	long long int totalSizeSpectrum = sx * sy*(sz / 2 + 1); // in complex floating format
	clock_t start, end;
	start = clock();
	float *h_StackA = h_img1, *h_StackB = h_img2, *h_StackE = h_decon;
	float *h_StackT = (float *)malloc(totalSize * sizeof(float));
	fftwf_complex *h_StackESpectrum = (fftwf_complex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));

	// initialize estimation
	maxvaluecpu(h_StackA, h_StackA, (float)(SMALLVALUE), totalSize);
	maxvaluecpu(h_StackB, h_StackB, (float)(SMALLVALUE), totalSize);
	// initialize estimation
	if (flagConstInitial) { // use constant mean value as initial
		float meanValue1 = (float)sumcpu(h_StackA, totalSize);
		float meanValue2 = (float)sumcpu(h_StackB, totalSize);
		memset(h_StackE, 0, totalSize * sizeof(float));
		addvaluecpu(h_StackE, h_StackE, (meanValue1 + meanValue2) / 2, totalSize);
	}
	else { // use measured images as initial
		addcpu(h_StackE, h_StackA, h_StackB, totalSize);
		multivaluecpu(h_StackE, h_StackE, (float)0.5, totalSize);
	}

	fftwf_plan stackE2Spectrum = fftwf_plan_dft_r2c_3d(sx, sy, sz, h_StackE, h_StackESpectrum, FFTW_MEASURE);
	fftwf_plan stackT2Spectrum = fftwf_plan_dft_r2c_3d(sx, sy, sz, h_StackT, h_StackESpectrum, FFTW_MEASURE);
	fftwf_plan spectrum2StackT = fftwf_plan_dft_c2r_3d(sx, sy, sz, h_StackESpectrum, h_StackT, FFTW_MEASURE);
	printf("...Start CPU Decon\n");
	for (int itNum = 1; itNum <= itNumForDecon; itNum++) {
		fftwf_execute(stackE2Spectrum);
		multicomplexcpu((fComplex *)h_StackESpectrum, (fComplex *)h_StackESpectrum, (fComplex *)h_OTF1, sx * sy * (sz / 2 + 1));
		fftwf_execute(spectrum2StackT);
		//printf("here!\n");

		divcpu(h_StackT, h_StackA, h_StackT, totalSize);

		fftwf_execute(stackT2Spectrum);
		multicomplexcpu((fComplex *)h_StackESpectrum, (fComplex *)h_StackESpectrum, (fComplex *)h_OTF_bp1, sx * sy * (sz / 2 + 1));
		fftwf_execute(spectrum2StackT);
		multicpu(h_StackE, h_StackE, h_StackT, totalSize);//
		return 0;
		fftwf_execute(stackE2Spectrum);
		multicomplexcpu((fComplex *)h_StackESpectrum, (fComplex *)h_StackESpectrum, (fComplex *)h_OTF2, sx * sy * (sz / 2 + 1));
		fftwf_execute(spectrum2StackT);

		divcpu(h_StackT, h_StackB, h_StackT, totalSize);

		fftwf_execute(stackT2Spectrum);
		multicomplexcpu((fComplex *)h_StackESpectrum, (fComplex *)h_StackESpectrum, (fComplex *)h_OTF_bp2, sx * sy * (sz / 2 + 1));
		fftwf_execute(spectrum2StackT);
		multicpu(h_StackE, h_StackE, h_StackT, totalSize);//
	}
	free(h_StackT);
	free(h_StackESpectrum);
	fftwf_destroy_plan(stackE2Spectrum);
	fftwf_destroy_plan(stackT2Spectrum);
	fftwf_destroy_plan(spectrum2StackT);
	end = clock();
	printf("...Time cost for decon is %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;
}
extern "C"
int decon_dualview_OTF1(float *d_decon, float *d_img1, float *d_img2, fComplex *d_OTF1, fComplex *d_OTF2, fComplex *d_OTF_bp1,
	fComplex *d_OTF_bp2, long long int sx, long long int sy, long long int sz, int itNumForDecon, bool flagConstInitial) {
	// **** dual-view deconvolution with OTF interface when GPU memory is sufficient: 9 images + 2 fftPlans ***
	// image size
	long long int totalSize = sx*sy*sz; // in floating format
	long long int totalSizeSpectrum = sx * sy*(sz / 2 + 1); // in complex floating format
	size_t freeMem = 0, totalMem = 0;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	clock_t start, end;
	start = clock();
	float *d_StackA = d_img1, *d_StackB = d_img2, *d_StackE = d_decon;
	float *d_StackT = NULL;
	fComplex *d_StackESpectrum = NULL;
	cudaMalloc((void **)&d_StackT, totalSize * sizeof(float));
	cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum * sizeof(fComplex));

	// initialize estimation
	maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), sx, sy, sz);
	maxvalue3Dgpu(d_StackB, d_StackB, (float)(SMALLVALUE), sx, sy, sz);
	// initialize estimation
	if (flagConstInitial) { // use constant mean value as initial
		float meanValue1 = (float)sum3Dgpu(d_StackA, sx, sy, sz);
		float meanValue2 = (float)sum3Dgpu(d_StackB, sx, sy, sz);
		cudaMemset(d_StackE, 0, totalSize * sizeof(float));
		addvaluegpu(d_StackE, d_StackE, (meanValue1 + meanValue2) / 2, sx, sy, sz);
	}
	else { // use measured images as initial
		add3Dgpu(d_StackE, d_StackA, d_StackB, sx, sy, sz);
		multivaluegpu(d_StackE, d_StackE, (float)0.5, sx, sy, sz); 
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: initial image preparation failed \n");
		exit(1);
	}
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, sx, sy, sz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, sx, sy, sz, CUFFT_C2R);
	cudaMemGetInfo(&freeMem, &totalMem);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: cufftPlan error \n");
		exit(1);
	}
	printf("...GPU free memory (before decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	for (int itNum = 1; itNum <= itNumForDecon; itNum++) {
		// ### 1st view
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF1, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
		div3Dgpu(d_StackT, d_StackA, d_StackT, sx, sy, sz);   
															
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF_bp1, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
																						  
		multi3Dgpu(d_StackE, d_StackE, d_StackT, sx, sy, sz);//
		maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), sx, sy, sz);

		// ### 2nd view
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF2, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
		div3Dgpu(d_StackT, d_StackB, d_StackT, sx, sy, sz);//
																
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF_bp2, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);

		multi3Dgpu(d_StackE, d_StackE, d_StackT, sx, sy, sz);
		maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), sx, sy, sz);
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: decon iterration error \n");
		exit(1);
	}
	cudaFree(d_StackT); cudaFree(d_StackESpectrum);
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory (after decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	end = clock();
	printf("...Time cost for decon is %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}
extern "C"
int decon_dualview_OTF2(float *d_decon, float *d_img1, float *h_img2, fComplex *h_OTF1, fComplex *h_OTF2, fComplex *h_OTF_bp1,
	fComplex *h_OTF_bp2, long long int sx, long long int sy, long long int sz, int itNumForDecon, bool flagConstInitial) {
	// **** dual-view deconvolution with OTF interface when GPU memory is insufficient: 2 images + 2 fftPlans ***
	// **** d_decon and d_img should have total size: sx * sy*(sz / 2 + 1) * sizeof(float) to store image spectrum
	// image size
	long long int totalSize = sx*sy*sz; // in floating format
	long long int totalSizeSpectrum = sx * sy*(sz / 2 + 1); // in complex floating format
															// *****
	size_t freeMem = 0, totalMem = 0;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	clock_t start, end;
	start = clock();
	float *h_StackA = NULL, *h_StackB = NULL, *h_StackE = NULL;
	h_StackA = (float *)malloc(totalSize * sizeof(float));
	h_StackB = (float *)malloc(totalSize * sizeof(float));
	h_StackE = (float *)malloc(totalSize * sizeof(float));

	float *d_StackA = d_img1, *d_StackE = d_decon;
	float *d_StackB = NULL;
	fComplex  *d_OTF = NULL, *d_StackESpectrum = NULL;
	d_StackESpectrum = (fComplex *)d_StackA;
	d_OTF = (fComplex *)d_StackE;
	cudaStatus = cudaGetLastError();

	// initialize estimation
	cudaMalloc((void **)&d_StackB, totalSize * sizeof(float));
	maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), sx, sy, sz);
	cudaMemcpy(d_StackB, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	maxvalue3Dgpu(d_StackB, d_StackB, (float)(SMALLVALUE), sx, sy, sz);
	cudaMemcpy(h_StackA, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_StackB, d_StackB, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (flagConstInitial) { // use constant mean value as initial
		float meanValue1 = (float)sum3Dgpu(d_StackA, sx, sy, sz);
		float meanValue2 = (float)sum3Dgpu(d_StackB, sx, sy, sz);
		cudaMemset(d_StackE, 0, totalSize * sizeof(float));
		addvaluegpu(d_StackE, d_StackE, (meanValue1 + meanValue2) / 2, sx, sy, sz);
	}
	else { // use measured images as initial
		add3Dgpu(d_StackE, d_StackA, d_StackB, sx, sy, sz);
		multivaluegpu(d_StackE, d_StackE, (float)0.5, sx, sy, sz);
	}
	cudaMemcpy(h_StackE, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_StackB); // release temperary variable
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: initial image preparation failed \n");
		exit(1);
	}
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, sx, sy, sz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, sx, sy, sz, CUFFT_C2R);
	cudaMemGetInfo(&freeMem, &totalMem);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: cufftPlan error \n");
		exit(1);
	}
	printf("...GPU free memory (before decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	for (int itNum = 1; itNum <= itNumForDecon; itNum++) {
		//printf("...Processing iteration %d\n", it);
		// ### 1st view
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_StackE, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

		cudaMemcpy(d_OTF, h_OTF1, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);

		cudaMemcpy(d_StackA, h_StackA, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		div3Dgpu(d_StackE, d_StackA, d_StackE, sx, sy, sz);   
															
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(d_OTF, h_OTF_bp1, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);
																				
		cudaMemcpy(d_StackA, h_StackE, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		multi3Dgpu(d_StackE, d_StackE, d_StackA, sx, sy, sz);//
		maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), sx, sy, sz);

		// ### 2nd view	
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
		cudaMemcpy(h_StackE, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

		cudaMemcpy(d_OTF, h_OTF2, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);
	
		cudaMemcpy(d_StackA, h_StackB, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		div3Dgpu(d_StackE, d_StackA, d_StackE, sx, sy, sz);//

		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(d_OTF, h_OTF_bp2, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyHostToDevice);
		multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_OTF, sx, sy, (sz / 2 + 1));
		cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);

		cudaMemcpy(d_StackA, h_StackE, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		multi3Dgpu(d_StackE, d_StackE, d_StackA, sx, sy, sz);
		maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), sx, sy, sz);
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "*** FAILED - ABORTING: decon iterration error \n");
		exit(1);
	}
	free(h_StackA); free(h_StackB); free(h_StackE);
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory (after decon iteration) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	end = clock();
	printf("...Time cost for decon is %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}

#undef SMALLVALUE
#undef NDIM