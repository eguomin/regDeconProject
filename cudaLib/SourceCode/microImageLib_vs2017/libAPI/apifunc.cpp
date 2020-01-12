#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>   
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <ctime>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
//#else
//#include <sys/stat.h>
#endif

// Includes CUDA
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>
//#include <cufftw.h> // ** cuFFT also comes with CPU-version FFTW, but seems not to work when image size is large.
#include "fftw3.h"
#include <memory>
#include "device_launch_parameters.h"

#include "tiff.h"
#include "tiffio.h"

extern "C"{
#include "powell.h"
}

#include "apifunc.h"
#include "apifunc_internal.h"
#define blockSize 1024
#define blockSize2Dx 32
#define blockSize2Dy 32
#define blockSize3Dx 16
#define blockSize3Dy 8
#define blockSize3Dz 8
#define NDIM 12
#define SMALLVALUE 0.01


cudaError_t __err;
#define cudaCheckErrors(msg) \
    do { \
        __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
								        } \
				    } while (0)
//concatenate any number of strings
char* concat(int count, ...)
{
	va_list ap;
	int i;

	// Find required length to store merged string
	int len = 1; // room for NULL
	va_start(ap, count);
	for (i = 0; i<count; i++)
		len += strlen(va_arg(ap, char*));
	va_end(ap);

	// Allocate memory to concat strings
	char *merged = (char*)calloc(sizeof(char), len);
	int null_pos = 0;

	// Actually concatenate strings
	va_start(ap, count);
	for (i = 0; i<count; i++)
	{
		char *s = va_arg(ap, char*);
		strcpy(merged + null_pos, s);
		null_pos += strlen(s);
	}
	va_end(ap);

	return merged;
}

//check file exists or not
bool fexists(const char * filename){
	if (FILE * file = fopen(filename, "r")) {
		fclose(file);
		return true;
	}
	return false;
}

#ifdef _WIN32
int findSubFolders(char *subFolderNames, char *pathIn)
{
	TCHAR szDir[MAX_PATH];
	StringCchCopy(szDir, MAX_PATH, pathIn);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));
	WIN32_FIND_DATA fd;

	HANDLE hFile = FindFirstFile((char*)szDir, &fd);
	int i = 0;
	if (hFile != INVALID_HANDLE_VALUE){
		do
		{
			if ((*(char*)fd.cFileName == '.') || (*(char*)fd.cFileName == '..'))
				continue;
			strcpy(&subFolderNames[i*MAX_PATH], fd.cFileName);
			i++;
		} while (FindNextFile(hFile, &fd));
	}
		
	return i;
}

#endif

unsigned short gettifinfo(char tifdir[], unsigned int *tifSize){
	if (!fexists(tifdir)) {
		fprintf(stderr, "*** File does not exist: %s\n", tifdir);
		exit(1);
	}
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &tifSize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &tifSize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);
	/// Get slice number
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	tifSize[2] = nSlice;
	(void)TIFFClose(tif);
	return bitPerSample;
}

// Read and write tiff image

void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize){
	// check if file exists
	if (!fexists(tifdir)) {
		fprintf(stderr, "***Failed to read image!!! File does not exist: %s\n", tifdir);
		exit(1);
	}

	// get TIFF image information
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	/// Get slice number
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	imsize[2] = nSlice;
	(void)TIFFClose(tif);

	tif = TIFFOpen(tifdir, "r");
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imsize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imsize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);        // uint16 bps;

	uint32 nByte = bitPerSample / 8;
	if (bitPerSample == 16){
		uint16 *buf = (uint16 *)_TIFFmalloc(imsize[0] * imsize[1] * imsize[2] * nByte);
		if (tif){
			uint32 n = 0; // slice number
			do{
				for (uint32 row = 0; row < imsize[1]; row++){
					TIFFReadScanline(tif, &buf[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
				}
				n++;
			} while (TIFFReadDirectory(tif));
		}
		(void)TIFFClose(tif);

		for (uint32 i = 0; i < imsize[0] * imsize[1] * imsize[2]; i++){
			h_Image[i] = (float)buf[i];
		}
		_TIFFfree(buf);
	}
	else if (bitPerSample == 32){
		if (tif){
			uint32 n = 0; // slice number
			do{
				for (uint32 row = 0; row < imsize[1]; row++){// possible to read in floating 32bit tiff images
					TIFFReadScanline(tif, &h_Image[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
				}
				n++;
			} while (TIFFReadDirectory(tif));
		}
		(void)TIFFClose(tif);
	}
}

void readtifstack_16to16(unsigned short *h_Image, char tifdir[], unsigned int *imsize){
	// check if file exists
	if (!fexists(tifdir)) {
		fprintf(stderr, "***Failed to read image!!! File does not exist: %s\n", tifdir);
		exit(1);
	}

	// get TIFF image information
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	/// Get slice number
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	imsize[2] = nSlice;
	(void)TIFFClose(tif);

	tif = TIFFOpen(tifdir, "r");
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imsize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imsize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);        // uint16 bps;

	if (bitPerSample == 16){
		if (tif){
			uint32 n = 0; // slice number
			do{
				for (uint32 row = 0; row < imsize[1]; row++){
					TIFFReadScanline(tif, &h_Image[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
				}
				n++;
			} while (TIFFReadDirectory(tif));
		}
		(void)TIFFClose(tif);

		
	}
	else
		printf("Image bit per sample is not supported, please set input image as 16 bit!!!\n\n");
}

// Write tiff image
void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample) {
	int imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(bitPerSample / 8);

	if (bitPerSample == 16) {
		uint16 *buf = (uint16 *)_TIFFmalloc(imTotalSize * sizeof(uint16));
		for (int i = 0; i < imTotalSize; i++) {
			buf[i] = (uint16)h_Image[i];
		}

		TIFF *tif = TIFFOpen(tifdir, "w");
		for (uint32 n = 0; n < imsize[2]; n++) {
			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
			TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
			TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
			TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			//   Some other essential fields to set that you do not have to understand for now.
			TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
			TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFWriteEncodedStrip(tif, 0, &buf[imxy * n], imxy * nByte);
			TIFFWriteDirectory(tif);
		}
		(void)TIFFClose(tif);
		_TIFFfree(buf);
	}
	else if (bitPerSample == 32) {
		TIFF *tif = TIFFOpen(tifdir, "w");
		for (uint32 n = 0; n < imsize[2]; n++) {

			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
			TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);    // ***set each pixel as floating point data ****
			TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
			TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
			TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			//   Some other essential fields to set that you do not have to understand for now.
			TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
			TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

			TIFFWriteEncodedStrip(tif, 0, &h_Image[imxy * n], imxy * nByte);
			TIFFWriteDirectory(tif);
		}
		(void)TIFFClose(tif);
	}
	else
		printf("Image bit per sample is not supported, please set bitPerPample to 16 or 32 !!!\n\n");
}

/*
void writetifstack_16to16(char tifdir[], unsigned short *h_Image, unsigned int *imsize) {
	int imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(16 / 8);
	TIFF *tif = TIFFOpen(tifdir, "w");
	for (uint32 n = 0; n < imsize[2]; n++) {
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);    // set the size of the channels
		TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
		TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		//   Some other essential fields to set that you do not have to understand for now.
		TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFWriteEncodedStrip(tif, 0, &h_Image[imxy * n], imxy * nByte);
		TIFFWriteDirectory(tif);
	}
	(void)TIFFClose(tif);
}
*/

/*
template <class T>
void readtifstack(T *h_Image, char tifdir[], unsigned int *imsize){
	// check if file exists
	if (!fexists(tifdir)){
		fprintf(stderr, "***Failed to read image!!! File does not exist: %s\n", tifdir);
		exit(1);
	}

	// get TIFF image information
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	imsize[2] = nSlice;
	(void)TIFFClose(tif);
	tif = TIFFOpen(tifdir, "r");
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imsize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imsize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);        // uint16 bps;
	uint32 nByte = bitPerSample / 8;

	// read image
	if (bitPerSample == 16){
		if (sizeof(T) == 2){ // read 16-bit TIFF to unsigned short format
			if (tif){
				uint32 n = 0; 
				do{
					for (uint32 row = 0; row < imsize[1]; row++){
						TIFFReadScanline(tif, &h_Image[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
					}
					n++;
				} while (TIFFReadDirectory(tif));
			}
		}
		else if (sizeof(T) == 4){// read 16-bit TIFF and convert to floating format
			uint16 *buf = (uint16 *)_TIFFmalloc(imsize[0] * imsize[1] * imsize[2] * nByte);
			if (tif){
				uint32 n = 0; 
				do{
					for (uint32 row = 0; row < imsize[1]; row++){
						TIFFReadScanline(tif, &buf[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
					}
					n++;
				} while (TIFFReadDirectory(tif));
			}
			for (uint32 i = 0; i < imsize[0] * imsize[1] * imsize[2]; i++){
				h_Image[i] = (float)buf[i];
			}
			_TIFFfree(buf);
		}
	}
	else if (bitPerSample == 32){
		if (sizeof(T) == 2){
			fprintf(stderr, "*** Failed to read image: input image format is not supported, please use 16-bit TIFF image!!!\n\n");
			exit(1);
		}
		else if (sizeof(T) == 4){
			if (tif){
				uint32 n = 0; // slice number
				do{
					for (uint32 row = 0; row < imsize[1]; row++){// possible to read in floating 32bit tiff images
						TIFFReadScanline(tif, &h_Image[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
					}
					n++;
				} while (TIFFReadDirectory(tif));
			}
		}
	}
	else {
		fprintf(stderr, "*** Failed to read image: input image format is not supported, please use 16-bit or 32-bit TIFF image!!!\n\n");
		exit(1);
	}
	(void)TIFFClose(tif);
}
template void readtifstack<unsigned short>(unsigned short *h_Image, char tifdir[], unsigned int *imsize);
template void readtifstack<float>(float *h_Image, char tifdir[], unsigned int *imsize);

template <class T>
void writetifstack(char tifdir[], T *h_Image, unsigned int *imsize, unsigned short bitPerSample){
	uint32 imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(bitPerSample / 8);

	if (bitPerSample == 16) {
		if (sizeof(T) == 2) {
			TIFF *tif = TIFFOpen(tifdir, "w");
			for (uint32 n = 0; n < imsize[2]; n++) {
				TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
				TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
				TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
				TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
				TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
				TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
				TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
				//   Some other essential fields to set that you do not have to understand for now.
				TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
				TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
				TIFFWriteEncodedStrip(tif, 0, &h_Image[imxy * n], imxy * nByte);
				TIFFWriteDirectory(tif);
			}
			(void)TIFFClose(tif);
		}
		else if (sizeof(T) == 4) {
			uint16 *buf = (uint16 *)malloc(imTotalSize * sizeof(uint16));
			for (uint32 i = 0; i < imTotalSize; i++) {
				buf[i] = (uint16)h_Image[i];
			}
			TIFF *tif = TIFFOpen(tifdir, "w");
			for (uint32 n = 0; n < imsize[2]; n++) {
				TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
				TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
				TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
				TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
				TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
				TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
				TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
				//   Some other essential fields to set that you do not have to understand for now.
				TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
				TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
				TIFFWriteEncodedStrip(tif, 0, &buf[imxy * n], imxy * nByte);
				TIFFWriteDirectory(tif);
			}
			(void)TIFFClose(tif);
			free(buf);
		}
		else {
			fprintf(stderr, "*** Failed to write image: only unsigned short or floating format input data is allowed for writing 32-bit TIFF image!!!\n\n");
			exit(1);
		}
	}
	else if (bitPerSample == 32) {
		if (sizeof(T) == 2) {
			float *buf = (float *)malloc(imTotalSize * sizeof(float));
			for (uint32 i = 0; i < imTotalSize; i++) {
				buf[i] = (float)h_Image[i];
			}
			TIFF *tif = TIFFOpen(tifdir, "w");
			for (uint32 n = 0; n < imsize[2]; n++) {
				TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
				TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
				TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
				TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
				TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
				TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
				TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
				//   Some other essential fields to set that you do not have to understand for now.
				TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
				TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
				TIFFWriteEncodedStrip(tif, 0, &buf[imxy * n], imxy * nByte);
				TIFFWriteDirectory(tif);
			}
			(void)TIFFClose(tif);
			free(buf);
		}
		else if (sizeof(T) == 4) {
			TIFF *tif = TIFFOpen(tifdir, "w");
			for (uint32 n = 0; n < imsize[2]; n++) {
				TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
				TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
				TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
				TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
				TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
				TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
				TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
				//   Some other essential fields to set that you do not have to understand for now.
				TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
				TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
				TIFFWriteEncodedStrip(tif, 0, &h_Image[imxy * n], imxy * nByte);
				TIFFWriteDirectory(tif);
			}
			(void)TIFFClose(tif);
		}
		else {
			fprintf(stderr, "*** Failed to write image: only unsigned short or floating format input data is allowed for writing 32-bit TIFF image!!!\n\n");
			exit(1);
		}
	}
	else {
		fprintf(stderr, "*** Failed to write image: output image format is not supported, please use 16-bit or 32-bit TIFF image!!!\n\n");
		exit(1);
	}
}
template void writetifstack<unsigned short>(char tifdir[], unsigned short *h_Image, unsigned int *imsize, unsigned short bitPerSample);
template void writetifstack<float>(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample);
*/

void queryDevice(){
	printf(" \n ===========================================\n");
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	//printf("Reseting GPU devices....\n");
	//cudaDeviceReset();
	//printf("...Reseting Done!!!\n");
	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
			(float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);

		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
	}

	printf(" ===========================================\n\n");

}

///// affine transformation
int atrans3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum){
	//image size
	long long int sx1 = imSize1[0], sy1 = imSize1[1], sz1 = imSize1[2];
	long long int sx2 = imSize2[0], sy2 = imSize2[1], sz2 = imSize2[2];
	// total pixel count for each images
	long long int totalSize1 = sx1*sy1*sz1;
	long long int totalSize2 = sx2*sy2*sz2;
	// GPU device
	cudaSetDevice(deviceNum);
	float *d_img3DTemp;
	cudaMalloc((void **)&d_img3DTemp, totalSize1 *sizeof(float));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_ArrayTemp;
	cudaMalloc3DArray(&d_ArrayTemp, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaThreadSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudaMemset(d_img3DTemp, 0, totalSize1*sizeof(float));
	cudacopyhosttoarray(d_ArrayTemp, channelDesc, h_img2, sx2, sy2, sz2);
	BindTexture(d_ArrayTemp, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_img3DTemp, sx1, sy1, sz1, sx2, sy2, sz2);
	UnbindTexture();
	cudaMemcpy(h_reg, d_img3DTemp, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFreeArray(d_ArrayTemp);
	cudaFree(d_img3DTemp);
	return 0;
}

int atrans3dgpu_16bit(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum){
	//image size
	long long int sx1 = imSize1[0], sy1 = imSize1[1], sz1 = imSize1[2];
	long long int sx2 = imSize2[0], sy2 = imSize2[1], sz2 = imSize2[2];
	// total pixel count for each images
	long long int totalSize1 = sx1*sy1*sz1;
	long long int totalSize2 = sx2*sy2*sz2;
	// GPU device
	unsigned short *d_img3D16;
	cudaSetDevice(deviceNum);
	cudaMalloc((void **)&d_img3D16, totalSize1 *sizeof(unsigned short));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaThreadSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudaMemset(d_img3D16, 0, totalSize1*sizeof(unsigned short));
	cudacopyhosttoarray(d_Array, channelDesc, h_img2, sx2, sy2, sz2);
	BindTexture16(d_Array, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_img3D16, sx1, sy1, sz1, sx2, sy2, sz2);
	UnbindTexture16();
	cudaMemcpy(h_reg, d_img3D16, totalSize1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaFreeArray(d_Array);
	cudaFree(d_img3D16);
	return 0;
}

///// 2D registration
int reg2d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records) {
	// **** 3D image registration: capable with phasor registraton and affine registration  ***
	/*
	*** registration choice: regChoice
	0: no phasor or affine registration; if flagTmx is true, transform h_img2 based on input matrix;
	1: phasor registraion (pixel-level translation only);
	2: affine registration (with or without input matrix);
	*
	*** flagTmx: only if regChoice == 0, 2
	true: use iTmx as input matrix;
	false: default;
	*
	*** gpuMemMode
	0: All on CPU. // need to add this option in the future
	1: sufficient GPU memory;
	*
	*** records: 11 element array
	[0]: actual gpu memory mode
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB), if use gpu
	*/

	// ************get basic input images information ******************	
	//image size
	long long int imx1, imy1, imx2, imy2;
	imx1 = imSize1[0]; imy1 = imSize1[1]; 
	imx2 = imSize2[0]; imy2 = imSize2[1];
	// total pixel count for each image
	long long int totalSize1 = imx1*imy1;
	long long int totalSize2 = imx2*imy2;
	long long int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;

	// ****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode == 1 ) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		records[8] = (float)freeMem / 1048576.0f;
		if (verbose) {
			printf("...GPU free memory before registration is %.0f MB\n", (float)freeMem / 1048576.0f);
		}
	}
	records[0] = gpuMemMode;
	int affMethod = 1;
	long long int shiftXY[2];
	float *d_imgT = NULL, *d_imgS = NULL;
	switch (gpuMemMode) {
	case 0:
		switch (regChoice) {
		case 0:
			break;
		case 1:
			break;
		case 2:
			break;
		default:
			printf("\n ****Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		printf("\n **** 2D CPU registration is currently not supported !!! **** \n");
		break;
	case 1:
		cudaMemGetInfo(&freeMem, &totalMem);
		records[9] = (float)freeMem / 1048576.0f;
		switch (regChoice) {
		case 0:
			if (flagTmx) {
				affMethod = 0;
				(void)reg2d_affine1(h_reg, iTmx, h_img1, h_img2, imx1, imy1, imx2, imy2, affMethod, flagTmx, FTOL, itLimit, records);
			}
			break;
		case 1:
			if ((imx1 != imx2) || (imy1 != imy2)) {
				printf("\n ****Image size of the 2D images is not matched, processing stop !!! **** \n");
				return 1;
			}
			cudaMalloc((void **)&d_imgT, totalSize1 * sizeof(float));
			cudaMalloc((void **)&d_imgS, totalSize1 * sizeof(float));
			cudaMemcpy(d_imgT, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_imgS, h_img2, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			(void)reg2d_phasor1(&shiftXY[0], d_imgT, d_imgS, imx1, imy1);
			imshiftgpu(d_imgT, d_imgS, imx1, imy1, 1, -shiftXY[0], -shiftXY[1], 0);
			cudaMemcpy(h_reg, d_imgT, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			iTmx[0] = 1; iTmx[1] = 0; iTmx[2] = shiftXY[0];
			iTmx[3] = 0; iTmx[4] = 1; iTmx[5] = shiftXY[1];
			break;
		case 2:
			(void)reg2d_affine1(h_reg, iTmx, h_img1, h_img2, imx1, imy1, imx2, imy2, affMethod, flagTmx, FTOL, itLimit, records);
			break;
		default:
			printf("\n ****Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return 1;
	}
	//
	if (gpuMemMode == 1) {
		cudaMemGetInfo(&freeMem, &totalMem);
		records[10] = (float)freeMem / 1048576.0f;
	}
	end = clock();
	records[7] = (float)(end - start) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("Total time cost for whole processing is %2.3f s\n", records[7]);
	}
	return 0;
}

///// 3D registration
bool checkmatrix(float *m, long long int sx, long long int sy, long long int sz) {
	// check if affine matrix is reasonable
	bool mMatrix = true;
	float scaleLow = 0.5, scaleUp = 1.4, scaleSumLow = 2, scaleSumUp = 4, shiftRatio = 0.8;
	if (m[0]<scaleLow || m[0]>scaleUp || m[5]<scaleLow || m[5]>scaleUp || m[10]<scaleLow || m[10]>scaleUp) {
		mMatrix = false;
	}
	if ((m[0] + m[5] + m[10]) < scaleSumLow || (m[0] + m[5] + m[10]) > scaleSumUp) {
		mMatrix = false;
	}
	if (abs(m[3])>shiftRatio * sx || abs(m[7])>shiftRatio * sy || abs(m[11])>shiftRatio * sz) {
		mMatrix = false;
	}
	// ... more checking
	return mMatrix;
}

int reg3d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice, int affMethod,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records) {
	// **** 3D image registration: capable with phasor registraton and affine registration  ***
	/*
	*** registration choice: regChoice
	0: no phasor or affine registration; if flagTmx is true, transform h_img2 based on input matrix;
	1: phasor registraion (pixel-level translation only);
	2: affine registration (with or without input matrix);
	3: phasor registration --> affine registration (input matrix disabled);
	4: 2D MIP registration --> affine registration (input matrix disabled);
	*
	*** affine registration method: only if regChoice == 2, 3, 4
	0: no registration; if inputTmx is true, transform d_img2 based on input matrix;
	1: translation only;
	2: rigid body;
	3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)
	4: 9 degrees of freedom(translation, rotation, scaling);
	5: 12 degrees of freedom;
	6: rigid body first, then do 12 degrees of freedom;
	7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*
	*** flagTmx: only if regChoice == 0, 2
	true: use iTmx as input matrix;
	false: default;
	*
	*** gpuMemMode  
	-1: Automatically set memory mode; 
	0: All on CPU. // need to add this option in the future
	1: sufficient GPU memory; 
	2: GPU memory optimized; 
	*
	*** records: 11 element array
	[0]: actual gpu memory mode
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB), if use gpu
	*/
	// ************get basic input images information ******************	
	//image size
	long long int imx1, imy1, imz1, imx2, imy2, imz2;
	imx1 = imSize1[0]; imy1 = imSize1[1]; imz1 = imSize1[2];
	imx2 = imSize2[0]; imy2 = imSize2[1]; imz2 = imSize2[2];
	// total pixel count for each image
	long long int totalSize1 = imx1*imy1*imz1;
	long long int totalSize2 = imx2*imy2*imz2;
	long long int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;

	// ****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		records[8] = (float)freeMem / 1048576.0f;
		if (verbose) {
			printf("...GPU free memory before registration is %.0f MB\n", (float)freeMem / 1048576.0f);
		}	
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
		cudaMemGetInfo(&freeMem, &totalMem);
		if ((regChoice == 0)||(regChoice == 2)|| (regChoice == 4)) {
			if (freeMem > (4 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 1;
				if (verbose) {
					printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
				}
			}
			else if (freeMem > (2 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 2;
				if (verbose) {
					printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
				}
			}
			else { // all processing in CPU
				gpuMemMode = 0;
				if (verbose) {
					printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
				}
			}
		}
		else if ((regChoice == 1) || (regChoice == 3)) {
			if (freeMem > (5 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 1;
				if (verbose) {
					printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
				}
			}
			else if (freeMem > (4 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 2;
				if (verbose) {
					printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
				}
			}
			else { // all processing in CPU
				gpuMemMode = 0;
				if (verbose) {
					printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
				}
			}
		}
	}
	records[0] = gpuMemMode;

	float *h_imgT = NULL, *h_imgS = NULL;
	float *d_imgT = NULL, *d_imgS = NULL, *d_reg = NULL;
	long long int shiftXYZ[3], shiftXY[2], shiftZX[2];
	float *d_imgTemp1 = NULL, *d_imgTemp2 = NULL;
	float *d_imgTemp3 = NULL, *d_imgTemp4 = NULL;
	unsigned int im2DSize[2];
	long long int totalSize2DMax;
	switch (gpuMemMode) {
	case 0: // CPU calculation
		printf("\n ****CPU registraion function is under developing **** \n");
		return -1;
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		//cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
		cudaMalloc((void **)&d_imgT, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_imgS, totalSize1 * sizeof(float));
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");
		if ((imx1 == imx2) && (imy1 == imy2) && (imz1 == imz2))
			cudaMemcpy(d_imgS, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
		else {
			cudaMemcpy(d_imgT, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			alignsize3Dgpu(d_imgS, d_imgT, imz1, imy1, imx1, imz2, imy2, imx2);
		}
		cudaMemcpy(d_imgT, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemGetInfo(&freeMem, &totalMem);
		records[9] = (float)freeMem / 1048576.0f;
		switch (regChoice) {
		case 0:
			affMethod = 0;
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 1:
			(void)reg3d_phasor1(&shiftXYZ[0], d_imgT, d_imgS, imx1, imy1, imz1);
			imshiftgpu(d_imgT, d_imgS, imx1, imy1, imz1, -shiftXYZ[0], -shiftXYZ[1], -shiftXYZ[2]);
			cudaMemcpy(h_reg, d_imgT, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0]; 
			iTmx[7] = shiftXYZ[1]; 
			iTmx[11] = shiftXYZ[2];
			break;
		case 2:
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 3:
			(void)reg3d_phasor1(&shiftXYZ[0], d_imgT, d_imgS, imx1, imy1, imz1);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0];
			iTmx[7] = shiftXYZ[1];
			iTmx[11] = shiftXYZ[2];
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			flagTmx = true;
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 4:
			printf("\n*** 2D MIP registration ... \n");
			totalSize2DMax = ((imx1 * imy1) > (imz1 * imx1)) ? (imx1 * imy1) : (imz1 * imx1);
			cudaMalloc((void **)&d_imgTemp3, totalSize2DMax * sizeof(float));
			cudaMalloc((void **)&d_imgTemp4, totalSize2DMax * sizeof(float));
			maxprojection(d_imgTemp3, d_imgT, imx1, imy1, imz1, 1);
			maxprojection(d_imgTemp4, d_imgS, imx1, imy1, imz1, 1);
			(void)reg2d_phasor1(&shiftXY[0], d_imgTemp3, d_imgTemp4, imx1, imy1);
			maxprojection(d_imgTemp3, d_imgT, imx1, imy1, imz1, 2);
			maxprojection(d_imgTemp4, d_imgS, imx1, imy1, imz1, 2);
			(void)reg2d_phasor1(&shiftZX[0], d_imgTemp3, d_imgTemp4, imz1, imx1);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = (shiftXY[0]+shiftZX[1])/2;
			iTmx[7] = shiftXY[1];
			iTmx[11] = shiftZX[0];
			cudaFree(d_imgTemp3); cudaFree(d_imgTemp4);
			printf("   ... 2D MIP registration completed. \n");
			printf("\n*** 3D registration ... \n");
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			flagTmx = true;
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		default:
			printf("\n*** Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		cudaFree(d_imgT);
		cudaFree(d_imgS);
		time2 = clock();
		break;
	case 2:
		time1 = clock();
		// allocate memory
		h_imgS = (float *)malloc(totalSize1 * sizeof(float));
		if ((imx1 == imx2) && (imy1 == imy2) && (imz1 == imz2))
			memcpy(h_imgS, h_img2, totalSize1 * sizeof(float));
		else {
			cudaMalloc((void **)&d_imgS, totalSize2 * sizeof(float));
			cudaMalloc((void **)&d_imgT, totalSize1 * sizeof(float));
			cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");
			cudaMemcpy(d_imgS, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			alignsize3Dgpu(d_imgT, d_imgS, imz1, imy1, imx1, imz2, imy2, imx2);
			cudaMemcpy(h_imgS, d_imgT, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_imgS);
			cudaFree(d_imgT);
		}
		cudaMemGetInfo(&freeMem, &totalMem);
		records[9] = (float)freeMem / 1048576.0f;
		switch (regChoice) {
		case 0:
			affMethod = 0;
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine2(d_reg, iTmx, h_img1, h_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 1:
			(void)reg3d_phasor2(&shiftXYZ[0], h_img1, h_imgS, imx1, imy1, imz1);
			cudaMalloc((void **)&d_imgTemp1, totalSize1 * sizeof(float));
			cudaMalloc((void **)&d_imgTemp2, totalSize1 * sizeof(float));
			cudaMemcpy(d_imgTemp1, h_imgS, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			imshiftgpu(d_imgTemp2, d_imgTemp1, imx1, imy1, imz1, -shiftXYZ[0], -shiftXYZ[1], -shiftXYZ[2]);
			cudaMemcpy(h_reg, d_imgTemp2, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0];
			iTmx[7] = shiftXYZ[1];
			iTmx[11] = shiftXYZ[2];
			cudaFree(d_imgTemp1);
			cudaFree(d_imgTemp2);
			break;
		case 2:
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine2(d_reg, iTmx, h_img1, h_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 3:
			(void)reg3d_phasor2(&shiftXYZ[0], h_img1, h_imgS, imx1, imy1, imz1);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0];
			iTmx[7] = shiftXYZ[1];
			iTmx[11] = shiftXYZ[2];
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			flagTmx = true;
			(void)reg3d_affine2(d_reg, iTmx, h_img1, h_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 4:
			printf("\n ****2D MIP registration --> affine registraion function is under developing **** \n");
			return -1;
			break;
		default:
			printf("\n ****Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		free(h_imgS);
		time2 = clock();
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return 1;
	}
	//
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		records[10] = (float)freeMem / 1048576.0f;
	}
	end = clock();
	records[7] = (float)(end - start) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("Total time cost for whole processing is %2.3f s\n", records[7]);
	}
	return 0;
}

int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int affMethod,
	int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords) {
	// **** 3D GPU affine registration: based on function reg3d  ***
	/*
	*** affine registration method
	0: no registration; if inputTmx is true, transform d_img2 based on input matrix;
	1: translation only;
	2: rigid body;
	3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)
	4: 9 degrees of freedom(translation, rotation, scaling);
	5: 12 degrees of freedom;
	6: rigid body first, then do 12 degrees of freedom;
	7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*
	*** flagTmx
	true: use iTmx as input matrix;
	false: default;

	*** regRecords: 11 element array
	[0]: actual gpu memory mode
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB);
	*/

	// subBgTrigger: no longer used 
	int regChoice = 4; // Do 2D regitration first
	bool flagTmx = false;
	if (inputTmx == 1) {
		flagTmx = true;
		regChoice = 2; // if use input matrix, do not perform 2D registration
	}
	int gpuMemMode = 1;
	bool verbose = false;
	int regStatus = reg3d(h_reg, iTmx, h_img1, h_img2, imSize1, imSize2, regChoice, affMethod,
		flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	bool mStatus = checkmatrix(iTmx, imSize1[0], imSize1[1], imSize1[2]);
	if (!mStatus) {
		regChoice = 2;
		regStatus = reg3d(h_reg, iTmx, h_img1, h_img2, imSize1, imSize2, regChoice, affMethod,
			flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	}
	return regStatus;
}

int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize, bool flagConstInitial,
	int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp) {
	// gpuMemMode --> -1: Automatically set memory mode; 0: All on CPU; 1: sufficient GPU memory; 2: GPU memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;	
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];
	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];
	// FFT size
	long long int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);
	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);

	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format

	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[1] = (float)freeMem / 1048576.0f;
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
		// Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 6 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}
	deconRecords[0] = gpuMemMode;

	float
		*h_StackA,
		*h_StackE,
		*d_StackA,
		*d_StackE;
	fComplex
		*h_PSFSpectrum,
		*h_FlippedPSFSpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_StackESpectrum;

	switch (gpuMemMode) {
	case 0:
		// CPU deconvolution
		time1 = clock();
		// allocate memory
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));

		// *** PSF Preparation
		// OTF 
		changestorageordercpu(h_StackA, h_psf, PSFx, PSFy, PSFz, 1);
		genOTFcpu((fftwf_complex *)h_PSFSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			changestorageordercpu(h_StackA, h_psf_bp, PSFx, PSFy, PSFz, 1);
			genOTFcpu((fftwf_complex *)h_FlippedPSFSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipcpu(h_StackE, h_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFcpu((fftwf_complex *)h_FlippedPSFSpectrum, h_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}

		// *** image  Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img, imx, imy, imz, 1);
			padstackcpu(h_StackA, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackA, h_img, imx, imy, imz, 1);
		}

		// *** deconvolution ****
		memset(h_StackE, 0, totalSizeFFT * sizeof(float));
		decon_singleview_OTF0(h_StackE, h_StackA, (fftwf_complex *)h_PSFSpectrum,
			(fftwf_complex *)h_FlippedPSFSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropcpu(h_StackA, h_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordercpu(h_decon, h_StackA, imx, imy, imz, -1);
		}
		else {
			changestorageordercpu(h_decon, h_StackE, imx, imy, imz, -1);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		// release variables
		free(h_StackA); free(h_StackE);
		free(h_PSFSpectrum);
		free(h_FlippedPSFSpectrum);
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFSpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum * sizeof(fComplex));
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;
		// *** PSF Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch){
			cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			genOTFgpu(d_FlippedPSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFgpu(d_FlippedPSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); //PSF already normalized
		}
		// *** image Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1); //1: change tiff storage order to C storage order
		}
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_singleview_OTF1(d_StackE, d_StackA, d_PSFSpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackA); cudaFree(d_StackE); 
		cudaFree(d_PSFSpectrum); cudaFree(d_FlippedPSFSpectrum); cudaFree(d_StackESpectrum);
		break;
	case 2:
		time1 = clock();

		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float)); // also to store spectrum images
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float)); // also to store spectrum images
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;

		// *** PSF Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		d_PSFSpectrum = (fComplex *)d_StackE;
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		cudaMemcpy(h_PSFSpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			d_FlippedPSFSpectrum = (fComplex *)d_StackE;
			genOTFgpu(d_FlippedPSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			d_FlippedPSFSpectrum = (fComplex *)d_StackA;
			genOTFgpu(d_FlippedPSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); 
		}
		cudaMemcpy(h_FlippedPSFSpectrum, d_FlippedPSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);

		// *** image Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1); //1: change tiff storage order to C storage order
		}
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_singleview_OTF2(d_StackE, d_StackA, h_PSFSpectrum, h_FlippedPSFSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release variables
		free(h_PSFSpectrum); free(h_FlippedPSFSpectrum);
		cudaFree(d_StackA); cudaFree(d_StackE);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return 1;
	}
	end = clock();
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	}
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, unsigned int *psfSize,
	bool flagConstInitial, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2) {
	// gpuMemMode --> -1: Automatically set memory mode; 0: All on CPU; 1: sufficient GPU memory; 2: GPU memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;	
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];
	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];
	// FFT size
	long long int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);
	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);

	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format

	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[1] = (float)freeMem / 1048576.0f;
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
						   // Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 9 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}
	deconRecords[0] = gpuMemMode;
	
	float
		*h_StackA,
		*h_StackB,
		*h_StackE,
		*d_StackA,
		*d_StackB,
		*d_StackE;

	fComplex
		*h_PSFASpectrum,
		*h_PSFBSpectrum,
		*h_FlippedPSFASpectrum,
		*h_FlippedPSFBSpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_PSFASpectrum,
		*d_PSFBSpectrum,
		*d_FlippedPSFASpectrum,
		*d_FlippedPSFBSpectrum;
	
	switch (gpuMemMode) {
	case 0:
		// CPU deconvolution
		time1 = clock();
		// allocate memory
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));

		// *** PSF A Preparation
		// OTF 
		changestorageordercpu(h_StackA, h_psf1, PSFx, PSFy, PSFz, 1);
		genOTFcpu((fftwf_complex *)h_PSFASpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			changestorageordercpu(h_StackA, h_psf_bp1, PSFx, PSFy, PSFz, 1);
			genOTFcpu((fftwf_complex *)h_FlippedPSFASpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipcpu(h_StackE, h_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFcpu((fftwf_complex *)h_FlippedPSFASpectrum, h_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}

		// *** PSF B Preparation
		// OTF 
		changestorageordercpu(h_StackA, h_psf2, PSFx, PSFy, PSFz, 1);
		genOTFcpu((fftwf_complex *)h_PSFBSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			changestorageordercpu(h_StackA, h_psf_bp2, PSFx, PSFy, PSFz, 1);
			genOTFcpu((fftwf_complex *)h_FlippedPSFBSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipcpu(h_StackE, h_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFcpu((fftwf_complex *)h_FlippedPSFBSpectrum, h_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img1, imx, imy, imz, 1);
			padstackcpu(h_StackA, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackA, h_img1, imx, imy, imz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img2, imx, imy, imz, 1);
			padstackcpu(h_StackB, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackB, h_img2, imx, imy, imz, 1);
		}
		// *** deconvolution ****
		memset(h_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF0(h_StackE, h_StackA, h_StackB, (fftwf_complex *)h_PSFASpectrum, (fftwf_complex *)h_PSFBSpectrum,
			(fftwf_complex *)h_FlippedPSFASpectrum, (fftwf_complex *)h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropcpu(h_StackA, h_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordercpu(h_decon, h_StackA, imx, imy, imz, -1);
		}
		else {
			changestorageordercpu(h_decon, h_StackE, imx, imy, imz, -1);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		// release variables
		free(h_StackA); free(h_StackB);  free(h_StackE);
		free(h_PSFASpectrum); free(h_PSFBSpectrum);
		free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum);
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_PSFASpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_PSFBSpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFASpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFBSpectrum, totalSizeSpectrum * sizeof(fComplex));
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;
		// *** PSF A Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
		genOTFgpu(d_PSFASpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_FlippedPSFASpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFgpu(d_FlippedPSFASpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		// *** PSF B Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
		genOTFgpu(d_PSFBSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_FlippedPSFBSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFgpu(d_FlippedPSFBSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); 
		}
		cudaCheckErrors("****PSF and OTF preparation failed !!!!*****");
		cudaMalloc((void **)&d_StackB, totalSizeMax * sizeof(float));
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); 
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackB, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1); 
			padstackgpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF1(d_StackE, d_StackA, d_StackB, d_PSFASpectrum, d_PSFBSpectrum, 
			d_FlippedPSFASpectrum, d_FlippedPSFBSpectrum,FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); 
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); 
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackA); cudaFree(d_StackB);  cudaFree(d_StackE);
		cudaFree(d_PSFASpectrum); cudaFree(d_PSFBSpectrum); 
		cudaFree(d_FlippedPSFASpectrum); cudaFree(d_FlippedPSFBSpectrum); 
		break;
	case 2:
		time1 = clock();
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float)); // also to store spectrum images
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float)); // also to store spectrum images
																	  //check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;

		// *** PSF A Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		d_PSFSpectrum = (fComplex *)d_StackE;
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		cudaMemcpy(h_PSFASpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			d_PSFSpectrum = (fComplex *)d_StackA;
			genOTFgpu(d_PSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); 
		}
		cudaMemcpy(h_FlippedPSFASpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);

		// *** PSF B Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
		d_PSFSpectrum = (fComplex *)d_StackE;
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		cudaMemcpy(h_PSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			d_PSFSpectrum = (fComplex *)d_StackA;
			genOTFgpu(d_PSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); //PSF already normalized
		}
		cudaMemcpy(h_FlippedPSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		cudaCheckErrors("****PSF and OTF preparation failed !!!!*****");

		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); 
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1); 
		}
		cudaMemcpy(h_StackB, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); 
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1); 
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");
		
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF2(d_StackE, d_StackA, h_StackB, h_PSFASpectrum, h_PSFBSpectrum,
			h_FlippedPSFASpectrum, h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release variables
		free(h_PSFASpectrum); free(h_PSFBSpectrum);
		free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum);
		cudaFree(d_StackA); cudaFree(d_StackE);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return -1;
	}
	end = clock();
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	}
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

// batch
int decon_dualview_batch(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, fComplex *OTF1, fComplex *OTF2, fComplex *OTF1_bp, fComplex *OTF2_bp, unsigned int *otfSize,
	bool flagConstInitial, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords) {
	// gpuMemMode --> -1: Automatically set memory mode; 0: All on CPU; 1: sufficient GPU memory; 2: GPU memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;	
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];
	// FFT size
	long long int
		FFTx, FFTy, FFTz;
	FFTx = long long int(otfSize[0]);
	FFTy = long long int(otfSize[1]);
	FFTz = long long int(otfSize[2]);

	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format

														// ****************** Processing Starts***************** //
														// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[1] = (float)freeMem / 1048576.0f;
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
							// Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 9 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}
	deconRecords[0] = gpuMemMode;

	float
		*h_StackA,
		*h_StackB,
		*h_StackE,
		*d_StackA,
		*d_StackB,
		*d_StackE;

	fComplex
		*h_PSFASpectrum,
		*h_PSFBSpectrum,
		*h_FlippedPSFASpectrum,
		*h_FlippedPSFBSpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_PSFASpectrum,
		*d_PSFBSpectrum,
		*d_FlippedPSFASpectrum,
		*d_FlippedPSFBSpectrum;

	switch (gpuMemMode) {
	case 0:
		// CPU deconvolution
		time1 = clock();
		// allocate memory
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		// *** PSF A Preparation
		// OTF 
		h_PSFASpectrum = OTF1;
		h_FlippedPSFASpectrum = OTF1_bp;
		// *** PSF B Preparation
		// OTF 
		h_PSFBSpectrum = OTF2;
		h_FlippedPSFBSpectrum = OTF2_bp;
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img1, imx, imy, imz, 1);
			padstackcpu(h_StackA, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackA, h_img1, imx, imy, imz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img2, imx, imy, imz, 1);
			padstackcpu(h_StackB, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackB, h_img2, imx, imy, imz, 1);
		}
		// *** deconvolution ****
		memset(h_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF0(h_StackE, h_StackA, h_StackB, (fftwf_complex *)h_PSFASpectrum, (fftwf_complex *)h_PSFBSpectrum,
			(fftwf_complex *)h_FlippedPSFASpectrum, (fftwf_complex *)h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropcpu(h_StackA, h_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordercpu(h_decon, h_StackA, imx, imy, imz, -1);
		}
		else {
			changestorageordercpu(h_decon, h_StackE, imx, imy, imz, -1);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		// release variables
		free(h_StackA); free(h_StackB);  free(h_StackE);
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float));
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		deconRecords[2] = (float)freeMem / 1048576.0f;
		// *** PSF A Preparation
		// OTF 
		h_PSFASpectrum = OTF1;
		h_FlippedPSFASpectrum = OTF1_bp;
		// *** PSF B Preparation
		// OTF 
		h_PSFBSpectrum = OTF2;
		h_FlippedPSFBSpectrum = OTF2_bp;
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1);
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackB, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1);
			padstackgpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF1(d_StackE, d_StackA, d_StackB, d_PSFASpectrum, d_PSFBSpectrum,
			d_FlippedPSFASpectrum, d_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1);
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1);
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackA); cudaFree(d_StackB);  cudaFree(d_StackE);
		break;
	case 2:
		time1 = clock();
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float)); // also to store spectrum images
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float)); // also to store spectrum images
																	  //check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

		deconRecords[2] = (float)freeMem / 1048576.0f;

		// *** PSF A Preparation
		// OTF 
		h_PSFASpectrum = OTF1;
		h_FlippedPSFASpectrum = OTF1_bp;
		// *** PSF B Preparation
		// OTF 
		h_PSFBSpectrum = OTF2;
		h_FlippedPSFBSpectrum = OTF2_bp;

		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1);
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1);
		}
		cudaMemcpy(h_StackB, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1);
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1);
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");

		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF2(d_StackE, d_StackA, h_StackB, h_PSFASpectrum, h_PSFBSpectrum,
			h_FlippedPSFASpectrum, h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release variables
		cudaFree(d_StackA); cudaFree(d_StackE);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return -1;
	}
	end = clock();
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	}
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

int decon_singleview_old(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize,
	int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp){
	// gpuMemMode --> 0: Automatically set memory mode based on calculations; 1: sufficient memory; 2: memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;

	float
		*h_StackA,
		*h_StackE,
		*d_StackA,
		*d_StackE,
		*d_StackT;

	fComplex
		*h_PSFSpectrum,
		*h_FlippedPSFSpectrum,
		*h_StackESpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_StackESpectrum;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];

	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];

	//FFT size
	long long int
		FFTx, FFTy, FFTz,
		PSFox, PSFoy, PSFoz,
		imox, imoy, imoz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);
	// set original points for padding and cropping
	//fftz.y.z?
	PSFox = round(PSFx / 2);// round((FFTx - PSFx) / 2);
	PSFoy = round(PSFy / 2);//round((FFTy - PSFy) / 2);
	PSFoz = round(PSFz / 2);//round((FFTz - PSFz) / 2 );
	imox = round((FFTx - imSize[0]) / 2);
	imoy = round((FFTy - imSize[1]) / 2);
	imoz = round((FFTz - imSize[2]) / 2);

	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);

	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);

	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format
	long long int totalSizeMax2 = totalSizeMax > totalSizePSF ? totalSizeMax : totalSizePSF; // in floating format: in case PSF has a larger size
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	// allocate memory
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[1] = (float)freeMem / 1048576.0f;
	cudaMalloc((void **)&d_StackA, totalSizeMax2 *sizeof(float));
	cudaMalloc((void **)&d_StackE, totalSizeMax2 *sizeof(float));
	cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
	cudaMemset(d_StackE, 0, totalSizeMax2*sizeof(float));
	//check GPU status
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory(after partially mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[2] = (float)freeMem / 1048576.0f;

	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// 0: Automatically set memory mode based on calculations; 
	// 1: sufficient memory; 2: memory optimized.
	if (gpuMemMode == 0){ //Automatically set memory mode based on calculations.
		if (freeMem > 4 * totalSizeMax * sizeof(float)){ // 7 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else {// no more GPU variables needed
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
	}
	deconRecords[0] = gpuMemMode;
	double mySumPSF = 0;
	switch (gpuMemMode){
	case 1:// efficient GPU calculation
		time1 = clock();
		cudaMalloc((void **)&d_StackT, totalSizeFFT *sizeof(float));
		cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		// *** PSF Preparation
		//PSF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		mySumPSF = sumcpu(h_psf, totalSizePSF);
		multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		if (!flagUnmatch){ // traditional backprojector matched PSF 
			flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
			cudaMemcpy(h_psf_bp, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
		//PSF bp
		cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		if (flagUnmatch){
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf_bp, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFSpectrum);

		// Prepare Stack Data
		cudaMemcpy(d_StackA, h_img, totalSize* sizeof(float), cudaMemcpyHostToDevice);
		//eliminate 0 in stacks
		maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
		changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
		padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		// initialize estimation
		cudaMemcpy(d_StackE, d_StackA, totalSizeFFT* sizeof(float), cudaMemcpyDeviceToDevice);
		cudaCheckErrors("image preparing fail");
		//printf("...Initializing deconvolution iteration...\n");
		for (int itNum = 1; itNum <= itNumForDecon; itNum++){
			// ### iterate with StackA and PSFA///////////////////
			// convolve StackE with PSFA
			//printf("...Processing iteration %d\n", it);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
			// divid StackA by StackTemp
			div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
			// convolve StackTemp with FlippedPSFA
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
			// multiply StackE and StackTemp
			multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
			maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
		}
		cropgpu(d_StackT, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
		//printf("...Deconvolution completed ! ! !\n");
		cudaThreadSynchronize();
		changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
		cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackT); cudaFree(d_PSFSpectrum); cudaFree(d_FlippedPSFSpectrum); cudaFree(d_StackESpectrum);
		break;
	case 2:
		time1 = clock();
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		h_StackESpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		d_StackESpectrum = (fComplex *)d_StackE; // share the same physic memory
		// *** PSF Preparation
		//PSF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		mySumPSF = sumcpu(h_psf, totalSizePSF);
		multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		if (!flagUnmatch){ // traditional backprojector matched PSF 
			flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
			cudaMemcpy(h_psf_bp, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_PSFSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
		//PSF bp
		cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		if (flagUnmatch){
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf_bp, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_FlippedPSFSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);

		// Prepare Stack Data
		cudaMemcpy(d_StackA, h_img, totalSize* sizeof(float), cudaMemcpyHostToDevice);
		//eliminate 0 in stacks
		maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
		changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
		padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//

		// initialize estimation
		cudaMemcpy(h_StackA, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_StackE, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

		d_PSFSpectrum = (fComplex *)d_StackA; // share the same physic memory
		d_FlippedPSFSpectrum = (fComplex *)d_StackA; // share the same physic memory
		cudaCheckErrors("image preparing fail");
		//printf("...Initializing deconvolution iteration...\n");
		for (int itNum = 1; itNum <= itNumForDecon; itNum++){
			// ### iterate with StackA and PSFA///////////////////
			// convolve StackE with PSFA
			//printf("...Processing iteration %d\n", it);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(d_PSFSpectrum, h_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);

			// divid StackA by StackTemp
			cudaMemcpy(d_StackE, h_StackA, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
			div3Dgpu(d_StackA, d_StackE, d_StackA, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
			// convolve StackTemp with FlippedPSFA
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(d_FlippedPSFSpectrum, h_FlippedPSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);//test
			// multiply StackE and StackTemp
			cudaMemcpy(d_StackE, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
			multi3Dgpu(d_StackA, d_StackE, d_StackA, FFTx, FFTy, FFTz);//
			cudaMemcpy(h_StackE, d_StackA, totalSizeFFT* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cropgpu(d_StackE, d_StackA, imx, imy, imz, FFTx, FFTy, FFTz);//
		//printf("...Deconvolution completed ! ! !\n");
		cudaThreadSynchronize();
		//## Write stack to tiff image
		changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
		cudaMemcpy(h_decon, d_StackA, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CPU memory
		free(h_StackA);  free(h_StackE); free(h_PSFSpectrum); free(h_FlippedPSFSpectrum); free(h_StackESpectrum);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return -1;
	}
	// release GPU memory
	cudaFree(d_StackA);
	cudaFree(d_StackE);
	// destroy plans
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

int decon_dualview_old(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, 
	unsigned int *psfSize, int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2){
	// gpuMemMode --> 0: Automatically set memory mode based on calculations; 1: sufficient memory; 2: memory optimized; 3: memory further optimized.
	//deconRecords: 10 elements
	//[0]:  the actual memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;
	float
		*h_StackA,
		*h_StackB,
		*h_StackE,
		*h_StackT,
		*d_StackA,
		*d_StackB,
		*d_StackE,
		*d_StackT;

	fComplex
		*h_PSFASpectrum,
		*h_PSFBSpectrum,
		*h_FlippedPSFASpectrum,
		*h_FlippedPSFBSpectrum,
		*h_StackESpectrum,
		*d_PSFSpectrum,
		*d_PSFASpectrum,
		*d_PSFBSpectrum,
		*d_FlippedPSFASpectrum,
		*d_FlippedPSFBSpectrum,
		*d_StackESpectrum;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];

	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];

	//FFT size
	long long int
		FFTx, FFTy, FFTz,
		PSFox, PSFoy, PSFoz,
		imox, imoy, imoz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);
	// set original points for padding and cropping
	//fftz.y.z?
	PSFox = round(PSFx / 2);// round((FFTx - PSFx) / 2);
	PSFoy = round(PSFy / 2);//round((FFTy - PSFy) / 2);
	PSFoz = round(PSFz / 2);//round((FFTz - PSFz) / 2 );
	imox = round((FFTx - imSize[0]) / 2);
	imoy = round((FFTy - imSize[1]) / 2);
	imoz = round((FFTz - imSize[2]) / 2);
	/*
	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);

	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);
	*/
	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format
	long long int totalSizeMax2 = totalSizeMax > totalSizePSF ? totalSizeMax : totalSizePSF; // in floating format: in case PSF has a larger size
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	// allocate memory
	cudaMemGetInfo(&freeMem, &totalMem);
	//printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[1] = (float)freeMem / 1048576.0f;
	cudaMalloc((void **)&d_StackA, totalSizeMax2 *sizeof(float));
	cudaMalloc((void **)&d_StackE, totalSizeMax2 *sizeof(float));
	cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
	cudaMemset(d_StackE, 0, totalSizeMax2*sizeof(float));
	//check GPU status
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

	cudaMemGetInfo(&freeMem, &totalMem);
	//printf("...GPU free memory(after partially mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[2] = (float)freeMem / 1048576.0f;

	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// 0: Automatically set memory mode based on calculations; 
	// 1: sufficient memory; 2: memory optimized; 3: memory further optimized.
	if (gpuMemMode == 0){ //Automatically set memory mode based on calculations.
		if (freeMem > 7 * totalSizeMax * sizeof(float)){ // 7 more GPU variables
			gpuMemMode = 1;
			//printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 4 * totalSizeMax * sizeof(float)){// 4 more GPU variables
			gpuMemMode = 2;
			//printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else {// no more GPU variables needed
			gpuMemMode = 3;
			//printf("\n GPU memory is futher optimized, processing in memory saved mode !!!\n");
		}
	}
	deconRecords[0] = gpuMemMode;
	double mySumPSF = 0;
	switch (gpuMemMode){
		case 1:// efficient GPU calculation
			time1 = clock();
			cudaMalloc((void **)&d_StackB, totalSizeFFT *sizeof(float));
			cudaMalloc((void **)&d_StackT, totalSizeFFT *sizeof(float));
			cudaMalloc((void **)&d_PSFASpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_PSFBSpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_FlippedPSFASpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_FlippedPSFBSpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			// *** PSF Preparation
			//PSF A 
			cudaMemcpy(d_StackE, h_psf1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf1, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp1, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFASpectrum);
			//PSF B 
			cudaMemcpy(d_StackE, h_psf2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf2, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp2, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFBSpectrum);
			// PSF bp A
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp1, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFASpectrum);
			// PSF bp B
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp2, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFBSpectrum);
			cudaCheckErrors("PSF preparing fail");
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[3] = (float)freeMem / 1048576.0f;
			// Prepare Stack Data
			cudaMemcpy(d_StackA, h_img1, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_StackB, h_img2, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			//eliminate 0 in stacks
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			maxvalue3Dgpu(d_StackB, d_StackB, (float)(SMALLVALUE), imx, imy, imz);
			/*
			double sumStackA = sum3Dgpu(d_StackA, d_2D, h_2D, imx, imy, imz);
			double sumStackB = sum3Dgpu(d_StackB, d_2D, h_2D, imx, imy, imz);
			//printf("Sum of Stack A: %.2f \n ", sumStackA);
			//printf("Sum of Stack B: %.2f \n ", sumStackB);
			multivaluegpu(d_StackB, d_StackB, (float)(sumStackA / sumStackB), imx, imy, imz);
			*/

			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//	

			// initialize estimation by average of StackA and StackB
			add3Dgpu(d_StackE, d_StackA, d_StackB, FFTx, FFTy, FFTz);
			multivaluegpu(d_StackE, d_StackE, (float)0.5, FFTx, FFTy, FFTz);
			cudaCheckErrors("image preparing fail");
			time2 = clock();
			// *****Joint deconvoultion	
			for (int itNum = 1; itNum <= itNumForDecon; itNum++){
				// ### iterate with StackA and PSFA///////////////////
				// convolve StackE with PSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackA by StackTemp
				div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
				// convolve StackTemp with FlippedPSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

				// ### iterate with StackB and PSFB /////////////////
				// convolve StackE with PSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackB by StackTemp
				div3Dgpu(d_StackT, d_StackB, d_StackT, FFTx, FFTy, FFTz);//
				// convolve StackTemp with FlippedPSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
			}
			cropgpu(d_StackT, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			//printf("...Deconvolution completed ! ! !\n");
			cudaThreadSynchronize();
			changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
			time3 = clock();
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[4] = (float)freeMem / 1048576.0f;
			// release CUDA variables
			cudaFree(d_StackB); cudaFree(d_StackT); cudaFree(d_PSFASpectrum); cudaFree(d_PSFBSpectrum);
			cudaFree(d_FlippedPSFASpectrum); cudaFree(d_FlippedPSFBSpectrum); cudaFree(d_StackESpectrum);
			break;
		case 2: // memory saved mode 2
			time1 = clock();
			h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_StackB, totalSizeMax *sizeof(float));
			cudaMalloc((void **)&d_StackT, totalSizeMax *sizeof(float));
			cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
			d_PSFASpectrum = d_PSFSpectrum;
			d_PSFBSpectrum = d_PSFSpectrum;
			d_FlippedPSFASpectrum = d_PSFSpectrum;
			d_FlippedPSFBSpectrum = d_PSFSpectrum;

			// *** PSF Preparation
			//PSF A 
			cudaMemcpy(d_StackE, h_psf1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf1, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp1, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_PSFASpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			//PSF B 
			cudaMemcpy(d_StackE, h_psf2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf2, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp2, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_PSFBSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp A
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp1, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_FlippedPSFASpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp B
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp2, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_FlippedPSFBSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			cudaCheckErrors("PSF preparing fail");
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[3] = (float)freeMem / 1048576.0f;
			// Prepare Stack Data
			//eliminate 0 in stacks
			cudaMemcpy(d_StackA, h_img1, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//

			cudaMemcpy(d_StackB, h_img2, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackB, d_StackB, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//

			// initialize estimation by average of StackA and StackB
			add3Dgpu(d_StackE, d_StackA, d_StackB, FFTx, FFTy, FFTz);
			multivaluegpu(d_StackE, d_StackE, (float)0.5, FFTx, FFTy, FFTz);
			cudaCheckErrors("image preparing fail");
			time2 = clock();
			for (int itNum = 1; itNum <= itNumForDecon; itNum++){
				// ### iterate with StackA and PSFA///////////////////
				// convolve StackE with PSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_PSFASpectrum, h_PSFASpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackA by StackTemp
				div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
				// convolve StackTemp with FlippedPSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_FlippedPSFASpectrum, h_FlippedPSFASpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

				// ### iterate with StackB and PSFB /////////////////
				// convolve StackE with PSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
				cudaMemcpy(d_PSFBSpectrum, h_PSFBSpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackB by StackTemp
				div3Dgpu(d_StackT, d_StackB, d_StackT, FFTx, FFTy, FFTz);//
				// convolve StackTemp with FlippedPSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_FlippedPSFBSpectrum, h_FlippedPSFBSpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

			}
			cropgpu(d_StackT, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			cudaThreadSynchronize();
			changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
			time3 = clock();
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[4] = (float)freeMem / 1048576.0f;
			// release CPU and GPU memory
			free(h_PSFASpectrum); free(h_PSFBSpectrum); free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum);
			cudaFree(d_StackB); cudaFree(d_StackT); cudaFree(d_PSFSpectrum); cudaFree(d_StackESpectrum);
			break;
		case 3: // memory saved mode 3
			time1 = clock();
			h_StackA = (float *)malloc(totalSizeFFT * sizeof(float));
			h_StackB = (float *)malloc(totalSizeFFT * sizeof(float));
			h_StackE = (float *)malloc(totalSizeFFT * sizeof(float));
			h_StackT = (float *)malloc(totalSizeFFT * sizeof(float));
			h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_StackESpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));

			d_StackESpectrum = (fComplex *)d_StackA;
			d_PSFSpectrum = (fComplex *)d_StackE;
			// *** PSF Preparation
			//PSF A 
			cudaMemcpy(d_StackE, h_psf1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf1, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp1, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_PSFASpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			//PSF B 
			cudaMemcpy(d_StackE, h_psf2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf2, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipgpu(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp2, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_PSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp A
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp1, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_FlippedPSFASpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp B
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp2, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSFgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_FlippedPSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			cudaCheckErrors("PSF preparing fail");
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[3] = (float)freeMem / 1048576.0f;
			// Prepare Stack Data
			//eliminate 0 in stacks
			cudaMemcpy(d_StackA, h_img1, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
			cudaMemcpy(h_StackA, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

			cudaMemcpy(d_StackA, h_img2, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
			cudaMemcpy(h_StackB, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

			// initialize estimation by average of StackA and StackB
			cudaMemcpy(d_StackE, h_StackA, totalSizeFFT * sizeof(float), cudaMemcpyHostToDevice);
			add3Dgpu(d_StackE, d_StackA, d_StackE, FFTx, FFTy, FFTz);
			multivaluegpu(d_StackE, d_StackE, (float)0.5, FFTx, FFTy, FFTz);
			cudaMemcpy(h_StackE, d_StackE, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
			cudaCheckErrors("image preparing fail");
			time2 = clock();
			for (int itNum = 1; itNum <= itNumForDecon; itNum++){
				//printf("...Processing iteration %d\n", it);
				// ### iterate with StackA and PSFA///////////////////
				// convolve StackE with PSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(h_StackE, d_StackE, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

				cudaMemcpy(d_PSFSpectrum, h_PSFASpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);

				// divid StackA by StackTemp
				cudaMemcpy(d_StackA, h_StackA, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				div3Dgpu(d_StackE, d_StackA, d_StackE, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
				// convolve StackTemp with FlippedPSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_PSFSpectrum, h_FlippedPSFASpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);//test
				// multiply StackE and StackTemp
				cudaMemcpy(d_StackA, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				multi3Dgpu(d_StackE, d_StackE, d_StackA, FFTx, FFTy, FFTz);//
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

				// ### iterate with StackB and PSFB /////////////////
				// convolve StackE with PSFB		
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
				cudaMemcpy(h_StackE, d_StackE, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

				cudaMemcpy(d_PSFSpectrum, h_PSFBSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);
				// divid StackB by StackTemp
				cudaMemcpy(d_StackA, h_StackB, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				div3Dgpu(d_StackE, d_StackA, d_StackE, FFTx, FFTy, FFTz);//

				// convolve StackTemp with FlippedPSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_PSFSpectrum, h_FlippedPSFBSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);
				// multiply StackE and StackTemp
				cudaMemcpy(d_StackA, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				multi3Dgpu(d_StackE, d_StackE, d_StackA, FFTx, FFTy, FFTz);
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
			}
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			cudaThreadSynchronize();
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
			time3 = clock();
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[4] = (float)freeMem / 1048576.0f;
			// release CPU memory
			free(h_StackA); free(h_StackB); free(h_StackE); free(h_StackT); free(h_PSFASpectrum);
			free(h_PSFBSpectrum); free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum); free(h_StackESpectrum);
			break;
		default:
			//printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
			return -1;
	}
	// release GPU memory
	cudaFree(d_StackA);
	cudaFree(d_StackE);
	// destroy plans
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	//printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

//// 3D fusion: registration and deconvolution
int fusion_dualview(float *h_decon, float *h_reg, float *h_prereg1, float *h_prereg2, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, bool flagTmx, int regChoice, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2){
	// **** registration and joint deconvolution for two images:  ***
	/*
	*** imBRotation: image B rotation
	0: no rotation; 
	1: 90deg rotation by y axis ; 
	-1: -90deg rotation by y axis;
	*
	*** registration choice: regChoice
	0: no phasor or affine registration; if flagTmx is true, transform d_img2 based on input matrix;
	1: phasor registraion (pixel-level translation only);
	2: affine registration (with or without input matrix); affine: 12 degrees of freedom;
	3: phasor registration --> affine registration (input matrix disabled); affine: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	4: 2D MIP registration --> affine registration (input matrix disabled); affine: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*
	*** flagTmx: only if regChoice == 0, 2
	true: use iTmx as input matrix;
	false: default;
	*
	*** gpuMemMode
	-1: Automatically set memory mode;
	0: All on CPU. // currently does not work
	1: sufficient GPU memory;
	2: GPU memory optimized;
	*
	*** fusionRecords: 22 element array
	--> 0-10: regRecords; 11-20: deconRecords; 21: total time;
	[0]: actual gpu memory mode for registration
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB), if use gpu
	[11]:  the actual GPU memory mode used for deconvolution;
	[12] -[16]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	[17] -[20]: initializing time, prepocessing time, decon time, total time;
	[21]: total time
	*
	*** flagUnmatach 
	false: use traditional backprojector (flipped PSF);
	true: use unmatch back projector;
	*/

	// ************get basic input images information ******************	
	// variables for memory and time cost records
	clock_t start, end;
	end = 0;
	start = clock();
	// ****************** calculate images' size ************************* //
	long long int imx, imy, imz;
	long long int imx1, imy1, imz1, imx2, imy2, imz2;
	unsigned int imSize[3], imSize1[3], imSize2[3], imSizeTemp[3]; // modify to long long 
	float pixelSize[3], pixelSizeTemp[3];
	bool flagInterp1 = true, flagInterp2 = true;
	if ((pixelSize1[0] == pixelSize1[1]) && (pixelSize1[0] == pixelSize1[2]))
		flagInterp1 = false;
	if ((pixelSize2[0] == pixelSize1[0]) && (pixelSize2[1] == pixelSize1[0]) && (pixelSize2[2] == pixelSize1[0]))
		flagInterp2 = false;
	// image A: base image
	pixelSize[0] = pixelSize[1] = pixelSize[2] = pixelSize1[0];
	imx1 = imSizeIn1[0];
	imy1 = round((float)imSizeIn1[1] * pixelSize1[1] / pixelSize[1]);
	imz1 = round((float)imSizeIn1[2] * pixelSize1[2] / pixelSize[2]);
	imSize[0] = imSize1[0] = imx1; imSize[1] = imSize1[1] = imy1; imSize[2] = imSize1[2] = imz1;
	imx = imx1; imy = imy1; imz = imz1; // also as output size

	// image B: target image
	imSizeTemp[0] = imSizeIn2[0]; imSizeTemp[1] = imSizeIn2[1]; imSizeTemp[2] = imSizeIn2[2];
	pixelSizeTemp[0] = pixelSize2[0]; pixelSizeTemp[1] = pixelSize2[1]; pixelSizeTemp[2] = pixelSize2[2];
	if ((imRotation == 1) || (imRotation == -1)){ //if there is rotation for B, change image dimemsion size
		imSizeIn2[0] = imSizeTemp[2];
		imSizeIn2[2] = imSizeTemp[0];
		pixelSize2[0] = pixelSizeTemp[2];
		pixelSize2[2] = pixelSizeTemp[0];
	}
	float pixelSizeRatioBx = pixelSize2[0] / pixelSize[0];
	float pixelSizeRatioBy = pixelSize2[1] / pixelSize[1];
	float pixelSizeRatioBz = pixelSize2[2] / pixelSize[2];
	imx2 = round((float)imSizeIn2[0] * pixelSizeRatioBx);
	imy2 = round((float)imSizeIn2[1] * pixelSizeRatioBy);
	imz2 = round((float)imSizeIn2[2] * pixelSizeRatioBz);
	imSize2[0] = imx2; imSize2[1] = imy2; imSize2[2] = imz2;

	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSizeIn[0], PSFy = psfSizeIn[1], PSFz = psfSizeIn[2];

	//FFT size
	long long int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	// total pixel count for each images
	long long int totalSizeIn1 = imSizeIn1[0] * imSizeIn1[1] * imSizeIn1[2]; // in floating format
	long long int totalSizeIn2 = imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2]; // in floating format
	long long int totalSize1 = imx1*imy1*imz1; // in floating format
	long long int totalSize2 = imx2*imy2*imz2; // in floating format
	long long int totalSize = totalSize1; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSize12 = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	long long int totalSizeMax = (totalSize1 > totalSizeFFT * 2) ? totalSize1 : totalSizeFFT * 2; // in floating format
	
	// ****************** Processing Starts*****************
	size_t totalMem = 0;
	size_t freeMem = 0;
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	}

	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
							// Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 9 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}

	if ((gpuMemMode != 1) || (gpuMemMode != 2)) {
		printf("\n****Wrong gpuMemMode setup (All in CPU is currently not supported), processing stopped !!! ****\n");
		return 1;
	}

	// ************** Registration *************
	// ***interpolation and rotation
	float
		*h_StackA,
		*h_StackB;
	float
		*d_imgE;
	float *d_img3D = NULL, *d_img2DMax = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array1, *d_Array2;
	float *h_aff12 = (float *)malloc((NDIM)* sizeof(float));
	h_StackA = (float *)malloc(totalSize12 * sizeof(float));
	h_StackB = (float *)malloc(totalSize12 * sizeof(float));
	cudaMalloc((void **)&d_img3D, totalSize12 *sizeof(float));
		
	//// image 1
	if (flagInterp1){
		cudaMalloc3DArray(&d_Array1, &channelDesc, make_cudaExtent(imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]));
		cudacopyhosttoarray(d_Array1, channelDesc, h_img1, imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]);
		BindTexture(d_Array1, channelDesc);
		cudaCheckErrors("Texture create fail");
		// transform matrix for Stack A interpolation
		h_aff12[0] = 1, h_aff12[1] = 0, h_aff12[2] = 0, h_aff12[3] = 0;
		h_aff12[4] = 0, h_aff12[5] = pixelSize[1] / pixelSize1[1], h_aff12[6] = 0, h_aff12[7] = 0;
		h_aff12[8] = 0, h_aff12[9] = 0, h_aff12[10] = pixelSize[2] / pixelSize1[2], h_aff12[11] = 0;
		CopyTranMatrix(h_aff12, NDIM * sizeof(float));
		affineTransform(d_img3D, imx1, imy1, imz1, imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]);
		UnbindTexture();
		cudaFreeArray(d_Array1);
		cudaMemcpy(h_StackA, d_img3D, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	}// after interpolation, Stack A size: imx x imy x imz;
	else
		memcpy(h_StackA, h_img1, totalSize * sizeof(float));
	cudaThreadSynchronize();

	//// image 2
	// rotation
	if ((imRotation == 1) || (imRotation == -1)){
		cudaMalloc((void **)&d_imgE, totalSizeIn2 * sizeof(float));
		cudaMemcpy(d_imgE, h_img2, totalSizeIn2 * sizeof(float), cudaMemcpyHostToDevice);
		rotbyyaxis(d_img3D, d_imgE, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2], imRotation);
		cudaMemcpy(h_StackB, d_img3D, imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2] * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_imgE);
	}
	if (flagInterp2){
		cudaMalloc3DArray(&d_Array2, &channelDesc, make_cudaExtent(imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]));
		if ((imRotation == 1) || (imRotation == -1))
			cudacopyhosttoarray(d_Array2, channelDesc, h_StackB, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]);
		else
			cudacopyhosttoarray(d_Array2, channelDesc, h_img2, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]);
		BindTexture(d_Array2, channelDesc);
		cudaCheckErrors("Texture create fail");
		// transform matrix for Stack A interpolation
		h_aff12[0] = pixelSize[0] / pixelSize2[0], h_aff12[1] = 0, h_aff12[2] = 0, h_aff12[3] = 0;
		h_aff12[4] = 0, h_aff12[5] = pixelSize[1] / pixelSize2[1], h_aff12[6] = 0, h_aff12[7] = 0;
		h_aff12[8] = 0, h_aff12[9] = 0, h_aff12[10] = pixelSize[2] / pixelSize2[2], h_aff12[11] = 0;
		CopyTranMatrix(h_aff12, NDIM * sizeof(float));
		affineTransform(d_img3D, imx2, imy2, imz2, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]);
		UnbindTexture();
		cudaFreeArray(d_Array2);
		cudaMemcpy(h_StackB, d_img3D, totalSize2 * sizeof(float), cudaMemcpyDeviceToHost);
	}// after interpolation, Stack A size: imx x imy x imz;
	else
		memcpy(h_StackB, h_img2, totalSize2 * sizeof(float));
	cudaThreadSynchronize();
	cudaFree(d_img3D);
	int runStatus = 0;
	memcpy(h_prereg1, h_StackA, totalSize * sizeof(float));
	runStatus = alignsize3d(h_prereg2, h_StackB, imz, imy, imx, imz2, imy2, imx2,gpuMemMode);
	// ***** perform registration
	printf("Running registration ...\n");
	int affMethod = 7;
	switch (regChoice) {
	case 0:
		break;
	case 1:
		break;
	case 2:
		if (flagTmx)
			affMethod = 5;
		else
			affMethod = 7;
		break;
	case 3:
		flagTmx = false;
		affMethod = 7;
		break;
	case 4:
		flagTmx = false;
		affMethod = 7;
		break;
	default:
		printf("Wrong registration choice, processing stopped !!!\n");
		return 1;
	}
	float *regRecords = (float *)malloc(11 * sizeof(float));
	int regStatus = reg3d(h_reg, iTmx, h_prereg1, h_prereg2, &imSize[0], &imSize[0], regChoice, affMethod,
		flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	memcpy(fusionRecords, regRecords, 11 * sizeof(float));
	free(h_StackB);
	free(regRecords);
	if (regStatus != 0) {
		printf("Registration error, processing stopped !!!\n");
		return 1;
	}
	bool mStatus = checkmatrix(iTmx, imx, imy, imz);
	if (!mStatus) {
		regChoice = 2;
		regStatus = reg3d(h_reg, iTmx, h_img1, h_img2, imSize1, imSize2, regChoice, affMethod,
			flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	}
	
	
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
	// ***** Joint deconvolution
	float *deconRecords = (float *)malloc(10 * sizeof(float));
	int deconStatus =  decon_dualview(h_decon, h_prereg1, h_reg, &imSize[0], h_psf1, h_psf2,
		psfSizeIn, true, itNumForDecon, deviceNum, gpuMemMode, verbose, deconRecords, flagUnmatch, h_psf_bp1, h_psf_bp2);
	memcpy(&fusionRecords[11], deconRecords, 10 * sizeof(float));
	free(deconRecords);
	free(h_StackA);
	free(h_aff12);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	end = clock();
	deconRecords[21] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;	
}


//// 3D image operations
int alignsize3d(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2, int gpuMemMode) {
	int runStatus = 0;
	float *d_img1=NULL, *d_img2 = NULL;
	switch (gpuMemMode) {
	case 0:
		alignsize3Dgpu(h_odata, h_idata, sx, sy, sz, sx2, sy2, sz2);
		break;
	case 1:
		cudaMalloc((void **)&d_img1, sx*sy*sz * sizeof(float));
		cudaMalloc((void **)&d_img2, sx2*sy2*sz2 * sizeof(float));
		cudaMemcpy(d_img2, h_idata, sx2*sy2*sz2 * sizeof(float), cudaMemcpyHostToDevice);
		alignsize3Dgpu(d_img1, d_img2, sx, sy, sz, sx2, sy2, sz2);
		cudaMemcpy(h_odata, d_img1, sx*sy*sz * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_img1);
		cudaFree(d_img2);
		break;
	case 2:
		cudaMalloc((void **)&d_img1, sx*sy*sz * sizeof(float));
		cudaMalloc((void **)&d_img2, sx2*sy2*sz2 * sizeof(float));
		cudaMemcpy(d_img2, h_idata, sx2*sy2*sz2 * sizeof(float), cudaMemcpyHostToDevice);
		alignsize3Dgpu(d_img1, d_img2, sx, sy, sz, sx2, sy2, sz2);
		cudaMemcpy(h_odata, d_img1, sx*sy*sz * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_img1);
		cudaFree(d_img2);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, processing stopped !!! ****\n");
		return 1;
	}
	return runStatus;
}
int mp2dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj){
	// sizeMP: sx, sy, sy, sz, sz, sx
	int sx = sizeImg[0], sy = sizeImg[1], sz = sizeImg[2];
	int totalSizeImg = sx*sy*sz; 
	int totalSizeMP = sx*sy + sy*sz + sz*sx;
	float *d_img, *d_MP;
	cudaMalloc((void **)&d_img, totalSizeImg * sizeof(float));
	cudaMalloc((void **)&d_MP, totalSizeMP * sizeof(float));
	cudaMemset(d_MP, 0, totalSizeMP*sizeof(float));
	cudaMemcpy(d_img, h_img, totalSizeImg* sizeof(float), cudaMemcpyHostToDevice);

	if(flagZProj) maxprojection(d_MP, d_img, sx, sy, sz, 1);
	if(flagXProj) maxprojection(&d_MP[sx*sy], d_img, sx, sy, sz, 3);
	if (flagZProj) maxprojection(&d_MP[sx*sy+sy*sz], d_img, sx, sy, sz, 2);
	sizeMP[0] = sx; sizeMP[1] = sy; sizeMP[2] = sy; 
	sizeMP[3] = sz; sizeMP[4] = sz; sizeMP[5] = sx;
	cudaMemcpy(h_MP, d_MP, totalSizeMP * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_img);
	cudaFree(d_MP);
	return 0;
}

int mp3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum){
	//sizeMP: sx, imRotationy, projectNum, imRotationx, sy, projectNum
	int sx = sizeImg[0], sy = sizeImg[1], sz = sizeImg[2];
	int imRotationx = round(sqrt(sx*sx + sz*sz));
	int imRotationy = round(sqrt(sy*sy + sz*sz));
	float projectAng = 0;
	float projectStep = 3.14159 * 2 / projectNum;
	float *h_affRot = (float *)malloc(NDIM * sizeof(float));
	float *d_StackProject, *d_StackRotation;
	int totalSizeProjectX = sx * imRotationy;
	int totalSizeProjectY = imRotationx * sy;
	int totalSizeProjectMax = totalSizeProjectX > totalSizeProjectY ? totalSizeProjectX : totalSizeProjectY;
	int totalSizeRotationX = sx * imRotationy * imRotationy; 
	int totalSizeRotationY = imRotationx * sy * imRotationx;
	int totalSizeRotationMax;
	if (flagXaxis&&flagYaxis) 
		totalSizeRotationMax = totalSizeRotationX > totalSizeRotationY ? totalSizeRotationX : totalSizeRotationY;
	else if (flagXaxis) 
		totalSizeRotationMax = totalSizeRotationX;
	else if (flagYaxis)
		totalSizeRotationMax = totalSizeRotationY;
	else
		return -1;
	cudaMalloc((void **)&d_StackRotation, totalSizeRotationMax * sizeof(float));
	cudaMalloc((void **)&d_StackProject, totalSizeProjectMax * sizeof(float));
	cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDescT, make_cudaExtent(sx, sy, sz));
	cudacopyhosttoarray(d_Array, channelDescT, h_img, sx, sy, sz);
	BindTexture(d_Array, channelDescT);
	cudaCheckErrors("Texture create fail");
	if (flagXaxis){// 3D projection by X axis
		for (int iProj = 0; iProj < projectNum; iProj++){
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 1);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, sx, imRotationy, imRotationy, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, sx, imRotationy, imRotationy, 1);
			cudaMemcpy(&h_MP[totalSizeProjectX*iProj], d_StackProject, totalSizeProjectX * sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		sizeMP[0] = sx; sizeMP[1] = imRotationy; sizeMP[2] = projectNum;
	}

	if (flagYaxis){// 3D projection by Y axis
		int Ystart = sx * imRotationy * projectNum;
		// 3D projection by Y axis
		for (int iProj = 0; iProj < projectNum; iProj++){
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 2);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, imRotationx, sy, imRotationx, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, imRotationx, sy, imRotationx, 1);
			cudaMemcpy(&h_MP[Ystart + totalSizeProjectY*iProj], d_StackProject, totalSizeProjectY * sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		sizeMP[3] = imRotationx; sizeMP[4] = sy; sizeMP[5] = projectNum;
	}
	UnbindTexture();
	free(h_affRot);
	cudaFree(d_StackRotation);
	cudaFree(d_StackProject);
	cudaFreeArray(d_Array);

	return 0;
}

int mip3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, int rAxis, int projectNum) {
	// bool flagXaxis, bool flagYaxis
	// if rAxis == 1: X axis; if rAxis==2: Y axis
	//sizeMP: sx, imRotationy, projectNum, imRotationx, sy, projectNum
	long long int sx = sizeImg[0], sy = sizeImg[1], sz = sizeImg[2];
	long long int sr = 1, imRotation = 1, totalSizeProject = 1;
	if (rAxis == 1) {
		sr = sx;
		imRotation = round(sqrt(sy*sy + sz*sz));	
	}
	else if (rAxis == 2) {
		sr = sy;
		imRotation = round(sqrt(sx*sx + sz*sz));
	}
	else
		return -1;
	totalSizeProject = sr * imRotation;
	long long int totalSizeRotation = sr * imRotation * imRotation;
	//long long int totalSizeProjectStack = sr * imRotation * (long long)projectNum;
	
	float projectAng = 0;
	float projectStep = 3.14159 * 2 / projectNum;
	float *h_affRot = (float *)malloc(NDIM * sizeof(float));
	float *d_StackProject, *d_StackRotation;
	
	cudaMalloc((void **)&d_StackRotation, totalSizeRotation * sizeof(float));
	cudaMalloc((void **)&d_StackProject, totalSizeProject * sizeof(float));
	cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDescT, make_cudaExtent(sx, sy, sz));
	cudacopyhosttoarray(d_Array, channelDescT, h_img, sx, sy, sz);
	BindTexture(d_Array, channelDescT);
	cudaCheckErrors("Texture create fail");
	if (rAxis == 1) {// 3D projection by X axis
		for (int iProj = 0; iProj < projectNum; iProj++) {
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 1);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, sr, imRotation, imRotation, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, sr, imRotation, imRotation, 1);
			cudaMemcpy(&h_MP[totalSizeProject*iProj], d_StackProject, totalSizeProject * sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		sizeMP[0] = sr; sizeMP[1] = imRotation; sizeMP[2] = projectNum;
	}

	else if (rAxis == 2) {// 3D projection by Y axis
		// 3D projection by Y axis
		for (int iProj = 0; iProj < projectNum; iProj++) {
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 2);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, imRotation, sy, imRotation, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, imRotation, sy, imRotation, 1);
			cudaMemcpy(&h_MP[totalSizeProject*iProj], d_StackProject, totalSizeProject * sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		sizeMP[3] = imRotation; sizeMP[4] = sy; sizeMP[5] = projectNum;
	}
	UnbindTexture();
	free(h_affRot);
	cudaFree(d_StackRotation);
	cudaFree(d_StackProject);
	cudaFreeArray(d_Array);

	return 0;
}

#undef blockSize
#undef blockSize2Dx
#undef blockSize2Dy
#undef blockSize3Dx
#undef blockSize3Dy
#undef blockSize3Dz
#undef SMALLVALUE
#undef NDIM
