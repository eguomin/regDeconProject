#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>       
#include <iostream>
#include <fstream>
#include <ctime>

extern "C" {
#include "libapi.h"
}


void helpmessage(char *appName, bool flagHelp) {
	printf("\n%s: single-view 3D image deconvolution\n", appName);
	printf("\nUsage:\t%s -i1 <inputImageName> -i2 <inputImageName> -fp1 <psfImageName> -fp2 <psfImageName>-o <outputImageName> [OPTIONS]\n", appName);
	if (!flagHelp) {
		printf("\nUse command for more details:\n\t%s -help or %s -h\n", appName, appName);
		return;
	}
	printf("\tOnly 16-bit or 32-bit standard TIFF images are currently supported.\n");
	printf("\n= = = [OPTIONS] = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
	printf("\t-i1 <filename>\t\tInput image filename (mandatory)\n");
	printf("\t-i2 <filename>\t\tInput image filename (mandatory)\n");
	printf("\t-fp1 <filename>\t\tPSF (forward projector) image filename (mandatory)\n");
	printf("\t-fp2 <filename>\t\tPSF (forward projector) image filename (mandatory)\n");
	printf("\t-o <filename>\t\tOutput filename of the deconvolved image (mandatory)\n");
	printf("\t-bp1 <filename>\t\tBackward projector filename [flip of PSF1]\n");
	printf("\t-bp2 <filename>\t\tBackward projector filename [flip of PSF2]\n");
	printf("\t-it <int>\t\tIteration number of the deconvolution [10]\n");
	printf("\t-gm <int>\t\tChoose CPU or GPU processing [-1]\n");
	printf("\t\t\t\t-1: automatically choose\n");
	printf("\t\t\t\t0: all in CPU (currently dose not work)\n");
	printf("\t\t\t\t1: efficient GPU mode if enough GPU memory\n");
	printf("\t\t\t\t2: memory-saved GPU mode if insufficient GPU memroy\n");
	printf("\t-dev <int>\t\tSpecify the GPU device if multiple GPUs on board [0]\n");
	printf("\t-cON or -cOFF\t\tON: constant as initialization; OFF: input image as initialization [OFF]\n");
	printf("\t-bit <int>\t\tSpecify output image bit: 16 or 32 [same as input image]\n");
	printf("\t-verbON or -verbOFF\tTurn on/off verbose information [ON]\n");
	printf("\t-log <filename>\t\tLog filename [no log] (currently dose not work)\n");
	return;
}

int main(int argc, char* argv[])
{
	if (argc == 1) {
		helpmessage(argv[0], false);
		return EXIT_SUCCESS;
	}

	char *filePSF1 = "../Data/PSFA.tif";
	char *filePSF2 = "../Data/PSFA.tif";
	char *filePSF1_bp = "../Data/PSFA_bp.tif";
	char *filePSF2_bp = "../Data/PSFB_bp.tif";
	char *fileImg1 = "../Data/SPIMA_0_crop.tif";
	char *fileImg2 = "../Data/SPIMB_0_crop.tif";
	char *fileDecon = "../Data/Decon_0.tif";
	unsigned int imSize[3], psfSize[3], tempSize[3];
	bool flagConstInitial = false;
	int itNumForDecon = 10; // decon it number
	int deviceNum = 0;
	int gpuMemMode = -1;
	bool verbose = true;
	bool flagUnmatch = false;
	unsigned int bitPerSample;
	bool flagBitInput = true;

	bool flagLog = false;
	// ****************** Processing Starts***************** //
	// *** variables for memory and time cost records
	clock_t start, time1, time2, end;
	start = clock();
	// *** get arguments
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			helpmessage(argv[0], true);
			return EXIT_SUCCESS;
		}
		else if (strcmp(argv[i], "-i1") == 0)
		{
			fileImg1 = argv[++i];
		}
		else if (strcmp(argv[i], "-i2") == 0)
		{
			fileImg2 = argv[++i];
		}
		else if (strcmp(argv[i], "-fp1") == 0)
		{
			filePSF1 = argv[++i];
		}
		else if (strcmp(argv[i], "-fp2") == 0)
		{
			filePSF2 = argv[++i];
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			fileDecon = argv[++i];
		}
		else if (strcmp(argv[i], "-bp1") == 0)
		{
			filePSF1_bp = argv[++i];
			flagUnmatch = true;
		}
		else if (strcmp(argv[i], "-bp2") == 0)
		{
			filePSF2_bp = argv[++i];
			flagUnmatch = true;
		}
		else if (strcmp(argv[i], "-it") == 0)
		{
			itNumForDecon = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-gm") == 0)
		{
			gpuMemMode = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-dev") == 0)
		{
			deviceNum = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-cON") == 0)
		{
			flagConstInitial = true;
		}
		else if (strcmp(argv[i], "-cOFF") == 0)
		{
			flagConstInitial = false;
		}
		else if (strcmp(argv[i], "-verbON") == 0)
		{
			verbose = true;
		}
		else if (strcmp(argv[i], "-verbOFF") == 0)
		{
			verbose = false;
		}
		else if (strcmp(argv[i], "-bit") == 0)
		{
			bitPerSample = atoi(argv[++i]);
			flagBitInput = false;
		}
		else if (strcmp(argv[i], "-log") == 0)
		{
			flagLog = true;
		}
		else if (strcmp(argv[i], "-log") == 0)
		{
			flagLog = false;
		}
	}

	printf("=====================================================\n");
	printf("=== Registration settings ...\n");
	printf("... Image information: \n");
	printf("\tInput image 1 path: %s\n", fileImg1);
	printf("\tInput image 2 path: %s\n", fileImg2);
	printf("\tPSF 1 (forward projector) image path: %s\n", filePSF1);
	printf("\tPSF 2 (forward projector) image path: %s\n", filePSF2);
	if (flagUnmatch) {
		printf("\tBackward projector 1 image path: %s\n", filePSF1_bp);
		printf("\tBackward projector 2 image path: %s\n", filePSF2_bp);
	}
	printf("\tOutput image path: %s\n", fileDecon);
	unsigned int bitPerSampleImg = gettifinfo(fileImg1, &imSize[0]);
	unsigned int bitPerSamplePSF = gettifinfo(filePSF1, &psfSize[0]);
	bitPerSampleImg = gettifinfo(fileImg2, &tempSize[0]);
	if ((imSize[0] != tempSize[0]) || (imSize[1] != tempSize[1]) || (imSize[2] != tempSize[2])) {
		printf("\tThe two input images don't have the same image size, processing stopped !!!\n");
		return 1;
	}
	bitPerSamplePSF = gettifinfo(filePSF2, &tempSize[0]);
	if ((psfSize[0] != tempSize[0]) || (psfSize[1] != tempSize[1]) || (psfSize[2] != tempSize[2])) {
		printf("\tThe two forward projectors don't have the same image size, processing stopped !!!\n");
		return 1;
	}

	if (flagUnmatch) {
		bitPerSamplePSF = gettifinfo(filePSF1_bp, &tempSize[0]);
		if ((psfSize[0] != tempSize[0]) || (psfSize[1] != tempSize[1]) || (psfSize[2] != tempSize[2])) {
			printf("\tForward projector and backward projector don't have the same image size, processing stopped !!!\n");
			return 1;
		}
		bitPerSamplePSF = gettifinfo(filePSF2_bp, &tempSize[0]);
		if ((psfSize[0] != tempSize[0]) || (psfSize[1] != tempSize[1]) || (psfSize[2] != tempSize[2])) {
			printf("\tForward projector and backward projector don't have the same image size, processing stopped !!!\n");
			return 1;
		}
	}
	if (flagBitInput) bitPerSample = bitPerSampleImg;
	printf("\tInput image size %d x %d x %d\n  ", imSize[0], imSize[1], imSize[2]);
	printf("\tPSF image size %d x %d x %d\n  ", psfSize[0], psfSize[1], psfSize[2]);
	printf("\tOutput image size %d x %d x %d\n  ", imSize[0], imSize[1], imSize[2]);
	printf("... Paremeters:\n");
	if (flagUnmatch) {
		printf("\tUse unmatched backward projector: yes\n");
	}
	printf("\tIteration number of the deconvolution: %d\n", itNumForDecon);
	switch (gpuMemMode) {
	case -1:
		printf("\tCPU or GPU processing: automatically setting\n");
		printf("\tPotential GPU device number: %d\n", deviceNum);
		break;
	case 0:
		printf("\tCPU or GPU processing: CPU\n");
		break;
	case 1:
		printf("\tCPU or GPU processing: efficient GPU\n");
		printf("\tGPU device number: %d\n", deviceNum);
		break;
	case 2:
		printf("\tCPU or GPU processing: memory-saved GPU\n");
		printf("\tGPU device number: %d\n", deviceNum);
		break;
	default:
		printf("\tWrong GPU mode setting, processing stopped !!!\n");
		return 1;
	}
	if (flagConstInitial) {
		printf("\tInitialization of the deconvolution: constant mean of the input image\n");
	}
	else {
		printf("\tInitialization of the deconvolution: the input image\n");
	}
	if (flagBitInput) {
		printf("\tOutput image bit: %d bit, same as input image\n", bitPerSample);
	}
	else {
		printf("\tOutput image bit: %d bit\n", bitPerSample);
	}
	if (verbose) {
		printf("\tverbose information: true\n");
	}
	else {
		printf("\tverbose information: false\n");
	}
	printf("=====================================================\n\n");

	long long int totalSize = (long long int(imSize[0])) * (long long int(imSize[1])) * (long long int(imSize[2]));
	long long int totalSizePSF = (long long int(psfSize[0])) * (long long int(psfSize[1])) * (long long int(psfSize[2]));
	float *h_decon = (float *)malloc(totalSize * sizeof(float));
	float *h_img1 = (float *)malloc(totalSize * sizeof(float));
	float *h_img2 = (float *)malloc(totalSize * sizeof(float));
	float *h_psf1 = (float *)malloc(totalSizePSF * sizeof(float));
	float *h_psf2 = (float *)malloc(totalSizePSF * sizeof(float));
	float *h_psf1_bp = (float *)malloc(totalSizePSF * sizeof(float));
	float *h_psf2_bp = (float *)malloc(totalSizePSF * sizeof(float));
	memset(h_decon, 0, totalSize * sizeof(float));
	readtifstack(h_img1, fileImg1, &imSize[0]);
	readtifstack(h_img2, fileImg2, &imSize[0]);
	readtifstack(h_psf1, filePSF1, &psfSize[0]);
	readtifstack(h_psf2, filePSF2, &psfSize[0]);
	if (flagUnmatch) {
		readtifstack(h_psf1_bp, filePSF1_bp, &tempSize[0]);
		readtifstack(h_psf2_bp, filePSF2_bp, &tempSize[0]);
	}
	// /	
	time1 = clock();
	int runStatus = -1;
	float *deconRecords = (float *)malloc(20 * sizeof(float));
	printf("=== Deconvolution starting ...\n");
	runStatus = decon_dualview(h_decon, h_img1, h_img2, &imSize[0], h_psf1, h_psf2, &psfSize[0], flagConstInitial,
		itNumForDecon, deviceNum, gpuMemMode, verbose, deconRecords, flagUnmatch, h_psf1_bp, h_psf2_bp);
	time2 = clock();
	printf("runStatus: %d\n", runStatus);
	printf("GPU mode: %d\n", int(deconRecords[0]));

	writetifstack(fileDecon, h_decon, &imSize[0], bitPerSample);

	//free CPU memory
	free(h_decon);
	free(h_img1);
	free(h_img2);
	free(h_psf1);
	free(h_psf2);
	free(h_psf1_bp);
	free(h_psf2_bp);
	free(deconRecords);

	end = clock();
	printf("\n****Time cost for  image reading/writing: %2.3f s\n", (float)(end - time2 + time1 - start) / CLOCKS_PER_SEC);
	printf("\n****Time cost for  deconvolution: %2.3f s\n", (float)(time2 - time1) / CLOCKS_PER_SEC);
	printf("\n****Time cost for  whole processing: %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;
}

