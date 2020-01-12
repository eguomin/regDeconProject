#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>       // va_*
#include <iostream>
#include <fstream>
#include <ctime>

extern "C" {
#include "libapi.h"
}

void helpmessage(char *appName, bool flagHelp) {
	printf("\n%s: intensity-based 3D image registration\n", appName);
	printf("\nUsage:\t%s -t <targetImageName> -s <sourceImageName> -o <outputImageName> [OPTIONS]\n", appName);
	if (!flagHelp) {
		printf("\nUse command for more details:\n\t%s -help or %s -h\n", appName, appName);
		return;
	}
	printf("\tOnly 16-bit or 32-bit standard TIFF images are currently supported.\n");
	printf("\n= = = [OPTIONS] = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
	printf("\t-t <filename>\t\tTarget image filename (Fixed or Base image) (mandatory)\n");
	printf("\t-s <filename>\t\tSource image filename (Moving or Floating image) (mandatory)\n");
	printf("\t-o <filename>\t\tOutput filename of the registered image (mandatory)\n");
	printf("\t-itmx <filename>\tInput tranformation matrix filename [identity matrix]\n");
	printf("\t-otmx <filename>\tOutput tranformation matrix filename [no output]\n");
	printf("\t-regc <int>\t\tOptions for registration choice [2]\n");
	printf("\t\t\t\t0: no registration, but transform source image based on input matrix\n");
	printf("\t\t\t\t1: phasor registraion (pixel-level translation, input matrix disabled)\n");
	printf("\t\t\t\t2: affine registration (with or without input matrix)\n");
	printf("\t\t\t\t3: phasor registration --> affine registration (input matrix disabled)\n");
	printf("\t\t\t\t4: 2D MIP registration --> affine registration (input matrix disabled)\n");
	printf("\t-affm <int>\t\tOptions for affine method [7]\n");
	printf("\t\t\t\t0: no affine, but transform source image based on input matrix\n");
	printf("\t\t\t\t1: translation only (3 DOF)\n");
	printf("\t\t\t\t2: rigid-body (6 DOF)\n");
	printf("\t\t\t\t3: 7 DOF (translation, rotation, scaling equally in 3 dimensions)\n");
	printf("\t\t\t\t4: 9 DOF (translation, rotation, scaling)\n");
	printf("\t\t\t\t5: directly 12 DOF\n");
	printf("\t\t\t\t6: rigid body (6 DOF) --> 12 DOF\n");
	printf("\t\t\t\t7: 3 DOF --> 6 DOF--> 9 DOF--> 12 DOF\n");
	printf("\t-ftol <float>\t\tTolerance or threshold of the stop point [0.0001]\n");
	printf("\t-it <int>\t\tMaximum iteration number [3000]\n");
	printf("\t-gm <int>\t\tChoose CPU or GPU processing [-1]\n");
	printf("\t\t\t\t-1: automatically choose\n");
	printf("\t\t\t\t0: all in CPU (currently dose not work)\n");
	printf("\t\t\t\t1: efficient GPU mode if enough GPU memory\n");
	printf("\t\t\t\t2: memory-saved GPU mode if insufficient GPU memroy\n");
	printf("\t-dev <int>\t\tSpecify the GPU device if multiple GPUs on board [0]\n");
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
	char *fileReg = "../Data/SPIMA_0_reg2.tif";
	float *iTmx = (float *)malloc(12 * sizeof(float));
	char *fileiTmx = "../Data/tmxIn.tmx";
	char *fileoTmx = "../Data/tmxOut.tmx";
	char *fileImg1 = "../Data/SPIMA_0_crop.tif";
	char *fileImg2 = "../Data/SPIMA_0_crop2.tif";
	unsigned int imSize1[3], imSize2[3];
	int regChoice = 2;
	int affMethod = 6;
	bool flagTmx = false;
	float FTOL = 0.0001;
	int itLimit = 3000;
	int deviceNum = 0;
	int gpuMemMode = -1;
	bool verbose = true;
	unsigned int bitPerSample;
	bool flagBitInput = true;
	float *records = (float *)malloc(11 * sizeof(float));

	bool flagoTmx = false;
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
		else if (strcmp(argv[i], "-t") == 0)
		{
			fileImg1 = argv[++i];
		}
		else if (strcmp(argv[i], "-s") == 0)
		{
			fileImg2 = argv[++i];
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			fileReg = argv[++i];
		}
		else if (strcmp(argv[i], "-itmx") == 0)
		{
			fileiTmx = argv[++i];
			flagTmx = true;
		}
		else if (strcmp(argv[i], "-otmx") == 0)
		{
			fileoTmx = argv[++i];
			flagoTmx = true;
		}
		else if (strcmp(argv[i], "-regc") == 0)
		{
			regChoice = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-affm") == 0)
		{
			affMethod = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-ftol") == 0)
		{
			FTOL = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-it") == 0)
		{
			itLimit = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-gm") == 0)
		{
			gpuMemMode = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-dev") == 0)
		{
			deviceNum = atoi(argv[++i]);
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
	printf("\tTarget (fixed) image path: %s\n", fileImg1);
	printf("\tSource (floating) image path: %s\n", fileImg2);
	printf("\tOutput (registered) image path: %s\n", fileReg);
	unsigned int bitPerSampleImg1 = gettifinfo(fileImg1, &imSize1[0]);
	unsigned int bitPerSampleImg2 = gettifinfo(fileImg2, &imSize2[0]);
	if (flagBitInput) bitPerSample = bitPerSampleImg1;
	printf("\tTarget (fixed) image size %d x %d x %d\n  ", imSize1[0], imSize1[1], imSize1[2]);
	printf("\tSource (floating) image size %d x %d x %d\n  ", imSize2[0], imSize2[1], imSize2[2]);
	printf("\tOutput (registered) image size %d x %d x %d\n  ", imSize1[0], imSize1[1], imSize1[2]);
	printf("... Paremeters:\n");
	if (flagTmx) {
		printf("\tInitial transformation matrix: %s\n", fileiTmx);
	}
	else {
		printf("\tInitial transformation matrix: Default\n");
	}
	if (flagoTmx) {
		printf("\tOutput transformation matrix: %s\n", fileoTmx);
	}
	else {
		printf("\tOutput transformation matrix: Default\n");
	}
	switch (regChoice) {
	case 0:
		printf("\tRegistration choice: no registration\n");
		break;
	case 1:
		printf("\tRegistration choice: phasor registration\n");
		break;
	case 2:
		printf("\tRegistration choice: affine registration\n");
		break;
	case 3:
		printf("\tRegistration choice: pahse registration --> affine registration\n");
		break;
	case 4:
		printf("\tRegistration choice: 2D registration --> affine registration\n");
		break;
	default:
		printf("\tWrong registration choice, processing stopped !!!\n");
		return 1;
	}
	if (regChoice >= 2) {
		switch (affMethod) {
		case 0:
			printf("\tAfine registration method: no registration\n");
			break;
		case 1:
			printf("\tAfine registration method: tanslation only\n");
			break;
		case 2:
			printf("\tAfine registration method: rigid body\n");
			break;
		case 3:
			printf("\tAfine registration method: 7 DOF\n");
			break;
		case 4:
			printf("\tAfine registration method: 9 DOF\n");
			break;
		case 5:
			printf("\tAfine registration method: 12 DOF\n");
			break;
		case 6:
			printf("\tAfine registration method: rigid body --> 12 DOF\n");
			break;
		case 7:
			printf("\tAfine registration method: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF\n");
			break;
		default:
			printf("\tWrong affine registration method, processing stopped !!!\n");
			return 1;
		}
	}
	printf("\tTolerance or threshold: %f\n", FTOL);
	printf("\tMaximum iteration number: %d\n", itLimit);
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
	if (flagBitInput) {
		printf("\tOutput image bit: %d bit, same as input image\n",bitPerSample);
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



	long long int totalSize1 = (long long int(imSize1[0])) * (long long int(imSize1[1])) * (long long int(imSize1[2]));
	long long int totalSize2 = (long long int(imSize2[0])) * (long long int(imSize2[1])) * (long long int(imSize2[2]));
	float *h_reg = (float *)malloc(totalSize1 * sizeof(float));
	float *h_img1 = (float *)malloc(totalSize1 * sizeof(float));
	float *h_img2 = (float *)malloc(totalSize2 * sizeof(float));

	memset(h_reg, 0, totalSize1 * sizeof(float));
	readtifstack(h_img1, fileImg1, &imSize1[0]);
	readtifstack(h_img2, fileImg2, &imSize2[0]);

	FILE *fTmxIn = NULL, *fTmxOut = NULL;
	if (flagTmx) {
		if (fexists(fileiTmx)) {
			fTmxIn = fopen(fileiTmx, "r");
			for (int j = 0; j<12; j++)
			{
				fscanf(fTmxIn, "%f", &iTmx[j]);
			}
			fclose(fTmxIn);
		}
		else {
			printf("***** Iput transformation matrix file does not exist: %s\n", fileiTmx);
			return 1;
		}		
	}


	int runStatus = -1;
	// /	
	time1 = clock();
	printf("=== Registration starting ...\n");
	runStatus = reg3d(h_reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
		flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, records);
	time2 = clock();
	printf("runStatus: %d\n", runStatus);

	printf("GPU mode: %d\n", int(records[0]));

	writetifstack(fileReg, h_reg, &imSize1[0], bitPerSample);

	// save matrix file
	if (flagoTmx) {
		fTmxOut = fopen(fileoTmx, "w");
		for (int j = 0; j<12; j++)
		{
			fprintf(fTmxOut, "%f\t", iTmx[j]);
			if ((j + 1) % 4 == 0)
				fprintf(fTmxOut, "\n");
		}
		fprintf(fTmxOut, "%f\t%f\t%f\t%f\n", 0.0, 0.0, 0.0, 1.0);
		fclose(fTmxOut);
	}

	//free CPU memory
	free(h_reg);
	free(h_img1);
	free(h_img2);
	free(records);

	end = clock();
	printf("\n****Time cost for  image reading/writing: %2.3f s\n", (float)(end - time2 + time1 - start) / CLOCKS_PER_SEC);
	printf("\n****Time cost for  registration: %2.3f s\n", (float)(time2 - time1) / CLOCKS_PER_SEC);
	printf("\n****Time cost for  whole processing: %2.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);
	return 0;
}