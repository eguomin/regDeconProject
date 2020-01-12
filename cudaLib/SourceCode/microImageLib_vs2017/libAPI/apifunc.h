#pragma once
//// *********** API functions ********************** ////
//// file I/O
extern "C"
__declspec(dllexport) char* concat(int count, ...);
extern "C"
__declspec(dllexport) bool fexists(const char * filename);
extern "C"
__declspec(dllexport) unsigned short gettifinfo(char tifdir[], unsigned int *tifSize);
extern "C"
__declspec(dllexport) void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize);
extern "C"
__declspec(dllexport) void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample);
extern "C"
//__declspec(dllexport) void readtifstack_16to16(unsigned short *h_Image, char tifdir[], unsigned int *imsize);
//extern "C"
//__declspec(dllexport) void writetifstack_16to16(char tifdir[], unsigned short *h_Image, unsigned int *imsize);

/*
template <class T>
__declspec(dllexport) void readtifstack(T *h_Image, char tifdir[], unsigned int *imsize);

template <class T>
__declspec(dllexport) void writetifstack(char tifdir[], T *h_Image, unsigned int *imsize, unsigned short bitPerSample);
*/

// Query GPU device
extern "C"
__declspec(dllexport) void queryDevice();

//// 2D registration
extern "C"
__declspec(dllexport) int reg2d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records);

//// 3D affine transformation
extern "C"
__declspec(dllexport) bool checkmatrix(float *iTmx, long long int sx, long long int sy, long long int sz);

extern "C"
__declspec(dllexport) int atrans3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

extern "C"
__declspec(dllexport) int atrans3dgpu_16bit(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

//// 3D registration
extern "C"
__declspec(dllexport) int reg3d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice, int regMethod,
	bool inputTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records);

extern "C"
__declspec(dllexport) int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords);

//// 3D deonvolution
extern "C"
__declspec(dllexport) int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize, bool initialFlag,
int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp);

extern "C"
__declspec(dllexport) int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, unsigned int *psfSize,
bool initialFlag, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

//// 3D fusion: registration and deconvolution
extern "C"
__declspec(dllexport) int fusion_dualview(float *h_decon, float *h_reg, float *h_prereg1, float *h_prereg2, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, bool flagTmx, int regChoice, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);
	
//// maximum intensity projectoions:
extern "C"
__declspec(dllexport) int mp2dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj);

extern "C"
__declspec(dllexport) int mp3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum);

extern "C"
__declspec(dllexport)
int mip3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, int rAxis, int projectNum);

extern "C"
__declspec(dllexport)
int alignsize3d(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2, int gpuMemMode);