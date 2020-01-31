//// *** API functions
//// file I/O
char* concat(int count, ...);
bool fexists(const char * filename);
unsigned short gettifinfo(char tifdir[], unsigned int *tifSize);
void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize);
void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample);

// 
void queryDevice();

//// 2D registration
int reg2d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records);
//// 3D affine transformation
// int atrans3d(float *h_out, float *iTmx, float *h_img, unsigned int *imSize1, unsigned int *imSize2, int deviceNum, int gpuMemMode, bool verbose, float *records);
int atrans3dgpu(float *h_out, float *iTmx, float *h_img, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);
int atrans3dgpu_16bit(unsigned short *h_out, float *iTmx, unsigned short *h_img, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

//// 3D registration
bool checkmatrix(float *iTmx, long long int sx, long long int sy, long long int sz);
int reg3d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice, int affMethod,
	bool inputTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records);

int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
	int inputTmx, float FTOL, int itLimit, int flagSubBg, int deviceNum, float *regRecords);

//// 3D deconvolution
// single view 
int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize, bool flagDeconInitial,
	int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp);
// dual view
int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, unsigned int *psfSize,
	bool flagDeconInitial, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

//// 3D fusion: dual view registration and deconvolution
int fusion_dualview(float *h_decon, float *h_reg, float *h_prereg1, float *h_prereg2, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, bool flagTmx, int regChoice, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

//// batch processing

//// maximum intensity projections: 
int mp2dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj);
int mp3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum);
int mip3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, int rAxis, int projectNum);

