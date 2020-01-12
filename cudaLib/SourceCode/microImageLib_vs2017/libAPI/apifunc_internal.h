#pragma once
//// *********** Internal functions ********************** ////
#ifdef __CUDACC__
typedef double2 dComplex;
#else
typedef struct {
	double x;
	double y;
} dComplex;

#endif

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct {
	float x;
	float y;
} fComplex;

#endif
// note if complex.h has been included we use the C99 complex types
#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined (complex)
typedef double _Complex fftw_complex;
typedef float _Complex fftwf_complex;
#else
typedef double fftw_complex[2];
typedef float fftwf_complex[2];
#endif

int snapTransformSize(int dataSize);

// CPU functions: basic operations
template <class T>
double sumcpu(T *h_idata, size_t totalSize);

template <class T>
void addcpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize);

template <class T>
void addvaluecpu(T *h_odata, T *h_idata1, T h_idata2, size_t totalSize);

template <class T>
void subcpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize);

template <class T>
void multicpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize);

template <class T>
void divcpu(T *h_odata, T *h_idata1, T *h_idata2, size_t totalSize);

template <class T>
void multivaluecpu(T *h_odata, T *h_idata1, T h_idata2, size_t totalSize);

extern "C"
void multicomplexcpu(fComplex *h_odata, fComplex *h_idata1, fComplex *h_idata2, size_t totalSize);

template <class T>
void maxvaluecpu(T *h_odata, T *h_idata1, T h_idata2, size_t totalSize);

template <class T>
T max3Dcpu(size_t *corXYZ, T *h_idata, size_t sx, size_t sy, size_t sz);

template <class T>
void changestorageordercpu(T *h_odata, T *h_idata, size_t sx, size_t sy, size_t sz, int orderMode);

// GPU functions: basic operations
template <class T>
void add3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz);

template <class T>
void addvaluegpu(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz);

template <class T>
void sub3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz);

template <class T>
void multi3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz);

template <class T>
void multivaluegpu(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz);

extern "C"
void multicomplex3Dgpu(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, size_t sx, size_t sy, size_t sz);

extern "C"
void multicomplexnorm3Dgpu(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, size_t sx, size_t sy, size_t sz);

extern "C"
void multidcomplex3Dgpu(dComplex *d_odata, dComplex *d_idata1, dComplex *d_idata2, size_t sx, size_t sy, size_t sz);

template <class T>
void div3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz);

extern "C"
void conj3Dgpu(fComplex *d_odata, fComplex *d_idata, size_t sx, size_t sy, size_t sz);

template <class T> // reduction example from internet: sum for small data size
T sumgpu(T *d_idata, int totalSize);

template <class T> // costumized reduction: sum for huge data size (3D data)
double sum3Dgpu(T *d_idata, size_t sx, size_t sy, size_t sz);

template <class T> // costumized reduction: sum for small data size (1D data)
T sumgpu1D(T *d_idata, size_t totalSize);

template <class T>
T max3Dgpu(size_t *corXYZ, T *d_idata, size_t sx, size_t sy, size_t sz);

template <class T>
void maxvalue3Dgpu(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz);

template <class T>
void maxprojection(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int pDirection);

template <class T> // reorder data
void changestorageordergpu(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int orderMode);
// orderMode
//  1: change tiff storage order to C storage order
// -1: change C storage order to tiff storage order

template <class T> // rotate 3D data +/- 90 degrees by Y axis
void rotbyyaxis(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int rotDirection);
// rotDirection: 1: 90 deg; -1:- 90 deg;

// convert affine cofficients between vector (12) and matrix (3x4)
void p2matrix(float *m, float *x); // vector to matrix
void matrix2p(float *m, float *x); // matrix to vector
extern "C" void matrixmultiply(float * m, float *m1, float *m2); // multiply two affine cofficient matrixes (3x4)

									   // void rot3Dbyyaxis(float *d_odata, float theta, int sx, int sz, int sx2, int sz2);
									   // rotation to affine matrix
extern "C" void rot2matrix(float * p_out, float theta, long long int sx, long long int sy, long long int sz, int rotAxis);
// DOF to affine matrix
extern "C" void dof9tomatrix(float * p_out, float *p_dof, int dofNum); // 9 DOF to matrix (3x4)

template <class T> // shift on 3D data
void imshiftgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);

template <class T> // do sircular shift on 3D data
void circshiftgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz);

extern "C" void CopyTranMatrix(float *p, int dataSize); // copy affine matrix from host to device const

// copy data and generate gpu array format
template <class T> // from host to gpu array
void cudacopyhosttoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, T *h_idata, size_t sx, size_t sy, size_t sz);
template <class T> // from device to gpu array
void cudacopydevicetoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, T *h_idata, size_t sx, size_t sy, size_t sz);

// bind gpu array to texture memory
extern "C" // floating dat to tex
void BindTexture(cudaArray *d_Stack, cudaChannelFormatDesc channelDesc);
extern "C" // floating data to tex2: second texture variable
void BindTexture2(cudaArray *d_Stack, cudaChannelFormatDesc channelDesc);
extern "C" // unsigned short data to tex16
void BindTexture16(cudaArray *d_Stack, cudaChannelFormatDesc channelDesc);
//unbind texture memory
extern "C"
void UnbindTexture();
extern "C"
void UnbindTexture2();
extern "C"
void UnbindTexture16();
//fetch single element from texture memory: tex
extern "C"
void AccessTexture(float x, float y, float z);

// affine transformation from tex: floating or unsigned short format
template <class T>
void affineTransform(T *d_s, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2);

// calculate cross-correlation between d_t (target) and and d_s (source, same size with d_t)
extern "C"
float zncc0(float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz);
extern "C"
float zncc1(float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz);
extern "C"
float zncc2(float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz);

// GPU optimized correlation function: calculate cross-correlation between d_t (target) and and a transformed image from tex (source)
float corrfunc(float *d_t, float sd_t, float *aff, long long int sx,
	long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2);


// CPU functions for correlation calculation
float ilerp(float x, float x1, float x2, float q00, float q01);
float ibilerp(float x, float y, float x1, float x2, float y1, float y2, float q11, float q12, float q21, float q22);
float itrilerp(float x, float y, float z, float x1, float x2, float y1, float y2, float z1, float z2,
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222);

void affinetransformcpu(float *h_s, float *h_t, float *aff, int sx, int sy, int sz, int sx2, int sy2, int sz2);
double corrfunccpu(float *h_s, float *h_t, float *aff, int sx, int sy, int sz, int sx2, int sy2, int sz2);
double corrfunccpu2(float *h_s, float *h_t, float *aff, int sx, int sy, int sz, int sx2, int sy2, int sz2);

// *** 2D data affine transformation and cross-correlation: GPU
extern "C"
void affineTransform2D(float *d_t, int sx, int sy, int sx2, int sy2);

float corrfunc2D(float *d_t, float sd_t, float *aff, long long int sx, long long int sy, long long int sx2, long long int sy2);

extern "C" void BindTexture2D(cudaArray *d_Array, cudaChannelFormatDesc channelDesc);

extern "C" void UnbindTexture2D();

// CPU
template <class T>
void flipcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz);

template <class T>
void padPSFcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void padstackcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void cropcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template <class T>
void cropcpu2(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);

template <class T>
void alignsize3Dcpu(T *h_odata, T *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

extern "C"
void genOTFcpu(fftwf_complex *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, bool normFlag);
// GPU
template <class T>
void flipgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz);

template <class T>
void padPSFgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void padstackgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);

template <class T>
void cropgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
template <class T>
void cropgpu2(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz);

template <class T>
void alignsize3Dgpu(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
// 2D registration
float costfunc2D(float *x);
extern "C"
int affinetrans2d0(float *h_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sx2, long long int sy2);
extern "C"
int affinetrans2d1(float *h_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sx2, long long int sy2);
extern "C"
int reg2d_phasor0(long long int *shiftXY, float *h_img1, float *h_img2, long long int sx, long long int sy);
extern "C"
int reg2d_phasor1(long long int *shiftXY, float *d_img1, float *d_img2, long long int sx, long long int sy);
extern "C"
int reg2d_affine0(float *h_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx1, long long int sy1,
	long long int sx2, long long int sy2, int affMethod, bool flagTmx, float FTOL, int itLimit, float *records);
extern "C"
int reg2d_affine1(float *h_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx1, long long int sy1,
	long long int sx2, long long int sy2, int affMethod, bool flagTmx, float FTOL, int itLimit, float *records);

// 3D registration
float costfunc(float *x);
extern "C"
float zncc1(float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz);
extern "C"
float zncc1(float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz);
extern "C"
float zncc1(float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz);
extern "C"
int affinetrans3d0(float *h_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
extern "C"
int affinetrans3d1(float *d_odata, float *iTmx, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
extern "C"
int affinetrans3d2(float *d_odata, float *iTmx, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2);
extern "C"
int reg3d_affine0(float *h_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz,
	int affMethod, bool flagTmx, float FTOL, int itLimit, bool verbose, float *records);
extern "C"
int reg3d_affine1(float *d_reg, float *iTmx, float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz,
	int affMethod, bool flagTmx, float FTOL, int itLimit, bool verbose, float *records);
extern "C"
int reg3d_affine2(float *d_reg, float *iTmx, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz,
	int affMethod, bool flagTmx, float FTOL, int itLimit, bool verbose, float *records);

extern "C"
int reg3d_phasor0(long long int *shiftXYZ, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz);
extern "C"
int reg3d_phasor1(long long int *shiftXYZ, float *d_img1, float *d_img2, long long int sx, long long int sy, long long int sz);
extern "C"
int reg3d_phasor2(long long int *shiftXYZ, float *h_img1, float *h_img2, long long int sx, long long int sy, long long int sz);

extern "C"
void genOTFgpu(fComplex *d_odata, float *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, bool normFlag);
extern "C"
int decon_singleview_OTF0(float *h_decon, float *h_img, fftwf_complex *h_OTF, fftwf_complex *h_OTF_bp,
	long long int sx, long long int sy, long long int sz, int itNumForDecon, bool initialFlag);
extern "C"
int decon_singleview_OTF1(float *d_decon, float *d_img, fComplex *d_OTF, fComplex *d_OTF_bp,
	long long int sx, long long int sy, long long int sz, int itNumForDecon, bool initialFlag);
extern "C"
int decon_singleview_OTF2(float *d_decon, float *d_img, fComplex *h_OTF, fComplex *h_OTF_bp,
	long long int sx, long long int sy, long long int sz, int itNumForDecon, bool initialFlag);
extern "C"
int decon_dualview_OTF0(float *h_decon, float *h_img1, float *h_img2, fftwf_complex *h_OTF1, fftwf_complex *h_OTF2, fftwf_complex *h_OTF_bp1,
	fftwf_complex *h_OTF_bp2, long long int sx, long long int sy, long long int sz, int itNumForDecon, bool initialFlag);
extern "C"
int decon_dualview_OTF1(float *d_decon, float *d_img1, float *d_img2, fComplex *d_OTF1, fComplex *d_OTF2, fComplex *d_OTF_bp1,
	fComplex *d_OTF_bp2, long long int sx, long long int sy, long long int sz, int itNumForDecon, bool initialFlag);
extern "C"
int decon_dualview_OTF2(float *d_decon, float *d_img1, float *h_img2, fComplex *h_OTF1, fComplex *h_OTF2, fComplex *h_OTF_bp1,
	fComplex *h_OTF_bp2, long long int sx, long long int sy, long long int sz, int itNumForDecon, bool initialFlag);