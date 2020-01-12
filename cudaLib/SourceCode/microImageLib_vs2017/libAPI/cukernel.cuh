#define blockSize 1024
#define blockSize2Dx 32
#define blockSize2Dy 32
#define blockSize3Dx 16
#define blockSize3Dy 8
#define blockSize3Dz 8

#ifdef __CUDACC__
typedef double2 dComplex;
#else
typedef struct{
	double x;
	double y;
} dComplex;

#endif

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct{
	float x;
	float y;
} fComplex;

#endif

//template <class T>
texture<float, 3, cudaReadModeElementType> tex, tex2; // declare texture

texture<unsigned short, 3, cudaReadModeElementType> tex16; // declare texture

__constant__ float d_aff[12]; // 3x4 Affine transform: constant array

texture<float, 2, cudaReadModeElementType> tex2D1; // declare texture

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

// Basic math functions kernels
template <class T>
__global__ void
add3Dkernel(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk] = d_idata1[ijk] + d_idata2[ijk];
	}

}

template <class T>
__global__ void
addvaluekernel(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk] = d_idata1[ijk] + d_idata2;
	}

}

template <class T>
__global__ void
sub3Dkernel(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk] = d_idata1[ijk] - d_idata2[ijk];
	}

}

template <class T>
__global__ void
multi3Dkernel(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk] = d_idata1[ijk] * d_idata2[ijk];
	}

}

template <class T>
__global__ void
multivaluekernel(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk] = d_idata1[ijk] * d_idata2;
	}

}

__global__ void
multicomplex3Dkernel(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		fComplex a = d_idata1[ijk];
		fComplex b = d_idata2[ijk];
		d_odata[ijk].x = a.x*b.x - a.y*b.y;
		d_odata[ijk].y = a.x*b.y + a.y*b.x;
	}

}

__global__ void
multicomplexnorm3Dkernel(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		fComplex a = d_idata1[ijk];
		fComplex b = d_idata2[ijk];
		float c = a.x*b.x - a.y*b.y;
		float d = a.x*b.y + a.y*b.x;
		float e = sqrt(c*c + d*d);
		if (e != 0){
			d_odata[ijk].x = c/e;
			d_odata[ijk].y = d/e;
		}
		else{
			d_odata[ijk].x = 0;
			d_odata[ijk].y = 0;
		}
		
	}

}

__global__ void
multidcomplex3Dkernel(dComplex *d_odata, dComplex *d_idata1, dComplex *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		dComplex a = d_idata1[ijk];
		dComplex b = d_idata2[ijk];
		d_odata[ijk].x = a.x*b.x - a.y*b.y;
		d_odata[ijk].y = a.x*b.y + a.y*b.x;
	}

}

template <class T>
__global__ void
div3Dkernel(T *d_odata, T *d_idata1, T *d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		//if (d_idata2[ijk] == 0) d_idata2[ijk] = 1e-10;
		d_odata[ijk] = d_idata1[ijk] / d_idata2[ijk];
	}

}

__global__ void
conj3Dkernel(fComplex *d_odata, fComplex *d_idata, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk].x = d_idata[ijk].x;
		d_odata[ijk].y = -d_idata[ijk].y;
	}

}

template <class T>
__global__ void
sumgpukernel(T *g_idata, T *g_temp, int n, bool nIsPow2)
{	// this function is downloaded from internet
	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T *smem = sdata;

		if (blockSize >= 64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >= 32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >= 16)
		{
			smem[tid] = mySum = mySum + smem[tid + 8];
		}

		if (blockSize >= 8)
		{
			smem[tid] = mySum = mySum + smem[tid + 4];
		}

		if (blockSize >= 4)
		{
			smem[tid] = mySum = mySum + smem[tid + 2];
		}

		if (blockSize >= 2)
		{
			smem[tid] = mySum = mySum + smem[tid + 1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_temp[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void
sumgpu1Dkernel(T *d_idata, T *d_temp, size_t totalSize)
{
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t tempDataSize = 5 * blockSize;
	const size_t sz = (totalSize % tempDataSize != 0) ? (totalSize / tempDataSize + 1) : (totalSize / tempDataSize);
	T mySum = 0;

	for (size_t k = 0; k < sz; k++)
	{
		if ((i + k*tempDataSize)<totalSize)
			mySum += d_idata[i + k*tempDataSize];
	}

	// each thread puts its local sum into shared memory
	d_temp[i] = mySum;
}

template <class T>
__global__ void
reduceZ(T *d_idata, double *d_temp, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < sx && j < sy){
		size_t zStep = sx*sy;
		d_temp[i + sx * j] = 0;
		for (size_t k = 0; k < sz; k++)
			d_temp[i + sx * j] += (double)d_idata[i + sx * j + zStep * k];
	}
}

template <class T>
__global__ void
maxZkernel(T *d_idata, T *d_temp1, size_t *d_temp2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < sx && j < sy){
		T p = d_idata[i + sx * j];
		size_t z0 = 0;
		size_t zStep = sx*sy;
		for (size_t k = 0; k < sz; k++)
			if (p < d_idata[i + sx * j + zStep * k]){
				p = d_idata[i + sx * j + zStep * k];
				z0 = k;
			}
		d_temp1[i + sx * j] = p;
		d_temp2[i + sx * j] = z0;
	}
}

template <class T>
__global__ void
maxvalue3Dgpukernel(T *d_odata, T *d_idata1, T d_idata2, size_t sx, size_t sy, size_t sz){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		size_t ijk = i + sx * j + sx * sy * k;
		d_odata[ijk] = (d_idata1[ijk] > d_idata2) ? d_idata1[ijk] : d_idata2;
	}

}

template <class T>
__global__ void
maxprojectionkernel(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, size_t psx, size_t psy, size_t psz, int pDirection){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < psx && j < psy){
		size_t zStep = sx*sy;
		T a = 0;
		if (pDirection == 1){
			for (size_t k = 0; k < psz; k++)
				a = (a>d_idata[i + sx*j + zStep*k]) ? a : d_idata[i + sx*j + zStep*k];
			d_odata[i + j*psx] = a;
		}
		if (pDirection == 2){
			for (size_t k = 0; k < psz; k++)
				a = (a>d_idata[j + sx*k + zStep*i]) ? a : d_idata[j + sx*k + zStep*i];
			d_odata[i + j*psx] = a;
		}
		if (pDirection == 3){
			for (size_t k = 0; k < psz; k++)
				a = (a>d_idata[k + sx*i + zStep*j]) ? a : d_idata[k + sx*i + zStep*j];
			d_odata[i + j*psx] = a;
		}
	}
}

///////////////////////////////////
// other functions
template <class T>
__global__ void changestorageordergpukernel(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int orderMode){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		if (orderMode==1)//change tiff storage order to C storage order: 
			//output[i][j][k] = input[k][j][i]
			d_odata[i*sy*sz + j*sz + k] = d_idata[k*sy*sx + j*sx + i];
		else if (orderMode == -1)//change C storage order to tiff storage order: 
			//output[k][j][i] = input[i][j][k]
			d_odata[k*sy*sx + j*sx + i] = d_idata[i*sy*sz + j*sz + k];
	}
}

template <class T>
__global__ void rotbyyaxiskernel(T *d_odata, T *d_idata, size_t sx, size_t sy, size_t sz, int rotDirection){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		if (rotDirection == 1)//rotate 90 deg around Y axis 
			//output[sz-k-1][j][i] = input[i][j][k]
			d_odata[(sz-k-1)*sx*sy + j*sx + i] = d_idata[i*sy*sz + j*sz + k];
		else if(rotDirection == -1)//rotate -90 deg around Y axis
			//output[k][j][sx - i - 1] = input[i][j][k]
			d_odata[k*sy*sx + j*sx + (sx-i-1)] = d_idata[i*sy*sz + j*sz + k];
	}
}


template <class T>
__global__ void circshiftgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz){
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x < sx && y < sy && z < sz){
		long long int tx, ty, tz;
		tx = x - dx; ty = y - dy; tz = z -dz;
		if (tx < 0) tx += sx;
		if (ty < 0) ty += sy;
		if (tz < 0) tz += sz;
		if (tx >= sx) tx = tx - sx;
		if (ty >= sy) ty = ty - sy;
		if (tz >= sz) tz = tz - sz;
		//output[z][y][x] = input[tz][ty][tx]
		d_odata[z*sy*sx + y*sx + x] = d_idata[tz*sy*sx + ty*sx + tx];
	}
}

template <class T>
__global__ void imshiftgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int dx, long long int dy, long long int dz) {
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x < sx && y < sy && z < sz) {
		long long int tx, ty, tz;
		tx = x - dx; ty = y - dy; tz = z - dz;
		//output[z][y][x] = input[tz][ty][tx]
		if ((tx < 0) || (tx >= sx) || (ty < 0) || (ty >= sy) || (tz < 0) || (tz >= sz))
			d_odata[z*sy*sx + y*sx + x] = 0;
		else
			d_odata[z*sy*sx + y*sx + x] = d_idata[tz*sy*sx + ty*sx + tx];
	}
}


__global__ void accesstexturekernel(float x, float y, float z){
	float xi = x + (float)threadIdx.x;
	float yi = y + (float)threadIdx.y;
	float zi = z + (float)threadIdx.z;
	float testValue = tex3D(tex, xi, yi, zi);//by using this the error occurs
	printf("Coordinates: %f,%f,%f, value: %f\n", xi, yi, zi, testValue);
}

template <class T>
__global__ void affinetransformkernel(T *d_t, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2){
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int z = blockDim.z * blockIdx.z + threadIdx.z;
	// coordinates transformation
	if (x < sx && y < sy && z < sz){
		float ix = (float)x;
		float iy = (float)y;
		float iz = (float)z;
		float tx = d_aff[0] * ix + d_aff[1] * iy + d_aff[2] * iz + d_aff[3]+0.5;
		float ty = d_aff[4] * ix + d_aff[5] * iy + d_aff[6] * iz + d_aff[7]+0.5;
		float tz = d_aff[8] * ix + d_aff[9] * iy + d_aff[10] * iz + d_aff[11]+0.5;
		// texture interpolation
		// d_Stack[x*imy*imz + y*imz + z] = tex3D(tex, tx, ty, tz); //d_Stack[k][j][i] = tex3D[i][j][k]
		if (tx >= 0 && tx < sx2 && ty >= 0 && ty < sy2 && tz >= 0 && tz < sz2){
			if (sizeof(T) == 2)
				d_t[x + y*sx + z*sx*sy] = tex3D(tex16, tx, ty, tz); // d_Stack[i][j][k] = tex3D[i][j][k], Target image in texture
			else
				d_t[x + y*sx + z*sx*sy] = tex3D(tex, tx, ty, tz); // d_Stack[i][j][k] = tex3D[i][j][k], Target image in texture
		}
		else
			d_t[x + y*sx + z*sx*sy] = 0;
	}
}

__global__ void corrkernel(float *d_t, double *d_temp1, double *d_temp2, long long int sx, long long int sy, 
	long long int sz, long long int sx2, long long int sy2, long long int sz2){
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	long long int z;
	float t, s;
	double ss = 0, st = 0;
	// coordinates transformation
	if (x < sx && y < sy){
		for (z = 0; z < sz; z++){
			float ix = (float)x;
			float iy = (float)y;
			float iz = (float)z;
			float tx = d_aff[0] * ix + d_aff[1] * iy + d_aff[2] * iz + d_aff[3] + 0.5;
			float ty = d_aff[4] * ix + d_aff[5] * iy + d_aff[6] * iz + d_aff[7] + 0.5;
			float tz = d_aff[8] * ix + d_aff[9] * iy + d_aff[10] * iz + d_aff[11] + 0.5;
			if (tx>0 && tx < sx2 && ty>0 && ty < sy2 && tz>0 && tz < sz2)
				s = tex3D(tex, tx, ty, tz); // d_Stack[i][j][k] = tex3D[i][j][k], Target image in texture
			else
				s = 0.0001;
			t = d_t[x + y*sx + z*sx*sy];
			ss += (double)s*s;
			st += (double)s*t;
		}
		d_temp1[x + y*sx] = ss;
		d_temp2[x + y*sx] = st;
	}
}

__global__ void affineTransform2Dkernel(float *d_t, int sx, int sy, int sx2, int sy2){
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	// coordinates transformation
	if (x < sx && y < sy){
		float ix = (float)x;
		float iy = (float)y;
		float tx = d_aff[0] * ix + d_aff[1] * iy + d_aff[2] + 0.5;
		float ty = d_aff[3] * ix + d_aff[4] * iy + d_aff[5] + 0.5;
		if (tx>0 && tx < sx2 && ty>0 && ty < sy2)
			d_t[x + y*sx] = tex2D(tex2D1, tx, ty);
		else
			d_t[x + y*sx] = 0;
		
	}
}

__global__ void corr2Dkernel(float *d_s, float *d_sqr, float *d_corr, int sx, int sy, int sx2, int sy2){
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	float t, s;
	// coordinates transformation
	if (x < sx && y < sy){
		float ix = (float)x;
		float iy = (float)y;
		//float tx = d_aff[0] * ix + d_aff[1] * iy + d_aff[2];
		//float ty = d_aff[3] * ix + d_aff[4] * iy + d_aff[5] ;
		float tx = d_aff[0] * ix + d_aff[1] * iy + d_aff[2] + 0.5;
		float ty = d_aff[3] * ix + d_aff[4] * iy + d_aff[5] + 0.5;
		// texture interpolation
		if (tx>0 && tx < sx2 && ty>0 && ty < sy2)
			t = tex2D(tex2D1, tx, ty);
		else
			t = 0;
		s = d_s[x + y*sx];
		d_sqr[x + y*sx] = t*t;
		d_corr[x + y*sx] = s * t;
	}
}

template <class T>
__global__ void flipPSFkernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz){
	const long long int i = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int j = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		//h_flippedPSFA[i][j][k] = h_PSFA[PSFx-i-1][PSFy-j-1][PSFz-k-1]
		d_odata[i*sy*sz + j*sz + k] = d_idata[(sx - i - 1) *sy*sz + (sy - j - 1)*sz + (sz - k - 1)];
	}
}

template <class T>
__global__ void padPSFKernel(T *d_PaddedPSF, T *d_PSF, long long int FFTx, long long int FFTy, long long int FFTz, long long int PSFx, 
	long long int PSFy, long long int PSFz, long long int PSFox, long long int PSFoy, long long int PSFoz){
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x < PSFx && y < PSFy && z < PSFz){
		long long int dx, dy, dz;
		dx = x - PSFox; dy = y - PSFoy; dz = z - PSFoz;
		if (dx < 0) dx += FFTx;
		if (dy < 0) dy += FFTy;
		if (dz < 0) dz += FFTz;
		//d_PaddedPSF[dx][dy][dz] = d_PSF[x][y][z]
		if (dx >= 0 && dx < FFTx && dy >= 0 && dy < FFTy && dz >= 0 && dz < FFTz)
			d_PaddedPSF[dx*FFTy*FFTz + dy*FFTz + dz] = d_PSF[x*PSFy*PSFz + y*PSFz + z];
	}
}

template <class T>
__global__ void padStackKernel(T *d_PaddedStack, T *d_Stack, long long int FFTx, long long int FFTy, long long int FFTz, long long int sx, 
	long long int sy, long long int sz, long long int imox, long long int imoy, long long int imoz){
	const long long int dx = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int dy = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int dz = blockDim.z * blockIdx.z + threadIdx.z;
	if (dx < FFTx && dy < FFTy && dz < FFTz){
		long long int x, y, z;
		if (dx < imox){
			x = 0;
		}
		if (dy < imoy){
			y = 0;
		}
		if (dz < imoz){
			z = 0;
		}
		if (dx >= imox && dx < (imox + sx)){
			x = dx - imox;
		}
		if (dy >= imoy && dy < (imoy + sy)){
			y = dy - imoy;
		}
		if (dz >= imoz && dz < (imoz + sz)){
			z = dz - imoz;
		}
		if (dx >= (imox + sx)){
			x = sx - 1;
		}
		if (dy >= (imoy + sy)){
			y = sy - 1;
		}
		if (dz >= (imoz + sz)){
			z = sz - 1;
		}
		d_PaddedStack[dx*FFTy*FFTz + dy*FFTz + dz] = d_Stack[x*sy*sz + y*sz + z];
	}
}

// ***** new functions
template <class T>
__global__ void flipgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz){
	const long long int i = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int j = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		//d_odata[k][j][i] = d_idata[sz-k-1][sy-j-1][sx-i-1]
		//d_odata[k*sy*sx + j*sx + i] = d_idata[(sz - k - 1) *sy*sx + (sy - j - 1)*sx + (sx - i - 1)];
		d_odata[i*sy*sz + j*sz + k] = d_idata[(sx - i - 1) *sy*sz + (sy - j - 1)*sz + (sz - k - 1)];
	}
}

template <class T>
__global__ void padPSFgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2, 
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz){
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x < sx2 && y < sy2 && z < sz2){
		long long int dx, dy, dz;
		dx = x - sox; dy = y - soy; dz = z - soz;
		if (dx < 0) dx += sx;
		if (dy < 0) dy += sy;
		if (dz < 0) dz += sz;
		//d_PaddedPSF[dz][dy][dx] = d_PSF[z][y][x]
		if (dx >= 0 && dx < sx && dy >= 0 && dy < sy && dz >= 0 && dz < sz){
			//d_odata[dz*sy*sx + dy*sx + dx] = d_idata[z*sy2*sx2 + y*sx2 + x];
			d_odata[dx*sy*sz + dy*sz + dz] = d_idata[x*sy2*sz2 + y*sz2 + z];
		}
	}
}

template <class T>
__global__ void padstackgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2, 
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz){
	const long long int dx = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int dy = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int dz = blockDim.z * blockIdx.z + threadIdx.z;
	if (dx < sx && dy < sy && dz < sz){
		long long int x, y, z;
		if (dx < sox){
			x = 0;
		}
		if (dy < soy){
			y = 0;
		}
		if (dz < soz){
			z = 0;
		}
		if (dx >= sox && dx < (sox + sx2)){
			x = dx - sox;
		}
		if (dy >= soy && dy < (soy + sy2)){
			y = dy - soy;
		}
		if (dz >= soz && dz < (soz + sz2)){
			z = dz - soz;
		}
		if (dx >= (sox + sx2)){
			x = sx2 - 1;
		}
		if (dy >= (soy + sy2)){
			y = sy2 - 1;
		}
		if (dz >= (soz + sz2)){
			z = sz2 - 1;
		}
		//d_odata[dz*sy*sx + dy*sx + dx] = d_idata[z*sy2*sx2 + y*sx2 + x];
		d_odata[dx*sy*sz + dy*sz + dz] = d_idata[x*sy2*sz2 + y*sz2 + z];
	}
}

template <class T>
__global__ void cropgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2, 
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz){
	const long long int x = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int y = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x < sx2 && y < sy2 && z < sz2){
		long long int dx, dy, dz;
		dx = sox + x; dy = soy + y; dz = soz + z;
		//d_odata[z*sy*sx + y*sx + x] = d_idata[dz*sy2*sx2 + dy*sx2 + dx];
		d_odata[x*sy*sz + y*sz + z] = d_idata[dx*sy2*sz2 + dy*sz2 + dz];
	}
}
template <class T>
__global__ void alignsize3Dgpukernel(T *d_odata, T *d_idata, long long int sx, long long int sy, long long int sz, long long int sx2,
	long long int sy2, long long int sz2, long long int sox, long long int soy, long long int soz) {
	const long long int dx = blockDim.x * blockIdx.x + threadIdx.x;
	const long long int dy = blockDim.y * blockIdx.y + threadIdx.y;
	const long long int dz = blockDim.z * blockIdx.z + threadIdx.z;
	if (dx < sx && dy < sy && dz < sz) {
		long long int x, y, z;
		x = dx - sox;
		y = dy - soy;
		z = dz - soz;
		if ((x < 0) || (y < 0) || (z < 0) || (x >= sx2) || (y >= sy2) || (z >= sz2))
			d_odata[dx*sy*sz + dy*sz + dz] = 0;
		else
			d_odata[dx*sy*sz + dy*sz + dz] = d_idata[x*sy2*sz2 + y*sz2 + z];
	}
}
