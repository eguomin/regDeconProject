#ifndef _POWELL_H_
#define _POWELL_H_

void nrerror(char error_text[]);
float *vector(long nl, long nh);
float **matrix(long nrl, long nrh, long ncl, long nch);
void free_vector(float *v, long nl, long nh);
void free_matrix(float **m, long nrl, long nrh, long ncl, long nch);

float brent(float ax, float bx, float cx,
	float(*f)(float), float tol, float *xmin);
float f1dim(float x);
void linmin(float p[], float xi[], int n, float *fret,
	float(*func)(float[]));
void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb,
	float *fc, float(*func)(float));
void powell(float p[], float **xi, int n, float ftol, int *iter, float *fret,
	float(*func)(float[]), int *totalIt, int itLimit);

#endif /* _NR_UTILS_H_ */