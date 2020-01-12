/* *** a collection of functions for powell searching method ****
*  source code oringed/modified from Cambridge - Numerical Recipes in C_ The Art of Scientific Computing, 2nd Ed
*  https://www.astro.umd.edu/~ricotti/NEWWEB/teaching/ASTR415/InClassExamples/NR3/license.htm
*/
// 
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "powell.h"

#define NR_END 1
#define FREE_ARG char*
void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v = (float *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v - nl + NR_END;
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((FREE_ARG)(v + nl - NR_END));
}

float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	float **m;

	/* allocate pointers to rows */
	m = (float **)malloc((size_t)((nrow + NR_END) * sizeof(float*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (float *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(float)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
	free((FREE_ARG)(m[nrl] + ncl - NR_END));
	free((FREE_ARG)(m + nrl - NR_END));
}

#undef NR_END
#undef FREE_ARG

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

static double dmaxarg1, dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
        (dmaxarg1) : (dmaxarg2))

static double dminarg1, dminarg2;
#define DMIN(a,b) (dminarg1=(a),dminarg2=(b),(dminarg1) < (dminarg2) ?\
        (dminarg1) : (dminarg2))

static float maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
        (maxarg1) : (maxarg2))

static float minarg1, minarg2;
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ?\
        (minarg1) : (minarg2))

static long lmaxarg1, lmaxarg2;
#define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ?\
        (lmaxarg1) : (lmaxarg2))

static long lminarg1, lminarg2;
#define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ?\
        (lminarg1) : (lminarg2))

static int imaxarg1, imaxarg2;
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
        (imaxarg1) : (imaxarg2))

static int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

// brent
#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

float brent(float ax, float bx, float cx, float(*f)(float), float tol,
	float *xmin)
{
	int iter;
	float a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
	float e = 0.0;

	a = (ax < cx ? ax : cx);
	b = (ax > cx ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = (*f)(x);
	for (iter = 1; iter <= ITMAX; iter++) {
		xm = 0.5*(a + b);
		tol2 = 2.0*(tol1 = tol*fabs(x) + ZEPS);
		if (fabs(x - xm) <= (tol2 - 0.5*(b - a))) {
			*xmin = x;
			return fx;
		}
		if (fabs(e) > tol1) {
			r = (x - w)*(fx - fv);
			q = (x - v)*(fx - fw);
			p = (x - v)*q - (x - w)*r;
			q = 2.0*(q - r);
			if (q > 0.0) p = -p;
			q = fabs(q);
			etemp = e;
			e = d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a - x) || p >= q*(b - x))
				d = CGOLD*(e = (x >= xm ? a - x : b - x));
			else {
				if (q == 0) return fx; //added by min
				d = p / q;
				u = x + d;
				if (u - a < tol2 || b - u < tol2)
					d = SIGN(tol1, xm - x);
			}
		}
		else {
			d = CGOLD*(e = (x >= xm ? a - x : b - x));
		}
		u = (fabs(d) >= tol1 ? x + d : x + SIGN(tol1, d));
		fu = (*f)(u);
		if (fu <= fx) {
			if (u >= x) a = x; else b = x;
			SHFT(v, w, x, u)
				SHFT(fv, fw, fx, fu)
		}
		else {
			if (u < x) a = u; else b = u;
			if (fu <= fw || w == x) {
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			}
			else if (fu <= fv || v == x || v == w) {
				v = u;
				fv = fu;
			}
		}
	}
	// nrerror("Too many iterations in brent"); //modified by min
	*xmin = x;
	return fx;
}
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT

// nmbrak 
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb, float *fc,
	float(*func)(float))
{
	float ulim, u, r, q, fu, dum;

	*fa = (*func)(*ax);
	*fb = (*func)(*bx);
	if (*fb > *fa) {
		SHFT(dum, *ax, *bx, dum)
			SHFT(dum, *fb, *fa, dum)
	}
	*cx = (*bx) + GOLD*(*bx - *ax);
	*fc = (*func)(*cx);
	while (*fb > *fc) {
		r = (*bx - *ax)*(*fb - *fc);
		q = (*bx - *cx)*(*fb - *fa);
		u = (*bx) - ((*bx - *cx)*q - (*bx - *ax)*r) /
			(2.0*SIGN(FMAX(fabs(q - r), TINY), q - r));
		ulim = (*bx) + GLIMIT*(*cx - *bx);
		if ((*bx - u)*(u - *cx) > 0.0) {
			fu = (*func)(u);
			if (fu < *fc) {
				*ax = (*bx);
				*bx = u;
				*fa = (*fb);
				*fb = fu;
				return;
			}
			else if (fu > *fb) {
				*cx = u;
				*fc = fu;
				return;
			}
			u = (*cx) + GOLD*(*cx - *bx);
			fu = (*func)(u);
		}
		else if ((*cx - u)*(u - ulim) > 0.0) {
			fu = (*func)(u);
			if (fu < *fc) {
				SHFT(*bx, *cx, u, *cx + GOLD*(*cx - *bx))
					SHFT(*fb, *fc, fu, (*func)(u))
			}
		}
		else if ((u - ulim)*(ulim - *cx) >= 0.0) {
			u = ulim;
			fu = (*func)(u);
		}
		else {
			u = (*cx) + GOLD*(*cx - *bx);
			fu = (*func)(u);
		}
		SHFT(*ax, *bx, *cx, u)
			SHFT(*fa, *fb, *fc, fu)
	}
}
#undef GOLD
#undef GLIMIT
#undef TINY
#undef SHFT


#define TOL 0.01 
int ncom;
float *pcom, *xicom, (*nrfunc)(float[]);

// f1dim 
float f1dim(float x)
{
	int j;
	float f, *xt;

	xt = vector(1, ncom);
	for (j = 1; j <= ncom; j++) xt[j] = pcom[j] + x*xicom[j];
	f = (*nrfunc)(xt);
	free_vector(xt, 1, ncom);
	return f;
}

// linmin
void linmin(float p[], float xi[], int n, float *fret, float(*func)(float[]))
{
	float brent(float ax, float bx, float cx,
		float(*f)(float), float tol, float *xmin);
	float f1dim(float x);
	void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb,
		float *fc, float(*func)(float));
	int j;
	float xx, xmin, fx, fb, fa, bx, ax;

	ncom = n;
	pcom = vector(1, n);
	xicom = vector(1, n);
	nrfunc = func;
	for (j = 1; j <= n; j++) {
		pcom[j] = p[j];
		xicom[j] = xi[j];
	}
	ax = 0.0;
	xx = 1.0;
	mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, f1dim);
	*fret = brent(ax, xx, bx, f1dim, TOL, &xmin);
	for (j = 1; j <= n; j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	free_vector(xicom, 1, n);
	free_vector(pcom, 1, n);
}
#undef TOL

#define ITMAX 100 // maximum number of powell loop
void powell(float p[], float **xi, int n, float ftol, int *iter, float *fret,
	float(*func)(float[]), int *totalIt, int itLimit)
{
	void linmin(float p[], float xi[], int n, float *fret,
		float(*func)(float[]));
	int i, ibig, j;
	float del, fp, fptt, t, *pt, *ptt, *xit;

	pt = vector(1, n);
	ptt = vector(1, n);
	xit = vector(1, n);
	*fret = (*func)(p);
	if (*fret >= 1.001) return 0; // set cost function value limit
	for (j = 1; j <= n; j++) pt[j] = p[j];
	for (*iter = 1;; ++(*iter)) {
		fp = (*fret);
		ibig = 0;
		del = 0.0;
		for (i = 1; i <= n; i++) {
			for (j = 1; j <= n; j++) xit[j] = xi[j][i];
			fptt = (*fret);
			linmin(p, xit, n, fret, func);
			if (fabs(fptt - (*fret)) > del) {
				del = fabs(fptt - (*fret));
				ibig = i;
			}
			if (*totalIt >= itLimit) return 0; // set sub-iteration limit
			if (*fret >= 1.001) return 1; // set cost function value limit
		}
		if (2.0*fabs(fp - (*fret)) <= ftol*(fabs(fp) + fabs(*fret))) {
			free_vector(xit, 1, n);
			free_vector(ptt, 1, n);
			free_vector(pt, 1, n);
			return;
		}
		if (*iter == ITMAX)  return 0; // nrerror("powell exceeding maximum iterations."); //
		for (j = 1; j <= n; j++) {
			ptt[j] = 2.0*p[j] - pt[j];
			xit[j] = p[j] - pt[j];
			pt[j] = p[j];
		}
		fptt = (*func)(ptt);
		if (fptt < fp) {
			t = 2.0*(fp - 2.0*(*fret) + fptt)*SQR(fp - (*fret) - del) - del*SQR(fp - fptt);
			if (t < 0.0) {
				linmin(p, xit, n, fret, func);
				for (j = 1; j <= n; j++) {
					xi[j][ibig] = xi[j][n];
					xi[j][n] = xit[j];
				}
				if (*totalIt >= itLimit) return 0; // set sub-iteration limit
				if (*fret >= 1.001) return 1; // set cost function value limit
			}
		}
	}
}
#undef ITMAX