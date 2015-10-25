
#include "imq.h"
#include <math.h>
#include <float.h>


const _INT32 filter_width = 7;
const double sigma_gauss = 1.5;
#ifdef CWSSIM_TEST
_INT32 K = 8;
_INT32 maxscale = 4;
#else
const _INT32 K = 8;
const _INT32 maxscale = 4;
#endif
const _INT32 rr = 40;
const double C = 0.03;
const _INT32 dowsampled_add  = 1;

#define RESIZE_IMAGE
#define use_ssim_on_HP
#define use_ssim_on_LP

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

bool wcustom_isnan(float var)
{
    volatile float d = var;
    return d != d;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

void kanaal(float *out, int w, int h, float *im) 
{
  	int i, j, index;
	
  	for(i = 0; i < w; ++i) 
	  {
	      for (j = 0; j < h; ++j) 
	        {
				index = i + w * j ;
				out[index] =  (im[index]);
	        }
	}
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

void expand (float *inputar, float *outputar, int w, int h, int rr)
  {
    int i, j, ii, jj, w1a, h1a;

    w1a = w + 2 * rr;
    h1a = h + 2 * rr;
    for (jj = 0; jj < h1a; ++jj)
      {
        for (ii = 0; ii < w1a; ++ii)
	  {
	    i = ii - rr;
	    if (i < 0)
	      {
	        i = rr - (ii + 1);
	      }
	    if (i > w - 1)
	      {
	        i = rr + 2 * w - 1 - ii;
	      }
	    j = jj - rr;
	    if (j < 0)
	      {
	        j = rr - (jj + 1);
	      }
	    if (j > h - 1)
	      {
	        j = rr + 2 * h - 1 - jj;
	      }
	    outputar[ii + w1a * jj] = inputar[i + w * j];
	  }
      }
  }

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------


double CalcLocalWSSIM (float *R1, float *I1, float *R2, float *I2, _INT32 W, _INT32 H, bool check_nan = true, bool fast = false){

double result = 0.0;
_INT32 size = W * H;

//
// NOW, WE CREATE THE FILTER, GAUSSIAN OR MEDIA FILTER, ACCORDING TO THE VALUE OF boolean "gaussian_window"
//
_INT32 filter_length = filter_width * filter_width;
float *window_weights = NULL;
double *array_gauss_window = NULL;
_INT32 pointer, i;

try {
	
		window_weights = new float [ filter_length ];
		array_gauss_window = new double [ filter_length ];
		
	} catch (...) { if (array_gauss_window) delete [] array_gauss_window; if (window_weights) delete [] window_weights; return result; }

double distance = 0.0;
_INT32 center = filter_width / 2;
double total = 0.0;
double sigma_sq = sigma_gauss * sigma_gauss;
		
  	for (_INT32 y = 0; y < filter_width; ++y){
		for (_INT32 x = 0; x < filter_width; ++x){
       				distance = abs(x-center) * abs(x-center) + abs(y-center) * abs(y-center);
				pointer = y * filter_width + x;
               			array_gauss_window[pointer] = (float) exp(-0.5 * distance / sigma_sq);
				total = total + array_gauss_window[pointer];
		}
	}
	
	for (pointer = 0; pointer < filter_length; ++pointer) {	
			array_gauss_window[pointer] = array_gauss_window[pointer] / total;
			window_weights [pointer] = (float) array_gauss_window[pointer];
		}

//
// END OF FILTER SELECTION							
//

if (check_nan) {
	for (i = 0; i < size; ++i)
		{
#ifdef _WIN32		     
					if (_isnan(R1[i])) R1[i] = 0.0f;
					if (_isnan(I1[i])) I1[i] = 0.0f;
					if (_isnan(R2[i])) R2[i] = 0.0f;
					if (_isnan(I2[i])) I2[i] = 0.0f;
#else
					if (wcustom_isnan(R1[i])) R1[i] = 0.0f;
					if (wcustom_isnan(I1[i])) I1[i] = 0.0f;
					if (wcustom_isnan(R2[i])) R2[i] = 0.0f;
					if (wcustom_isnan(I2[i])) I2[i] = 0.0f;
#endif
}}

bool stop = false;
if (!convolve (I2, W, H, window_weights, filter_length, filter_width, filter_width,fast)) stop = true;
if (!convolve (R1, W, H, window_weights, filter_length, filter_width, filter_width,fast)) stop = true;
if (!convolve (I1, W, H, window_weights, filter_length, filter_width, filter_width,fast)) stop = true;
if (!convolve (R2, W, H, window_weights, filter_length, filter_width, filter_width,fast)) stop = true;
if (stop) { if (array_gauss_window) delete [] array_gauss_window; if (window_weights) delete [] window_weights; return result; }

for (i = 0; i < size; ++i) {

	/*
#ifdef _WIN32		     
					if (_isnan(R1[i])) R1[i] = 0.0f;
					if (_isnan(I1[i])) I1[i] = 0.0f;
					if (_isnan(I2[i])) R2[i] = 0.0f;
					if (_isnan(R2[i])) R2[i] = 0.0f;
#else
					if (wcustom_isnan(R1[i])) R1[i] = 0.0f;
					if (wcustom_isnan(I1[i])) I1[i] = 0.0f;
					if (wcustom_isnan(R2[i])) R2[i] = 0.0f;
#endif
					*/

		double A = -I1[i]*I2[i] - R1[i]*R2[i];
		double B = -I1[i]*R2[i] + I2[i]*R1[i];
		double C = R1[i]*R1[i] + I1[i]*I1[i] + R2[i]*R2[i] + I2[i]*I2[i];
		R2[i] = (float)sqrt(C);
		R1[i] = (float)sqrt(A*A + B*B);

}

_INT32 M = size;

if (fast) {

for (M = i = 0; i < size; ++i) if ((R1[i] != 0.0) && (R2[i] != 0.0))
{ result += (2.0 * double(R1[i]) + C) / (double(R2[i])*double(R2[i]) + C); ++M; }

} else
for (i = 0; i < size; ++i) {
	result += (2.0 * double(R1[i]) + C) / (double(R2[i])*double(R2[i]) + C);
}

result /= double(M);

if (array_gauss_window) delete [] array_gauss_window; 
if (window_weights) delete [] window_weights;
	
return result;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double CW_SSIMF(float *imin_1, float *imin_2, int PX, int PY, int BPP, bool fast) {

double result = 0.0;
double M = 0.0;
_INT32 size = PX * PY;
_INT32 w1a, h1a, i, j;

_INT32 w = PX;
_INT32 h = PY;


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
#ifdef RESIZE_IMAGE
double downsampled = dowsampled_add + (_INT32) h / 256;

if (downsampled > 1.0) {

			_INT32 new_image_width = (_INT32) (w/downsampled);
			bool stop = false;
			if (!resize(imin_1,new_image_width,&w,&h,false,true)) stop = true;
			if (!resize(imin_2,new_image_width,&w,&h,true,true)) stop = true;
			if (stop)
			{ 	
				return result; 
			}}
#endif
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------

float *input2_1 = NULL, **Ar_1 = NULL, **Ac_1 = NULL, ***Br_1 = NULL, ***Bc_1 = NULL, **L1_1 = NULL;
float *input2_2 = NULL, **Ar_2 = NULL, **Ac_2 = NULL, ***Br_2 = NULL, ***Bc_2 = NULL, **L1_2 = NULL;

w1a = w + 2 * rr;
h1a = h + 2 * rr;

try {

input2_1 = new float [ w1a * h1a ];
Ar_1 = new float* [ K ];
Ac_1 = new float* [ K ];
Br_1 = new float** [ K ];
Bc_1 = new float** [ K ];
L1_1    = new float* [ maxscale + 1 ];

for (i = 0; i < K; ++i)
	  {
		Br_1[i] = new float* [ maxscale ];
	    Bc_1[i] = new float* [ maxscale ];
	  }

for (i = 0; i < K; ++i)
  {
		for (j = 0; j < maxscale; ++j)
			{
				_INT32 w1, h1;
				w1 = w1a / my_pow(2, j);
				h1 = h1a / my_pow(2, j); 
				Br_1[i][j] = new float [ w1 * h1 ];
				Bc_1[i][j] = new float [ w1 * h1 ];
			}
		Ar_1[i] = new float [ w1a * h1a ];
	    Ac_1[i] = new float [ w1a * h1a ];
  }

	
for (i = 0; i < maxscale; ++i)
	  {
	     _INT32 w1, h1;
	     w1 = w1a / my_pow(2, i);
	     h1 = h1a / my_pow(2, i);
	  }

for (i = 0; i < maxscale + 1; ++i)
	  {
	     _INT32 w1, h1;
	     w1 = w1a / my_pow(2, i);
	     h1 = h1a / my_pow(2, i);
	     L1_1[i] = new float [ w1 * h1 ];
	  }

}
catch (...) {

	return result;

}

expand (imin_1, input2_1, w, h, rr); 
decompose(maxscale, K, input2_1, w1a, h1a, L1_1, Br_1, Bc_1, Ar_1, Ac_1);


delete [] input2_1;

try {

input2_2 = new float [ w1a * h1a ];
Ar_2 = new float* [ K ];
Ac_2 = new float* [ K ];
Br_2 = new float** [ K ];
Bc_2 = new float** [ K ];
L1_2    = new float* [ maxscale + 1 ];

for (i = 0; i < K; ++i)
	  {
	    Br_2[i] =new float* [ maxscale ];
	    Bc_2[i] =new float* [ maxscale ];
	  }

for (i = 0; i < K; ++i)
  {
		for (j = 0; j < maxscale; ++j)
			{
				_INT32 w1, h1;
				w1 = w1a / my_pow(2, j);
				h1 = h1a / my_pow(2, j); 
				Br_2[i][j] = new float [ w1 * h1 ];
				Bc_2[i][j] = new float [ w1 * h1 ];
			}
		Ar_2[i] = new float [ w1a * h1a ];
	    Ac_2[i] = new float [ w1a * h1a ];
  }

	
for (i = 0; i < maxscale; ++i)
	  {
	     _INT32 w1, h1;
	     w1 = w1a / my_pow(2, i);
	     h1 = h1a / my_pow(2, i);
	  }

for (i = 0; i < maxscale + 1; ++i)
	  {
	     _INT32 w1, h1;
	     w1 = w1a / my_pow(2, i);
	     h1 = h1a / my_pow(2, i);
	     L1_2[i]    = new float [ w1 * h1 ];
	  }

}
catch (...) {

	return result;
}

expand (imin_2, input2_2, w, h, rr); 
decompose(maxscale, K, input2_2, w1a, h1a, L1_2, Br_2, Bc_2, Ar_2, Ac_2);


delete [] input2_2;

double LWSSIM; 

j = maxscale - 1;
{
		_INT32 w1, h1;
		w1 = w1a / my_pow(2, j);
		h1 = h1a / my_pow(2, j);
		double KOR = 0.0;
		for (i = 0; i < K ; ++i)
			{
				
				KOR += CalcLocalWSSIM(Br_1[i][j],Bc_1[i][j],Br_2[i][j],Bc_2[i][j],w1,h1,false,fast);
				
			}

	LWSSIM = KOR / double(K);		
			
}

_INT32 MM = 1;

#ifdef use_ssim_on_HP
j = 0;
double KOR = 0.0;
for (i = 0; i < K ; ++i)
		{
				
			_INT32 w1, h1;
			w1 = w1a / my_pow(2,j);
			h1 = h1a / my_pow(2,j);
			KOR += CalcLocalWSSIM(Ar_1[i],Ac_1[i],Ar_2[i],Ac_2[i],w1,h1,false,fast);
				
	}

result += KOR / double(K);
++MM;
#endif 

#ifdef use_ssim_on_LP
	
j = 0;
_INT32 w1, h1;
w1 = w1a / my_pow(2, j);
h1 = h1a / my_pow(2, j);
double SSIM = MS_SSIMF(L1_1[j],L1_2[j],w1,h1,false,true,8,fast,1.0f,1.0f,1.0f); 
result += SSIM;
++MM;

#endif 

result += LWSSIM;
result /= double(MM);

for (i = 0; i < K; ++i) { delete [] Ar_2[i]; delete [] Ac_2[i]; }
for (i = 0; i < K; ++i) { delete [] Ar_1[i]; delete [] Ac_1[i]; }
delete [] Ar_2;
delete [] Ac_2;
delete [] Ar_1;
delete [] Ac_1;
for (i = 0; i < maxscale + 1; ++i) delete [] L1_2[i];
for (i = 0; i < maxscale + 1; ++i) delete [] L1_1[i];
delete [] L1_1;
delete [] L1_2;

for (i = 0; i < K; ++i)
		for (j = 0; j < maxscale; ++j)
			{
				delete [] Br_1[i][j];
				delete [] Bc_1[i][j];
				delete [] Br_2[i][j];
				delete [] Bc_2[i][j];
			}

for (i = 0; i < K; ++i)
	  {
		delete [] Br_1[i];
	    delete [] Bc_1[i];
	    delete [] Br_2[i];
	    delete [] Bc_2[i];
	  }

delete [] Br_1;
delete [] Bc_1;
delete [] Br_2;
delete [] Bc_2;

return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double DoCW_SSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

_INT32 size = PX * PY;
double result = 0.0;

float *orig_imgb = NULL;
float *comp_imgb = NULL;

try {
      orig_imgb = new float [ size ];
      comp_imgb = new float [ size ];
} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }


switch(BPP) {


	case 8: {


		for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MBYTE); comp_imgb[i] = (float)(comp_img[i]&_MBYTE); }	
		result = CW_SSIMF(orig_imgb,comp_imgb,PX,PY,BPP,fast);


	};
	break;

	case 16: {

		for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MINT16); comp_imgb[i] = (float)(comp_img[i]&_MINT16); }	
		result = CW_SSIMF(orig_imgb,comp_imgb,PX,PY,BPP,fast);
                    

	};
	break;


	default: break;


}

delete [] orig_imgb;
delete [] comp_imgb;
return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double CW_SSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast) {

_INT32 size = PX * PY;
double result = 0.0;

float *orig_imgb = NULL;
float *comp_imgb = NULL;
try {
      orig_imgb = new float [ size ];
      comp_imgb = new float [ size ];
} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }


for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MBYTE); comp_imgb[i] = (float)(comp_img[i]&_MBYTE); }	
result = CW_SSIMF(orig_imgb,comp_imgb,PX,PY,8,fast);

delete [] orig_imgb;
delete [] comp_imgb;
return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double CW_SSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast) {

_INT32 size = PX * PY;
double result = 0.0;

float *orig_imgb = NULL;
float *comp_imgb = NULL;
try {
      orig_imgb = new float [ size ];
      comp_imgb = new float [ size ];
} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }


for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MINT16); comp_imgb[i] = (float)(comp_img[i]&_MINT16); }	
result = CW_SSIMF(orig_imgb,comp_imgb,PX,PY,16,fast);

delete [] orig_imgb;
delete [] comp_imgb;
return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
#ifdef CWSSIM_TEST
double DoCW_SSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, WSS *wss, bool fast)
{
::K = wss->K;
::maxscale = wss->L;
#else
double DoCW_SSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast)
{
#endif
	_INT32 size = PX * PY;
	double result = 0.0;
	
	switch (BPP) {

	case 8: case 16:
				result = DoCW_SSIM(orig_img, comp_img, PX,PY,BPP, fast);
				break;

	case 24:  {
			
			float *orig_imgb = NULL;
			float *comp_imgb = NULL;
			try {
		      		orig_imgb = new float [ size ];
		      		comp_imgb = new float [ size ];
			} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }

			for(_INT32 i=0; i < size; ++i) {
			double Y1 = rgB * (double)((orig_img[i]>>16)&_MBYTE) + rGb * (double)((orig_img[i]>>8)&_MBYTE) + Rgb * (double)(orig_img[i]&_MBYTE) + Crgb;
			double Y2 = rgB * (double)((comp_img[i]>>16)&_MBYTE) + rGb * (double)((comp_img[i]>>8)&_MBYTE) + Rgb * (double)(comp_img[i]&_MBYTE) + Crgb;
			comp_imgb[i] = (float)Y2;
			orig_imgb[i] = (float)Y1;
			}

			result = CW_SSIMF(orig_imgb, comp_imgb, PX,PY,BPP, fast);
			delete [] orig_imgb;
			delete [] comp_imgb;
		};
		break;

	default: break;

	}

return result;

}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double DoCW_SSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast)
{

	_INT32 size = PX * PY;
	_INT32 bsize = size * 3;
	double result = 0.0;
	
	switch (BPP) {

	case 24:  {
			
			float *orig_imgb = NULL;
			float *comp_imgb = NULL;
			try {
		      		orig_imgb = new float [ size ];
		      		comp_imgb = new float [ size ];
			} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }

			for(_INT32 i=0,j=0; i < bsize; i+=3,++j) {
			if ((i < bsize) && (i + 1 < bsize) && (i + 2 < bsize) && (j < size)) {
				double Y1 = rgB * (double)(orig_img[i]) + rGb * (double)(orig_img[i+1]) + Rgb * (double)(orig_img[i+2]) + Crgb;
				double Y2 = rgB * (double)(comp_img[i]) + rGb * (double)(comp_img[i+1]) + Rgb * (double)(comp_img[i+2]) + Crgb;
				comp_imgb[j] = (float)Y2;
				orig_imgb[j] = (float)Y1;
			} 			
			}

			result = CW_SSIMF(orig_imgb, comp_imgb, PX,PY,BPP,fast);
			delete [] orig_imgb;
			delete [] comp_imgb;
		};
		break;

	default: break;

	}

return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
