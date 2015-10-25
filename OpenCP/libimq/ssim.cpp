#include <math.h>
#include "imq.h"

//MS_SSIMF(float *forig_img, float *fcomp_img, _INT32 PX, _INT32 PY, bool Wang, bool SSIM, _INT32 bits_per_pixel_1, bool fast, float a, float b, float g)

const float aa = 0.05f;
const float bb = 0.15f;
const float gg = 0.10f;

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double ABGDoSSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast, float a, float b, float g) {

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
		result = MS_SSIMF(orig_imgb,comp_imgb,PX,PY,false,true,8,fast,a,b,g);


	};
	break;

	case 16: {

		for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MINT16); comp_imgb[i] = (float)(comp_img[i]&_MINT16); }	
		result = MS_SSIMF(orig_imgb,comp_imgb,PX,PY,false,true,16,fast,a,b,g);
                    

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

double ABGSSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast, float a, float b, float g) {

_INT32 size = PX * PY;
double result = 0.0;

float *orig_imgb = NULL;
float *comp_imgb = NULL;
try {
      orig_imgb = new float [ size ];
      comp_imgb = new float [ size ];
} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }


for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MBYTE); comp_imgb[i] = (float)(comp_img[i]&_MBYTE); }	
result = MS_SSIMF(orig_imgb,comp_imgb,PX,PY,false,true,8,fast,a,b,g);

delete [] orig_imgb;
delete [] comp_imgb;
return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double ABGSSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast, float a, float b, float g) {

_INT32 size = PX * PY;
double result = 0.0;

float *orig_imgb = NULL;
float *comp_imgb = NULL;
try {
      orig_imgb = new float [ size ];
      comp_imgb = new float [ size ];
} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }


for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i]&_MINT16); comp_imgb[i] = (float)(comp_img[i]&_MINT16); }	
result = MS_SSIMF(orig_imgb,comp_imgb,PX,PY,false,true,16,fast,a,b,g);

delete [] orig_imgb;
delete [] comp_imgb;
return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------


double ABGDoSSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast, float a, float b, float g)
{

	_INT32 size = PX * PY;
	double result = 0.0;
	
	switch (BPP) {

	case 8: case 16:
				result = ABGDoSSIM(orig_img, comp_img, PX,PY,BPP, fast,a,b,g);
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

			result = MS_SSIMF(orig_imgb,comp_imgb,PX,PY,false,true,8,fast,a,b,g);
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

double ABGDoSSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast, float a, float b, float g)
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

			result = MS_SSIMF(orig_imgb,comp_imgb,PX,PY,false,true,8,fast,a,b,g);
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


double DoSSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

	return ABGDoSSIMY(orig_img,comp_img,PX,PY,BPP,fast,1.0f,1.0f,1.0f);

}

double DoSSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

	return ABGDoSSIMY(orig_img,comp_img,PX,PY,BPP,fast,1.0f,1.0f,1.0f);

}


double DoSSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

	return ABGDoSSIM(orig_img,comp_img,PX,PY,BPP,fast, 1.0f, 1.0f, 1.0f);

}

double SSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast) {

	return ABGSSIM16bit(orig_img,comp_img,PX,PY,fast,1.0f,1.0f,1.0f);

}

double SSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast) {

	return ABGSSIM8bit(orig_img, comp_img, PX, PY,fast,1.0f,1.0f,1.0f);

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double mDoSSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

	return ABGDoSSIMY(orig_img,comp_img,PX,PY,BPP,fast,aa,bb,gg);

}

double mDoSSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

	return ABGDoSSIMY(orig_img,comp_img,PX,PY,BPP,fast,aa,bb,gg);

}


double mDoSSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast) {

	return ABGDoSSIM(orig_img,comp_img,PX,PY,BPP,fast,aa,bb,gg);

}

double mSSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast) {

	return ABGSSIM16bit(orig_img,comp_img,PX,PY,fast,aa,bb,gg);

}

double mSSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast) {

	return ABGSSIM8bit(orig_img, comp_img, PX, PY,fast,aa,bb,gg);

}

//----------------------------------------------------------------------------------------------------------------------------------------------
double __DoSSIM(_INT32 *orig_img,_INT32 *comp_img,_INT32 PX,_INT32 PY,_INT32 BPP, float a,float b,float g, bool fast) {

return ABGDoSSIMY(orig_img,comp_img,PX,PY,BPP,fast,a,b,g);

}
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
