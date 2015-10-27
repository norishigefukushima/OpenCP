#pragma once

#define PC601
#include <tchar.h>
#include <fftw3.h>

#ifdef REC601

#define Rgb 0.257
#define rGb 0.504
#define rgB 0.098
#define Crgb 16.0

#define Rgbf 0.257f
#define rGbf 0.504f
#define rgBf 0.098f
#define Crgbf 16.0f

#endif


#ifdef PC601

#define Rgb 0.299
#define rGb 0.587
#define rgB 0.114
#define Crgb 0.0

#endif

#ifndef _INT32
#define _INT32 int
#endif
#ifndef _UINT32
#define _UINT32 unsigned int
#endif
#ifndef _MINT32
#define _MINT32 0xFFFFFFFF
#endif

#ifndef _INT16
#define _INT16 short int
#endif
#ifndef _UINT16
#define _UINT16 unsigned short int
#endif
#ifndef _MINT16
#define _MINT16 0xFFFF
#endif

#ifndef _BYTE
#define _BYTE unsigned char
#endif
#ifndef _MBYTE
#define _MBYTE 0xFF
#endif

#ifndef _INT64
#define _INT64 int
#endif
#ifndef _UINT64
#define _UINT64 unsigned int
#endif

// ----------------------------------

#ifndef WIN32
#ifndef WORD
#define WORD unsigned short int
#endif
#ifndef DWORD
#define DWORD unsigned int
#endif
#ifndef LONG
#define LONG long
#endif
#endif

double DoDeltaY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);
double DoDeltaY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double DoDelta(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double Delta16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY);
double Delta8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY);




bool decompose(int maxscale, int K, float *kanaal, int w, int h, float **L1, float ***Br, float ***Bc, float  **Ar, float  **Ac);
int my_pow(int a, int b);

double DoMSADY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);
double DoMSADY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double DoMSAD(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double MSAD16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY);
double MSAD8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY);


double DoMSEY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);
double DoMSEY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double DoMSE(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double MSE16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY);
double MSE8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY);


double DoMS_SSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool Wang, bool fast = false);
double DoMS_SSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool Wang, bool fast = false);

double DoMS_SSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool Wang, bool fast = false);

double MS_SSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool Wang, bool fast = false);
double MS_SSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool Wang, bool fast = false);

double MS_SSIMF(float *forig_img, float *fcomp_img, _INT32 PX, _INT32 PY, bool Wang, bool SSIM, _INT32 bits_per_pixel_1, bool fast, float a, float b, float g);

bool convolve(float *image, _INT32 width, _INT32 height, float* kernel, _INT32 kernel_len, _INT32 kw, _INT32 kh, bool fast);
bool resize(float *mas, _INT32 dstWidth, _INT32 *WW, _INT32 *HH, bool modify, bool interpolationMethod = false);


double DoSSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);
double DoSSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);

double DoSSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);

double SSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast);
double SSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast);

double mDoSSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);
double mDoSSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);

double mDoSSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);

double mSSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast);
double mSSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast);

double __DoSSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, float a, float b, float g, bool fast);


//#define CWSSIM_TEST

#ifdef CWSSIM_TEST

struct WSS {

	int K;
	int L;

};

#endif

double DoCW_SSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);

#ifdef CWSSIM_TEST
double DoCW_SSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, WSS *wss, bool fast);
#else
double DoCW_SSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);
#endif


double DoCW_SSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool fast);

double CW_SSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool fast);
double CW_SSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool fast);

double CW_SSIMF(float *forig_img, float *fcomp_img, _INT32 PX, _INT32 PY, bool fast);



double DoPSNRY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);
double DoPSNRY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double DoPSNR(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP);

double PSNR16bitL8(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY);
double PSNR16bitL16(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY);
double PSNR8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY);







