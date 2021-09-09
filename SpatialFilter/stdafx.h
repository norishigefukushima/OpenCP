#pragma once
#define  _CRT_SECURE_NO_WARNINGS

//#define USING_AVX512 // for AVX512

#define COEFFICIENTS_SMALLEST_FIRST

//#define USE_SET_VEC 
#ifdef USE_SET_VEC 
/*
use float LUT
float a[n];
__m256 mval = _mm256_set1_ps(a[n]);
*/
#define SETVEC float
#define SETVECD double
#define _MM256_SET_VEC(a) _mm256_set1_ps(a)
#define _MM256_SETLUT_VEC(a) a
#define _MM256_SET_VECD(a) _mm256_set1_pd(a)
#define _MM256_SETLUT_VECD(a) a

#else 
/*
//usually better
use __m256 LUT
__m256 a[n];
__m256 mval = a[n];
*/
#define SETVEC __m256
#define _MM256_SET_VEC(a) a
#define _MM256_SETLUT_VEC(a) _mm256_set1_ps(a)

#define SETVECD __m256d
#define _MM256_SET_VECD(a) a
#define _MM256_SETLUT_VECD(a) _mm256_set1_pd(a)
#endif

//for experiment
//#define WITHOUT_FMA 1

//not using FMA
#ifdef WITHOUT_FMA
#define _mm256_fmadd_ps(a,x,b) _mm256_add_ps(_mm256_mul_ps(a,x),b)
#define _mm256_fmsub_ps(a,x,b) _mm256_sub_ps(_mm256_mul_ps(a,x),b)
#define _mm256_fnmadd_ps(a,x,b) _mm256_sub_ps(b,_mm256_mul_ps(a,x))
#endif

//for optimizeSpectrum in SlidingDCT
//#define USE_EIGEN
//for optimizeSpectrum in SlidingDCT
//#define USE_OPTIMIZE_DCT_SWICH

//IIR
#define DERICHE_TIMEPLOT 1
#define VYV_TIMEPLOT 1

//Plot DCT kernel in setRadius
//#define PLOT_DCT_KERNEL

//Plot profile plot in Debug
#define PLOT_Zk

//#define PRINT_RANK_DEFICIENT

#define COMPILE_GF_DCT1_32F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-1) 
#define COMPILE_GF_DCT1_64F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-1) 

#define COMPILE_GF_DCT3_32F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-3) 
#define COMPILE_GF_DCT3_64F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-3) 

#define COMPILE_GF_DCT5_32F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-5) 
#define COMPILE_GF_DCT5_64F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-5) 

#define COMPILE_GF_DCT7_32F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-7) 
#define COMPILE_GF_DCT7_64F_ORDER_TEMPLATE 1 //use template for approximation order(DCT-7) 

//#define INIT_TIME_TEMPLATE //print init time for range table

#include <opencp.hpp>
//#define __AVX2__ 1
//#define __FMA__ 1
//#define __AVX__ 1

#include "patchSIMDFunctions.hpp"
#include "search1D.hpp"
#include "spatialfilter/SpatialFilter.hpp"
//#include "GaussianFilter128.hpp"