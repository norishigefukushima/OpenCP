#include <nmmintrin.h> //SSE4.2
#define  _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
//info: code
//http://d.hatena.ne.jp/shiku_otomiya/20100902/p1 (in japanese)

//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.

void transpose4x4(float* src);
void transpose4x4(float* src, float* dest);
void transpose8x8(float* src);
void transpose8x8(const float* src, float* dest);
#include <stdio.h>
void print_m128(__m128 src);
#define _KEEP_00_COEF_

inline int getNonzero(float* s, int size)
{
	int ret = 0;
	for (int i = 0; i < size; i++)
	{
		if (s[i] != 0.f)ret++;
	}

	return ret;
}

void fDCT2D8x4_and_threshold_keep00_32f(const float* x, float* y, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);

	__m128 v = _mm_mul_ps(_mm_add_ps(c0, c1), invsqrt2h);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	// keep 00 coef.
	__m128 v2 = _mm_blendv_ps(zeros, v, msk);
	v2 = _mm_blend_ps(v2, v, 1);
	_mm_store_ps(y, v2);

	v = _mm_mul_ps(_mm_sub_ps(c0, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 32, v);

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	v = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 16, v);

	v = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 48, v);
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	v = _mm_mul_ps(_mm_sub_ps(c0, c2), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 24, v);

	v = _mm_mul_ps(_mm_sub_ps(c3, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 40, v);
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	v = _mm_mul_ps(_mm_add_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 8, v);

	v = _mm_mul_ps(_mm_sub_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 56, v);
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}

void fDCT2D8x4_and_threshold_32f(const float* x, float* y, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);

	__m128 v = _mm_mul_ps(_mm_add_ps(c0, c1), invsqrt2h);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y, v);

	v = _mm_mul_ps(_mm_sub_ps(c0, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 32, v);

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	v = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 16, v);

	v = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 48, v);
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	v = _mm_mul_ps(_mm_sub_ps(c0, c2), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 24, v);

	v = _mm_mul_ps(_mm_sub_ps(c3, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 40, v);
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	v = _mm_mul_ps(_mm_add_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 8, v);

	v = _mm_mul_ps(_mm_sub_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 56, v);
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}

void fDCT2D8x4noscale_32f(const float* x, float* y)
{
	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/


	_mm_store_ps(y, _mm_add_ps(c0, c1));
	_mm_store_ps(y + 32, _mm_sub_ps(c0, c1));

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	_mm_store_ps(y + 16, _mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)));
	_mm_store_ps(y + 48, _mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)));
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	_mm_store_ps(y + 24, _mm_sub_ps(c0, c2));
	_mm_store_ps(y + 40, _mm_sub_ps(c3, c1));
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	_mm_store_ps(y + 8, _mm_add_ps(c0, c3));
	_mm_store_ps(y + 56, _mm_sub_ps(c0, c3));
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}
void fDCT2D8x4_32f(const float* x, float* y)
{
	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);
	_mm_store_ps(y, _mm_mul_ps(_mm_add_ps(c0, c1), invsqrt2h));
	_mm_store_ps(y + 32, _mm_mul_ps(_mm_sub_ps(c0, c1), invsqrt2h));

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	_mm_store_ps(y + 16, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)), invsqrt2h));
	_mm_store_ps(y + 48, _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)), invsqrt2h));
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	_mm_store_ps(y + 24, _mm_mul_ps(_mm_sub_ps(c0, c2), invsqrt2h));
	_mm_store_ps(y + 40, _mm_mul_ps(_mm_sub_ps(c3, c1), invsqrt2h));
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	_mm_store_ps(y + 8, _mm_mul_ps(_mm_add_ps(c0, c3), invsqrt2h));
	_mm_store_ps(y + 56, _mm_mul_ps(_mm_sub_ps(c0, c3), invsqrt2h));
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}

void fDCT8x8_32f_and_threshold(const float* s, float* d, float threshold, float* temp)
{
	transpose8x8(s, temp);

	/*for (int j = 0; j < 8; j ++)
	{
	for (int i = 0; i < 8; i ++)
	{
	temp[8*i+j] =s[8*j+i];
	}
	}*/

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	/*for (int j = 0; j < 8; j ++)
	{
	for (int i = 0; i < 8; i ++)
	{
	temp[8*i+j] =d[8*j+i];
	}
	}*/
	fDCT2D8x4_and_threshold_32f(temp, d, threshold);
	fDCT2D8x4_and_threshold_32f(temp + 4, d + 4, threshold);

}
void fDCT8x8_32f(const float* s, float* d, float* temp)
{
	//for (int j = 0; j < 8; j ++)
	//{
	//	for (int i = 0; i < 8; i ++)
	//	{
	//		temp[8*i+j] =s[8*j+i];
	//	}
	//}
	transpose8x8(s, temp);

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);

	//for (int j = 0; j < 8; j ++)
	//{
	//	for (int i = 0; i < 8; i ++)
	//	{
	//		temp[8*i+j] =d[8*j+i];
	//	}
	//}
	transpose8x8(d, temp);
	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);
}

void fDCT1Dllm_32f(const float* x, float* y)
{
	float t0, t1, t2, t3, t4, t5, t6, t7; float c0, c1, c2, c3; float r[8]; int i;

	for (i = 0; i < 8; i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
	const float invsqrt2 = 0.707107f;//(float)(1.0f / M_SQRT2);
	const float invsqrt2h = 0.353554f;//invsqrt2*0.5f;

	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;

	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;

	y[0] = c0 + c1;
	y[4] = c0 - c1;
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];

	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];

	y[5] = c3 - c1; y[3] = c0 - c2;
	c0 = (c0 + c2) * invsqrt2;
	c3 = (c3 + c1) * invsqrt2;
	y[1] = c0 + c3; y[7] = c0 - c3;

	for (i = 0; i < 8; i++)
	{
		y[i] *= invsqrt2h;
	}
}

void fDCT2Dllm_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j++)
	{
		fDCT1Dllm_32f(s + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
	for (int j = 0; j < 8; j++)
	{
		fDCT1Dllm_32f(d + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
}

void iDCT1Dllm_32f(const float* y, float* x)
{
	float a0, a1, a2, a3, b0, b1, b2, b3; float z0, z1, z2, z3, z4; float r[8]; int i;

	for (i = 0; i < 8; i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }

	z0 = y[1] + y[7]; z1 = y[3] + y[5]; z2 = y[3] + y[7]; z3 = y[1] + y[5];
	z4 = (z0 + z1) * r[3];

	z0 = z0 * (-r[3] + r[7]);
	z1 = z1 * (-r[3] - r[1]);
	z2 = z2 * (-r[3] - r[5]) + z4;
	z3 = z3 * (-r[3] + r[5]) + z4;

	b3 = y[7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2;
	b2 = y[5] * (r[1] + r[3] - r[5] + r[7]) + z1 + z3;
	b1 = y[3] * (r[1] + r[3] + r[5] - r[7]) + z1 + z2;
	b0 = y[1] * (r[1] + r[3] - r[5] - r[7]) + z0 + z3;

	z4 = (y[2] + y[6]) * r[6];
	z0 = y[0] + y[4]; z1 = y[0] - y[4];
	z2 = z4 - y[6] * (r[2] + r[6]);
	z3 = z4 + y[2] * (r[2] - r[6]);
	a0 = z0 + z3; a3 = z0 - z3;
	a1 = z1 + z2; a2 = z1 - z2;

	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;

	for (i = 0; i < 8; i++){ x[i] *= 0.353554f; }
}

void iDCT2Dllm_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j++)
	{
		iDCT1Dllm_32f(s + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
	for (int j = 0; j < 8; j++)
	{
		iDCT1Dllm_32f(d + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
}

void iDCT2D8x4_32f(const float* y, float* x)
{
	/*
	float a0,a1,a2,a3,b0,b1,b2,b3; float z0,z1,z2,z3,z4; float r[8]; int i;
	for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
	*/
	/*
	0: 1.414214
	1: 1.387040
	2: 1.306563
	3:
	4: 1.000000
	5: 0.785695
	6:
	7: 0.275899
	*/
	__m128 my1 = _mm_load_ps(y + 8);
	__m128 my7 = _mm_load_ps(y + 56);
	__m128 mz0 = _mm_add_ps(my1, my7);

	__m128 my3 = _mm_load_ps(y + 24);
	__m128 mz2 = _mm_add_ps(my3, my7);
	__m128 my5 = _mm_load_ps(y + 40);
	__m128 mz1 = _mm_add_ps(my3, my5);
	__m128 mz3 = _mm_add_ps(my1, my5);

	__m128 w = _mm_set1_ps(1.175876f);
	__m128 mz4 = _mm_mul_ps(_mm_add_ps(mz0, mz1), w);
	//z0 = y[1] + y[7]; z1 = y[3] + y[5]; z2 = y[3] + y[7]; z3 = y[1] + y[5];
	//z4 = (z0 + z1) * r[3];

	w = _mm_set1_ps(-1.961571f);
	mz2 = _mm_add_ps(_mm_mul_ps(mz2, w), mz4);
	w = _mm_set1_ps(-0.390181f);
	mz3 = _mm_add_ps(_mm_mul_ps(mz3, w), mz4);
	w = _mm_set1_ps(-0.899976f);
	mz0 = _mm_mul_ps(mz0, w);
	w = _mm_set1_ps(-2.562915f);
	mz1 = _mm_mul_ps(mz1, w);


	/*
	-0.899976
	-2.562915
	-1.961571
	-0.390181
	z0 = z0 * (-r[3] + r[7]);
	z1 = z1 * (-r[3] - r[1]);
	z2 = z2 * (-r[3] - r[5]) + z4;
	z3 = z3 * (-r[3] + r[5]) + z4;*/

	w = _mm_set1_ps(0.298631f);
	__m128 mb3 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my7, w), mz0), mz2);
	w = _mm_set1_ps(2.053120f);
	__m128 mb2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my5, w), mz1), mz3);
	w = _mm_set1_ps(3.072711f);
	__m128 mb1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my3, w), mz1), mz2);
	w = _mm_set1_ps(1.501321f);
	__m128 mb0 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my1, w), mz0), mz3);
	/*
	0.298631
	2.053120
	3.072711
	1.501321
	b3 = y[7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2;
	b2 = y[5] * ( r[1] + r[3] - r[5] + r[7]) + z1 + z3;
	b1 = y[3] * ( r[1] + r[3] + r[5] - r[7]) + z1 + z2;
	b0 = y[1] * ( r[1] + r[3] - r[5] - r[7]) + z0 + z3;
	*/

	__m128 my2 = _mm_load_ps(y + 16);
	__m128 my6 = _mm_load_ps(y + 48);
	w = _mm_set1_ps(0.541196f);
	mz4 = _mm_mul_ps(_mm_add_ps(my2, my6), w);
	__m128 my0 = _mm_load_ps(y);
	__m128 my4 = _mm_load_ps(y + 32);
	mz0 = _mm_add_ps(my0, my4);
	mz1 = _mm_sub_ps(my0, my4);


	w = _mm_set1_ps(-1.847759f);
	mz2 = _mm_add_ps(mz4, _mm_mul_ps(my6, w));
	w = _mm_set1_ps(0.765367f);
	mz3 = _mm_add_ps(mz4, _mm_mul_ps(my2, w));

	my0 = _mm_add_ps(mz0, mz3);
	my3 = _mm_sub_ps(mz0, mz3);
	my1 = _mm_add_ps(mz1, mz2);
	my2 = _mm_sub_ps(mz1, mz2);
	/*
	1.847759
	0.765367
	z4 = (y[2] + y[6]) * r[6];
	z0 = y[0] + y[4]; z1 = y[0] - y[4];
	z2 = z4 - y[6] * (r[2] + r[6]);
	z3 = z4 + y[2] * (r[2] - r[6]);
	a0 = z0 + z3; a3 = z0 - z3;
	a1 = z1 + z2; a2 = z1 - z2;
	*/

	w = _mm_set1_ps(0.353554f);
	_mm_store_ps(x, _mm_mul_ps(w, _mm_add_ps(my0, mb0)));
	_mm_store_ps(x + 56, _mm_mul_ps(w, _mm_sub_ps(my0, mb0)));
	_mm_store_ps(x + 8, _mm_mul_ps(w, _mm_add_ps(my1, mb1)));
	_mm_store_ps(x + 48, _mm_mul_ps(w, _mm_sub_ps(my1, mb1)));
	_mm_store_ps(x + 16, _mm_mul_ps(w, _mm_add_ps(my2, mb2)));
	_mm_store_ps(x + 40, _mm_mul_ps(w, _mm_sub_ps(my2, mb2)));
	_mm_store_ps(x + 24, _mm_mul_ps(w, _mm_add_ps(my3, mb3)));
	_mm_store_ps(x + 32, _mm_mul_ps(w, _mm_sub_ps(my3, mb3)));
	/*
	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;
	for(i = 0;i < 8;i++){ x[i] *= 0.353554f; }
	*/
}



void iDCT8x8_32f(const float* s, float* d, float* temp)
{
	transpose8x8((float*)s, temp);
	//for (int j = 0; j < 8; j ++)
	//{
	//	for (int i = 0; i < 8; i ++)
	//	{
	//		temp[8*i+j] =s[8*j+i];
	//	}
	//}
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	/*for (int j = 0; j < 8; j ++)
	{
	for (int i = 0; i < 8; i ++)
	{
	temp[8*i+j] =d[8*j+i];
	}
	}*/
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);
}



#ifdef UNDERCONSTRUCTION_____
//internal simd using sse3
void LLMDCTOpt(const float* x, float* y)
{
	float t4,t5,t6,t7; float c0,c1,c2,c3; 
	float* r = dct_tbl;

	const float invsqrt2= 0.707107f;//(float)(1.0f / M_SQRT2);
	const float invsqrt2h=0.353554f;//invsqrt2*0.5f;

	{
		__m128 mc1 = _mm_load_ps(x);
		__m128 mc2 = _mm_loadr_ps(x+4);

		__m128 mt1 = _mm_add_ps(mc1,mc2);
		__m128 mt2 = _mm_sub_ps(mc1,mc2);//rev

		mc1 = _mm_addsub_ps(_mm_shuffle_ps(mt1,mt1,_MM_SHUFFLE(1,1,0,0)),_mm_shuffle_ps(mt1,mt1,_MM_SHUFFLE(2,2,3,3)));
		mc1 = _mm_shuffle_ps(mc1,mc1,_MM_SHUFFLE(0,2,3,1));

		_mm_store_ps(y,mc1);
		_mm_store_ps(y+4,mt2);

	}
	c0=y[0];
	c1=y[1];
	c2=y[2];
	c3=y[3];
	/*c3=y[0];
	c0=y[1];
	c2=y[2];
	c1=y[3];*/

	t7=y[4];
	t6=y[5];
	t5=y[6];
	t4=y[7];

	y[0] = c0 + c1;
	y[4] = c0 - c1;
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];

	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];

	y[5] = c3 - c1; y[3] = c0 - c2;
	c0 = (c0 + c2) * invsqrt2;
	c3 = (c3 + c1) * invsqrt2;
	y[1] = c0 + c3; y[7] = c0 - c3;

	const __m128 invsqh = _mm_set_ps1(invsqrt2h);
	__m128 my = _mm_load_ps(y);
	_mm_store_ps(y,_mm_mul_ps(my,invsqh));

	my = _mm_load_ps(y+4);
	_mm_store_ps(y+4,_mm_mul_ps(my,invsqh));
}
#endif

void fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(float* s, float threshold)
{
	fDCT2D8x4_32f(s, s);
	fDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
#ifdef _KEEP_00_COEF_
	fDCT2D8x4_and_threshold_keep00_32f(s, s, threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s,threshold);
#endif
	fDCT2D8x4_and_threshold_32f(s + 4, s + 4, threshold);
	//ommiting transform
	//transpose8x8(s);
	//transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);

	return;
}

int fDCT8x8_32f_and_threshold_and_iDCT8x8_nonzero_32f(float* s, float threshold)
{
	fDCT2D8x4_32f(s, s);
	fDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
#ifdef _KEEP_00_COEF_
	fDCT2D8x4_and_threshold_keep00_32f(s, s, threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s,threshold);
#endif
	fDCT2D8x4_and_threshold_32f(s + 4, s + 4, threshold);
	int ret = getNonzero(s, 64);
	//ommiting transform
	//transpose8x8(s);
	//transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);

	return ret;
}


//2x2

void dct1d2_32f(float* src, float* dest)
{
	dest[0] = 0.7071067812f*(src[0] + src[1]);
	dest[1] = 0.7071067812f*(src[0] - src[1]);
}

void fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack(float* src, float* dest, float thresh)
{
	__m128 ms0 = _mm_load_ps(src);
	__m128 ms1 = _mm_load_ps(src + 4);
	const __m128 mm = _mm_set1_ps(0.5f);
	__m128 a = _mm_add_ps(ms0, ms1);
	__m128 b = _mm_sub_ps(ms0, ms1);

	__m128 t1 = _mm_unpacklo_ps(a, b);
	__m128 t2 = _mm_unpackhi_ps(a, b);
	ms0 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(1, 0, 1, 0));
	ms1 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(3, 2, 3, 2));

	a = _mm_mul_ps(mm, _mm_add_ps(ms0, ms1));
	b = _mm_mul_ps(mm, _mm_sub_ps(ms0, ms1));

	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);

	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(a, *(const __m128*)v32f_absmask), mth);
	ms0 = _mm_blendv_ps(_mm_setzero_ps(), a, msk);
#ifdef _KEEP_00_COEF_
	ms0 = _mm_blend_ps(ms0, a, 1);
#endif
	msk = _mm_cmpgt_ps(_mm_and_ps(b, *(const __m128*)v32f_absmask), mth);
	ms1 = _mm_blendv_ps(_mm_setzero_ps(), b, msk);

	a = _mm_add_ps(ms0, ms1);
	b = _mm_sub_ps(ms0, ms1);

	t1 = _mm_unpacklo_ps(a, b);
	t2 = _mm_unpackhi_ps(a, b);
	ms0 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(1, 0, 1, 0));
	ms1 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(3, 2, 3, 2));

	a = _mm_mul_ps(mm, _mm_add_ps(ms0, ms1));
	b = _mm_mul_ps(mm, _mm_sub_ps(ms0, ms1));

	_mm_store_ps(dest, a);
	_mm_store_ps(dest + 4, b);
}

void DCT2x2_32f(float* src, float* dest, float* temp)
{
	dct1d2_32f(src, temp);
	dct1d2_32f(src + 2, temp + 2);
	float v = temp[1];
	temp[1] = temp[2];
	temp[2] = v;

	dct1d2_32f(temp, dest);
	dct1d2_32f(temp + 2, dest + 2);

	v = dest[1];
	dest[1] = dest[2];
	dest[2] = v;
}

#define fDCT2x2_32f DCT2x2_32f
#define iDCT2x2_32f DCT2x2_32f

void dct1d2_32f_and_thresh(float* src, float* dest, float thresh)
{
	float v = 0.7071068f*(src[0] + src[1]);
	dest[0] = (fabs(v) < thresh) ? 0.f : v;
	v = 0.7071068f*(src[0] - src[1]);
	dest[1] = (fabs(v) < thresh) ? 0 : v;
}
void fDCT2x2_32f_and_threshold(float* src, float* dest, float* temp, float thresh)
{
	dct1d2_32f(src, temp);
	dct1d2_32f(src + 2, temp + 2);
	float v = temp[1];
	temp[1] = temp[2];
	temp[2] = v;

	dct1d2_32f_and_thresh(temp, dest, thresh);
	dct1d2_32f_and_thresh(temp + 2, dest + 2, thresh);
	//dct1d2_32f(temp,dest);
	//dct1d2_32f(temp+2,dest+2);

	v = dest[1];
	dest[1] = dest[2];
	dest[2] = v;
}

void dct4x4_1d_llm_fwd_sse(float* s, float* d)//8add, 4 mul
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	_mm_store_ps(d, _mm_add_ps(p03, p12));
	_mm_store_ps(d + 4, _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12)));
	_mm_store_ps(d + 8, _mm_sub_ps(p03, p12));
	_mm_store_ps(d + 12, _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12)));
}

void dct4x4_1d_llm_fwd_sse_and_transpose(float* s, float* d)//8add, 4 mul
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	s0 = _mm_add_ps(p03, p12);
	s1 = _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12));
	s2 = _mm_sub_ps(p03, p12);
	s3 = _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12));
	_MM_TRANSPOSE4_PS(s0, s1, s2, s3);
	_mm_store_ps(d, s0);
	_mm_store_ps(d + 4, s1);
	_mm_store_ps(d + 8, s2);
	_mm_store_ps(d + 12, s3);
}

void dct4x4_1d_llm_inv_sse(float* s, float* d)
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 t10 = _mm_add_ps(s0, s2);
	__m128 t12 = _mm_sub_ps(s0, s2);

	__m128 t0 = _mm_add_ps(_mm_mul_ps(c2, s1), _mm_mul_ps(c6, s3));
	__m128 t2 = _mm_sub_ps(_mm_mul_ps(c6, s1), _mm_mul_ps(c2, s3));

	_mm_store_ps(d, _mm_add_ps(t10, t0));
	_mm_store_ps(d + 4, _mm_add_ps(t12, t2));
	_mm_store_ps(d + 8, _mm_sub_ps(t12, t2));
	_mm_store_ps(d + 12, _mm_sub_ps(t10, t0));
}

void dct4x4_1d_llm_inv_sse_and_transpose(float* s, float* d)
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 t10 = _mm_add_ps(s0, s2);
	__m128 t12 = _mm_sub_ps(s0, s2);

	__m128 t0 = _mm_add_ps(_mm_mul_ps(c2, s1), _mm_mul_ps(c6, s3));
	__m128 t2 = _mm_sub_ps(_mm_mul_ps(c6, s1), _mm_mul_ps(c2, s3));

	s0 = _mm_add_ps(t10, t0);
	s1 = _mm_add_ps(t12, t2);
	s2 = _mm_sub_ps(t12, t2);
	s3 = _mm_sub_ps(t10, t0);
	_MM_TRANSPOSE4_PS(s0, s1, s2, s3);
	_mm_store_ps(d, s0);
	_mm_store_ps(d + 4, s1);
	_mm_store_ps(d + 8, s2);
	_mm_store_ps(d + 12, s3);
}

void dct4x4_llm_sse(float* a, float* b, float* temp, int flag)
{
	if (flag == 0)
	{
		dct4x4_1d_llm_fwd_sse(a, temp);
		transpose4x4(temp);
		dct4x4_1d_llm_fwd_sse(temp, b);
		transpose4x4(b);
		__m128 c = _mm_set1_ps(0.250f);
		_mm_store_ps(b, _mm_mul_ps(_mm_load_ps(b), c));
		_mm_store_ps(b + 4, _mm_mul_ps(_mm_load_ps(b + 4), c));
		_mm_store_ps(b + 8, _mm_mul_ps(_mm_load_ps(b + 8), c));
		_mm_store_ps(b + 12, _mm_mul_ps(_mm_load_ps(b + 12), c));
	}
	else
	{
		dct4x4_1d_llm_inv_sse(a, temp);
		transpose4x4(temp);
		dct4x4_1d_llm_inv_sse(temp, b);
		transpose4x4(b);
		__m128 c = _mm_set1_ps(0.250f);
		_mm_store_ps(b, _mm_mul_ps(_mm_load_ps(b), c));
		_mm_store_ps(b + 4, _mm_mul_ps(_mm_load_ps(b + 4), c));
		_mm_store_ps(b + 8, _mm_mul_ps(_mm_load_ps(b + 8), c));
		_mm_store_ps(b + 12, _mm_mul_ps(_mm_load_ps(b + 12), c));
	}
}
void fDCT2D4x4_and_threshold_keep00_32f(float* s, float* d, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	__m128 v = _mm_add_ps(p03, p12);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	// keep 00 coef.
	__m128 v2 = _mm_blendv_ps(zeros, v, msk);
	v2 = _mm_blend_ps(v2, v, 1);
	_mm_store_ps(d, v2);

	v = _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 4, v);

	v = _mm_sub_ps(p03, p12);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 8, v);

	v = _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 12, v);
}

void fDCT2D4x4_and_threshold_32f(float* s, float* d, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	__m128 v = _mm_add_ps(p03, p12);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d, v);

	v = _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 4, v);

	v = _mm_sub_ps(p03, p12);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 8, v);

	v = _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 12, v);
}

void fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(float* s, float threshold)
{
	dct4x4_1d_llm_fwd_sse_and_transpose(s, s);
#ifdef _KEEP_00_COEF_
	fDCT2D4x4_and_threshold_keep00_32f(s, s, 4 * threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s,4*threshold);
#endif
	//ommiting transform
	//transpose4x4(s);
	dct4x4_1d_llm_inv_sse_and_transpose(s, s);//transpose4x4(s);
	dct4x4_1d_llm_inv_sse(s, s);
	//ommiting transform
	//transpose4x4(s);

	__m128 c = _mm_set1_ps(0.06250f);
	_mm_store_ps(s, _mm_mul_ps(_mm_load_ps(s), c));
	_mm_store_ps(s + 4, _mm_mul_ps(_mm_load_ps(s + 4), c));
	_mm_store_ps(s + 8, _mm_mul_ps(_mm_load_ps(s + 8), c));
	_mm_store_ps(s + 12, _mm_mul_ps(_mm_load_ps(s + 12), c));
}

int fDCT4x4_32f_and_threshold_and_iDCT4x4_nonzero_32f(float* s, float threshold)
{
	dct4x4_1d_llm_fwd_sse_and_transpose(s, s);
#ifdef _KEEP_00_COEF_
	fDCT2D4x4_and_threshold_keep00_32f(s, s, 4 * threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s,4*threshold);
#endif
	//ommiting transform
	//transpose4x4(s);
	int v = getNonzero(s, 16);
	dct4x4_1d_llm_inv_sse_and_transpose(s, s);//transpose4x4(s);
	dct4x4_1d_llm_inv_sse(s, s);
	//ommiting transform
	//transpose4x4(s);

	__m128 c = _mm_set1_ps(0.06250f);
	_mm_store_ps(s, _mm_mul_ps(_mm_load_ps(s), c));
	_mm_store_ps(s + 4, _mm_mul_ps(_mm_load_ps(s + 4), c));
	_mm_store_ps(s + 8, _mm_mul_ps(_mm_load_ps(s + 8), c));
	_mm_store_ps(s + 12, _mm_mul_ps(_mm_load_ps(s + 12), c));
	return v;
}