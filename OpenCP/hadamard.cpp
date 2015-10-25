#include <nmmintrin.h> //SSE4.2
#include <string.h>
#include <stdio.h>
void transpose4x4(float* src);
void transpose4x4(float* src, float* dest);
void transpose8x8(float* src);
void transpose8x8(const float* src, float* dest);
void transpose16x16(float* src);

void print_m128(__m128 src);
#define _KEEP_00_COEF_

void Hadamard1D4(float *val)
{
	__m128 xmm0, xmm1, xmm2, xmm3;
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };

	xmm2 = _mm_load_ps(sign[0]);
	xmm3 = _mm_load_ps(sign[1]);

	xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	xmm0 = _mm_add_ps(xmm0, xmm1);
	_mm_store_ps(val, xmm0);
};

void Hadamard1D8(float *val)
{
	__m128 xmm0, xmm1, xmm2;

	__declspec(align(16)) float sign0[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	__declspec(align(16)) float sign1[4] = { 1.0f, 1.0f, -1.0f, -1.0f };

	xmm2 = _mm_load_ps(sign0);

	xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

	xmm2 = _mm_load_ps(sign1);

	xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

	xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

	xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
	xmm2 = mmaddvalue1; //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
	xmm0 = _mm_add_ps(xmm0, xmm2);
	_mm_store_ps(val, xmm0);
	xmm1 = _mm_sub_ps(xmm1, xmm2);
	_mm_store_ps(val + 4, xmm1);
};

void Hadamard1D16(float *val)
{
	__m128 xmm0, xmm1, xmm2;
	__m128 mmadd0, mmadd1, mmadd2, mmadd3;
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };

	xmm2 = _mm_load_ps(sign[0]);


	xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd0 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd1 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 8); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd2 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 12); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd3 = _mm_add_ps(xmm0, xmm1);


	////////////
	xmm2 = _mm_load_ps(sign[1]);

	xmm1 = xmm0 = mmadd0; //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd0 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = mmadd1;
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd1 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = mmadd2;
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd2 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = mmadd3;
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd3 = _mm_add_ps(xmm0, xmm1);

	////////////

	xmm0 = xmm1 = mmadd0; //x[p+1] x[p+2] x[p+3] x[p+4]
	xmm2 = mmadd1; //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
	mmadd0 = _mm_add_ps(xmm0, xmm2);
	mmadd1 = _mm_sub_ps(xmm1, xmm2);

	xmm0 = xmm1 = mmadd2;
	xmm2 = mmadd3;
	mmadd2 = _mm_add_ps(xmm0, xmm2);
	mmadd3 = _mm_sub_ps(xmm1, xmm2);

	_mm_store_ps(val, _mm_add_ps(mmadd0, mmadd2));
	_mm_store_ps(val + 4, _mm_add_ps(mmadd1, mmadd3));
	_mm_store_ps(val + 8, _mm_sub_ps(mmadd0, mmadd2));
	_mm_store_ps(val + 12, _mm_sub_ps(mmadd1, mmadd3));
};


void Hadamard1D16x16(float *val)
{
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };
	const __m128 sgn0 = _mm_load_ps(sign[0]);
	const __m128 sgn1 = _mm_load_ps(sign[1]);

	for (int i = 0; i < 16; i++)
	{
		__m128 xmm0, xmm1, xmm2;
		__m128 mmadd0, mmadd1, mmadd2, mmadd3;


		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 8); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd2 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 12); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd3 = _mm_add_ps(xmm0, xmm1);


		////////////

		xmm1 = xmm0 = mmadd0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmadd1;
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmadd2;
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd2 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmadd3;
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd3 = _mm_add_ps(xmm0, xmm1);

		////////////

		xmm0 = xmm1 = mmadd0; //x[p+1] x[p+2] x[p+3] x[p+4]
		xmm2 = mmadd1; //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		mmadd0 = _mm_add_ps(xmm0, xmm2);
		mmadd1 = _mm_sub_ps(xmm1, xmm2);

		xmm0 = xmm1 = mmadd2;
		xmm2 = mmadd3;
		mmadd2 = _mm_add_ps(xmm0, xmm2);
		mmadd3 = _mm_sub_ps(xmm1, xmm2);

		_mm_store_ps(val, _mm_add_ps(mmadd0, mmadd2));
		_mm_store_ps(val + 4, _mm_add_ps(mmadd1, mmadd3));
		_mm_store_ps(val + 8, _mm_sub_ps(mmadd0, mmadd2));
		_mm_store_ps(val + 12, _mm_sub_ps(mmadd1, mmadd3));
		val += 16;
	}
};



void Hadamard1Dn(float *val, size_t n)
{
	size_t i, j, k;
	__m128 xmm0, xmm1, xmm2;
	float *addvalue, *subvalue;
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };

	xmm2 = _mm_load_ps(sign[0]);
	for (i = 0, addvalue = val; i < n; i += 4, addvalue++)
	{
		xmm1 = xmm0 = _mm_load_ps(addvalue); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(addvalue, xmm0);
	}

	xmm2 = _mm_load_ps(sign[1]);
	for (i = 0, addvalue = val; i < n; i += 4, addvalue++)
	{
		xmm1 = xmm0 = _mm_load_ps(addvalue); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(addvalue, xmm0);
	}

	for (i = 4; i < n; i <<= 1)
	{
		for (j = 0; j < n; j += i * 2)
		{
			addvalue = (val + j + 0);
			subvalue = (val + j + i);
			for (k = 0; k < i; k += 4, addvalue++, subvalue++)
			{
				xmm0 = xmm1 = _mm_load_ps(addvalue); //x[p+1] x[p+2] x[p+3] x[p+4]
				xmm2 = _mm_load_ps(subvalue); //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
				xmm0 = _mm_add_ps(xmm0, xmm2);
				xmm1 = _mm_sub_ps(xmm1, xmm2);
				_mm_store_ps(addvalue, xmm0);
				_mm_store_ps(subvalue, xmm1);
			}
		}
	}
};

void divval(float* src, int size, float div)
{
	const __m128 h = _mm_set1_ps(div);
	for (int i = 0; i < size; i += 4)
	{
		_mm_store_ps(src + i, _mm_mul_ps(_mm_load_ps(src + i), h));
	}
}



void divvalandthresh(float* src, int size, float thresh, float div)
{
	const __m128 h = _mm_set1_ps(div);
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	for (int i = 0; i < size; i += 4)
	{
		__m128 v = _mm_mul_ps(_mm_load_ps(src + i), h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(src + i, v);
	}
}

void Hadamard2D16x16(float* src)
{
	for (int i = 0; i < 16; i++)
		Hadamard1D16(src + 16 * i);


	transpose16x16(src);

	for (int i = 0; i < 16; i++)
		Hadamard1D16(src + 16 * i);

	transpose16x16(src);
	divval(src, 256, 0.0625f);
}

void Hadamard2D16x16andThreshandIDHT(float* src, float thresh)
{
	Hadamard1D16x16(src);
	transpose16x16(src);
	Hadamard1D16x16(src);
#ifdef _KEEP_00_COEF_
	float f0 = src[0] * 0.0625f;
#endif
	divvalandthresh(src, 256, thresh, 0.0625f);
#ifdef _KEEP_00_COEF_
	src[0] = f0;
#endif
	Hadamard1D16x16(src);
	transpose16x16(src);
	Hadamard1D16x16(src);

	divval(src, 256, 0.0625f);
}

void Hadamard2D4x4(float* src)
{
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	divval(src, 16, 0.5f);
	transpose4x4(src);
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	transpose4x4(src);
	divval(src, 16, 0.5f);
}

void Hadamard2D4x4andThresh(float* src, float thresh)
{
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	divval(src, 16, 0.5f);
	transpose4x4(src);
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	transpose4x4(src);
	divvalandthresh(src, 16, thresh, 0.5f);
}

void Hadamard2D4x4andThreshandIDHT(float* src, float thresh)
{
	const __m128 h = _mm_set1_ps(0.25f);
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };
	float* val = src;

	for (int i = 0; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(val, xmm0);
		val += 4;
	};
	{
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 4);
		__m128 m2 = _mm_load_ps(src + 8);
		__m128 m3 = _mm_load_ps(src + 12);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 4, m1);
		_mm_store_ps(src + 8, m2);
		_mm_store_ps(src + 12, m3);
	}
	val = src;
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);

		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
#ifdef _KEEP_00_COEF_
		__m128 v2 = _mm_blendv_ps(zeros, v, msk);
		v2 = _mm_blend_ps(v2, v, 1);
		_mm_store_ps(val, v2);
#else
		v = _mm_blendv_ps(zeros,v,msk);
		_mm_store_ps(val,v);
#endif		
		val += 4;
	}
	for (int i = 1; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);

		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);

		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val, v);

		val += 4;
	}

	val = src;
	for (int i = 0; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(val, xmm0);
		val += 4;
	}
	{
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 4);
		__m128 m2 = _mm_load_ps(src + 8);
		__m128 m3 = _mm_load_ps(src + 12);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 4, m1);
		_mm_store_ps(src + 8, m2);
		_mm_store_ps(src + 12, m3);
	}
	val = src;
	for (int i = 0; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(val, _mm_mul_ps(xmm0, h));

		val += 4;
	};

}

void Hadamard2D8x8(float* src)
{
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	divval(src, 64, 0.5f);
	transpose8x8(src);
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	transpose8x8(src);
	divval(src, 64, 0.5f);
}

void Hadamard2D8x8andThresh(float* src, float thresh)
{
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	divval(src, 64, 0.5f);
	transpose8x8(src);
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	transpose8x8(src);
	divvalandthresh(src, 64, thresh, 0.25f);
}


void Hadamard2D8x8i(float *vall)
{
	float* val = vall;
	__declspec(align(16)) float sign0[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	__declspec(align(16)) float sign1[4] = { 1.0f, 1.0f, -1.0f, -1.0f };

	const __m128 sgn0 = _mm_load_ps(sign0);
	const __m128 sgn1 = _mm_load_ps(sign1);
	__m128 xmm0, xmm1;
	for (int i = 0; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		_mm_store_ps(val, xmm0);
		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		_mm_store_ps(val + 4, xmm1);
		val += 8;
	}

	float* src = vall;
	{
		__declspec(align(16)) float temp[16];
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 8);
		__m128 m2 = _mm_load_ps(src + 16);
		__m128 m3 = _mm_load_ps(src + 24);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 8, m1);
		_mm_store_ps(src + 16, m2);
		_mm_store_ps(src + 24, m3);


		m0 = _mm_load_ps(src + 4);
		m1 = _mm_load_ps(src + 12);
		m2 = _mm_load_ps(src + 20);
		m3 = _mm_load_ps(src + 28);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		/*_mm_store_ps(dest+32,m0);
		_mm_store_ps(dest+40,m1);
		_mm_store_ps(dest+48,m2);
		_mm_store_ps(dest+56,m3);*/
		_mm_store_ps(temp, m0);
		_mm_store_ps(temp + 4, m1);
		_mm_store_ps(temp + 8, m2);
		_mm_store_ps(temp + 12, m3);

		m0 = _mm_load_ps(src + 32);
		m1 = _mm_load_ps(src + 40);
		m2 = _mm_load_ps(src + 48);
		m3 = _mm_load_ps(src + 56);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 4, m0);
		_mm_store_ps(src + 12, m1);
		_mm_store_ps(src + 20, m2);
		_mm_store_ps(src + 28, m3);

		memcpy(src + 32, temp, sizeof(float) * 4);
		memcpy(src + 40, temp + 4, sizeof(float) * 4);
		memcpy(src + 48, temp + 8, sizeof(float) * 4);
		memcpy(src + 56, temp + 12, sizeof(float) * 4);


		m0 = _mm_load_ps(src + 36);
		m1 = _mm_load_ps(src + 44);
		m2 = _mm_load_ps(src + 52);
		m3 = _mm_load_ps(src + 60);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 36, m0);
		_mm_store_ps(src + 44, m1);
		_mm_store_ps(src + 52, m2);
		_mm_store_ps(src + 60, m3);
	}
	val = vall;
	const __m128 h = _mm_set1_ps(0.125f);
	for (int i = 0; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		_mm_store_ps(val, _mm_mul_ps(h, xmm0));
		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		_mm_store_ps(val + 4, _mm_mul_ps(h, xmm1));
		val += 8;
	}
};

void Hadamard2D8x8i_and_thresh(float *vall, float thresh)
{
	float* val = vall;
	__declspec(align(16)) float sign0[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	__declspec(align(16)) float sign1[4] = { 1.0f, 1.0f, -1.0f, -1.0f };

	const __m128 sgn0 = _mm_load_ps(sign0);
	const __m128 sgn1 = _mm_load_ps(sign1);
	__m128 xmm0, xmm1;
	for (int i = 0; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		_mm_store_ps(val, xmm0);
		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		_mm_store_ps(val + 4, xmm1);
		val += 8;
	}

	float* src = vall;
	{
		__declspec(align(16)) float temp[16];
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 8);
		__m128 m2 = _mm_load_ps(src + 16);
		__m128 m3 = _mm_load_ps(src + 24);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 8, m1);
		_mm_store_ps(src + 16, m2);
		_mm_store_ps(src + 24, m3);


		m0 = _mm_load_ps(src + 4);
		m1 = _mm_load_ps(src + 12);
		m2 = _mm_load_ps(src + 20);
		m3 = _mm_load_ps(src + 28);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		/*_mm_store_ps(dest+32,m0);
		_mm_store_ps(dest+40,m1);
		_mm_store_ps(dest+48,m2);
		_mm_store_ps(dest+56,m3);*/
		_mm_store_ps(temp, m0);
		_mm_store_ps(temp + 4, m1);
		_mm_store_ps(temp + 8, m2);
		_mm_store_ps(temp + 12, m3);

		m0 = _mm_load_ps(src + 32);
		m1 = _mm_load_ps(src + 40);
		m2 = _mm_load_ps(src + 48);
		m3 = _mm_load_ps(src + 56);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 4, m0);
		_mm_store_ps(src + 12, m1);
		_mm_store_ps(src + 20, m2);
		_mm_store_ps(src + 28, m3);

		memcpy(src + 32, temp, sizeof(float) * 4);
		memcpy(src + 40, temp + 4, sizeof(float) * 4);
		memcpy(src + 48, temp + 8, sizeof(float) * 4);
		memcpy(src + 56, temp + 12, sizeof(float) * 4);


		m0 = _mm_load_ps(src + 36);
		m1 = _mm_load_ps(src + 44);
		m2 = _mm_load_ps(src + 52);
		m3 = _mm_load_ps(src + 60);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 36, m0);
		_mm_store_ps(src + 44, m1);
		_mm_store_ps(src + 52, m2);
		_mm_store_ps(src + 60, m3);
	}
	val = vall;

	const __m128 h = _mm_set1_ps(0.125f);
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
#ifdef _KEEP_00_COEF_
		__m128 v2 = _mm_blendv_ps(zeros, v, msk);
		v2 = _mm_blend_ps(v2, v, 1);
		_mm_store_ps(val, v2);
#else
		v = _mm_blendv_ps(zeros,v,msk);
		_mm_store_ps(val,v);
#endif


		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		v = _mm_mul_ps(xmm1, h);
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val + 4, v);
		val += 8;
	}
	for (int i = 1; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val, v);

		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		v = _mm_mul_ps(xmm1, h);
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val + 4, v);
		val += 8;
	}
};
void Hadamard2D8x8andThreshandIDHT(float* src, float thresh)
{
	Hadamard2D8x8i_and_thresh(src, thresh);
	Hadamard2D8x8i(src);
}