#include <nmmintrin.h> //SSE4.2
#include <string.h>
#include <stdio.h>
void transpose4x4(float* src)
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

void transpose4x4(float* src, float* dest)
{
	__m128 m0 = _mm_load_ps(src);
	__m128 m1 = _mm_load_ps(src + 4);
	__m128 m2 = _mm_load_ps(src + 8);
	__m128 m3 = _mm_load_ps(src + 12);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest, m0);
	_mm_store_ps(dest + 4, m1);
	_mm_store_ps(dest + 8, m2);
	_mm_store_ps(dest + 12, m3);
}

void transpose8x8(const float* src, float* dest)
{
	__m128 m0 = _mm_load_ps(src);
	__m128 m1 = _mm_load_ps(src + 8);
	__m128 m2 = _mm_load_ps(src + 16);
	__m128 m3 = _mm_load_ps(src + 24);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest, m0);
	_mm_store_ps(dest + 8, m1);
	_mm_store_ps(dest + 16, m2);
	_mm_store_ps(dest + 24, m3);

	m0 = _mm_load_ps(src + 4);
	m1 = _mm_load_ps(src + 12);
	m2 = _mm_load_ps(src + 20);
	m3 = _mm_load_ps(src + 28);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest + 32, m0);
	_mm_store_ps(dest + 40, m1);
	_mm_store_ps(dest + 48, m2);
	_mm_store_ps(dest + 56, m3);

	m0 = _mm_load_ps(src + 32);
	m1 = _mm_load_ps(src + 40);
	m2 = _mm_load_ps(src + 48);
	m3 = _mm_load_ps(src + 56);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest + 4, m0);
	_mm_store_ps(dest + 12, m1);
	_mm_store_ps(dest + 20, m2);
	_mm_store_ps(dest + 28, m3);

	m0 = _mm_load_ps(src + 36);
	m1 = _mm_load_ps(src + 44);
	m2 = _mm_load_ps(src + 52);
	m3 = _mm_load_ps(src + 60);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest + 36, m0);
	_mm_store_ps(dest + 44, m1);
	_mm_store_ps(dest + 52, m2);
	_mm_store_ps(dest + 60, m3);
}

void transpose8x8(float* src)
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

void transpose16x16(float* src)
{
	__declspec(align(16)) float temp[64];
	__declspec(align(16)) float tmp[64];
	int sz = sizeof(float) * 8;
	for (int i = 0; i < 8; i++)
	{
		memcpy(temp + 8 * i, src + 16 * i, sz);
	}
	transpose8x8(temp);
	for (int i = 0; i < 8; i++)
	{
		memcpy(src + 16 * i, temp + 8 * i, sz);
	}

	for (int i = 0; i < 8; i++)
	{
		memcpy(tmp + 8 * i, src + 16 * i + 8, sz);
		memcpy(temp + 8 * i, src + 16 * (i + 8), sz);
	}
	transpose8x8(tmp);
	transpose8x8(temp);
	for (int i = 0; i < 8; i++)
	{
		memcpy(src + 16 * i + 8, temp + 8 * i, sz);
		memcpy(src + 16 * (i + 8), tmp + 8 * i, sz);
	}

	for (int i = 0; i < 8; i++)
	{
		memcpy(temp + 8 * i, src + 16 * (i + 8) + 8, sz);
	}
	transpose8x8(temp);
	for (int i = 0; i < 8; i++)
	{
		memcpy(src + 16 * (i + 8) + 8, temp + 8 * i, sz);
	}
}