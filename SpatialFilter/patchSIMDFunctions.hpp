#pragma once
#include <iostream>
#include <inlineSIMDFunctions.hpp>
#include <intrin.h>

enum
{
	SSE_ALIGNMENT = 16,
	AVX_ALIGNMENT = 32,
	AVX512_ALIGNMENT = 64,
};

typedef union __declspec(intrin_type)_CRT_ALIGN(4) __m32
{
	//unsigned __int64    m64_u64;
	float               m64_f32;
	__int8              m64_i8[4];
	__int16             m64_i16[2];
	__int32             m64_i32;
	//__int64             m64_i64;
	unsigned __int8     m64_u8[4];
	unsigned __int16    m64_u16[2];
	unsigned __int32    m64_u32;
} __m32;


template<typename destT>
inline void store_auto(destT* dest, __m256 src)
{
	;
}

template<>
inline void store_auto<float>(float* dest, __m256 src)
{
	_mm256_storeu_ps(dest, src);
}

template<>
inline void store_auto<uchar>(uchar* dest, __m256 src)
{
	_mm_storel_epi64((__m128i*)dest, _mm256_cvtps_epu8(src));
}

template<typename destT>
inline void store_auto(destT* dest, __m256d src)
{
	;
}

template<>
inline void store_auto<double>(double* dest, __m256d src)
{
	_mm256_storeu_pd(dest, src);
}

template<>
inline void store_auto<float>(float* dest, __m256d src)
{
	_mm_storeu_ps(dest, _mm256_cvtpd_ps(src));
}

template<>
inline void store_auto<uchar>(uchar* dest, __m256d src)
{
	//_mm_storel_epi64((__m128i*)dest, _mm_cvtepi32_epu8(_mm_cvtps_epi32(_mm256_cvtpd_ps(src))));
	uchar b[8];
	_mm_storel_epi64((__m128i*)b, _mm_cvtepi32_epu8(_mm_cvtps_epi32(_mm256_cvtpd_ps(src))));

	dest[0] = b[0];
	dest[1] = b[1];
	dest[2] = b[2];
	dest[3] = b[3];
}