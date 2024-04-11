#pragma once
#ifndef __CP_INLINE_SIMD_FUNCTIONS__
#define __CP_INLINE_SIMD_FUNCTIONS__

#pragma warning(disable:4309)

#ifdef __INTEL_LLVM_COMPILER //#ifdef __INTEL_COMPILER
#include <immintrin.h>
//#define __INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES 1

//#define _mm512_permutex2var_epi8(a, b, c) _mm512_setzero_si512()
#else
#include <intrin.h>
#define __AVX512FP16__
#endif

#include <opencv2/core.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#define SSE_ALIGN 16
#define AVX_ALIGN 32
#define AVX512_ALIGN 64

//#define STATIC_INLINE static __forceinline
#define STATIC_INLINE static inline

//0 src1 low, 1 src1 high, //2 src2 low, 3 src2 high, 
#define _MM_SELECT4(x,y) (((y)<<4) + (x))


//#define CP_ROUND_NEAREST _MM_FROUND_TO_NEAREST_INT 
//#define CP_ROUND_NEAREST _MM_ROUND_MODE_NEAREST
#define CP_ROUND_NEAREST 0

//#ifdef __AVX512F__
#ifdef __INTEL_LLVM_COMPILER
//#define _mm_setr_epi64x _mm_setr_epi64

STATIC_INLINE __m512i _mm512_setr_epi8(char s0, char s1, char s2, char s3, char s4, char s5, char s6, char s7, char s8, char s9, char s10, char s11, char s12, char s13, char s14, char s15, char s16, char s17, char s18, char s19, char s20, char s21, char s22, char s23, char s24, char s25, char s26, char s27, char s28, char s29, char s30, char s31, char s32, char s33, char s34, char s35, char s36, char s37, char s38, char s39, char s40, char s41, char s42, char s43, char s44, char s45, char s46, char s47, char s48, char s49, char s50, char s51, char s52, char s53, char s54, char s55, char s56, char s57, char s58, char s59, char s60, char s61, char s62, char s63)
{
	return _mm512_set_epi8(s63, s62, s61, s60, s59, s58, s57, s56, s55, s54, s53, s52, s51, s50, s49, s48, s47, s46, s45, s44, s43, s42, s41, s40, s39, s38, s37, s36, s35, s34, s33, s32, s31, s30, s29, s28, s27, s26, s25, s24, s23, s22, s21, s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0);
}

#define _mm512_permutex2var_ph _mm512_permutex2var_epi16
#define _mm512_mask_blend_ph _mm512_mask_blend_epi16
#endif
//#endif

//#define AVX512_VBMI //enable _mm512_permutex2var_epi8 >=TigerLake 

//#define USE_SET4GATHER //for AMD CPU
//#define UNUSE_FMA

#pragma region swich_for_AVX(sandybridge_ivybridge)
static inline __m256i _my256_cvtepu8_epi32(__m128i src)
{
	const __m128i a16 = _mm_unpacklo_epi8(src, _mm_setzero_si128());
	return _mm256_set_m128i(_mm_unpackhi_epi16(a16, _mm_setzero_si128()), _mm_unpacklo_epi16(a16, _mm_setzero_si128()));
}
#ifndef __AVX2__
#define _mm256_cvtepu8_epi32(a) _my256_cvtepu8_epi32(a)

#ifdef AMD_OPTIMIZATION
inline __m256 _mm256_i32gather_ps(float* s, __m256i index, int val)
{
	return _mm256_setr_ps(s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]], s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]]);
}

#define _mm256_permute4x64_ps(src,imm8) (_mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(src), imm8)))
//for _MM_SHUFFLE(3, 1, 2, 0) 
/*
inline __m256 _mm256_permute4x64_ps(__m256 src, const int imm8)
{
	//perm128x1, shuffle_pd(1,0.5) x2
	__m256 tmp = _mm256_permute2f128_ps(src, src, 0x01);
	__m256d tm2 = _mm256_shuffle_pd(_mm256_castps_pd(src), _mm256_castps_pd(tmp), 0b1100);
	__m256 ret = _mm256_castpd_ps(_mm256_shuffle_pd(tm2, tm2, 0b0110));
	return ret;
}
*/
/*
inline __m256 _mm256_permute4x64_ps(__m256 src, const int imm8)
{
	//perm128x1 shuffle_ps(1,0.5) x2, blend(1,0.33) x1
	__m256 tmp = _mm256_permute2f128_ps(src, src, 0x01);
	__m256 rt1 = _mm256_shuffle_ps(src, tmp, _MM_SHUFFLE(1, 0, 1, 0));
	__m256 rt2 = _mm256_shuffle_ps(tmp, src, _MM_SHUFFLE(3, 2, 3, 2));
	__m256 ret = _mm256_blend_ps(rt1, rt2, 0b11110000);
	return ret;
}*/
#else
#define _mm256_permute4x64_ps(src,imm8) (_mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(src), imm8)))
//#define _mm256_permute4x64_ps(src,imm8) (_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(src), imm8)))
#endif 
#endif

#if !defined(__AVX2__) || defined(UNUSE_FMA)
#define _mm256_fmadd_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#define _mm256_fmsub_ps(a,b,c) _mm256_sub_ps(_mm256_mul_ps(a,b),c)
#define _mm256_fnmadd_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(-1.f),a),b),c)
#define _mm256_fnmsub_ps(a,b,c) _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(-1.f),a),b),c)
#define _mm256_fmadd_pd(a,b,c) _mm256_add_pd(_mm256_mul_pd(a,b),c)
#define _mm256_fmsub_pd(a,b,c) _mm256_sub_pd(_mm256_mul_pd(a,b),c)
#define _mm256_fnmadd_pd(a,b,c) _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0),a),b),c)
#define _mm256_fnmsub_pd(a,b,c) _mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0),a),b),c)
#endif


STATIC_INLINE __m256 _mm256_i32gatherset_ps(const float* s, __m256i index, int offset)
{
	return _mm256_setr_ps(s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]], s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]]);
}
STATIC_INLINE __m256i _mm256_i32gatherset_epi32(const int* s, __m256i index, int offset)
{
	return _mm256_setr_epi32(s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]], s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]]);
}
#ifdef __AVX512F__
STATIC_INLINE __m512 _mm512_i32gatherset_ps(__m512i index, const float* s, int offset)
{
	return _mm512_setr_ps(
		s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]], 
		s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]],
		s[((int*)&index)[8]], s[((int*)&index)[9]], s[((int*)&index)[10]], s[((int*)&index)[11]], 
		s[((int*)&index)[12]], s[((int*)&index)[13]], s[((int*)&index)[14]], s[((int*)&index)[15]]
	);
}
STATIC_INLINE __m512i _mm512_i32gatherset_epi32(__m512i index, const int* s, int offset)
{
	return _mm512_setr_epi32(
		s[((int*)&index)[0]], s[((int*)&index)[1]], s[((int*)&index)[2]], s[((int*)&index)[3]],
		s[((int*)&index)[4]], s[((int*)&index)[5]], s[((int*)&index)[6]], s[((int*)&index)[7]],
		s[((int*)&index)[8]], s[((int*)&index)[9]], s[((int*)&index)[10]], s[((int*)&index)[11]],
		s[((int*)&index)[12]], s[((int*)&index)[13]], s[((int*)&index)[14]], s[((int*)&index)[15]]
	);
}
#endif
#if !defined(__AVX2__) || defined(USE_SET4GATHER)
#define _mm256_i32gather_force_ps(s, idx, val) _mm256_i32gather_ps(s, idx, val)
#define _mm256_i32gather_force_epi32(s, idx, val) _mm256_i32gather_epi32(s, idx, val)
#define _mm256_i32gather_ps(s, idx, val) _mm256_i32gatherset_ps(s, idx, val)
#define _mm256_i32gather_epi32(s, idx, val) _mm256_i32gatherset_epi32(s, idx, val)

#endif


#pragma endregion

#pragma region xxx
#pragma endregion

#pragma region utility
STATIC_INLINE std::string getRoundingModeName(const int mode)
{
	std::string ret = "";
	switch (mode)
	{
	case _MM_ROUND_NEAREST:		ret = "_MM_ROUND_NEAREST"; break;
	case _MM_ROUND_DOWN:		ret = "_MM_ROUND_DOWN"; break;
	case _MM_ROUND_UP:			ret = "_MM_ROUND_UP"; break;
	case _MM_ROUND_TOWARD_ZERO:	ret = "_MM_ROUND_TOWARD_ZERO"; break;
	default: break;
	}
	return ret;
}

STATIC_INLINE int _mm_cvtf32_i32(const float src)
{
	__m128i d = _mm_cvtps_epi32(_mm_set1_ps(src));
	return ((int*)&d)[0];
}
#pragma endregion

//template for array
//const int CV_DECL_ALIGNED(AVX_ALIGN) a[10]
#pragma region loop manipulation
STATIC_INLINE int get_loop_end(int begin, int end, int step)
{
	const int rem = ((end - begin) % step == 0) ? 0 : 1;
	const int count = (end - begin) / step + rem;
	const int ret = begin + count * step;

	/*
	int i = begin;
	for (; i < end; i += step)
	{
		;
	}
	cout <<begin<<": "<< i << "," << ret <<","<<end<< endl;
	*/
	return ret;
}

STATIC_INLINE int get_simd_ceil(const int val, const int simdwidth)
{
	return (val % simdwidth == 0) ? val : (val / simdwidth + 1) * simdwidth;
}

STATIC_INLINE int get_simd_floor(const int val, const int simdwidth)
{
	return (val / simdwidth) * simdwidth;
}

STATIC_INLINE void get_simd_width_end(const int cv_depth, const int channels, const int vector_length, const int image_width, int& dest_endwidth, int& dest_pad_pixels)
{
	if (cv_depth == CV_32F && image_width % vector_length == 0)
	{
		dest_endwidth = image_width;
		dest_pad_pixels = 0;
	}
	else if (cv_depth == CV_32F && image_width % vector_length != 0)
	{
		dest_endwidth = get_simd_floor(image_width, vector_length);
		dest_pad_pixels = (image_width - dest_endwidth) * channels;
	}
	else if (cv_depth == CV_8U)
	{
		dest_endwidth = get_simd_floor(image_width - vector_length, vector_length);
		dest_pad_pixels = (image_width - dest_endwidth) * channels;
	}
}

STATIC_INLINE __m256i get_simd_residualmask_epi32(const int width)
{
	const int rem = width - get_simd_floor(width, 8);
	__m256i ret = _mm256_undefined_si256();
	switch (rem)
	{
	case 0: ret = _mm256_cmpeq_epi32(_mm256_set1_epi32(1), _mm256_setzero_si256()); break;
	case 1: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 1, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256()); break;
	case 2: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 1, 1, 1, 1, 1, 1), _mm256_setzero_si256()); break;
	case 3: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 1, 1, 1, 1, 1), _mm256_setzero_si256()); break;
	case 4: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), _mm256_setzero_si256()); break;
	case 5: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 1, 1, 1), _mm256_setzero_si256()); break;
	case 6: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 1, 1), _mm256_setzero_si256()); break;
	case 7: ret = _mm256_cmpeq_epi32(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 1), _mm256_setzero_si256()); break;
	}
	return ret;
}

STATIC_INLINE __m256 get_simd_residualmask_ps(const int width)
{
	const int rem = width - get_simd_floor(width, 8);
	__m256 ret = _mm256_undefined_ps();
	switch (rem)
	{
	case 0: ret = _mm256_cmp_ps(_mm256_set1_ps(1), _mm256_setzero_ps(), 0); break;
	case 1: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 1, 1, 1, 1, 1, 1, 1), _mm256_setzero_ps(), 0); break;
	case 2: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 0, 1, 1, 1, 1, 1, 1), _mm256_setzero_ps(), 0); break;
	case 3: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 0, 0, 1, 1, 1, 1, 1), _mm256_setzero_ps(), 0); break;
	case 4: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 0, 0, 0, 1, 1, 1, 1), _mm256_setzero_ps(), 0); break;
	case 5: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 0, 0, 0, 0, 1, 1, 1), _mm256_setzero_ps(), 0); break;
	case 6: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 0, 0, 0, 0, 0, 1, 1), _mm256_setzero_ps(), 0); break;
	case 7: ret = _mm256_cmp_ps(_mm256_setr_ps(0, 0, 0, 0, 0, 0, 0, 1), _mm256_setzero_ps(), 0); break;
	}
	return ret;
}

STATIC_INLINE __m256d get_simd_residualmask_pd(const int width)
{
	const int rem = width - get_simd_floor(width, 4);
	__m256d ret = _mm256_undefined_pd();
	switch (rem)
	{
	case 0: ret = _mm256_cmp_pd(_mm256_set1_pd(1), _mm256_setzero_pd(), 0); break;
	case 1: ret = _mm256_cmp_pd(_mm256_setr_pd(0, 1, 1, 1), _mm256_setzero_pd(), 0); break;
	case 2: ret = _mm256_cmp_pd(_mm256_setr_pd(0, 0, 1, 1), _mm256_setzero_pd(), 0); break;
	case 3: ret = _mm256_cmp_pd(_mm256_setr_pd(0, 0, 0, 1), _mm256_setzero_pd(), 0); break;

	}
	return ret;
}

inline unsigned short get_simd512_residualmask_epi64(const int width)
{
	return UCHAR_MAX >> ((8 - (width % 8)) % 8);
}

#define get_simd512_residualmask_pd get_simd512_residualmask_epi64

inline unsigned short get_simd512_residualmask_epi32(const int width)
{
	return USHRT_MAX >> ((16 - (width % 16)) % 16);
}

#define get_simd512_residualmask_ps get_simd512_residualmask_epi32

inline unsigned int get_simd512_residualmask_epi16(const int width)
{
	return UINT_MAX >> ((32 - (width % 32)) % 32);
}

#define get_simd512_residualmask_ph get_simd512_residualmask_epi16

inline unsigned long long get_simd512_residualmask_epi8(const int width)
{
	return ULLONG_MAX >> ((64 - (width % 64)) % 64);
}

#pragma endregion

#pragma region transpose
static inline __m128i _mm_movehl_si128(const __m128i& xmm0)
{
	return _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm0),
		_mm_castsi128_ps(xmm0)));
}
//8 in 8 out
STATIC_INLINE void _mm_transpose8_epi8(__m128i& s0, __m128i& s1, __m128i& s2, __m128i& s3, __m128i& s4, __m128i& s5, __m128i& s6, __m128i& s7)
{
	s0 = _mm_unpacklo_epi8(s0, s1);
	s1 = _mm_unpacklo_epi8(s2, s3);
	s2 = _mm_unpacklo_epi8(s4, s5);
	s3 = _mm_unpacklo_epi8(s6, s7);

	s4 = _mm_unpacklo_epi16(s0, s1);
	s5 = _mm_unpackhi_epi16(s0, s1);
	s6 = _mm_unpacklo_epi16(s2, s3);
	s7 = _mm_unpackhi_epi16(s2, s3);

	s0 = _mm_unpacklo_epi32(s4, s6);
	s1 = _mm_movehl_si128(s0);
	s2 = _mm_unpackhi_epi32(s4, s6);
	s3 = _mm_movehl_si128(s2);
	s4 = _mm_unpacklo_epi32(s5, s7);
	s6 = s5;
	s5 = _mm_movehl_si128(s4);
	s6 = _mm_unpackhi_epi32(s6, s7);
	s7 = _mm_movehl_si128(s6);
}

//x in 4 out(s0, s1, s2, s3)
STATIC_INLINE void _mm_transpose84_epi8(__m128i& s0, __m128i& s1, __m128i& s2, __m128i& s3, __m128i& s4, __m128i& s5, __m128i& s6, __m128i& s7)
{
	s0 = _mm_unpacklo_epi8(s0, s1);
	s1 = _mm_unpacklo_epi8(s2, s3);
	s2 = _mm_unpacklo_epi8(s4, s5);
	s3 = _mm_unpacklo_epi8(s6, s7);

	s4 = _mm_unpacklo_epi16(s0, s1);
	s5 = _mm_unpackhi_epi16(s0, s1);
	s6 = _mm_unpacklo_epi16(s2, s3);
	s7 = _mm_unpackhi_epi16(s2, s3);

	s0 = _mm_unpacklo_epi32(s4, s6);
	s1 = _mm_unpackhi_epi32(s4, s6);
	s2 = _mm_unpacklo_epi32(s5, s7);
	s3 = _mm_unpackhi_epi32(s5, s7);
}

STATIC_INLINE void _mm_transposel_epi8(__m128i& s0, __m128i& s1, __m128i& s2, __m128i& s3, __m128i& s4, __m128i& s5, __m128i& s6, __m128i& s7)
{
	__m128i t[8];
	for (int i = 0; i < 8; i++)
	{
		((uchar*)&t[i])[0] = ((uchar*)&s0)[i];
		((uchar*)&t[i])[1] = ((uchar*)&s1)[i];
		((uchar*)&t[i])[2] = ((uchar*)&s2)[i];
		((uchar*)&t[i])[3] = ((uchar*)&s3)[i];
		((uchar*)&t[i])[4] = ((uchar*)&s4)[i];
		((uchar*)&t[i])[5] = ((uchar*)&s5)[i];
		((uchar*)&t[i])[6] = ((uchar*)&s6)[i];
		((uchar*)&t[i])[7] = ((uchar*)&s7)[i];
	}
	s0 = t[0];
	s1 = t[1];
	s2 = t[2];
	s3 = t[3];
	s4 = t[4];
	s5 = t[5];
	s6 = t[6];
	s7 = t[7];
}

#define _MM256_TRANSPOSE4_PD(in_row0, in_row1, in_row2, in_row3		\
						, out_row0, out_row1, out_row2, out_row3) {	\
	__m256d tmp0, tmp1, tmp2, tmp3;									\
																	\
	tmp0 = _mm256_unpackhi_pd((in_row0), (in_row1));				\
	tmp1 = _mm256_unpackhi_pd((in_row2), (in_row3));				\
	tmp2 = _mm256_unpacklo_pd((in_row0), (in_row1));				\
	tmp3 = _mm256_unpacklo_pd((in_row2), (in_row3));				\
																	\
	(out_row3) = _mm256_permute2f128_pd(tmp0, tmp1,0x31);			\
	(out_row2) = _mm256_permute2f128_pd(tmp2, tmp3,0x31);			\
	(out_row1) = _mm256_permute2f128_pd(tmp0, tmp1,0x20);			\
	(out_row0) = _mm256_permute2f128_pd(tmp2, tmp3,0x20);			\
}



#define _MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3, out4, out5, out6, out7){	\
__m256 __tmp0 = _mm256_unpacklo_ps((in0), (in1)); \
__m256 __tmp1 = _mm256_unpackhi_ps((in0), (in1)); \
__m256 __tmp2 = _mm256_unpacklo_ps((in2), (in3)); \
__m256 __tmp3 = _mm256_unpackhi_ps((in2), (in3)); \
__m256 __tmp4 = _mm256_unpacklo_ps((in4), (in5)); \
__m256 __tmp5 = _mm256_unpackhi_ps((in4), (in5)); \
__m256 __tmp6 = _mm256_unpacklo_ps((in6), (in7)); \
__m256 __tmp7 = _mm256_unpackhi_ps((in6), (in7)); \
__m256 __tmpp0 = _mm256_shuffle_ps(__tmp0, __tmp2, 0x44); \
__m256 __tmpp1 = _mm256_shuffle_ps(__tmp0, __tmp2, 0xEE); \
__m256 __tmpp2 = _mm256_shuffle_ps(__tmp1, __tmp3, 0x44); \
__m256 __tmpp3 = _mm256_shuffle_ps(__tmp1, __tmp3, 0xEE); \
__m256 __tmpp4 = _mm256_shuffle_ps(__tmp4, __tmp6, 0x44); \
__m256 __tmpp5 = _mm256_shuffle_ps(__tmp4, __tmp6, 0xEE); \
__m256 __tmpp6 = _mm256_shuffle_ps(__tmp5, __tmp7, 0x44); \
__m256 __tmpp7 = _mm256_shuffle_ps(__tmp5, __tmp7, 0xEE); \
(out0) = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x20); \
(out1) = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x20); \
(out2) = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x20); \
(out3) = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x20); \
(out4) = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x31); \
(out5) = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x31); \
(out6) = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x31); \
(out7) = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x31); \
}

#define _MM256_TRANSPOSE8INPLACE_PS(in_row0, in_row1, in_row2, in_row3, in_row4, in_row5, in_row6, in_row7){	\
	__m256 tmp0, tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;		\
															\
	tmp0 = _mm256_unpackhi_ps((in_row0), (in_row2));		\
	tmp1 = _mm256_unpackhi_ps((in_row1), (in_row3));		\
	tmp2 = _mm256_unpacklo_ps((in_row0), (in_row2));		\
	tmp3 = _mm256_unpacklo_ps((in_row1), (in_row3));		\
	tmp4 = _mm256_unpackhi_ps((in_row4), (in_row6));		\
	tmp5 = _mm256_unpackhi_ps((in_row5), (in_row7));		\
	tmp6 = _mm256_unpacklo_ps((in_row4), (in_row6));		\
	tmp7 = _mm256_unpacklo_ps((in_row5), (in_row7));		\
															\
	in_row0 = _mm256_unpackhi_ps(tmp0, tmp1);				\
	in_row1 = _mm256_unpacklo_ps(tmp0, tmp1);				\
	in_row2 = _mm256_unpackhi_ps(tmp2, tmp3);				\
	in_row3 = _mm256_unpacklo_ps(tmp2, tmp3);				\
	in_row4 = _mm256_unpackhi_ps(tmp4, tmp5);				\
	in_row5 = _mm256_unpacklo_ps(tmp4, tmp5);				\
	in_row6 = _mm256_unpackhi_ps(tmp6, tmp7);				\
	in_row7 = _mm256_unpacklo_ps(tmp6, tmp7);				\
															\
	(tmp7) = _mm256_permute2f128_ps(in_row0, in_row4, 0x31);\
	(tmp6) = _mm256_permute2f128_ps(in_row1, in_row5, 0x31);\
	(tmp5) = _mm256_permute2f128_ps(in_row2, in_row6, 0x31);\
	(tmp4) = _mm256_permute2f128_ps(in_row3, in_row7, 0x31);\
	(tmp3) = _mm256_permute2f128_ps(in_row0, in_row4, 0x20);\
	(tmp2) = _mm256_permute2f128_ps(in_row1, in_row5, 0x20);\
	(tmp1) = _mm256_permute2f128_ps(in_row2, in_row6, 0x20);\
	(tmp0) = _mm256_permute2f128_ps(in_row3, in_row7, 0x20);\
    (in_row0) = (tmp0);										\
    (in_row1) = (tmp1);										\
    (in_row2) = (tmp2);										\
    (in_row3) = (tmp3);										\
    (in_row4) = (tmp4);										\
    (in_row5) = (tmp5);										\
    (in_row6) = (tmp6);										\
    (in_row7) = (tmp7);										\
}

STATIC_INLINE void _mm256_transpose_epi32(__m256i& in_row0, __m256i& in_row1, __m256i& in_row2, __m256i& in_row3, __m256i& in_row4, __m256i& in_row5, __m256i& in_row6, __m256i& in_row7)
{
	__m256i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	tmp0 = _mm256_unpackhi_epi32((in_row0), (in_row2));
	tmp1 = _mm256_unpackhi_epi32((in_row1), (in_row3));
	tmp2 = _mm256_unpacklo_epi32((in_row0), (in_row2));
	tmp3 = _mm256_unpacklo_epi32((in_row1), (in_row3));
	tmp4 = _mm256_unpackhi_epi32((in_row4), (in_row6));
	tmp5 = _mm256_unpackhi_epi32((in_row5), (in_row7));
	tmp6 = _mm256_unpacklo_epi32((in_row4), (in_row6));
	tmp7 = _mm256_unpacklo_epi32((in_row5), (in_row7));

	in_row0 = _mm256_unpackhi_epi32(tmp0, tmp1);
	in_row1 = _mm256_unpacklo_epi32(tmp0, tmp1);
	in_row2 = _mm256_unpackhi_epi32(tmp2, tmp3);
	in_row3 = _mm256_unpacklo_epi32(tmp2, tmp3);
	in_row4 = _mm256_unpackhi_epi32(tmp4, tmp5);
	in_row5 = _mm256_unpacklo_epi32(tmp4, tmp5);
	in_row6 = _mm256_unpackhi_epi32(tmp6, tmp7);
	in_row7 = _mm256_unpacklo_epi32(tmp6, tmp7);

	(tmp7) = _mm256_permute2f128_si256(in_row0, in_row4, 0x31);
	(tmp6) = _mm256_permute2f128_si256(in_row1, in_row5, 0x31);
	(tmp5) = _mm256_permute2f128_si256(in_row2, in_row6, 0x31);
	(tmp4) = _mm256_permute2f128_si256(in_row3, in_row7, 0x31);
	(tmp3) = _mm256_permute2f128_si256(in_row0, in_row4, 0x20);
	(tmp2) = _mm256_permute2f128_si256(in_row1, in_row5, 0x20);
	(tmp1) = _mm256_permute2f128_si256(in_row2, in_row6, 0x20);
	(tmp0) = _mm256_permute2f128_si256(in_row3, in_row7, 0x20);
	(in_row0) = (tmp0);
	(in_row1) = (tmp1);
	(in_row2) = (tmp2);
	(in_row3) = (tmp3);
	(in_row4) = (tmp4);
	(in_row5) = (tmp5);
	(in_row6) = (tmp6);
	(in_row7) = (tmp7);
}

STATIC_INLINE void _mm256_transpose8_ps(__m256* in_row, __m256* out_row)
{
	_MM256_TRANSPOSE8_PS(in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], in_row[5], in_row[6], in_row[7],
		out_row[0], out_row[1], out_row[2], out_row[3], out_row[4], out_row[5], out_row[6], out_row[7]);
}

STATIC_INLINE void _mm256_transpose8_ps(__m256* in_row)
{
	//_MM256_TRANSPOSE8_PS(in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], in_row[5], in_row[6], in_row[7]
	//	, in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], in_row[5], in_row[6], in_row[7]);
	_MM256_TRANSPOSE8INPLACE_PS(in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], in_row[5], in_row[6], in_row[7]);
}

#ifdef CV_AVX_512
STATIC_INLINE void _mm512_transpose16_ps(__m512* in_row)
{
	__m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
	__m512i* r0, * r1, * r2, * r3, * r4, * r5, * r6, * r7, * r8, * r9, * ra, * rb, * rc, * rd, * re, * rf;

	r0 = (__m512i*)&in_row[0];
	r1 = (__m512i*)&in_row[1];
	r2 = (__m512i*)&in_row[2];
	r3 = (__m512i*)&in_row[3];
	r4 = (__m512i*)&in_row[4];
	r5 = (__m512i*)&in_row[5];
	r6 = (__m512i*)&in_row[6];
	r7 = (__m512i*)&in_row[7];
	r8 = (__m512i*)&in_row[8];
	r9 = (__m512i*)&in_row[9];
	ra = (__m512i*)&in_row[10];
	rb = (__m512i*)&in_row[11];
	rc = (__m512i*)&in_row[12];
	rd = (__m512i*)&in_row[13];
	re = (__m512i*)&in_row[14];
	rf = (__m512i*)&in_row[15];

	t0 = _mm512_unpacklo_epi32(*r0, *r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
	t1 = _mm512_unpackhi_epi32(*r0, *r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
	t2 = _mm512_unpacklo_epi32(*r2, *r3); //  32  48  33  49 ...
	t3 = _mm512_unpackhi_epi32(*r2, *r3); //  34  50  35  51 ...
	t4 = _mm512_unpacklo_epi32(*r4, *r5); //  64  80  65  81 ...  
	t5 = _mm512_unpackhi_epi32(*r4, *r5); //  66  82  67  83 ...
	t6 = _mm512_unpacklo_epi32(*r6, *r7); //  96 112  97 113 ...
	t7 = _mm512_unpackhi_epi32(*r6, *r7); //  98 114  99 115 ...
	t8 = _mm512_unpacklo_epi32(*r8, *r9); // 128 ...
	t9 = _mm512_unpackhi_epi32(*r8, *r9); // 130 ...
	ta = _mm512_unpacklo_epi32(*ra, *rb); // 160 ...
	tb = _mm512_unpackhi_epi32(*ra, *rb); // 162 ...
	tc = _mm512_unpacklo_epi32(*rc, *rd); // 196 ...
	td = _mm512_unpackhi_epi32(*rc, *rd); // 198 ...
	te = _mm512_unpacklo_epi32(*re, *rf); // 228 ...
	tf = _mm512_unpackhi_epi32(*re, *rf); // 230 ...

	*r0 = _mm512_unpacklo_epi64(t0, t2); //   0  16  32  48 ...
	*r1 = _mm512_unpackhi_epi64(t0, t2); //   1  17  33  49 ...
	*r2 = _mm512_unpacklo_epi64(t1, t3); //   2  18  34  49 ...
	*r3 = _mm512_unpackhi_epi64(t1, t3); //   3  19  35  51 ...
	*r4 = _mm512_unpacklo_epi64(t4, t6); //  64  80  96 112 ...  
	*r5 = _mm512_unpackhi_epi64(t4, t6); //  65  81  97 114 ...
	*r6 = _mm512_unpacklo_epi64(t5, t7); //  66  82  98 113 ...
	*r7 = _mm512_unpackhi_epi64(t5, t7); //  67  83  99 115 ...
	*r8 = _mm512_unpacklo_epi64(t8, ta); // 128 144 160 176 ...  
	*r9 = _mm512_unpackhi_epi64(t8, ta); // 129 145 161 178 ...
	*ra = _mm512_unpacklo_epi64(t9, tb); // 130 146 162 177 ... 
	*rb = _mm512_unpackhi_epi64(t9, tb); // 131 147 163 179 ...
	*rc = _mm512_unpacklo_epi64(tc, te); // 192 208 228 240 ... 
	*rd = _mm512_unpackhi_epi64(tc, te); // 193 209 229 241 ...
	*re = _mm512_unpacklo_epi64(td, tf); // 194 210 230 242 ...
	*rf = _mm512_unpackhi_epi64(td, tf); // 195 211 231 243 ...

	//_mm512_shuffle_f32x4
	t0 = _mm512_shuffle_i32x4(*r0, *r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
	t1 = _mm512_shuffle_i32x4(*r1, *r5, 0x88); //   1  17  33  49 ...
	t2 = _mm512_shuffle_i32x4(*r2, *r6, 0x88); //   2  18  34  50 ...
	t3 = _mm512_shuffle_i32x4(*r3, *r7, 0x88); //   3  19  35  51 ...
	t4 = _mm512_shuffle_i32x4(*r0, *r4, 0xdd); //   4  20  36  52 ...
	t5 = _mm512_shuffle_i32x4(*r1, *r5, 0xdd); //   5  21  37  53 ...
	t6 = _mm512_shuffle_i32x4(*r2, *r6, 0xdd); //   6  22  38  54 ...
	t7 = _mm512_shuffle_i32x4(*r3, *r7, 0xdd); //   7  23  39  55 ...
	t8 = _mm512_shuffle_i32x4(*r8, *rc, 0x88); // 128 144 160 176 ...
	t9 = _mm512_shuffle_i32x4(*r9, *rd, 0x88); // 129 145 161 177 ...
	ta = _mm512_shuffle_i32x4(*ra, *re, 0x88); // 130 146 162 178 ...
	tb = _mm512_shuffle_i32x4(*rb, *rf, 0x88); // 131 147 163 179 ...
	tc = _mm512_shuffle_i32x4(*r8, *rc, 0xdd); // 132 148 164 180 ...
	td = _mm512_shuffle_i32x4(*r9, *rd, 0xdd); // 133 149 165 181 ...
	te = _mm512_shuffle_i32x4(*ra, *re, 0xdd); // 134 150 166 182 ...
	tf = _mm512_shuffle_i32x4(*rb, *rf, 0xdd); // 135 151 167 183 ...

	*r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
	*r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
	*r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
	*r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
	*r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
	*r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
	*r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
	*r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
	*r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
	*r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
	*ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
	*rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
	*rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
	*rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
	*re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
	*rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
}
#endif

void show_patch(__m256* patch);



STATIC_INLINE void _mm256_storepatch_ph(short* base, __m256* patch, const int index)
{
	_mm_store_si128((__m128i*)(base + 0 * index), _mm256_cvtps_ph(patch[0], 0));
	_mm_store_si128((__m128i*)(base + 1 * index), _mm256_cvtps_ph(patch[1], 0));
	_mm_store_si128((__m128i*)(base + 2 * index), _mm256_cvtps_ph(patch[2], 0));
	_mm_store_si128((__m128i*)(base + 3 * index), _mm256_cvtps_ph(patch[3], 0));
	_mm_store_si128((__m128i*)(base + 4 * index), _mm256_cvtps_ph(patch[4], 0));
	_mm_store_si128((__m128i*)(base + 5 * index), _mm256_cvtps_ph(patch[5], 0));
	_mm_store_si128((__m128i*)(base + 6 * index), _mm256_cvtps_ph(patch[6], 0));
	_mm_store_si128((__m128i*)(base + 7 * index), _mm256_cvtps_ph(patch[7], 0));
}

STATIC_INLINE void _mm256_storeupatch_ph(short* base, __m256* patch, const int index)
{
	_mm_storeu_si128((__m128i*)(base + 0 * index), _mm256_cvtps_ph(patch[0], 0));
	_mm_storeu_si128((__m128i*)(base + 1 * index), _mm256_cvtps_ph(patch[1], 0));
	_mm_storeu_si128((__m128i*)(base + 2 * index), _mm256_cvtps_ph(patch[2], 0));
	_mm_storeu_si128((__m128i*)(base + 3 * index), _mm256_cvtps_ph(patch[3], 0));
	_mm_storeu_si128((__m128i*)(base + 4 * index), _mm256_cvtps_ph(patch[4], 0));
	_mm_storeu_si128((__m128i*)(base + 5 * index), _mm256_cvtps_ph(patch[5], 0));
	_mm_storeu_si128((__m128i*)(base + 6 * index), _mm256_cvtps_ph(patch[6], 0));
	_mm_storeu_si128((__m128i*)(base + 7 * index), _mm256_cvtps_ph(patch[7], 0));
}

STATIC_INLINE void _mm256_storepatch_ps(float* base, __m256* patch, const int index)
{
	_mm256_store_ps(base, patch[0]);
	_mm256_store_ps(base + index, patch[1]);
	_mm256_store_ps(base + 2 * index, patch[2]);
	_mm256_store_ps(base + 3 * index, patch[3]);
	_mm256_store_ps(base + 4 * index, patch[4]);
	_mm256_store_ps(base + 5 * index, patch[5]);
	_mm256_store_ps(base + 6 * index, patch[6]);
	_mm256_store_ps(base + 7 * index, patch[7]);
}

STATIC_INLINE void _mm256_storeupatch_ps(float* base, __m256* patch, const int index)
{
	_mm256_storeu_ps(base, patch[0]);
	_mm256_storeu_ps(base + index, patch[1]);
	_mm256_storeu_ps(base + 2 * index, patch[2]);
	_mm256_storeu_ps(base + 3 * index, patch[3]);
	_mm256_storeu_ps(base + 4 * index, patch[4]);
	_mm256_storeu_ps(base + 5 * index, patch[5]);
	_mm256_storeu_ps(base + 6 * index, patch[6]);
	_mm256_storeu_ps(base + 7 * index, patch[7]);
}

#ifdef CV_AVX_512
STATIC_INLINE void _mm512_storeupatch_ps(float* base, __m512* patch, const int index)
{
	_mm512_storeu_ps(base, patch[0]);
	_mm512_storeu_ps(base + index, patch[1]);
	_mm512_storeu_ps(base + 2 * index, patch[2]);
	_mm512_storeu_ps(base + 3 * index, patch[3]);
	_mm512_storeu_ps(base + 4 * index, patch[4]);
	_mm512_storeu_ps(base + 5 * index, patch[5]);
	_mm512_storeu_ps(base + 6 * index, patch[6]);
	_mm512_storeu_ps(base + 7 * index, patch[7]);
	_mm512_storeu_ps(base + 8 * index, patch[8]);
	_mm512_storeu_ps(base + 9 * index, patch[9]);
	_mm512_storeu_ps(base + 10 * index, patch[10]);
	_mm512_storeu_ps(base + 11 * index, patch[11]);
	_mm512_storeu_ps(base + 12 * index, patch[12]);
	_mm512_storeu_ps(base + 13 * index, patch[13]);
	_mm512_storeu_ps(base + 14 * index, patch[14]);
	_mm512_storeu_ps(base + 15 * index, patch[15]);
}
#endif
STATIC_INLINE void _mm256_addstorepatch_ps(float* base, __m256* patch, const int index)
{
	_mm256_store_ps(base, _mm256_add_ps(*(__m256*) base, patch[0]));
	_mm256_store_ps(base + index, _mm256_add_ps(*(__m256*)(base + index), patch[1]));
	_mm256_store_ps(base + 2 * index, _mm256_add_ps(*(__m256*)(base + 2 * index), patch[2]));
	_mm256_store_ps(base + 3 * index, _mm256_add_ps(*(__m256*)(base + 3 * index), patch[3]));
	_mm256_store_ps(base + 4 * index, _mm256_add_ps(*(__m256*)(base + 4 * index), patch[4]));
	_mm256_store_ps(base + 5 * index, _mm256_add_ps(*(__m256*)(base + 5 * index), patch[5]));
	_mm256_store_ps(base + 6 * index, _mm256_add_ps(*(__m256*)(base + 6 * index), patch[6]));
	_mm256_store_ps(base + 7 * index, _mm256_add_ps(*(__m256*)(base + 7 * index), patch[7]));
}

STATIC_INLINE void show_patch(__m256* patch)
{
	std::cout << std::fixed;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			std::cout << "," << std::setw(10) << std::setprecision(4) << *((float*)&patch[i] + j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

#define _MM256_TRANSPOSE4INPLACE_PD(inout_row0, inout_row1, inout_row2, inout_row3){		\
	__m256d tmp0, tmp1, tmp2, tmp3;									\
																	\
	tmp0 = _mm256_unpackhi_pd((inout_row0), (inout_row1));				\
	tmp1 = _mm256_unpackhi_pd((inout_row2), (inout_row3));				\
	tmp2 = _mm256_unpacklo_pd((inout_row0), (inout_row1));				\
	tmp3 = _mm256_unpacklo_pd((inout_row2), (inout_row3));				\
																	\
	(inout_row3) = _mm256_permute2f128_pd(tmp0, tmp1,0x31);			\
	(inout_row2) = _mm256_permute2f128_pd(tmp2, tmp3,0x31);			\
	(inout_row1) = _mm256_permute2f128_pd(tmp0, tmp1,0x20);			\
	(inout_row0) = _mm256_permute2f128_pd(tmp2, tmp3,0x20);			\
}

//for 64F
STATIC_INLINE void _mm256_transpose4_pd(__m256d* in_row, __m256d* out_row)
{
	_MM256_TRANSPOSE4_PD(in_row[0], in_row[1], in_row[2], in_row[3], out_row[0], out_row[1], out_row[2], out_row[3]);
}

STATIC_INLINE void _mm256_transpose4_pd(__m256d* inout_row)
{
	_MM256_TRANSPOSE4INPLACE_PD(inout_row[0], inout_row[1], inout_row[2], inout_row[3]);
}

STATIC_INLINE void _mm256_storepatch_pd(double* base, __m256d* patch, const int index)
{
	_mm256_store_pd(base + 0 * index, patch[0]);
	_mm256_store_pd(base + 1 * index, patch[1]);
	_mm256_store_pd(base + 2 * index, patch[2]);
	_mm256_store_pd(base + 3 * index, patch[3]);
}

STATIC_INLINE void _mm256_storeupatch_pd(double* base, __m256d* patch, const int index)
{
	_mm256_store_pd(base + 0 * index, patch[0]);
	_mm256_store_pd(base + 1 * index, patch[1]);
	_mm256_store_pd(base + 2 * index, patch[2]);
	_mm256_store_pd(base + 3 * index, patch[3]);
}

STATIC_INLINE void _mm_storepatch_pdps(float* base, __m256d* patch, const int index)
{
	*(__m128*) base = _mm256_cvtpd_ps(patch[0]);
	*(__m128*)(base + index) = _mm256_cvtpd_ps(patch[1]);
	*(__m128*)(base + 2 * index) = _mm256_cvtpd_ps(patch[2]);
	*(__m128*)(base + 3 * index) = _mm256_cvtpd_ps(patch[3]);
}

STATIC_INLINE void _mm_storepatch_ps(float* base, __m128* patch, const int index)
{
	_mm_storeu_ps(base + 0 * index, patch[0]);
	_mm_storeu_ps(base + 1 * index, patch[1]);
	_mm_storeu_ps(base + 2 * index, patch[2]);
	_mm_storeu_ps(base + 3 * index, patch[3]);
}

STATIC_INLINE void _mm256_addstorepatch_pd(double* base, __m256d* patch, const int index)
{
	*(__m256d*)base = _mm256_add_pd(*(__m256d*)base, patch[0]);
	*(__m256d*)(base + index) = _mm256_add_pd(*(__m256d*)(base + index), patch[1]);
	*(__m256d*)(base + 2 * index) = _mm256_add_pd(*(__m256d*)(base + 2 * index), patch[2]);
	*(__m256d*)(base + 3 * index) = _mm256_add_pd(*(__m256d*)(base + 3 * index), patch[3]);
}

STATIC_INLINE void show_patch(__m256d* patch)
{
	std::cout << std::fixed;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			std::cout << "," << std::setw(10) << std::setprecision(4) << *((double*)&patch[i] + j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
#pragma endregion

#pragma region convert
#define _mm256_cvtss_si64(a) (_mm_cvtss_si64(_mm256_castpd256_pd128(a)))
#define _mm256_cvtss_si32(a) (_mm_cvtss_si32(_mm256_castpd256_pd128(a)))
#define _mm256_cvttss_si64(a) (_mm_cvttss_si64(_mm256_castpd256_pd128(a)))
#define _mm256_cvttss_si32(a) (_mm_cvttss_si32(_mm256_castpd256_pd128(a)))
#define _mm256_cvtsd_si64(a) (_mm_cvtsd_si64(_mm256_castpd256_pd128(a)))
#define _mm256_cvtsd_si32(a) (_mm_cvtsd_si32(_mm256_castpd256_pd128(a)))
#define _mm256_cvttsd_si64(a) (_mm_cvttsd_si64(_mm256_castpd256_pd128(a)))
#define _mm256_cvttsd_si32(a) (_mm_cvttsd_si32(_mm256_castpd256_pd128(a)))

//cast
//cast ps to ph and then return cast value as float
STATIC_INLINE float cvtps_ph(float v)
{
	__m256 mv = _mm256_set1_ps(v);
	__m128i ms = _mm256_cvtps_ph(mv, 0);
	mv = _mm256_cvtph_ps(ms);
	return _mm256_cvtss_f32(mv);
}

//opencp cast __m256i of hi register ->__m128i
STATIC_INLINE __m128i _mm256_castsi256hi_si128(__m256i src)
{
	return _mm256_extractf128_si256(src, 1);
}

STATIC_INLINE __m128 _mm256_castps256hi_ps128(__m256 src)
{
	return _mm256_extractf128_ps(src, 1);
}
//opencp (same as _mm256_extractf128_si256(src, 1))
//#define _mm256_castsi256hi_si128(src) *((__m128i*)&(src) + 1)

//opencp uchar->int (0123)
STATIC_INLINE __m128i _mm_cvtepu80_epi32(__m128i src)
{
	return _mm_cvtepu8_epi32(src);
}

//opencp uchar->int (4567)
STATIC_INLINE __m128i _mm_cvtepu81_epi32(__m128i src)
{
	return _mm_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(1, 1, 1, 1)));
}

//opencp uchar->int (891011)
STATIC_INLINE __m128i _mm_cvtepu82_epi32(__m128i src)
{
	return _mm_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(2, 2, 2, 2)));
}

//opencp uchar->int (12131415)
STATIC_INLINE __m128i _mm_cvtepu83_epi32(__m128i src)
{
	return _mm_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(3, 3, 3, 3)));
}

//opencp uchar->double (low 4 elements)
STATIC_INLINE __m256d _mm256_cvtepu8_pd(__m128i src)
{
	//_mm256_cvtepi64_pd(_mm256_cvtepu8_epi64(src));AVX512
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(src)));
}

//opencp uchar->double (0123 elements): same _mm256_cvtepu8_pd
STATIC_INLINE __m256d _mm256_cvtepu80_pd(__m128i src)
{
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(src)));
}

//opencp uchar->double (4567 elements) same _mm256_cvtepu8_pd
STATIC_INLINE __m256d _mm256_cvtepu81_pd(__m128i src)
{
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(1, 1, 1, 1)))));
}

//opencp uchar->double (891011 elements)
STATIC_INLINE __m256d _mm256_cvtepu82_pd(__m128i src)
{
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(2, 2, 2, 2)))));
}

//opencp uchar->double (12131415 elements)
STATIC_INLINE __m256d _mm256_cvtepu83_pd(__m128i src)
{
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(3, 3, 3, 3)))));
}

//opencp short->double (low 4 elements)
STATIC_INLINE __m256d _mm256_cvtepi16_pd(__m128i src)
{
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(src)));
}

//opencp short->double (last low 4 elements)
STATIC_INLINE __m256d _mm256_cvtepi16hi_pd(__m128i src)
{
	return _mm256_cvtps_pd(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(0, 1, 3, 2)))));
}

//opencp uchar->float
STATIC_INLINE __m256 _mm256_cvtepu8_ps(__m128i src)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
}

//opencp uchar->floatx2
STATIC_INLINE void _mm256_cvtepu8_psx2(__m128i src, __m256& dest0, __m256& dest1)
{
	dest0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
	dest1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(1, 0, 3, 2))));
}

//opencp uchar->floatx4
STATIC_INLINE void _mm256_cvtepu8_psx4(__m256i src, __m256& dest0, __m256& dest1, __m256& dest2, __m256& dest3)
{
	dest0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm256_castsi256_si128(src)));
	dest1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(_mm256_castsi256_si128(src), _MM_SHUFFLE(1, 0, 3, 2))));
	__m128i v = _mm256_castsi256hi_si128(src);
	dest2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
	dest3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2))));
}

STATIC_INLINE void _mm256_cvtepu8_epi32x4(__m256i src, __m256i& dest0, __m256i& dest1, __m256i& dest2, __m256i& dest3)
{
	dest0 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(src));
	dest1 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(_mm256_castsi256_si128(src), _MM_SHUFFLE(1, 0, 3, 2)));
	__m128i v = _mm256_castsi256hi_si128(src);
	dest2 = _mm256_cvtepu8_epi32(v);
	dest3 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
}

//opencp int ->uchar
STATIC_INLINE __m128i _mm256_cvtepi32_epu8(const __m256i v0)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, _mm256_setzero_si256()), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

STATIC_INLINE __m128i _mm_cvtepi32_epu8(const __m128i v0)
{
	return _mm_packus_epi16(_mm_packs_epi32(v0, _mm_setzero_si128()), _mm_setzero_si128());
}

//opencp float->uchar
STATIC_INLINE __m128i _mm256_cvtps_epu8(const __m256 ms)
{
	return _mm256_cvtepi32_epu8(_mm256_cvtps_epi32(ms));
}

//opencp floatx2->uchar
STATIC_INLINE __m128i _mm256_cvtpsx2_epu8(const __m256 v0, const __m256 v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(v0), _mm256_cvtps_epi32(v1)), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//opencp intx2 ->uchar
STATIC_INLINE __m128i _mm256_cvtepi32x2_epu8(const __m256i v0, const __m256i v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, v1), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//opencp intx4 ->uchar
STATIC_INLINE __m256i _mm256_cvtepi32x4_epu8(const __m256i v0, const __m256i v1, const __m256i v2, const __m256i v3)
{
	__m256i shlmask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
	return _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, v1), _mm256_packs_epi32(v2, v3)), shlmask);
}

//opencp floatx4 ->uchar
STATIC_INLINE __m256i _mm256_cvtpsx4_epu8(const __m256 v0, const __m256 v1, const __m256 v2, const __m256 v3)
{
	static __m256i shlmask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
	return _mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(v0), _mm256_cvtps_epi32(v1)), _mm256_packs_epi32(_mm256_cvtps_epi32(v2), _mm256_cvtps_epi32(v3))), shlmask);
}

STATIC_INLINE __m256i _mm256_cvtepu16x2_epi8(__m256i a, __m256i b)
{
	return _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), _MM_SHUFFLE(3, 1, 2, 0));
}

STATIC_INLINE __m256i _mm256_cvtepi16x2_epi8(__m256i a, __m256i b)
{
	return _mm256_permute4x64_epi64(_mm256_packs_epi16(a, b), _MM_SHUFFLE(3, 1, 2, 0));
}

//_mm256_cvtepi32_epi16 is already defined in zmmintrin.h (AVX512)
STATIC_INLINE __m128i _mm256_cvtepi32_epi16_v2(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

//_mm256_cvtepi16_epi8 is already defined in zmmintrin.h (AVX512), but this is ep`u`
STATIC_INLINE __m128i _mm256_cvtepi16_epu8(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

STATIC_INLINE __m128i _mm256_cvtepi32_epu16(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

STATIC_INLINE __m256i _mm256_cvepi32x2_epi16(__m256i src1, __m256i src2)
{
	return _mm256_permute4x64_epi64(_mm256_packs_epi32(src1, src2), _MM_SHUFFLE(3, 1, 2, 0));
}

STATIC_INLINE __m256 _mm256_cvtpdx2_ps(__m256d src1, __m256d src2)
{
	return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(src1)), _mm256_cvtpd_ps(src2), 1);
}
#pragma endregion

#pragma region load
STATIC_INLINE __m128 _mm_lddqu_ps(const float* src)
{
	return _mm_castsi128_ps(_mm_lddqu_si128((__m128i*)src));
}

STATIC_INLINE __m128d _mm_lddqu_pd(const double* src)
{
	return _mm_castsi128_pd(_mm_lddqu_si128((__m128i*)src));
}

STATIC_INLINE __m128 _mm_stream_load_ps(const float* src)
{
	return _mm_castsi128_ps(_mm_stream_load_si128((__m128i*)src));
}

STATIC_INLINE __m128d _mm_stream_load_pd(const double* src)
{
	return _mm_castsi128_pd(_mm_stream_load_si128((__m128i*)src));
}

STATIC_INLINE __m256 _mm256_lddqu_ps(const float* src)
{
	return _mm256_castsi256_ps(_mm256_lddqu_si256((__m256i*)src));
}

STATIC_INLINE __m256d _mm256_lddqu_pd(const double* src)
{
	return _mm256_castsi256_pd(_mm256_lddqu_si256((__m256i*)src));
}

STATIC_INLINE __m256 _mm256_stream_load_ps(const float* src)
{
	return _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i*)src));
}

STATIC_INLINE __m256d _mm256_stream_load_pd(const double* src)
{
	return _mm256_castsi256_pd(_mm256_stream_load_si256((__m256i*)src));
}

#pragma endregion

#pragma region load and cast 
//opencp: uchar->short
STATIC_INLINE __m256i _mm256_load_epu8cvtepi16(const __m128i* P)
{
	return _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)P));
}

//opencp: short->int
STATIC_INLINE __m256i _mm256_load_epi16cvtepi32(const __m128i* P)
{
	return _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)P));
}

//opencp: uchar->int
STATIC_INLINE __m256i _mm256_load_epu8cvtepi32(const __m128i* P)
{
	return _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P));
}

//opencp: uchar->intx2
STATIC_INLINE void _mm256_load_epu8cvtepi32x2(const __m128i* P, __m256i& d0, __m256i& d1)
{
	__m128i s = _mm_load_si128((__m128i*)P);
	d0 = _mm256_cvtepu8_epi32(s);
	d1 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
}

//opencp: uchar->float loadu and load is the same intrinsics (AVX2: _mm256_cvtepu8_epi32)
STATIC_INLINE __m256 _mm256_load_epu8cvtps(const __m128i* P)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)P)));
	//return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
}

//opencp: uchar->float loadu and load is the same intrinsics  (AVX2: _mm256_cvtepu8_epi32)
STATIC_INLINE __m256 _mm256_loadu_epu8cvtps(const __m128i* P)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)P)));
	//return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
}

//opencp: uchar->floatx2
STATIC_INLINE void _mm256_load_epu8cvtpsx2(const __m128i* P, __m256& d0, __m256& d1)
{
	__m128i t = _mm_loadu_si128((__m128i*)P);
	d0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(t));
	d1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2))));
}

//opencp: uchar->int
STATIC_INLINE __m256i _mm256_load_cvtepu8_epi32(const uchar* src)
{
	return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src)));
}

//opencp: uchar->float
STATIC_INLINE __m256 _mm256_load_cvtepu8_ps(const uchar* src)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src))));
}

//opencp: uchar->double
STATIC_INLINE __m256d _mm256_load_cvtepu8_pd(const uchar* src)
{
	return _mm256_cvtepu8_pd(_mm_loadl_epi64((const __m128i*)src));
}

//opencp: float->double
STATIC_INLINE __m256d _mm256_load_cvtps_pd(const float* src)
{
	return _mm256_cvtps_pd(_mm_load_ps(src));
}

STATIC_INLINE __m256d _mm256_loadu_cvtps_pd(const float* src)
{
	return _mm256_cvtps_pd(_mm_loadu_ps(src));
}
#pragma endregion

#pragma region color_convert
STATIC_INLINE __m256 _mm256_unpackeven_ps(__m256 src1, __m256 src2)
{
	//__m256i t0 = _mm256_castps_si256(_mm256_shuffle_ps(src1, src2, _MM_SHUFFLE(2, 0, 2, 0)));
	//return _mm256_castsi256_ps(_mm256_permute4x64_epi64(t0, _MM_SHUFFLE(3, 1, 2, 0)));

	__m256i unpackeven_mask1 = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
	__m256i unpackeven_mask2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 2, 4, 6);
	__m256 t0 = _mm256_permutevar8x32_ps(src1, unpackeven_mask1);
	__m256 t1 = _mm256_permutevar8x32_ps(src2, unpackeven_mask2);
	return _mm256_blend_ps(t0, t1, 0b11110000);
}

STATIC_INLINE void _mm256_load_deinterleave_ps(const float* src, __m256& d0, __m256& d1)
{
	__m256 v1 = _mm256_load_ps(src);
	__m256 v2 = _mm256_load_ps(src + 8);
	__m256 s1 = _mm256_shuffle_ps(v1, v2, _MM_SHUFFLE(2, 0, 2, 0));
	__m256 s2 = _mm256_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 1, 3, 1));
	d0 = _mm256_permutevar8x32_ps(s1, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
	d1 = _mm256_permutevar8x32_ps(s2, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}

STATIC_INLINE void _mm256_store_interleave_ps(void* dst, const __m256 d0, const __m256 d1)
{
	__m256 s1 = _mm256_unpacklo_ps(d0, d1);
	__m256 s2 = _mm256_unpackhi_ps(d0, d1);
	_mm256_store_ps((float*)dst + 0, _mm256_permute2f128_ps(s1, s2, 0x20));
	_mm256_store_ps((float*)dst + 8, _mm256_permute2f128_ps(s1, s2, 0x31));
}

STATIC_INLINE void _mm256_storeu_interleave_ps(void* dst, const __m256 d0, const __m256 d1)
{
	__m256 s1 = _mm256_unpacklo_ps(d0, d1);
	__m256 s2 = _mm256_unpackhi_ps(d0, d1);
	_mm256_storeu_ps((float*)dst + 0, _mm256_permute2f128_ps(s1, s2, 0x20));
	_mm256_storeu_ps((float*)dst + 8, _mm256_permute2f128_ps(s1, s2, 0x31));
}

STATIC_INLINE void _mm256_load_cvtpd_bgr2planar_pd(const double* ptr, __m256d& b, __m256d& g, __m256d& r)
{
	__m256d bgr0 = _mm256_loadu_pd(ptr);
	__m256d bgr1 = _mm256_loadu_pd(ptr + 4);
	__m256d bgr2 = _mm256_loadu_pd(ptr + 8);

	__m256d s02_low = _mm256_permute2f128_pd(bgr0, bgr2, _MM_SELECT4(0, 2));
	__m256d s02_high = _mm256_permute2f128_pd(bgr0, bgr2, _MM_SELECT4(1, 3));

	r = _mm256_blend_pd(_mm256_blend_pd(s02_low, s02_high, 0b1001), bgr1, 0b0010);
	__m256d g0 = _mm256_blend_pd(_mm256_blend_pd(bgr0, bgr2, 0b1100), bgr1, 0b1001);
	g = _mm256_shuffle_pd(g0, g0, 0b0101);
	b = _mm256_blend_pd(_mm256_blend_pd(s02_high, s02_low, 0b1001), bgr1, 0b0100);
}

STATIC_INLINE void _mm256_load_cvtps_bgr2planar_ps(const float* ptr, __m256& b, __m256& g, __m256& r)
{
	const __m256 bgr0 = _mm256_load_ps(ptr);
	const __m256 bgr2 = _mm256_load_ps(ptr + 16);
	const __m256 s02_low = _mm256_permute2f128_ps(bgr0, bgr2, 0 + 2 * 16);
	const __m256 s02_high = _mm256_permute2f128_ps(bgr0, bgr2, 1 + 3 * 16);
	const __m256 bgr1 = _mm256_load_ps(ptr + 8);
	__m256 b0 = _mm256_blend_ps(_mm256_blend_ps(s02_low, s02_high, 0x24), bgr1, 0x92);
	b = _mm256_shuffle_ps(b0, b0, 0x6c);
	__m256 g0 = _mm256_blend_ps(_mm256_blend_ps(s02_high, s02_low, 0x92), bgr1, 0x24);
	g = _mm256_shuffle_ps(g0, g0, 0xb1);
	__m256 r0 = _mm256_blend_ps(_mm256_blend_ps(bgr1, s02_low, 0x24), s02_high, 0x92);
	r = _mm256_shuffle_ps(r0, r0, 0xc6);
}

STATIC_INLINE void _mm256_loadu_cvtps_bgr2planar_ps(const float* ptr, __m256& b, __m256& g, __m256& r)
{

	const __m256 bgr0 = _mm256_loadu_ps(ptr);
	const __m256 bgr2 = _mm256_loadu_ps(ptr + 16);
	const __m256 s02_low = _mm256_permute2f128_ps(bgr0, bgr2, 0 + 2 * 16);
	const __m256 s02_high = _mm256_permute2f128_ps(bgr0, bgr2, 1 + 3 * 16);
	const __m256 bgr1 = _mm256_loadu_ps(ptr + 8);

	__m256 b0 = _mm256_blend_ps(_mm256_blend_ps(s02_low, s02_high, 0x24), bgr1, 0x92);
	b = _mm256_shuffle_ps(b0, b0, 0x6c);
	__m256 g0 = _mm256_blend_ps(_mm256_blend_ps(s02_high, s02_low, 0x92), bgr1, 0x24);
	g = _mm256_shuffle_ps(g0, g0, 0xb1);
	__m256 r0 = _mm256_blend_ps(_mm256_blend_ps(bgr1, s02_low, 0x24), s02_high, 0x92);
	r = _mm256_shuffle_ps(r0, r0, 0xc6);

}
#define _mm256_shuffle_epi64(/*__m256i*/ a, /*__m256i*/ b, /*const int*/ imm8)\
 _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), imm8))

STATIC_INLINE void _mm256_load_cvtepi16bgr2planar_epi16(const short* src, __m256i& b, __m256i& g, __m256i& r)
{
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15
	//  16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47
	__m256i a0 = _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src + 0)));
	__m256i a1 = _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src + 8)));
	__m256i a2 = _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src + 16)));
	a0 = _mm256_inserti128_si256(a0, ((const __m128i*)src)[3], 1);
	a1 = _mm256_inserti128_si256(a1, ((const __m128i*)src)[4], 1);
	a2 = _mm256_inserti128_si256(a2, ((const __m128i*)src)[5], 1);
	//a0:   0   1   2   3;  4   5   6   7 | 24  25  26  27; 28  29  30  31
	//a1:   8   9  10  11; 12  13  14  15 | 32  33  34  35; 36  37  38  39
	//a2:  16  17  18  19; 20  21  22  23 | 40  41  42  43; 44  45  46  47
	b = _mm256_blend_epi16(a0, a1, 0xf0);
	g = _mm256_shuffle_epi64(a0, a2, 0b0101);
	r = _mm256_blend_epi16(a1, a2, 0xf0);
	//b:   0   1   2   3; 12  13  14  15 | 24  25  26  27; 36  37  38  39
	//g:   4   5   6   7; 16  17  18  19 | 28  29  30  31; 40  41  42  43
	//r:   8   9  10  11; 20  21  22  23 | 32  33  34  35; 44  45  46  47
	a0 = _mm256_blend_epi16(b, g, 0xcc);
	a1 = _mm256_shuffle_epi32(_mm256_blend_epi16(r, b, 0xcc), _MM_SHUFFLE(2, 3, 0, 1));
	a2 = _mm256_blend_epi16(g, r, 0xcc);
	//a0:   0   1   6   7; 12  13  18  19 | 24  25  30  31; 36  37  42  43
	//a1:   2   3   8   9; 14  15  20  21 | 26  27  32  33; 38  39  44  45
	//a2:   4   5  10  11; 16  17  22  23 | 28  29  34  35; 40  41  46  47
	//__m128i mask128 = _mm_setr_epi16(255, 0, 3, 2, 5, 4, 7, 6);
	// #ifdef 

#ifdef __INTEL_LLVM_COMPILER
	__m128i mask128 = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
#else
	static __m128i mask128 = _mm_setr_epi64x(361421592464458498i64, 940142975169071882i64);
#endif
	static __m256i mask = _mm256_setr_m128i(mask128, mask128);
	b = _mm256_blend_epi16(a0, a1, 0xaa);
	g = _mm256_shuffle_epi8(_mm256_blend_epi16(a2, a0, 0xaa), mask);
	r = _mm256_blend_epi16(a1, a2, 0xaa);
	//r:   0   3   6   9; 12  15  18  21 | 24  27  30  33; 36  39  42  45
	//g:   1   4   7  10; 13  16  19  22 | 25  28  31  34; 37  40  43  46
	//b:   2   5   8  11; 14  17  20  23 | 26  29  32  35; 38  41  44  47
}

//opencp: BGR2Planar (uchar). Same function of OpenCV for SSE4.1.
STATIC_INLINE void _mm_load_cvtepu8bgr2planar_si128(const uchar* ptr, __m128i& b, __m128i& g, __m128i& r)
{
	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	__m128i s0 = _mm_loadu_si128((const __m128i*)ptr);
	__m128i s1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
	__m128i s2 = _mm_loadu_si128((const __m128i*)(ptr + 32));
	__m128i a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
	__m128i b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
	__m128i c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
	const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
	const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
	a0 = _mm_shuffle_epi8(a0, sh_b);
	b0 = _mm_shuffle_epi8(b0, sh_g);
	c0 = _mm_shuffle_epi8(c0, sh_r);
	b = a0;
	g = b0;
	r = c0;
}

//for SSE4.1
STATIC_INLINE void _mm_load_cvtepu8bgr2planar_epi64(const uchar* ptr, __m128i& b, __m128i& g, __m128i& r)
{
	//b = _mm_setr_epi8(ptr[0], ptr[3], ptr[6], ptr[9], ptr[12], ptr[15], ptr[18], ptr[21], 0, 0, 0, 0, 0, 0, 0, 0);
	//g = _mm_setr_epi8(ptr[1], ptr[4], ptr[7], ptr[10], ptr[13], ptr[16], ptr[19], ptr[22], 0, 0, 0, 0, 0, 0, 0, 0);
	//r = _mm_setr_epi8(ptr[2], ptr[5], ptr[8], ptr[11], ptr[14], ptr[17], ptr[20], ptr[23], 0, 0, 0, 0, 0, 0, 0, 0);

	const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
	const __m128i mask2 = _mm_setr_epi8(0, 3, 6, 0, 0, 0, 2, 5, 0, 0, 0, 1, 4, 7, 0, 0);
	__m128i s0 = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(ptr)), mask1);      //bbbbbbgggggrrrrr
	__m128i s1 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(ptr + 16)), mask2);//ggg000bb000rrr00


	const __m128i bmask1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	b = _mm_blendv_epi8(s1, s0, bmask1);//bbbbbbbb
	const __m128i smask1 = _mm_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
	const __m128i bmask2 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	s0 = _mm_shuffle_epi8(s0, smask1);  //gggggbbbbbbrrrrr
	s1 = _mm_shuffle_epi8(s1, smask1);  //bb000ggg000rrr00
	g = _mm_blendv_epi8(s1, s0, bmask2);//gggggggg

	const __m128i smask2 = _mm_setr_epi8(11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	const __m128i smask3 = _mm_setr_epi8(0, 0, 0, 0, 0, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0);
	s0 = _mm_shuffle_epi8(s0, smask2);  //rrrrr00000000000
	s1 = _mm_shuffle_epi8(s1, smask3);  //00000rrr00000000
	r = _mm_blendv_epi8(s1, s0, bmask2);//rrrrrrrr
}
//BGR2Planar (uchar). Same function of OpenCV for AVX.
STATIC_INLINE void _mm256_load_cvtepu8bgr2planar_si256(const uchar* ptr, __m256i& b, __m256i& g, __m256i& r)
{
	__m256i bgr0 = _mm256_load_si256((const __m256i*)ptr);
	__m256i bgr1 = _mm256_load_si256((const __m256i*)(ptr + 32));
	__m256i bgr2 = _mm256_load_si256((const __m256i*)(ptr + 64));

	__m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
	__m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

	const __m256i blendmask_bgrdeinterleave0 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	const __m256i blendmask_bgrdeinterleave1 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

	__m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, blendmask_bgrdeinterleave0), bgr1, blendmask_bgrdeinterleave1);
	__m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, blendmask_bgrdeinterleave1), bgr1, blendmask_bgrdeinterleave0);
	__m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, blendmask_bgrdeinterleave0), s02_high, blendmask_bgrdeinterleave1);

	const __m256i shufflemask_bgrdeinterleaveb = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	const __m256i shufflemask_bgrdeinterleaveg = _mm256_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
	const __m256i shufflemask_bgrdeinterleaver = _mm256_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
	b = _mm256_shuffle_epi8(b0, shufflemask_bgrdeinterleaveb);
	g = _mm256_shuffle_epi8(g0, shufflemask_bgrdeinterleaveg);
	r = _mm256_shuffle_epi8(r0, shufflemask_bgrdeinterleaver);
}

STATIC_INLINE void _mm256_loadu_cvtepu8bgr2planar_si256(const uchar* ptr, __m256i& b, __m256i& g, __m256i& r)
{
	__m256i bgr0 = _mm256_loadu_si256((const __m256i*)ptr);
	__m256i bgr1 = _mm256_loadu_si256((const __m256i*)(ptr + 32));
	__m256i bgr2 = _mm256_loadu_si256((const __m256i*)(ptr + 64));

	__m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
	__m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

	const __m256i blendmask_bgrdeinterleave0 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	const __m256i blendmask_bgrdeinterleave1 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

	__m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, blendmask_bgrdeinterleave0), bgr1, blendmask_bgrdeinterleave1);
	__m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, blendmask_bgrdeinterleave1), bgr1, blendmask_bgrdeinterleave0);
	__m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, blendmask_bgrdeinterleave0), s02_high, blendmask_bgrdeinterleave1);

	const __m256i shufflemask_bgrdeinterleaveb = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	const __m256i shufflemask_bgrdeinterleaveg = _mm256_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
	const __m256i shufflemask_bgrdeinterleaver = _mm256_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
	b = _mm256_shuffle_epi8(b0, shufflemask_bgrdeinterleaveb);
	g = _mm256_shuffle_epi8(g0, shufflemask_bgrdeinterleaveg);
	r = _mm256_shuffle_epi8(r0, shufflemask_bgrdeinterleaver);
}

//opencp: BGR2Planar (uchar->float). psx2 is more effective
STATIC_INLINE void _mm256_load_cvtepu8bgr2planar_ps(const uchar* ptr, __m256& b, __m256& g, __m256& r)
{
	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0);
	const __m128i m2 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	__m128i s0 = _mm_load_si128((const __m128i*)ptr);
	__m128i s1 = _mm_load_si128((const __m128i*)(ptr + 16));

	__m128i a0 = _mm_blendv_epi8(s0, s1, m0);
	__m128i b0 = _mm_blendv_epi8(s0, s1, m1);
	__m128i c0 = _mm_blendv_epi8(s0, s1, m2);
	const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0);
	const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0);
	const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0);
	a0 = _mm_shuffle_epi8(a0, sh_b);
	b0 = _mm_shuffle_epi8(b0, sh_g);
	c0 = _mm_shuffle_epi8(c0, sh_r);
	b = _mm256_cvtepu8_ps(a0);
	g = _mm256_cvtepu8_ps(b0);
	r = _mm256_cvtepu8_ps(c0);
}

//opencp: BGR2Planar (uchar->float) SSE shuffle and then cvtepu8_ps. psx4 has almost the same performance.
STATIC_INLINE void _mm256_load_cvtepu8bgr2planar_psx2(const uchar* ptr, __m256& b0, __m256& b1, __m256& g0, __m256& g1, __m256& r0, __m256& r1)
{
	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	__m128i s0 = _mm_loadu_si128((const __m128i*)ptr);
	__m128i s1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
	__m128i s2 = _mm_loadu_si128((const __m128i*)(ptr + 32));
	__m128i t0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
	__m128i t1 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
	__m128i t2 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
	const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
	const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
	t0 = _mm_shuffle_epi8(t0, sh_b);
	t1 = _mm_shuffle_epi8(t1, sh_g);
	t2 = _mm_shuffle_epi8(t2, sh_r);

	_mm256_cvtepu8_psx2(t0, b0, b1);
	_mm256_cvtepu8_psx2(t1, g0, g1);
	_mm256_cvtepu8_psx2(t2, r0, r1);
}

//opencp: BGR2Planar (uchar->float) AVX shuffle and then cvtepu8_ps. psx2 has almost the same performance.
STATIC_INLINE void _mm256_load_cvtepu8bgr2planar_psx4(const uchar* ptr,
	__m256& b0, __m256& b1, __m256& b2, __m256& b3,
	__m256& g0, __m256& g1, __m256& g2, __m256& g3,
	__m256& r0, __m256& r1, __m256& r2, __m256& r3)
{
	__m256i bgr0 = _mm256_load_si256((const __m256i*)ptr);
	__m256i bgr1 = _mm256_load_si256((const __m256i*)(ptr + 32));
	__m256i bgr2 = _mm256_load_si256((const __m256i*)(ptr + 64));

	__m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
	__m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

	const __m256i blendmask_bgrdeinterleave0 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	const __m256i blendmask_bgrdeinterleave1 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

	__m256i t0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, blendmask_bgrdeinterleave0), bgr1, blendmask_bgrdeinterleave1);
	__m256i t1 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, blendmask_bgrdeinterleave1), bgr1, blendmask_bgrdeinterleave0);
	__m256i t2 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, blendmask_bgrdeinterleave0), s02_high, blendmask_bgrdeinterleave1);

	const __m256i shufflemask_bgrdeinterleaveb = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	const __m256i shufflemask_bgrdeinterleaveg = _mm256_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
	const __m256i shufflemask_bgrdeinterleaver = _mm256_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
	t0 = _mm256_shuffle_epi8(t0, shufflemask_bgrdeinterleaveb);
	t1 = _mm256_shuffle_epi8(t1, shufflemask_bgrdeinterleaveg);
	t2 = _mm256_shuffle_epi8(t2, shufflemask_bgrdeinterleaver);
	_mm256_cvtepu8_psx2(_mm256_castsi256_si128(t0), b0, b1);
	_mm256_cvtepu8_psx2(_mm256_castsi256hi_si128(t0), b2, b3);
	_mm256_cvtepu8_psx2(_mm256_castsi256_si128(t1), g0, g1);
	_mm256_cvtepu8_psx2(_mm256_castsi256hi_si128(t1), g2, g3);
	_mm256_cvtepu8_psx2(_mm256_castsi256_si128(t2), r0, r1);
	_mm256_cvtepu8_psx2(_mm256_castsi256hi_si128(t2), r2, r3);
}


STATIC_INLINE void _mm_store_interleave_epi8_epi64(uchar* dst, __m128i b, __m128i g, __m128i r)
{
	const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

	__m128i a0 = _mm_shuffle_epi8(b, sh_a);
	__m128i b0 = _mm_shuffle_epi8(g, sh_b);
	__m128i c0 = _mm_shuffle_epi8(r, sh_c);

	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

	_mm_storeu_si128((__m128i*)(dst), _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0));
	_mm_storel_epi64((__m128i*)(dst + 16), _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0));
}

#define _mm_store_epi8_color _mm_store_interleave_epi8_si128
STATIC_INLINE void _mm_store_epi8_color(uchar* dst, __m128i b, __m128i g, __m128i r)
{
	const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	__m128i a0 = _mm_shuffle_epi8(b, sh_a);
	__m128i b0 = _mm_shuffle_epi8(g, sh_b);
	__m128i c0 = _mm_shuffle_epi8(r, sh_c);

	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

	_mm_store_si128((__m128i*)(dst), _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0));
	_mm_store_si128((__m128i*)(dst + 16), _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0));
	_mm_store_si128((__m128i*)(dst + 32), _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0));
}


#define _mm_storeu_epi8_color _mm_storeu_interleave_epi8_si128
STATIC_INLINE void _mm_storeu_epi8_color(uchar* dst, __m128i b, __m128i g, __m128i r)
{
	const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	__m128i a0 = _mm_shuffle_epi8(b, sh_a);
	__m128i b0 = _mm_shuffle_epi8(g, sh_b);
	__m128i c0 = _mm_shuffle_epi8(r, sh_c);

	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

	_mm_storeu_si128((__m128i*)(dst), _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0));
	_mm_storeu_si128((__m128i*)(dst + 16), _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0));
	_mm_storeu_si128((__m128i*)(dst + 32), _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0));
}

#define _mm_stream_epi8_color _mm_stream_interleave_epi8_si256
STATIC_INLINE void _mm_stream_epi8_color(uchar* dst, __m128i b, __m128i g, __m128i r)
{
	const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	__m128i a0 = _mm_shuffle_epi8(b, sh_a);
	__m128i b0 = _mm_shuffle_epi8(g, sh_b);
	__m128i c0 = _mm_shuffle_epi8(r, sh_c);

	const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
	const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

	_mm_stream_si128((__m128i*)(dst), _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0));
	_mm_stream_si128((__m128i*)(dst + 16), _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0));
	_mm_stream_si128((__m128i*)(dst + 32), _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0));
}

#define _mm256_store_epi8_color _mm256_store_interleave_epi8_si256
STATIC_INLINE void _mm256_store_epi8_color(uchar* dst, __m256i b, __m256i g, __m256i r)
{
	const __m256i sh_b = _mm256_setr_epi8(
		0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
		0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m256i sh_g = _mm256_setr_epi8(
		5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
		5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m256i sh_r = _mm256_setr_epi8(
		10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
		10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

	__m256i b0 = _mm256_shuffle_epi8(b, sh_b);
	__m256i g0 = _mm256_shuffle_epi8(g, sh_g);
	__m256i r0 = _mm256_shuffle_epi8(r, sh_r);

	const __m256i m0 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
		0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	const __m256i m1 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
		0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);

	__m256i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
	__m256i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
	__m256i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

	_mm256_store_si256((__m256i*)dst, _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16));
	_mm256_store_si256((__m256i*)(dst + 32), _mm256_permute2x128_si256(p2, p0, 0 + 3 * 16));
	_mm256_store_si256((__m256i*)(dst + 64), _mm256_permute2x128_si256(p1, p2, 1 + 3 * 16));

#if 0
	const __m256i mask1 = _mm256_set_epi8(
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0,
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0
	);
	const __m256i mask2 = _mm256_set_epi8(
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5,
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5
	);
	const __m256i mask3 = _mm256_set_epi8(
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10,
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10
	);

	const __m256i aa = _mm256_shuffle_epi8(b, mask1);
	const __m256i bb = _mm256_shuffle_epi8(g, mask2);
	const __m256i cc = _mm256_shuffle_epi8(r, mask3);

	const __m256i bmask1 = _mm256_set_epi8(
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255,
		0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0
	);
	const __m256i bmask2 = _mm256_set_epi8(
		255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255,
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255
	);

	__m256i aaa = _mm256_permute2x128_si256(aa, aa, 0x00);
	__m256i bbb = _mm256_permute2x128_si256(bb, bb, 0x00);
	__m256i ccc = _mm256_permute2x128_si256(cc, cc, 0x00);
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst)), _mm256_blendv_epi8(ccc, _mm256_blendv_epi8(aaa, bbb, bmask1), bmask2));
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 32), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(bb, aa, bmask2), bmask1));
	aaa = _mm256_permute2x128_si256(aa, aa, 0x11);
	bbb = _mm256_permute2x128_si256(bb, bb, 0x11);
	ccc = _mm256_permute2x128_si256(cc, cc, 0x11);
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 64), _mm256_blendv_epi8(aaa, _mm256_blendv_epi8(bbb, ccc, bmask1), bmask2));
#endif
}

#define _mm256_storeu_epi8_color _mm256_storeu_interleave_epi8_si256
STATIC_INLINE void _mm256_storeu_epi8_color(uchar* dst, __m256i b, __m256i g, __m256i r)
{
	const __m256i sh_b = _mm256_setr_epi8(
		0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
		0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m256i sh_g = _mm256_setr_epi8(
		5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
		5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m256i sh_r = _mm256_setr_epi8(
		10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
		10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

	__m256i b0 = _mm256_shuffle_epi8(b, sh_b);
	__m256i g0 = _mm256_shuffle_epi8(g, sh_g);
	__m256i r0 = _mm256_shuffle_epi8(r, sh_r);

	const __m256i m0 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
		0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
	const __m256i m1 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
		0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);

	__m256i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
	__m256i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
	__m256i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

	_mm256_storeu_si256((__m256i*)dst, _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16));
	_mm256_storeu_si256((__m256i*)(dst + 32), _mm256_permute2x128_si256(p2, p0, 0 + 3 * 16));
	_mm256_storeu_si256((__m256i*)(dst + 64), _mm256_permute2x128_si256(p1, p2, 1 + 3 * 16));

#if 0
	const __m256i mask1 = _mm256_set_epi8(
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0,
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0
	);
	const __m256i mask2 = _mm256_set_epi8(
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5,
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5
	);
	const __m256i mask3 = _mm256_set_epi8(
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10,
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10
	);

	const __m256i aa = _mm256_shuffle_epi8(b, mask1);
	const __m256i bb = _mm256_shuffle_epi8(g, mask2);
	const __m256i cc = _mm256_shuffle_epi8(r, mask3);

	const __m256i bmask1 = _mm256_set_epi8(
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255,
		0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0
	);
	const __m256i bmask2 = _mm256_set_epi8(
		255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255,
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255
	);

	__m256i aaa = _mm256_permute2x128_si256(aa, aa, 0x00);
	__m256i bbb = _mm256_permute2x128_si256(bb, bb, 0x00);
	__m256i ccc = _mm256_permute2x128_si256(cc, cc, 0x00);
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst)), _mm256_blendv_epi8(ccc, _mm256_blendv_epi8(aaa, bbb, bmask1), bmask2));
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 32), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(bb, aa, bmask2), bmask1));
	aaa = _mm256_permute2x128_si256(aa, aa, 0x11);
	bbb = _mm256_permute2x128_si256(bb, bb, 0x11);
	ccc = _mm256_permute2x128_si256(cc, cc, 0x11);
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 64), _mm256_blendv_epi8(aaa, _mm256_blendv_epi8(bbb, ccc, bmask1), bmask2));
#endif
}


STATIC_INLINE void _mm256_storeu_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r)
{
	__m256i mb = _mm256_cvtps_epi32(b);
	__m256i mg = _mm256_cvtps_epi32(g);
	__m256i mr = _mm256_cvtps_epi32(r);
	mb = _mm256_packus_epi16(mb, mg);
	mb = _mm256_packus_epi16(mb, mr);
	const __m256i mask = _mm256_setr_epi8(0, 4, 8, 1, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 0, 4, 8, 1);
	mb = _mm256_shuffle_epi8(mb, mask);
	__m256i mp = _mm256_permute2f128_si256(mb, mb, 0x11);
	mb = _mm256_blend_epi32(mb, mp, 8);

	_mm256_storeu_si256((__m256i*)dst, mb);
}

#define  _mm256_stream_ps2epu8_color  _mm256_store_ps2epu8_color
STATIC_INLINE void _mm256_stream_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r)
{
	__m256i mb = _mm256_cvtps_epi32(b);
	__m256i mg = _mm256_cvtps_epi32(g);
	__m256i mr = _mm256_cvtps_epi32(r);
	mb = _mm256_packus_epi16(mb, mg);
	mb = _mm256_packus_epi16(mb, mr);
	const __m256i mask = _mm256_setr_epi8(0, 4, 8, 1, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 0, 4, 8, 1);
	mb = _mm256_shuffle_epi8(mb, mask);
	__m256i mp = _mm256_permute2f128_si256(mb, mb, 0x11);
	mb = _mm256_blend_epi32(mb, mp, 8);

	_mm256_store_si256((__m256i*)dst, mb);//interleved data cannot be stream
}

STATIC_INLINE void _mm256_storescalar_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r, const int numpixel = 24)
{
	__m256i mb = _mm256_cvtps_epi32(b);
	__m256i mg = _mm256_cvtps_epi32(g);
	__m256i mr = _mm256_cvtps_epi32(r);
	mb = _mm256_packus_epi16(mb, mg);
	mb = _mm256_packus_epi16(mb, mr);
	const __m256i mask = _mm256_setr_epi8(0, 4, 8, 1, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 0, 4, 8, 1);
	mb = _mm256_shuffle_epi8(mb, mask);
	__m256i mp = _mm256_permute2f128_si256(mb, mb, 0x11);
	mb = _mm256_blend_epi32(mb, mp, 8);

	uchar CV_DECL_ALIGNED(32) buffscalarstore[32];
	_mm256_store_si256((__m256i*)buffscalarstore, mb);
	uchar* dest = (uchar*)dst;
	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}

#define _mm256_store_ps_color _mm256_store_interleave_ps
STATIC_INLINE void _mm256_store_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
	__m256 b0 = _mm256_shuffle_ps(b, b, 0x6c);
	__m256 g0 = _mm256_shuffle_ps(g, g, 0xb1);
	__m256 r0 = _mm256_shuffle_ps(r, r, 0xc6);

	__m256 p0 = _mm256_blend_ps(_mm256_blend_ps(b0, g0, 0x92), r0, 0x24);
	__m256 p1 = _mm256_blend_ps(_mm256_blend_ps(g0, r0, 0x92), b0, 0x24);
	__m256 p2 = _mm256_blend_ps(_mm256_blend_ps(r0, b0, 0x92), g0, 0x24);

	__m256 bgr0 = _mm256_permute2f128_ps(p0, p1, 0 + 2 * 16);
	//__m256i bgr1 = p2;
	__m256 bgr2 = _mm256_permute2f128_ps(p0, p1, 1 + 3 * 16);

	_mm256_store_ps((float*)dst, bgr0);
	_mm256_store_ps((float*)dst + 8, p2);
	_mm256_store_ps((float*)dst + 16, bgr2);
}

#define _mm256_storeu_ps_color _mm256_storeu_interleave_ps
STATIC_INLINE void _mm256_storeu_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
	__m256 b0 = _mm256_shuffle_ps(b, b, 0x6c);
	__m256 g0 = _mm256_shuffle_ps(g, g, 0xb1);
	__m256 r0 = _mm256_shuffle_ps(r, r, 0xc6);

	__m256 p0 = _mm256_blend_ps(_mm256_blend_ps(b0, g0, 0x92), r0, 0x24);
	__m256 p1 = _mm256_blend_ps(_mm256_blend_ps(g0, r0, 0x92), b0, 0x24);
	__m256 p2 = _mm256_blend_ps(_mm256_blend_ps(r0, b0, 0x92), g0, 0x24);

	__m256 bgr0 = _mm256_permute2f128_ps(p0, p1, 0 + 2 * 16);
	//__m256i bgr1 = p2;
	__m256 bgr2 = _mm256_permute2f128_ps(p0, p1, 1 + 3 * 16);

	_mm256_storeu_ps((float*)dst, bgr0);
	_mm256_storeu_ps((float*)dst + 8, p2);
	_mm256_storeu_ps((float*)dst + 16, bgr2);
}

STATIC_INLINE void _mm256_storescalar_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r, const int numpixel = 8)
{
	__m256 b0 = _mm256_shuffle_ps(b, b, 0x6c);
	__m256 g0 = _mm256_shuffle_ps(g, g, 0xb1);
	__m256 r0 = _mm256_shuffle_ps(r, r, 0xc6);

	__m256 p0 = _mm256_blend_ps(_mm256_blend_ps(b0, g0, 0x92), r0, 0x24);
	__m256 p1 = _mm256_blend_ps(_mm256_blend_ps(g0, r0, 0x92), b0, 0x24);
	__m256 p2 = _mm256_blend_ps(_mm256_blend_ps(r0, b0, 0x92), g0, 0x24);

	__m256 bgr0 = _mm256_permute2f128_ps(p0, p1, 0 + 2 * 16);
	//__m256i bgr1 = p2;
	__m256 bgr2 = _mm256_permute2f128_ps(p0, p1, 1 + 3 * 16);

	float CV_DECL_ALIGNED(32) buffscalarstore[24];
	_mm256_store_ps(buffscalarstore + 0, bgr0);
	_mm256_store_ps(buffscalarstore + 8, p2);
	_mm256_store_ps(buffscalarstore + 16, bgr2);
	float* dest = (float*)dst;
	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}

#define _mm256_stream_ps_color _mm256_stream_interleve_ps
STATIC_INLINE void _mm256_stream_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
	__m256 b0 = _mm256_shuffle_ps(b, b, 0x6c);
	__m256 g0 = _mm256_shuffle_ps(g, g, 0xb1);
	__m256 r0 = _mm256_shuffle_ps(r, r, 0xc6);

	__m256 p0 = _mm256_blend_ps(_mm256_blend_ps(b0, g0, 0x92), r0, 0x24);
	__m256 p1 = _mm256_blend_ps(_mm256_blend_ps(g0, r0, 0x92), b0, 0x24);
	__m256 p2 = _mm256_blend_ps(_mm256_blend_ps(r0, b0, 0x92), g0, 0x24);

	__m256 bgr0 = _mm256_permute2f128_ps(p0, p1, 0 + 2 * 16);
	//__m256i bgr1 = p2;
	__m256 bgr2 = _mm256_permute2f128_ps(p0, p1, 1 + 3 * 16);

	_mm256_stream_ps((float*)dst, bgr0);
	_mm256_stream_ps((float*)dst + 8, p2);
	_mm256_stream_ps((float*)dst + 16, bgr2);
}

STATIC_INLINE void _mm256_stream_ps_color_2(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
	const int smask1 = _MM_SHUFFLE(1, 2, 3, 0);
	const int smask2 = _MM_SHUFFLE(2, 3, 0, 1);
	const int smask3 = _MM_SHUFFLE(3, 0, 1, 2);
	const int bmask1 = 0x44;
	const int bmask2 = 0x22;
	const int pmask1 = 0x20;
	const int pmask2 = 0x30;
	const int pmask3 = 0x31;
	const __m256 aa = _mm256_shuffle_ps(b, b, smask1);
	const __m256 bb = _mm256_shuffle_ps(g, g, smask2);
	const __m256 cc = _mm256_shuffle_ps(r, r, smask3);
	__m256 bval = _mm256_blend_ps(_mm256_blend_ps(aa, cc, bmask1), bb, bmask2);
	__m256 gval = _mm256_blend_ps(_mm256_blend_ps(cc, bb, bmask1), aa, bmask2);
	__m256 rval = _mm256_blend_ps(_mm256_blend_ps(bb, aa, bmask1), cc, bmask2);
	_mm256_stream_ps((float*)dst + 0, _mm256_permute2f128_ps(bval, rval, pmask1));
	_mm256_stream_ps((float*)dst + 8, _mm256_permute2f128_ps(gval, bval, pmask2));
	_mm256_stream_ps((float*)dst + 16, _mm256_permute2f128_ps(rval, gval, pmask3));
}

void STATIC_INLINE _mm256_store_ps_color_v2(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
	static const int smask1 = _MM_SHUFFLE(1, 2, 3, 0);
	static const int smask2 = _MM_SHUFFLE(2, 3, 0, 1);
	static const int smask3 = _MM_SHUFFLE(3, 0, 1, 2);
	static const int bmask1 = 0x44;
	static const int bmask2 = 0x22;
	static const int pmask1 = 0x20;
	static const int pmask2 = 0x30;
	static const int pmask3 = 0x31;
	const __m256 aa = _mm256_shuffle_ps(b, b, smask1);
	const __m256 bb = _mm256_shuffle_ps(g, g, smask2);
	const __m256 cc = _mm256_shuffle_ps(r, r, smask3);
	__m256 bval = _mm256_blend_ps(_mm256_blend_ps(aa, cc, bmask1), bb, bmask2);
	__m256 gval = _mm256_blend_ps(_mm256_blend_ps(cc, bb, bmask1), aa, bmask2);
	__m256 rval = _mm256_blend_ps(_mm256_blend_ps(bb, aa, bmask1), cc, bmask2);
	_mm256_store_ps((float*)dst + 0, _mm256_permute2f128_ps(bval, rval, pmask1));
	_mm256_store_ps((float*)dst + 8, _mm256_permute2f128_ps(gval, bval, pmask2));
	_mm256_store_ps((float*)dst + 16, _mm256_permute2f128_ps(rval, gval, pmask3));
}

#define _mm256_storeu_pd_color _mm256_storeu_interleave_pd
STATIC_INLINE void _mm256_storeu_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
{
	const __m256d b0 = _mm256_permute2f128_pd(b, b, 0b00000000);
	const __m256d b1 = _mm256_permute2f128_pd(b, b, 0b00010001);

	const __m256d g0 = _mm256_shuffle_pd(g, g, 0b0101);

	const __m256d r0 = _mm256_permute2f128_pd(r, r, 0b00000000);
	const __m256d r1 = _mm256_permute2f128_pd(r, r, 0b00010001);

	_mm256_storeu_pd(static_cast<double*>(dst) + 0, _mm256_blend_pd(_mm256_blend_pd(b0, g0, 0b0010), r0, 0b0100));
	_mm256_storeu_pd(static_cast<double*>(dst) + 4, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b1001), r0, 0b0010));
	_mm256_storeu_pd(static_cast<double*>(dst) + 8, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b0100), r1, 0b1001));
}

#define _mm256_store_pd_color _mm256_store_interleave_pd
STATIC_INLINE void _mm256_store_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
{
	const __m256d b0 = _mm256_permute2f128_pd(b, b, 0b00000000);
	const __m256d b1 = _mm256_permute2f128_pd(b, b, 0b00010001);

	const __m256d g0 = _mm256_shuffle_pd(g, g, 0b0101);

	const __m256d r0 = _mm256_permute2f128_pd(r, r, 0b00000000);
	const __m256d r1 = _mm256_permute2f128_pd(r, r, 0b00010001);

	_mm256_store_pd(static_cast<double*>(dst) + 0, _mm256_blend_pd(_mm256_blend_pd(b0, g0, 0b0010), r0, 0b0100));
	_mm256_store_pd(static_cast<double*>(dst) + 4, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b1001), r0, 0b0010));
	_mm256_store_pd(static_cast<double*>(dst) + 8, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b0100), r1, 0b1001));
}

#define _mm256_stream_pd_color _mm256_stream_interleave_pd
STATIC_INLINE void _mm256_stream_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
{
	const __m256d b0 = _mm256_permute2f128_pd(b, b, 0b00000000);
	const __m256d b1 = _mm256_permute2f128_pd(b, b, 0b00010001);

	const __m256d g0 = _mm256_shuffle_pd(g, g, 0b0101);

	const __m256d r0 = _mm256_permute2f128_pd(r, r, 0b00000000);
	const __m256d r1 = _mm256_permute2f128_pd(r, r, 0b00010001);

	_mm256_stream_pd(static_cast<double*>(dst) + 0, _mm256_blend_pd(_mm256_blend_pd(b0, g0, 0b0010), r0, 0b0100));
	_mm256_stream_pd(static_cast<double*>(dst) + 4, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b1001), r0, 0b0010));
	_mm256_stream_pd(static_cast<double*>(dst) + 8, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b0100), r1, 0b1001));
}

#pragma region gray2bgr
//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm_cvtepi8_gray2bgr(const __m128i src, __m128i& d0, __m128i& d1, __m128i& d2)
{
	static const __m128i g2rgbmask0 = _mm_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5);
	static const __m128i g2rgbmask1 = _mm_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10);
	static const __m128i g2rgbmask2 = _mm_setr_epi8(10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15);
	d0 = _mm_shuffle_epi8(src, g2rgbmask0);
	d1 = _mm_shuffle_epi8(src, g2rgbmask1);
	d2 = _mm_shuffle_epi8(src, g2rgbmask2);
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtepi8_gray2bgr(const __m256i src, __m256i& d0, __m256i& d1, __m256i& d2)
{
	static const __m256i g2rgbmask0 = _mm256_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21);
	static const __m256i g2rgbmask1 = _mm256_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26);
	static const __m256i g2rgbmask2 = _mm256_setr_epi8(10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31);
	__m256i md0 = _mm256_shuffle_epi8(src, g2rgbmask0);
	__m256i md1 = _mm256_shuffle_epi8(src, g2rgbmask1);
	__m256i md2 = _mm256_shuffle_epi8(src, g2rgbmask2);

	d0 = _mm256_permute2x128_si256(md0, md1, _MM_SELECT4(0, 2));
	d1 = _mm256_permute2x128_si256(md2, md0, _MM_SELECT4(0, 3));
	d2 = _mm256_permute2x128_si256(md1, md2, _MM_SELECT4(1, 3));
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtepi16_gray2bgr(const __m256i src, __m256i& d0, __m256i& d1, __m256i& d2)
{
	static const __m256i g2rgbmask0 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5,/**/  16, 17, 16, 17, 16, 17, 18, 19, 18, 19, 18, 19, 20, 21, 20, 21);
	static const __m256i g2rgbmask1 = _mm256_setr_epi8(4, 5, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 10, 11,/**/ 20, 21, 22, 23, 22, 23, 22, 23, 24, 25, 24, 25, 24, 25, 26, 27);
	static const __m256i g2rgbmask2 = _mm256_setr_epi8(10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15,/**/ 26, 27, 26, 27, 28, 29, 28, 29, 28, 29, 30, 31, 30, 31, 30, 31);
	__m256i md0 = _mm256_shuffle_epi8(src, g2rgbmask0);
	__m256i md1 = _mm256_shuffle_epi8(src, g2rgbmask1);
	__m256i md2 = _mm256_shuffle_epi8(src, g2rgbmask2);

	d0 = _mm256_permute2x128_si256(md0, md1, _MM_SELECT4(0, 2));
	d1 = _mm256_permute2x128_si256(md2, md0, _MM_SELECT4(0, 3));
	d2 = _mm256_permute2x128_si256(md1, md2, _MM_SELECT4(1, 3));
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtepi32_gray2bgr(const __m256i src, __m256i& d0, __m256i& d1, __m256i& d2)
{
	static const int smask1 = _MM_SHUFFLE(1, 0, 0, 0);
	static const int smask2 = _MM_SHUFFLE(2, 2, 1, 1);
	static const int smask3 = _MM_SHUFFLE(3, 3, 3, 2);
	const __m256i md0 = _mm256_shuffle_epi32(src, smask1);
	const __m256i md1 = _mm256_shuffle_epi32(src, smask2);
	const __m256i md2 = _mm256_shuffle_epi32(src, smask3);
	d0 = _mm256_permute2x128_si256(md0, md1, _MM_SELECT4(0, 2));
	d1 = _mm256_permute2x128_si256(md2, md0, _MM_SELECT4(0, 3));
	d2 = _mm256_permute2x128_si256(md1, md2, _MM_SELECT4(1, 3));
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtps_gray2bgr(const __m256 srcf, __m256& d0, __m256& d1, __m256& d2)
{
	__m256i src = _mm256_castps_si256(srcf);
	static const int smask1 = _MM_SHUFFLE(1, 0, 0, 0);
	static const int smask2 = _MM_SHUFFLE(2, 2, 1, 1);
	static const int smask3 = _MM_SHUFFLE(3, 3, 3, 2);
	const __m256i md0 = _mm256_shuffle_epi32(src, smask1);
	const __m256i md1 = _mm256_shuffle_epi32(src, smask2);
	const __m256i md2 = _mm256_shuffle_epi32(src, smask3);
	d0 = _mm256_castsi256_ps(_mm256_permute2x128_si256(md0, md1, _MM_SELECT4(0, 2)));
	d1 = _mm256_castsi256_ps(_mm256_permute2x128_si256(md2, md0, _MM_SELECT4(0, 3)));
	d2 = _mm256_castsi256_ps(_mm256_permute2x128_si256(md1, md2, _MM_SELECT4(1, 3)));
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtps_gray2bgr_v2(const __m256 src, __m256& d0, __m256& d1, __m256& d2)
{
	static const int smask1 = _MM_SHUFFLE(1, 0, 0, 0);
	static const int smask2 = _MM_SHUFFLE(2, 2, 1, 1);
	static const int smask3 = _MM_SHUFFLE(3, 3, 3, 2);
	const __m256 md0 = _mm256_shuffle_ps(src, src, smask1);
	const __m256 md1 = _mm256_shuffle_ps(src, src, smask2);
	const __m256 md2 = _mm256_shuffle_ps(src, src, smask3);
	d0 = _mm256_permute2f128_ps(md0, md1, _MM_SELECT4(0, 2));
	d1 = _mm256_permute2f128_ps(md2, md0, _MM_SELECT4(0, 3));
	d2 = _mm256_permute2f128_ps(md1, md2, _MM_SELECT4(1, 3));
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtps_gray2bgr_v3(const __m256 src, __m256& d0, __m256& d1, __m256& d2)
{
	static const __m256i pmask0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
	static const __m256i pmask1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
	static const __m256i pmask2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
	d0 = _mm256_permutevar8x32_ps(src, pmask0);
	d1 = _mm256_permutevar8x32_ps(src, pmask1);
	d2 = _mm256_permutevar8x32_ps(src, pmask2);
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtpd_gray2bgr(const __m256d src, __m256d& d0, __m256d& d1, __m256d& d2)
{
	static const int smask1 = _MM_SHUFFLE(1, 0, 0, 0);
	//static const int smask2 = _MM_SHUFFLE(2, 2, 1, 1);
	static const int smask3 = _MM_SHUFFLE(3, 3, 3, 2);

	d0 = _mm256_permute4x64_pd(src, smask1);
	//d1 = _mm256_permute4x64_pd(src, smask2);
	d1 = _mm256_shuffle_pd(src, src, _MM_SHUFFLE2(1, 1));
	d2 = _mm256_permute4x64_pd(src, smask3);
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtpd_gray2bgr_v2(const __m256d src, __m256d& d0, __m256d& d1, __m256d& d2)
{
	const __m256d md0 = _mm256_shuffle_pd(src, src, 0b0000);
	const __m256d md1 = _mm256_shuffle_pd(src, src, 0b1111);
	const __m256d md2 = _mm256_permute2f128_pd(src, src, _MM_SELECT4(1, 0));
	d0 = _mm256_blend_pd(md0, md2, 0b1100);
	d1 = _mm256_blend_pd(md1, md0, 0b1100);
	d2 = _mm256_blend_pd(md2, md1, 0b1100);
}

//gray2bgr: broadcast 3 channels
STATIC_INLINE void _mm256_cvtpd_gray2bgr_v3(const __m256d src, __m256d& d0, __m256d& d1, __m256d& d2)
{
	const __m256d md0 = _mm256_shuffle_pd(src, src, 0b0000);
	const __m256d md1 = _mm256_shuffle_pd(src, src, 0b1111);
	d0 = _mm256_permute2f128_pd(md0, src, _MM_SELECT4(0, 2));
	d1 = _mm256_blend_pd(md1, md0, 0b1100);
	d2 = _mm256_permute2f128_pd(src, md1, _MM_SELECT4(1, 3));
}
#pragma endregion

//plain2bgr: plain b,g,r image to interleave rgb. SoA->AoS
STATIC_INLINE void _mm256_cvtps_planar2bgr(const __m256 b, const __m256 g, const __m256 r, __m256& d0, __m256& d1, __m256& d2)
{
	const __m256 aa = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(1, 2, 3, 0));
	const __m256 bb = _mm256_shuffle_ps(g, g, _MM_SHUFFLE(2, 3, 0, 1));
	const __m256 cc = _mm256_shuffle_ps(r, r, _MM_SHUFFLE(3, 0, 1, 2));
	__m256 bval = _mm256_blend_ps(_mm256_blend_ps(aa, cc, 0x44), bb, 0x22);
	__m256 gval = _mm256_blend_ps(_mm256_blend_ps(cc, bb, 0x44), aa, 0x22);
	__m256 rval = _mm256_blend_ps(_mm256_blend_ps(bb, aa, 0x44), cc, 0x22);
	d0 = _mm256_permute2f128_ps(bval, rval, 0x20);
	d1 = _mm256_permute2f128_ps(gval, bval, 0x30);
	d2 = _mm256_permute2f128_ps(rval, gval, 0x31);
}

#pragma endregion

#pragma region arithmetic

STATIC_INLINE __m128 _mm_abs_ps(__m128 src)
{
	//return _mm256_and_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
	return _mm_andnot_ps(_mm_set1_ps(-0.f), src);
}

STATIC_INLINE __m256 _mm256_abs_ps(__m256 src)
{
	//return _mm256_and_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
	return _mm256_andnot_ps(_mm256_set1_ps(-0.f), src);
}

STATIC_INLINE __m256 _mm256_signinv_ps(__m256 src)
{
	return _mm256_xor_ps(src, _mm256_set1_ps(-0.f));
}

STATIC_INLINE __m256 _mm256_signbit_ps(__m256 src)
{
	return _mm256_and_ps(_mm256_set1_ps(-0.f), src);
}

STATIC_INLINE __m256 _mm256_signbitinv_ps(__m256 src)
{
	return _mm256_andnot_ps(src, _mm256_set1_ps(-0.f));
}

STATIC_INLINE __m256 _mm256_sign_ps(__m256 src)
{
	__m256 mask = _mm256_cmp_ps(src, _mm256_setzero_ps(), _CMP_GE_OS);
	return _mm256_blendv_ps(_mm256_set1_ps(-1.f), _mm256_set1_ps(1.f), mask);
}

STATIC_INLINE float _mm_reduceadd_ps(__m128 src)
{
	src = _mm_add_ps(src, _mm_shuffle_ps(src, src, _MM_SHUFFLE(2, 3, 0, 1)));
	src = _mm_add_ps(src, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 2, 0, 2)));
	return _mm_cvtss_f32(src);
}

STATIC_INLINE double _mm256_reduceadd_pspd(__m256 src)
{
	float CV_DECL_ALIGNED(32) buff[8];
	_mm256_store_ps(buff, src);
	double ret = buff[0];
	for (int i = 1; i < 8; i++)
		ret += buff[i];
	/*double ret = src.m256_f32[0];
	for (int i = 1; i < 8; i++)
		ret += src.m256_f32[i];
*/
	return ret;
}

//hadd output is same d0 = d1 = a0+a1, d2 = d3 = a2+a3, d4 = d5 = a4+a5, d6 = d7 = a6+a7
STATIC_INLINE __m256 _mm256_hadd_shuffle_ps(__m256 a)
{
	__m256 ret = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
	return _mm256_add_ps(a, ret);
}

//hadd output is same d0 = d2 = a0+a2, d1 = d3 = a1+a3, d4 = d6 = a4+a6, d5 = d7 = a5+a7
STATIC_INLINE __m256 _mm256_hadd_shuffle_evenodd_ps(__m256 a)
{
	__m256 ret = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(1, 0, 3, 2));
	return _mm256_add_ps(a, ret);
}

STATIC_INLINE float _mm256_reduceadd_ps(__m256 src)
{
	src = _mm256_hadd_shuffle_ps(src);
	src = _mm256_hadd_shuffle_evenodd_ps(src);
	return _mm256_cvtss_f32(_mm256_add_ps(src, _mm256_permute2f128_ps(src, src, 1)));
	//return (src.m256_f32[0] + src.m256_f32[4]);

	//__m256 rsum = _mm256_permute2f128_ps(src, src, 0 << 4 | 1);
	//src = _mm256_unpacklo_ps(src, rsum);
	//return _mm256_hadd_ps(src, src).m256_f32[0];
}

STATIC_INLINE int _mm256_reduceadd_epi32(__m256i src)
{
	int* data = (int*)_mm_malloc(sizeof(int) * 8, AVX_ALIGN);
	_mm256_store_si256((__m256i*)data, src);
	int ret = data[0];
	for (int i = 1; i < 8; i++)
	{
		ret += data[i];
	}
	_mm_free(data);
	return ret;
}

STATIC_INLINE float _mm256_reducemax_ps(__m256 src)
{
	float* data = (float*)_mm_malloc(sizeof(float) * 8, AVX_ALIGN);
	_mm256_store_ps(data, src);
	float ret = data[0];
	for (int i = 1; i < 8; i++)
	{
		ret = std::max(ret, data[i]);
	}
	_mm_free(data);
	return ret;
}

STATIC_INLINE void _mm256_reduceadd_highlow_ps(__m256 src, float& low, float& high)
{
	src = _mm256_hadd_ps(src, src);
	src = _mm256_hadd_ps(src, src);
	low = _mm256_cvtss_f32(src);
	high = _mm_cvtss_f32(_mm256_extractf128_ps(src, 1));
	//low = src.m256_f32[0];
	//high = src.m256_f32[4];
}
STATIC_INLINE double _mm256_reduceadd_pd(__m256d src)
{
	src = _mm256_hadd_pd(src, src);
	return _mm256_cvtsd_f64(src) + _mm_cvtsd_f64(_mm256_extractf128_pd(src, 1));
	//return (src.m256d_f64[0] + src.m256d_f64[2]);
}

#pragma region kahan
STATIC_INLINE __m256d _mm256_cvtkahanlo_pd(const __m256 sum, const __m256 carry)
{
	return _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(sum)), _mm256_cvtps_pd(_mm256_castps256_ps128(carry)));
}

STATIC_INLINE __m256d _mm256_cvtkahanhi_pd(const __m256 sum, const __m256 carry)
{
	return _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256hi_ps128(sum)), _mm256_cvtps_pd(_mm256_castps256hi_ps128(carry)));
}

STATIC_INLINE double _mm256_reduceadd_kahan_64f(const __m256 src, const __m256 carry, const double v = 0.0)
{
	__m256d l = _mm256_cvtkahanlo_pd(src, carry);
	__m256d h = _mm256_cvtkahanhi_pd(src, carry);

	h = _mm256_hadd_pd(h, l);
	h = _mm256_add_pd(h, _mm256_shuffle_pd(h, h, 0b0101));

	return v + ((double*)&h)[0] + ((double*)&h)[2];
}

STATIC_INLINE float _mm256_reduceadd_kahan_32f(const __m256 src, const __m256 carry, const float v = 0.f)
{
	__m256d l = _mm256_cvtkahanlo_pd(src, carry);
	__m256d h = _mm256_cvtkahanhi_pd(src, carry);

	h = _mm256_hadd_pd(h, l);
	h = _mm256_add_pd(h, _mm256_shuffle_pd(h, h, 0b0101));
	return (float)(double(v) + ((double*)&h)[0] + ((double*)&h)[2]);
}

STATIC_INLINE float _mm256_reduceadd_kahanfast_32f(const __m256 src, const __m256 carry, const float v = 0.f)
{
	float sum = v;
	float c = 0.f;
	for (int i = 0; i < 8; i++)
		((float*)&carry)[i];


	for (int i = 0; i < 8; i++)
	{
		float y = ((float*)&src)[i] - c;
		float t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	return sum;
}

STATIC_INLINE double _mm256_reduceadd_kahan_64f(const __m256d src, const __m256d carry, const double v = 0.0)
{
	double sum = v;
	double c = ((double*)&carry)[0] + ((double*)&carry)[1] + ((double*)&carry)[2] + ((double*)&carry)[3];
	/*
	* double c = 0.0;
	for (int i = 0; i < 4; i++)
	{
		double y = -carry.m256d_f64[i] - c;
		double t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}*/

	for (int i = 0; i < 4; i++)
	{
		double y = ((double*)&src)[i] - c;
		double t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	return sum;
}
#pragma endregion

STATIC_INLINE __m256 _mm256_div_avoidzerodiv_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_div_ps(src1, _mm256_blendv_ps(src2, _mm256_set1_ps(FLT_MIN), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0)));
	//return _mm256_div_ps(src1, _mm256_max_ps(src2, _mm256_set1_ps(FLT_EPSILON)));
}

STATIC_INLINE __m256 _mm256_div_zerodivzero_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_blendv_ps(_mm256_div_ps(src1, src2), _mm256_set1_ps(FLT_MIN), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0));
}

STATIC_INLINE __m256 _mm256_div_zerodivone_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_blendv_ps(_mm256_div_ps(src1, src2), _mm256_set1_ps(1.f), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0));
}


STATIC_INLINE __m256 _mm256_ssd_ps(__m256 src, __m256 ref)
{
	__m256 diff = _mm256_sub_ps(src, ref);
	return _mm256_mul_ps(diff, diff);
}

STATIC_INLINE __m256 _mm256_ssd_ps(__m256 src0, __m256 src1, __m256 src2, __m256 ref0, __m256 ref1, __m256 ref2)
{
	__m256 diff = _mm256_sub_ps(src0, ref0);
	__m256 ret = _mm256_mul_ps(diff, diff);
	diff = _mm256_sub_ps(src1, ref1);
	ret = _mm256_fmadd_ps(diff, diff, ret);
	diff = _mm256_sub_ps(src2, ref2);
	ret = _mm256_fmadd_ps(diff, diff, ret);
	return ret;
}

STATIC_INLINE __m256 _mm256_ssdepi32_ps(__m256i src0, __m256i src1, __m256i src2, __m256i ref0, __m256i ref1, __m256i ref2)
{
	__m256 diff = _mm256_cvtepi32_ps(_mm256_sub_epi32(src0, ref0));
	__m256 ret = _mm256_mul_ps(diff, diff);
	diff = _mm256_cvtepi32_ps(_mm256_sub_epi32(src1, ref1));
	ret = _mm256_fmadd_ps(diff, diff, ret);
	diff = _mm256_cvtepi32_ps(_mm256_sub_epi32(src2, ref2));
	ret = _mm256_fmadd_ps(diff, diff, ret);
	return ret;
}

//return offset + ssd
STATIC_INLINE __m256 _mm256_ssdadd_ps(__m256 offset, __m256 src, __m256 ref)
{
	__m256 diff = _mm256_sub_ps(src, ref);
	return _mm256_fmadd_ps(diff, diff, offset);
}

//return offset + ssd
STATIC_INLINE __m256 _mm256_ssdadd_ps(__m256 offset, __m256 src0, __m256 src1, __m256 src2, __m256 ref0, __m256 ref1, __m256 ref2)
{
	__m256 diff = _mm256_sub_ps(src0, ref0);
	__m256 ret = _mm256_fmadd_ps(diff, diff, offset);
	diff = _mm256_sub_ps(src1, ref1);
	ret = _mm256_fmadd_ps(diff, diff, ret);
	diff = _mm256_sub_ps(src2, ref2);
	ret = _mm256_fmadd_ps(diff, diff, ret);
	return ret;
}

//(s0-r0)^2+(s1-r1)^2+(s2-r2)^2
STATIC_INLINE __m256 _mm256_quadrance_ps(__m256 src0, __m256 src1, __m256 src2)
{
	__m256 ret = _mm256_mul_ps(src0, src0);
	ret = _mm256_fmadd_ps(src1, src1, ret);
	ret = _mm256_fmadd_ps(src2, src2, ret);
	return ret;
}

STATIC_INLINE __m128i _mm_absdiff_epu8(__m128i src1, __m128i src2)
{
	return _mm_max_epu8(_mm_subs_epu8(src1, src2), _mm_subs_epu8(src2, src1));
}

STATIC_INLINE __m256i _mm256_absdiff_epu8(__m256i src1, __m256i src2)
{
	return _mm256_max_epu8(_mm256_subs_epu8(src1, src2), _mm256_subs_epu8(src2, src1));
}

STATIC_INLINE __m128i _mm_absdiff_epi32(__m128i src1, __m128i src2)
{
	return _mm_abs_epi32(_mm_sub_epi32(src1, src2));
}

STATIC_INLINE __m256i _mm256_absdiff_epi32(__m256i src1, __m256i src2)
{
	return _mm256_abs_epi32(_mm256_sub_epi32(src1, src2));
}

STATIC_INLINE __m128 _mm_absdiff_ps(__m128 src1, __m128 src2)
{
	return _mm_abs_ps(_mm_sub_ps(src1, src2));
}

STATIC_INLINE __m256 _mm256_absdiff_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_andnot_ps(_mm256_set1_ps(-0.f), _mm256_sub_ps(src1, src2));
}

STATIC_INLINE void _mm256_absdiff_psx2(const __m256 src0, const __m256 ref0, const __m256 src1, const __m256 ref1, __m256& d0, __m256& d1)
{
	d0 = _mm256_sub_ps(src0, ref0);
	d1 = _mm256_sub_ps(src1, ref1);
	d0 = _mm256_abs_ps(d0);
	d1 = _mm256_abs_ps(d1);
}

STATIC_INLINE void _mm256_absdiff_psx4(const __m256 src0, const __m256 ref0, const __m256 src1, const __m256 ref1, const __m256 src2, const __m256 ref2, const __m256 src3, const __m256 ref3, __m256& d0, __m256& d1, __m256& d2, __m256& d3)
{
	d0 = _mm256_sub_ps(src0, ref0);
	d1 = _mm256_sub_ps(src1, ref1);
	d2 = _mm256_sub_ps(src2, ref2);
	d3 = _mm256_sub_ps(src3, ref3);
	d0 = _mm256_abs_ps(d0);
	d1 = _mm256_abs_ps(d1);
	d2 = _mm256_abs_ps(d2);
	d3 = _mm256_abs_ps(d3);
}

STATIC_INLINE void _mm256_absdiff_psx8(const __m256 src0, const __m256 ref0, const __m256 src1, const __m256 ref1, const __m256 src2, const __m256 ref2, const __m256 src3, const __m256 ref3, const __m256 src4, const __m256 ref4, const __m256 src5, const __m256 ref5, const __m256 src6, const __m256 ref6, const __m256 src7, const __m256 ref7, __m256& d0, __m256& d1, __m256& d2, __m256& d3, __m256& d4, __m256& d5, __m256& d6, __m256& d7)
{
	d0 = _mm256_sub_ps(src0, ref0);
	d1 = _mm256_sub_ps(src1, ref1);
	d2 = _mm256_sub_ps(src2, ref2);
	d3 = _mm256_sub_ps(src3, ref3);
	d4 = _mm256_sub_ps(src4, ref4);
	d5 = _mm256_sub_ps(src5, ref5);
	d6 = _mm256_sub_ps(src6, ref6);
	d7 = _mm256_sub_ps(src7, ref7);
	d0 = _mm256_abs_ps(d0);
	d1 = _mm256_abs_ps(d1);
	d2 = _mm256_abs_ps(d2);
	d3 = _mm256_abs_ps(d3);
	d4 = _mm256_abs_ps(d4);
	d5 = _mm256_abs_ps(d5);
	d6 = _mm256_abs_ps(d6);
	d7 = _mm256_abs_ps(d7);
}

//rsqrt->rcp
STATIC_INLINE __m256 _mm256_fastsqrt_ps(__m256 x)
{
	return _mm256_rcp_ps(_mm256_rsqrt_ps(x));
}

//rcp with newton-raphson 1-iteration
STATIC_INLINE __m256 _mm256_rcpnr_ps(__m256 x)
{
	__m256 res = _mm256_rcp_ps(x);
	//rcp*(2-rcp*x)->(rcp+rcp)-rcp*rcp*x
	return res = _mm256_sub_ps(_mm256_add_ps(res, res), _mm256_mul_ps(x, _mm256_mul_ps(res, res)));
}

//rcp with newton-raphson 1-iteration (FMA ver) requided set2
STATIC_INLINE __m256 _mm256_rcpnr_fma_ps(__m256 x, __m256 two = _mm256_set1_ps(2.f))
{
	__m256 rcp = _mm256_rcp_ps(x);
	//rcp*(2-rcp*x)
	return _mm256_mul_ps(rcp, _mm256_fnmadd_ps(x, rcp, two));
}


#pragma endregion

#pragma region compare
STATIC_INLINE __m128i _mm_cmpgt_epu8(__m128i x, __m128i y)
{
	return _mm_andnot_si128(_mm_cmpeq_epi8(x, y), _mm_cmpeq_epi8(_mm_max_epu8(x, y), x));
}


STATIC_INLINE __m256i _mm256_cmpgt_epu8(__m256i x, __m256i y)
{
	//return _mm256_andnot_si256(_mm256_cmpeq_epi8(x, y), _mm256_cmpeq_epi8(_mm256_max_epu8(x, y), x));
	const __m256i highBit = _mm256_set1_epi8(0x80);
	return _mm256_cmpgt_epi8(_mm256_xor_si256(x, highBit), _mm256_xor_si256(y, highBit));
}

#pragma endregion

#pragma region bit manipulation
STATIC_INLINE __m256i _mm256_not_si256(__m256i src)
{
	return _mm256_xor_si256(src, _mm256_cmpeq_epi8(src, src));
	//return _mm256_xor_si256(src, _mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()));
}

STATIC_INLINE __m256 _mm256_not_ps(__m256 src)
{
	return _mm256_xor_ps(src, _mm256_cmp_ps(src, src, 0));
}

STATIC_INLINE __m256d _mm256_not_pd(__m256d src)
{
	return _mm256_xor_pd(src, _mm256_cmp_pd(src, src, 0));
}

//1: same, 0: not same
STATIC_INLINE int _mm256_testsame_ps(__m256 a, __m256 b)
{
	return _mm256_testc_si256(_mm256_castps_si256(a), _mm256_castps_si256(b));
}

//1: same, 0: not same
STATIC_INLINE int _mm256_testsame_pd(__m256d a, __m256d b)
{
	return _mm256_testc_si256(_mm256_castpd_si256(a), _mm256_castpd_si256(b));
}
#pragma endregion

#pragma region print
STATIC_INLINE void print(__m128d src)
{
	printf_s("%5.3f %5.3f\n",
		((double*)&src)[0], ((double*)&src)[1]);
	//
	//src.m128d_f64[0], src.m128d_f64[1]);
}

#define print_m128d(src) printf_s("%s: %6.2f %6.2f\n",#src,((double*)&src)[0], ((double*)&src)[1]);
#define print_m256d(src) printf_s("%s: %6.2f %6.2f | %6.2f %6.2f\n",#src,((double*)&src)[0], ((double*)&src)[1], ((double*)&src)[2], ((double*)&src)[3]);

STATIC_INLINE void print(__m256d src)
{
	printf_s("%6.2f %6.2f %6.2f %6.2f\n", ((double*)&src)[0], ((double*)&src)[1], ((double*)&src)[2], ((double*)&src)[3]);
}

STATIC_INLINE void print(__m128 src)
{
	printf_s("%5.3f %5.3f %5.3f %5.3f\n",
		((float*)&src)[0], ((float*)&src)[1], ((float*)&src)[2], ((float*)&src)[3]);
}


#define print_m128(src) printf_s("%s: %6.2f %6.2f %6.2f %6.2f\n",#src,((float*)&src)[0], ((float*)&src)[1], ((float*)&src)[2], ((float*)&src)[3]);
#define print_m256(src) printf_s("%s: %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f\n",#src,((float*)&src)[0], ((float*)&src)[1], ((float*)&src)[2], ((float*)&src)[3], ((float*)&src)[4], ((float*)&src)[5], ((float*)&src)[6], ((float*)&src)[7]);

STATIC_INLINE void print(__m256 src)
{
	printf_s("%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f\n",
		((float*)&src)[0], ((float*)&src)[1], ((float*)&src)[2], ((float*)&src)[3], ((float*)&src)[4], ((float*)&src)[5], ((float*)&src)[6], ((float*)&src)[7]);
}

STATIC_INLINE void print_char(__m128i src)
{
	char* data = (char*)&src;
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_char(__m256i src)
{
	char* data = (char*)&src;
	for (int i = 0; i < 32; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_uchar(__m128i src)
{
	uchar* data = (uchar*)&src;
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

#define print_m128i_uchar(src) printf_s("%s: %3d %3d %3d %3d;%3d %3d %3d %3d;%3d %3d %3d %3d;%3d %3d %3d %3d\n",#src,((uchar*)&src)[0], ((uchar*)&src)[1], ((uchar*)&src)[2], ((uchar*)&src)[3], ((uchar*)&src)[4], ((uchar*)&src)[5], ((uchar*)&src)[6], ((uchar*)&src)[7],((uchar*)&src)[8], ((uchar*)&src)[9], ((uchar*)&src)[10], ((uchar*)&src)[11], ((uchar*)&src)[12], ((uchar*)&src)[13], ((uchar*)&src)[14], ((uchar*)&src)[15]);
#define print_m256i_uchar(src) printf_s("%s: %3d %3d %3d %3d;%3d %3d %3d %3d;%3d %3d %3d %3d;%3d %3d %3d %3d|%3d %3d %3d %3d;%3d %3d %3d %3d;%3d %3d %3d %3d;%3d %3d %3d %3d\n",#src,((uchar*)&src)[0], ((uchar*)&src)[1], ((uchar*)&src)[2], ((uchar*)&src)[3], ((uchar*)&src)[4], ((uchar*)&src)[5], ((uchar*)&src)[6], ((uchar*)&src)[7],((uchar*)&src)[8], ((uchar*)&src)[9], ((uchar*)&src)[10], ((uchar*)&src)[11], ((uchar*)&src)[12], ((uchar*)&src)[13], ((uchar*)&src)[14], ((uchar*)&src)[15],((uchar*)&src)[16], ((uchar*)&src)[17], ((uchar*)&src)[18], ((uchar*)&src)[19], ((uchar*)&src)[20], ((uchar*)&src)[21], ((uchar*)&src)[22], ((uchar*)&src)[23],((uchar*)&src)[24], ((uchar*)&src)[25], ((uchar*)&src)[26], ((uchar*)&src)[27], ((uchar*)&src)[28], ((uchar*)&src)[29], ((uchar*)&src)[30], ((uchar*)&src)[31]);
#define print_m256i_char(src) printf_s("%s: %4d %4d %4d %4d;%4d %4d %4d %4d;%4d %4d %4d %4d;%4d %4d %4d %4d|%4d %4d %4d %4d;%4d %4d %4d %4d;%4d %4d %4d %4d;%4d %4d %4d %4d\n",#src,((char*)&src)[0], ((char*)&src)[1], ((char*)&src)[2], ((char*)&src)[3], ((char*)&src)[4], ((char*)&src)[5], ((char*)&src)[6], ((char*)&src)[7],((char*)&src)[8], ((char*)&src)[9], ((char*)&src)[10], ((char*)&src)[11], ((char*)&src)[12], ((char*)&src)[13], ((char*)&src)[14], ((char*)&src)[15],((char*)&src)[16], ((char*)&src)[17], ((char*)&src)[18], ((char*)&src)[19], ((char*)&src)[20], ((char*)&src)[21], ((char*)&src)[22], ((char*)&src)[23],((char*)&src)[24], ((char*)&src)[25], ((char*)&src)[26], ((char*)&src)[27], ((char*)&src)[28], ((char*)&src)[29], ((char*)&src)[30], ((char*)&src)[31]);

#define print_uchar(src) std::cout << #src << ": ";\
for (int i = 0; i < 32; i++){if(i%8==0)printf_s("|%3d ", ((uchar*)&src)[i]);else printf_s("%3d ", ((uchar*)&src)[i]);}printf_s("|\n");


STATIC_INLINE void print_short(__m128i src)
{
	short* data = (short*)&src;
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

#define print_m128i_short(src) printf_s("%s: %3d %3d %3d %3d;%3d %3d %3d %3d\n",#src,((short*)&src)[0], ((short*)&src)[1], ((short*)&src)[2], ((short*)&src)[3], ((short*)&src)[4], ((short*)&src)[5], ((short*)&src)[6], ((short*)&src)[7]);
#define print_m256i_short(src) printf_s("%s: %3d %3d %3d %3d;%3d %3d %3d %3d|%3d %3d %3d %3d;%3d %3d %3d %3d\n",#src,((short*)&src)[0], ((short*)&src)[1], ((short*)&src)[2], ((short*)&src)[3], ((short*)&src)[4], ((short*)&src)[5], ((short*)&src)[6], ((short*)&src)[7],((short*)&src)[8], ((short*)&src)[9], ((short*)&src)[10], ((short*)&src)[11], ((short*)&src)[12], ((short*)&src)[13], ((short*)&src)[14], ((short*)&src)[15]);

#define print_short(src) std::cout << #src << ": ";\
for (int i = 0; i < 16; i++){if(i%8==0)printf_s("|%4d ", ((short*)&src)[i]);else printf_s("%4d ", ((short*)&src)[i]);}printf_s("|\n");

STATIC_INLINE void print_ushort(__m128i src)
{
	ushort* data = (ushort*)&src;
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_ushort(__m256i src)
{
	ushort* data = (ushort*)&src;
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

#define print_m128i_int(src) printf_s("%s: %3d %3d %3d %3d\n",#src,((int*)&src)[0], ((int*)&src)[1], ((int*)&src)[2], ((int*)&src)[3]);
#define print_m256i_int(src) printf_s("%s: %3d %3d %3d %3d | %3d %3d %3d %3d\n",#src,((int*)&src)[0], ((int*)&src)[1], ((int*)&src)[2], ((int*)&src)[3], ((int*)&src)[4], ((int*)&src)[5], ((int*)&src)[6], ((int*)&src)[7]);

STATIC_INLINE void print_int(__m128i src)
{
	int* data = (int*)&src;
	for (int i = 0; i < 4; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_int(__m256i src)
{
	int* data = (int*)&src;
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_uint(__m128i src)
{
	unsigned int* data = (unsigned int*)&src;
	for (int i = 0; i < 4; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_uint(__m256i src)
{
	unsigned int* data = (unsigned int*)&src;
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_long(__m256i src)
{
	long long* data = (long long*)&src;
	for (int i = 0; i < 4; i++)
	{
		printf_s("%3lld ", data[i]);
	}
	printf_s("\n");
}

STATIC_INLINE void print_simd(__m128 src) { print_m128(src); }
STATIC_INLINE void print_simd(__m128i src) { print_int(src); }
STATIC_INLINE void print_simd(__m128d src) { print_m128d(src); }
STATIC_INLINE void print_simd(__m256i src) { print_int(src); }
STATIC_INLINE void print_simd(__m256  src) { print_m256(src); }
STATIC_INLINE void print_simd(__m256d src) { print_m256d(src); }
#ifdef CP_AVX512
STATIC_INLINE void print_simd(__m512  src) {}
STATIC_INLINE void print_simd(__m512d src) {}
STATIC_INLINE void print_simd(__m512i src) {}
#endif
#pragma endregion


//broadcast
/*
__m128 xxxx = _mm_shuffle_ps(first, first, 0x00); // _MM_SHUFFLE(0, 0, 0, 0)
__m128 yyyy = _mm_shuffle_ps(first, first, 0x55); // _MM_SHUFFLE(1, 1, 1, 1)
__m128 zzzz = _mm_shuffle_ps(first, first, 0xAA); // _MM_SHUFFLE(2, 2, 2, 2)
__m128 wwww = _mm_shuffle_ps(first, first, 0xFF); // _MM_SHUFFLE(3, 3, 3, 3)
*/
#pragma region store_auto_color
STATIC_INLINE void _mm256_storeu_auto_color(float* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_storeu_ps_color(dest, b, g, r);
}

STATIC_INLINE void _mm256_storeu_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_storeu_ps2epu8_color(dest, b, g, r);
}

STATIC_INLINE void _mm256_store_auto_color(float* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_store_ps_color(dest, b, g, r);
}

STATIC_INLINE void _mm256_store_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_store_ps2epu8_color(dest, b, g, r);
}

STATIC_INLINE void _mm256_store_auto_color(uchar* dest, __m256i b, __m256i g, __m256i r)
{
	_mm256_store_epi8_color(dest, b, g, r);
}

STATIC_INLINE void _mm256_storeu_auto_color(uchar* dest, __m256i b, __m256i g, __m256i r)
{
	_mm256_storeu_epi8_color(dest, b, g, r);
}


STATIC_INLINE void _mm256_stream_auto_color(float* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_stream_ps_color(dest, b, g, r);
}

STATIC_INLINE void _mm256_stream_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_stream_ps2epu8_color(dest, b, g, r);
}
#pragma endregion

#pragma region store

STATIC_INLINE void _mm256_store_cvtps_epu8(__m128i* dest, __m256 ms)
{
	_mm_storel_epi64(dest, _mm256_cvtps_epu8(ms));
}

STATIC_INLINE void _mm256_storescalar_cvtps_epu8(void* dst, __m256 src, const int numpixel)
{
	uchar CV_DECL_ALIGNED(32) buffscalarstore[32];
	_mm256_store_cvtps_epu8((__m128i*)buffscalarstore, src);
	uchar* dest = (uchar*)dst;
	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm256_storescalar_pd(uchar* dst, __m256d src, const int numpixel)
{
	double CV_DECL_ALIGNED(32) buffscalarstore[4];
	_mm256_store_pd(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = cv::saturate_cast<uchar>(buffscalarstore[i]);
}

STATIC_INLINE void _mm256_storescalar_pd(float* dst, __m256d src, const int numpixel)
{
	double CV_DECL_ALIGNED(32) buffscalarstore[4];
	_mm256_store_pd(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = (float)buffscalarstore[i];
}

STATIC_INLINE void _mm256_storescalar_pd(double* dst, __m256d src, const int numpixel)
{
	double CV_DECL_ALIGNED(32) buffscalarstore[4];
	_mm256_store_pd(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm256_storescalar_ps(float* dst, __m256 src, const int numpixel)
{
	float CV_DECL_ALIGNED(32) buffscalarstore[8];
	_mm256_store_ps(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm256_i32scaterscalar_epu8(uchar* dest, __m256i vindex, __m256 src)
{
	__m128i v = _mm256_cvtps_epu8(src);
	dest[((int*)&vindex)[0]] = ((uchar*)&v)[0];
	dest[((int*)&vindex)[1]] = ((uchar*)&v)[1];
	dest[((int*)&vindex)[2]] = ((uchar*)&v)[2];
	dest[((int*)&vindex)[3]] = ((uchar*)&v)[3];
	dest[((int*)&vindex)[4]] = ((uchar*)&v)[4];
	dest[((int*)&vindex)[5]] = ((uchar*)&v)[5];
	dest[((int*)&vindex)[6]] = ((uchar*)&v)[6];
	dest[((int*)&vindex)[7]] = ((uchar*)&v)[7];
}

STATIC_INLINE void _mm_i32scaterscalar_ps(float* dest, __m128i vindex, __m128 src)
{
	dest[((int*)&vindex)[0]] = ((float*)&src)[0];
	dest[((int*)&vindex)[1]] = ((float*)&src)[1];
	dest[((int*)&vindex)[2]] = ((float*)&src)[2];
	dest[((int*)&vindex)[3]] = ((float*)&src)[3];
}

STATIC_INLINE void _mm256_i32scaterscalar_ps(float* dest, __m256i vindex, __m256 src)
{
	dest[((int*)&vindex)[0]] = ((float*)&src)[0];
	dest[((int*)&vindex)[1]] = ((float*)&src)[1];
	dest[((int*)&vindex)[2]] = ((float*)&src)[2];
	dest[((int*)&vindex)[3]] = ((float*)&src)[3];
	dest[((int*)&vindex)[4]] = ((float*)&src)[4];
	dest[((int*)&vindex)[5]] = ((float*)&src)[5];
	dest[((int*)&vindex)[6]] = ((float*)&src)[6];
	dest[((int*)&vindex)[7]] = ((float*)&src)[7];
}

STATIC_INLINE void _mm256_i32scaterscalar_epu8_color(uchar* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	__m128i bb = _mm256_cvtps_epu8(b);
	__m128i gb = _mm256_cvtps_epu8(g);
	__m128i rb = _mm256_cvtps_epu8(r);
	int idx = ((int*)&vindex)[0];
	dest[idx + 0] = ((uchar*)&bb)[0];	dest[idx + 1] = ((uchar*)&gb)[0];	dest[idx + 2] = ((uchar*)&rb)[0];
	idx = ((int*)&vindex)[1];
	dest[idx + 0] = ((uchar*)&bb)[1];	dest[idx + 1] = ((uchar*)&gb)[1];	dest[idx + 2] = ((uchar*)&rb)[1];
	idx = ((int*)&vindex)[2];
	dest[idx + 0] = ((uchar*)&bb)[2];	dest[idx + 1] = ((uchar*)&gb)[2];	dest[idx + 2] = ((uchar*)&rb)[2];
	idx = ((int*)&vindex)[3];
	dest[idx + 0] = ((uchar*)&bb)[3];	dest[idx + 1] = ((uchar*)&gb)[3];	dest[idx + 2] = ((uchar*)&rb)[3];
	idx = ((int*)&vindex)[4];
	dest[idx + 0] = ((uchar*)&bb)[4];	dest[idx + 1] = ((uchar*)&gb)[4];	dest[idx + 2] = ((uchar*)&rb)[4];
	idx = ((int*)&vindex)[5];
	dest[idx + 0] = ((uchar*)&bb)[5];	dest[idx + 1] = ((uchar*)&gb)[5];	dest[idx + 2] = ((uchar*)&rb)[5];
	idx = ((int*)&vindex)[6];
	dest[idx + 0] = ((uchar*)&bb)[6];	dest[idx + 1] = ((uchar*)&gb)[6];	dest[idx + 2] = ((uchar*)&rb)[6];
	idx = ((int*)&vindex)[7];
	dest[idx + 0] = ((uchar*)&bb)[7];	dest[idx + 1] = ((uchar*)&gb)[7];	dest[idx + 2] = ((uchar*)&rb)[7];
}

STATIC_INLINE void _mm256_i32scaterscalar_ps_color(float* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	int idx = ((int*)&vindex)[0];
	dest[idx + 0] = ((float*)&b)[0];	dest[idx + 1] = ((float*)&g)[0];	dest[idx + 2] = ((float*)&r)[0];
	idx = ((int*)&vindex)[1];
	dest[idx + 0] = ((float*)&b)[1];	dest[idx + 1] = ((float*)&g)[1];	dest[idx + 2] = ((float*)&r)[1];
	idx = ((int*)&vindex)[2];
	dest[idx + 0] = ((float*)&b)[2];	dest[idx + 1] = ((float*)&g)[2];	dest[idx + 2] = ((float*)&r)[2];
	idx = ((int*)&vindex)[3];
	dest[idx + 0] = ((float*)&b)[3];	dest[idx + 1] = ((float*)&g)[3];	dest[idx + 2] = ((float*)&r)[3];
	idx = ((int*)&vindex)[4];
	dest[idx + 0] = ((float*)&b)[4];	dest[idx + 1] = ((float*)&g)[4];	dest[idx + 2] = ((float*)&r)[4];
	idx = ((int*)&vindex)[5];
	dest[idx + 0] = ((float*)&b)[5];	dest[idx + 1] = ((float*)&g)[5];	dest[idx + 2] = ((float*)&r)[5];
	idx = ((int*)&vindex)[6];
	dest[idx + 0] = ((float*)&b)[6];	dest[idx + 1] = ((float*)&g)[6];	dest[idx + 2] = ((float*)&r)[6];
	idx = ((int*)&vindex)[7];
	dest[idx + 0] = ((float*)&b)[7];	dest[idx + 1] = ((float*)&g)[7];	dest[idx + 2] = ((float*)&r)[7];
}
#pragma endregion

#pragma region set
STATIC_INLINE __m256i _mm256_set1_auto(uchar src)
{
	return _mm256_set1_epi8(src);
}
STATIC_INLINE __m256i _mm256_set1_auto(short src)
{
	return _mm256_set1_epi16(src);
}
STATIC_INLINE __m256i _mm256_set1_auto(int src)
{
	return _mm256_set1_epi32(src);
}
STATIC_INLINE __m256 _mm256_set1_auto(float src)
{
	return _mm256_set1_ps(src);
}
STATIC_INLINE __m256d _mm256_set1_auto(double src)
{
	return _mm256_set1_pd(src);
}

#pragma endregion
STATIC_INLINE void _mm256_stream_auto(uchar* dest, __m256 ms)
{
	_mm256_store_cvtps_epu8((__m128i*)dest, ms);
}

STATIC_INLINE void _mm256_stream_auto(float* dest, __m256 ms)
{
	_mm256_stream_ps(dest, ms);
}

STATIC_INLINE void _mm256_maskstore_auto(float* dest, __m256i mask, __m256 ms)
{
	_mm256_maskstore_ps(dest, mask, ms);
}

STATIC_INLINE void _mm256_maskstore_auto(double* dest, __m256i mask, __m256d ms)
{
	_mm256_maskstore_pd(dest, mask, ms);
}

STATIC_INLINE void _mm256_maskstore_auto(uchar* dest, __m256i mask, __m256 ms)
{
	uchar CV_DECL_ALIGNED(32) buffscalarstore[32];
	_mm256_store_cvtps_epu8((__m128i*)buffscalarstore, ms);
	for (int i = 0; i < 8; i++)
	{
		if (((int*)&mask)[i] == 255) dest[i] = buffscalarstore[i];
	}
}

STATIC_INLINE void _mm256_storescalar_auto(uchar* dest, __m256 ms, const int numpixel)
{
	_mm256_storescalar_cvtps_epu8(dest, ms, numpixel);
}

STATIC_INLINE void _mm256_storescalar_auto(float* dest, __m256 ms, const int numpixel)
{
	_mm256_storescalar_ps(dest, ms, numpixel);
}

STATIC_INLINE void _mm256_storescalar_auto(uchar* dest, __m256d ms, const int numpixel)
{
	_mm256_storescalar_pd(dest, ms, numpixel);
}

STATIC_INLINE void _mm256_storescalar_auto(float* dest, __m256d ms, const int numpixel)
{
	_mm256_storescalar_pd(dest, ms, numpixel);
}

STATIC_INLINE void _mm256_storescalar_auto(double* dest, __m256d ms, const int numpixel)
{
	_mm256_storescalar_pd(dest, ms, numpixel);
}

STATIC_INLINE void _mm256_storescalar_auto_color(float* dest, __m256 b, __m256 g, __m256 r, const int numpixel)
{
	_mm256_storescalar_ps_color(dest, b, g, r, numpixel);
}

STATIC_INLINE void _mm256_storescalar_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r, const int numpixel)
{
	_mm256_storescalar_ps2epu8_color(dest, b, g, r, numpixel);
}






#pragma region gather-scatter
//return 8 uchar elements
STATIC_INLINE __m128i _mm_i8gather_epi32(const uchar* src, __m128i idx)
{
	return _mm_srli_epi32(_mm_i32gather_epi32(reinterpret_cast<const int*>(&src[-3]), idx, 1), 24);
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

//gather bgr interleved uchar data with convert epi32->ps
STATIC_INLINE void _mm256_i32gather_bgr_ps(const uchar* src, __m256i idx, __m256& b, __m256& g, __m256& r)
{
	__m256i v = _mm256_i32gather_epi32((int*)(src), idx, 1);

	b = _mm256_cvtepi32_ps(_mm256_blendv_epi8(v, _mm256_setzero_si256(), _mm256_setr_epi8(0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF)));
	g = _mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_blendv_epi8(v, _mm256_setzero_si256(), _mm256_setr_epi8(0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF)), 8));
	r = _mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_blendv_epi8(v, _mm256_setzero_si256(), _mm256_setr_epi8(0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF)), 16));
}

STATIC_INLINE void _mm256_i32gather_bgr_epi32(const uchar* src, __m256i idx, __m256i& b, __m256i& g, __m256i& r)
{
	__m256i v = _mm256_i32gather_epi32((int*)(src - 3), idx, 1);
	b = _mm256_srli_epi32(v, 24);
	g = _mm256_srli_epi32(v, 25);
	r = _mm256_srli_epi32(v, 26);
}

//gather 8 uchar elements and output 32bit integer
//old STATIC_INLINE __m256i _mm256_i32gather_epi32(const uchar* src, __m256i idx)
STATIC_INLINE __m256i _mm256_i32gather_epi8epi32(const uchar* src, __m256i idx)
{
	return _mm256_srli_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(&src[-3]), idx, 1), 24);
	//return _mm256_setr_epi32(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]]);
}

STATIC_INLINE __m128 _mm_i8gather_ps(const uchar* src, __m128i idx)
{
	return _mm_cvtepi32_ps(_mm_srli_epi32(_mm_i32gather_epi32(reinterpret_cast<const int*>(&src[-3]), idx, 1), 24));
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

//gather 4 uchar elements
STATIC_INLINE __m128i _mm_i32gather_epu8(const uchar* src, __m128i idx)
{
	return _mm_cvtepi32_epu8(_mm_srli_epi32(_mm_i32gather_epi32(reinterpret_cast<const int*>(&src[-3]), idx, 1), 24));
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

//gather 16 uchar elements
STATIC_INLINE __m128i _mm_i8gather_epu8(const uchar* src, __m128i idx)
{
	const char* i = (char*)&idx;
	return _mm_setr_epi8(src[i[0]], src[i[1]], src[i[2]], src[i[3]], src[i[4]], src[i[5]], src[i[6]], src[i[7]], src[i[8]], src[i[9]], src[i[10]], src[i[11]], src[i[12]], src[i[13]], src[i[14]], src[i[15]]);
}

//gather 16 uchar elements
STATIC_INLINE __m128i _mm256_i16gather_epu8(const uchar* src, __m256i idx)
{
	const short* i = (short*)&idx;
	return _mm_setr_epi8(src[i[0]], src[i[1]], src[i[2]], src[i[3]], src[i[4]], src[i[5]], src[i[6]], src[i[7]], src[i[8]], src[i[9]], src[i[10]], src[i[11]], src[i[12]], src[i[13]], src[i[14]], src[i[15]]);
}

//dest: 8 uchar elements
STATIC_INLINE __m128i _mm256_i32gather_epu8(const uchar* src, __m256i idx)
{
	return _mm256_cvtepi32_epu8(_mm256_srli_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(src - 3), idx, 1), 24));
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

//dest: 8 uchar elements
STATIC_INLINE __m256 _mm256_i32gather_epu8cvtps(const uchar* src, __m256i idx)
{
	return _mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(src - 3), idx, 1), 24));
}

STATIC_INLINE __m256 _mm256_i32gather_auto(float* src, __m256i idx)
{
	return _mm256_i32gather_ps(src, idx, 4);
}

STATIC_INLINE __m256 _mm256_i32gather_auto(uchar* src, __m256i idx)
{
	return _mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(&src[-3]), idx, 1), 24));
}

STATIC_INLINE __m256 _mm256_i32gatherset_auto(uchar* src, __m256i idx)
{

	return _mm256_cvtepi32_ps(_mm256_setr_epi8(src[((int*)&idx)[0]], src[((int*)&idx)[1]], src[((int*)&idx)[2]], src[((int*)&idx)[3]], src[((int*)&idx)[4]], src[((int*)&idx)[5]], src[((int*)&idx)[6]], src[((int*)&idx)[7]],
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

STATIC_INLINE __m256 _mm256_i32gatherset_auto(float* src, __m256i idx)
{
	return _mm256_setr_ps(src[((int*)&idx)[0]], src[((int*)&idx)[1]], src[((int*)&idx)[2]], src[((int*)&idx)[3]], src[((int*)&idx)[4]], src[((int*)&idx)[5]], src[((int*)&idx)[6]], src[((int*)&idx)[7]]);
}

STATIC_INLINE void _mm256_i32scaterscalar_auto(uchar* dest, __m256i vindex, __m256 src)
{
	_mm256_i32scaterscalar_epu8(dest, vindex, src);
}

STATIC_INLINE void _mm256_i32scaterscalar_auto(float* dest, __m256i vindex, __m256 src)
{
	_mm256_i32scaterscalar_ps(dest, vindex, src);
}

STATIC_INLINE void _mm256_i32scaterscalar_auto_color(uchar* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	_mm256_i32scaterscalar_epu8_color(dest, vindex, b, g, r);
}

STATIC_INLINE void _mm256_i32scaterscalar_auto_color(float* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	_mm256_i32scaterscalar_ps_color(dest, vindex, b, g, r);
}
#pragma endregion

STATIC_INLINE __m128 _mm_set_step_ps(float v, float step = 1.f)
{
	return _mm_setr_ps(v, v + step, v + 2.f * step, v + 3.f * step);
}

//_mm256_setr_ps(v + step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step);
STATIC_INLINE __m256 _mm256_set_step_ps(float v, float step = 1.f)
{
	return _mm256_setr_ps(v, v + step, v + 2.f * step, v + 3.f * step, v + 4.f * step, v + 5.f * step, v + 6.f * step, v + 7.f * step);
}

STATIC_INLINE __m128i _mm_set_step_epi8(char v, char step = 1)
{
	return _mm_setr_epi8(
		v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step,
		v + 8 * step, v + 9 * step, v + 10 * step, v + 11 * step, v + 12 * step, v + 13 * step, v + 14 * step, v + 15 * step);
}

STATIC_INLINE __m256i _mm256_set_step_epi8(char v, char step = 1)
{
	return _mm256_setr_epi8(
		v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step,
		v + 8 * step, v + 9 * step, v + 10 * step, v + 11 * step, v + 12 * step, v + 13 * step, v + 14 * step, v + 15 * step,
		v + 16 * step, v + 17 * step, v + 18 * step, v + 19 * step, v + 20 * step, v + 21 * step, v + 22 * step, v + 23 * step,
		v + 24 * step, v + 25 * step, v + 26 * step, v + 27 * step, v + 28 * step, v + 29 * step, v + 30 * step, v + 31 * step);
}

STATIC_INLINE __m256i _mm256_set_step_epi16(short v, short step = 1)
{
	return _mm256_setr_epi16(
		v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step,
		v + 8 * step, v + 9 * step, v + 10 * step, v + 11 * step, v + 12 * step, v + 13 * step, v + 14 * step, v + 15 * step);
}

STATIC_INLINE __m256i _mm256_set_step_epi32(int v, int step = 1)
{
	return _mm256_setr_epi32(v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step);
}

STATIC_INLINE __m256i _mm256_set_step_epi64(long v, long step = 1L)
{
	return _mm256_setr_epi64x(v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step);
}

//_mm256_setr_pd(v + step, v + 2 * step, v + 3 * step);
STATIC_INLINE __m256d _mm256_set_step_pd(double v, double step = 1.0)
{
	return _mm256_setr_pd(v, v + step, v + 2.0 * step, v + 3.0 * step);
}

STATIC_INLINE __m256 _mm256_reverse_ps(__m256 src)
{
	__m256 ret = _mm256_shuffle_ps(src, src, _MM_SHUFFLE(0, 1, 2, 3));
	ret = _mm256_permute2f128_ps(ret, ret, 1);
	return ret;
}

STATIC_INLINE __m256 _mm256_load_reverse_ps(const float* src)
{
	__m256 ret = _mm256_load_ps(src);
	return _mm256_reverse_ps(ret);
}

STATIC_INLINE __m256 _mm256_loadu_reverse_ps(const float* src)
{
	__m256 ret = _mm256_loadu_ps(src);
	return _mm256_reverse_ps(ret);
}

STATIC_INLINE __m256d _mm256_reverse_pd(__m256d src)
{
	return _mm256_permute4x64_pd(src, _MM_SHUFFLE(0, 1, 2, 3));
}

STATIC_INLINE __m256d _mm256_load_reverse_pd(const double* src)
{
	__m256d ret = _mm256_load_pd(src);
	return _mm256_reverse_pd(ret);
}

STATIC_INLINE __m256d _mm256_loadu_reverse_pd(const double* src)
{
	__m256d ret = _mm256_loadu_pd(src);
	return _mm256_reverse_pd(ret);
}

STATIC_INLINE __m256 _mm256_load_auto(const uchar* src)
{
	return _mm256_load_epu8cvtps((const __m128i*)src);
}

STATIC_INLINE __m256 _mm256_load_auto(const float* src)
{
	return _mm256_load_ps(src);
}

STATIC_INLINE __m256d _mm256_load_auto(const double* src)
{
	return _mm256_load_pd(src);
}

STATIC_INLINE __m256 _mm256_loadu_auto(const float* src)
{
	return _mm256_loadu_ps(src);
}

STATIC_INLINE __m256d _mm256_loadu_auto(const double* src)
{
	return _mm256_loadu_pd(src);
}

STATIC_INLINE __m256 _mm256_loadu_auto(const uchar* src)
{
	return _mm256_load_epu8cvtps((const __m128i*)src);
}

STATIC_INLINE __m256 _mm256_lddqu_auto(const float* src)
{
	return _mm256_lddqu_ps(src);
}

STATIC_INLINE __m256d _mm256_lddqu_auto(const double* src)
{
	return _mm256_lddqu_pd(src);
}

STATIC_INLINE __m256 _mm256_lddqu_auto(const uchar* src)
{
	return _mm256_load_epu8cvtps((const __m128i*)src);
}

STATIC_INLINE void _mm256_store_auto(uchar* dest, __m256 src)
{
	_mm256_store_cvtps_epu8((__m128i*)dest, src);
}

STATIC_INLINE void _mm256_store_auto(float* dest, __m256 src)
{
	_mm256_store_ps(dest, src);
}

STATIC_INLINE void _mm256_storeu_auto(uchar* dest, __m256 src)
{
	_mm256_store_cvtps_epu8((__m128i*)dest, src);
}

STATIC_INLINE void _mm256_storeu_auto(float* dest, __m256 src)
{
	_mm256_storeu_ps(dest, src);
}

STATIC_INLINE void _mm256_storeu_auto(double* dest, __m256d src)
{
	_mm256_storeu_pd(dest, src);
}




STATIC_INLINE __m256i _mm256_alphablend_epu8(__m256i a, __m256i b, __m256i ma)
{
	__m256i a2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a));
	__m256i b2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b));
	__m128i d1 = _mm256_cvtepi16_epu8(_mm256_add_epi16(b2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(a2, b2))));

	a2 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(a));
	b2 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(b));
	__m128i d2 = _mm256_cvtepi16_epu8(_mm256_add_epi16(b2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(a2, b2))));
	return _mm256_set_m128i(d2, d1);
}

STATIC_INLINE void _mm256_argmin_ps(const __m256 src, __m256& minval, __m256& argment, const float index)
{
	const __m256 mask = _mm256_cmp_ps(src, minval, 2);
	argment = _mm256_blendv_ps(argment, _mm256_set1_ps(index), mask);
	minval = _mm256_blendv_ps(minval, src, mask);
}

STATIC_INLINE __m256i _mm256_get_gatherIndex_border(int pad, int borderType)
{
	__m256i rem_idx;
	switch (borderType)
	{
	case cv::BORDER_REPLICATE:
	{
		switch (pad)
		{
		case 7: rem_idx = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0); break;
		case 6: rem_idx = _mm256_setr_epi32(0, 1, 1, 1, 1, 1, 1, 1); break;
		case 5: rem_idx = _mm256_setr_epi32(0, 1, 2, 2, 2, 2, 2, 2); break;
		case 4: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 3, 3, 3, 3); break;
		case 3: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 4, 4, 4); break;
		case 2: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 5, 5); break;
		case 1: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 6); break;
		default:rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); break;
		}
	}
	break;

	case cv::BORDER_REFLECT:
	{
		switch (pad)
		{
		case 7: rem_idx = _mm256_setr_epi32(0, 0, -1, -2, -3, -4, -5, -6); break;
		case 6: rem_idx = _mm256_setr_epi32(0, 1, 1, 0, -1, -2, -3, -4); break;
		case 5: rem_idx = _mm256_setr_epi32(0, 1, 2, 2, 1, 0, -1, -2); break;
		case 4: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 3, 2, 1, 0); break;
		case 3: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 4, 3, 2); break;
		case 2: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 5, 4); break;
		case 1: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 6); break;
		default:rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); break;
		}
	}
	break;

	default:
	case cv::BORDER_REFLECT101:
	{
		switch (pad)
		{
		case 7: rem_idx = _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7); break;
		case 6: rem_idx = _mm256_setr_epi32(0, 1, 0, -1, -2, -3, -4, -5); break;
		case 5: rem_idx = _mm256_setr_epi32(0, 1, 2, 1, 0, -1, -2, -3); break;
		case 4: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 2, 1, 0, -1); break;
		case 3: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 3, 2, 1); break;
		case 2: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 4, 3); break;
		case 1: rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 5); break;
		default:rem_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); break;
		}
	}
	}
	return rem_idx;
}

STATIC_INLINE __m128i _mm_get_gatherIndex_border(int pad, int borderType)
{
	__m128i rem_idx;
	switch (borderType)
	{
	case cv::BORDER_REPLICATE:
	{
		switch (pad)
		{
		case 3: rem_idx = _mm_setr_epi32(0, 0, 0, 0); break;
		case 2: rem_idx = _mm_setr_epi32(0, 1, 1, 1); break;
		case 1: rem_idx = _mm_setr_epi32(0, 1, 2, 2); break;
		default:rem_idx = _mm_setr_epi32(0, 1, 2, 3); break;
		}
	}
	break;

	case cv::BORDER_REFLECT:
	{
		switch (pad)
		{
		case 3: rem_idx = _mm_setr_epi32(0, 0, -1, -2); break;
		case 2: rem_idx = _mm_setr_epi32(0, 1, 1, 0); break;
		case 1: rem_idx = _mm_setr_epi32(0, 1, 2, 2); break;
		default:rem_idx = _mm_setr_epi32(0, 1, 2, 3); break;
		}
	}
	break;

	default:
	case cv::BORDER_REFLECT101:
	{
		switch (pad)
		{
		case 3: rem_idx = _mm_setr_epi32(0, -1, -2, -3); break;
		case 2: rem_idx = _mm_setr_epi32(0, 1, 0, -1); break;
		case 1: rem_idx = _mm_setr_epi32(0, 1, 2, 1); break;
		default:rem_idx = _mm_setr_epi32(0, 1, 2, 3); break;
		}
	}
	}
	return rem_idx;
}

enum class MM_PRINT_EXCEPTION
{
	ALL,
	NO_PRINT,
	NO_INEXACT,
};

STATIC_INLINE std::vector<std::string> _MM_PRINT_EXCEPTION(std::string mes = "", const MM_PRINT_EXCEPTION isPrint = MM_PRINT_EXCEPTION::ALL)
{
	if (mes.size() != 0) std::cout << mes << ": " << std::endl;
	std::vector<std::string> ret;
	if (_MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_INVALID)
	{
		ret.push_back("_MM_EXCEPT_INVALID");
	}
	if (_MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_DENORM)
	{
		ret.push_back("_MM_EXCEPT_DENORM");
	}
	if (_MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_DIV_ZERO)
	{
		ret.push_back("_MM_EXCEPT_DIV_ZERO");
	}
	if (_MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_OVERFLOW)
	{
		ret.push_back("_MM_EXCEPT_OVERFLOW");
	}
	if (_MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_UNDERFLOW)
	{
		ret.push_back("_MM_EXCEPT_UNDERFLOW");
	}
	if (_MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_INEXACT)
	{
		if (isPrint != MM_PRINT_EXCEPTION::NO_INEXACT)
			ret.push_back("_MM_EXCEPT_INEXACT");
	}
	if (isPrint == MM_PRINT_EXCEPTION::ALL || isPrint == MM_PRINT_EXCEPTION::NO_INEXACT)
	{
		if (ret.size() == 0) std::cout << "NO_EXCEPTION" << std::endl;

		for (int i = 0; i < ret.size(); i++)
		{
			std::cout << ret[i] << std::endl;
		}
	}
	return ret;
}

STATIC_INLINE __m128i _mm256_cvtps_pbh(__m256 a)
{
	static __m256i mask = _mm256_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 6, 7, 10, 11, 14, 15);
	__m256i t = _mm256_shuffle_epi8(_mm256_castps_si256(a), mask);
	__m256i s = _mm256_permute2x128_si256(t, t, 0x01);
	return _mm256_castsi256_si128(_mm256_blend_epi16(t, s, 0b11110000));
}
#ifdef __AVX512F__
#define _mm512_set_m128i(/* __m128i */ src3, /* __m128i */ src2, /* __m128i */ src1, /* __m128i */ src0) \
    _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(src0), (src1), 0x1),(src2), 0x2), (src3), 0x3)

#define _mm512_setr_m128i(/* __m128i */ src0, /* __m128i */ src1, /* __m128i */ src2, /* __m128i */ src3) \
    _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(src0), (src1), 0x1),(src2), 0x2), (src3), 0x3)

#define _mm512_set_m256i(/* __m256i */ src1, /* __m256i */ src0) \
    _mm512_inserti64x4(_mm512_castsi256_si512(src0), (src1), 0x1)

#define _mm512_setr_m256i(/* __m256i */ src0, /* __m256i */ src1) \
    _mm512_inserti64x4(_mm512_castsi256_si512(src0), (src1), 0x1)

//cvt
STATIC_INLINE __m128i _mm512_cvtps_epu8(const __m512 ms)
{
	//return _mm256_cvtepi32_epu8(_mm512_cvtps_epi32(ms));
	return _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(ms));
}

STATIC_INLINE __m128i _mm512_cvtepi32_epu8(const __m512i v0)
{
	return _mm_setr_epi8(((char*)&v0)[0], ((char*)&v0)[1], ((char*)&v0)[2], ((char*)&v0)[3], ((char*)&v0)[4], ((char*)&v0)[5], ((char*)&v0)[6], ((char*)&v0)[7], ((char*)&v0)[8], ((char*)&v0)[9], ((char*)&v0)[10], ((char*)&v0)[11], ((char*)&v0)[12], ((char*)&v0)[13], ((char*)&v0)[14], ((char*)&v0)[15]);
	//return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, _mm256_setzero_si256()), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//_mm512_permutexvar_ps: L3, T1
STATIC_INLINE void _mm512_cvtsoa2aos_ps(const __m512 b, const __m512 g, const __m512 r, __m512& db, __m512& dg, __m512& dr)
{
#if __USE_SCATTER_INSTRUCTION__
	static const __m512i idx = _mm512_set_epi32(45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);
	_mm512_i32scatter_ps((float*)dst + 0, idx, b, 4);
	_mm512_i32scatter_ps((float*)dst + 1, idx, g, 4);
	_mm512_i32scatter_ps((float*)dst + 2, idx, r, 4);
#else
	static const __m512i permuteIndexB = _mm512_setr_epi32(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	static const __m512i permuteIndexG = _mm512_setr_epi32(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	static const __m512i permuteIndexR = _mm512_setr_epi32(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

	static const __mmask16 blendMask1 = 0x4924;
	static const __mmask16 blendMask2 = 0x2492;
	static const __mmask16 blendMask3 = 0x9249;

	const __m512 aa = _mm512_permutexvar_ps(permuteIndexB, b);
	const __m512 bb = _mm512_permutexvar_ps(permuteIndexG, g);
	const __m512 cc = _mm512_permutexvar_ps(permuteIndexR, r);

	db = _mm512_mask_blend_ps(blendMask1, _mm512_mask_blend_ps(blendMask2, aa, bb), cc);
	dg = _mm512_mask_blend_ps(blendMask2, _mm512_mask_blend_ps(blendMask3, aa, bb), cc);
	dr = _mm512_mask_blend_ps(blendMask3, _mm512_mask_blend_ps(blendMask1, aa, bb), cc);
#endif
}

STATIC_INLINE void _mm512_cvtsoa2aos_pd(const __m512d b, const __m512d g, const __m512d r, __m512d& db, __m512d& dg, __m512d& dr)
{
#if __USE_SCATTER_INSTRUCTION__
	static const __m512i idx = _mm512_set_epi64(21, 18, 15, 12, 9, 6, 3, 0);
	_mm512_i64scatter_pd((double*)dst + 0, idx, b, 8);
	_mm512_i64scatter_pd((double*)dst + 1, idx, g, 8);
	_mm512_i64scatter_pd((double*)dst + 2, idx, r, 8);
#else
	static const __m512i permuteIndexB = _mm512_setr_epi64(0, 3, 6, 1, 4, 7, 2, 5);
	static const __m512i permuteIndexG = _mm512_setr_epi64(5, 0, 3, 6, 1, 4, 7, 2);
	static const __m512i permuteIndexR = _mm512_setr_epi64(2, 5, 0, 3, 6, 1, 4, 7);

	static const __mmask16 blendMask1 = 0b00100100;
	static const __mmask16 blendMask2 = 0b10010010;
	static const __mmask16 blendMask3 = 0b01001001;

	const __m512d aa = _mm512_permutexvar_pd(permuteIndexB, b);
	const __m512d bb = _mm512_permutexvar_pd(permuteIndexG, g);
	const __m512d cc = _mm512_permutexvar_pd(permuteIndexR, r);

	db = _mm512_mask_blend_pd(blendMask1, _mm512_mask_blend_pd(blendMask2, aa, bb), cc);
	dg = _mm512_mask_blend_pd(blendMask3, _mm512_mask_blend_pd(blendMask1, aa, bb), cc);
	dr = _mm512_mask_blend_pd(blendMask2, _mm512_mask_blend_pd(blendMask3, aa, bb), cc);
#endif
}

STATIC_INLINE void _mm512_cvtsoa2aos_epi8(const __m512i b, const __m512i g, const __m512i r, __m512i& db, __m512i& dg, __m512i& dr)
{
	static const __m512i mask1 = _mm512_set_epi8(
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0,
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0
	);
	static const __m512i mask2 = _mm512_set_epi8(
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5,
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5
	);
	static const __m512i mask3 = _mm512_set_epi8(
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10,
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10
	);
	static const __m512i pmask1 = _mm512_set_epi64(3, 2, 1, 0, 1, 0, 1, 0);
	static const __m512i pmask2 = _mm512_set_epi64(5, 4, 5, 4, 3, 2, 3, 2);
	static const __m512i pmask3 = _mm512_set_epi64(7, 6, 7, 6, 7, 6, 5, 4);

	static const __mmask64 blendMask1 = 0x4924924924924924;
	static const __mmask64 blendMask2 = 0x2492492492492492;
	static const __mmask64 blendMask3 = 0x9249249249249249;

	const __m512i aa = _mm512_shuffle_epi8(b, mask1);
	const __m512i bb = _mm512_shuffle_epi8(g, mask2);
	const __m512i cc = _mm512_shuffle_epi8(r, mask3);

	__m512i aaa = _mm512_permutexvar_epi64(pmask1, aa);
	__m512i bbb = _mm512_permutexvar_epi64(pmask1, bb);
	__m512i ccc = _mm512_permutexvar_epi64(pmask1, cc);
	db = _mm512_mask_blend_epi8(blendMask1, _mm512_mask_blend_epi8(blendMask2, aaa, bbb), ccc);

	aaa = _mm512_permutexvar_epi64(pmask2, aa);
	bbb = _mm512_permutexvar_epi64(pmask2, bb);
	ccc = _mm512_permutexvar_epi64(pmask2, cc);
	dg = _mm512_mask_blend_epi8(blendMask2, _mm512_mask_blend_epi8(blendMask3, aaa, bbb), ccc);

	aaa = _mm512_permutexvar_epi64(pmask3, aa);
	bbb = _mm512_permutexvar_epi64(pmask3, bb);
	ccc = _mm512_permutexvar_epi64(pmask3, cc);
	dr = _mm512_mask_blend_epi8(blendMask3, _mm512_mask_blend_epi8(blendMask1, aaa, bbb), ccc);
}
#pragma region load
STATIC_INLINE __m512 _mm512_load_epu8cvtps(const __m128i* P)
{
	return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)P)));
	//return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
}

STATIC_INLINE __m512 _mm512_load_auto(const uchar* src)
{
	return _mm512_load_epu8cvtps((const __m128i*)src);
}

STATIC_INLINE __m512 _mm512_load_auto(const float* src)
{
	return _mm512_load_ps(src);
}

STATIC_INLINE __m512d _mm512_load_auto(const double* src)
{
	return _mm512_load_pd(src);
}

STATIC_INLINE __m512 _mm512_loadu_epu8cvtps(const __m128i* P)
{
	return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
	//return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
}

STATIC_INLINE __m512 _mm512_loadu_auto(const uchar* src)
{
	return _mm512_loadu_epu8cvtps((const __m128i*)src);
}

STATIC_INLINE __m512 _mm512_loadu_auto(const float* src)
{
	return _mm512_loadu_ps(src);
}

STATIC_INLINE __m512d _mm512_loadu_auto(const double* src)
{
	return _mm512_loadu_pd(src);
}

#define _mm512_shuffle_epi64(/*__m512i*/ a, /*__m512i*/ b, /*const int*/ imm8)\
 _mm512_castpd_si512(_mm512_shuffle_pd(_mm512_castsi512_pd(a), _mm512_castsi512_pd(b), imm8))

#define _mm512_shuffle2_epi32(/*__m512i*/ a, /*__m512i*/ b, /*const int*/ imm8)\
 _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(a), _mm512_castsi512_ps(b), imm8))

STATIC_INLINE void _mm512_load_cvtepu8_bgr2planar_si512(const uchar* ptr, __m512i& b, __m512i& g, __m512i& r)
{
	//b:    0 100 200   1 | 101 201   2 102 | 202   3 103 203 |   4 104 204   5 | 105 205   6 106 | 206   7 107 207 |   8 108 208   9 | 109 209  10 110 | 210  11 111 211 |  12 112 212  13 | 113 213  14 114 | 214  15 115 215 |  16 116 216  17 | 117 217  18 118 | 218  19 119 219 |  20 120 220  21
	//g : 121 221  22 122 | 222  23 123 223 |  24 124 224  25 | 125 225  26 126 | 226  27 127 227 |  28 128 228  29 | 129 229  30 130 | 230  31 131 231 |  32 132 232  33 | 133 233  34 134 | 234  35 135 235 |  36 136 236  37 | 137 237  38 138 | 238  39 139 239 |  40 140 240  41 | 141 241  42 142
	//r : 242  43 143 243 |  44 144 244  45 | 145 245  46 146 | 246  47 147 247 |  48 148 248  49 | 149 249  50 150 | 250  51 151 251 |  52 152 252  53 | 153 253  54 154 | 254  55 155 255 |  56 156   0  57 | 157   1  58 158 |   2  59 159   3 |  60 160   4  61 | 161   5  62 162 |   6  63 163   7
	__m512i a0 = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*)(ptr + 0)));
	__m512i a1 = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*)(ptr + 32)));
	__m512i a2 = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*)(ptr + 64)));
	a0 = _mm512_inserti32x8(a0, _mm256_load_si256((const __m256i*)(ptr + 96)), 1);
	a1 = _mm512_inserti32x8(a1, _mm256_load_si256((const __m256i*)(ptr + 128)), 1);
	a2 = _mm512_inserti32x8(a2, _mm256_load_si256((const __m256i*)(ptr + 160)), 1);

	//static const __m512i idx0 = _mm512_setr_epi32(4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 24, 25, 26, 27);
	static const __m512i idx0 = _mm512_setr_epi64(21474836484, 30064771078, 73014444048, 81604378642, 55834574860, 64424509454, 107374182424, 115964117018);
	g = _mm512_permutex2var_epi32(a0, idx0, a2);
	b = _mm512_mask_blend_epi32(0xf0f0, a0, a1);
	r = _mm512_mask_blend_epi32(0xf0f0, a1, a2);

	a0 = _mm512_mask_blend_epi32(0xcccc, b, g);
	a1 = _mm512_shuffle2_epi32(b, r, 78);
	a2 = _mm512_mask_blend_epi32(0xcccc, g, r);

	//static const __m512i idx1 = _mm512_setr_epi32(1, 16, 3, 18, 5, 20, 7, 22, 9, 24, 11, 26, 13, 28, 15, 30);
	static const __m512i idx1 = _mm512_setr_epi64(68719476737, 77309411331, 85899345925, 94489280519, 103079215113, 111669149707, 120259084301, 128849018895);
	b = _mm512_mask_blend_epi32(0xaaaa, a0, a1);
	g = _mm512_permutex2var_epi32(a0, idx1, a2);
	r = _mm512_mask_blend_epi32(0xaaaa, a1, a2);

	//static const __m512i idx3 = _mm512_setr_epi16(1, 32, 3, 34, 5, 36, 7, 38, 9, 40, 11, 42, 13, 44, 15, 46, 17, 48, 19, 50, 21, 52, 23, 54, 25, 56, 27, 58, 29, 60, 31, 62	);
	static const __m512i idx3 = _mm512_setr_epi64(9570162095161345, 10696079182135301, 11821996269109257, 12947913356083213, 14073830443057169, 15199747530031125, 16325664617005081, 17451581703979037);
	a1 = _mm512_permutex2var_epi16(b, idx3, r);
	a0 = _mm512_mask_blend_epi16(0xaaaaaaaa, b, g);
	a2 = _mm512_mask_blend_epi16(0xaaaaaaaa, g, r);

#ifdef AVX512_VBMI
	//static const __m512i idx4 = _mm512_setr_epi8(1, 64, 3, 66, 5, 68, 7, 70, 9, 72, 11, 74, 13, 76, 15, 78, 17, 80, 19, 82, 21, 84, 23, 86, 25, 88, 27, 90, 29, 92, 31, 94, 33, 96, 35, 98, 37, 100, 39, 102, 41, 104, 43, 106, 45, 108, 47, 110, 49, 112, 51, 114, 53, 116, 55, 118, 57, 120, 59, 122, 61, 124, 63, 126);
	static const __m512i idx4 = _mm512_setr_epi64(5046076696864964609, 5624798079569577993, 6203519462274191377, 6782240844978804761, 7360962227683418145, 7939683610388031529, 8518404993092644913, 9097126375797258297);
	g = _mm512_permutex2var_epi8(a0, idx4, a2);
#else
	//static const __m512i mask = _mm512_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
	__m128i sfl = _mm_setr_epi64x(434320308619640833, 1013041691324254217);
	static const __m512i mask = _mm512_setr_m128i(sfl, sfl, sfl, sfl);
	g = _mm512_shuffle_epi8(_mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, a2, a0), mask);
#endif
	b = _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, a0, a1);
	r = _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, a1, a2);

}

STATIC_INLINE void _mm512_loadu_cvtepu8_bgr2planar_si512(const uchar* ptr, __m512i& b, __m512i& g, __m512i& r)
{
	//b:    0 100 200   1 | 101 201   2 102 | 202   3 103 203 |   4 104 204   5 | 105 205   6 106 | 206   7 107 207 |   8 108 208   9 | 109 209  10 110 | 210  11 111 211 |  12 112 212  13 | 113 213  14 114 | 214  15 115 215 |  16 116 216  17 | 117 217  18 118 | 218  19 119 219 |  20 120 220  21
	//g : 121 221  22 122 | 222  23 123 223 |  24 124 224  25 | 125 225  26 126 | 226  27 127 227 |  28 128 228  29 | 129 229  30 130 | 230  31 131 231 |  32 132 232  33 | 133 233  34 134 | 234  35 135 235 |  36 136 236  37 | 137 237  38 138 | 238  39 139 239 |  40 140 240  41 | 141 241  42 142
	//r : 242  43 143 243 |  44 144 244  45 | 145 245  46 146 | 246  47 147 247 |  48 148 248  49 | 149 249  50 150 | 250  51 151 251 |  52 152 252  53 | 153 253  54 154 | 254  55 155 255 |  56 156   0  57 | 157   1  58 158 |   2  59 159   3 |  60 160   4  61 | 161   5  62 162 |   6  63 163   7
	__m512i a0 = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr + 0)));
	__m512i a1 = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr + 32)));
	__m512i a2 = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr + 64)));
	a0 = _mm512_inserti32x8(a0, _mm256_loadu_si256((const __m256i*)(ptr + 96)), 1);
	a1 = _mm512_inserti32x8(a1, _mm256_loadu_si256((const __m256i*)(ptr + 128)), 1);
	a2 = _mm512_inserti32x8(a2, _mm256_loadu_si256((const __m256i*)(ptr + 160)), 1);

	//static const __m512i idx0 = _mm512_setr_epi32(4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 24, 25, 26, 27);
	static const __m512i idx0 = _mm512_setr_epi64(21474836484, 30064771078, 73014444048, 81604378642, 55834574860, 64424509454, 107374182424, 115964117018);
	g = _mm512_permutex2var_epi32(a0, idx0, a2);
	b = _mm512_mask_blend_epi32(0xf0f0, a0, a1);
	r = _mm512_mask_blend_epi32(0xf0f0, a1, a2);

	a0 = _mm512_mask_blend_epi32(0xcccc, b, g);
	a1 = _mm512_shuffle2_epi32(b, r, 78);
	a2 = _mm512_mask_blend_epi32(0xcccc, g, r);

	//static const __m512i idx1 = _mm512_setr_epi32(1, 16, 3, 18, 5, 20, 7, 22, 9, 24, 11, 26, 13, 28, 15, 30);
	static const __m512i idx1 = _mm512_setr_epi64(68719476737, 77309411331, 85899345925, 94489280519, 103079215113, 111669149707, 120259084301, 128849018895);
	b = _mm512_mask_blend_epi32(0xaaaa, a0, a1);
	g = _mm512_permutex2var_epi32(a0, idx1, a2);
	r = _mm512_mask_blend_epi32(0xaaaa, a1, a2);

	//static const __m512i idx3 = _mm512_setr_epi16(1, 32, 3, 34, 5, 36, 7, 38, 9, 40, 11, 42, 13, 44, 15, 46, 17, 48, 19, 50, 21, 52, 23, 54, 25, 56, 27, 58, 29, 60, 31, 62	);
	static const __m512i idx3 = _mm512_setr_epi64(9570162095161345, 10696079182135301, 11821996269109257, 12947913356083213, 14073830443057169, 15199747530031125, 16325664617005081, 17451581703979037);
	a1 = _mm512_permutex2var_epi16(b, idx3, r);
	a0 = _mm512_mask_blend_epi16(0xaaaaaaaa, b, g);
	a2 = _mm512_mask_blend_epi16(0xaaaaaaaa, g, r);

#ifdef AVX512_VBMI
	//static const __m512i idx4 = _mm512_setr_epi8(1, 64, 3, 66, 5, 68, 7, 70, 9, 72, 11, 74, 13, 76, 15, 78, 17, 80, 19, 82, 21, 84, 23, 86, 25, 88, 27, 90, 29, 92, 31, 94, 33, 96, 35, 98, 37, 100, 39, 102, 41, 104, 43, 106, 45, 108, 47, 110, 49, 112, 51, 114, 53, 116, 55, 118, 57, 120, 59, 122, 61, 124, 63, 126);
	static const __m512i idx4 = _mm512_setr_epi64(5046076696864964609, 5624798079569577993, 6203519462274191377, 6782240844978804761, 7360962227683418145, 7939683610388031529, 8518404993092644913, 9097126375797258297);
	g = _mm512_permutex2var_epi8(a0, idx4, a2);
#else
	//static const __m512i mask = _mm512_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
	__m128i sfl = _mm_setr_epi64x(434320308619640833, 1013041691324254217);
	static const __m512i mask = _mm512_setr_m128i(sfl, sfl, sfl, sfl);
	g = _mm512_shuffle_epi8(_mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, a2, a0), mask);
#endif
	b = _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, a0, a1);
	r = _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, a1, a2);

}

STATIC_INLINE void _mm512_loadu_cvtph_bgr2planar_ph2(const short* ptr, __m512h& b, __m512h& g, __m512h& r, const __m512i idx0, const __m512i idx1, const __m512i idx2, const __m512i idx3)
{
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47 | 48 49 50 51 52 53 54 55 | 56 57 58 59 60 61 62 63
	//  64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79 | 80 81 82 83 84 85 86 87 | 88 89 90 91 92 93 94 95

	b = _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(ptr + 0)));
	g = _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(ptr + 16)));
	r = _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)(ptr + 32)));
	b = _mm512_inserti32x8(b, ((const __m256i*)ptr)[3], 1);
	g = _mm512_inserti32x8(g, ((const __m256i*)ptr)[4], 1);
	r = _mm512_inserti32x8(r, ((const __m256i*)ptr)[5], 1);
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 48 49 50 51 52 53 54 55 | 56 57 58 59 60 61 62 63
	//  16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31 | 64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47 | 80 81 82 83 84 85 86 87 | 88 89 90 91 92 93 94 95

	__m512i a1 = _mm512_permutex2var_ph(b, idx0, r);
	__m512i a0 = _mm512_mask_blend_epi16(0xff00ff00, b, g);
	__m512i a2 = _mm512_mask_blend_epi16(0xff00ff00, g, r);
	//   0  1  2  3  4  5  6  7 | 24 25 26 27 28 29 30 31 | 48 49 50 51 52 53 54 55 | 72 73 74 75 76 77 78 79
	//   8  9 10 11 12 13 14 15 | 32 33 34 35 36 37 38 39 | 56 57 58 59 60 61 62 63 | 80 81 82 83 84 85 86 87
	//  16 17 18 19 20 21 22 23 | 40 41 42 43 44 45 46 47 | 64 65 66 67 68 69 70 71 | 88 89 90 91 92 93 94 95

	g = _mm512_permutex2var_ph(a0, idx1, a2);
	b = _mm512_mask_blend_ph(0xf0f0f0f0, a0, a1);
	r = _mm512_mask_blend_ph(0xf0f0f0f0, a1, a2);
	//   0  1  2  3 12 13 14 15 | 24 25 26 27 36 37 38 39 | 48 49 50 51 60 61 62 63 | 72 73 74 75 84 85 86 87
	//   4  5  6  7 16 17 18 19 | 28 29 30 31 40 41 42 43 | 52 53 54 55 64 65 66 67 | 76 77 78 79 88 89 90 91
	//   8  9 10 11 20 21 22 23 | 32 33 34 35 44 45 46 47 | 56 57 58 59 68 69 70 71 | 80 81 82 83 92 93 94 95


	a1 = _mm512_permutex2var_ph(b, idx2, r);
	a0 = _mm512_mask_blend_epi16(0xcccccccc, b, g);
	a2 = _mm512_mask_blend_epi16(0xcccccccc, g, r);
	//   0  1  6  7 12 13 18 19 | 24 25 30 31 36 37 42 43 | 48 49 54 55 60 61 66 67 | 72 73 78 79 84 85 90 91
	//   2  3  8  9 14 15 20 21 | 26 27 32 33 38 39 44 45 | 50 51 56 57 62 63 68 69 | 74 75 80 81 86 87 92 93
	//   4  5 10 11 16 17 22 23 | 28 29 34 35 40 41 46 47 | 52 53 58 59 64 65 70 71 | 76 77 82 83 88 89 94 95

	g = _mm512_permutex2var_ph(a0, idx3, a2);
	b = _mm512_mask_blend_epi16(0xaaaaaaaa, a0, a1);
	r = _mm512_mask_blend_epi16(0xaaaaaaaa, a1, a2);
}

STATIC_INLINE void _mm512_load_cvtps_bgr2planar_ps(const float* ptr, __m512& b, __m512& g, __m512& r)
{
	//   0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
	//  16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31
	//  32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47

	__m512 a0 = _mm512_castps256_ps512(_mm256_load_ps(ptr + 0));
	__m512 a1 = _mm512_castps256_ps512(_mm256_load_ps(ptr + 8));
	__m512 a2 = _mm512_castps256_ps512(_mm256_load_ps(ptr + 16));
	a0 = _mm512_insertf32x8(a0, _mm256_load_ps(ptr + 24), 1);
	a1 = _mm512_insertf32x8(a1, _mm256_load_ps(ptr + 32), 1);
	a2 = _mm512_insertf32x8(a2, _mm256_load_ps(ptr + 40), 1);
	//   0  1  2  3 |  4  5  6  7 | 24 25 26 27 | 28 29 30 31
	//   8  9 10 11 | 12 13 14 15 | 32 33 34 35 | 36 37 38 39
	//  16 17 18 19 | 20 21 22 23 | 40 41 42 43 | 44 45 46 47

	//static const __m512i idx0 = _mm512_setr_epi32(4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 24, 25, 26, 27);
	static const __m512i idx0 = _mm512_setr_epi64(21474836484, 30064771078, 73014444048, 81604378642, 55834574860, 64424509454, 107374182424, 115964117018);
	g = _mm512_permutex2var_ps(a0, idx0, a2);
	b = _mm512_mask_blend_ps(0xf0f0, a0, a1);
	r = _mm512_mask_blend_ps(0xf0f0, a1, a2);
	//   0  1  2  3 | 12 13 14 15 | 24 25 26 27 | 36 37 38 39
	//   4  5  6  7 | 16 17 18 19 | 28 29 30 31 | 40 41 42 43
	//   8  9 10 11 | 20 21 22 23 | 32 33 34 35 | 44 45 46 47

	a0 = _mm512_mask_blend_ps(0xcccc, b, g);
	a1 = _mm512_shuffle_ps(b, r, 78);
	a2 = _mm512_mask_blend_ps(0xcccc, g, r);
	//   0  1  6  7 | 12 13 18 19 | 24 25 30 31 | 36 37 42 43
	//   2  3  8  9 | 14 15 20 21 | 26 27 32 33 | 38 39 44 45
	//   4  5 10 11 | 16 17 22 23 | 28 29 34 35 | 40 41 46 47

	//static const __m512i idx1 = _mm512_setr_epi32(1, 16, 3, 18, 5, 20, 7, 22, 9, 24, 11, 26, 13, 28, 15, 30);
	static const __m512i idx1 = _mm512_setr_epi64(68719476737, 77309411331, 85899345925, 94489280519, 103079215113, 111669149707, 120259084301, 128849018895);
	b = _mm512_mask_blend_ps(0xaaaa, a0, a1);
	g = _mm512_permutex2var_ps(a0, idx1, a2);
	r = _mm512_mask_blend_ps(0xaaaa, a1, a2);
	//   0  3  6  9 | 12 15 18 21 | 24  27 30 33 | 36 39 42 45
	//   1  4  7 10 | 13 16 19 22 | 25  28 31 34 | 37 40 43 46
	//   2  5  8 11 | 14 17 20 23 | 26  29 32 35 | 38 41 44 47
}

STATIC_INLINE void _mm512_loadu_cvtps_bgr2planar_ps(const float* ptr, __m512& b, __m512& g, __m512& r)
{
	//   0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
	//  16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31
	//  32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47

	__m512 a0 = _mm512_castps256_ps512(_mm256_loadu_ps(ptr + 0));
	__m512 a1 = _mm512_castps256_ps512(_mm256_loadu_ps(ptr + 8));
	__m512 a2 = _mm512_castps256_ps512(_mm256_loadu_ps(ptr + 16));
	a0 = _mm512_insertf32x8(a0, ((const __m256*)ptr)[3], 1);
	a1 = _mm512_insertf32x8(a1, ((const __m256*)ptr)[4], 1);
	a2 = _mm512_insertf32x8(a2, ((const __m256*)ptr)[5], 1);
	//   0  1  2  3 |  4  5  6  7 | 24 25 26 27 | 28 29 30 31
	//   8  9 10 11 | 12 13 14 15 | 32 33 34 35 | 36 37 38 39
	//  16 17 18 19 | 20 21 22 23 | 40 41 42 43 | 44 45 46 47

	//static const __m512i idx0 = _mm512_setr_epi32(4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 24, 25, 26, 27);
	static const __m512i idx0 = _mm512_setr_epi64(21474836484, 30064771078, 73014444048, 81604378642, 55834574860, 64424509454, 107374182424, 115964117018);
	g = _mm512_permutex2var_ps(a0, idx0, a2);
	b = _mm512_mask_blend_ps(0xf0f0, a0, a1);
	r = _mm512_mask_blend_ps(0xf0f0, a1, a2);
	//   0  1  2  3 | 12 13 14 15 | 24 25 26 27 | 36 37 38 39
	//   4  5  6  7 | 16 17 18 19 | 28 29 30 31 | 40 41 42 43
	//   8  9 10 11 | 20 21 22 23 | 32 33 34 35 | 44 45 46 47

	a0 = _mm512_mask_blend_ps(0xcccc, b, g);
	a1 = _mm512_shuffle_ps(b, r, 78);
	a2 = _mm512_mask_blend_ps(0xcccc, g, r);
	//   0  1  6  7 | 12 13 18 19 | 24 25 30 31 | 36 37 42 43
	//   2  3  8  9 | 14 15 20 21 | 26 27 32 33 | 38 39 44 45
	//   4  5 10 11 | 16 17 22 23 | 28 29 34 35 | 40 41 46 47

	//static const __m512i idx1 = _mm512_setr_epi32(1, 16, 3, 18, 5, 20, 7, 22, 9, 24, 11, 26, 13, 28, 15, 30);
	static const __m512i idx1 = _mm512_setr_epi64(68719476737, 77309411331, 85899345925, 94489280519, 103079215113, 111669149707, 120259084301, 128849018895);
	b = _mm512_mask_blend_ps(0xaaaa, a0, a1);
	g = _mm512_permutex2var_ps(a0, idx1, a2);
	r = _mm512_mask_blend_ps(0xaaaa, a1, a2);
	//   0  3  6  9 | 12 15 18 21 | 24  27 30 33 | 36 39 42 45
	//   1  4  7 10 | 13 16 19 22 | 25  28 31 34 | 37 40 43 46
	//   2  5  8 11 | 14 17 20 23 | 26  29 32 35 | 38 41 44 47
}


STATIC_INLINE __m512 _mm512_i32gather_auto(const float* src, __m512i idx)
{
	return _mm512_i32gather_ps(idx, src, 4);
}

STATIC_INLINE __m512 _mm512_i32gather_auto(const uchar* src, __m512i idx)
{
	return _mm512_cvtepi32_ps(_mm512_srli_epi32(_mm512_i32gather_epi32(idx, reinterpret_cast<const int*>(&src[-3]), 1), 24));
}
#pragma endregion

#pragma region store
STATIC_INLINE void _mm512_store_cvtps_epu8(__m128i* dest, __m512 ms)
{
	_mm_store_si128(dest, _mm512_cvtps_epu8(ms));
}

STATIC_INLINE void _mm512_storeu_cvtps_epu8(__m128i* dest, __m512 ms)
{
	_mm_storeu_si128(dest, _mm512_cvtps_epu8(ms));
}

STATIC_INLINE void _mm512_store_auto(uchar* dest, __m512 src)
{
	_mm512_store_cvtps_epu8((__m128i*)dest, src);
}

STATIC_INLINE void _mm512_store_auto(float* dest, __m512 src)
{
	_mm512_store_ps(dest, src);
}

STATIC_INLINE void _mm512_store_auto(double* dest, __m512d src)
{
	_mm512_store_pd(dest, src);
}

STATIC_INLINE void _mm512_storeu_auto(uchar* dest, __m512 src)
{
	_mm512_storeu_cvtps_epu8((__m128i*)dest, src);
}

STATIC_INLINE void _mm512_storeu_auto(float* dest, __m512 src)
{
	_mm512_storeu_ps(dest, src);
}

STATIC_INLINE void _mm512_storeu_auto(double* dest, __m512d src)
{
	_mm512_storeu_pd(dest, src);
}

STATIC_INLINE void _mm512_stream_auto(uchar* dest, __m512 ms)
{
	_mm512_store_cvtps_epu8((__m128i*)dest, ms);
}

STATIC_INLINE void _mm512_stream_auto(float* dest, __m512 ms)
{
	_mm512_stream_ps(dest, ms);
}

STATIC_INLINE void _mm512_storescalar_cvtps_epu8(void* dst, __m512 src, const int numpixel)
{
	uchar CV_DECL_ALIGNED(64) buffscalarstore[64];
	_mm512_store_cvtps_epu8((__m128i*)buffscalarstore, src);
	uchar* dest = (uchar*)dst;
	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm512_storescalar_pd(uchar* dst, __m512d src, const int numpixel)
{
	double CV_DECL_ALIGNED(64) buffscalarstore[8];
	_mm512_store_pd(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = cv::saturate_cast<uchar>(buffscalarstore[i]);
}

STATIC_INLINE void _mm512_storescalar_pd(float* dst, __m512d src, const int numpixel)
{
	double CV_DECL_ALIGNED(64) buffscalarstore[8];
	_mm512_store_pd(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = (float)buffscalarstore[i];
}

STATIC_INLINE void _mm512_storescalar_pd(double* dst, __m512d src, const int numpixel)
{
	double CV_DECL_ALIGNED(64) buffscalarstore[8];
	_mm512_store_pd(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm512_storescalar_ps(float* dst, __m512 src, const int numpixel)
{
	float CV_DECL_ALIGNED(64) buffscalarstore[16];
	_mm512_store_ps(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm512_storescalar_auto(uchar* dest, __m512 ms, const int numpixel)
{
	_mm512_storescalar_cvtps_epu8(dest, ms, numpixel);
}

STATIC_INLINE void _mm512_storescalar_auto(float* dest, __m512 ms, const int numpixel)
{
	_mm512_storescalar_ps(dest, ms, numpixel);
}

STATIC_INLINE void _mm512_storescalar_auto(uchar* dest, __m512d ms, const int numpixel)
{
	_mm512_storescalar_pd(dest, ms, numpixel);
}

STATIC_INLINE void _mm512_storescalar_auto(float* dest, __m512d ms, const int numpixel)
{
	_mm512_storescalar_pd(dest, ms, numpixel);
}

STATIC_INLINE void _mm512_storescalar_auto(double* dest, __m512d ms, const int numpixel)
{
	_mm512_storescalar_pd(dest, ms, numpixel);
}

STATIC_INLINE void _mm512_store_epi8_color(void* dst, const __m512i b, const __m512i g, const __m512i r)
{
	__m512i dr, dg, db;
	_mm512_cvtsoa2aos_epi8(b, g, r, db, dg, dr);
	_mm512_store_si512((uchar*)dst + 0, db);
	_mm512_store_si512((uchar*)dst + 64, dg);
	_mm512_store_si512((uchar*)dst + 128, dr);
}

STATIC_INLINE void _mm512_storeu_epi8_color(void* dst, const __m512i b, const __m512i g, const __m512i r)
{
	__m512i dr, dg, db;
	_mm512_cvtsoa2aos_epi8(b, g, r, db, dg, dr);
	_mm512_storeu_si512((uchar*)dst + 0, db);
	_mm512_storeu_si512((uchar*)dst + 64, dg);
	_mm512_storeu_si512((uchar*)dst + 128, dr);
}

STATIC_INLINE void _mm512_stream_epi8_color(void* dst, const __m512i b, const __m512i g, const __m512i r)
{
	__m512i dr, dg, db;
	_mm512_cvtsoa2aos_epi8(b, g, r, db, dg, dr);
	_mm512_stream_si512((uchar*)dst + 0, db);
	_mm512_stream_si512((uchar*)dst + 64, dg);
	_mm512_stream_si512((uchar*)dst + 128, dr);
}

STATIC_INLINE void _mm512_store_ps_color(void* dst, const __m512 b, const __m512 g, const __m512 r)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	_mm512_store_ps((float*)dst + 0, db);
	_mm512_store_ps((float*)dst + 16, dg);
	_mm512_store_ps((float*)dst + 32, dr);
}

STATIC_INLINE void _mm512_storeu_ps_color(void* dst, const __m512 b, const __m512 g, const __m512 r)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	_mm512_storeu_ps((float*)dst + 0, db);
	_mm512_storeu_ps((float*)dst + 16, dg);
	_mm512_storeu_ps((float*)dst + 32, dr);
}

STATIC_INLINE void _mm512_stream_ps_color(void* dst, const __m512 b, const __m512 g, const __m512 r)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	_mm512_stream_ps((float*)dst + 0, db);
	_mm512_stream_ps((float*)dst + 16, dg);
	_mm512_stream_ps((float*)dst + 32, dr);
}

STATIC_INLINE void _mm512_store_pd_color(void* dst, const __m512d b, const __m512d g, const __m512d r)
{
	__m512d db, dg, dr;
	_mm512_cvtsoa2aos_pd(b, g, r, db, dg, dr);

	_mm512_store_pd((double*)dst + 0, db);
	_mm512_store_pd((double*)dst + 8, dg);
	_mm512_store_pd((double*)dst + 16, dr);
}

STATIC_INLINE void _mm512_storeu_pd_color(void* dst, const __m512d b, const __m512d g, const __m512d r)
{
	__m512d db, dg, dr;
	_mm512_cvtsoa2aos_pd(b, g, r, db, dg, dr);

	_mm512_storeu_pd((double*)dst + 0, db);
	_mm512_storeu_pd((double*)dst + 8, dg);
	_mm512_storeu_pd((double*)dst + 16, dr);
}

STATIC_INLINE void _mm512_stream_pd_color(void* dst, const __m512d b, const __m512d g, const __m512d r)
{
	__m512d db, dg, dr;
	_mm512_cvtsoa2aos_pd(b, g, r, db, dg, dr);

	_mm512_stream_pd((double*)dst + 0, db);
	_mm512_stream_pd((double*)dst + 8, dg);
	_mm512_stream_pd((double*)dst + 16, dr);
}

STATIC_INLINE void _mm512_storeu_auto_color(float* dest, __m512 b, __m512 g, __m512 r)
{
	_mm512_storeu_ps_color(dest, b, g, r);
}

STATIC_INLINE void _mm512_store_auto_color(float* dest, __m512 b, __m512 g, __m512 r)
{
	_mm512_store_ps_color(dest, b, g, r);
}

STATIC_INLINE void _mm512_store_auto_color(uchar* dest, __m512i b, __m512i g, __m512i r)
{
	_mm512_store_epi8_color(dest, b, g, r);
}

STATIC_INLINE void _mm512_storeu_auto_color(uchar* dest, __m512i b, __m512i g, __m512i r)
{
	_mm512_storeu_epi8_color(dest, b, g, r);
}

STATIC_INLINE void _mm512_storeu_auto_color(uchar* dest, __m512 b, __m512 g, __m512 r)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	_mm_storeu_si128((__m128i*)(dest), _mm512_cvtps_epu8(db));
	_mm_storeu_si128((__m128i*)(dest + 8), _mm512_cvtps_epu8(dg));
	_mm_storeu_si128((__m128i*)(dest + 16), _mm512_cvtps_epu8(dr));
}

STATIC_INLINE void _mm512_store_auto_color(uchar* dest, __m512 b, __m512 g, __m512 r)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	_mm_store_si128((__m128i*)(dest), _mm512_cvtps_epu8(db));
	_mm_store_si128((__m128i*)(dest + 8), _mm512_cvtps_epu8(dg));
	_mm_store_si128((__m128i*)(dest + 16), _mm512_cvtps_epu8(dr));
}

STATIC_INLINE void _mm512_stream_auto_color(float* dest, __m512 b, __m512 g, __m512 r)
{
	_mm512_stream_ps_color(dest, b, g, r);
}

STATIC_INLINE void _mm512_stream_auto_color(uchar* dest, __m512 b, __m512 g, __m512 r)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	_mm_stream_si128((__m128i*)(dest), _mm512_cvtps_epu8(db));
	_mm_stream_si128((__m128i*)(dest + 8), _mm512_cvtps_epu8(dg));
	_mm_stream_si128((__m128i*)(dest + 16), _mm512_cvtps_epu8(dr));
}

STATIC_INLINE void _mm512_storescalar_auto_color(uchar* dest, __m512 b, __m512 g, __m512 r, const int numpixel)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	uchar CV_DECL_ALIGNED(64) buffscalarstore[48];

	_mm_store_si128((__m128i*)(buffscalarstore + 0), _mm512_cvtps_epu8(db));
	_mm_store_si128((__m128i*)(buffscalarstore + 16), _mm512_cvtps_epu8(dg));
	_mm_store_si128((__m128i*)(buffscalarstore + 32), _mm512_cvtps_epu8(dr));
	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}

STATIC_INLINE void _mm512_storescalar_auto_color(float* dest, __m512 b, __m512 g, __m512 r, const int numpixel)
{
	__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	float CV_DECL_ALIGNED(64) buffscalarstore[48];
	_mm512_store_ps(buffscalarstore + 0, db);
	_mm512_store_ps(buffscalarstore + 16, dg);
	_mm512_store_ps(buffscalarstore + 32, dr);

	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}


STATIC_INLINE void _mm512_i32scaterscalar_epu8(uchar* dest, __m512i vindex, __m512 src)
{
	__m128i v = _mm512_cvtps_epu8(src);
	dest[((int*)&vindex)[0]] = ((uchar*)&v)[0];
	dest[((int*)&vindex)[1]] = ((uchar*)&v)[1];
	dest[((int*)&vindex)[2]] = ((uchar*)&v)[2];
	dest[((int*)&vindex)[3]] = ((uchar*)&v)[3];
	dest[((int*)&vindex)[4]] = ((uchar*)&v)[4];
	dest[((int*)&vindex)[5]] = ((uchar*)&v)[5];
	dest[((int*)&vindex)[6]] = ((uchar*)&v)[6];
	dest[((int*)&vindex)[7]] = ((uchar*)&v)[7];
	dest[((int*)&vindex)[8]] = ((uchar*)&v)[8];
	dest[((int*)&vindex)[9]] = ((uchar*)&v)[9];
	dest[((int*)&vindex)[10]] = ((uchar*)&v)[10];
	dest[((int*)&vindex)[11]] = ((uchar*)&v)[11];
	dest[((int*)&vindex)[12]] = ((uchar*)&v)[12];
	dest[((int*)&vindex)[13]] = ((uchar*)&v)[13];
	dest[((int*)&vindex)[14]] = ((uchar*)&v)[14];
	dest[((int*)&vindex)[15]] = ((uchar*)&v)[15];
}

STATIC_INLINE void _mm512_i32scaterscalar_auto(uchar* dest, __m512i vindex, __m512 src)
{
	_mm512_i32scaterscalar_epu8(dest, vindex, src);
}

STATIC_INLINE void _mm512_i32scaterscalar_auto(float* dest, __m512i vindex, __m512 src)
{
	_mm512_i32scatter_ps(dest, vindex, src, sizeof(float));
}

STATIC_INLINE void _mm512_i32scaterscalar_epu8_color(uchar* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	__m128i bb = _mm512_cvtps_epu8(b);
	__m128i gb = _mm512_cvtps_epu8(g);
	__m128i rb = _mm512_cvtps_epu8(r);
	int idx = ((int*)&vindex)[0];
	dest[idx + 0] = ((uchar*)&bb)[0];	dest[idx + 1] = ((uchar*)&gb)[0];	dest[idx + 2] = ((uchar*)&rb)[0];
	idx = ((int*)&vindex)[1];
	dest[idx + 0] = ((uchar*)&bb)[1];	dest[idx + 1] = ((uchar*)&gb)[1];	dest[idx + 2] = ((uchar*)&rb)[1];
	idx = ((int*)&vindex)[2];
	dest[idx + 0] = ((uchar*)&bb)[2];	dest[idx + 1] = ((uchar*)&gb)[2];	dest[idx + 2] = ((uchar*)&rb)[2];
	idx = ((int*)&vindex)[3];
	dest[idx + 0] = ((uchar*)&bb)[3];	dest[idx + 1] = ((uchar*)&gb)[3];	dest[idx + 2] = ((uchar*)&rb)[3];
	idx = ((int*)&vindex)[4];
	dest[idx + 0] = ((uchar*)&bb)[4];	dest[idx + 1] = ((uchar*)&gb)[4];	dest[idx + 2] = ((uchar*)&rb)[4];
	idx = ((int*)&vindex)[5];
	dest[idx + 0] = ((uchar*)&bb)[5];	dest[idx + 1] = ((uchar*)&gb)[5];	dest[idx + 2] = ((uchar*)&rb)[5];
	idx = ((int*)&vindex)[6];
	dest[idx + 0] = ((uchar*)&bb)[6];	dest[idx + 1] = ((uchar*)&gb)[6];	dest[idx + 2] = ((uchar*)&rb)[6];
	idx = ((int*)&vindex)[7];
	dest[idx + 0] = ((uchar*)&bb)[7];	dest[idx + 1] = ((uchar*)&gb)[7];	dest[idx + 2] = ((uchar*)&rb)[7];

	idx = ((int*)&vindex)[8];
	dest[idx + 0] = ((uchar*)&bb)[8];	dest[idx + 1] = ((uchar*)&gb)[8];	dest[idx + 2] = ((uchar*)&rb)[8];
	idx = ((int*)&vindex)[9];
	dest[idx + 0] = ((uchar*)&bb)[9];	dest[idx + 1] = ((uchar*)&gb)[9];	dest[idx + 2] = ((uchar*)&rb)[9];
	idx = ((int*)&vindex)[10];
	dest[idx + 0] = ((uchar*)&bb)[10];	dest[idx + 1] = ((uchar*)&gb)[10];	dest[idx + 2] = ((uchar*)&rb)[10];
	idx = ((int*)&vindex)[11];
	dest[idx + 0] = ((uchar*)&bb)[11];	dest[idx + 1] = ((uchar*)&gb)[11];	dest[idx + 2] = ((uchar*)&rb)[11];
	idx = ((int*)&vindex)[12];
	dest[idx + 0] = ((uchar*)&bb)[12];	dest[idx + 1] = ((uchar*)&gb)[12];	dest[idx + 2] = ((uchar*)&rb)[12];
	idx = ((int*)&vindex)[13];
	dest[idx + 0] = ((uchar*)&bb)[13];	dest[idx + 1] = ((uchar*)&gb)[13];	dest[idx + 2] = ((uchar*)&rb)[13];
	idx = ((int*)&vindex)[14];
	dest[idx + 0] = ((uchar*)&bb)[14];	dest[idx + 1] = ((uchar*)&gb)[14];	dest[idx + 2] = ((uchar*)&rb)[14];
	idx = ((int*)&vindex)[15];
	dest[idx + 0] = ((uchar*)&bb)[15];	dest[idx + 1] = ((uchar*)&gb)[15];	dest[idx + 2] = ((uchar*)&rb)[15];
}

STATIC_INLINE void _mm512_i32scater_ps_color(float* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	_mm512_i32scatter_ps(dest + 0, vindex, b, sizeof(float));
	_mm512_i32scatter_ps(dest + 1, vindex, g, sizeof(float));
	_mm512_i32scatter_ps(dest + 2, vindex, r, sizeof(float));
}

STATIC_INLINE void _mm512_i32scaterscalar_ps_color(float* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	_mm512_i32scatter_ps(dest + 0, vindex, b, sizeof(float));
	_mm512_i32scatter_ps(dest + 1, vindex, g, sizeof(float));
	_mm512_i32scatter_ps(dest + 2, vindex, r, sizeof(float));
	return;
	int idx = ((int*)&vindex)[0];
	dest[idx + 0] = ((float*)&b)[0];	dest[idx + 1] = ((float*)&g)[0];	dest[idx + 2] = ((float*)&r)[0];
	idx = ((int*)&vindex)[1];
	dest[idx + 0] = ((float*)&b)[1];	dest[idx + 1] = ((float*)&g)[1];	dest[idx + 2] = ((float*)&r)[1];
	idx = ((int*)&vindex)[2];
	dest[idx + 0] = ((float*)&b)[2];	dest[idx + 1] = ((float*)&g)[2];	dest[idx + 2] = ((float*)&r)[2];
	idx = ((int*)&vindex)[3];
	dest[idx + 0] = ((float*)&b)[3];	dest[idx + 1] = ((float*)&g)[3];	dest[idx + 2] = ((float*)&r)[3];
	idx = ((int*)&vindex)[4];
	dest[idx + 0] = ((float*)&b)[4];	dest[idx + 1] = ((float*)&g)[4];	dest[idx + 2] = ((float*)&r)[4];
	idx = ((int*)&vindex)[5];
	dest[idx + 0] = ((float*)&b)[5];	dest[idx + 1] = ((float*)&g)[5];	dest[idx + 2] = ((float*)&r)[5];
	idx = ((int*)&vindex)[6];
	dest[idx + 0] = ((float*)&b)[6];	dest[idx + 1] = ((float*)&g)[6];	dest[idx + 2] = ((float*)&r)[6];
	idx = ((int*)&vindex)[7];
	dest[idx + 0] = ((float*)&b)[7];	dest[idx + 1] = ((float*)&g)[7];	dest[idx + 2] = ((float*)&r)[7];

	idx = ((int*)&vindex)[8];
	dest[idx + 0] = ((float*)&b)[8];	dest[idx + 1] = ((float*)&g)[8];	dest[idx + 2] = ((float*)&r)[8];
	idx = ((int*)&vindex)[9];
	dest[idx + 0] = ((float*)&b)[9];	dest[idx + 1] = ((float*)&g)[9];	dest[idx + 2] = ((float*)&r)[9];
	idx = ((int*)&vindex)[10];
	dest[idx + 0] = ((float*)&b)[10];	dest[idx + 1] = ((float*)&g)[10];	dest[idx + 2] = ((float*)&r)[10];
	idx = ((int*)&vindex)[11];
	dest[idx + 0] = ((float*)&b)[11];	dest[idx + 1] = ((float*)&g)[11];	dest[idx + 2] = ((float*)&r)[11];
	idx = ((int*)&vindex)[12];
	dest[idx + 0] = ((float*)&b)[12];	dest[idx + 1] = ((float*)&g)[12];	dest[idx + 2] = ((float*)&r)[12];
	idx = ((int*)&vindex)[13];
	dest[idx + 0] = ((float*)&b)[13];	dest[idx + 1] = ((float*)&g)[13];	dest[idx + 2] = ((float*)&r)[13];
	idx = ((int*)&vindex)[14];
	dest[idx + 0] = ((float*)&b)[14];	dest[idx + 1] = ((float*)&g)[14];	dest[idx + 2] = ((float*)&r)[14];
	idx = ((int*)&vindex)[15];
	dest[idx + 0] = ((float*)&b)[15];	dest[idx + 1] = ((float*)&g)[15];	dest[idx + 2] = ((float*)&r)[15];
	/*idx = ((int*)&vindex)[8] + 3;
	dest[idx + 0] = ((float*)&b)[8];	dest[idx + 1] = ((float*)&g)[8];	dest[idx + 2] = ((float*)&r)[8];
	idx = ((int*)&vindex)[9] + 3;
	dest[idx + 0] = ((float*)&b)[9];	dest[idx + 1] = ((float*)&g)[9];	dest[idx + 2] = ((float*)&r)[9];
	idx = ((int*)&vindex)[10] + 3;
	dest[idx + 0] = ((float*)&b)[10];	dest[idx + 1] = ((float*)&g)[10];	dest[idx + 2] = ((float*)&r)[10];
	idx = ((int*)&vindex)[11] + 3;
	dest[idx + 0] = ((float*)&b)[11];	dest[idx + 1] = ((float*)&g)[11];	dest[idx + 2] = ((float*)&r)[11];
	idx = ((int*)&vindex)[12] + 3;
	dest[idx + 0] = ((float*)&b)[12];	dest[idx + 1] = ((float*)&g)[12];	dest[idx + 2] = ((float*)&r)[12];
	idx = ((int*)&vindex)[13] + 3;
	dest[idx + 0] = ((float*)&b)[13];	dest[idx + 1] = ((float*)&g)[13];	dest[idx + 2] = ((float*)&r)[13];
	idx = ((int*)&vindex)[14] + 3;
	dest[idx + 0] = ((float*)&b)[14];	dest[idx + 1] = ((float*)&g)[14];	dest[idx + 2] = ((float*)&r)[14];
	idx = ((int*)&vindex)[15] + 3;
	dest[idx + 0] = ((float*)&b)[15];	dest[idx + 1] = ((float*)&g)[15];	dest[idx + 2] = ((float*)&r)[15];*/
}

STATIC_INLINE void _mm512_i32scater_auto_color(uchar* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	_mm512_i32scaterscalar_epu8_color(dest, vindex, b, g, r);
}

STATIC_INLINE void _mm512_i32scater_auto_color(float* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	_mm512_i32scater_ps_color(dest, vindex, b, g, r);
}

STATIC_INLINE void _mm512_i32scaterscalar_auto_color(uchar* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	_mm512_i32scaterscalar_epu8_color(dest, vindex, b, g, r);
}

STATIC_INLINE void _mm512_i32scaterscalar_auto_color(float* dest, __m512i vindex, __m512 b, __m512 g, __m512 r)
{
	_mm512_i32scaterscalar_ps_color(dest, vindex, b, g, r);
}

#pragma endregion

#pragma region arithmetic
STATIC_INLINE __m512 _mm256_abs_ps(__m512 src)
{
	//return _mm256_and_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
	static __m512 mask = _mm512_set1_ps(-0.f); return _mm512_andnot_ps(mask, src);
	//return _mm512_andnot_ps(_mm512_set1_ps(-0.f), src);
}

STATIC_INLINE __m512 _mm512_absdiff_ps(__m512 src1, __m512 src2)
{
	return _mm512_abs_ps(_mm512_sub_ps(src1, src2));
}

STATIC_INLINE void _mm512_absdiff_psx2(const __m512 src0, const __m512 ref0, const __m512 src1, const __m512 ref1, __m512& d0, __m512& d1)
{
	d0 = _mm512_sub_ps(src0, ref0);
	d1 = _mm512_sub_ps(src1, ref1);
	d0 = _mm512_abs_ps(d0);
	d1 = _mm512_abs_ps(d1);
}

STATIC_INLINE void _mm512_absdiff_psx4(const __m512 src0, const __m512 ref0, const __m512 src1, const __m512 ref1, const __m512 src2, const __m512 ref2, const __m512 src3, const __m512 ref3, __m512& d0, __m512& d1, __m512& d2, __m512& d3)
{
	d0 = _mm512_sub_ps(src0, ref0);
	d1 = _mm512_sub_ps(src1, ref1);
	d2 = _mm512_sub_ps(src2, ref2);
	d3 = _mm512_sub_ps(src3, ref3);
	d0 = _mm512_abs_ps(d0);
	d1 = _mm512_abs_ps(d1);
	d2 = _mm512_abs_ps(d2);
	d3 = _mm512_abs_ps(d3);
}

STATIC_INLINE void _mm512_absdiff_psx8(const __m512 src0, const __m512 ref0, const __m512 src1, const __m512 ref1, const __m512 src2, const __m512 ref2, const __m512 src3, const __m512 ref3, const __m512 src4, const __m512 ref4, const __m512 src5, const __m512 ref5, const __m512 src6, const __m512 ref6, const __m512 src7, const __m512 ref7, __m512& d0, __m512& d1, __m512& d2, __m512& d3, __m512& d4, __m512& d5, __m512& d6, __m512& d7)
{
	d0 = _mm512_sub_ps(src0, ref0);
	d1 = _mm512_sub_ps(src1, ref1);
	d2 = _mm512_sub_ps(src2, ref2);
	d3 = _mm512_sub_ps(src3, ref3);
	d4 = _mm512_sub_ps(src4, ref4);
	d5 = _mm512_sub_ps(src5, ref5);
	d6 = _mm512_sub_ps(src6, ref6);
	d7 = _mm512_sub_ps(src7, ref7);
	d0 = _mm512_abs_ps(d0);
	d1 = _mm512_abs_ps(d1);
	d2 = _mm512_abs_ps(d2);
	d3 = _mm512_abs_ps(d3);
	d4 = _mm512_abs_ps(d4);
	d5 = _mm512_abs_ps(d5);
	d6 = _mm512_abs_ps(d6);
	d7 = _mm512_abs_ps(d7);
}

STATIC_INLINE __m512i _mm512_absdiff_epu8(__m512i src1, __m512i src2)
{
	return _mm512_max_epu8(_mm512_subs_epu8(src1, src2), _mm512_subs_epu8(src2, src1));
}

STATIC_INLINE __m512 _mm512_ssd_ps(__m512 src, __m512 ref)
{
	__m512 diff = _mm512_sub_ps(src, ref);
	return _mm512_mul_ps(diff, diff);
}

STATIC_INLINE __m512 _mm512_ssd_ps(__m512 src0, __m512 src1, __m512 src2, __m512 ref0, __m512 ref1, __m512 ref2)
{
	__m512 diff = _mm512_sub_ps(src0, ref0);
	__m512 difft = _mm512_mul_ps(diff, diff);
	diff = _mm512_sub_ps(src1, ref1);
	difft = _mm512_fmadd_ps(diff, diff, difft);
	diff = _mm512_sub_ps(src2, ref2);
	difft = _mm512_fmadd_ps(diff, diff, difft);
	return difft;
}

//rcp with newton-raphson 1-iteration
STATIC_INLINE __m512 _mm512_rcpnr_ps(__m512 x)
{
	__m512 res = _mm512_rcp14_ps(x);
	//rcp*(2-rcp*x)->(rcp+rcp)-rcp*rcp*x
	return res = _mm512_sub_ps(_mm512_add_ps(res, res), _mm512_mul_ps(x, _mm512_mul_ps(res, res)));
}

//rcp with newton-raphson 1-iteration (FMA ver) requided set2
STATIC_INLINE __m512 _mm512_rcpnr_fma_ps(__m512 x, __m512 two = _mm512_set1_ps(2.f))
{
	__m512 rcp = _mm512_rcp14_ps(x);
	//rcp*(2-rcp*x)
	return _mm512_mul_ps(rcp, _mm512_fnmadd_ps(x, rcp, two));
}
#pragma endregion

STATIC_INLINE void _mm512_transpose16_ps(__m512& s00, __m512& s01, __m512& s02, __m512& s03, __m512& s04, __m512& s05, __m512& s06, __m512& s07, __m512& s08, __m512& s09, __m512& s10, __m512& s11, __m512& s12, __m512& s13, __m512& s14, __m512& s15)
{
	__m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
	__m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

	r0 = _mm512_castps_si512(s00);
	r1 = _mm512_castps_si512(s01);
	r2 = _mm512_castps_si512(s02);
	r3 = _mm512_castps_si512(s03);
	r4 = _mm512_castps_si512(s04);
	r5 = _mm512_castps_si512(s05);
	r6 = _mm512_castps_si512(s06);
	r7 = _mm512_castps_si512(s07);
	r8 = _mm512_castps_si512(s08);
	r9 = _mm512_castps_si512(s09);
	ra = _mm512_castps_si512(s10);
	rb = _mm512_castps_si512(s11);
	rc = _mm512_castps_si512(s12);
	rd = _mm512_castps_si512(s13);
	re = _mm512_castps_si512(s14);
	rf = _mm512_castps_si512(s15);

	t0 = _mm512_unpacklo_epi32(r0, r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
	t1 = _mm512_unpackhi_epi32(r0, r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
	t2 = _mm512_unpacklo_epi32(r2, r3); //  32  48  33  49 ...
	t3 = _mm512_unpackhi_epi32(r2, r3); //  34  50  35  51 ...
	t4 = _mm512_unpacklo_epi32(r4, r5); //  64  80  65  81 ...  
	t5 = _mm512_unpackhi_epi32(r4, r5); //  66  82  67  83 ...
	t6 = _mm512_unpacklo_epi32(r6, r7); //  96 112  97 113 ...
	t7 = _mm512_unpackhi_epi32(r6, r7); //  98 114  99 115 ...
	t8 = _mm512_unpacklo_epi32(r8, r9); // 128 ...
	t9 = _mm512_unpackhi_epi32(r8, r9); // 130 ...
	ta = _mm512_unpacklo_epi32(ra, rb); // 160 ...
	tb = _mm512_unpackhi_epi32(ra, rb); // 162 ...
	tc = _mm512_unpacklo_epi32(rc, rd); // 196 ...
	td = _mm512_unpackhi_epi32(rc, rd); // 198 ...
	te = _mm512_unpacklo_epi32(re, rf); // 228 ...
	tf = _mm512_unpackhi_epi32(re, rf); // 230 ...

	r0 = _mm512_unpacklo_epi64(t0, t2); //   0  16  32  48 ...
	r1 = _mm512_unpackhi_epi64(t0, t2); //   1  17  33  49 ...
	r2 = _mm512_unpacklo_epi64(t1, t3); //   2  18  34  49 ...
	r3 = _mm512_unpackhi_epi64(t1, t3); //   3  19  35  51 ...
	r4 = _mm512_unpacklo_epi64(t4, t6); //  64  80  96 112 ...  
	r5 = _mm512_unpackhi_epi64(t4, t6); //  65  81  97 114 ...
	r6 = _mm512_unpacklo_epi64(t5, t7); //  66  82  98 113 ...
	r7 = _mm512_unpackhi_epi64(t5, t7); //  67  83  99 115 ...
	r8 = _mm512_unpacklo_epi64(t8, ta); // 128 144 160 176 ...  
	r9 = _mm512_unpackhi_epi64(t8, ta); // 129 145 161 178 ...
	ra = _mm512_unpacklo_epi64(t9, tb); // 130 146 162 177 ... 
	rb = _mm512_unpackhi_epi64(t9, tb); // 131 147 163 179 ...
	rc = _mm512_unpacklo_epi64(tc, te); // 192 208 228 240 ... 
	rd = _mm512_unpackhi_epi64(tc, te); // 193 209 229 241 ...
	re = _mm512_unpacklo_epi64(td, tf); // 194 210 230 242 ...
	rf = _mm512_unpackhi_epi64(td, tf); // 195 211 231 243 ...

	//_mm512_shuffle_f32x4
	t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
	t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
	t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
	t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
	t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
	t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
	t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
	t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
	t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
	t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
	ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
	tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
	tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
	td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
	te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
	tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

	s00 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t0, t8, 0x88)); //   0  16  32  48  64  80  96 112 ... 240
	s01 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t1, t9, 0x88)); //   1  17  33  49  66  81  97 113 ... 241
	s02 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t2, ta, 0x88)); //   2  18  34  50  67  82  98 114 ... 242
	s03 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t3, tb, 0x88)); //   3  19  35  51  68  83  99 115 ... 243
	s04 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t4, tc, 0x88)); //   4 ...
	s05 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t5, td, 0x88)); //   5 ...
	s06 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t6, te, 0x88)); //   6 ...
	s07 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t7, tf, 0x88)); //   7 ...
	s08 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t0, t8, 0xdd)); //   8 ...
	s09 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t1, t9, 0xdd)); //   9 ...
	s10 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t2, ta, 0xdd)); //  10 ...
	s11 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t3, tb, 0xdd)); //  11 ...
	s12 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t4, tc, 0xdd)); //  12 ...
	s13 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t5, td, 0xdd)); //  13 ...
	s14 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t6, te, 0xdd)); //  14 ...
	s15 = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t7, tf, 0xdd)); //  15  31  47  63  79  96 111 127 ... 255
}

//util
//_mm512_setr_ps(v, v + step, v + 2.f * step, v + 3.f * step, v + 4.f * step, v + 5.f * step, v + 6.f * step, v + 7.f * step, v + 8.f * step, v + 9.f * step, v + 10.f * step, v + 11.f * step, v + 12.f * step, v + 13.f * step, v + 14.f * step, v + 15.f * step);
STATIC_INLINE __m512 _mm512_set_step_ps(float v, float step = 1.f)
{
	return _mm512_setr_ps(v, v + step, v + 2.f * step, v + 3.f * step, v + 4.f * step, v + 5.f * step, v + 6.f * step, v + 7.f * step, v + 8.f * step, v + 9.f * step, v + 10.f * step, v + 11.f * step, v + 12.f * step, v + 13.f * step, v + 14.f * step, v + 15.f * step);
}

STATIC_INLINE __m512i _mm512_set_step_epi8(char v, char step = 1)
{
	return _mm512_setr_epi8(
		v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step, v + 8 * step, v + 9 * step, v + 10 * step, v + 11 * step, v + 12 * step, v + 13 * step, v + 14 * step, v + 15 * step,
		v + 16 * step, v + 17 * step, v + 18 * step, v + 19 * step, v + 20 * step, v + 21 * step, v + 22 * step, v + 23 * step, v + 24 * step, v + 25 * step, v + 26 * step, v + 27 * step, v + 28 * step, v + 29 * step, v + 30 * step, v + 31 * step,
		v + 32 * step, v + 33 * step, v + 34 * step, v + 35 * step, v + 36 * step, v + 37 * step, v + 38 * step, v + 39 * step, v + 40 * step, v + 41 * step, v + 42 * step, v + 43 * step, v + 44 * step, v + 45 * step, v + 46 * step, v + 47 * step,
		v + 48 * step, v + 49 * step, v + 50 * step, v + 51 * step, v + 52 * step, v + 53 * step, v + 54 * step, v + 55 * step, v + 56 * step, v + 57 * step, v + 58 * step, v + 59 * step, v + 60 * step, v + 61 * step, v + 62 * step, v + 63 * step
	);
}

STATIC_INLINE __m512i _mm512_set_step_epi16(short v, short step = 1)
{
	return _mm512_setr_epi16(
		v + 0 * step, v + 1 * step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step, v + 8 * step, v + 9 * step, v + 10 * step, v + 11 * step, v + 12 * step, v + 13 * step, v + 14 * step, v + 15 * step,
		v + 16 * step, v + 17 * step, v + 18 * step, v + 19 * step, v + 20 * step, v + 21 * step, v + 22 * step, v + 23 * step, v + 24 * step, v + 25 * step, v + 26 * step, v + 27 * step, v + 28 * step, v + 29 * step, v + 30 * step, v + 31 * step
	);
}

STATIC_INLINE __m512i _mm512_set_step_epi32(int v, int step = 1)
{
	return _mm512_setr_epi32(v, v + step, v + 2 * step, v + 3 * step, v + 4 * step, v + 5 * step, v + 6 * step, v + 7 * step, v + 8 * step, v + 9 * step, v + 10 * step, v + 11 * step, v + 12 * step, v + 13 * step, v + 14 * step, v + 15 * step);
}



#define print_m512(src) printf_s("%s: %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f\n",#src,((float*)&src)[0], ((float*)&src)[1], ((float*)&src)[2], ((float*)&src)[3], ((float*)&src)[4], ((float*)&src)[5], ((float*)&src)[6], ((float*)&src)[7], ((float*)&src)[8], ((float*)&src)[9], ((float*)&src)[10], ((float*)&src)[11], ((float*)&src)[12], ((float*)&src)[13], ((float*)&src)[14], ((float*)&src)[15]);
#define print_m512i_int(src) printf_s("%s: %d %d %d %d | %d %d %d %d | %d %d %d %d | %d %d %d %d\n",#src,((int*)&src)[0], ((int*)&src)[1], ((int*)&src)[2], ((int*)&src)[3], ((int*)&src)[4], ((int*)&src)[5], ((int*)&src)[6], ((int*)&src)[7], ((int*)&src)[8], ((int*)&src)[9], ((int*)&src)[10], ((int*)&src)[11], ((int*)&src)[12], ((int*)&src)[13], ((int*)&src)[14], ((int*)&src)[15]);
#define print_m512i_int_align4(src) printf_s("%s: %4d %4d %4d %4d | %4d %4d %4d %4d | %4d %4d %4d %4d | %4d %4d %4d %4d\n",#src,((int*)&src)[0], ((int*)&src)[1], ((int*)&src)[2], ((int*)&src)[3], ((int*)&src)[4], ((int*)&src)[5], ((int*)&src)[6], ((int*)&src)[7], ((int*)&src)[8], ((int*)&src)[9], ((int*)&src)[10], ((int*)&src)[11], ((int*)&src)[12], ((int*)&src)[13], ((int*)&src)[14], ((int*)&src)[15]);
#define print_m512i_short(src) printf_s("%s: %6d %6d %6d %6d | %6d %6d %6d %6d | %6d %6d %6d %6d | %6d %6d %6d %6d | %6d %6d %6d %6d | %6d %6d %6d %6d | %6d %6d %6d %6d | %6d %6d %6d %6d\n",#src,((short*)&src)[0], ((short*)&src)[1], ((short*)&src)[2], ((short*)&src)[3], ((short*)&src)[4], ((short*)&src)[5], ((short*)&src)[6], ((short*)&src)[7], ((short*)&src)[8], ((short*)&src)[9], ((short*)&src)[10], ((short*)&src)[11], ((short*)&src)[12], ((short*)&src)[13], ((short*)&src)[14], ((short*)&src)[15],((short*)&src)[16], ((short*)&src)[17], ((short*)&src)[18], ((short*)&src)[19], ((short*)&src)[20], ((short*)&src)[21], ((short*)&src)[22], ((short*)&src)[23], ((short*)&src)[24], ((short*)&src)[25], ((short*)&src)[26], ((short*)&src)[27], ((short*)&src)[28], ((short*)&src)[29], ((short*)&src)[30], ((short*)&src)[31]);
#define print_m512i_uchar(src) printf_s("%s: %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d\n",#src,((uchar*)&src)[0], ((uchar*)&src)[1], ((uchar*)&src)[2], ((uchar*)&src)[3], ((uchar*)&src)[4], ((uchar*)&src)[5], ((uchar*)&src)[6], ((uchar*)&src)[7], ((uchar*)&src)[8], ((uchar*)&src)[9], ((uchar*)&src)[10], ((uchar*)&src)[11], ((uchar*)&src)[12], ((uchar*)&src)[13], ((uchar*)&src)[14], ((uchar*)&src)[15],((uchar*)&src)[16], ((uchar*)&src)[17], ((uchar*)&src)[18], ((uchar*)&src)[19], ((uchar*)&src)[20], ((uchar*)&src)[21], ((uchar*)&src)[22], ((uchar*)&src)[23], ((uchar*)&src)[24], ((uchar*)&src)[25], ((uchar*)&src)[26], ((uchar*)&src)[27], ((uchar*)&src)[28], ((uchar*)&src)[29], ((uchar*)&src)[30], ((uchar*)&src)[31],((uchar*)&src)[32], ((uchar*)&src)[33], ((uchar*)&src)[34], ((uchar*)&src)[35], ((uchar*)&src)[36], ((uchar*)&src)[37], ((uchar*)&src)[38], ((uchar*)&src)[39], ((uchar*)&src)[40], ((uchar*)&src)[41], ((uchar*)&src)[42], ((uchar*)&src)[43], ((uchar*)&src)[44], ((uchar*)&src)[45], ((uchar*)&src)[46], ((uchar*)&src)[47],((uchar*)&src)[48], ((uchar*)&src)[49], ((uchar*)&src)[50], ((uchar*)&src)[51], ((uchar*)&src)[52], ((uchar*)&src)[53], ((uchar*)&src)[54], ((uchar*)&src)[55], ((uchar*)&src)[56], ((uchar*)&src)[57], ((uchar*)&src)[58], ((uchar*)&src)[59], ((uchar*)&src)[60], ((uchar*)&src)[61], ((uchar*)&src)[62], ((uchar*)&src)[63]);
#define print_m512i_char(src) printf_s("%s: %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d | %3d %3d %3d %3d\n",#src,((char*)&src)[0], ((char*)&src)[1], ((char*)&src)[2], ((char*)&src)[3], ((char*)&src)[4], ((char*)&src)[5], ((char*)&src)[6], ((char*)&src)[7], ((char*)&src)[8], ((char*)&src)[9], ((char*)&src)[10], ((char*)&src)[11], ((char*)&src)[12], ((char*)&src)[13], ((char*)&src)[14], ((char*)&src)[15],((char*)&src)[16], ((char*)&src)[17], ((char*)&src)[18], ((char*)&src)[19], ((char*)&src)[20], ((char*)&src)[21], ((char*)&src)[22], ((char*)&src)[23], ((char*)&src)[24], ((char*)&src)[25], ((char*)&src)[26], ((char*)&src)[27], ((char*)&src)[28], ((char*)&src)[29], ((char*)&src)[30], ((char*)&src)[31],((char*)&src)[32], ((char*)&src)[33], ((char*)&src)[34], ((char*)&src)[35], ((char*)&src)[36], ((char*)&src)[37], ((char*)&src)[38], ((char*)&src)[39], ((char*)&src)[40], ((char*)&src)[41], ((char*)&src)[42], ((char*)&src)[43], ((char*)&src)[44], ((char*)&src)[45], ((char*)&src)[46], ((char*)&src)[47],((char*)&src)[48], ((char*)&src)[49], ((char*)&src)[50], ((char*)&src)[51], ((char*)&src)[52], ((char*)&src)[53], ((char*)&src)[54], ((char*)&src)[55], ((char*)&src)[56], ((char*)&src)[57], ((char*)&src)[58], ((char*)&src)[59], ((char*)&src)[60], ((char*)&src)[61], ((char*)&src)[62], ((char*)&src)[63]);

//opencp: not implement in VS2022 (checked 2023/8/12)
#ifndef __INTEL_LLVM_COMPILER
STATIC_INLINE __m512h _mm512_exp_ph(__m512h src)
{
	const __m512 slo = _mm512_cvtph_ps(_mm512_castsi512_si256(src));
	const __m512 shi = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(src, 1));
	return _mm512_setr_m256i(_mm512_cvtps_ph(_mm512_exp_ps(slo), CP_ROUND_NEAREST), _mm512_cvtps_ph(_mm512_exp_ps(shi), CP_ROUND_NEAREST));
}
//opencp: not implement in VS2022 (checked 2023/8/12)
inline __m512h _mm512_pow_ph(__m512h src, __m512h pow)
{
	const __m512 slo = _mm512_cvtph_ps(_mm512_castsi512_si256(src));
	const __m512 shi = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(src, 1));
	const __m512 plo = _mm512_cvtph_ps(_mm512_castsi512_si256(pow));
	const __m512 phi = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(pow, 1));
	return _mm512_setr_m256i(_mm512_cvtps_ph(_mm512_pow_ps(slo, plo), CP_ROUND_NEAREST), _mm512_cvtps_ph(_mm512_pow_ps(shi, phi), CP_ROUND_NEAREST));
}
inline __m512h _mm512_cbrt_ph(__m512h src)
{
	const __m512 slo = _mm512_cvtph_ps(_mm512_castsi512_si256(src));
	const __m512 shi = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(src, 1));
	return _mm512_setr_m256i(_mm512_cvtps_ph(_mm512_cbrt_ps(slo), CP_ROUND_NEAREST), _mm512_cvtps_ph(_mm512_cbrt_ps(shi), CP_ROUND_NEAREST));
}
#endif

#ifdef __AVX512FP16__
STATIC_INLINE void _print_m512h(const char* name, __m512h src)
{
	__m512 slo = _mm512_cvtph_ps(_mm512_castsi512_si256(src));
	__m512 shi = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(src, 1));
	printf_s("%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f\n",
		((float*)&slo)[0], ((float*)&slo)[1], ((float*)&slo)[2], ((float*)&slo)[3], ((float*)&slo)[4], ((float*)&slo)[5], ((float*)&slo)[6], ((float*)&slo)[7],
		((float*)&slo)[8], ((float*)&slo)[9], ((float*)&slo)[10], ((float*)&slo)[11], ((float*)&slo)[12], ((float*)&slo)[13], ((float*)&slo)[14], ((float*)&slo)[15],
		((float*)&shi)[0], ((float*)&shi)[1], ((float*)&shi)[2], ((float*)&shi)[3], ((float*)&shi)[4], ((float*)&shi)[5], ((float*)&shi)[6], ((float*)&shi)[7],
		((float*)&shi)[8], ((float*)&shi)[9], ((float*)&shi)[10], ((float*)&shi)[11], ((float*)&shi)[12], ((float*)&shi)[13], ((float*)&shi)[14], ((float*)&shi)[15]);
}
STATIC_INLINE void _printInt6_m512h(const char* name, __m512h src)
{
	__m512i sint = _mm512_cvtph_epi16(src);
	printf_s("%6d %6d %6d %6d %6d %6d %6d %6d | %6d %6d %6d %6d %6d %6d %6d %6d | %6d %6d %6d %6d %6d %6d %6d %6d | %6d %6d %6d %6d %6d %6d %6d %6d\n",
		((short*)&sint)[0], ((short*)&sint)[1], ((short*)&sint)[2], ((short*)&sint)[3], ((short*)&sint)[4], ((short*)&sint)[5], ((short*)&sint)[6], ((short*)&sint)[7],
		((short*)&sint)[8], ((short*)&sint)[9], ((short*)&sint)[10], ((short*)&sint)[11], ((short*)&sint)[12], ((short*)&sint)[13], ((short*)&sint)[14], ((short*)&sint)[15],
		((short*)&sint)[16], ((short*)&sint)[17], ((short*)&sint)[18], ((short*)&sint)[19], ((short*)&sint)[20], ((short*)&sint)[21], ((short*)&sint)[22], ((short*)&sint)[23],
		((short*)&sint)[24], ((short*)&sint)[25], ((short*)&sint)[26], ((short*)&sint)[27], ((short*)&sint)[28], ((short*)&sint)[29], ((short*)&sint)[30], ((short*)&sint)[31]);
}

STATIC_INLINE void _printInt3_m512h(const char* name, __m512h src)
{
	__m512i sint = _mm512_cvtph_epi16(src);
	printf_s("%3d %3d %3d %3d %3d %3d %3d %3d | %3d %3d %3d %3d %3d %3d %3d %3d | %3d %3d %3d %3d %3d %3d %3d %3d | %3d %3d %3d %3d %3d %3d %3d %3d\n",
		((short*)&sint)[0], ((short*)&sint)[1], ((short*)&sint)[2], ((short*)&sint)[3], ((short*)&sint)[4], ((short*)&sint)[5], ((short*)&sint)[6], ((short*)&sint)[7],
		((short*)&sint)[8], ((short*)&sint)[9], ((short*)&sint)[10], ((short*)&sint)[11], ((short*)&sint)[12], ((short*)&sint)[13], ((short*)&sint)[14], ((short*)&sint)[15],
		((short*)&sint)[16], ((short*)&sint)[17], ((short*)&sint)[18], ((short*)&sint)[19], ((short*)&sint)[20], ((short*)&sint)[21], ((short*)&sint)[22], ((short*)&sint)[23],
		((short*)&sint)[24], ((short*)&sint)[25], ((short*)&sint)[26], ((short*)&sint)[27], ((short*)&sint)[28], ((short*)&sint)[29], ((short*)&sint)[30], ((short*)&sint)[31]);
}
#define print_m512h(src) _print_m512h(#src, src);
#define printInt6_m512h(src) _printInt6_m512h(#src, src);
#define printInt3_m512h(src) _printInt3_m512h(#src, src);

inline __m512h _mm512_set1_ph(float src)
{
	__m128h rh = _mm_cvtps_ph(_mm_set1_ps(src), _MM_FROUND_TO_NEAREST_INT);
	return _mm512_broadcastw_epi16(rh);
}

inline __m256h _mm256_set1_ph(float src)
{
	__m128h rh = _mm_cvtps_ph(_mm_set1_ps(src), _MM_FROUND_TO_NEAREST_INT);
	return _mm256_broadcastw_epi16(rh);
}

STATIC_INLINE __m512h _mm512_set_step_ph(float v, float step = 1.f)
{
	return _mm512_setr_m256i(
		_mm512_cvtps_ph(_mm512_set_step_ps(v, step), CP_ROUND_NEAREST),
		_mm512_cvtps_ph(_mm512_set_step_ps((v + 16.f) * step, step), CP_ROUND_NEAREST)
	);
}

STATIC_INLINE void _mm512_load_cvtph_bgr2planar_ph(const short* ptr, __m512h& b, __m512h& g, __m512h& r)
{
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47 | 48 49 50 51 52 53 54 55 | 56 57 58 59 60 61 62 63
	//  64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79 | 80 81 82 83 84 85 86 87 | 88 89 90 91 92 93 94 95

	b = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*)(ptr + 0)));
	g = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*)(ptr + 16)));
	r = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*)(ptr + 32)));
	b = _mm512_inserti32x8(b, _mm256_load_si256((const __m256i*)(ptr + 48)), 1);
	g = _mm512_inserti32x8(g, _mm256_load_si256((const __m256i*)(ptr + 64)), 1);
	r = _mm512_inserti32x8(r, _mm256_load_si256((const __m256i*)(ptr + 80)), 1);
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 48 49 50 51 52 53 54 55 | 56 57 58 59 60 61 62 63
	//  16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31 | 64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47 | 80 81 82 83 84 85 86 87 | 88 89 90 91 92 93 94 95

	//static const __m512i idx0 = _mm512_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55);
	static const __m512i idx0 = _mm512_setr_epi64(3096267694080008, 4222184781053964, 9851770215923744, 10977687302897700, 7599936041975832, 8725853128949788, 14355438563819568, 15481355650793524);
	__m512i a1 = _mm512_permutex2var_ph(b, idx0, r);
	__m512i a0 = _mm512_mask_blend_epi16(0xff00ff00, b, g);
	__m512i a2 = _mm512_mask_blend_epi16(0xff00ff00, g, r);
	//   0  1  2  3  4  5  6  7 | 24 25 26 27 28 29 30 31 | 48 49 50 51 52 53 54 55 | 72 73 74 75 76 77 78 79
	//   8  9 10 11 12 13 14 15 | 32 33 34 35 36 37 38 39 | 56 57 58 59 60 61 62 63 | 80 81 82 83 84 85 86 87
	//  16 17 18 19 20 21 22 23 | 40 41 42 43 44 45 46 47 | 64 65 66 67 68 69 70 71 | 88 89 90 91 92 93 94 95

	//static const __m512i idx1 = _mm512_setr_epi16(4, 5, 6, 7, 32, 33, 34, 35, 12, 13, 14, 15, 40, 41, 42, 43, 20, 21, 22, 23, 48, 49, 50, 51, 28, 29, 30, 31, 56, 57, 58, 59);
	//static const __m512i idx1 = _mm512_setr_epi64(1970350607106052, 9851770215923744, 4222184781053964, 12103604389871656, 6474018955001876, 14355438563819568, 8725853128949788, 16607272737767480);
	//g = _mm512_permutex2var_ph(a0, idx1, a2);
	b = _mm512_mask_blend_ph(0xf0f0f0f0, a0, a1);
	g = _mm512_shuffle2_epi32(a0, a2, _MM_SHUFFLE(1, 0, 3, 2));//78
	r = _mm512_mask_blend_ph(0xf0f0f0f0, a1, a2);
	//   0  1  2  3 12 13 14 15 | 24 25 26 27 36 37 38 39 | 48 49 50 51 60 61 62 63 | 72 73 74 75 84 85 86 87
	//   4  5  6  7 16 17 18 19 | 28 29 30 31 40 41 42 43 | 52 53 54 55 64 65 66 67 | 76 77 78 79 88 89 90 91
	//   8  9 10 11 20 21 22 23 | 32 33 34 35 44 45 46 47 | 56 57 58 59 68 69 70 71 | 80 81 82 83 92 93 94 95

	//static const __m512i idx2 = _mm512_setr_epi16(2, 3, 32, 33, 6, 7, 36, 37, 10, 11, 40, 41, 14, 15, 44, 45, 18, 19, 48, 49, 22, 23, 52, 53, 26, 27, 56, 57, 30, 31, 60, 61);
	static const __m512i idx2 = _mm512_setr_epi64(9288811670601730, 10414728757575686, 11540645844549642, 12666562931523598, 13792480018497554, 14918397105471510, 16044314192445466, 17170231279419422);
	a1 = _mm512_permutex2var_ph(b, idx2, r);
	a0 = _mm512_mask_blend_epi16(0xcccccccc, b, g);
	a2 = _mm512_mask_blend_epi16(0xcccccccc, g, r);
	//   0  1  6  7 12 13 18 19 | 24 25 30 31 36 37 42 43 | 48 49 54 55 60 61 66 67 | 72 73 78 79 84 85 90 91
	//   2  3  8  9 14 15 20 21 | 26 27 32 33 38 39 44 45 | 50 51 56 57 62 63 68 69 | 74 75 80 81 86 87 92 93
	//   4  5 10 11 16 17 22 23 | 28 29 34 35 40 41 46 47 | 52 53 58 59 64 65 70 71 | 76 77 82 83 88 89 94 95

	//static const __m512i idx3 = _mm512_setr_epi16(1, 32, 3, 34, 5, 36, 7, 38, 9, 40, 11, 42, 13, 44, 15, 46, 17, 48, 19, 50, 21, 52, 23, 54, 25, 56, 27, 58, 29, 60, 31, 62	);
	static const __m512i idx3 = _mm512_setr_epi64(9570162095161345, 10696079182135301, 11821996269109257, 12947913356083213, 14073830443057169, 15199747530031125, 16325664617005081, 17451581703979037);
	g = _mm512_permutex2var_ph(a0, idx3, a2);
	b = _mm512_mask_blend_epi16(0xaaaaaaaa, a0, a1);
	r = _mm512_mask_blend_epi16(0xaaaaaaaa, a1, a2);
}

STATIC_INLINE void _mm512_loadu_cvtph_bgr2planar_ph(const short* ptr, __m512h& b, __m512h& g, __m512h& r)
{
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47 | 48 49 50 51 52 53 54 55 | 56 57 58 59 60 61 62 63
	//  64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79 | 80 81 82 83 84 85 86 87 | 88 89 90 91 92 93 94 95

	b = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr + 0)));
	g = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr + 16)));
	r = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr + 32)));
	b = _mm512_inserti32x8(b, _mm256_loadu_si256((const __m256i*)(ptr + 48)), 1);
	g = _mm512_inserti32x8(g, _mm256_loadu_si256((const __m256i*)(ptr + 64)), 1);
	r = _mm512_inserti32x8(r, _mm256_loadu_si256((const __m256i*)(ptr + 80)), 1);
	//   0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 48 49 50 51 52 53 54 55 | 56 57 58 59 60 61 62 63
	//  16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31 | 64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79
	//  32 33 34 35 36 37 38 39 | 40 41 42 43 44 45 46 47 | 80 81 82 83 84 85 86 87 | 88 89 90 91 92 93 94 95

	//__m512i a1 = _mm512_shuffle_i32x4(b, r, _MM_SHUFFLE(0, 1, 0, 1));
		//static const __m512i idx0 = _mm512_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55);
	static const __m512i idx0 = _mm512_setr_epi64(3096267694080008, 4222184781053964, 9851770215923744, 10977687302897700, 7599936041975832, 8725853128949788, 14355438563819568, 15481355650793524);
	__m512i a1 = _mm512_permutex2var_ph(b, idx0, r);
	__m512i a0 = _mm512_mask_blend_epi16(0xff00ff00, b, g);
	__m512i a2 = _mm512_mask_blend_epi16(0xff00ff00, g, r);
	//   0  1  2  3  4  5  6  7 | 24 25 26 27 28 29 30 31 | 48 49 50 51 52 53 54 55 | 72 73 74 75 76 77 78 79
	//   8  9 10 11 12 13 14 15 | 32 33 34 35 36 37 38 39 | 56 57 58 59 60 61 62 63 | 80 81 82 83 84 85 86 87
	//  16 17 18 19 20 21 22 23 | 40 41 42 43 44 45 46 47 | 64 65 66 67 68 69 70 71 | 88 89 90 91 92 93 94 95

	//static const __m512i idx1 = _mm512_setr_epi16(4, 5, 6, 7, 32, 33, 34, 35, 12, 13, 14, 15, 40, 41, 42, 43, 20, 21, 22, 23, 48, 49, 50, 51, 28, 29, 30, 31, 56, 57, 58, 59);
	//static const __m512i idx1 = _mm512_setr_epi64(1970350607106052, 9851770215923744, 4222184781053964, 12103604389871656, 6474018955001876, 14355438563819568, 8725853128949788, 16607272737767480);
	//g = _mm512_permutex2var_ph(a0, idx1, a2);
	b = _mm512_mask_blend_ph(0xf0f0f0f0, a0, a1);
	g = _mm512_shuffle2_epi32(a0, a2, _MM_SHUFFLE(1, 0, 3, 2));//78
	r = _mm512_mask_blend_ph(0xf0f0f0f0, a1, a2);
	//   0  1  2  3 12 13 14 15 | 24 25 26 27 36 37 38 39 | 48 49 50 51 60 61 62 63 | 72 73 74 75 84 85 86 87
	//   4  5  6  7 16 17 18 19 | 28 29 30 31 40 41 42 43 | 52 53 54 55 64 65 66 67 | 76 77 78 79 88 89 90 91
	//   8  9 10 11 20 21 22 23 | 32 33 34 35 44 45 46 47 | 56 57 58 59 68 69 70 71 | 80 81 82 83 92 93 94 95

	//static const __m512i idx2 = _mm512_setr_epi16(2, 3, 32, 33, 6, 7, 36, 37, 10, 11, 40, 41, 14, 15, 44, 45, 18, 19, 48, 49, 22, 23, 52, 53, 26, 27, 56, 57, 30, 31, 60, 61);
	static const __m512i idx2 = _mm512_setr_epi64(9288811670601730, 10414728757575686, 11540645844549642, 12666562931523598, 13792480018497554, 14918397105471510, 16044314192445466, 17170231279419422);
	a1 = _mm512_permutex2var_ph(b, idx2, r);
	a0 = _mm512_mask_blend_epi16(0xcccccccc, b, g);
	a2 = _mm512_mask_blend_epi16(0xcccccccc, g, r);
	//   0  1  6  7 12 13 18 19 | 24 25 30 31 36 37 42 43 | 48 49 54 55 60 61 66 67 | 72 73 78 79 84 85 90 91
	//   2  3  8  9 14 15 20 21 | 26 27 32 33 38 39 44 45 | 50 51 56 57 62 63 68 69 | 74 75 80 81 86 87 92 93
	//   4  5 10 11 16 17 22 23 | 28 29 34 35 40 41 46 47 | 52 53 58 59 64 65 70 71 | 76 77 82 83 88 89 94 95

	//static const __m512i idx3 = _mm512_setr_epi16(1, 32, 3, 34, 5, 36, 7, 38, 9, 40, 11, 42, 13, 44, 15, 46, 17, 48, 19, 50, 21, 52, 23, 54, 25, 56, 27, 58, 29, 60, 31, 62	);
	static const __m512i idx3 = _mm512_setr_epi64(9570162095161345, 10696079182135301, 11821996269109257, 12947913356083213, 14073830443057169, 15199747530031125, 16325664617005081, 17451581703979037);
	g = _mm512_permutex2var_ph(a0, idx3, a2);
	b = _mm512_mask_blend_epi16(0xaaaaaaaa, a0, a1);
	r = _mm512_mask_blend_epi16(0xaaaaaaaa, a1, a2);
}

STATIC_INLINE __m512h _mm512_loadu_auto16(const float* src)
{
	__m512i v0 = _mm512_castsi256_si512(_mm512_cvtps_ph(_mm512_loadu_ps(src), _MM_FROUND_TO_NEAREST_INT));
	__m256i v1 = _mm512_cvtps_ph(_mm512_loadu_ps(src + 16), _MM_FROUND_TO_NEAREST_INT);
	return _mm512_castsi512_ph(_mm512_inserti64x4(v0, v1, 1));
}

STATIC_INLINE __m512h _mm512_loadu_auto16(const unsigned char* src)
{
	return _mm512_cvtepi16_ph(_mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)src)));
}

STATIC_INLINE __m512h _mm512_loadu_auto16(const short* src)
{
	return _mm512_loadu_si512((__m512h*)src);
}

STATIC_INLINE __m256h _mm512_castph512_ph256hi(__m512h src)
{
	return _mm512_extracti32x8_epi32(src, 1);
}

STATIC_INLINE void _mm512_storeu_auto(unsigned char* dest, __m512h src)
{
	_mm_storeu_si128((__m128i*)(dest + 0), _mm512_cvtps_epu8(_mm512_cvtph_ps(_mm512_castph512_ph256(src))));
	_mm_storeu_si128((__m128i*)(dest + 16), _mm512_cvtps_epu8(_mm512_cvtph_ps(_mm512_castph512_ph256hi(src))));
}

STATIC_INLINE void _mm512_storeu_auto(float* dest, __m512h src)
{
	_mm512_storeu_ps(dest + 0, _mm512_cvtph_ps(_mm512_castph512_ph256(src)));
	_mm512_storeu_ps(dest + 16, _mm512_cvtph_ps(_mm512_castph512_ph256hi(src)));
}

STATIC_INLINE void _mm512_storescalar_auto(float* dest, __m512h ms, const int numpixel)
{
	__m512 ms0 = _mm512_cvtph_ps(_mm512_castph512_ph256(ms));
	__m512 ms1 = _mm512_cvtph_ps(_mm512_castph512_ph256hi(ms));
	_mm512_storescalar_ps(dest, ms0, numpixel);
	_mm512_storescalar_ps(dest + 16, ms1, numpixel);
}

STATIC_INLINE void _mm512_storeu_auto_color(float* dest, __m512h b, __m512h g, __m512h r)
{
	__m512 b0 = _mm512_cvtph_ps(_mm512_castph512_ph256(b));
	__m512 g0 = _mm512_cvtph_ps(_mm512_castph512_ph256(g));
	__m512 r0 = _mm512_cvtph_ps(_mm512_castph512_ph256(r));
	_mm512_storeu_ps_color(dest, b0, g0, r0);
	__m512 b1 = _mm512_cvtph_ps(_mm512_castph512_ph256hi(b));
	__m512 g1 = _mm512_cvtph_ps(_mm512_castph512_ph256hi(g));
	__m512 r1 = _mm512_cvtph_ps(_mm512_castph512_ph256hi(r));
	_mm512_storeu_ps_color(dest + 48, b1, g1, r1);
}

STATIC_INLINE void _mm512_storescalar_auto_color(float* dest, __m512h b, __m512h g, __m512h r, const int numpixel)
{
	/*__m512 dr, dg, db;
	_mm512_cvtsoa2aos_ps(b, g, r, db, dg, dr);
	float CV_DECL_ALIGNED(64) buffscalarstore[48];
	_mm512_store_ps(buffscalarstore + 0, db);
	_mm512_store_ps(buffscalarstore + 16, dg);
	_mm512_store_ps(buffscalarstore + 32, dr);

	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];*/
}
#endif //__AVX512FP16__

#endif //__AVX512F__

#endif //__CP_INLINE_SIMD_FUNCTIONS__