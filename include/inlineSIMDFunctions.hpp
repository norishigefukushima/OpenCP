#pragma once

#define SSE_ALIGN 16
#define AVX_ALIGN 32
#define AVX512_ALIGN 64

//template for array
//const int CV_DECL_ALIGNED(AVX_ALIGN) a[10]


inline int get_simd_ceil(const int val, const int simdwidth)
{
	return (val%simdwidth == 0) ? val : (val / simdwidth + 1)*simdwidth;
}

inline int get_simd_floor(const int val, const int simdwidth)
{
	return (val / simdwidth)*simdwidth;
}

inline float _mm256_reduceadd_ps(__m256 src)
{
	src = _mm256_hadd_ps(src, src);
	src = _mm256_hadd_ps(src, src);
	return (src.m256_f32[0] + src.m256_f32[4]);
}

inline double _mm256_reduceadd_pd(__m256d src)
{
	src = _mm256_hadd_pd(src, src);
	return (src.m256d_f64[0] + src.m256d_f64[2]);
}

inline __m128i _mm256_cvtepi32x2_epu8(const __m256i v0, const __m256i v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, v1), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

inline __m256 _mm256_load_epu8cvtps(const __m128i* P)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
}

inline void _mm256_store_epi8_color(uchar* dst, __m256i b, __m256i g, __m256i r)
{
	static const __m256i mask1 = _mm256_set_epi8(
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0,
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0
	);
	static const __m256i mask2 = _mm256_set_epi8(
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5,
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5
	);
	static const __m256i mask3 = _mm256_set_epi8(
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10,
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10
	);

	static const __m256i bmask1 = _mm256_set_epi8(
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255,
		0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0
	);
	static const __m256i bmask2 = _mm256_set_epi8(
		255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255,
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255
	);

	const __m256i aa = _mm256_shuffle_epi8(b, mask1);
	const __m256i bb = _mm256_shuffle_epi8(g, mask2);
	const __m256i cc = _mm256_shuffle_epi8(r, mask3);

	__m256i aaa = _mm256_permute2x128_si256(aa, aa, 0x00);
	__m256i bbb = _mm256_permute2x128_si256(bb, bb, 0x00);
	__m256i ccc = _mm256_permute2x128_si256(cc, cc, 0x00);
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst)), _mm256_blendv_epi8(ccc, _mm256_blendv_epi8(aaa, bbb, bmask1), bmask2));
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 32), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(bb, aa, bmask2), bmask1));
	aaa = _mm256_permute2x128_si256(aa, aa, 0x11);
	bbb = _mm256_permute2x128_si256(bb, bb, 0x11);
	ccc = _mm256_permute2x128_si256(cc, cc, 0x11);
	_mm256_store_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 64), _mm256_blendv_epi8(aaa, _mm256_blendv_epi8(bbb, ccc, bmask1), bmask2));
}

inline void _mm256_storeu_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r)
{
	__m256i mb = _mm256_cvtps_epi32(b);
	__m256i mg = _mm256_cvtps_epi32(g);
	__m256i mr = _mm256_cvtps_epi32(r);
	mb = _mm256_packus_epi16(mb, mg);
	mb = _mm256_packus_epi16(mb, mr);
	static const __m256i mask = _mm256_setr_epi8(0, 4, 8, 1, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 0, 4, 8, 1);
	mb = _mm256_shuffle_epi8(mb, mask);
	__m256i mp = _mm256_permute2f128_si256(mb, mb, 0x11);
	mb = _mm256_blend_epi32(mb, mp, 8);

	_mm256_storeu_si256((__m256i*)dst, mb);
}

inline void _mm256_storescalar_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r)
{
	__m256i mb = _mm256_cvtps_epi32(b);
	__m256i mg = _mm256_cvtps_epi32(g);
	__m256i mr = _mm256_cvtps_epi32(r);
	mb = _mm256_packus_epi16(mb, mg);
	mb = _mm256_packus_epi16(mb, mr);
	static const __m256i mask = _mm256_setr_epi8(0, 4, 8, 1, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 0, 4, 8, 1);
	mb = _mm256_shuffle_epi8(mb, mask);
	__m256i mp = _mm256_permute2f128_si256(mb, mb, 0x11);
	mb = _mm256_blend_epi32(mb, mp, 8);

	uchar CV_DECL_ALIGNED(32) buffscalarstore[32];
	_mm256_storeu_si256((__m256i*)buffscalarstore, mb);
	uchar* dest = (uchar*)dst;
	for (int i = 0; i < 24; i++)
		dest[i] = buffscalarstore[i];
}

void inline _mm256_stream_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
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
	_mm256_stream_ps((float*)dst + 0, _mm256_permute2f128_ps(bval, rval, pmask1));
	_mm256_stream_ps((float*)dst + 8, _mm256_permute2f128_ps(gval, bval, pmask2));
	_mm256_stream_ps((float*)dst + 16, _mm256_permute2f128_ps(rval, gval, pmask3));
}

void inline _mm256_store_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
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