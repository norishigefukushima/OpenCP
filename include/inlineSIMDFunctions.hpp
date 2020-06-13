#pragma once

#pragma warning(disable:4309)

#define SSE_ALIGN 16
#define AVX_ALIGN 32
#define AVX512_ALIGN 64

//0 src1 low, 1 src1 high, //2 src2 low, 3 src2 high, 
#define _MM_SELECT4(x,y) (((y)<<4) + (x))

//template for array
//const int CV_DECL_ALIGNED(AVX_ALIGN) a[10]

inline int get_simd_ceil(const int val, const int simdwidth)
{
	return (val % simdwidth == 0) ? val : (val / simdwidth + 1) * simdwidth;
}

inline int get_simd_floor(const int val, const int simdwidth)
{
	return (val / simdwidth) * simdwidth;
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

//cast

inline __m128i _mm256_cvtpsx2_epu8(const __m256 v0, const __m256 v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(v0), _mm256_cvtps_epi32(v1)), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

inline __m128i _mm256_cvtepi32_epu8(const __m256i v0)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, _mm256_setzero_si256()), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

inline __m128i _mm256_cvtepi32x2_epu8(const __m256i v0, const __m256i v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, v1), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//load and cast 

//defined in opencp uchar->short
inline __m256i _mm256_load_epu8cvtepi16(const __m128i* P)
{
	return _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)P));
}

//defined in opencp short->int
inline __m256i _mm256_load_epi16cvtepi32(const __m128i* P)
{
	return _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)P));
}

//defined in opencp uchar->int
inline __m256i _mm256_load_epu8cvtepi32(const __m128i* P)
{
	return _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P));
}

//defined in opencp uchar->intx2
inline void _mm256_load_epu8cvtepi32x2(const __m128i* P, __m256i& d0, __m256i& d1)
{
	__m128i s = _mm_loadu_si128((__m128i*)P);
	d0 = _mm256_cvtepu8_epi32(s);
	d1 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
}

//defined in opencp uchar->float
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

//gray2bgr: broadcast 3 channels
inline void _mm_cvtepi8_gray2bgr(const __m128i src, __m128i& d0, __m128i& d1, __m128i& d2)
{
	static const __m128i g2rgbmask0 = _mm_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5);
	static const __m128i g2rgbmask1 = _mm_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10);
	static const __m128i g2rgbmask2 = _mm_setr_epi8(10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15);
	d0 = _mm_shuffle_epi8(src, g2rgbmask0);
	d1 = _mm_shuffle_epi8(src, g2rgbmask1);
	d2 = _mm_shuffle_epi8(src, g2rgbmask2);
}

//gray2bgr: broadcast 3 channels
inline void _mm256_cvtepi8_gray2bgr(const __m256i src, __m256i& d0, __m256i& d1, __m256i& d2)
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
inline void _mm256_cvtepi16_gray2bgr(const __m256i src, __m256i& d0, __m256i& d1, __m256i& d2)
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
inline void _mm256_cvtepi32_gray2bgr(const __m256i src, __m256i& d0, __m256i& d1, __m256i& d2)
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
inline void _mm256_cvtps_gray2bgr(const __m256 srcf, __m256& d0, __m256& d1, __m256& d2)
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
inline void _mm256_cvtps_gray2bgr_v2(const __m256 src, __m256& d0, __m256& d1, __m256& d2)
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
inline void _mm256_cvtps_gray2bgr_v3(const __m256 src, __m256& d0, __m256& d1, __m256& d2)
{
	static const __m256i pmask0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
	static const __m256i pmask1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
	static const __m256i pmask2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
	d0 = _mm256_permutevar8x32_ps(src, pmask0);
	d1 = _mm256_permutevar8x32_ps(src, pmask1);
	d2 = _mm256_permutevar8x32_ps(src, pmask2);
}

//gray2bgr: broadcast 3 channels
inline void _mm256_cvtpd_gray2bgr(const __m256d src, __m256d& d0, __m256d& d1, __m256d& d2)
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
inline void _mm256_cvtpd_gray2bgr_v2(const __m256d src, __m256d& d0, __m256d& d1, __m256d& d2)
{
	const __m256d md0 = _mm256_shuffle_pd(src, src, 0b0000);
	const __m256d md1 = _mm256_shuffle_pd(src, src, 0b1111);
	const __m256d md2 = _mm256_permute2f128_pd(src, src, _MM_SELECT4(1, 0));
	d0 = _mm256_blend_pd(md0, md2, 0b1100);
	d1 = _mm256_blend_pd(md1, md0, 0b1100);
	d2 = _mm256_blend_pd(md2, md1, 0b1100);
}

//gray2bgr: broadcast 3 channels
inline void _mm256_cvtpd_gray2bgr_v3(const __m256d src, __m256d& d0, __m256d& d1, __m256d& d2)
{
	const __m256d md0 = _mm256_shuffle_pd(src, src, 0b0000);
	const __m256d md1 = _mm256_shuffle_pd(src, src, 0b1111);
	d0 = _mm256_permute2f128_pd(md0, src, _MM_SELECT4(0, 2));
	d1 = _mm256_blend_pd(md1, md0, 0b1100);
	d2 = _mm256_permute2f128_pd(src, md1, _MM_SELECT4(1, 3));
}

//plain2bgr: plain b,g,r image to interleave rgb. SoA->AoS
inline void _mm256_cvtps_planar2bgr(const __m256 b, const __m256 g, const __m256 r, __m256& d0, __m256& d1, __m256& d2)
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
	d0 = _mm256_permute2f128_ps(bval, rval, pmask1);
	d1 = _mm256_permute2f128_ps(gval, bval, pmask2);
	d2 = _mm256_permute2f128_ps(rval, gval, pmask3);
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

inline void _mm256_stream_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
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

inline __m256 _mm256_div_avoidzerodiv_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_div_ps(src1, _mm256_blendv_ps(src2, _mm256_set1_ps(FLT_MIN), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0)));
}

inline __m256 _mm256_div_zerodivzero_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_blendv_ps(_mm256_div_ps(src1, src2), _mm256_set1_ps(FLT_MIN), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0));
}

inline __m128i _mm_cmpgt_epu8(__m128i x, __m128i y)
{
	return _mm_andnot_si128(_mm_cmpeq_epi8(x, y), _mm_cmpeq_epi8(_mm_max_epu8(x, y), x));
}

inline __m256i _mm256_cmpgt_epu8(__m256i x, __m256i y)
{
	return _mm256_andnot_si256(_mm256_cmpeq_epi8(x, y), _mm256_cmpeq_epi8(_mm256_max_epu8(x, y), x));
}

//_mm256_cvtepi32_epi16 is already defined in zmmintrin.h (AVX512)
inline __m128i _mm256_cvtint_short(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

//_mm256_cvtepi16_epi8 is already defined in zmmintrin.h (AVX512), but this is ep`u`
inline __m128i _mm256_cvtepi16_epu8(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

inline __m128i _mm256_cvtint_ushort(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

inline __m256i _mm256_cvtintx2_short(__m256i src1, __m256i src2)
{
	return _mm256_permute4x64_epi64(_mm256_packs_epi32(src1, src2), _MM_SHUFFLE(3, 1, 2, 0));
}

inline void print(__m128d src)
{
	printf_s("%5.3f %5.3f\n",
		src.m128d_f64[0], src.m128d_f64[1]);
}

inline void print(__m256d src)
{
	printf_s("%5.3f %5.3f %5.3f %5.3f\n",
		src.m256d_f64[0], src.m256d_f64[1], src.m256d_f64[2], src.m256d_f64[3]);
}

inline void print(__m128 src)
{
	printf_s("%5.3f %5.3f %5.3f %5.3f\n",
		src.m128_f32[0], src.m128_f32[1],
		src.m128_f32[2], src.m128_f32[3]);
}

inline void print(__m256 src)
{
	printf_s("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n",
		src.m256_f32[0], src.m256_f32[1], src.m256_f32[2], src.m256_f32[3], src.m256_f32[4], src.m256_f32[5], src.m256_f32[6], src.m256_f32[7]);
}

inline void print_char(__m128i src)
{
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", src.m128i_i8[i]);
	}
	printf_s("\n");
}

inline void print_char(__m256i src)
{
	for (int i = 0; i < 32; i++)
	{
		printf_s("%3d ", src.m256i_i8[i]);
	}
	printf_s("\n");
}

inline void print_uchar(__m128i src)
{
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", src.m128i_u8[i]);
	}
	printf_s("\n");
}

inline void print_uchar(__m256i src)
{
	for (int i = 0; i < 32; i++)
	{
		printf_s("%3d ", src.m256i_u8[i]);
	}
	printf_s("\n");
}

inline void print_short(__m128i src)
{
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", src.m128i_i16[i]);
	}
	printf_s("\n");
}

inline void print_short(__m256i src)
{
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", src.m256i_i16[i]);
	}
	printf_s("\n");
}

inline void print_ushort(__m128i src)
{
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", src.m128i_u16[i]);
	}
	printf_s("\n");
}

inline void print_ushort(__m256i src)
{
	for (int i = 0; i < 16; i++)
	{
		printf_s("%3d ", src.m256i_u16[i]);
	}
	printf_s("\n");
}

inline void print_int(__m128i src)
{
	for (int i = 0; i < 4; i++)
	{
		printf_s("%3d ", src.m128i_i32[i]);
	}
	printf_s("\n");
}

inline void print_int(__m256i src)
{
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", src.m256i_i32[i]);
	}
	printf_s("\n");
}

inline void print_uint(__m128i src)
{
	for (int i = 0; i < 4; i++)
	{
		printf_s("%3d ", src.m128i_u32[i]);
	}
	printf_s("\n");
}

inline void print_uint(__m256i src)
{
	for (int i = 0; i < 8; i++)
	{
		printf_s("%3d ", src.m256i_u32[i]);
	}
	printf_s("\n");
}

//broadcast
/*
__m128 xxxx = _mm_shuffle_ps(first, first, 0x00); // _MM_SHUFFLE(0, 0, 0, 0)
__m128 yyyy = _mm_shuffle_ps(first, first, 0x55); // _MM_SHUFFLE(1, 1, 1, 1)
__m128 zzzz = _mm_shuffle_ps(first, first, 0xAA); // _MM_SHUFFLE(2, 2, 2, 2)
__m128 wwww = _mm_shuffle_ps(first, first, 0xFF); // _MM_SHUFFLE(3, 3, 3, 3)
*/