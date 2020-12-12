#pragma once

#pragma warning(disable:4309)

#define SSE_ALIGN 16
#define AVX_ALIGN 32
#define AVX512_ALIGN 64

//0 src1 low, 1 src1 high, //2 src2 low, 3 src2 high, 
#define _MM_SELECT4(x,y) (((y)<<4) + (x))

#pragma region xxx
#pragma endregion

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

inline void get_simd_widthend(const int cv_depth, const int channels, const int image_width, int& dest_endwidth, int& dest_pad_pixels)
{
	if (cv_depth == CV_32F && image_width % 8 == 0)
	{
		dest_endwidth = image_width;
		dest_pad_pixels = 0;
	}
	else if (cv_depth == CV_32F && image_width % 8 != 0)
	{
		dest_endwidth = get_simd_floor(image_width, 8);
		dest_pad_pixels = (image_width - dest_endwidth) * channels;
	}
	else if (cv_depth == CV_8U)
	{
		dest_endwidth = get_simd_floor(image_width - 8, 8);
		dest_pad_pixels = (image_width - dest_endwidth) * channels;
	}
}

inline void _mm_transposel_epi8(__m128i& s0, __m128i& s1, __m128i& s2, __m128i& s3, __m128i& s4, __m128i& s5, __m128i& s6, __m128i& s7)
{
	__m128i t[8];
	for (int i = 0; i < 8; i++)
	{
		t[i].m128i_u8[0] = s0.m128i_u8[i];
		t[i].m128i_u8[1] = s1.m128i_u8[i];
		t[i].m128i_u8[2] = s2.m128i_u8[i];
		t[i].m128i_u8[3] = s3.m128i_u8[i];
		t[i].m128i_u8[4] = s4.m128i_u8[i];
		t[i].m128i_u8[5] = s5.m128i_u8[i];
		t[i].m128i_u8[6] = s6.m128i_u8[i];
		t[i].m128i_u8[7] = s7.m128i_u8[i];
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

#define ___MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3, out4, out5, out6, out7, __in0, __in1, __in2, __in3, __in4, __in5, __in6, __in7, __out0, __out1, __out2, __out3, __out4, __out5, __out6, __out7, __tmp0, __tmp1, __tmp2, __tmp3, __tmp4, __tmp5, __tmp6, __tmp7, __tmpp0, __tmpp1, __tmpp2, __tmpp3, __tmpp4, __tmpp5, __tmpp6, __tmpp7) \
  do { \
    __m256 __in0 = (in0), __in1 = (in1), __in2 = (in2), __in3 = (in3), __in4 = (in4), __in5 = (in5), __in6 = (in6), __in7 = (in7); \
    __m256 __tmp0, __tmp1, __tmp2, __tmp3, __tmp4, __tmp5, __tmp6, __tmp7; \
    __m256 __tmpp0, __tmpp1, __tmpp2, __tmpp3, __tmpp4, __tmpp5, __tmpp6, __tmpp7; \
    __m256 __out0, __out1, __out2, __out3, __out4, __out5, __out6, __out7; \
    __tmp0  = _mm256_unpacklo_ps(__in0, __in1); \
    __tmp1  = _mm256_unpackhi_ps(__in0, __in1); \
    __tmp2  = _mm256_unpacklo_ps(__in2, __in3); \
    __tmp3  = _mm256_unpackhi_ps(__in2, __in3); \
    __tmp4  = _mm256_unpacklo_ps(__in4, __in5); \
    __tmp5  = _mm256_unpackhi_ps(__in4, __in5); \
    __tmp6  = _mm256_unpacklo_ps(__in6, __in7); \
    __tmp7  = _mm256_unpackhi_ps(__in6, __in7); \
    __tmpp0 = _mm256_shuffle_ps(__tmp0, __tmp2, 0x44); \
    __tmpp1 = _mm256_shuffle_ps(__tmp0, __tmp2, 0xEE); \
    __tmpp2 = _mm256_shuffle_ps(__tmp1, __tmp3, 0x44); \
    __tmpp3 = _mm256_shuffle_ps(__tmp1, __tmp3, 0xEE); \
    __tmpp4 = _mm256_shuffle_ps(__tmp4, __tmp6, 0x44); \
    __tmpp5 = _mm256_shuffle_ps(__tmp4, __tmp6, 0xEE); \
    __tmpp6 = _mm256_shuffle_ps(__tmp5, __tmp7, 0x44); \
    __tmpp7 = _mm256_shuffle_ps(__tmp5, __tmp7, 0xEE); \
    __out0  = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x20); \
    __out1  = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x20); \
    __out2  = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x20); \
    __out3  = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x20); \
    __out4  = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x31); \
    __out5  = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x31); \
    __out6  = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x31); \
    __out7  = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x31); \
    (out0)  = __out0, (out1) = __out1, (out2) = __out2, (out3) = __out3, (out4) = __out4, (out5) = __out5, (out6) = __out6, (out7) = __out7; \
          } while (0)
#define _MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7) \
      ___MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4, in5, in6, in7, \
          __in0##__LINE__, __in1##__LINE__, __in2##__LINE__, __in3##__LINE__, __in4##__LINE__, __in5##__LINE__, __in6##__LINE__, __in7##__LINE__, \
          __out0##__LINE__, __out1##__LINE__, __out2##__LINE__, __out3##__LINE__, __out4##__LINE__, __out5##__LINE__, __out6##__LINE__, __out7##__LINE__, \
          __tmp0##__LINE__, __tmp1##__LINE__, __tmp2##__LINE__, __tmp3##__LINE__, __tmp4##__LINE__, __tmp5##__LINE__, __tmp6##__LINE__, __tmp7##__LINE__, \
          __tmpp0##__LINE__, __tmpp1##__LINE__, __tmpp2##__LINE__, __tmpp3##__LINE__, __tmpp4##__LINE__, __tmpp5##__LINE__, __tmpp6##__LINE__, __tmpp7##__LINE__)


#pragma region convert

//cast
//opencp cast __m256i of hi register ->__m128i
inline __m128i _mm256_castsi256hi_si128(__m256i src)
{
	return _mm256_extractf128_si256(src, 1);
}

//opencp (same as _mm256_extractf128_si256(src, 1))
//#define _mm256_castsi256hi_si128(src) *((__m128i*)&(src) + 1)

//opencp uchar->float
inline __m256 _mm256_cvtepu8_ps(__m128i src)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
}

//opencp uchar->floatx2
inline void _mm256_cvtepu8_psx2(__m128i src, __m256& dest0, __m256& dest1)
{
	dest0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
	dest1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(src, _MM_SHUFFLE(1, 0, 3, 2))));
}

//opencp int ->uchar
inline __m128i _mm256_cvtepi32_epu8(const __m256i v0)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, _mm256_setzero_si256()), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//opencp float->uchar
inline __m128i _mm256_cvtps_epu8(__m256 ms)
{
	return _mm256_cvtepi32_epu8(_mm256_cvtps_epi32(ms));
}

//opencp floatx2->uchar
inline __m128i _mm256_cvtpsx2_epu8(const __m256 v0, const __m256 v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(v0), _mm256_cvtps_epi32(v1)), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//opencp intx2 ->uchar
inline __m128i _mm256_cvtepi32x2_epu8(const __m256i v0, const __m256i v1)
{
	return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_packus_epi16(_mm256_packs_epi32(v0, v1), _mm256_setzero_si256()), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
}

//_mm256_cvtepi32_epi16 is already defined in zmmintrin.h (AVX512)
inline __m128i _mm256_cvtepi32_epi16_v2(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

//_mm256_cvtepi16_epi8 is already defined in zmmintrin.h (AVX512), but this is ep`u`
inline __m128i _mm256_cvtepi16_epu8(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

inline __m128i _mm256_cvtepi32_epu16(__m256i src)
{
	return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(src, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0)));
}

inline __m256i _mm256_cvepi32x2_epi16(__m256i src1, __m256i src2)
{
	return _mm256_permute4x64_epi64(_mm256_packs_epi32(src1, src2), _MM_SHUFFLE(3, 1, 2, 0));
}

#pragma endregion

#pragma region load and cast 
//opencp: uchar->short
inline __m256i _mm256_load_epu8cvtepi16(const __m128i* P)
{
	return _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)P));
}

//opencp: short->int
inline __m256i _mm256_load_epi16cvtepi32(const __m128i* P)
{
	return _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)P));
}

//opencp: uchar->int
inline __m256i _mm256_load_epu8cvtepi32(const __m128i* P)
{
	return _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P));
}

//opencp: uchar->intx2
inline void _mm256_load_epu8cvtepi32x2(const __m128i* P, __m256i& d0, __m256i& d1)
{
	__m128i s = _mm_load_si128((__m128i*)P);
	d0 = _mm256_cvtepu8_epi32(s);
	d1 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
}

//opencp: uchar->float
inline __m256 _mm256_load_epu8cvtps(const __m128i* P)
{
	return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)P)));
	//return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)P)));
}

//opencp: uchar->floatx2
inline void _mm256_load_epu8cvtpsx2(const __m128i* P, __m256& d0, __m256& d1)
{
	__m128i t = _mm_loadu_si128((__m128i*)P);
	d0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(t));
	d1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2))));
}
#pragma endregion

#pragma region color_convert

inline void _mm256_load_cvtpd_bgr2planar_pd(const double* ptr, __m256d& b, __m256d& g, __m256d& r)
{
	__m256d bgr0 = _mm256_loadu_pd(ptr);
	__m256d bgr1 = _mm256_loadu_pd(ptr + 4);
	__m256d bgr2 = _mm256_loadu_pd(ptr + 8);

	__m256d s02_low = _mm256_permute2f128_pd(bgr0, bgr2, _MM_SELECT4(0, 2));
	__m256d s02_high = _mm256_permute2f128_pd(bgr0, bgr2, _MM_SELECT4(1, 3));

	r = _mm256_blend_pd(_mm256_blend_pd(s02_low, s02_high, 0b1001), bgr1,0b0010);
	__m256d g0 = _mm256_blend_pd(_mm256_blend_pd(bgr0, bgr2, 0b1100), bgr1, 0b1001);
	g = _mm256_shuffle_pd(g0, g0, 0b0101);
	b = _mm256_blend_pd(_mm256_blend_pd(s02_high, s02_low, 0b1001), bgr1, 0b0100);
}

inline void _mm256_load_cvtps_bgr2planar_ps(const float* ptr, __m256& b, __m256& g, __m256& r)
{
	__m256 bgr0 = _mm256_loadu_ps(ptr);
	__m256 bgr1 = _mm256_loadu_ps(ptr + 8);
	__m256 bgr2 = _mm256_loadu_ps(ptr + 16);

	__m256 s02_low = _mm256_permute2f128_ps(bgr0, bgr2, 0 + 2 * 16);
	__m256 s02_high = _mm256_permute2f128_ps(bgr0, bgr2, 1 + 3 * 16);

	__m256 b0 = _mm256_blend_ps(_mm256_blend_ps(s02_low, s02_high, 0x24), bgr1, 0x92);
	__m256 g0 = _mm256_blend_ps(_mm256_blend_ps(s02_high, s02_low, 0x92), bgr1, 0x24);
	__m256 r0 = _mm256_blend_ps(_mm256_blend_ps(bgr1, s02_low, 0x24), s02_high, 0x92);

	b = _mm256_shuffle_ps(b0, b0, 0x6c);
	g = _mm256_shuffle_ps(g0, g0, 0xb1);
	r = _mm256_shuffle_ps(r0, r0, 0xc6);
}

//opencp: BGR2Planar (uchar). Same function of OpenCV for SSE4.1.
inline void _mm_load_cvtepu8bgr2planar_si128(const uchar* ptr, __m128i& b, __m128i& g, __m128i& r)
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
inline void _mm_load_cvtepu8bgr2planar_epi64(const uchar* ptr, __m128i& b, __m128i& g, __m128i& r)
{
	//b = _mm_setr_epi8(ptr[0], ptr[3], ptr[6], ptr[9], ptr[12], ptr[15], ptr[18], ptr[21], 0, 0, 0, 0, 0, 0, 0, 0);
	//g = _mm_setr_epi8(ptr[1], ptr[4], ptr[7], ptr[10], ptr[13], ptr[16], ptr[19], ptr[22], 0, 0, 0, 0, 0, 0, 0, 0);
	//r = _mm_setr_epi8(ptr[2], ptr[5], ptr[8], ptr[11], ptr[14], ptr[17], ptr[20], ptr[23], 0, 0, 0, 0, 0, 0, 0, 0);

	const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
	const __m128i mask2 = _mm_setr_epi8(0, 3, 6, 0, 0, 0, 2, 5, 0, 0, 0, 1, 4, 7, 0, 0);

	const __m128i smask1 = _mm_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
	const __m128i smask2 = _mm_setr_epi8(11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	const __m128i smask3 = _mm_setr_epi8(0, 0, 0, 0, 0, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0);

	const __m128i bmask1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	const __m128i bmask2 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	__m128i s0 = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(ptr)), mask1);      //bbbbbbgggggrrrrr
	__m128i s1 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(ptr + 16)), mask2);//ggg000bb000rrr00

	b = _mm_blendv_epi8(s1, s0, bmask1);//bbbbbbbb

	s0 = _mm_shuffle_epi8(s0, smask1);  //gggggbbbbbbrrrrr
	s1 = _mm_shuffle_epi8(s1, smask1);  //bb000ggg000rrr00
	g = _mm_blendv_epi8(s1, s0, bmask2);//gggggggg

	s0 = _mm_shuffle_epi8(s0, smask2);  //rrrrr00000000000
	s1 = _mm_shuffle_epi8(s1, smask3);  //00000rrr00000000
	r = _mm_blendv_epi8(s1, s0, bmask2);//rrrrrrrr
}
//BGR2Planar (uchar). Same function of OpenCV for AVX.
inline void _mm256_load_cvtepu8bgr2planar_si256(const uchar* ptr, __m256i& b, __m256i& g, __m256i& r)
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

//opencp: BGR2Planar (uchar->float). psx2 is more effective
inline void _mm256_load_cvtepu8bgr2planar_ps(const uchar* ptr, __m256& b, __m256& g, __m256& r)
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
inline void _mm256_load_cvtepu8bgr2planar_psx2(const uchar* ptr, __m256& b0, __m256& b1, __m256& g0, __m256& g1, __m256& r0, __m256& r1)
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
inline void _mm256_load_cvtepu8bgr2planar_psx4(const uchar* ptr,
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
	const __m256i mask = _mm256_setr_epi8(0, 4, 8, 1, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 5, 10, 2, 6, 12, 3, 7, 14, 0, 0, 0, 0, 0, 4, 8, 1);
	mb = _mm256_shuffle_epi8(mb, mask);
	__m256i mp = _mm256_permute2f128_si256(mb, mb, 0x11);
	mb = _mm256_blend_epi32(mb, mp, 8);

	_mm256_storeu_si256((__m256i*)dst, mb);
}

#define  _mm256_stream_ps2epu8_color  _mm256_store_ps2epu8_color
inline void _mm256_stream_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r)
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

inline void _mm256_storescalar_ps2epu8_color(void* dst, __m256 b, __m256 g, __m256 r, const int numpixel = 24)
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

inline void _mm256_store_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
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

inline void _mm256_storeu_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
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

inline void _mm256_storescalar_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r, const int numpixel = 8)
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

inline void _mm256_stream_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
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

inline void _mm256_stream_ps_color_2(void* dst, const __m256 b, const __m256 g, const __m256 r)
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

void inline _mm256_store_ps_color_v2(void* dst, const __m256 b, const __m256 g, const __m256 r)
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


inline void _mm256_storeu_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
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

inline void _mm256_store_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
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

#pragma region gray2bgr
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
#pragma endregion

//plain2bgr: plain b,g,r image to interleave rgb. SoA->AoS
inline void _mm256_cvtps_planar2bgr(const __m256 b, const __m256 g, const __m256 r, __m256& d0, __m256& d1, __m256& d2)
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

inline __m256 _mm256_abs_ps(__m256 src)
{
	return _mm256_and_ps(src, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}

inline float _mm_reduceadd_ps(__m128 src)
{
	src = _mm_hadd_ps(src, src);
	src = _mm_hadd_ps(src, src);
	return _mm_cvtss_f32(src);
}

inline float _mm256_reduceadd_ps(__m256 src)
{
	src = _mm256_hadd_ps(src, src);
	src = _mm256_hadd_ps(src, src);
	return (src.m256_f32[0] + src.m256_f32[4]);
	//__m256 rsum = _mm256_permute2f128_ps(src, src, 0 << 4 | 1);
	//src = _mm256_unpacklo_ps(src, rsum);
	//return _mm256_hadd_ps(src, src).m256_f32[0];
}

inline void _mm256_reduceadd_highlow_ps(__m256 src, float& low, float& high)
{
	src = _mm256_hadd_ps(src, src);
	src = _mm256_hadd_ps(src, src);
	low = src.m256_f32[0];
	high = src.m256_f32[4];
}
inline double _mm256_reduceadd_pd(__m256d src)
{
	src = _mm256_hadd_pd(src, src);
	return (src.m256d_f64[0] + src.m256d_f64[2]);
}

inline __m256 _mm256_div_avoidzerodiv_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_div_ps(src1, _mm256_blendv_ps(src2, _mm256_set1_ps(FLT_MIN), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0)));
}

inline __m256 _mm256_div_zerodivzero_ps(const __m256 src1, const __m256 src2)
{
	return _mm256_blendv_ps(_mm256_div_ps(src1, src2), _mm256_set1_ps(FLT_MIN), _mm256_cmp_ps(src2, _mm256_setzero_ps(), 0));
}

inline __m256 _mm256_ssd_ps(__m256 src, __m256 ref)
{
	__m256 diff = _mm256_sub_ps(src, ref);
	return _mm256_mul_ps(diff, diff);
}

inline __m256 _mm256_ssd_ps(__m256 src0, __m256 src1, __m256 src2, __m256 ref0, __m256 ref1, __m256 ref2)
{
	__m256 diff = _mm256_sub_ps(src0, ref0);
	__m256 difft = _mm256_mul_ps(diff, diff);
	diff = _mm256_sub_ps(src1, ref1);
	difft = _mm256_fmadd_ps(diff, diff, difft);
	diff = _mm256_sub_ps(src2, ref2);
	difft = _mm256_fmadd_ps(diff, diff, difft);
	return difft;
}

#pragma endregion

#pragma region compare
inline __m128i _mm_cmpgt_epu8(__m128i x, __m128i y)
{
	return _mm_andnot_si128(_mm_cmpeq_epi8(x, y), _mm_cmpeq_epi8(_mm_max_epu8(x, y), x));
}

inline __m256i _mm256_cmpgt_epu8(__m256i x, __m256i y)
{
	return _mm256_andnot_si256(_mm256_cmpeq_epi8(x, y), _mm256_cmpeq_epi8(_mm256_max_epu8(x, y), x));
}
#pragma endregion

#pragma region print
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
#pragma endregion


//broadcast
/*
__m128 xxxx = _mm_shuffle_ps(first, first, 0x00); // _MM_SHUFFLE(0, 0, 0, 0)
__m128 yyyy = _mm_shuffle_ps(first, first, 0x55); // _MM_SHUFFLE(1, 1, 1, 1)
__m128 zzzz = _mm_shuffle_ps(first, first, 0xAA); // _MM_SHUFFLE(2, 2, 2, 2)
__m128 wwww = _mm_shuffle_ps(first, first, 0xFF); // _MM_SHUFFLE(3, 3, 3, 3)
*/

inline void _mm256_storeu_auto_color(float* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_storeu_ps_color(dest, b, g, r);
}

inline void _mm256_storeu_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_storeu_ps2epu8_color(dest, b, g, r);
}

inline void _mm256_store_auto_color(float* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_store_ps_color(dest, b, g, r);
}

inline void _mm256_store_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_store_ps2epu8_color(dest, b, g, r);
}

inline void _mm256_stream_auto_color(float* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_stream_ps_color(dest, b, g, r);
}

inline void _mm256_stream_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r)
{
	_mm256_stream_ps2epu8_color(dest, b, g, r);
}


#pragma region store

inline void _mm256_store_cvtps_epu8(__m128i* dest, __m256 ms)
{
	_mm_storel_epi64(dest, _mm256_cvtps_epu8(ms));
}

inline void _mm256_storescalar_cvtps_epu8(void* dst, __m256 src, const int numpixel)
{
	uchar CV_DECL_ALIGNED(32) buffscalarstore[32];
	_mm256_store_cvtps_epu8((__m128i*)buffscalarstore, src);
	uchar* dest = (uchar*)dst;
	for (int i = 0; i < numpixel; i++)
		dest[i] = buffscalarstore[i];
}

inline void _mm256_storescalar_ps(float* dst, __m256 src, const int numpixel)
{
	float CV_DECL_ALIGNED(32) buffscalarstore[8];
	_mm256_store_ps(buffscalarstore, src);
	for (int i = 0; i < numpixel; i++)
		dst[i] = buffscalarstore[i];
}

inline void _mm256_i32scaterscalar_epu8(uchar* dest, __m256i vindex, __m256 src)
{
	__m128i v = _mm256_cvtps_epu8(src);
	dest[vindex.m256i_i32[0]] = v.m128i_u8[0];
	dest[vindex.m256i_i32[1]] = v.m128i_u8[1];
	dest[vindex.m256i_i32[2]] = v.m128i_u8[2];
	dest[vindex.m256i_i32[3]] = v.m128i_u8[3];
	dest[vindex.m256i_i32[4]] = v.m128i_u8[4];
	dest[vindex.m256i_i32[5]] = v.m128i_u8[5];
	dest[vindex.m256i_i32[6]] = v.m128i_u8[6];
	dest[vindex.m256i_i32[7]] = v.m128i_u8[7];
}

inline void _mm256_i32scaterscalar_ps(float* dest, __m256i vindex, __m256 src)
{
	dest[vindex.m256i_i32[0]] = src.m256_f32[0];
	dest[vindex.m256i_i32[1]] = src.m256_f32[1];
	dest[vindex.m256i_i32[2]] = src.m256_f32[2];
	dest[vindex.m256i_i32[3]] = src.m256_f32[3];
	dest[vindex.m256i_i32[4]] = src.m256_f32[4];
	dest[vindex.m256i_i32[5]] = src.m256_f32[5];
	dest[vindex.m256i_i32[6]] = src.m256_f32[6];
	dest[vindex.m256i_i32[7]] = src.m256_f32[7];
}

inline void _mm256_i32scaterscalar_epu8_color(uchar* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	__m128i bb = _mm256_cvtps_epu8(b);
	__m128i gb = _mm256_cvtps_epu8(g);
	__m128i rb = _mm256_cvtps_epu8(r);
	int idx = vindex.m256i_i32[0];
	dest[idx + 0] = bb.m128i_u8[0];	dest[idx + 1] = gb.m128i_u8[0];	dest[idx + 2] = rb.m128i_u8[0];
	idx = vindex.m256i_i32[1];
	dest[idx + 0] = bb.m128i_u8[1];	dest[idx + 1] = gb.m128i_u8[1];	dest[idx + 2] = rb.m128i_u8[1];
	idx = vindex.m256i_i32[2];
	dest[idx + 0] = bb.m128i_u8[2];	dest[idx + 1] = gb.m128i_u8[2];	dest[idx + 2] = rb.m128i_u8[2];
	idx = vindex.m256i_i32[3];
	dest[idx + 0] = bb.m128i_u8[3];	dest[idx + 1] = gb.m128i_u8[3];	dest[idx + 2] = rb.m128i_u8[3];
	idx = vindex.m256i_i32[4];
	dest[idx + 0] = bb.m128i_u8[4];	dest[idx + 1] = gb.m128i_u8[4];	dest[idx + 2] = rb.m128i_u8[4];
	idx = vindex.m256i_i32[5];
	dest[idx + 0] = bb.m128i_u8[5];	dest[idx + 1] = gb.m128i_u8[5];	dest[idx + 2] = rb.m128i_u8[5];
	idx = vindex.m256i_i32[6];
	dest[idx + 0] = bb.m128i_u8[6];	dest[idx + 1] = gb.m128i_u8[6];	dest[idx + 2] = rb.m128i_u8[6];
	idx = vindex.m256i_i32[7];
	dest[idx + 0] = bb.m128i_u8[7];	dest[idx + 1] = gb.m128i_u8[7];	dest[idx + 2] = rb.m128i_u8[7];
}

inline void _mm256_i32scaterscalar_ps_color(float* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	int idx = vindex.m256i_i32[0];
	dest[idx + 0] = b.m256_f32[0];	dest[idx + 1] = g.m256_f32[0];	dest[idx + 2] = r.m256_f32[0];
	idx = vindex.m256i_i32[1];
	dest[idx + 0] = b.m256_f32[1];	dest[idx + 1] = g.m256_f32[1];	dest[idx + 2] = r.m256_f32[1];
	idx = vindex.m256i_i32[2];
	dest[idx + 0] = b.m256_f32[2];	dest[idx + 1] = g.m256_f32[2];	dest[idx + 2] = r.m256_f32[2];
	idx = vindex.m256i_i32[3];
	dest[idx + 0] = b.m256_f32[3];	dest[idx + 1] = g.m256_f32[3];	dest[idx + 2] = r.m256_f32[3];
	idx = vindex.m256i_i32[4];
	dest[idx + 0] = b.m256_f32[4];	dest[idx + 1] = g.m256_f32[4];	dest[idx + 2] = r.m256_f32[4];
	idx = vindex.m256i_i32[5];
	dest[idx + 0] = b.m256_f32[5];	dest[idx + 1] = g.m256_f32[5];	dest[idx + 2] = r.m256_f32[5];
	idx = vindex.m256i_i32[6];
	dest[idx + 0] = b.m256_f32[6];	dest[idx + 1] = g.m256_f32[6];	dest[idx + 2] = r.m256_f32[6];
	idx = vindex.m256i_i32[7];
	dest[idx + 0] = b.m256_f32[7];	dest[idx + 1] = g.m256_f32[7];	dest[idx + 2] = r.m256_f32[7];
}
#pragma endregion

inline void _mm256_stream_auto(uchar* dest, __m256 ms)
{
	_mm256_store_cvtps_epu8((__m128i*)dest, ms);
}

inline void _mm256_stream_auto(float* dest, __m256 ms)
{
	_mm256_stream_ps(dest, ms);
}

inline void _mm256_storescalar_auto(uchar* dest, __m256 ms, const int numpixel)
{
	_mm256_storescalar_cvtps_epu8(dest, ms, numpixel);
}

inline void _mm256_storescalar_auto(float* dest, __m256 ms, const int numpixel)
{
	_mm256_storescalar_ps(dest, ms, numpixel);
}

inline void _mm256_storescalar_auto_color(float* dest, __m256 b, __m256 g, __m256 r, const int numpixel)
{
	_mm256_storescalar_ps_color(dest, b, g, r, numpixel);
}

inline void _mm256_storescalar_auto_color(uchar* dest, __m256 b, __m256 g, __m256 r, const int numpixel)
{
	_mm256_storescalar_ps2epu8_color(dest, b, g, r, numpixel);
}

//return 8 uchar elements
inline __m128i _mm_i32gather_epi32(const uchar* src, __m128i idx)
{
	return _mm_srli_epi32(_mm_i32gather_epi32((int*)(src - 3), idx, 1), 24);
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}


//gather bgr interleved uchar data with convert epi32->ps
inline void _mm256_i32gather_bgr_ps(const uchar* src, __m256i idx, __m256& b, __m256& g, __m256& r)
{
	__m256i v = _mm256_i32gather_epi32((int*)(src), idx, 1);
	
	b = _mm256_cvtepi32_ps(_mm256_blendv_epi8(v, _mm256_setzero_si256(), _mm256_setr_epi8(0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF)));
	g = _mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_blendv_epi8(v, _mm256_setzero_si256(), _mm256_setr_epi8(0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF)), 8));
	r = _mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_blendv_epi8(v, _mm256_setzero_si256(), _mm256_setr_epi8(0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0, 0xFF)), 16));
}

inline void _mm256_i32gather_bgr_epi32(const uchar* src, __m256i idx, __m256i& b, __m256i& g, __m256i& r)
{
	__m256i v = _mm256_i32gather_epi32((int*)(src - 3), idx, 1);
	b = _mm256_srli_epi32(v, 24);
	g = _mm256_srli_epi32(v, 25);
	r = _mm256_srli_epi32(v, 26);
}

inline __m256i _mm256_i32gather_epi32(const uchar* src, __m256i idx)
{
	return _mm256_srli_epi32(_mm256_i32gather_epi32((int*)(src - 3), idx, 1), 24);
	//return _mm256_setr_epi32(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]]);
}

inline __m128 _mm_i32gather_ps(const uchar* src, __m128i idx)
{
	return _mm_cvtepi32_ps(_mm_srli_epi32(_mm_i32gather_epi32((int*)(src - 3), idx, 1), 24));
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

inline __m128i _mm_i32gather_epu8(const uchar* src, __m128i idx)
{
	return _mm_cvtepi32_epi8(_mm_srli_epi32(_mm_i32gather_epi32((int*)(src - 3), idx, 1), 24));
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

inline __m128i _mm256_i32gather_epu8(const uchar* src, __m256i idx)
{
	return _mm256_cvtepi32_epu8(_mm256_srli_epi32(_mm256_i32gather_epi32((int*)(src - 3), idx, 1), 24));
	//return _mm_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]], 0, 0, 0, 0, 0, 0, 0, 0);
}

inline __m256 _mm256_i32gather_auto(float* src, __m256i idx)
{
	return _mm256_i32gather_ps(src, idx, 4);
}

inline __m256 _mm256_i32gather_auto(uchar* src, __m256i idx)
{
	return _mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_i32gather_epi32((int*)(src - 3), idx, 1), 24));
}

inline __m256 _mm256_i32gatherset_auto(uchar* src, __m256i idx)
{
	return _mm256_cvtepi32_ps(_mm256_setr_epi8(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]],
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

inline __m256 _mm256_i32gatherset_auto(float* src, __m256i idx)
{
	return _mm256_setr_ps(src[idx.m256i_i32[0]], src[idx.m256i_i32[1]], src[idx.m256i_i32[2]], src[idx.m256i_i32[3]], src[idx.m256i_i32[4]], src[idx.m256i_i32[5]], src[idx.m256i_i32[6]], src[idx.m256i_i32[7]]);
}

inline void _mm256_i32scaterscalar_auto(uchar* dest, __m256i vindex, __m256 src)
{
	_mm256_i32scaterscalar_epu8(dest, vindex, src);
}

inline void _mm256_i32scaterscalar_auto(float* dest, __m256i vindex, __m256 src)
{
	_mm256_i32scaterscalar_ps(dest, vindex, src);
}

inline void _mm256_i32scaterscalar_auto_color(uchar* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	_mm256_i32scaterscalar_epu8_color(dest, vindex, b, g, r);
}

inline void _mm256_i32scaterscalar_auto_color(float* dest, __m256i vindex, __m256 b, __m256 g, __m256 r)
{
	_mm256_i32scaterscalar_ps_color(dest, vindex, b, g, r);
}


inline __m256 _mm256_load_reverse_ps(const float* src)
{
	__m256 ret = _mm256_load_ps(src);
	ret = _mm256_shuffle_ps(ret, ret, _MM_SHUFFLE(0, 1, 2, 3));
	ret = _mm256_permute2f128_ps(ret, ret, 1);
	return ret;
}

inline __m256 _mm256_loadu_reverse_ps(const float* src)
{
	__m256 ret = _mm256_loadu_ps(src);
	ret = _mm256_shuffle_ps(ret, ret, _MM_SHUFFLE(0, 1, 2, 3));
	ret = _mm256_permute2f128_ps(ret, ret, 1);
	return ret;
}

inline __m256 _mm256_load_auto(const uchar* src)
{
	return _mm256_load_epu8cvtps((const __m128i*)src);
}

inline __m256 _mm256_loadu_auto(const uchar* src)
{
	return _mm256_load_epu8cvtps((const __m128i*)src);
}

inline __m256 _mm256_load_auto(const float* src)
{
	return _mm256_load_ps(src);
}

inline __m256 _mm256_loadu_auto(const float* src)
{
	return _mm256_loadu_ps(src);
}

inline __m256d _mm256_load_auto(const double* src)
{
	return _mm256_load_pd(src);
}

inline __m256d _mm256_loadu_auto(const double* src)
{
	return _mm256_loadu_pd(src);
}


inline void _mm256_store_auto(uchar* dest, __m256 src)
{
	_mm256_store_cvtps_epu8((__m128i*)dest, src);
}

inline void _mm256_store_auto(float* dest, __m256 src)
{
	_mm256_store_ps(dest, src);
}

inline void _mm256_storeu_auto(uchar* dest, __m256 src)
{
	_mm256_store_cvtps_epu8((__m128i*)dest, src);
}

inline void _mm256_storeu_auto(float* dest, __m256 src)
{
	_mm256_storeu_ps(dest, src);
}

inline void _mm256_storeu_auto(double* dest, __m256d src)
{
	_mm256_storeu_pd(dest, src);
}




inline __m256i _mm256_alphablend_epu8(__m256i a, __m256i b, __m256i ma)
{
	__m256i a2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a));
	__m256i b2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b));
	__m128i d1 = _mm256_cvtepi16_epu8(_mm256_add_epi16(b2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(a2, b2))));

	a2 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(a));
	b2 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(b));
	__m128i d2 = _mm256_cvtepi16_epu8(_mm256_add_epi16(b2, _mm256_mulhrs_epi16(ma, _mm256_sub_epi16(a2, b2))));
	return _mm256_set_m128i(d2, d1);
}