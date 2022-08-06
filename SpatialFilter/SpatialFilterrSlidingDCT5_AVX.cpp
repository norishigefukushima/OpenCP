#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{

#pragma region SlidingDCT5_AVX_32F

	void SpatialFilterSlidingDCT5_AVX_32F::allocBuffer()
	{
		const int simdUnrollSize = 8;
		const int internalWidth = get_simd_ceil(imgSize.width, simdUnrollSize);
		const int internalHeight = get_simd_ceil(imgSize.height, simdUnrollSize) + simdUnrollSize;
		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			inter.create(Size(internalWidth, internalHeight), CV_16F);
		}
		else
		{
			inter.create(Size(internalWidth, internalHeight), CV_32F);
		}

		_mm_free(fn_hfilter);
		_mm_free(buffVFilter);
		fn_hfilter = (__m256*)_mm_malloc((internalWidth + get_simd_ceil(2 * radius + 3, simdUnrollSize)) * sizeof(__m256), AVX_ALIGNMENT);
		buffVFilter = (__m256*)_mm_malloc(int((2 * gf_order + 1) * internalWidth / simdUnrollSize) * sizeof(__m256), AVX_ALIGNMENT);
	}

	int SpatialFilterSlidingDCT5_AVX_32F::getRadius(const double sigma, const int order)
	{
		cv::AutoBuffer<double> spect(order + 1);
		if (radius == 0)
		{
			return argminR_BruteForce_DCT(sigma, order, 5, spect, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
		}
		else
		{
			return radius;
		}
	}

	void SpatialFilterSlidingDCT5_AVX_32F::computeRadius(const int rad, const bool isOptimize)
	{
		if (gf_order == 0)
		{
			this->radius = rad;
			return;
		}

		cv::AutoBuffer<double> Gk(gf_order + 1);
		if (rad == 0)
		{
			this->radius = argminR_BruteForce_DCT(sigma, gf_order, 5, Gk, isOptimize);
		}
		else
		{
			this->radius = rad;
		}

		if (isOptimize)optimizeSpectrum(sigma, gf_order, radius, 5, Gk, 0);
		else computeSpectrumGaussianClosedForm(sigma, gf_order, radius, 5, Gk);

		const double omega = CV_2PI / (2.0 * radius + 1.0);
		const int GCnSize = (gf_order + 0) * (radius + 1);//for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		_mm_free(GCn);
		GCn = (float*)_mm_malloc(GCnSize * sizeof(float), AVX_ALIGN);
		AutoBuffer<double> GCn64F(GCnSize);

		double totalInv = 0.0;
		generateCosKernel(GCn64F, totalInv, 5, Gk, radius, gf_order);
		for (int i = 0; i < GCnSize; i++)
		{
			GCn[i] = float(GCn64F[i] * totalInv);
			if (abs(GCn[i]) < FLT_MIN)
			{
				GCn[i] = 0.f;
			}
		}
		G0 = float(Gk[0] * totalInv);
#ifdef PLOT_DCT_KERNEL	
		plotDCTKernel("DCT-5", false, GCn, radius, gf_order, G0, sigma);
#endif

		_mm_free(shift);
		shift = (float*)_mm_malloc((2 * gf_order) * sizeof(float), AVX_ALIGN);
		for (int k = 1; k <= gf_order; ++k)
		{
			const double C1 = cos(k * omega * 1);
			const double CR = cos(k * omega * radius);
			shift[2 * (k - 1) + 0] = float(C1 * 2.0);//2*C1
			shift[2 * (k - 1) + 1] = float(CR * Gk[k] * totalInv);//CR
			//shift[2 * (k - 1) + 0] = float(C1 * 2.0)< FLT_MIN ? 0.f: float(C1 * 2.0);//2*C1
			//shift[2 * (k - 1) + 1] = float(CR * Gk[k] * totalInv)< FLT_MIN ? 0.f: float(CR * Gk[k] * totalInv);//CR
		}
	}

	SpatialFilterSlidingDCT5_AVX_32F::SpatialFilterSlidingDCT5_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		//cout << "init sliding DCT5 AVX 32F" << endl;
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT5_AVX;
		this->gf_order = order;
		this->sigma = sigma;
		this->dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;
		computeRadius(radius, true);

		this->imgSize = imgSize;
		allocBuffer();
	}

	SpatialFilterSlidingDCT5_AVX_32F::SpatialFilterSlidingDCT5_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule, const SpatialKernel skernel)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT5_AVX;
		this->schedule = schedule;
		this->depth = CV_32F;
		this->dest_depth = dest_depth;
		this->dct_coeff_method = method;
	}

	SpatialFilterSlidingDCT5_AVX_32F::~SpatialFilterSlidingDCT5_AVX_32F()
	{
		_mm_free(GCn);
		_mm_free(shift);
		_mm_free(fn_hfilter);
		_mm_free(buffVFilter);
	}

	inline void mm256_transpose2_inplace(
		__m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4, __m256& s5, __m256& s6, __m256& s7)
	{
		__m256 temp0 = _mm256_unpacklo_ps(s0, s1);
		__m256 temp1 = _mm256_unpacklo_ps(s2, s3);
		__m256 temp2 = _mm256_unpackhi_ps(s0, s1);
		__m256 temp3 = _mm256_unpacklo_ps(s4, s5);
		s0 = _mm256_unpacklo_ps(s6, s7);

		__m256 temp4 = _mm256_shuffle_ps(temp0, temp1, 0x4E);
		s1 = _mm256_blend_ps(temp0, temp4, 0xCC);//blend
		temp0 = _mm256_shuffle_ps(temp3, s0, 0x4E);

		__m256 temp5 = _mm256_unpackhi_ps(s2, s3);
		s2 = _mm256_blend_ps(temp3, temp0, 0xCC);//blend
		s3 = _mm256_blend_ps(temp4, temp1, 0xCC);//blend
		temp4 = _mm256_permute2f128_ps(s1, s2, 0x20);//dest

		temp3 = _mm256_unpackhi_ps(s4, s5);
		s4 = _mm256_blend_ps(temp0, s0, 0xCC);//blend
		s0 = temp4;//out s0
		temp4 = _mm256_unpackhi_ps(s6, s7);
		s7 = _mm256_permute2f128_ps(s3, s4, 0x20);//dest

		s5 = _mm256_shuffle_ps(temp2, temp5, 0x4E);//4E
		s6 = _mm256_blend_ps(s5, temp5, 0xCC);//blend
		temp5 = _mm256_shuffle_ps(temp3, temp4, 0x4E);
		temp2 = _mm256_blend_ps(temp2, s5, 0xCC);//blend
		temp3 = _mm256_blend_ps(temp3, temp5, 0xCC);//blend
		temp0 = _mm256_permute2f128_ps(temp2, temp3, 0x20);//dest	

		temp4 = _mm256_shuffle_ps(temp5, temp4, 0xE4);
		temp5 = _mm256_permute2f128_ps(s6, temp4, 0x20);//dest	
		temp1 = _mm256_permute2f128_ps(s1, s2, 0x31);//dest
		s2 = temp0;
		s5 = _mm256_permute2f128_ps(s3, s4, 0x31);//dest
		s3 = temp5;
		s4 = temp1;
		temp1 = s6;
		s6 = _mm256_permute2f128_ps(temp2, temp3, 0x31);//dest
		s1 = s7;
		s7 = _mm256_permute2f128_ps(temp1, temp4, 0x31);//dest
	}

	void SpatialFilterSlidingDCT5_AVX_32F::interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad)
	{
		const int pad = inter.cols - imgSize.width;
		const int R = radius + 2;
		const int Rm = radius + 1;//left side does not access r+2

		int length = src.cols - right - left + Rm + R;
		bool isLpad = false;
		bool isRpad = false;
		int xst = 0;
		int xed = src.cols;
		if (left < Rm)
		{
			isLpad = true;
			length -= Rm;
		}
		else
		{
			xst = left - Rm;
		}
		if (src.cols - right + R > src.cols)
		{
			isRpad = true;
			length -= R;
		}
		else
		{
			xed = src.cols - right;
		}
		const int mainloop_simdsize = get_simd_floor(length, 8) / 8;
		const int L = mainloop_simdsize * 8;
		const int rem = length - L;
		if (xst + L > imgSize.width) xst -= (xst + L - imgSize.width);

		__m256* dest = &this->fn_hfilter[Rm];
		__m256* buffPtr = dest + xst;
		const int step0 = 0 * imgSize.width;
		const int step1 = 1 * imgSize.width;
		const int step2 = 2 * imgSize.width;
		const int step3 = 3 * imgSize.width;
		const int step4 = 4 * imgSize.width;
		const int step5 = 5 * imgSize.width;
		const int step6 = 6 * imgSize.width;
		const int step7 = 7 * imgSize.width;

		if (vpad == 0)
		{
			const __m256i gidx = _mm256_setr_epi32(step0, step1, step2, step3, step4, step5, step6, step7);
			if (src.depth() == CV_32F)
			{
				const float* ptr = src.ptr<float>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					buffPtr[0] = _mm256_loadu_ps(ptr + step0);
					buffPtr[1] = _mm256_loadu_ps(ptr + step1);
					buffPtr[2] = _mm256_loadu_ps(ptr + step2);
					buffPtr[3] = _mm256_loadu_ps(ptr + step3);
					buffPtr[4] = _mm256_loadu_ps(ptr + step4);
					buffPtr[5] = _mm256_loadu_ps(ptr + step5);
					buffPtr[6] = _mm256_loadu_ps(ptr + step6);
					buffPtr[7] = _mm256_loadu_ps(ptr + step7);
					//_MM256_TRANSPOSE8INPLACE_PS(buffPtr[0], buffPtr[1], buffPtr[2], buffPtr[3], buffPtr[4], buffPtr[5], buffPtr[6], buffPtr[7]);
					//_MM256_TRANSPOSEBLEND8_PS(buffPtr[0], buffPtr[1], buffPtr[2], buffPtr[3], buffPtr[4], buffPtr[5], buffPtr[6], buffPtr[7]);
					mm256_transpose2_inplace(buffPtr[0], buffPtr[1], buffPtr[2], buffPtr[3], buffPtr[4], buffPtr[5], buffPtr[6], buffPtr[7]);
					//_mm256_transpose8_ps(buffPtr);
					ptr += 8;
					buffPtr += 8;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_i32gather_ps(ptr, gidx, sizeof(float));
					ptr++;
					buffPtr++;
				}
			}
			else if (src.depth() == CV_8U)
			{
				const uchar* ptr = src.ptr<uchar>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					buffPtr[0] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step0)));
					buffPtr[1] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step1)));
					buffPtr[2] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step2)));
					buffPtr[3] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step3)));
					buffPtr[4] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step4)));
					buffPtr[5] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step5)));
					buffPtr[6] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step6)));
					buffPtr[7] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + step7)));
					_mm256_transpose8_ps(buffPtr);
					ptr += 8;
					buffPtr += 8;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(ptr, gidx));
					ptr++;
					buffPtr++;
				}
			}
		}
		else
		{
			int step[8];
			for (int i = 0; i < vpad; i++)step[i] = imgSize.width * i;
			for (int i = vpad; i < 8; i++)step[i] = step[vpad - 1];
			const __m256i gidx = _mm256_setr_epi32(step[0], step[1], step[2], step[3], step[4], step[5], step[6], step[7]);
			if (src.depth() == CV_32F)
			{
				const float* ptr = src.ptr<float>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					for (int i = 0; i < vpad; i++)
					{
						buffPtr[i] = _mm256_loadu_ps(ptr + imgSize.width * i);
					}
					_mm256_transpose8_ps(buffPtr);
					ptr += 8;
					buffPtr += 8;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_i32gather_ps(ptr, gidx, sizeof(float));
					ptr++;
					buffPtr++;
				}
			}
			else if (src.depth() == CV_8U)
			{
				const uchar* ptr = src.ptr<uchar>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					for (int i = 0; i < vpad; i++)
					{
						buffPtr[i] = _mm256_cvtepu8_ps(_mm_loadl_epi64((const __m128i*)(ptr + imgSize.width * i)));
					}
					_mm256_transpose8_ps(buffPtr);
					ptr += 8;
					buffPtr += 8;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_cvtepu8_ps(_mm256_i32gather_epu8(ptr, gidx));
					ptr++;
					buffPtr++;
				}
			}
		}

		if (isLpad && isRpad)
		{
			switch (borderType)
			{
			case cv::BORDER_REPLICATE:
			{
				for (int x = 1; x < Rm; ++x)
				{
					dest[-x] = dest[0];
				}
				for (int x = 1; x < R; ++x)
				{
					dest[src.cols - 1 + x] = dest[src.cols - 1];
				}
			}
			break;

			case cv::BORDER_REFLECT:
			{
				for (int x = 1; x < Rm; ++x)
				{
					dest[-x] = dest[x - 1];
				}
				for (int x = 1; x < R; ++x)
				{
					dest[src.cols - 1 + x] = dest[src.cols - 1 - x + 1];
				}
			}
			break;

			default:
			case cv::BORDER_REFLECT101:
			{
				for (int x = 1; x < Rm; ++x)
				{
					dest[-x] = dest[x];
				}
				for (int x = 1; x < R; ++x)
				{
					dest[src.cols - 1 + x] = dest[src.cols - 1 - x];
				}
			}
			break;
			}
		}
		else
		{
			if (isLpad)
			{
				switch (borderType)
				{
				case cv::BORDER_REPLICATE:
				{
					for (int x = 1; x < Rm; ++x)
					{
						dest[-x] = dest[0];
					}
				}
				break;

				case cv::BORDER_REFLECT:
				{
					for (int x = 1; x < Rm; ++x)
					{
						dest[-x] = dest[x - 1];
					}
				}
				break;

				default:
				case cv::BORDER_REFLECT101:
				{
					for (int x = 1; x < Rm; ++x)
					{
						dest[-x] = dest[x];
					}
				}
				break;
				}
			}
			if (isRpad)
			{
				switch (borderType)
				{
				case cv::BORDER_REPLICATE:
				{
					for (int x = 1; x < R; ++x)
					{
						dest[src.cols - 1 + x] = dest[src.cols - 1];
					}
				}
				break;

				case cv::BORDER_REFLECT:
				{
					for (int x = 1; x < R; ++x)
					{
						dest[src.cols - 1 + x] = dest[src.cols - 1 - x + 1];
					}
				}
				break;

				default:
				case cv::BORDER_REFLECT101:
				{
					for (int x = 1; x < R; ++x)
					{
						dest[src.cols - 1 + x] = dest[src.cols - 1 - x];
					}
				}
				break;
				}
			}
		}
	}

	//Naive DCT Convolution: O(KR)
	void SpatialFilterSlidingDCT5_AVX_32F::horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		__m256 total[8];
		__m256 F0;
		AutoBuffer<__m256> Z(order);
		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 8)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			for (int x = xstart; x < (mainloop_simdsize + 1) * 8; x += simdUnrollSize)
			{
				for (int j = 0; j < 8; j++)
				{
					// 1) initilization of Z (1 <= n <= radius)
					for (int k = 0; k <= order; ++k)Z[k] = _mm256_setzero_ps();
					F0 = _mm256_setzero_ps();

					for (int n = radius; n >= 1; --n)
					{
						const __m256 sumA = _mm256_add_ps(fn_hfilter[x + j - n], fn_hfilter[x + j + n]);
						F0 = _mm256_add_ps(F0, sumA);
						float* Cn_ = GCn + order * n;
						for (int k = 0; k < order; ++k)
						{
							__m256 Cnk = _mm256_set1_ps(Cn_[k]);
							Z[k] = _mm256_fmadd_ps(Cnk, sumA, Z[k]);
						}
					}

					// 1) initilization of Z (n=0) adding small->large
					F0 = _mm256_add_ps(F0, fn_hfilter[x + j]);
					for (int k = 0; k < order; ++k)
					{
						__m256 Cnk = _mm256_set1_ps(GCn[k]);
						Z[k] = _mm256_fmadd_ps(Cnk, fn_hfilter[x + j], Z[k]);
					}

					total[j] = _mm256_setzero_ps();
					for (int k = order - 1; k >= 0; --k)
					{
						total[j] = _mm256_add_ps(total[j], Z[k]);
					}
					total[j] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[j]);
				}
				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dst.cols);
				dstPtr += 8;
			}
		}
	}

	//32F hfilter O(K) (Default)
	template<int order>
	void SpatialFilterSlidingDCT5_AVX_32F::horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVEC C1_2[order];
		SETVEC CR_g[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int k = 0; k < order; ++k)
		{
			C1_2[k] = _MM256_SETLUT_VEC(shift[k * 2 + 0]);
			CR_g[k] = _MM256_SETLUT_VEC(shift[k * 2 + 1]);
		}

		__m256 total[8];
		__m256 F0;
		__m256 Zp[order];
		__m256 Zc[order];
		__m256 delta_inner;//f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256 dc;//f(x+R+1)-f(x-R)
		__m256 dp;//f(x+R)-f(x-R-1)

		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 8)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			F0 = fn_hfilter[xstart];
			for (int k = 0; k < order; ++k)
			{
				__m256 Cnk = _mm256_set1_ps(GCn[k]);
				Zp[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 1]);
			}

			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[xstart + 0 - n], fn_hfilter[xstart + 0 + n]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[xstart + 1 - n], fn_hfilter[xstart + 1 + n]);
				F0 = _mm256_add_ps(F0, sumA);
				float* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				dc = _mm256_sub_ps(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);

				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);
				//	F0 = _mm256_add_ps(F0, dp);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			__m256* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];//f(x+R+1)
			__m256* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius + 0];//f(x-R)
			for (int x = 0; x < mainloop_simdsize; ++x)
			{
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[0] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);
				//	F0 = _mm256_add_ps(F0, dp);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}
		}
	}

	//non template version 32F hfilter O(K) (Default)
	void SpatialFilterSlidingDCT5_AVX_32F::horizontalFilteringInnerXKn(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;//SIMDSIZE

		AutoBuffer<SETVEC> C1_2(order);
		AutoBuffer<SETVEC> CR_g(order);
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int k = 0; k < order; ++k)
		{
			C1_2[k] = _MM256_SETLUT_VEC(shift[k * 2 + 0]);
			CR_g[k] = _MM256_SETLUT_VEC(shift[k * 2 + 1]);
		}

		__m256 total[8];
		__m256 F0;
		AutoBuffer<__m256> Zp(order);
		AutoBuffer<__m256> Zc(order);
		__m256 delta_inner;//f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256 dc;//f(x+R+1)-f(x-R)
		__m256 dp;//f(x+R)-f(x-R-1)

		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 8)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			F0 = fn_hfilter[xstart];
			for (int k = 0; k < order; ++k)
			{
				__m256 Cnk = _mm256_set1_ps(GCn[k]);
				Zp[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 1]);
			}

			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[xstart + 0 - n], fn_hfilter[xstart + 0 + n]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[xstart + 1 - n], fn_hfilter[xstart + 1 + n]);
				F0 = _mm256_add_ps(F0, sumA);
				float* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				dc = _mm256_sub_ps(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);

				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);
				//	F0 = _mm256_add_ps(F0, dp);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			__m256* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];//f(x+R+1)
			__m256* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius + 0];//f(x-R)
			for (int x = 0; x < mainloop_simdsize; ++x)
			{
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[0] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);
				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);
				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);
				//	F0 = _mm256_add_ps(F0, dp);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}
		}
	}

	//32F vfilter O(K) Y-XLoop (Default)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
		const int dwidth = dst.cols;//dst.cols

		const int xstart = left;
		const int xlength = dst.cols - left - right;
		const int xlengthC = get_simd_ceil(xlength, simdUnrollSize);
		const int xend = xstart + xlengthC;//xendC
		const int rem = xlength - get_simd_floor(xlength, simdUnrollSize);
		const int simdWidth = xlengthC / simdUnrollSize;
		const int simdend = (rem == 0) ? simdWidth : simdWidth - 1;

		__m256i storemask = _mm256_setzero_si256();
		for (int i = 0; i < rem; i++)
		{
			((int*)&storemask)[i] = -1;
		}

		const int ylength = dst.rows - (top + bottom);
		const bool isEven = ((ylength) % 2 == 0);
		const int hend = ylength + ((isEven) ? 0 : -1);

		const float* srcPtr = src.ptr<float>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		SETVEC C1_2[order];
		SETVEC CR_g[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA;//dc-dp: f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256 deltaB;//dn-dc
		__m256 dp;//f(x+R)-f(x-R-1)
		__m256 dc;//f(x+R+1)-f(x-R)
		__m256 dn;//f(x+R+2)-f(x-R+1)

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (top + 1) + x]);

			for (int i = order - 1; i >= 0; i--)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
			*ws++ = pA;
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		__m256 mGCn[order];
		for (int r = 1; r <= radius; ++r)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + 0 - r, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + 1 - r, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[swidth * (top + 0 + r) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + 1 + r) + xstart]);

			ws = buffVFilter;

			for (int i = order - 1; i >= 0; i--) mGCn[i] = _mm256_set1_ps(GCn[order * r + i]);

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				const __m256 pB = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				for (int i = order - 1; i >= 0; i--)
				{
					*ws++ = _mm256_fmadd_ps(pA, mGCn[i], *ws);
					*ws++ = _mm256_fmadd_ps(pB, mGCn[i], *ws);
				}
				*ws++ = _mm256_add_ps(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 1) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 temp = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_ps(temp, dn);
			}

			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storeu_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}

			if (rem != 0)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_maskstore_auto(dstPtr2, storemask, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				//_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_maskstore_auto(dstPtr2 + dwidth, storemask, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
				//_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
				*ws++ = _mm256_add_ps(F0, dn);
			}

			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;

			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
		}
	}

	template<int order, typename destT>
	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYKReuseAll(const cv::Mat& src, cv::Mat& dst, int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
		const int dwidth = dst.cols;//dst.cols

		const int xstart = left;
		const int xlength = dst.cols - left - right;
		const int xlengthC = get_simd_ceil(xlength, simdUnrollSize);
		const int xend = xstart + xlengthC;//xendC
		const int rem = xlength - get_simd_floor(xlength, simdUnrollSize);
		const int simdWidth = xlengthC / simdUnrollSize;
		const int simdend = (rem == 0) ? simdWidth : simdWidth - 1;

		const int ylength = dst.rows - (top + bottom);
		const bool isEven = ((ylength) % 2 == 0);
		const int hend = ylength + ((isEven) ? 0 : -1);

		const float* srcPtr = src.ptr<float>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		SETVEC C1_2[order];
		SETVEC CR_g[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA;//dc-dp: f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256 deltaB;//dn-dc
		//__m256 dp;//f(x+R)-f(x-R-1)
		__m256 dc;//f(x+R+1)-f(x-R)
		__m256 dn;//f(x+R+2)-f(x-R+1)
		AutoBuffer<__m256> DP(simdend + 1);

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (top + 1) + x]);

			for (int i = order - 1; i >= 0; i--)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
			*ws++ = pA;
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		for (int r = 1; r <= radius; ++r)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + 0 - r, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + 1 - r, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[swidth * (top + 0 + r) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + 1 + r) + xstart]);

			ws = buffVFilter;

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				const __m256 pB = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				for (int i = order - 1; i >= 0; i--)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[order * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[order * r + i]), *ws);
				}
				*ws++ = _mm256_add_ps(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 1) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				DP[x] = dn;
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				DP[simdend + 1] = dn;
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 temp = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}

			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			//float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			//float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			destT* dstPtr3 = dstPtr + dwidth;

			for (int x = 0; x < simdend; ++x)
			{
				//dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, DP[x]);
				DP[x] = dn;
				deltaB = _mm256_sub_ps(dn, dc);
				//pAP += 8;
				pBP += 8;
				pCP += 8;
				//pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storeu_auto(dstPtr3, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
				dstPtr3 += 8;
			}

			if (rem != 0)
			{
				//dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, DP[simdend + 1]);
				DP[simdend + 1] = dn;
				deltaB = _mm256_sub_ps(dn, dc);
				//pAP += 8;
				pBP += 8;
				pCP += 8;
				//pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr3, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
				*ws++ = _mm256_add_ps(F0, dn);
			}

			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;

			//float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			//float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{

				//dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, DP[x]);
				DP[x] = dn;
				deltaB = _mm256_sub_ps(dn, dc);
				//pAP += 8;
				pBP += 8;
				pCP += 8;
				//pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				//dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, DP[simdend + 1]);
				DP[simdend + 1] = dn;
				deltaB = _mm256_sub_ps(dn, dc);
				//pAP += 8;
				pBP += 8;
				pCP += 8;
				//pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
		}
	}
	//non template version 32F vfilter O(K) Y-XLoop (Default)
	template<typename destT>
	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYKn(const cv::Mat& src, cv::Mat& dst, const int order, int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
		const int dwidth = dst.cols;//dst.cols

		const int xstart = left;
		const int xlength = dst.cols - left - right;
		const int xlengthC = get_simd_ceil(xlength, simdUnrollSize);
		const int xend = xstart + xlengthC;//xendC
		const int rem = xlength - get_simd_floor(xlength, simdUnrollSize);
		const int simdWidth = xlengthC / simdUnrollSize;
		const int simdend = (rem == 0) ? simdWidth : simdWidth - 1;

		const int ylength = dst.rows - (top + bottom);
		const bool isEven = ((ylength) % 2 == 0);
		const int hend = ylength + ((isEven) ? 0 : -1);

		const float* srcPtr = src.ptr<float>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		AutoBuffer<SETVEC> C1_2(order);
		AutoBuffer<SETVEC> CR_g(order);
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA;//dc-dp: f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256 deltaB;//dn-dc
		__m256 dp;//f(x+R)-f(x-R-1)
		__m256 dc;//f(x+R+1)-f(x-R)
		__m256 dn;//f(x+R+2)-f(x-R+1)

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (top + 1) + x]);

			for (int i = order - 1; i >= 0; i--)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
			*ws++ = pA;
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		AutoBuffer<__m256> mGCn(order);
		for (int r = 1; r <= radius; ++r)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + 0 - r, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + 1 - r, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[swidth * (top + 0 + r) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + 1 + r) + xstart]);

			ws = buffVFilter;

			for (int i = order - 1; i >= 0; i--) mGCn[i] = _mm256_set1_ps(GCn[order * r + i]);

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				const __m256 pB = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				for (int i = order - 1; i >= 0; i--)
				{
					*ws++ = _mm256_fmadd_ps(pA, mGCn[i], *ws);
					*ws++ = _mm256_fmadd_ps(pB, mGCn[i], *ws);
				}
				*ws++ = _mm256_add_ps(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 1) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 temp = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_ps(temp, dn);
			}

			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storeu_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}

			if (rem != 0)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
				*ws++ = _mm256_add_ps(F0, dn);
			}

			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;

			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
		}
	}
	//32F vfilter O(K) X-YLoop (tend to cause cache slashing)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYK_XYLoop(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
		const int dwidth = dst.cols;//dst.cols

		const int xstart = left;
		const int xend = get_simd_ceil(swidth - left - right, simdUnrollSize) + xstart;
		//const int simdWidth = (xend - xstart) / simdUnrollSize;

		const int yst = get_loop_end(top + 2, radius + 1, 2);
		const int yed = get_simd_floor(imgSize.height - radius - 1 - yst, 2) + yst;
		const int yend = get_simd_ceil(imgSize.height - (top + bottom), 2) + top - 2 * ((imgSize.height) % 2);
		const bool isylast = (imgSize.height % 2 == 1);
		//print_debug4(height, radius, top, bottom);
		//print_debug4(yst, yed, yend, height - 2 - radius);

		SETVEC C1_2[order];
		SETVEC CR_g[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 total;
		__m256 F0;
		__m256 delta_inner;
		__m256 Zp[order];
		__m256 Zc[order];
		__m256 dp, dc;

		const int xrem = get_loop_end(xstart, xend, simdUnrollSize) - dwidth;
		const int xstop = (xrem > 0) ? xend - simdUnrollSize : xend;
		const int xpad = simdUnrollSize - xrem;

		for (int x = xstart; x < xstop; x += 8)
		{
			const int ystart = top;
			destT* dstPtr = dst.ptr<destT>(ystart, x);
			const float* srcPtr = src.ptr<float>(0, x);

			// 1) initilization of Z0 and Z1 (n=radius)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - radius, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + radius)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius)));
				F0 = sumA;
				float* Cn_ = GCn + order * radius;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_mul_ps(Cnk, sumA);
					Zc[k] = _mm256_mul_ps(Cnk, sumB);
				}
			}
			// 1) initilization of Z0 and Z1 (1<=n<radius)
			for (int n = radius - 1; n >= 1; --n)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + n)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + n)));
				F0 = _mm256_add_ps(F0, sumA);
				float* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}
			// 1) initilization of Z0 and Z1 (n=0)
			{
				const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (ystart + 0)]);
				const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (ystart + 1)]);
				F0 = _mm256_add_ps(F0, pA);
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(GCn[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, pA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, pB, Zc[k]);
				}
			}

			if (top < radius)
			{
				// 2) initial output computing for y=0,1
				{
					//F0 is already computed by initilization
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - radius, swidth, borderType)));
					total = Zp[order - 1];
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;
				}

				// 3) main loop
				//top part
				for (int y = top + 2; y < yst; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 1 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y + 0 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 2 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y + 1 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;
				}

				//mid part
				float* s0 = (float*)(srcPtr + swidth * (yst + 0 - radius));
				float* s1 = (float*)(srcPtr + swidth * (yst + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (yst + 1 - radius));
				float* s3 = (float*)(srcPtr + swidth * (yst + 2 + radius));
				for (int y = yst; y < yed; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}

				//bottom part
				for (int y = yed; y < yend; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 0 - radius)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 2 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 1 - radius)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;
				}
				if (isylast)
				{
					const int y = yed - 2;

					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 0 - radius)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
				}
			}
			else
			{
				// 2) initial output computing for y=0
				float* s0 = (float*)(srcPtr + swidth * (top + 0 - radius));
				float* s1 = (float*)(srcPtr + swidth * (top + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (top + 1 - radius));
				float* s3 = (float*)(srcPtr + swidth * (top + 2 + radius));
				{
					//F0 is already computed by initilization
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[order - 1];
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;
				}

				// 3) main loop
				s0 += 2 * swidth;
				s1 += 2 * swidth;
				s2 += 2 * swidth;
				s3 += 2 * swidth;
				for (int y = top + 2; y < yend; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}
			}
		}

		//last loop
		/*
		if (xrem > 0)
		{
			const int x = xstop;
			const int ystart = top;

			destT* dstPtr = dst.ptr<destT>(ystart, x);
			const float* srcPtr = src.ptr<float>(0, x);

			// 1) initilization of Z0 and Z1 (n=radius)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - radius, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + radius)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius)));
				F0 = sumA;
				float* Cn_ = GCn + order * radius;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_mul_ps(Cnk, sumA);
					Zc[k] = _mm256_mul_ps(Cnk, sumB);
				}
			}
			// 1) initilization of Z0 and Z1 (1<=n<radius)
			for (int n = radius - 1; n >= 1; --n)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + n)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + n)));
				F0 = _mm256_add_ps(F0, sumA);
				float* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}
			// 1) initilization of Z0 and Z1 (n=0)
			{
				const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (ystart + 0)]);
				const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (ystart + 1)]);
				F0 = _mm256_add_ps(F0, pA);
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(GCn[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, pA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, pB, Zc[k]);
				}
			}

			if (top < radius)
			{
				// 2) initial output computing for y=0,1
				{
					//F0 is already computed by initilization
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - radius, swidth, borderType)));
					total = Zp[order - 1];
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}

					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;
				}

				// 3) main loop
				//top part
				for (int y = top + 2; y < yst; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 1 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y + 0 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 2 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y + 1 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;
				}

				//mid part
				float* s0 = (float*)(srcPtr + swidth * (yst + 0 - radius));
				float* s1 = (float*)(srcPtr + swidth * (yst + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (yst + 1 - radius));
				float* s3 = (float*)(srcPtr + swidth * (yst + 2 + radius));
				for (int y = yst; y < yed; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}

				//bottom part
				for (int y = yed; y < yend; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 0 - radius)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 2 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 1 - radius)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;
				}

				if (isylast)
				{
					int y = yend - 2;
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 0 - radius)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;
				}
			}
			else
			{
				// 2) initial output computing for y=0
				float* s0 = (float*)(srcPtr + swidth * (top + 0 - radius));
				float* s1 = (float*)(srcPtr + swidth * (top + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (top + 1 - radius));
				float* s3 = (float*)(srcPtr + swidth * (top + 2 + radius));
				{
					//F0 is already computed by initilization
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[order - 1];
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;
				}

				// 3) main loop
				s0 += 2 * swidth;
				s1 += 2 * swidth;
				s2 += 2 * swidth;
				s3 += 2 * swidth;
				for (int y = top + 2; y < yend; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total), xpad);
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}
			}
		}
		*/
	}
	//16F
	template<int order>
	void SpatialFilterSlidingDCT5_AVX_32F::horizontalFilteringInnerXKdest16F(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;//SIMDSIZE

		SETVEC C1_2[order];
		SETVEC CR_g[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 total[8];
		__m256 F0;
		__m256 Zp[order];
		__m256 Zc[order];
		__m256 delta_inner;//f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256 dc;//f(x+R+1)-f(x-R)
		__m256 dp;//f(x+R)-f(x-R-1)

		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 8)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			short* dstPtr = dst.ptr<short>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			F0 = fn_hfilter[xstart];
			for (int k = 0; k < order; ++k)
			{
				__m256 Cnk = _mm256_set1_ps(GCn[k]);
				Zp[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 1]);
			}

			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[xstart + 0 - n], fn_hfilter[xstart + 0 + n]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[xstart + 1 - n], fn_hfilter[xstart + 1 + n]);
				F0 = _mm256_add_ps(F0, sumA);
				float* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				dc = _mm256_sub_ps(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);

				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius)]);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius)]);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);
				//	F0 = _mm256_add_ps(F0, dp);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ph(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			__m256* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];//f(x+R+1)
			__m256* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius + 0];//f(x-R)

			for (int x = 0; x < mainloop_simdsize; x++)
			{
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[0] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, dp);

				dc = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dc, dp);

				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, dc);

				dp = _mm256_sub_ps(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_ps(dp, dc);

				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i--)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);
				//	F0 = _mm256_add_ps(F0, dp);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ph(dstPtr, total, dwidth);
				dstPtr += 8;
			}
		}
	}
	//16F
	template<int order, typename destT>
	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYKsrc16F(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
		const int dwidth = dst.cols;//dst.cols

		const int xstart = left;
		const int xlength = dst.cols - left - right;
		const int xlengthC = get_simd_ceil(xlength, simdUnrollSize);
		const int xend = xstart + xlengthC;//xendC
		const int rem = xlength - get_simd_floor(xlength, simdUnrollSize);
		const int simdWidth = xlengthC / simdUnrollSize;
		const int simdend = (rem == 0) ? simdWidth : simdWidth - 1;

		const int ylength = dst.rows - (top + bottom);
		const bool isEven = ((ylength) % 2 == 0);
		const int hend = ylength + ((isEven) ? 0 : -1);

		const short* srcPtr = src.ptr<short>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		SETVEC C1_2[order];
		SETVEC CR_g[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA, deltaB;
		__m256 dp, dc, dn;

		__m256* ws = buffVFilter;

		// 1) initilization of F0 and F1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_cvtph_ps(*(__m128i*) & srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_cvtph_ps(*(__m128i*) & srcPtr[swidth * (top + 1) + x]);

			for (int k = order - 1; k >= 0; k--)
			{
				__m256 mCn = _mm256_set1_ps(GCn[k]);
				*ws++ = _mm256_mul_ps(pA, mCn);
				*ws++ = _mm256_mul_ps(pB, mCn);
			}
			*ws++ = pA;
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		for (int r = 1; r <= radius; ++r)
		{
			__m128i* pAM = (__m128i*) & srcPtr[ref_tborder(top + 0 - r, swidth, borderType) + xstart];
			__m128i* pAP = (__m128i*) & srcPtr[swidth * (top + 0 + r) + xstart];
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + 1 - r, swidth, borderType) + xstart];
			__m128i* pBP = (__m128i*) & srcPtr[swidth * (top + 1 + r) + xstart];

			ws = buffVFilter;

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256 pA = _mm256_add_ps(_mm256_cvtph_ps(*pAM++), _mm256_cvtph_ps(*pAP++));
				const __m256 pB = _mm256_add_ps(_mm256_cvtph_ps(*pBM++), _mm256_cvtph_ps(*pBP++));

				for (int i = order - 1; i >= 0; i--)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[order * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[order * r + i]), *ws);
				}
				*ws++ = _mm256_add_ps(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart];
			__m128i* pCM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart];
			__m128i* pBP = (__m128i*) & srcPtr[swidth * (top + y + radius + 1) + xstart];
			__m128i* pCP = (__m128i*) & srcPtr[swidth * (top + y + radius + 2) + xstart];

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			for (int x = 0; x < simdend; ++x)
			{
				dc = _mm256_sub_ps(_mm256_cvtph_ps(*pBP++), _mm256_cvtph_ps(*pBM++));
				dn = _mm256_sub_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pCM++));
				deltaB = _mm256_sub_ps(dn, dc);

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_cvtph_ps(*pBP++), _mm256_cvtph_ps(*pBM++));
				dn = _mm256_sub_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pCM++));
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 temp = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			__m128i* pAM = (__m128i*) & srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart];
			__m128i* pAP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart];
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart];
			__m128i* pBP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart];
			__m128i* pCM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart];
			__m128i* pCP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart];

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_ps(_mm256_cvtph_ps(*pAP++), _mm256_cvtph_ps(*pAM++));
				dc = _mm256_sub_ps(_mm256_cvtph_ps(*pBP++), _mm256_cvtph_ps(*pBM++));
				dn = _mm256_sub_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pCM++));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dp = _mm256_sub_ps(_mm256_cvtph_ps(*pAP++), _mm256_cvtph_ps(*pAM++));
				dc = _mm256_sub_ps(_mm256_cvtph_ps(*pBP++), _mm256_cvtph_ps(*pBM++));
				dn = _mm256_sub_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pCM++));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 temp = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_ps(temp, dn);
			}
			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;
			__m128i* pAM = (__m128i*) & srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart];
			__m128i* pAP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart];
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart];
			__m128i* pBP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart];
			__m128i* pCM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart];
			__m128i* pCP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart];

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_ps(_mm256_cvtph_ps(*pAP++), _mm256_cvtph_ps(*pAM++));
				dc = _mm256_sub_ps(_mm256_cvtph_ps(*pBP++), _mm256_cvtph_ps(*pBM++));
				dn = _mm256_sub_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pCM++));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dp = _mm256_sub_ps(_mm256_cvtph_ps(*pAP++), _mm256_cvtph_ps(*pAM++));
				dc = _mm256_sub_ps(_mm256_cvtph_ps(*pBP++), _mm256_cvtph_ps(*pBM++));
				dn = _mm256_sub_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pCM++));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 temp = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(temp, dn);
			}
		}
	}


	//(under debug!!)32F hfilter O(K) 
	__m256* SpatialFilterSlidingDCT5_AVX_32F::computeZConvHFilter(const float* srcPtr, const int y, const int x, const int width, const int height, const int order, float* GCn, const int radius, const float G0, const int borderType)
	{
#if 0
		float* sptr = const_cast<float*>(srcPtr + width * y);

		bool is64FAcc = true;

		__m256* ret = new __m256[order + 2];

		if (is64FAcc)
		{
			__m256d* ret20 = new __m256d[order + 2];
			__m256d* ret21 = new __m256d[order + 2];
			// 1) initilization of Z0and Z1(n = 0)
			{
				const __m256d pA0 = _mm256_cvtps_pd(_mm_loadu_ps(&sptr[x + 0]));
				const __m256d pA1 = _mm256_cvtps_pd(_mm_loadu_ps(&sptr[x + 4]));
				ret20[0] = pA0;
				ret21[0] = pA1;
				for (int i = order - 1; i >= 0; i--)
				{
					ret20[i + 1] = _mm256_mul_pd(pA0, _mm256_set1_pd(GCn[i]));
					ret21[i + 1] = _mm256_mul_pd(pA1, _mm256_set1_pd(GCn[i]));
				}
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int r = 1; r <= radius; ++r)
			{
				float* pAM0 = const_cast<float*>(&srcPtr[ref_lborder(x - r + 0, borderType)]);
				float* pAP0 = const_cast<float*>(&srcPtr[ref_rborder(x + r + 0, imgSize.width, borderType)]);
				float* pAM1 = const_cast<float*>(&srcPtr[ref_lborder(x - r + 4, borderType)]);
				float* pAP1 = const_cast<float*>(&srcPtr[ref_rborder(x + r + 4, imgSize.width, borderType)]);
				{
					const __m256d pA0 = _mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(pAM0)), _mm256_cvtps_pd(_mm_loadu_ps(pAP0)));
					const __m256d pA1 = _mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(pAM1)), _mm256_cvtps_pd(_mm_loadu_ps(pAP1)));

					ret20[0] = _mm256_add_pd(ret20[0], pA0);
					ret21[0] = _mm256_add_pd(ret21[0], pA1);
					for (int i = order - 1; i >= 0; i--)
					{
						ret20[i + 1] = _mm256_fmadd_pd(pA0, _mm256_set1_pd(GCn[order * r + i]), ret20[i + 1]);
						ret21[i + 1] = _mm256_fmadd_pd(pA1, _mm256_set1_pd(GCn[order * r + i]), ret21[i + 1]);
					}
				}
			}
			ret20[0] = _mm256_mul_pd(ret20[0], _MM256_SET_VECD(mG0));
			ret21[0] = _mm256_mul_pd(ret21[0], _MM256_SET_VECD(mG0));

			//output total and set
			ret20[order + 1] = _mm256_setzero_pd();
			ret21[order + 1] = _mm256_setzero_pd();
			for (int i = 0; i <= order; i++)
			{
				ret20[order + 1] = _mm256_add_pd(ret20[i], ret20[order + 1]);
				ret21[order + 1] = _mm256_add_pd(ret21[i], ret21[order + 1]);
				ret[i] = _mm256_set_m128(_mm256_cvtpd_ps(ret21[i]), _mm256_cvtpd_ps(ret20[i]));
			}
			ret[order + 1] = _mm256_set_m128(_mm256_cvtpd_ps(ret21[order + 1]), _mm256_cvtpd_ps(ret20[order + 1]));
			delete[] ret20;
			delete[] ret21;
		}
		else
		{
			// 1) initilization of Z0and Z1(n = 0)
			{
				const __m256 pA = _mm256_loadu_ps(&srcPtr[(y)*width + x]);
				ret[0] = pA;
				for (int i = order - 1; i >= 0; i--)
				{
					ret[i + 1] = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				}
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int r = 1; r <= radius; ++r)
			{
				float* pAM = const_cast<float*>(&srcPtr[UREF(y - r) + x]);
				float* pAP = const_cast<float*>(&srcPtr[DREF(y + r) + x]);
				{
					const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
					ret[0] = _mm256_add_ps(ret[0], pA);
					for (int i = order - 1; i >= 0; i--)
					{
						ret[i + 1] = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[order * r + i]), ret[i + 1]);
					}
				}
			}
			ret[0] = _mm256_mul_ps(ret[0], _MM256_SET_VEC(mG0));

			//output total
			ret[order + 1] = _mm256_setzero_ps();
			for (int i = 0; i < order + 1; i++)
			{
				ret[order + 1] = _mm256_add_ps(ret[i], ret[order + 1]);
			}

		}
		return ret;
#endif
		__m256* a = nullptr;
		return a;
	}
	//(under debug!!)32F hfilter O(K) 
	void SpatialFilterSlidingDCT5_AVX_32F::horizontalFilteringInnerXKn_32F_Debug(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		;
	}

	//plofile plot 32F hfilter O(K)
	__m256* SpatialFilterSlidingDCT5_AVX_32F::computeZConvVFilter(const float* srcPtr, const int y, const int x, const int width, const int height, const int order, float* GCn, const int radius, const float G0, const int borderType)
	{
		bool is64FAcc = true;

		__m256* ret = new __m256[order + 2];

		if (is64FAcc)
		{
			SETVECD mG0 = _MM256_SETLUT_VECD(G0);
			__m256d* ret20 = new __m256d[order + 2];
			__m256d* ret21 = new __m256d[order + 2];
			// 1) initilization of Z0and Z1(n = 0)
			{
				const __m256d pA0 = _mm256_cvtps_pd(_mm_loadu_ps(&srcPtr[(y)*width + x + 0]));
				const __m256d pA1 = _mm256_cvtps_pd(_mm_loadu_ps(&srcPtr[(y)*width + x + 4]));
				ret20[0] = pA0;
				ret21[0] = pA1;
				for (int i = order - 1; i >= 0; i--)
				{
					ret20[i + 1] = _mm256_mul_pd(pA0, _mm256_set1_pd(GCn[i]));
					ret21[i + 1] = _mm256_mul_pd(pA1, _mm256_set1_pd(GCn[i]));
				}
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int r = 1; r <= radius; ++r)
			{
				float* pAM0 = const_cast<float*>(&srcPtr[ref_tborder(y - r, width, borderType) + x + 0]);
				float* pAM1 = const_cast<float*>(&srcPtr[ref_tborder(y - r, width, borderType) + x + 4]);
				float* pAP0 = const_cast<float*>(&srcPtr[ref_bborder(y + r, width, height, borderType) + x + 0]);
				float* pAP1 = const_cast<float*>(&srcPtr[ref_bborder(y + r, width, height, borderType) + x + 4]);
				{
					const __m256d pA0 = _mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(pAM0)), _mm256_cvtps_pd(_mm_loadu_ps(pAP0)));
					const __m256d pA1 = _mm256_add_pd(_mm256_cvtps_pd(_mm_loadu_ps(pAM1)), _mm256_cvtps_pd(_mm_loadu_ps(pAP1)));

					ret20[0] = _mm256_add_pd(ret20[0], pA0);
					ret21[0] = _mm256_add_pd(ret21[0], pA1);
					for (int i = order - 1; i >= 0; i--)
					{
						ret20[i + 1] = _mm256_fmadd_pd(pA0, _mm256_set1_pd(GCn[order * r + i]), ret20[i + 1]);
						ret21[i + 1] = _mm256_fmadd_pd(pA1, _mm256_set1_pd(GCn[order * r + i]), ret21[i + 1]);
					}
				}
			}
			ret20[0] = _mm256_mul_pd(ret20[0], _MM256_SET_VECD(mG0));
			ret21[0] = _mm256_mul_pd(ret21[0], _MM256_SET_VECD(mG0));

			//output total and set
			ret20[order + 1] = _mm256_setzero_pd();
			ret21[order + 1] = _mm256_setzero_pd();
			for (int i = 0; i <= order; i++)
			{
				ret20[order + 1] = _mm256_add_pd(ret20[i], ret20[order + 1]);
				ret21[order + 1] = _mm256_add_pd(ret21[i], ret21[order + 1]);
				ret[i] = _mm256_set_m128(_mm256_cvtpd_ps(ret21[i]), _mm256_cvtpd_ps(ret20[i]));
			}
			ret[order + 1] = _mm256_set_m128(_mm256_cvtpd_ps(ret21[order + 1]), _mm256_cvtpd_ps(ret20[order + 1]));
			delete[] ret20;
			delete[] ret21;
		}
		else
		{
			SETVEC mG0 = _MM256_SETLUT_VEC(G0);
			// 1) initilization of Z0and Z1(n = 0)
			{
				const __m256 pA = _mm256_loadu_ps(&srcPtr[(y)*width + x]);
				ret[0] = pA;
				for (int i = order - 1; i >= 0; i--)
				{
					ret[i + 1] = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				}
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int r = 1; r <= radius; ++r)
			{
				float* pAM = const_cast<float*>(&srcPtr[ref_tborder(y - r, width, borderType) + x]);
				float* pAP = const_cast<float*>(&srcPtr[ref_bborder(y + r, width, height, borderType) + x]);
				{
					const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
					ret[0] = _mm256_add_ps(ret[0], pA);
					for (int i = order - 1; i >= 0; i--)
					{
						ret[i + 1] = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[order * r + i]), ret[i + 1]);
					}
				}
			}
			ret[0] = _mm256_mul_ps(ret[0], _MM256_SET_VEC(mG0));

			//output total
			ret[order + 1] = _mm256_setzero_ps();
			for (int i = 0; i < order + 1; i++)
			{
				ret[order + 1] = _mm256_add_ps(ret[i], ret[order + 1]);
			}

		}
		return ret;
	}
	//plofile plot 32F hfilter O(K) 
	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYKn_32F_Debug(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
#ifdef PLOT_Zk
		string wname = "DCT-5 V plofile plot";
		static int pre_order = 0;
		namedWindow(wname);
		static int line = 8;//plofile plot
		createTrackbar("x", wname, &line, src.cols - 1 - radius - 1);
		setTrackbarMin("x", wname, radius + 1);
		static int scale = 45; createTrackbar("scale", wname, &scale, 80);
		const int totalIndex = order + 1;
		static int plot_order = totalIndex; createTrackbar("plotorder", wname, &plot_order, totalIndex);
		if (pre_order != order)
		{
			pre_order = order;
			plot_order = totalIndex;
			setTrackbarMax("plotorder", wname, totalIndex);
			setTrackbarPos("plotorder", wname, plot_order);
		}

		cp::Plot pt;
		const bool isDiff = true;
		vector<bool> isPlot(totalIndex + 1);

		for (int i = 0; i < isPlot.size(); i++)isPlot[i] = false;

		if (plot_order == totalIndex)
		{
			for (int i = 0; i < isPlot.size(); i++)isPlot[i] = true;
		}
		else
		{
			isPlot[order - plot_order] = true;
		}

		for (int i = 0; i <= order; i++)
		{
			pt.setPlotTitle(i, cv::format("order %d", i));
		}
		pt.setPlotTitle(totalIndex, "total");
#endif

		const int simdUnrollSize = 8;//8

		const int width = src.cols;
		const int dwidth = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols;

		const int xstart = left;
		const int xend = get_simd_ceil(dwidth - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>();
		float* dstPtr = dst.ptr<float>(top, xstart);

		AutoBuffer <SETVEC> C1_2(order);
		AutoBuffer <SETVEC> CR(order);
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA, deltaB;
		__m256 dp, dc, dn;

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += simdUnrollSize)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[(top + 0) * width + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[(top + 1) * width + x]);

			for (int i = order - 1; i >= 0; i--)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
			*ws++ = pA;
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		for (int r = 1; r <= radius; ++r)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + 0 - r, width, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + 1 - r, width, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[width * (top + 0 + r) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[width * (top + 1 + r) + xstart]);

			ws = buffVFilter;

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				const __m256 pB = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				for (int i = order - 1; i >= 0; i--)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[order * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[order * r + i]), *ws);
				}
				*ws++ = _mm256_add_ps(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, width, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, width, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[width * (top + y + radius + 1) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[width * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			float* dstPtr2 = dstPtr;
			for (int x = 0; x < simdWidth; ++x)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaB = _mm256_sub_ps(dn, dc);
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[order - 1]), deltaB, *ws));
				ws += 2;

				for (int i = order - 2; i >= 0; i--)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[i]), deltaB, *ws));
					ws += 2;
				}

				store_auto<float>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<float>(dstPtr2 + dstStep, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			dstPtr += 2 * dstStep;
		}

		// 3) main loop
		for (int y = 2; y < height - (top + bottom); y += 2)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, width, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 0, width, height, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, width, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, width, height, borderType) + xstart]);
			float* pCM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 1, width, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, width, height, borderType) + xstart]);

			ws = buffVFilter;
			float* dstPtr2 = dstPtr;

			for (int x = 0; x < simdWidth; ++x)
			{
				dp = _mm256_sub_ps(_mm256_loadu_ps(pAP), _mm256_loadu_ps(pAM));
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				deltaA = _mm256_sub_ps(dc, dp);
				deltaB = _mm256_sub_ps(dn, dc);
				pAP += 8;
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;
				pCM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR[order - 1]), deltaA, *(ws + 1)));

#ifdef PLOT_Zk
				__m256* conv0 = nullptr;
				if (x == line / 8)
				{
					conv0 = computeZConvVFilter(srcPtr, top + y + 0, xstart + 8 * x, width, height, order, GCn, radius, G0, borderType);
				}
				if (x == line / 8 && isPlot[order])// plot last order
				{
					if (isDiff)
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(*ws, conv0[order]);
						pt.push_back(y + 0, ((float*)&sub)[idx], order);
					}
					else
					{
						;
					}
				}
#endif
				* ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[order - 1]), deltaB, *ws));
				ws += 2;
				for (int i = order - 2; i >= 0; i--)
				{
#ifdef PLOT_Zk
					if (x == line / 8 && isPlot[i + 1])
					{
						if (isDiff)
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(*ws, conv0[i + 1]);
							//	std::cout << "sub = " << ((float*)&sub)[0] << std::endl;

							pt.push_back(y, ((float*)&sub)[idx], (i + 1));
						}
						else
						{
							;
						}
					}
#endif
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR[i]), deltaA, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[i]), deltaB, *ws));
					ws += 2;
				}

#ifdef PLOT_Zk
				if (x == line / 8)
				{
					__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), *ws);
					__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA);
					if (isDiff)
					{
						if (isPlot[0])
						{
							__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
							const int idx = line % 8;
							pt.push_back(y, ((float*)&sub)[idx], 0);
						}
						if (isPlot[totalIndex])
						{
							__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
							const int idx = line % 8;
							pt.push_back(y, ((float*)&sub)[idx], totalIndex);
						}
					}
					else
					{
						;
					}

					delete[] conv0;
				}
#endif
				store_auto<float>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				store_auto<float>(dstPtr2 + dstStep, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);
				dstPtr2 += 8;
			}
			dstPtr += 2 * dstStep;
		}

#ifdef PLOT_Zk
		pt.setYRange(-pow(10.0, -scale * 0.1), pow(10.0, -scale * 0.1));
		//pt.setKey(cp::Plot::FLOATING);
		pt.plot(wname, false);
#endif
	}

	void SpatialFilterSlidingDCT5_AVX_32F::verticalFilteringInnerXYK_XYLoop_Debug(const cv::Mat& src, cv::Mat& dst, int order, const int borderType)
	{
		static int showOrder = 0;
		Mat show = Mat::zeros(dst.size(), CV_32F);
#ifdef PLOT_Zk
		createTrackbar("showorder", "", &showOrder, order);
		setTrackbarMax("showorder", "", order);
		string wname = "DCT-5 V plofile plot";
		static int pre_order = 0;
		namedWindow(wname);
		static int line = 8;//plofile plot
		createTrackbar("x", wname, &line, src.cols - 1 - radius - 1);
		setTrackbarMin("x", wname, radius + 1);
		static int scale = 45; createTrackbar("scale", wname, &scale, 80);
		const int totalIndex = order + 1;
		static int plot_order = totalIndex; createTrackbar("plotorder", wname, &plot_order, totalIndex);
		if (pre_order != order)
		{
			pre_order = order;
			plot_order = totalIndex;
			setTrackbarMax("plotorder", wname, totalIndex);
			setTrackbarPos("plotorder", wname, plot_order);
		}

		cp::Plot pt(Size(src.rows * 2, 512));
		const bool isDiff = true;
		vector<bool> isPlot(totalIndex + 1);

		for (int i = 0; i < isPlot.size(); i++)isPlot[i] = false;

		if (plot_order == totalIndex)
		{
			for (int i = 0; i < isPlot.size(); i++)isPlot[i] = true;
		}
		else
		{
			isPlot[order - plot_order] = true;
		}

		for (int i = 0; i <= order; i++)
		{
			pt.setPlotTitle(i, cv::format("order %d", i));
		}
		pt.setPlotTitle(totalIndex, "total");
#endif
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;
		const int dwidth = dst.cols;
		const int height = imgSize.height;

		const int xstart = left;
		const int xend = get_simd_ceil(dwidth - left - right, simdUnrollSize) + xstart;
		//const int simdWidth = (xend - xstart) / simdUnrollSize;

		const int yst = get_loop_end(top + 2, radius + 1, 2);
		const int yed = get_simd_floor(imgSize.height - radius - 1 - yst, 2) + yst;
		const int yend = get_simd_ceil(imgSize.height - (top + bottom), 2) + top - 2 * ((imgSize.height) % 2);
		const bool isylast = (imgSize.height % 2 == 1);
		//print_debug4(height, radius, top, bottom);
		//print_debug4(yst, yed, yend, height - 2 - radius);

		AutoBuffer <SETVEC>C1_2(order);
		AutoBuffer <SETVEC>CR_g(order);
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 total;
		__m256 F0;
		__m256 delta_inner;
		AutoBuffer<__m256> Zp(order);
		AutoBuffer<__m256> Zc(order);
		__m256 dp, dc;
		__m256* conv0 = nullptr;

		for (int x = xstart; x < xend; x += simdUnrollSize)
		{
			int ystart = top;
			float* dstPtr = dst.ptr<float>(ystart, x);
			float* showPtr = show.ptr<float>(ystart, x);
			const float* srcPtr = src.ptr<float>(0, x);

			// 1) initilization of Z0 and Z1 (n=radius)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - radius, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + radius)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius)));
				F0 = sumA;
				float* Cn_ = GCn + order * radius;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_mul_ps(Cnk, sumA);
					Zc[k] = _mm256_mul_ps(Cnk, sumB);
				}
			}
			// 1) initilization of Z0 and Z1 (1<=n<radius)
			for (int n = radius - 1; n >= 1; --n)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + n)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + n)));
				F0 = _mm256_add_ps(F0, sumA);
				float* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}
			// 1) initilization of Z0 and Z1 (n=0)
			{
				const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (ystart + 0)]);
				const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (ystart + 1)]);
				F0 = _mm256_add_ps(F0, pA);
				for (int k = 0; k < order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(GCn[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, pA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, pB, Zc[k]);
				}
			}

			if (left < radius)
			{
				// 2) initial output computing for y=0,1
				{
					//F0 is already computed by initilization
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - radius, swidth, borderType)));
					total = Zp[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zp[order - 1]);
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zp[k]);
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zc[order - 1]);
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zc[k]);
					}
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;
				}

				// 3) main loop
				//top part
				for (int y = top + 2; y < yst; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 1 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y + 0 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zp[order - 1]);
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 0, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zp[order - 1], conv0[order]);
						pt.push_back(y + 0, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zp[k], conv0[k + 1]);
							pt.push_back(y + 0, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zp[k]);
					}
#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 2 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y + 1 - radius, swidth, borderType)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zc[order - 1]);
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 1, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zc[order - 1], conv0[order]);
						pt.push_back(y + 1, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zc[k], conv0[k + 1]);
							pt.push_back(y + 1, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zc[k]);
					}

#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					_mm256_storeu_auto(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;
				}

				//mid part
				float* s0 = (float*)(srcPtr + swidth * (yst + 0 - radius));
				float* s1 = (float*)(srcPtr + swidth * (yst + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (yst + 1 - radius));
				float* s3 = (float*)(srcPtr + swidth * (yst + 2 + radius));
				for (int y = yst; y < yed; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zp[order - 1]);

					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 0, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zp[order - 1], conv0[order]);
						pt.push_back(y + 0, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zp[k], conv0[k + 1]);
							pt.push_back(y + 0, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zp[k]);
					}

#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zc[order - 1]);
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 1, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zc[order - 1], conv0[order]);
						pt.push_back(y + 1, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zc[k], conv0[k + 1]);
							pt.push_back(y + 1, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zc[k]);
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}

				//bottom part
				for (int y = yed; y < yend; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 1 + radius, swidth, height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 0 - radius)));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zp[order - 1]);
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 0, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zp[order - 1], conv0[order]);
						pt.push_back(y + 0, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zp[k], conv0[k + 1]);
							pt.push_back(y + 0, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zp[k]);
					}

#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));
					dstPtr += dwidth;
					showPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(srcPtr + ref_bborder(top + y + 2 + radius, swidth, height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (top + y + 1 - radius)));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					if (order == showOrder)_mm256_storeu_auto(showPtr, Zc[order - 1]);
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 1, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zc[order - 1], conv0[order]);
						pt.push_back(y + 1, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zc[k], conv0[k + 1]);
							pt.push_back(y + 1, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
						if (k + 1 == showOrder)_mm256_storeu_auto(showPtr, Zc[k]);
					}

#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					if (showOrder == 0)_mm256_storeu_auto(showPtr, _mm256_mul_ps(_MM256_SET_VEC(mG0), F0));

					dstPtr += dwidth;
					showPtr += dwidth;
				}
			}
			else
			{
				// 2) initial output computing for y=0
				float* s0 = (float*)(srcPtr + swidth * (top + 0 - radius));
				float* s1 = (float*)(srcPtr + swidth * (top + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (top + 1 - radius));
				float* s3 = (float*)(srcPtr + swidth * (top + 2 + radius));
				{
					//F0 is already computed by initilization
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[order - 1];
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
					for (int k = order - 2; k >= 0; k--)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;
				}

				// 3) main loop
				s0 += 2 * swidth;
				s1 += 2 * swidth;
				s2 += 2 * swidth;
				s3 += 2 * swidth;
				for (int y = top + 2; y < yend; y += 2)
				{
					F0 = _mm256_add_ps(F0, dp);//computing F0(x+1) = F0(x)+dc(x)
					dc = _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					delta_inner = _mm256_sub_ps(dc, dp);
					total = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zc[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 0, 0, swidth, height, order, GCn, radius, G0, borderType);

					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zc[order - 1], conv0[order]);
						pt.push_back(y + 0, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zc[k], conv0[k + 1]);
							pt.push_back(y + 0, ((float*)&sub)[idx], (k + 1));
						}
#endif

						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}

#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 0, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					F0 = _mm256_add_ps(F0, dc);//computing F0(x+1) = F0(x)+dc(x)
					dp = _mm256_sub_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					delta_inner = _mm256_sub_ps(dp, dc);
					total = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[order - 1]), delta_inner, Zp[order - 1]));
#ifdef PLOT_Zk
					conv0 = nullptr;
					if (x <= line && line < x + 8)
					{
						conv0 = computeZConvVFilter(srcPtr, y + 1, 0, swidth, height, order, GCn, radius, G0, borderType);
					}
					if ((x <= line && line < x + 8) && isPlot[order])
					{
						const int idx = line % 8;
						__m256 sub = _mm256_sub_ps(Zp[order - 1], conv0[order]);
						pt.push_back(y + 1, ((float*)&sub)[idx], order);
					}
#endif
					for (int k = order - 2; k >= 0; k--)
					{
#ifdef PLOT_Zk
						if ((x <= line && line < x + 8) && isPlot[k + 1])
						{
							const int idx = line % 8;
							__m256 sub = _mm256_sub_ps(Zc[k], conv0[k + 1]);
							pt.push_back(y + 1, ((float*)&sub)[idx], (k + 1));
						}
#endif
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}

#ifdef PLOT_Zk
					if (x <= line && line < x + 8)
					{
						__m256 order_zero = _mm256_mul_ps(_MM256_SET_VEC(mG0), F0);
						__m256 plot_total = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total);
						if (isDiff)
						{
							if (isPlot[0])
							{
								__m256 sub = _mm256_sub_ps(order_zero, conv0[0]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], 0);
							}
							if (isPlot[totalIndex])
							{
								__m256 sub = _mm256_sub_ps(plot_total, conv0[totalIndex]);
								const int idx = line % 8;
								pt.push_back(y + 1, ((float*)&sub)[idx], totalIndex);
							}
						}
						else
						{
							;
						}

						delete[] conv0;
					}
#endif
					store_auto<float>(dstPtr, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total));
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}
			}
		}
#ifdef PLOT_Zk
		pt.setYRange(-pow(10.0, -scale * 0.1), pow(10.0, -scale * 0.1));
		//	pt.setKey(cp::Plot::FLOATING);
		pt.plot(wname, false);
		cp::imshowNormalize("order", show);
#endif
	}

	void SpatialFilterSlidingDCT5_AVX_32F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		//cout<<"filtering sliding DCT5 GF AVX 32F" << endl;
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

		dst.create(imgSize, (dest_depth < 0) ? src.depth() : dest_depth);

		bool isFloat = (dest_depth == CV_32F || (src.depth() == CV_32F && dest_depth < 0)) ? true : false;

		//schedule = SLIDING_DCT_SCHEDULE::DEBUG;
		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT5_32F_ORDER_TEMPLATE
			case 1:
				horizontalFilteringInnerXKdest16F<1>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<1, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<1, uchar>(inter, dst, borderType);
				break;
			case 2:
				horizontalFilteringInnerXKdest16F<2>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<2, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<2, uchar>(inter, dst, borderType);
				break;
			case 3:
				horizontalFilteringInnerXKdest16F<3>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<3, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<3, uchar>(inter, dst, borderType);
				break;
			case 4:
				horizontalFilteringInnerXKdest16F<4>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<4, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<4, uchar>(inter, dst, borderType);
				break;
			case 5:
				horizontalFilteringInnerXKdest16F<5>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<5, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<5, uchar>(inter, dst, borderType);
				break;
			case 6:
				horizontalFilteringInnerXKdest16F<6>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<6, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<6, uchar>(inter, dst, borderType);
				break;
			case 7:
				horizontalFilteringInnerXKdest16F<7>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<7, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<7, uchar>(inter, dst, borderType);
				break;
			case 8:
				horizontalFilteringInnerXKdest16F<8>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<8, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<8, uchar>(inter, dst, borderType);
				break;
			case 9:
				horizontalFilteringInnerXKdest16F<9>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<9, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<9, uchar>(inter, dst, borderType);
				break;
			case 10:
				horizontalFilteringInnerXKdest16F<10>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<10, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<10, uchar>(inter, dst, borderType);
				break;
			case 11:
				horizontalFilteringInnerXKdest16F<11>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<11, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<11, uchar>(inter, dst, borderType);
				break;
			case 12:
				horizontalFilteringInnerXKdest16F<12>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<12, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<12, uchar>(inter, dst, borderType);
				break;
			case 13:
				horizontalFilteringInnerXKdest16F<13>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<13, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<13, uchar>(inter, dst, borderType);
				break;
			case 14:
				horizontalFilteringInnerXKdest16F<14>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<14, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<14, uchar>(inter, dst, borderType);
				break;
			case 15:
				horizontalFilteringInnerXKdest16F<15>(src, inter, borderType);
				if (isFloat)verticalFilteringInnerXYKsrc16F<15, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<15, uchar>(inter, dst, borderType);
				break;
#endif
			default:
				std::cout << "do not support this order (GaussianFilterSlidingDCT5_AVX_32F::)" << std::endl;
				//horizontalFilteringInnerXKn(src, inter);
				//verticalFilteringInnerXYKn(inter, dst);
				break;
			}
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::CONVOLUTION)
		{
			CV_Assert(src.cols == src.rows);
			horizontalFilteringNaiveConvolution(src, inter, gf_order, borderType);
			Mat temp;
			inter(Rect(0, 0, src.cols, src.rows)).copyTo(temp);
			transpose(temp, temp);
			horizontalFilteringNaiveConvolution(temp, inter, gf_order, borderType);
			inter(Rect(0, 0, src.cols, src.rows)).copyTo(temp);
			transpose(temp, dst);
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::DEBUG)
		{
			CV_Assert(isFloat);//dest type must be float
			//horizontalFilteringNaiveConvolution(src, inter, gf_order);
			//verticalFilteringInnerXYKn_32F_Debug(inter, dst, gf_order);

			Mat temp = Mat::zeros(src.size(), CV_32F);
			horizontalFilteringNaiveConvolution(src, temp, gf_order, borderType);
			verticalFilteringInnerXYK_XYLoop_Debug(temp, dst, gf_order, borderType);

			/*

			transpose(src, temp);
			//	verticalFilteringInnerXYKn_32F_Debug(src, temp, gf_order);
			verticalFilteringInnerXYK_XYLoop_Debug(src, temp, gf_order, borderType);
			horizontalFilteringNaiveConvolution(temp, dst, gf_order, borderType);
			*/
			//verticalFilteringInnerXYKn_32F_Debug(src, dst, gf_order);
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::V_XY_LOOP)
		{
			//test
			bool isTestForPad2 = true;
			if (isTestForPad2)
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT5_32F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXK<1>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYKReuseAll<1, float>(inter, dst, borderType);
					else verticalFilteringInnerXYKReuseAll<1, uchar>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXK<2>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYKReuseAll<2, float>(inter, dst, borderType);
					else verticalFilteringInnerXYKReuseAll<2, uchar>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXK<3>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYKReuseAll<3, float>(inter, dst, borderType);
					else verticalFilteringInnerXYKReuseAll<3, uchar>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXK<4>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYKReuseAll<4, float>(inter, dst, borderType);
					else verticalFilteringInnerXYKReuseAll<4, uchar>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXK<5>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYKReuseAll<5, float>(inter, dst, borderType);
					else verticalFilteringInnerXYKReuseAll<5, uchar>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXK<6>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<6, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<6, uchar>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXK<7>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<7, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<7, uchar>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXK<8>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<8, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<8, uchar>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXK<9>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<9, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<9, uchar>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXK<10>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<10, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<10, uchar>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXK<11>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<11, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<11, uchar>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXK<12>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<12, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<12, uchar>(inter, dst, borderType);
					break;
				case 13:
					horizontalFilteringInnerXK<13>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<13, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<13, uchar>(inter, dst, borderType);
					break;
				case 14:
					horizontalFilteringInnerXK<14>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<14, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<14, uchar>(inter, dst, borderType);
					break;
				case 15:
					horizontalFilteringInnerXK<15>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK<15, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK<15, uchar>(inter, dst, borderType);
					break;
#endif
				default:
					horizontalFilteringInnerXKn(src, inter, gf_order, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYKn<float>(inter, dst, gf_order, borderType);
					else verticalFilteringInnerXYKn<uchar>(inter, dst, gf_order, borderType);
					break;
				}
			}
			else
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT5_32F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXK<1>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<1, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<1, uchar>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXK<2>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<2, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<2, uchar>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXK<3>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<3, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<3, uchar>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXK<4>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<4, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<4, uchar>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXK<5>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<5, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<5, uchar>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXK<6>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<6, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<6, uchar>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXK<7>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<7, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<7, uchar>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXK<8>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<8, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<8, uchar>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXK<9>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<9, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<9, uchar>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXK<10>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<10, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<10, uchar>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXK<11>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<11, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<11, uchar>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXK<12>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<12, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<12, uchar>(inter, dst, borderType);
					break;
				case 13:
					horizontalFilteringInnerXK<13>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<13, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<13, uchar>(inter, dst, borderType);
					break;
				case 14:
					horizontalFilteringInnerXK<14>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<14, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<14, uchar>(inter, dst, borderType);
					break;
				case 15:
					horizontalFilteringInnerXK<15>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_XYLoop<15, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_XYLoop<15, uchar>(inter, dst, borderType);
					break;
#endif
				default:
					std::cout << "do not support this order (GaussianFilterSlidingDCT5_AVX_32F)" << std::endl;
					break;
				}
			}
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::DEFAULT)
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT5_32F_ORDER_TEMPLATE
			case 1:
				horizontalFilteringInnerXK<1>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<1, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<1, uchar>(inter, dst, borderType);
				break;
			case 2:
				horizontalFilteringInnerXK<2>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<2, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<2, uchar>(inter, dst, borderType);
				break;
			case 3:
				horizontalFilteringInnerXK<3>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<3, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<3, uchar>(inter, dst, borderType);

				break;
			case 4:
				horizontalFilteringInnerXK<4>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<4, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<4, uchar>(inter, dst, borderType);
				break;
			case 5:
				horizontalFilteringInnerXK<5>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<5, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<5, uchar>(inter, dst, borderType);
				break;
			case 6:
				horizontalFilteringInnerXK<6>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<6, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<6, uchar>(inter, dst, borderType);
				break;
			case 7:
				horizontalFilteringInnerXK<7>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<7, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<7, uchar>(inter, dst, borderType);
				break;
			case 8:
				horizontalFilteringInnerXK<8>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<8, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<8, uchar>(inter, dst, borderType);
				break;
			case 9:
				horizontalFilteringInnerXK<9>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<9, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<9, uchar>(inter, dst, borderType);
				break;
			case 10:
				horizontalFilteringInnerXK<10>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<10, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<10, uchar>(inter, dst, borderType);
				break;
			case 11:
				horizontalFilteringInnerXK<11>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<11, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<11, uchar>(inter, dst, borderType);
				break;
			case 12:
				horizontalFilteringInnerXK<12>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<12, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<12, uchar>(inter, dst, borderType);
				break;
			case 13:
				horizontalFilteringInnerXK<13>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<13, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<13, uchar>(inter, dst, borderType);
				break;
			case 14:
				horizontalFilteringInnerXK<14>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<14, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<14, uchar>(inter, dst, borderType);
				break;
			case 15:
				horizontalFilteringInnerXK<15>(src, inter, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYK<15, float>(inter, dst, borderType);
				else verticalFilteringInnerXYK<15, uchar>(inter, dst, borderType);
				break;
#endif
			default:
				horizontalFilteringInnerXKn(src, inter, gf_order, borderType);
				if (dst.depth() == CV_32F)verticalFilteringInnerXYKn<float>(inter, dst, gf_order, borderType);
				else verticalFilteringInnerXYKn<uchar>(inter, dst, gf_order, borderType);
				break;
			}
		}

		//cp::imshowScale("inter", inter);
	}

	void SpatialFilterSlidingDCT5_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		if (this->radius == 0 || this->sigma != sigma || this->gf_order != order || this->imgSize != src.size() || this->fn_hfilter == nullptr)
		{
			//cp::Timer t;
			this->sigma = sigma;
			this->gf_order = order;
			//print_debug(radius);
			if (!isUseFixRadius) computeRadius(0, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
			else computeRadius(radius, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
			
			imgSize = src.size();
			allocBuffer();
			//print_debug5("initDCT5", radius, this->sigma, this->gf_order, imgSize);
		}
		//cout << this->radius << endl;
		body(src, dst, borderType);
	}

#pragma endregion

#pragma region SlidingDCT5_AVX512_32F
#ifdef CP_AVX_512
	void GaussianFilterSlidingDCT5_AVX512_32F::setRadius(const int rad, const bool isOptimize)
	{
		double* spect = new double[gf_order + 1];
		if (rad == 0)radius = argminR_BruteForce_DCT(sigma, gf_order, 5, spect, isOptimize);
		else radius = rad;

		const double phase = CV_2PI / (2.0 * radius + 1.0);

		computeSpectrum(sigma, gf_order, radius, 5, spect);
		if (isOptimize) optimizeSpectrum(sigma, gf_order, radius, 5, spect);

		delete[] table;
		//table = new float[(order + 1)*(1 + radius)];for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		table = new float[gf_order * (1 + radius)];

		double sum = 2.0 * radius + 1.0;
		for (int r = 0; r <= radius; ++r)
		{
			for (int k = 1; k <= gf_order; ++k)
			{
				//table[order*r + k - 1] = float(cos(k*phase*r)*spect[k]);
				table[gf_order * r + gf_order - k] = float(cos(k * phase * r) * spect[k]);

				if (r == 0) sum += cos(k * phase * r) * spect[k];
				else sum += 2.0 * cos(k * phase * r) * spect[k];
			}
		}
		G0 = (float)(1.0 / sum);
		//vizKernel(sigma, order, radius, G0, table, 5);
		//G0 = float(1.0 / (2 * radius + 1));

		//cout << spect[0] << endl;
		//cout << spect[1] << endl;
		//cout << sum * norm << endl;

		delete[] shift;
		shift = new float[2 * gf_order];
		for (int k = 1; k <= gf_order; ++k)
		{
			//shift[2 * (k - 1) + 0] = (float)(cos(k*phase * 1)*spect[k] * 2.0 / spect[k]);
			//shift[2 * (k - 1) + 1] = (float)(cos(k*phase * radius)*spect[k]);

			shift[2 * (gf_order - k) + 0] = (float)(cos(k * phase * 1) * 2.0);
			shift[2 * (gf_order - k) + 1] = (float)(cos(k * phase * radius) * spect[k]);
		}

		_mm_free(buffHFilter);
		_mm_free(buffVFilter);
		buffHFilter = (__m512*)_mm_malloc(imgSize.width * sizeof(__m512), AVX512_ALIGNMENT);
		buffVFilter = (__m512*)_mm_malloc(int((2 * gf_order + 1) * imgSize.width / 16) * sizeof(__m512), AVX512_ALIGNMENT);

		delete[] spect;
	}

	GaussianFilterSlidingDCT5_AVX512_32F::GaussianFilterSlidingDCT5_AVX512_32F(cv::Size imgSize, float sigma, int order)
		: GaussianFilterBase(imgSize, CV_32F)
	{
		gf_order = order;
		CV_Assert(imgSize.width % 16 == 0);
		CV_Assert(imgSize.height % 16 == 0);

		this->sigma = sigma;
		inter.create(imgSize, CV_32F);

		setRadius(radius, true);
	}

	GaussianFilterSlidingDCT5_AVX512_32F::~GaussianFilterSlidingDCT5_AVX512_32F()
	{
		delete[] table;
		delete[] shift;
		_mm_free(buffHFilter);
		_mm_free(buffVFilter);
	}


	void GaussianFilterSlidingDCT5_AVX512_32F::copyPatchHorizontalbody(cv::Mat& src, const int y)
	{
		const int simdUnrollSize = 16;
		const int simdWidth = imgSize.width / simdUnrollSize;

		__m512* buffPtr = buffHFilter;

		if (src.depth() == CV_8U)
		{
			__m128i* srcPtr = (__m128i*)src.ptr<uchar>(y);

			for (int x = 0; x < simdWidth; ++x)
			{
				buffPtr[0] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[0 * simdWidth]));
				buffPtr[1] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[1 * simdWidth]));
				buffPtr[2] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[2 * simdWidth]));
				buffPtr[3] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[3 * simdWidth]));
				buffPtr[4] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[4 * simdWidth]));
				buffPtr[5] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[5 * simdWidth]));
				buffPtr[6] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[6 * simdWidth]));
				buffPtr[7] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[7 * simdWidth]));
				buffPtr[8] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[8 * simdWidth]));
				buffPtr[9] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[9 * simdWidth]));
				buffPtr[10] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[10 * simdWidth]));
				buffPtr[11] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[11 * simdWidth]));
				buffPtr[12] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[12 * simdWidth]));
				buffPtr[13] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[13 * simdWidth]));
				buffPtr[14] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[14 * simdWidth]));
				buffPtr[15] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i*) & srcPtr[15 * simdWidth]));
				_mm512_transpose16_ps(buffPtr);

				++srcPtr;
				buffPtr += simdUnrollSize;
			}
		}
		else if (src.depth() == CV_32F)
		{
			__m512* srcPtr = (__m512*)src.ptr<float>(y);

			for (int x = 0; x < simdWidth; ++x)
			{
				buffPtr[0] = srcPtr[0 * simdWidth];
				buffPtr[1] = srcPtr[1 * simdWidth];
				buffPtr[2] = srcPtr[2 * simdWidth];
				buffPtr[3] = srcPtr[3 * simdWidth];
				buffPtr[4] = srcPtr[4 * simdWidth];
				buffPtr[5] = srcPtr[5 * simdWidth];
				buffPtr[6] = srcPtr[6 * simdWidth];
				buffPtr[7] = srcPtr[7 * simdWidth];
				buffPtr[8] = srcPtr[8 * simdWidth];
				buffPtr[9] = srcPtr[9 * simdWidth];
				buffPtr[10] = srcPtr[10 * simdWidth];
				buffPtr[11] = srcPtr[11 * simdWidth];
				buffPtr[12] = srcPtr[12 * simdWidth];
				buffPtr[13] = srcPtr[13 * simdWidth];
				buffPtr[14] = srcPtr[14 * simdWidth];
				buffPtr[15] = srcPtr[15 * simdWidth];
				_mm512_transpose16_ps(buffPtr);

				++srcPtr;
				buffPtr += simdUnrollSize;
			}
		}
	}

#if 0
	void GaussianFilterSlidingDCT5_AVX512_32F::horizontalFilteringInnerXK1(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 8;//8

		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;
		const int xend = get_simd_ceil(width - (left + right), simdUnrollSize) + xstart;
		int xst = xstart + simdUnrollSize;
		while (xst < radius + 1)
		{
			xst += simdUnrollSize;
		}
		int xed = get_simd_floor(width - radius - 1 - xst, simdUnrollSize) + xst;

		const float cf11 = shift[0], cfR1 = shift[1];

		__m256 a1, b1;
		__m256 sum;
		__m256 sumA, sumB;
		__m256 dvA, dvB, delta;

		int x, y, r;
		for (y = 0; y < height; y += simdUnrollSize)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			sum = buffHFilter[xstart];
			a1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buffHFilter[xstart + 0]);
			b1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buffHFilter[xstart + 1]);

			if (left < radius)
			{
				for (r = 1; r <= left; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
				}

				for (r = left + 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[LREF(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[LREF(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);
					sum = _mm256_add_ps(sum, sumA);

					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
				}

				//for (x = 0; x < simdUnrollSize; x += simdUnrollSize)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 0 + radius + 1)], buffHFilter[LREF(xstart + 0 - radius)]);
					sum = _mm256_add_ps(sum, dvA);

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 1 + radius + 1)], buffHFilter[LREF(xstart + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 2 + radius + 1)], buffHFilter[LREF(xstart + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 3 + radius + 1)], buffHFilter[LREF(xstart + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 4 + radius + 1)], buffHFilter[LREF(xstart + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 5 + radius + 1)], buffHFilter[LREF(xstart + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xstart + simdUnrollSize; x < xst; x += simdUnrollSize)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[LREF(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[LREF(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[LREF(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[LREF(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[LREF(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[LREF(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xst; x < xed; x += simdUnrollSize)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xed; x < xend; x += simdUnrollSize)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}
			}
			else
			{
				__m256* buffHR = &buffHFilter[xstart + 1];
				__m256* buffHL = &buffHFilter[xstart - 1];
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(*buffHL, *buffHR);
					sumB = _mm256_add_ps(*(buffHL + 1), *(buffHR + 1));
					sum = _mm256_add_ps(sum, sumA);
					buffHR++;
					buffHL--;

					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
				}

				buffHR = &buffHFilter[(xstart + 0 + radius + 1)];
				buffHL = &buffHFilter[(xstart + 0 - radius + 0)];
				//for (x = 0; x < 8; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					sum = _mm256_add_ps(sum, dvA);

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}

				const int simdWidth = (xend - (xstart + simdUnrollSize)) / simdUnrollSize;
				//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				for (x = 0; x < simdWidth; ++x)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, a1));
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, b1));
					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}
			}
		}
	}

	void GaussianFilterSlidingDCT5_AVX512_32F::verticalFilteringInnerXYK1(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 8;//8

		const int width = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols / 8;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>(0);
		__m256* dstPtr = (__m256*)dst.ptr<float>(top, xstart);

		const float cf11 = shift[0], cfR1 = shift[1];

		__m256 totalA, totalB;
		__m256 pA, pB;
		__m256* pAP, * pAM, * pBP, * pBM, * pCP, * pCM;
		__m256 dvA, dvB, dvC, deltaB, deltaC;
		__m256* ws = buffVFilter;

		int x, y, r;
		for (x = xstart; x < xend; x += simdUnrollSize)
		{
			pA = _mm256_loadu_ps(&srcPtr[top * width + x]);
			pB = _mm256_loadu_ps(&srcPtr[top * width + x + width]);

			*ws++ = pA;
			*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[0]));
			*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[0]));
		}

		int rstop = min(top, radius);
		if (top < radius)
		{
			for (r = 1; r <= top; ++r)
			{
				pAM = (__m256*) & srcPtr[width * (top + 0 - r) + xstart];
				pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
				pBM = (__m256*) & srcPtr[width * (top + 1 - r) + xstart];
				pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
				ws = buffVFilter;
				for (x = 0; x < simdWidth; ++x)
				{
					pA = _mm256_add_ps(*pAM++, *pAP++);
					pB = _mm256_add_ps(*pBM++, *pBP++);
					*ws++ = _mm256_add_ps(*ws, pA);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
				}
			}
			for (r = top + 1; r <= radius; ++r)
			{
				pAM = (__m256*) & srcPtr[UREF(top + 0 - r) + xstart];
				pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
				pBM = (__m256*) & srcPtr[UREF(top + 1 - r) + xstart];
				pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
				ws = buffVFilter;
				for (x = 0; x < simdWidth; ++x)
				{
					pA = _mm256_add_ps(*pAM++, *pAP++);
					pB = _mm256_add_ps(*pBM++, *pBP++);
					*ws++ = _mm256_add_ps(*ws, pA);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
				}
			}
		}
		else
		{
			for (r = 1; r <= radius; ++r)
			{
				pAM = (__m256*) & srcPtr[width * (top + 0 - r) + xstart];
				pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
				pBM = (__m256*) & srcPtr[width * (top + 1 - r) + xstart];
				pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
				ws = buffVFilter;

				for (x = 0; x < simdWidth; ++x)
				{
					pA = _mm256_add_ps(*pAM++, *pAP++);
					pB = _mm256_add_ps(*pBM++, *pBP++);
					*ws++ = _mm256_add_ps(*ws, pA);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pBP = (__m256*) & srcPtr[width * (top + y + radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[width * (top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaC = _mm256_sub_ps(dvC, dvB);

				totalA = *ws;
				totalB = _mm256_add_ps(totalA, dvB);
				*ws++ = _mm256_add_ps(totalB, dvC);
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf11), *(ws + 1), *(ws)));

					ws += 2;
				}
				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m256*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pAP = (__m256*) & srcPtr[DREF(top + y + radius + 0) + xstart];
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = xstart; x < xend; x += 8)
			{
				dvA = _mm256_sub_ps(*pAP++, *pAM++);
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaB = _mm256_sub_ps(dvB, dvA);
				deltaC = _mm256_sub_ps(dvC, dvB);

				totalA = *ws;
				totalB = _mm256_add_ps(totalA, dvB);
				*ws++ = _mm256_add_ps(totalB, dvC);
				{
					*(ws + 1) = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(cf11), *ws, *(ws + 1)));
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf11), *(ws + 1), *ws));
					ws += 2;
				}
				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), totalB);
				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
	}

	void GaussianFilterSlidingDCT5_AVX512_32F::horizontalFilteringInnerXK2(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;
		const int xend = get_simd_ceil(width - (left + right), 8) + xstart;

		int xst = xstart + 8;
		while (xst < radius + 1)
		{
			xst += 8;
		}
		//int xed = get_simd_floor(width - radius - 8, 8);
		int xed = get_simd_floor(width - radius - 1 - xst, 8) + xst;

		const float cf11 = shift[0], cfR1 = shift[1];
		const float cf12 = shift[2], cfR2 = shift[3];
		int x, y, r;
		__m256 sum, a1, b1, a2, b2;
		__m256 sumA, sumB;
		__m256 dvA, dvB, delta;

		for (y = 0; y < height; y += 8)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			sum = buffHFilter[xstart];
			a1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buffHFilter[xstart]);
			b1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buffHFilter[xstart + 1]);
			a2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), buffHFilter[xstart]);
			b2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), buffHFilter[xstart + 1]);

			if (left < radius)
			{
				for (r = 1; r <= left; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumA, a2);
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumB, b2);
				}
				for (r = left + 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[LREF(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[LREF(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumA, a2);
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumB, b2);
				}

				//for (x = xstart; x < xstart+8; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 0 + radius + 1)], buffHFilter[LREF(xstart + 0 - radius)]);
					sum = _mm256_add_ps(sum, dvA);

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 1 + radius + 1)], buffHFilter[LREF(xstart + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 2 + radius + 1)], buffHFilter[LREF(xstart + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 3 + radius + 1)], buffHFilter[LREF(xstart + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 4 + radius + 1)], buffHFilter[LREF(xstart + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 5 + radius + 1)], buffHFilter[LREF(xstart + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xstart + 8; x < xst; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[LREF(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[LREF(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[LREF(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[LREF(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[LREF(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[LREF(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xst; x < xed; x += 8)
				{
					//RREF
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xed; x < xend; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}
			}
			else
			{
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumA, a2);
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumB, b2);
				}

				//for (x = xstart; x < xstart+8; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 0 + radius + 1)], buffHFilter[(xstart + 0 - radius)]);
					sum = _mm256_add_ps(sum, dvA);

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 1 + radius + 1)], buffHFilter[(xstart + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 2 + radius + 1)], buffHFilter[(xstart + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 3 + radius + 1)], buffHFilter[(xstart + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 4 + radius + 1)], buffHFilter[(xstart + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 5 + radius + 1)], buffHFilter[(xstart + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[(xstart + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[(xstart + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xstart + 8; x < xend; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}
			}
		}
		/*
		const int width = imgSize.width;
		const int height = imgSize.height;

		int pad = radius;
		int xstart = pad;
		int xend = get_simd_ceil(width - 2 * pad, 8) + xstart;

		const float cf11 = shift[0], cfR1 = shift[1];
		const float cf12 = shift[2], cfR2 = shift[3];
		int x, y, r;
		__m256 sum, a1, b1, a2, b2;
		__m256 sumA, sumB;
		__m256 dvA, dvB, delta;

		for (y = 0; y < height; y += 8)
		{
			float* dstPtr = dst.ptr<float>(y, xstart);
			{
				__m256* srcPtr = (__m256*)src.ptr<float>(y);
				for (x = 0; x < width; x += 8)
				{
					for (int i = 0; i < 8; ++i)
					{
						patch[i] = srcPtr[i * width / 8 + 0];
					}
					_mm256_transpose8_ps(patch, buff + x);
					++srcPtr;
				}
			}

			sum = buff[xstart];
			a1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buff[xstart]);
			b1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buff[xstart + 1]);
			a2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), buff[xstart]);
			b2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), buff[xstart + 1]);

			for (r = 1; r <= radius; ++r)
			{
				sumA = _mm256_add_ps(buff[LREF(xstart + 0 - r)], buff[(xstart + 0 + r)]);
				sumB = _mm256_add_ps(buff[LREF(xstart + 1 - r)], buff[(xstart + 1 + r)]);

				sum = _mm256_add_ps(sum, sumA);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[order*r + 0]), sumA, a1);
				b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[order*r + 0]), sumB, b1);
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[order*r + 1]), sumA, a2);
				b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[order*r + 1]), sumB, b2);
			}

			//for (x = xstart; x < xstart+8; x += 8)
			{
				patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
				dvA = _mm256_sub_ps(buff[(xstart + 0 + radius + 1)], buff[LREF(xstart + 0 - radius)]);
				sum = _mm256_add_ps(sum, dvA);

				patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
				dvB = _mm256_sub_ps(buff[(xstart + 1 + radius + 1)], buff[LREF(xstart + 1 - radius)]);
				delta = _mm256_sub_ps(dvB, dvA);
				sum = _mm256_add_ps(sum, dvB);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

				patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
				dvA = _mm256_sub_ps(buff[(xstart + 2 + radius + 1)], buff[LREF(xstart + 2 - radius)]);
				delta = _mm256_sub_ps(dvA, dvB);
				sum = _mm256_add_ps(sum, dvA);
				b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
				b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

				patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
				dvB = _mm256_sub_ps(buff[(xstart + 3 + radius + 1)], buff[LREF(xstart + 3 - radius)]);
				delta = _mm256_sub_ps(dvB, dvA);
				sum = _mm256_add_ps(sum, dvB);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

				patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
				dvA = _mm256_sub_ps(buff[(xstart + 4 + radius + 1)], buff[LREF(xstart + 4 - radius)]);
				delta = _mm256_sub_ps(dvA, dvB);
				sum = _mm256_add_ps(sum, dvA);
				b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
				b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

				patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
				dvB = _mm256_sub_ps(buff[(xstart + 5 + radius + 1)], buff[LREF(xstart + 5 - radius)]);
				delta = _mm256_sub_ps(dvB, dvA);
				sum = _mm256_add_ps(sum, dvB);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

				patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
				dvA = _mm256_sub_ps(buff[(xstart + 6 + radius + 1)], buff[LREF(xstart + 6 - radius)]);
				delta = _mm256_sub_ps(dvA, dvB);
				sum = _mm256_add_ps(sum, dvA);
				b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
				b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

				patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
				dvB = _mm256_sub_ps(buff[(xstart + 7 + radius + 1)], buff[LREF(xstart + 7 - radius)]);
				delta = _mm256_sub_ps(dvB, dvA);
				sum = _mm256_add_ps(sum, dvB);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

				_mm256_transpose8_ps(patch);
				_mm256_storeupatch_ps(dstPtr, patch, width);
				dstPtr += 8;
			}


		}
		*/
	}

	void GaussianFilterSlidingDCT5_AVX512_32F::verticalFilteringInnerXYK2(const cv::Mat& src, cv::Mat& dst)
	{
		const int width = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols / 8;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, 8) + xstart;

		const float* srcPtr = src.ptr<float>(0);
		__m256* dstPtr = (__m256*)dst.ptr<float>(top, xstart);

		const float cf11 = shift[0], cfR1 = shift[1];
		const float cf12 = shift[2], cfR2 = shift[3];
		int x, y, r;
		__m256 b1, a1, b2, a2;
		__m256 sum;
		__m256 pA, pB;
		__m256* pAP, * pAM, * pBP, * pBM, * pCP, * pCM;
		__m256 dvA, dvB, dvC, deltaB, deltaC;
		__m256* ws;

		ws = buffVFilter;
		for (x = xstart; x < xend; x += 8)
		{
			pA = _mm256_load_ps(&srcPtr[top * width + x]);
			pB = _mm256_load_ps(&srcPtr[top * width + x + width]);
			*ws++ = pA;
			*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[0]));
			*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[0]));
			*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[1]));
			*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[1]));
		}

		int rstop = min(top, radius);
		if (top < radius)
		{
			for (r = 1; r <= top; ++r)
			{
				pAM = (__m256*) & srcPtr[width * (top + 0 - r) + xstart];
				pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
				pBM = (__m256*) & srcPtr[width * (top + 1 - r) + xstart];
				pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
				ws = buffVFilter;
				for (x = xstart; x < xend; x += 8)
				{
					pA = _mm256_add_ps(*pAM++, *pAP++);
					pB = _mm256_add_ps(*pBM++, *pBP++);
					*ws++ = _mm256_add_ps(*ws, pA);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
				}
			}
			for (r = top + 1; r <= radius; ++r)
			{
				pAM = (__m256*) & srcPtr[UREF(top + 0 - r) + xstart];
				pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
				pBM = (__m256*) & srcPtr[UREF(top + 1 - r) + xstart];
				pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
				ws = buffVFilter;
				for (x = xstart; x < xend; x += 8)
				{
					pA = _mm256_add_ps(*pAM++, *pAP++);
					pB = _mm256_add_ps(*pBM++, *pBP++);
					*ws++ = _mm256_add_ps(*ws, pA);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
				}
			}
		}
		else
		{
			for (r = 1; r <= radius; ++r)
			{
				pAM = (__m256*) & srcPtr[width * (top + 0 - r) + xstart];
				pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
				pBM = (__m256*) & srcPtr[width * (top + 1 - r) + xstart];
				pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
				ws = buffVFilter;
				for (x = xstart; x < xend; x += 8)
				{
					pA = _mm256_add_ps(*pAM++, *pAP++);
					pB = _mm256_add_ps(*pBM++, *pBP++);
					*ws++ = _mm256_add_ps(*ws, pA);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = xstart; x < xend; x += 8)
			{
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaC = _mm256_sub_ps(dvC, dvB);
				sum = ws[0];
				a1 = ws[1];
				b1 = ws[2];
				a2 = ws[3];
				b2 = ws[4];

				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
				sum = _mm256_add_ps(sum, dvB);

				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
				sum = _mm256_add_ps(sum, dvC);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

				*ws++ = sum;
				*ws++ = a1;
				*ws++ = b1;
				*ws++ = a2;
				*ws++ = b2;
				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m256*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pAP = (__m256*) & srcPtr[DREF(top + y + radius + 0) + xstart];
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = xstart; x < xend; x += 8)
			{
				dvA = _mm256_sub_ps(*pAP++, *pAM++);
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaB = _mm256_sub_ps(dvB, dvA);
				deltaC = _mm256_sub_ps(dvC, dvB);
				sum = ws[0];
				a1 = ws[1];
				b1 = ws[2];
				a2 = ws[3];
				b2 = ws[4];

				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(a1, a2)));
				sum = _mm256_add_ps(sum, dvB);
				b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
				b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));

				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(b1, b2)));
				sum = _mm256_add_ps(sum, dvC);
				a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
				a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));

				*ws++ = sum;
				*ws++ = a1;
				*ws++ = b1;
				*ws++ = a2;
				*ws++ = b2;
				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
	}

	void GaussianFilterSlidingDCT5_AVX512_32F::horizontalFilteringInnerXK3(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 8;

		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;
		const int xend = get_simd_ceil(width - (left + right), simdUnrollSize) + xstart;
		int xst = xstart + simdUnrollSize;
		while (xst < radius + 1)
		{
			xst += simdUnrollSize;
		}
		int xed = get_simd_floor(width - radius - 1 - xst, simdUnrollSize) + xst;

		const float cf11 = shift[0], cfR1 = shift[1];
		const float cf12 = shift[2], cfR2 = shift[3];
		const float cf13 = shift[4], cfR3 = shift[5];

		__m256 a1, b1, a2, b2, a3, b3;
		__m256 sum;
		__m256 sumA, sumB;
		__m256 dvA, dvB, delta;

		int x, y, r;
		for (y = 0; y < height; y += simdUnrollSize)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			sum = buffHFilter[xstart];
			a1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buffHFilter[xstart]);
			b1 = _mm256_mul_ps(_mm256_set1_ps(table[0]), buffHFilter[xstart + 1]);
			a2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), buffHFilter[xstart]);
			b2 = _mm256_mul_ps(_mm256_set1_ps(table[1]), buffHFilter[xstart + 1]);
			a3 = _mm256_mul_ps(_mm256_set1_ps(table[2]), buffHFilter[xstart]);
			b3 = _mm256_mul_ps(_mm256_set1_ps(table[2]), buffHFilter[xstart + 1]);

			if (left < radius)
			{
				for (r = 1; r <= left; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumA, a2);
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumB, b2);
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 2]), sumA, a3);
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 2]), sumB, b3);
				}
				for (r = left + 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[LREF(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[LREF(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumA, a2);
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumB, b2);
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 2]), sumA, a3);
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 2]), sumB, b3);
				}

				//for (x = xstart+0; x < xstart+8; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 0 + radius + 1)], buffHFilter[LREF(xstart + 0 - radius)]);
					sum = _mm256_add_ps(sum, dvA);

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 1 + radius + 1)], buffHFilter[LREF(xstart + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 2 + radius + 1)], buffHFilter[LREF(xstart + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 3 + radius + 1)], buffHFilter[LREF(xstart + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 4 + radius + 1)], buffHFilter[LREF(xstart + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 5 + radius + 1)], buffHFilter[LREF(xstart + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xstart + 8; x < xst; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[LREF(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[LREF(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[LREF(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[LREF(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[LREF(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[LREF(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xst; x < xed; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(buffHFilter[(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(buffHFilter[(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);

					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xed; x < xend; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 4 + radius + 1)], buffHFilter[(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 5 + radius + 1)], buffHFilter[(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));

					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));
					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}
			}
			else
			{
				__m256* buffHR = &buffHFilter[xstart + 1];
				__m256* buffHL = &buffHFilter[xstart - 1];
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(*buffHL, *buffHR);
					sumB = _mm256_add_ps(*(buffHL + 1), *(buffHR + 1));
					buffHR++;
					buffHL--;

					sum = _mm256_add_ps(sum, sumA);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumA, a1);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 0]), sumB, b1);
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumA, a2);
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 1]), sumB, b2);
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 2]), sumA, a3);
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + 2]), sumB, b3);
				}

				buffHR = &buffHFilter[(xstart + 0 + radius + 1)];
				buffHL = &buffHFilter[(xstart + 0 - radius - 0)];
				//for (x = 0; x < 8; x += 8)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					sum = _mm256_add_ps(sum, dvA);
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				const int simdWidth = (xend - (xstart + simdUnrollSize)) / simdUnrollSize;
				//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				for (x = 0; x < simdWidth; ++x)
				{
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a1, a2), a3)));

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					sum = _mm256_add_ps(sum, dvA);
					b1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), a1, b1));
					b2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), a2, b2));
					b3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), a3, b3));
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(b1, b2), b3)));

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					sum = _mm256_add_ps(sum, dvB);
					a1 = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf11), b1, a1));
					a2 = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf12), b2, a2));
					a3 = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), delta, _mm256_fmsub_ps(_mm256_set1_ps(cf13), b3, a3));

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}
			}
		}
	}

	void GaussianFilterSlidingDCT5_AVX512_32F::verticalFilteringInnerXYK3(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 8;//8

		const int width = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols / simdUnrollSize;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>(0);
		__m256* dstPtr = (__m256*)dst.ptr<float>(top, xstart);

		const float cf11 = shift[0], cfR1 = shift[1];
		const float cf12 = shift[2], cfR2 = shift[3];
		const float cf13 = shift[4], cfR3 = shift[5];

		__m256 totalA, totalB;
		__m256 pA, pB;
		__m256* pAP, * pAM, * pBP, * pBM, * pCP, * pCM;
		__m256 dvA, dvB, dvC, deltaB, deltaC;
		__m256* ws = buffVFilter;

		int x, y, r;
		for (x = xstart; x < xend; x += simdUnrollSize)
		{
			pA = _mm256_load_ps(&srcPtr[top * width + x]);
			pB = _mm256_load_ps(&srcPtr[top * width + x + width]);
			*ws++ = pA;
			*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[0]));
			*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[0]));
			*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[1]));
			*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[1]));
			*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[2]));
			*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[2]));
		}

		for (r = 1; r <= radius; ++r)
		{
			pAM = (__m256*) & srcPtr[UREF(top + 0 - r) + xstart];
			pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
			pBM = (__m256*) & srcPtr[UREF(top + 1 - r) + xstart];
			pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];
			ws = buffVFilter;

			for (x = 0; x < simdWidth; ++x)
			{
				pA = _mm256_add_ps(*pAM++, *pAP++);
				pB = _mm256_add_ps(*pBM++, *pBP++);
				*ws++ = _mm256_add_ps(*ws, pA);
				*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
				*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 0]), *ws);
				*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
				*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 1]), *ws);
				*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + 2]), *ws);
				*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + 2]), *ws);
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaC = _mm256_sub_ps(dvC, dvB);

				totalA = *ws;
				totalB = _mm256_add_ps(totalA, dvB);
				*ws++ = _mm256_add_ps(totalB, dvC);
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf11), *(ws + 1), *(ws)));
					ws += 2;

					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf12), *(ws + 1), *(ws)));
					ws += 2;

					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf13), *(ws + 1), *(ws)));
					ws += 2;
				}
				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m256*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pAP = (__m256*) & srcPtr[DREF(top + y + radius + 0) + xstart];
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvA = _mm256_sub_ps(*pAP++, *pAM++);
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaB = _mm256_sub_ps(dvB, dvA);
				deltaC = _mm256_sub_ps(dvC, dvB);

				totalA = *ws;
				totalB = _mm256_add_ps(totalA, dvB);
				*ws++ = _mm256_add_ps(totalB, dvC);

				{
					*(ws + 1) = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(cf11), *ws, *(ws + 1)));
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR1), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf11), *(ws + 1), *ws));
					ws += 2;

					*(ws + 1) = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(cf12), *ws, *(ws + 1)));
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR2), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf12), *(ws + 1), *ws));
					ws += 2;

					*(ws + 1) = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(cf13), *ws, *(ws + 1)));
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(cfR3), deltaC, _mm256_fmsub_ps(_mm256_set1_ps(cf13), *(ws + 1), *ws));
					ws += 2;
				}
				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
	}


	void GaussianFilterSlidingDCT5_AVX512_32F::verticalFilteringInnerXYKn(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 16;//16

		const int width = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols / simdUnrollSize;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>(0);
		__m256* dstPtr = (__m256*)dst.ptr<float>(top, xstart);

		float* shift0 = new float[gf_order + 1];
		float* shift1 = new float[gf_order + 1];
		for (int i = 1; i <= gf_order; i++)
		{
			shift0[i] = shift[2 * (i - 1) + 0];
			shift1[i] = shift[2 * (i - 1) + 1];
		}

		__m256 totalA, totalB;
		__m256 sumA, sumB;
		__m256 pA, pB;
		__m256* pAP, * pAM, * pBP, * pBM, * pCP, * pCM;
		__m256 dvA, dvB, dvC, deltaA, deltaB;
		__m256* ws = buffVFilter;

		int x, y, r, i;
		for (x = xstart; x < xend; x += simdUnrollSize)
		{
			pA = _mm256_load_ps(&srcPtr[top * width + x]);
			pB = _mm256_load_ps(&srcPtr[top * width + x + width]);

			*ws++ = pA;
			for (i = 1; i <= gf_order; ++i)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(table[i - 1]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(table[i - 1]));
			}
		}

		for (r = 1; r <= radius; ++r)
		{
			pAM = (__m256*) & srcPtr[UREF(top + 0 - r) + xstart];
			pAP = (__m256*) & srcPtr[width * (top + 0 + r) + xstart];
			pBM = (__m256*) & srcPtr[UREF(top + 1 - r) + xstart];
			pBP = (__m256*) & srcPtr[width * (top + 1 + r) + xstart];

			ws = buffVFilter;
			for (x = 0; x < simdWidth; ++x)
			{
				pA = _mm256_add_ps(*pAM++, *pAP++);
				pB = _mm256_add_ps(*pBM++, *pBP++);

				*ws++ = _mm256_add_ps(*ws, pA);
				for (i = 1; i <= gf_order; ++i)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(table[gf_order * r + i - 1]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(table[gf_order * r + i - 1]), *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaB = _mm256_sub_ps(dvC, dvB);

				sumA = *ws;
				sumB = _mm256_add_ps(sumA, dvB);
				*ws++ = _mm256_add_ps(sumB, dvC);

				totalB = sumB;
				totalA = sumA;
				for (i = 1; i <= gf_order; ++i)
				{
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					totalA = _mm256_add_ps(totalA, *ws);
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(shift1[i]), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(shift0[i]), *(ws + 1), *(ws)));

					ws += 2;
				}
				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m256*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pAP = (__m256*) & srcPtr[DREF(top + y + radius + 0) + xstart];
			pBM = (__m256*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m256*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m256*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m256* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvA = _mm256_sub_ps(*pAP++, *pAM++);
				dvB = _mm256_sub_ps(*pBP++, *pBM++);
				dvC = _mm256_sub_ps(*pCP++, *pCM++);
				deltaA = _mm256_sub_ps(dvB, dvA);
				deltaB = _mm256_sub_ps(dvC, dvB);

				sumA = *ws;
				sumB = _mm256_add_ps(sumA, dvB);
				*ws++ = _mm256_add_ps(sumB, dvC);

				totalA = sumA;
				totalB = sumB;
				for (i = 1; i <= gf_order; ++i)
				{
					*(ws + 1) = _mm256_fmadd_ps(_mm256_set1_ps(shift1[i]), deltaA, _mm256_fmsub_ps(_mm256_set1_ps(shift0[i]), *ws, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					totalA = _mm256_add_ps(totalA, *ws);
					*ws = _mm256_fmadd_ps(_mm256_set1_ps(shift1[i]), deltaB, _mm256_fmsub_ps(_mm256_set1_ps(shift0[i]), *(ws + 1), *ws));
					ws += 2;
				}
				*dstPtr2 = _mm256_mul_ps(_mm256_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm256_mul_ps(_mm256_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
		delete[] shift0;
		delete[] shift1;
	}

	void GaussianFilterSlidingDCT5_AVX512_32F::horizontalFilteringInnerXKn(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 8;

		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;
		const int xend = get_simd_ceil(width - (left + right), simdUnrollSize) + xstart;
		int xst = xstart + simdUnrollSize;
		while (xst < radius + 1)
		{
			xst += simdUnrollSize;
		}
		int xed = get_simd_floor(width - radius - 1 - xst, simdUnrollSize) + xst;

		__m256* mshift0 = (__m256*)_mm_malloc(sizeof(__m256) * (gf_order + 1), AVX_ALIGNMENT);
		__m256* mshift1 = (__m256*)_mm_malloc(sizeof(__m256) * (gf_order + 1), AVX_ALIGNMENT);
		for (int i = 1; i <= gf_order; ++i)
		{
			mshift0[i] = _mm256_set1_ps(shift[(i - 1) * 2 + 1]);
			mshift1[i] = _mm256_set1_ps(shift[(i - 1) * 2 + 0]);
		}
		__m256 total;

		__m256* a = (__m256*)_mm_malloc(sizeof(__m256) * (gf_order + 1), AVX_ALIGNMENT);
		__m256* b = (__m256*)_mm_malloc(sizeof(__m256) * (gf_order + 1), AVX_ALIGNMENT);

		__m256 sum;
		__m256 sumA, sumB;
		__m256 dvA, dvB, delta;

		int x, y, r, i;
		for (y = 0; y < height; y += simdUnrollSize)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			sum = buffHFilter[xstart];
			for (i = 1; i <= gf_order; ++i)
			{
				a[i] = _mm256_mul_ps(_mm256_set1_ps(table[i - 1]), buffHFilter[xstart + 0]);
				b[i] = _mm256_mul_ps(_mm256_set1_ps(table[i - 1]), buffHFilter[xstart + 1]);
			}

			if (left < radius)
			{
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(buffHFilter[LREF(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm256_add_ps(buffHFilter[LREF(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);
					sum = _mm256_add_ps(sum, sumA);

					for (i = 1; i <= gf_order; ++i)
					{
						a[i] = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + i - 1]), sumA, a[i]);
						b[i] = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + i - 1]), sumB, b[i]);
					}
				}

				//for (x = 0; x < simdUnrollSize; x += simdUnrollSize)
				{
					dvA = _mm256_sub_ps(buffHFilter[(xstart + 0 + radius + 1)], buffHFilter[LREF(xstart + 0 - radius)]);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
					}
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[(xstart + 1 + radius + 1)], buffHFilter[LREF(xstart + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(buffHFilter[(xstart + 2 + radius + 1)], buffHFilter[LREF(xstart + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[(xstart + 3 + radius + 1)], buffHFilter[LREF(xstart + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(buffHFilter[(xstart + 4 + radius + 1)], buffHFilter[LREF(xstart + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[(xstart + 5 + radius + 1)], buffHFilter[LREF(xstart + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));

					}
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}

				for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				{
					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 0 + radius + 1)], buffHFilter[LREF(x + 0 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 1 + radius + 1)], buffHFilter[LREF(x + 1 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 2 + radius + 1)], buffHFilter[LREF(x + 2 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 3 + radius + 1)], buffHFilter[LREF(x + 3 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 4 + radius + 1)], buffHFilter[LREF(x + 4 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 5 + radius + 1)], buffHFilter[LREF(x + 5 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += 8;
				}
			}
			else
			{
				__m256* buffHR = &buffHFilter[xstart + 1];
				__m256* buffHL = &buffHFilter[xstart - 1];
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_ps(*buffHL, *buffHR);
					sumB = _mm256_add_ps(*(buffHL + 1), *(buffHR + 1));
					sum = _mm256_add_ps(sum, sumA);
					buffHR++;
					buffHL--;

					for (i = 1; i <= gf_order; ++i)
					{
						a[i] = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + i - 1]), sumA, a[i]);
						b[i] = _mm256_fmadd_ps(_mm256_set1_ps(table[gf_order * r + i - 1]), sumB, b[i]);
					}
				}

				buffHR = &buffHFilter[(xstart + 0 + radius + 1)];
				buffHL = &buffHFilter[(xstart + 0 - radius + 0)];
				//for (x = 0; x < simdUnrollSize; x += simdUnrollSize)
				{
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
					}
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));

					}
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}

				const int simdWidth = (xend - (xstart + simdUnrollSize)) / simdUnrollSize;
				//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				for (x = 0; x < simdWidth; ++x)
				{
					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[0] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[5] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					dvA = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, a[i]);
						b[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvA);

					dvB = _mm256_sub_ps(*buffHR++, *buffHL++);
					delta = _mm256_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= gf_order; ++i)
					{
						total = _mm256_add_ps(total, b[i]);
						a[i] = _mm256_fmadd_ps(mshift0[i], delta, _mm256_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm256_mul_ps(_mm256_set1_ps(norm), total);
					sum = _mm256_add_ps(sum, dvB);

					_mm256_transpose8_ps(patch);
					_mm256_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}
			}
		}

		_mm_free(mshift0);
		_mm_free(mshift1);
		_mm_free(a);
		_mm_free(b);
	}

#endif
	template<int order>
	void GaussianFilterSlidingDCT5_AVX512_32F::horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 16;

		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;
		const int xend = get_simd_ceil(width - (left + right), simdUnrollSize) + xstart;
		int xst = xstart + simdUnrollSize;
		while (xst < radius + 1)
		{
			xst += simdUnrollSize;
		}
		int xed = get_simd_floor(width - radius - 1 - xst, simdUnrollSize) + xst;

		__m512 mshift0[order + 1];
		__m512 mshift1[order + 1];
		for (int i = 1; i <= order; ++i)
		{
			mshift0[i] = _mm512_set1_ps(shift[(i - 1) * 2 + 1]);
			mshift1[i] = _mm512_set1_ps(shift[(i - 1) * 2 + 0]);
		}

		__m512 total;
		__m512 a[order + 1];
		__m512 b[order + 1];
		__m512 sum;
		__m512 sumA, sumB;
		__m512 dvA, dvB, delta;

		int x, y, r, i;
		for (y = 0; y < height; y += simdUnrollSize)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			sum = buffHFilter[xstart];
			for (i = 1; i <= order; ++i)
			{
				a[i] = _mm512_mul_ps(_mm512_set1_ps(table[i - 1]), buffHFilter[xstart + 0]);
				b[i] = _mm512_mul_ps(_mm512_set1_ps(table[i - 1]), buffHFilter[xstart + 1]);
			}

			if (left < radius)
			{
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm512_add_ps(buffHFilter[LREF(xstart + 0 - r)], buffHFilter[(xstart + 0 + r)]);
					sumB = _mm512_add_ps(buffHFilter[LREF(xstart + 1 - r)], buffHFilter[(xstart + 1 + r)]);
					sum = _mm512_add_ps(sum, sumA);

					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm512_fmadd_ps(_mm512_set1_ps(table[order * r + i - 1]), sumA, a[i]);
						b[i] = _mm512_fmadd_ps(_mm512_set1_ps(table[order * r + i - 1]), sumB, b[i]);
					}
				}

				//for (x = 0; x < simdUnrollSize; x += simdUnrollSize)
				{
					dvA = _mm512_sub_ps(buffHFilter[(xstart + 0 + radius + 1)], buffHFilter[LREF(xstart + 0 - radius)]);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
					}
					patch[0] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 1 + radius + 1)], buffHFilter[LREF(xstart + 1 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 2 + radius + 1)], buffHFilter[LREF(xstart + 2 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 3 + radius + 1)], buffHFilter[LREF(xstart + 3 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 4 + radius + 1)], buffHFilter[LREF(xstart + 4 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 5 + radius + 1)], buffHFilter[LREF(xstart + 5 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));

					}
					patch[5] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[8] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[9] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[10] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[11] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[12] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[13] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[(xstart + 6 + radius + 1)], buffHFilter[LREF(xstart + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[14] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[(xstart + 7 + radius + 1)], buffHFilter[LREF(xstart + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[15] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					_mm512_transpose16_ps(patch);
					_mm512_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}

				for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				{
					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 0 + radius + 1)], buffHFilter[LREF(x + 0 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[0] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 1 + radius + 1)], buffHFilter[LREF(x + 1 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 2 + radius + 1)], buffHFilter[LREF(x + 2 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 3 + radius + 1)], buffHFilter[LREF(x + 3 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 4 + radius + 1)], buffHFilter[LREF(x + 4 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 5 + radius + 1)], buffHFilter[LREF(x + 5 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[5] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[8] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[9] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[10] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[11] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[12] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[13] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(buffHFilter[RREF(x + 6 + radius + 1)], buffHFilter[LREF(x + 6 - radius)]);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[14] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(buffHFilter[RREF(x + 7 + radius + 1)], buffHFilter[LREF(x + 7 - radius)]);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[15] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					_mm512_transpose16_ps(patch);
					_mm512_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}
			}
			else
			{
				__m512* buffHR = &buffHFilter[xstart + 1];
				__m512* buffHL = &buffHFilter[xstart - 1];
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm512_add_ps(*buffHL, *buffHR);
					sumB = _mm512_add_ps(*(buffHL + 1), *(buffHR + 1));
					sum = _mm512_add_ps(sum, sumA);
					buffHR++;
					buffHL--;

					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm512_fmadd_ps(_mm512_set1_ps(table[order * r + i - 1]), sumA, a[i]);
						b[i] = _mm512_fmadd_ps(_mm512_set1_ps(table[order * r + i - 1]), sumB, b[i]);
					}
				}

				buffHR = &buffHFilter[(xstart + 0 + radius + 1)];
				buffHL = &buffHFilter[(xstart + 0 - radius + 0)];
				//for (x = 0; x < simdUnrollSize; x += simdUnrollSize)
				{
					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
					}
					patch[0] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));

					}
					patch[5] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[8] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[9] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[10] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[11] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[12] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[13] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[14] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[15] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					_mm512_transpose16_ps(patch);
					_mm512_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}

				const int simdWidth = (xend - (xstart + simdUnrollSize)) / simdUnrollSize;
				//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				for (x = 0; x < simdWidth; ++x)
				{
					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[0] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[1] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[2] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[3] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[4] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[5] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[6] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[7] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[8] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[9] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[10] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[11] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[12] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[13] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					dvA = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvA, dvB);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, a[i]);
						b[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], a[i], b[i]));
					}
					patch[14] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvA);

					dvB = _mm512_sub_ps(*buffHR++, *buffHL++);
					delta = _mm512_sub_ps(dvB, dvA);
					total = sum;
					for (i = 1; i <= order; ++i)
					{
						total = _mm512_add_ps(total, b[i]);
						a[i] = _mm512_fmadd_ps(mshift0[i], delta, _mm512_fmsub_ps(mshift1[i], b[i], a[i]));
					}
					patch[15] = _mm512_mul_ps(_mm512_set1_ps(norm), total);
					sum = _mm512_add_ps(sum, dvB);

					_mm512_transpose16_ps(patch);
					_mm512_storeupatch_ps(dstPtr, patch, width);
					dstPtr += simdUnrollSize;
				}
			}
		}
	}

	template<int order>
	void GaussianFilterSlidingDCT5_AVX512_32F::verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 16;//16

		const int width = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols / simdUnrollSize;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>(0);
		__m512* dstPtr = (__m512*)dst.ptr<float>(top, xstart);

		float shift0[order + 1];
		float shift1[order + 1];
		for (int i = 1; i <= order; i++)
		{
			shift0[i] = shift[2 * (i - 1) + 0];
			shift1[i] = shift[2 * (i - 1) + 1];
		}
		__m512 t[order + 1];

		__m512 totalA, totalB;
		__m512 pA, pB;
		__m512* pAP, * pAM, * pBP, * pBM, * pCP, * pCM;
		__m512 dvA, dvB, dvC, deltaA, deltaB;
		__m512* ws = buffVFilter;

		int x, y, r, i;
		for (x = xstart; x < xend; x += simdUnrollSize)
		{
			pA = _mm512_loadu_ps(&srcPtr[top * width + x]);
			pB = _mm512_loadu_ps(&srcPtr[top * width + x + width]);

			*ws++ = pA;
			for (i = 1; i <= order; ++i)
			{
				*ws++ = _mm512_mul_ps(pA, _mm512_set1_ps(table[i - 1]));
				*ws++ = _mm512_mul_ps(pB, _mm512_set1_ps(table[i - 1]));
			}
		}

		for (r = 1; r <= radius; ++r)
		{
			pAM = (__m512*)&srcPtr[UREF(top + 0 - r) + xstart];
			pAP = (__m512*)&srcPtr[width * (top + 0 + r) + xstart];
			pBM = (__m512*)&srcPtr[UREF(top + 1 - r) + xstart];
			pBP = (__m512*)&srcPtr[width * (top + 1 + r) + xstart];

			ws = buffVFilter;

			for (i = 1; i <= order; ++i)
			{
				t[i] = _mm512_set1_ps(table[order * r + i - 1]);
			}

			for (x = 0; x < simdWidth; ++x)
			{
				pA = _mm512_add_ps(*pAM++, *pAP++);
				pB = _mm512_add_ps(*pBM++, *pBP++);

				*ws++ = _mm512_add_ps(*ws, pA);
				for (i = 1; i <= order; ++i)
				{
					//*ws++ = _mm512_fmadd_ps(pA, _mm512_set1_ps(table[order*r + i - 1]), *ws);
					//*ws++ = _mm512_fmadd_ps(pB, _mm512_set1_ps(table[order*r + i - 1]), *ws);
					*ws++ = _mm512_fmadd_ps(pA, t[i], *ws);
					*ws++ = _mm512_fmadd_ps(pB, t[i], *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m512*)&srcPtr[UREF(top + y - radius + 0) + xstart];
			pCM = (__m512*)&srcPtr[UREF(top + y - radius + 1) + xstart];
			pBP = (__m512*)&srcPtr[width * (top + y + radius + 1) + xstart];
			pCP = (__m512*)&srcPtr[width * (top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m512* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvB = _mm512_sub_ps(*pBP++, *pBM++);
				dvC = _mm512_sub_ps(*pCP++, *pCM++);
				deltaB = _mm512_sub_ps(dvC, dvB);

				totalA = *ws;
				totalB = _mm512_add_ps(totalA, dvB);
				*ws++ = _mm512_add_ps(totalB, dvC);

				for (i = 1; i <= order; ++i)
				{
					totalA = _mm512_add_ps(totalA, *ws);
					totalB = _mm512_add_ps(totalB, *(ws + 1));
					*ws = _mm512_fmadd_ps(_mm512_set1_ps(shift1[i]), deltaB, _mm512_fmsub_ps(_mm512_set1_ps(shift0[i]), *(ws + 1), *(ws)));

					ws += 2;
				}
				*dstPtr2 = _mm512_mul_ps(_mm512_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm512_mul_ps(_mm512_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m512*)&srcPtr[UREF(top + y - radius - 1) + xstart];
			pAP = (__m512*)&srcPtr[DREF(top + y + radius + 0) + xstart];
			pBM = (__m512*)&srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m512*)&srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m512*)&srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m512*)&srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m512* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvA = _mm512_sub_ps(*pAP++, *pAM++);
				dvB = _mm512_sub_ps(*pBP++, *pBM++);
				dvC = _mm512_sub_ps(*pCP++, *pCM++);
				deltaA = _mm512_sub_ps(dvB, dvA);
				deltaB = _mm512_sub_ps(dvC, dvB);

				totalA = *ws;
				totalB = _mm512_add_ps(totalA, dvB);
				*ws++ = _mm512_add_ps(totalB, dvC);

				for (i = 1; i <= order; ++i)
				{
					*(ws + 1) = _mm512_fmadd_ps(_mm512_set1_ps(shift1[i]), deltaA, _mm512_fmsub_ps(_mm512_set1_ps(shift0[i]), *ws, *(ws + 1)));
					totalA = _mm512_add_ps(totalA, *ws);
					totalB = _mm512_add_ps(totalB, *(ws + 1));
					*ws = _mm512_fmadd_ps(_mm512_set1_ps(shift1[i]), deltaB, _mm512_fmsub_ps(_mm512_set1_ps(shift0[i]), *(ws + 1), *ws));
					ws += 2;
				}
				*dstPtr2 = _mm512_mul_ps(_mm512_set1_ps(norm), totalA);
				*(dstPtr2 + dstStep) = _mm512_mul_ps(_mm512_set1_ps(norm), totalB);

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
	}


	void GaussianFilterSlidingDCT5_AVX512_32F::body(const cv::Mat& src, cv::Mat& dst)
	{
#ifndef __AVX512__
		src.copyTo(dst);
		return;
#endif

		if (dst.size() != imgSize || dst.depth() != depth)
			dst.create(imgSize, depth);

		switch (gf_order)
		{
		case 1:
			//horizontalFilteringInnerXK1(src, inter);
			//verticalFilteringInnerXYK1(inter, dst);
			horizontalFilteringInnerXK<1>(src, inter);
			verticalFilteringInnerXYK<1>(inter, dst);
			break;

		case 2:
			//horizontalFilteringInnerXK2(src, inter);
			//verticalFilteringInnerXYK2(inter, dst);

			horizontalFilteringInnerXK<2>(src, inter);
			verticalFilteringInnerXYK<2>(inter, dst);
			break;
		case 3:
			//horizontalFilteringInnerXK3(src, inter);
			//verticalFilteringInnerXYK3(inter, dst);

			horizontalFilteringInnerXK<3>(src, inter);
			verticalFilteringInnerXYK<3>(inter, dst);
			break;
		case 4:
			horizontalFilteringInnerXK<4>(src, inter);
			verticalFilteringInnerXYK<4>(inter, dst);
			break;

		case 5:
			horizontalFilteringInnerXK<5>(src, inter);
			verticalFilteringInnerXYK<5>(inter, dst);
			break;

		case 6:
			horizontalFilteringInnerXK<6>(src, inter);
			verticalFilteringInnerXYK<6>(inter, dst);
			break;

		case 7:
			horizontalFilteringInnerXK<7>(src, inter);
			verticalFilteringInnerXYK<7>(inter, dst);
			break;

		case 8:
			horizontalFilteringInnerXK<8>(src, inter);
			verticalFilteringInnerXYK<8>(inter, dst);
			break;

		case 9:
			horizontalFilteringInnerXK<9>(src, inter);
			verticalFilteringInnerXYK<9>(inter, dst);
			break;
		case 10:
			horizontalFilteringInnerXK<10>(src, inter);
			verticalFilteringInnerXYK<10>(inter, dst);
			break;

		case 11:
			horizontalFilteringInnerXK<11>(src, inter);
			verticalFilteringInnerXYK<11>(inter, dst);
			break;

		case 12:
			horizontalFilteringInnerXK<12>(src, inter);
			verticalFilteringInnerXYK<12>(inter, dst);
			break;

		case 13:
			horizontalFilteringInnerXK<13>(src, inter);
			verticalFilteringInnerXYK<13>(inter, dst);
			break;

		case 14:
			horizontalFilteringInnerXK<14>(src, inter);
			verticalFilteringInnerXYK<14>(inter, dst);
			break;

		case 15:
			horizontalFilteringInnerXK<15>(src, inter);
			verticalFilteringInnerXYK<15>(inter, dst);
			break;

		default:
			//horizontalFilteringInnerXKn(src, inter);
			//verticalFilteringInnerXYKn(inter, dst);
			break;
		}
	}
#endif
#pragma endregion

#pragma region SlidingDCT5_AVX_64F

	void SpatialFilterSlidingDCT5_AVX_64F::allocBuffer()
	{
		const int simdUnrollSize = 4;
		const int internalWidth = get_simd_ceil(imgSize.width, simdUnrollSize);
		const int internalHeight = get_simd_ceil(imgSize.height, simdUnrollSize) + simdUnrollSize;
		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			inter.create(Size(internalWidth, internalHeight), CV_32F);
		}
		else
		{
			inter.create(Size(internalWidth, internalHeight), CV_64F);
		}

		_mm_free(fn_hfilter);
		_mm_free(buffVFilter);
		fn_hfilter = (__m256d*)_mm_malloc((internalWidth + get_simd_ceil(2 * radius + 3, simdUnrollSize)) * sizeof(__m256d), AVX_ALIGNMENT);
		buffVFilter = (__m256d*)_mm_malloc(int((2 * gf_order + 1) * internalWidth / simdUnrollSize) * sizeof(__m256d), AVX_ALIGNMENT);
	}

	int SpatialFilterSlidingDCT5_AVX_64F::getRadius(const double sigma, const int order)
	{
		cv::AutoBuffer<double> spect(order + 1);
		if (radius == 0)
			return argminR_BruteForce_DCT(sigma, order, 5, spect, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
		else return radius;
	}

	void SpatialFilterSlidingDCT5_AVX_64F::computeRadius(const int rad, const bool isOptimize)
	{
		if (gf_order == 0)
		{
			radius = rad;
			return;
		}

		cv::AutoBuffer<double> Gk(gf_order + 1);
		if (rad == 0)
		{
			radius = argminR_BruteForce_DCT(sigma, gf_order, 5, Gk, isOptimize);
		}
		else
		{
			radius = rad;
		}

		const double omega = CV_2PI / (2.0 * radius + 1.0);

		if (isOptimize)
		{
			bool ret = optimizeSpectrum(sigma, gf_order, radius, 5, Gk, 0);
			//if (!ret)cout << "does not optimized (DCT-5 64F)" << endl;
		}
		else computeSpectrumGaussianClosedForm(sigma, gf_order, radius, 5, Gk);

		_mm_free(GCn);
		const int GCnSize = (gf_order + 0) * (radius + 1);//for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		GCn = (double*)_mm_malloc(GCnSize * sizeof(double), AVX_ALIGN);

		double totalInv = 0;
		generateCosKernel(GCn, totalInv, 5, Gk, radius, gf_order);
		for (int i = 0; i < GCnSize; i++)
		{
			GCn[i] *= totalInv;
		}
		G0 = Gk[0] * totalInv;

		_mm_free(shift);
		shift = (double*)_mm_malloc((2 * gf_order) * sizeof(double), AVX_ALIGN);
		for (int k = 1; k <= gf_order; ++k)
		{
			const double C1 = cos(k * omega * 1);
			const double CR = cos(k * omega * radius);
			shift[2 * (k - 1) + 0] = C1 * 2.0;
			shift[2 * (k - 1) + 1] = CR * Gk[k] * totalInv;
		}
	}

	SpatialFilterSlidingDCT5_AVX_64F::SpatialFilterSlidingDCT5_AVX_64F(cv::Size imgSize, double sigma, int order, const bool isBuff32F)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		//cout << "init sliding DCT5 AVX 64F" << endl;
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT5_64_AVX;
		this->gf_order = order;
		this->sigma = sigma;
		this->schedule = SLIDING_DCT_SCHEDULE::DEFAULT;
		computeRadius(radius, true);

		this->imgSize = imgSize;
		allocBuffer();
	}

	SpatialFilterSlidingDCT5_AVX_64F::SpatialFilterSlidingDCT5_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule, const SpatialKernel skernel)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT5_64_AVX;
		this->schedule = schedule;
		this->depth = CV_64F;
		this->dest_depth = dest_depth;
		this->dct_coeff_method = method;
	}

	SpatialFilterSlidingDCT5_AVX_64F::~SpatialFilterSlidingDCT5_AVX_64F()
	{
		_mm_free(GCn);
		_mm_free(shift);
		_mm_free(buffVFilter);
		_mm_free(fn_hfilter);
	}


	void SpatialFilterSlidingDCT5_AVX_64F::interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad)
	{
		const int pad = inter.cols - imgSize.width;
		const int R = radius + 2;
		const int Rm = radius + 1;//left side does not access r+2

		int length = src.cols - right - left + Rm + R;
		bool isLpad = false;
		bool isRpad = false;
		int xst = 0;
		int xed = src.cols;
		if (left < Rm)
		{
			isLpad = true;
			length -= Rm;
		}
		else
		{
			xst = left - Rm;
		}
		if (src.cols - right + R > src.cols)
		{
			isRpad = true;
			length -= R;
		}
		else
		{
			xed = src.cols - right;
		}
		const int mainloop_simdsize = get_simd_floor(length, 4) / 4;
		const int L = mainloop_simdsize * 4;
		const int rem = length - L;
		if (xst + L > imgSize.width) xst -= (xst + L - imgSize.width);

		__m256d* dest = &this->fn_hfilter[Rm];
		__m256d* buffPtr = dest + xst;
		const int step0 = 0 * imgSize.width;
		const int step1 = 1 * imgSize.width;
		const int step2 = 2 * imgSize.width;
		const int step3 = 3 * imgSize.width;

		if (vpad == 0)
		{
			const __m128i gidx = _mm_setr_epi32(step0, step1, step2, step3);
			if (src.depth() == CV_64F)
			{
				const double* ptr = src.ptr<double>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					buffPtr[0] = _mm256_loadu_pd(ptr + step0);
					buffPtr[1] = _mm256_loadu_pd(ptr + step1);
					buffPtr[2] = _mm256_loadu_pd(ptr + step2);
					buffPtr[3] = _mm256_loadu_pd(ptr + step3);
					_mm256_transpose4_pd(buffPtr);
					ptr += 4;
					buffPtr += 4;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_i32gather_pd(ptr, gidx, sizeof(double));
					ptr++;
					buffPtr++;
				}
			}
			else if (src.depth() == CV_32F)
			{
				const float* ptr = src.ptr<float>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					buffPtr[0] = _mm256_loadu_cvtps_pd(ptr + step0);
					buffPtr[1] = _mm256_loadu_cvtps_pd(ptr + step1);
					buffPtr[2] = _mm256_loadu_cvtps_pd(ptr + step2);
					buffPtr[3] = _mm256_loadu_cvtps_pd(ptr + step3);
					_mm256_transpose4_pd(buffPtr);
					ptr += 4;
					buffPtr += 4;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_cvtps_pd(_mm_i32gather_ps(ptr, gidx, sizeof(float)));
					ptr++;
					buffPtr++;
				}
			}
			else if (src.depth() == CV_8U)
			{
				const uchar* ptr = src.ptr<uchar>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					buffPtr[0] = _mm256_load_cvtepu8_pd(ptr + step0);
					buffPtr[1] = _mm256_load_cvtepu8_pd(ptr + step1);
					buffPtr[2] = _mm256_load_cvtepu8_pd(ptr + step2);
					buffPtr[3] = _mm256_load_cvtepu8_pd(ptr + step3);
					_mm256_transpose4_pd(buffPtr);
					ptr += 4;
					buffPtr += 4;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_cvtepu8_pd(_mm_i32gather_epu8(ptr, gidx));
					ptr++;
					buffPtr++;
				}
			}
		}
		else
		{
			int step[4];
			for (int i = 0; i < vpad; i++)step[i] = imgSize.width * i;
			for (int i = vpad; i < 4; i++)step[i] = step[vpad - 1];
			const __m128i gidx = _mm_setr_epi32(step[0], step[1], step[2], step[3]);
			if (src.depth() == CV_64F)
			{
				const double* ptr = src.ptr<double>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					for (int i = 0; i < vpad; i++)
					{
						buffPtr[i] = _mm256_loadu_pd(ptr + imgSize.width * i);
					}
					_mm256_transpose4_pd(buffPtr);
					ptr += 4;
					buffPtr += 4;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_i32gather_pd(ptr, gidx, sizeof(double));
					ptr++;
					buffPtr++;
				}
			}
			else if (src.depth() == CV_32F)
			{
				const float* ptr = src.ptr<float>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					for (int i = 0; i < vpad; i++)
					{
						buffPtr[i] = _mm256_loadu_cvtps_pd(ptr + imgSize.width * i);
					}
					_mm256_transpose4_pd(buffPtr);
					ptr += 4;
					buffPtr += 4;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_cvtps_pd(_mm_i32gather_ps(ptr, gidx, sizeof(float)));
					ptr++;
					buffPtr++;
				}
			}
			else if (src.depth() == CV_8U)
			{
				const uchar* ptr = src.ptr<uchar>(y, xst);
				for (int x = 0; x < mainloop_simdsize; ++x)
				{
					for (int i = 0; i < vpad; i++)
					{
						buffPtr[i] = _mm256_load_cvtepu8_pd(ptr + imgSize.width * i);
					}
					_mm256_transpose4_pd(buffPtr);
					ptr += 4;
					buffPtr += 4;
				}
				for (int x = 0; x < rem; ++x)
				{
					*buffPtr = _mm256_cvtepu8_pd(_mm_i32gather_epu8(ptr, gidx));
					ptr++;
					buffPtr++;
				}
			}
		}

		if (isLpad && isRpad)
		{
			//print_debug7(radius, left, src.cols-right, xst, xed, length, L);
			switch (borderType)
			{
			case cv::BORDER_REPLICATE:
			{
				for (int x = 1; x < Rm; ++x)
				{
					dest[-x] = dest[0];
				}
				for (int x = 1; x < R; ++x)
				{
					dest[src.cols - 1 + x] = dest[src.cols - 1];
				}
			}
			break;

			case cv::BORDER_REFLECT:
			{
				for (int x = 1; x < Rm; ++x)
				{
					dest[-x] = dest[x - 1];
				}
				for (int x = 1; x < R; ++x)
				{
					dest[src.cols - 1 + x] = dest[src.cols - 1 - x + 1];
				}
			}
			break;

			default:
			case cv::BORDER_REFLECT101:
			{
				for (int x = 1; x < Rm; ++x)
				{
					dest[-x] = dest[x];
				}
				for (int x = 1; x < R; ++x)
				{
					dest[src.cols - 1 + x] = dest[src.cols - 1 - x];
				}
			}
			break;
			}
		}
		else
		{
			if (isLpad)
			{
				switch (borderType)
				{
				case cv::BORDER_REPLICATE:
				{
					for (int x = 1; x < Rm; ++x)
					{
						dest[-x] = dest[0];
					}
				}
				break;

				case cv::BORDER_REFLECT:
				{
					for (int x = 1; x < Rm; ++x)
					{
						dest[-x] = dest[x - 1];
					}
				}
				break;

				default:
				case cv::BORDER_REFLECT101:
				{
					for (int x = 1; x < Rm; ++x)
					{
						dest[-x] = dest[x];
					}
				}
				break;
				}
			}
			if (isRpad)
			{
				switch (borderType)
				{
				case cv::BORDER_REPLICATE:
				{
					for (int x = 1; x < R; ++x)
					{
						dest[src.cols - 1 + x] = dest[src.cols - 1];
					}
				}
				break;

				case cv::BORDER_REFLECT:
				{
					for (int x = 1; x < R; ++x)
					{
						dest[src.cols - 1 + x] = dest[src.cols - 1 - x + 1];
					}
				}
				break;

				default:
				case cv::BORDER_REFLECT101:
				{
					for (int x = 1; x < R; ++x)
					{
						dest[src.cols - 1 + x] = dest[src.cols - 1 - x];
					}
				}
				break;
				}
			}
		}
	}

	//Naive DCT Convolution: O(KR)
	void SpatialFilterSlidingDCT5_AVX_64F::horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 4;//4

		const int xstart = left;//left
		const int xend = get_simd_ceil(imgSize.width - (left + right), simdUnrollSize) + xstart;
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		__m256d total[4];
		__m256d F0;
		AutoBuffer<__m256d> Z(order);
		__m256d* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = 0; y < imgSize.height; y += simdUnrollSize)
		{
			interleaveVerticalPixel(const_cast<Mat&>(src), y, borderType);

			double* dstPtr = dst.ptr<double>(y, xstart);

			for (int x = xstart; x < xend; x += simdUnrollSize)
			{
				for (int j = 0; j < simdUnrollSize; j++)
				{
					// 1) initilization of Z (1 <= n <= radius)
					for (int k = 0; k <= order; ++k)Z[k] = _mm256_setzero_pd();
					F0 = _mm256_setzero_pd();

					for (int n = radius; n >= 1; --n)
					{
						const __m256d sumA = _mm256_add_pd(fn_hfilter[ref_lborder(x + j - n, borderType)], fn_hfilter[ref_rborder(x + j + n, imgSize.width, borderType)]);
						F0 = _mm256_add_pd(F0, sumA);
						double* Cn_ = GCn + order * n;
						for (int k = 0; k < order; ++k)
						{
							__m256d Cnk = _mm256_set1_pd(Cn_[k]);
							Z[k] = _mm256_fmadd_pd(Cnk, sumA, Z[k]);
						}
					}

					// 1) initilization of Z (n=0) adding small->large
					F0 = _mm256_add_pd(F0, fn_hfilter[x + j]);
					for (int k = 0; k < order; ++k)
					{
						__m256d Cnk = _mm256_set1_pd(GCn[k]);
						Z[k] = _mm256_fmadd_pd(Cnk, fn_hfilter[x + j], Z[k]);
					}

					total[j] = _mm256_setzero_pd();
					for (int k = order - 1; k >= 0; --k)
					{
						total[j] = _mm256_add_pd(total[j], Z[k]);
					}
					total[j] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[j]);
				}
				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, imgSize.width);
				dstPtr += simdUnrollSize;
			}
		}
	}

	//64F
	template<int order>
	void SpatialFilterSlidingDCT5_AVX_64F::horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 4;

		const int dwidth = dst.cols;//not always same as width for cache thrashing

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVECD C1_2[order];
		SETVECD CR_g[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
		}

		__m256d total[4];
		__m256d F0;
		__m256d Zp[order];
		__m256d Zc[order];
		__m256d delta_inner;//f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256d dc;//f(x+R+1)-f(x-R)
		__m256d dp;//f(x+R)-f(x-R-1)

		__m256d* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 4)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			double* dstPtr = dst.ptr<double>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			F0 = fn_hfilter[xstart];
			for (int k = 0; k < order; ++k)
			{
				__m256d Cnk = _mm256_set1_pd(GCn[k]);
				Zp[k] = _mm256_mul_pd(Cnk, fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_pd(Cnk, fn_hfilter[xstart + 1]);
			}

			for (int n = 1; n <= radius; ++n)
			{
				const __m256d sumA = _mm256_add_pd(fn_hfilter[xstart + 0 - n], fn_hfilter[xstart + 0 + n]);
				const __m256d sumB = _mm256_add_pd(fn_hfilter[xstart + 1 - n], fn_hfilter[xstart + 1 + n]);
				F0 = _mm256_add_pd(F0, sumA);
				double* Cn_ = GCn + order * n;
				for (int k = 0; k < order; ++k)
				{
					__m256d Cnk = _mm256_set1_pd(Cn_[k]);
					Zp[k] = _mm256_fmadd_pd(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_pd(Cnk, sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				dc = _mm256_sub_pd(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);

				total[0] = Zp[order - 1];
				for (int k = order - 2; k >= 0; k--)
				{
					total[0] = _mm256_add_pd(total[0], Zp[k]);
				}
				total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_sub_pd(dp, dc);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[1] = _mm256_add_pd(total[1], Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_sub_pd(dc, dp);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[2] = _mm256_add_pd(total[2], Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zp[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zc[k]));
				}
				total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_sub_pd(dp, dc);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[3] = _mm256_add_pd(total[3], Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}

			// 3) main loop
			__m256d* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];//f(x+R+1)
			__m256d* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius + 0];//f(x-R)
			for (int x = 0; x < mainloop_simdsize; x++)
			{
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dc, dp);
				total[0] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[0] = _mm256_add_pd(total[0], Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zp[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zc[k]));
				}
				total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
				F0 = _mm256_add_pd(F0, dc);//computing F0(x+1) = F0(x)+dc(x)

				dp = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dp, dc);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[1] = _mm256_add_pd(total[1], Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dc, dp);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[2] = _mm256_add_pd(total[2], Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zp[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zc[k]));
				}
				total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dp, dc);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total[3] = _mm256_add_pd(total[3], Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}
		}
	}

	//64F
	template<int order, typename destT>
	void SpatialFilterSlidingDCT5_AVX_64F::verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 4;//4

		const int swidth = src.cols;
		const int dwidth = dst.cols;

		const int xstart = left;
		const int xlength = dst.cols - left - right;
		const int xlengthC = get_simd_ceil(xlength, simdUnrollSize);
		const int xend = xstart + xlengthC;//xendC
		const int rem = xlength - get_simd_floor(xlength, simdUnrollSize);
		const int simdWidth = xlengthC / simdUnrollSize;
		const int simdend = (rem == 0) ? simdWidth : simdWidth - 1;

		const int ylength = dst.rows - (top + bottom);
		const bool isEven = ((ylength) % 2 == 0);
		const int hend = ylength + ((isEven) ? 0 : -1);

		const double* srcPtr = src.ptr<double>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		SETVECD C1_2[order];
		SETVECD CR_g[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
		}

		__m256d totalA, totalB;
		__m256d deltaA, deltaB;
		__m256d dp, dc, dn;

		__m256d* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 4)
		{
			const __m256d pA = _mm256_loadu_pd(&srcPtr[(top + 0) * swidth + x]);
			const __m256d pB = _mm256_loadu_pd(&srcPtr[(top + 1) * swidth + x]);

			for (int k = order - 1; k >= 0; k--)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(GCn[k]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(GCn[k]));
			}
			*ws++ = pA;
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		for (int r = 1; r <= radius; ++r)
		{
			double* pAM = const_cast<double*>(&srcPtr[ref_tborder(top + 0 - r, swidth, borderType) + xstart]);
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + 1 - r, swidth, borderType) + xstart]);
			double* pAP = const_cast<double*>(&srcPtr[swidth * (top + 0 + r) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[swidth * (top + 1 + r) + xstart]);

			ws = buffVFilter;

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256d pA = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pAP));
				const __m256d pB = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pBP));
				pAP += 4;
				pBP += 4;
				pAM += 4;
				pBM += 4;

				for (int k = order - 1; k >= 0; k--)
				{
					*ws++ = _mm256_fmadd_pd(pA, _mm256_set1_pd(GCn[order * r + k]), *ws);
					*ws++ = _mm256_fmadd_pd(pB, _mm256_set1_pd(GCn[order * r + k]), *ws);
				}
				*ws++ = _mm256_add_pd(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			double* pCM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[swidth * (top + y + radius + 1) + xstart]);
			double* pCP = const_cast<double*>(&srcPtr[swidth * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dc = _mm256_sub_pd(_mm256_loadu_pd(pBP), _mm256_loadu_pd(pBM));
				dn = _mm256_sub_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pCM));
				deltaB = _mm256_sub_pd(dn, dc);
				pBP += 4;
				pCP += 4;
				pBM += 4;
				pCM += 4;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int k = order - 2; k >= 0; k--)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA));
				__m256d temp = _mm256_add_pd(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB));
				*ws++ = _mm256_add_pd(temp, dn);

				dstPtr2 += 4;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_pd(_mm256_loadu_pd(pBP), _mm256_loadu_pd(pBM));
				dn = _mm256_sub_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pCM));
				deltaB = _mm256_sub_pd(dn, dc);
				pBP += 4;
				pCP += 4;
				pBM += 4;
				pCM += 4;

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaB, *ws));
				ws += 2;

				for (int k = order - 2; k >= 0; k--)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
				__m256d temp = _mm256_add_pd(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_pd(temp, dn);
			}
			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			double* pAM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			double* pCM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			double* pAP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			double* pCP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_pd(_mm256_loadu_pd(pAP), _mm256_loadu_pd(pAM));
				dc = _mm256_sub_pd(_mm256_loadu_pd(pBP), _mm256_loadu_pd(pBM));
				dn = _mm256_sub_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pCM));
				deltaA = _mm256_sub_pd(dc, dp);
				deltaB = _mm256_sub_pd(dn, dc);
				pAP += 4;
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;
				pCM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int k = order - 2; k >= 0; k--)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaA, *(ws + 1)));
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA));
				__m256d temp = _mm256_add_pd(*ws, dc);
				store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB));
				*ws++ = _mm256_add_pd(temp, dn);

				dstPtr2 += 4;
			}
			if (rem != 0)
			{
				dp = _mm256_sub_pd(_mm256_loadu_pd(pAP), _mm256_loadu_pd(pAM));
				dc = _mm256_sub_pd(_mm256_loadu_pd(pBP), _mm256_loadu_pd(pBM));
				dn = _mm256_sub_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pCM));
				deltaA = _mm256_sub_pd(dc, dp);
				deltaB = _mm256_sub_pd(dn, dc);
				pAP += 4;
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;
				pCM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int k = order - 2; k >= 0; k--)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaA, *(ws + 1)));
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
				__m256d temp = _mm256_add_pd(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB), rem);
				*ws++ = _mm256_add_pd(temp, dn);
			}

			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;

			double* pAM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			double* pAP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 0, swidth, imgSize.height, borderType) + xstart]);
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			double* pCM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 1, swidth, borderType) + xstart]);
			double* pCP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				dp = _mm256_sub_pd(_mm256_loadu_pd(pAP), _mm256_loadu_pd(pAM));
				dc = _mm256_sub_pd(_mm256_loadu_pd(pBP), _mm256_loadu_pd(pBM));
				dn = _mm256_sub_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pCM));
				deltaA = _mm256_sub_pd(dc, dp);
				deltaB = _mm256_sub_pd(dn, dc);
				pAP += 4;
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;
				pCM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int k = order - 2; k >= 0; k--)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaA, *(ws + 1)));
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaB, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA));
				__m256d temp = _mm256_add_pd(*ws, dc);
				*ws++ = _mm256_add_pd(temp, dn);

				dstPtr2 += 4;
			}
			if (rem != 0)
			{
				dp = _mm256_sub_pd(_mm256_loadu_pd(pAP), _mm256_loadu_pd(pAM));
				dc = _mm256_sub_pd(_mm256_loadu_pd(pBP), _mm256_loadu_pd(pBM));
				dn = _mm256_sub_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pCM));
				deltaA = _mm256_sub_pd(dc, dp);
				deltaB = _mm256_sub_pd(dn, dc);
				pAP += 4;
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;
				pCM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaA, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), deltaB, *ws));
				ws += 2;
				for (int k = order - 2; k >= 0; k--)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaA, *(ws + 1)));
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaB, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
				__m256d temp = _mm256_add_pd(*ws, dc);
				*ws++ = _mm256_add_pd(temp, dn);
			}
		}
	}

	template<int order>
	void SpatialFilterSlidingDCT5_AVX_64F::horizontalFilteringInnerXStore32FK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 4;

		const int dwidth = dst.cols;//not always same as width for cache thrashing
		const int dheight = dst.rows;

		const int xstart = left;
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), imgSize.width, simdUnrollSize);

		SETVECD C1_2[order];
		SETVECD CR_g[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
		}

		__m128 total_sse[4];
		__m256d total;
		__m256d F0;
		__m256d Zp[order];
		__m256d Zc[order];
		__m256d delta_inner;//f(x+R+1)-f(x-R)-f(x+R)+f(x-R-1)
		__m256d dc, dp;

		for (int y = 0; y < dheight; y += simdUnrollSize)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : dheight - imgSize.height;
			interleaveVerticalPixel(src, y, borderType, vpad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			F0 = fn_hfilter[xstart];
			for (int k = 0; k < order; ++k)
			{
				Zp[k] = _mm256_mul_pd(_mm256_set1_pd(GCn[k - 1]), fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_pd(_mm256_set1_pd(GCn[k - 1]), fn_hfilter[xstart + 1]);
			}

			__m256d* buffHR = &fn_hfilter[xstart + 1];
			__m256d* buffHL = &fn_hfilter[xstart - 1];
			for (int n = 1; n <= radius; ++n)
			{
				const __m256d sumA = _mm256_add_pd(*buffHL, *buffHR);
				const __m256d sumB = _mm256_add_pd(*(buffHL + 1), *(buffHR + 1));
				F0 = _mm256_add_pd(F0, sumA);
				buffHR++;
				buffHL--;

				for (int k = 0; k < order; ++k)
				{
					Zp[k] = _mm256_fmadd_pd(_mm256_set1_pd(GCn[order * n + k - 1]), sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_mm256_set1_pd(GCn[order * n + k - 1]), sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			buffHR = &fn_hfilter[(xstart + 0 + radius + 1)];
			buffHL = &fn_hfilter[(xstart + 0 - radius + 0)];
			{
				dc = _mm256_sub_pd(*buffHR++, *buffHL++);

				total = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total = _mm256_add_pd(total, Zp[i]);
				}
				total_sse[0] = _mm256_cvtpd_ps(_mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total));
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dp, dc);
				total = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total_sse[1] = _mm256_cvtpd_ps(_mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total));
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dc, dp);
				total = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zp[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zc[k]));
				}
				total_sse[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_MM256_SET_VECD(mG0), total));
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dp, dc);
				total = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total_sse[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_MM256_SET_VECD(mG0), total));
				//F0 = _mm256_add_pd(F0, dp);

				_MM_TRANSPOSE4_PS(total_sse[0], total_sse[1], total_sse[2], total_sse[3]);
				_mm_storepatch_ps(dstPtr, total_sse, dwidth);
				dstPtr += 4;
			}
			// 3) main loop
			const int SIMDSIZE = (xend - xstart) / simdUnrollSize - 1;
			//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
			for (int x = 0; x < SIMDSIZE; ++x)
			{
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dc, dp);
				total = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zp[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zc[k]));
				}
				total_sse[0] = _mm256_cvtpd_ps(_mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total));
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dp, dc);
				total = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total_sse[1] = _mm256_cvtpd_ps(_mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total));
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dc, dp);
				total = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zc[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zp[k]);
					Zc[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zp[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zc[k]));
				}
				total_sse[2] = _mm256_cvtpd_ps(_mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total));
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHR++, *buffHL++);
				delta_inner = _mm256_sub_pd(dp, dc);
				total = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[order - 1]), delta_inner, Zp[order - 1]));
				for (int k = order - 2; k >= 0; k--)
				{
					total = _mm256_add_pd(total, Zc[k]);
					Zp[k] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), Zc[k], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), delta_inner, Zp[k]));
				}
				total_sse[3] = _mm256_cvtpd_ps(_mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total));

				_MM_TRANSPOSE4_PS(total_sse[0], total_sse[1], total_sse[2], total_sse[3]);
				_mm_storepatch_ps(dstPtr, total_sse, dwidth);
				dstPtr += 4;
			}
		}
	}

	template<int order>
	void SpatialFilterSlidingDCT5_AVX_64F::verticalFilteringInnerXYLoadStore32FK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
#if 0
		const int simdUnrollSize = 4;//4

		const int width = imgSize.width;
		const int height = imgSize.height;
		const int dstStep = dst.cols / simdUnrollSize;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>(0);
		__m128* dstPtr = (__m128*)dst.ptr<float>(top, xstart);

		double shift0[order + 1];
		double shift1[order + 1];
		for (int i = 1; i <= order; i++)
		{
			shift0[i] = shift[2 * (i - 1) + 0];
			shift1[i] = shift[2 * (i - 1) + 1];
		}
		__m256d t[order + 1];

		__m256d totalA, totalB;
		__m256d pA, pB;
		__m128* pAP, * pAM, * pBP, * pBM, * pCP, * pCM;
		__m256d dvA, dvB, dvC, deltaA, deltaB;
		__m256d* ws = buffVFilter;

		int x, y, r, i;
		for (x = xstart; x < xend; x += simdUnrollSize)
		{
			pA = _mm256_cvtps_pd(_mm_load_ps(&srcPtr[top * width + x]));
			pB = _mm256_cvtps_pd(_mm_load_ps(&srcPtr[top * width + x + width]));

			*ws++ = pA;
			for (i = 1; i <= order; ++i)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(GCn[i - 1]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(GCn[i - 1]));
			}
		}

		for (r = 1; r <= radius; ++r)
		{
			pAM = (__m128*) & srcPtr[UREF(top + 0 - r) + xstart];
			pAP = (__m128*) & srcPtr[width * (top + 0 + r) + xstart];
			pBM = (__m128*) & srcPtr[UREF(top + 1 - r) + xstart];
			pBP = (__m128*) & srcPtr[width * (top + 1 + r) + xstart];

			ws = buffVFilter;

			for (i = 1; i <= order; ++i)
			{
				t[i] = _mm256_set1_pd(GCn[order * r + i - 1]);
			}

			for (x = 0; x < simdWidth; ++x)
			{
				pA = _mm256_add_pd(_mm256_cvtps_pd(*pAM++), _mm256_cvtps_pd(*pAP++));
				pB = _mm256_add_pd(_mm256_cvtps_pd(*pBM++), _mm256_cvtps_pd(*pBP++));

				*ws++ = _mm256_add_pd(*ws, pA);
				for (i = 1; i <= order; ++i)
				{
					//*ws++ = _mm256_fmadd_pd(pA, _mm256_set1_pd(table[order*r + i - 1]), *ws);
					//*ws++ = _mm256_fmadd_pd(pB, _mm256_set1_pd(table[order*r + i - 1]), *ws);
					*ws++ = _mm256_fmadd_pd(pA, t[i], *ws);
					*ws++ = _mm256_fmadd_pd(pB, t[i], *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m128*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m128*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m128*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m128*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m128* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvB = _mm256_sub_pd(_mm256_cvtps_pd(*pBP++), _mm256_cvtps_pd(*pBM++));
				dvC = _mm256_sub_pd(_mm256_cvtps_pd(*pCP++), _mm256_cvtps_pd(*pCM++));
				deltaB = _mm256_sub_pd(dvC, dvB);

				totalA = *ws;
				totalB = _mm256_add_pd(totalA, dvB);
				*ws++ = _mm256_add_pd(totalB, dvC);

				for (i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_mm256_set1_pd(shift1[i]), deltaB, _mm256_fmsub_pd(_mm256_set1_pd(shift0[i]), *(ws + 1), *(ws)));

					ws += 2;
				}
				*dstPtr2 = _mm256_cvtpd_ps(_mm256_mul_pd(_MM256_SET_VECD(mG0), totalA));
				*(dstPtr2 + dstStep) = _mm256_cvtpd_ps(_mm256_mul_pd(_MM256_SET_VECD(mG0), totalB));

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m128*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pAP = (__m128*) & srcPtr[DREF(top + y + radius + 0) + xstart];
			pBM = (__m128*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m128*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCM = (__m128*) & srcPtr[UREF(top + y - radius + 1) + xstart];
			pCP = (__m128*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m128* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				dvA = _mm256_sub_pd(_mm256_cvtps_pd(*pAP++), _mm256_cvtps_pd(*pAM++));
				dvB = _mm256_sub_pd(_mm256_cvtps_pd(*pBP++), _mm256_cvtps_pd(*pBM++));
				dvC = _mm256_sub_pd(_mm256_cvtps_pd(*pCP++), _mm256_cvtps_pd(*pCM++));
				deltaA = _mm256_sub_pd(dvB, dvA);
				deltaB = _mm256_sub_pd(dvC, dvB);

				totalA = *ws;
				totalB = _mm256_add_pd(totalA, dvB);
				*ws++ = _mm256_add_pd(totalB, dvC);

				for (i = 1; i <= order; ++i)
				{
					*(ws + 1) = _mm256_fmadd_pd(_mm256_set1_pd(shift1[i]), deltaA, _mm256_fmsub_pd(_mm256_set1_pd(shift0[i]), *ws, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_mm256_set1_pd(shift1[i]), deltaB, _mm256_fmsub_pd(_mm256_set1_pd(shift0[i]), *(ws + 1), *ws));
					ws += 2;
				}
				*dstPtr2 = _mm256_cvtpd_ps(_mm256_mul_pd(_MM256_SET_VECD(mG0), totalA));
				*(dstPtr2 + dstStep) = _mm256_cvtpd_ps(_mm256_mul_pd(_MM256_SET_VECD(mG0), totalB));

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
#endif
	}

	void SpatialFilterSlidingDCT5_AVX_64F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);

		dst.create(imgSize, (dest_depth < 0) ? src.depth() : dest_depth);

		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			cout << "not implemented (GaussianFilterSlidingDCT5_AVX_64F::filtering)" << endl;
			/*
			if (isHBuff32F)
			{
				switch (gf_order)
				{
				case 1:
					horizontalFilteringInnerXLoadStore32FK<1>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<1>(inter, dst);
					break;
				case 2:
					horizontalFilteringInnerXLoadStore32FK<2>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<2>(inter, dst);
					break;
				case 3:
					horizontalFilteringInnerXLoadStore32FK<3>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<3>(inter, dst);
					break;
				case 4:
					horizontalFilteringInnerXLoadStore32FK<4>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<4>(inter, dst);
					break;
				case 5:
					horizontalFilteringInnerXLoadStore32FK<5>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<5>(inter, dst);
					break;
				case 6:
					horizontalFilteringInnerXLoadStore32FK<6>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<6>(inter, dst);
					break;
				case 7:
					horizontalFilteringInnerXLoadStore32FK<7>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<7>(inter, dst);
					break;
				case 8:
					horizontalFilteringInnerXLoadStore32FK<8>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<8>(inter, dst);
					break;
				case 9:
					horizontalFilteringInnerXLoadStore32FK<9>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<9>(inter, dst);
					break;
				case 10:
					horizontalFilteringInnerXLoadStore32FK<10>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<10>(inter, dst);
					break;
				case 11:
					horizontalFilteringInnerXLoadStore32FK<11>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<11>(inter, dst);
					break;
				case 12:
					horizontalFilteringInnerXLoadStore32FK<12>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<12>(inter, dst);
					break;
				default:
					cout << "do not support this order" << endl;
					break;
				}
			}
			else*/
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT5_64F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXStore32FK<1>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<1>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXStore32FK<2>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<2>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXStore32FK<3>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<3>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXStore32FK<4>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<4>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXStore32FK<5>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<5>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXStore32FK<6>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<6>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXStore32FK<7>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<7>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXStore32FK<8>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<8>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXStore32FK<9>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<9>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXStore32FK<10>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<10>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXStore32FK<11>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<11>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXStore32FK<12>(src, inter, borderType);
					verticalFilteringInnerXYLoadStore32FK<12>(inter, dst, borderType);
					break;
#endif
				default:
					//horizontalFilteringInnerXStore32FKn(src, inter, borderType);
					//verticalFilteringInnerXYLoadStore32FKn(inter, dst, borderType);
					break;
				}
			}
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::CONVOLUTION)
		{
			CV_Assert(src.cols == src.rows);
			horizontalFilteringNaiveConvolution(src, inter, gf_order, borderType);
			Mat temp;
			transpose(inter, temp);
			horizontalFilteringNaiveConvolution(temp, inter, gf_order, borderType);
			transpose(inter, dst);
		}
		else
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT5_64F_ORDER_TEMPLATE
			case 1:
				horizontalFilteringInnerXK<1>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<1, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<1, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<1, uchar>(inter, dst, borderType);
				break;
			case 2:
				horizontalFilteringInnerXK<2>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<2, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<2, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<2, uchar>(inter, dst, borderType);
				break;
			case 3:
				horizontalFilteringInnerXK<3>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<3, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<3, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<3, uchar>(inter, dst, borderType);
				break;
			case 4:
				horizontalFilteringInnerXK<4>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<4, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<4, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<4, uchar>(inter, dst, borderType);
				break;
			case 5:
				horizontalFilteringInnerXK<5>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<5, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<5, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<5, uchar>(inter, dst, borderType);

				break;
			case 6:
				horizontalFilteringInnerXK<6>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<6, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<6, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<6, uchar>(inter, dst, borderType);
				break;
			case 7:
				horizontalFilteringInnerXK<7>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<7, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<7, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<7, uchar>(inter, dst, borderType);
				break;
			case 8:
				horizontalFilteringInnerXK<8>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<8, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<8, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<8, uchar>(inter, dst, borderType);
				break;
			case 9:
				horizontalFilteringInnerXK<9>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<9, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<9, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<9, uchar>(inter, dst, borderType);
				break;
			case 10:
				horizontalFilteringInnerXK<10>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<10, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<10, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<10, uchar>(inter, dst, borderType);
				break;
			case 11:
				horizontalFilteringInnerXK<11>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<11, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<11, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<11, uchar>(inter, dst, borderType);
				break;
			case 12:
				horizontalFilteringInnerXK<12>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<12, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<12, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<12, uchar>(inter, dst, borderType);
				break;
			case 13:
				horizontalFilteringInnerXK<13>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<13, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<13, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<13, uchar>(inter, dst, borderType);
				break;
			case 14:
				horizontalFilteringInnerXK<14>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<14, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<14, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<14, uchar>(inter, dst, borderType);
				break;
			case 15:
				horizontalFilteringInnerXK<15>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<15, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<15, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<15, uchar>(inter, dst, borderType);
				break;
			case 16:
				horizontalFilteringInnerXK<16>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<16, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<16, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<16, uchar>(inter, dst, borderType);
				break;
			case 17:
				horizontalFilteringInnerXK<17>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<17, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<17, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<17, uchar>(inter, dst, borderType);
				break;
			case 18:
				horizontalFilteringInnerXK<18>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<18, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<18, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<18, uchar>(inter, dst, borderType);
				break;
			case 19:
				horizontalFilteringInnerXK<19>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<19, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<19, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<19, uchar>(inter, dst, borderType);
				break;
			case 20:
				horizontalFilteringInnerXK<20>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<20, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<20, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<20, uchar>(inter, dst, borderType);
				break;
			case 21:
				horizontalFilteringInnerXK<21>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<21, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<21, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<21, uchar>(inter, dst, borderType);
				break;
			case 22:
				horizontalFilteringInnerXK<22>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<22, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<22, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<22, uchar>(inter, dst, borderType);
				break;
			case 23:
				horizontalFilteringInnerXK<23>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<23, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<23, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<23, uchar>(inter, dst, borderType);
				break;
			case 24:
				horizontalFilteringInnerXK<24>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<24, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<24, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<24, uchar>(inter, dst, borderType);
				break;
			case 25:
				horizontalFilteringInnerXK<25>(src, inter, borderType);
				if (dst.depth() == CV_64F) verticalFilteringInnerXYK<25, double>(inter, dst, borderType);
				else if (dst.depth() == CV_32F) verticalFilteringInnerXYK<25, float>(inter, dst, borderType);
				else if (dst.depth() == CV_8U) verticalFilteringInnerXYK<25, uchar>(inter, dst, borderType);
				break;
#endif
			default:
				cout << "max order is 25" << endl;
				break;
			}
		}
	}

	void SpatialFilterSlidingDCT5_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		if (this->radius == 0 || this->sigma != sigma || this->gf_order != order || this->imgSize != src.size() || this->fn_hfilter == nullptr)
		{
			this->sigma = sigma;
			this->gf_order = order;
			if (!isUseFixRadius) computeRadius(0, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
			else computeRadius(radius, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);

			imgSize = src.size();
			allocBuffer();
		}

		body(src, dst, borderType);
	}

#pragma endregion
}
