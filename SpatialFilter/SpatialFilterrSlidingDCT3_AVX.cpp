#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
#pragma region SlidingDCT3_AVX_32F

	void SpatialFilterSlidingDCT3_AVX_32F::allocBuffer()
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
		buffVFilter = (__m256*)_mm_malloc(int((2 * gf_order + 2) * internalWidth / simdUnrollSize) * sizeof(__m256), AVX_ALIGNMENT);
	}

	int SpatialFilterSlidingDCT3_AVX_32F::getRadius(const double sigma, const int order)
	{
		cv::AutoBuffer<double> spect(order + 1);
		if (radius == 0)
			return argminR_BruteForce_DCT(sigma, order, 3, spect, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
		else return radius;
	}

	void SpatialFilterSlidingDCT3_AVX_32F::computeRadius(const int rad, const bool isOptimize)
	{
		if (gf_order == 0)
		{
			radius = rad;
			return;
		}

		cv::AutoBuffer<double> Gk(gf_order + 1);
		if (rad == 0)
		{
			radius = argminR_BruteForce_DCT(sigma, gf_order, 3, Gk, isOptimize);
		}
		else
		{
			radius = rad;
		}

		if (isOptimize) optimizeSpectrum(sigma, gf_order, radius, 3, Gk, 0);
		else computeSpectrumGaussianClosedForm(sigma, gf_order, radius, 3, Gk);

		const double omega = CV_2PI / (2.0 * radius + 2.0);

		_mm_free(GCn);
		const int GCnSize = (gf_order + 1) * (radius + 1);//for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		GCn = (float*)_mm_malloc(GCnSize * sizeof(float), AVX_ALIGN);
		AutoBuffer<double> GCn64F(GCnSize);

		double totalInv = 0.0;
		generateCosKernel(GCn64F, totalInv, 3, Gk, radius, gf_order);
		for (int i = 0; i < (gf_order + 1) * (radius + 1); i++)
		{
			GCn[i] = float(GCn64F[i] * totalInv);
		}

#ifdef PLOT_DCT_KERNEL
		plotDCTKernel("DCT-3", false, GCn, radius, gf_order, 0.0, sigma);
#endif
		_mm_free(shift);
		shift = (float*)_mm_malloc(2 * (gf_order + 1) * sizeof(float), AVX_ALIGN);
		for (int k = 0; k <= gf_order; k++)
		{
			const double C1 = cos((k + 0.5) * omega * 1);
			const double CR = cos((k + 0.5) * omega * radius);
#ifdef COEFFICIENTS_SMALLEST_FIRST
			shift[2 * (gf_order - k) + 0] = float(C1 * 2.0);
			shift[2 * (gf_order - k) + 1] = float(CR * Gk[k] * totalInv);
#else
			shift[2 * k + 0] = float(C1 * 2.0);
			shift[2 * k + 1] = float(CR * Gk[k] * totalInv);
#endif
		}
	}

	SpatialFilterSlidingDCT3_AVX_32F::SpatialFilterSlidingDCT3_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		//cout << "init sliding DCT3 GF AVX 32F" << endl;
		this->gf_order = order;
		this->sigma = sigma;
		this->dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;
		computeRadius(radius, true);

		this->imgSize = imgSize;
		allocBuffer();
	}

	SpatialFilterSlidingDCT3_AVX_32F::SpatialFilterSlidingDCT3_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule, const SpatialKernel skernel)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT3_AVX;
		this->schedule = schedule;
		this->depth = CV_32F;
		this->dest_depth = dest_depth;
		this->dct_coeff_method = method;
	}

	SpatialFilterSlidingDCT3_AVX_32F::~SpatialFilterSlidingDCT3_AVX_32F()
	{
		_mm_free(GCn);
		_mm_free(shift);
		_mm_free(buffVFilter);
		_mm_free(fn_hfilter);
	}


	void SpatialFilterSlidingDCT3_AVX_32F::interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad)
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
	void SpatialFilterSlidingDCT3_AVX_32F::horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int xstart = left;//left
		const int xend = get_simd_ceil(src.cols - (left + right), simdUnrollSize) + xstart;

		__m256 total[8];
		AutoBuffer<__m256> Z(order + 1);
		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		const int pad = get_simd_ceil(src.rows, 8) - src.rows;

		for (int y = 0; y < src.rows; y += simdUnrollSize)
		{
			interleaveVerticalPixel(src, y, borderType, pad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			for (int x = xstart; x < xend; x += simdUnrollSize)
			{
				for (int j = 0; j < 8; j++)
				{
					// 1) initilization of Z (1 <= n <= radius) adding small->large
					for (int k = 0; k <= order; ++k)Z[k] = _mm256_setzero_ps();

					for (int n = radius; n >= 1; --n)
					{
						const __m256 sumA = _mm256_add_ps(fn_hfilter[(x + j - n)], fn_hfilter[(x + j + n)]);
						float* Cn_ = GCn + (order + 1) * n;
						for (int k = 0; k <= order; ++k)
						{
							__m256 Cnk = _mm256_set1_ps(Cn_[k]);
							Z[k] = _mm256_fmadd_ps(Cnk, sumA, Z[k]);
						}
					}

					// 1) initilization of Z (n=0) adding small->large
					for (int k = 0; k <= order; ++k)
					{
						__m256 Cnk = _mm256_set1_ps(GCn[k]);
						Z[k] = _mm256_fmadd_ps(Cnk, fn_hfilter[x + j], Z[k]);
					}

					total[j] = _mm256_setzero_ps();
					for (int k = 0; k <= order; k++)
					{
						total[j] = _mm256_add_ps(total[j], Z[k]);
					}
				}
				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dst.cols);
				dstPtr += 8;
			}
		}
	}

	//32F hfilter O(K) (Default)
	template<int order>
	void SpatialFilterSlidingDCT3_AVX_32F::horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVEC C1_2[order + 1];//2*C_1
		SETVEC CR_g[order + 1];//G*C_R
		for (int i = 0; i < order + 1; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 total[8];
		__m256 Zp[order + 1];
		__m256 Zc[order + 1];
		__m256 delta_inner;//f(x-R-1)-f(x+R+1)

		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 8)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			for (int k = 0; k <= order; ++k)
			{
				__m256 GCnk = _mm256_set1_ps(GCn[k]);
				Zp[k] = _mm256_mul_ps(GCnk, fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_ps(GCnk, fn_hfilter[xstart + 1]);
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
				float* Cn_ = GCn + (order + 1) * n;
				for (int k = 0; k <= order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_fmadd_ps(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_ps(Cnk, sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				total[0] = _mm256_add_ps(Zp[0], Zp[1]);
				for (int i = 2; i <= order; ++i)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius - 1)]);
				total[4] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius - 1)]);
				total[5] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));

				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius - 1)]);
				total[6] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius - 1)]);
				total[7] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			__m256* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius - 1];
			for (int x = 0; x < mainloop_simdsize; x++)
			{
				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[0] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[4] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[5] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[6] = Zp[0];
				Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[7] = Zc[0];
				Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Zp[i]));
				}

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}
		}
	}

	//32F vfilter O(K) Y-XLoop (Default)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT3_AVX_32F::verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

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

		const float* srcPtr = src.ptr<float>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		SETVEC C1_2[order + 1];
		SETVEC CR_g[order + 1];
		for (int i = 0; i <= order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA, deltaB;

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (top + 1) + x]);

			for (int i = 0; i <= order; ++i)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
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

				for (int i = 0; i <= order; ++i)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[(order + 1) * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[(order + 1) * r + i]), *ws);
				}
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[swidth * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				pBP += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1);

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;

				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				store_auto<destT>(dstPtr2, totalA);
				store_auto<destT>(dstPtr2 + dwidth, totalB);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				pBP += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1);

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;

				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
				_mm256_storescalar_auto(dstPtr2 + dwidth, totalB, rem);

				dstPtr2 += 8;
			}

			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				deltaB = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storeu_auto(dstPtr2, totalA);
				_mm256_storeu_auto(dstPtr2 + dwidth, totalB);

				dstPtr2 += 8;
			}

			if (rem != 0)
			{
				deltaB = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
				_mm256_storescalar_auto(dstPtr2 + dwidth, totalB, rem);

				dstPtr2 += 8;
			}
			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;

			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			float* pAP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				deltaB = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storeu_auto(dstPtr2, totalA);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				deltaB = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pBP));
				pAP += 8;
				pBP += 8;
				pAM += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);

				dstPtr2 += 8;
			}
		}
	}

	//32F debug plot
	__m256* SpatialFilterSlidingDCT3_AVX_32F::computeZConvVFilter(const float* srcPtr, const int y, const int x, const int width, const int height, const int order, float* GCn, const int radius, const float G0, const int borderType)
	{
		bool is64FAcc = true;

		__m256* ret = new __m256[order + 2];

		if (is64FAcc)
		{
			__m256d* ret20 = new __m256d[order + 2];
			__m256d* ret21 = new __m256d[order + 2];
			// 1) initilization of Z0and Z1(n = 0)
			{
				const __m256d pA0 = _mm256_cvtps_pd(_mm_loadu_ps(&srcPtr[(y)*width + x + 0]));
				const __m256d pA1 = _mm256_cvtps_pd(_mm_loadu_ps(&srcPtr[(y)*width + x + 4]));
				//no DC
				for (int i = 0; i <= order; i++)
				{
					ret20[i] = _mm256_mul_pd(pA0, _mm256_set1_pd(GCn[i]));
					ret21[i] = _mm256_mul_pd(pA1, _mm256_set1_pd(GCn[i]));
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

					//	no DC
					for (int i = 0; i <= order; i++)
					{
						ret20[i] = _mm256_fmadd_pd(pA0, _mm256_set1_pd(GCn[(order + 1) * r + i]), ret20[i]);
						ret21[i] = _mm256_fmadd_pd(pA1, _mm256_set1_pd(GCn[(order + 1) * r + i]), ret21[i]);
					}
				}
			}
			// no DC

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
				for (int i = 0; i <= order; i++)
				{
					ret[i] = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				}
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int r = 1; r <= radius; ++r)
			{
				float* pAM = const_cast<float*>(&srcPtr[ref_tborder(y - r, width, borderType) + x]);
				float* pAP = const_cast<float*>(&srcPtr[width * (y + r) + x]);
				{
					const __m256 pA = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pAP));
					//	ret[0] = _mm256_add_ps(ret[0], pA);
					for (int i = 0; i <= order; i++)
					{
						ret[i] = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[(order + 1) * r + i]), ret[i]);
					}
				}
			}
			//	ret[0] = _mm256_mul_ps(ret[0], _mm256_set1_ps(G0));
		}
		return ret;
	}
	//32F debug plot
	void SpatialFilterSlidingDCT3_AVX_32F::verticalFilteringInnerXYKn_32F_Debug(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		static int showOrder = 0;
		Mat show = Mat::zeros(dst.size(), CV_32F);
#ifdef PLOT_Zk
		createTrackbar("showorder", "", &showOrder, order);
		setTrackbarMax("showorder", "", order);
		string wname = "DCT-3 V plofile plot";
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
		const int height = imgSize.height;
		const int dstStep = dst.cols;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>();
		float* dstPtr = dst.ptr<float>(top, xstart);
		float* showPtr = show.ptr<float>(top, xstart);

		AutoBuffer <SETVEC> C1_2(order + 1);
		AutoBuffer <SETVEC> CR(order + 1);
		for (int i = 0; i <= order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA, deltaB;

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += simdUnrollSize)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[(top + 0) * width + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[(top + 1) * width + x]);

			for (int i = 0; i <= order; ++i)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
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

				for (int i = 0; i <= order; ++i)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[(order + 1) * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[(order + 1) * r + i]), *ws);
				}
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, width, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[width * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			float* dstPtr2 = dstPtr;
			float* showPtr2 = showPtr;
			for (int x = 0; x < simdWidth; ++x)
			{
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pBM));
				pCP += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1);
				if (showOrder == 0) { _mm256_storeu_auto(showPtr2, *ws); _mm256_storeu_auto(showPtr2 + dstStep, *(ws + 1)); }
				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[0]), deltaA, *ws));
				ws += 2;

				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					if (showOrder == i) { _mm256_storeu_auto(showPtr2, *ws); _mm256_storeu_auto(showPtr2 + dstStep, *(ws + 1)); }
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[i]), deltaA, *ws));
					ws += 2;
				}
				store_auto<float>(dstPtr2, totalA);
				store_auto<float>(dstPtr2 + dstStep, totalB);

				dstPtr2 += 8;
				showPtr2 += 8;
			}
			dstPtr += 2 * dstStep;
			showPtr += 2 * dstStep;
		}

		// 3) main loop
		for (int y = 2; y < height - (top + bottom); y += 2)
		{
			float* pAM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius - 1, width, borderType) + xstart]);
			float* pBM = const_cast<float*>(&srcPtr[ref_tborder(top + y - radius + 0, width, borderType) + xstart]);
			float* pBP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 1, width, height, borderType) + xstart]);
			float* pCP = const_cast<float*>(&srcPtr[ref_bborder(top + y + radius + 2, width, height, borderType) + xstart]);

			ws = buffVFilter;
			float* dstPtr2 = dstPtr;
			float* showPtr2 = showPtr;
			for (int x = 0; x < simdWidth; ++x)
			{
				deltaB = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
				deltaA = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				pBP += 8;
				pCP += 8;
				pAM += 8;
				pBM += 8;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR[0]), deltaB, *(ws + 1)));
				if (showOrder == 0) { _mm256_storeu_auto(showPtr2, *ws); _mm256_storeu_auto(showPtr2 + dstStep, *(ws + 1)); }
#ifdef PLOT_Zk
				__m256* conv = nullptr;
				//if (x == line / 8) conv = computeZConvVFilter(srcPtr, top + y, xstart + 8 * x, width, height, order, GCn, radius, invsum, borderType);
				if (x == line / 8) conv = computeZConvVFilter(srcPtr, top + y, xstart + 8 * x, width, height, order, GCn, radius, 1.f, borderType);
				if (x == line / 8 && isPlot[0])
				{
					if (isDiff)
					{
						__m256 sub = _mm256_sub_ps(*ws, conv[0]); //low ws->k=order , high ws ->k=0
						const int idx = line % 8;
						pt.push_back(y, ((float*)&sub)[idx], order);
					}
					else
					{
						;
					}
				}
#endif
				* ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
#ifdef PLOT_Zk
					if (x == line / 8 && isPlot[i])
					{
						if (isDiff)
						{
							__m256 sub = _mm256_sub_ps(*ws, conv[i]);
							const int idx = line % 8;
							pt.push_back(y, ((float*)&sub)[idx], (order - i));
						}
						else
						{
							;
						}
					}
#endif
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					if (showOrder == i) { _mm256_storeu_auto(showPtr2, *ws); _mm256_storeu_auto(showPtr2 + dstStep, *(ws + 1)); }
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR[i]), deltaA, *ws));
					ws += 2;
				}
#ifdef PLOT_Zk
				if (x == line / 8)
				{
					__m256 plot_total = totalA;
					if (isDiff)
					{
						if (isPlot[totalIndex])
						{
							__m256 sub = _mm256_sub_ps(plot_total, conv[totalIndex]);
							const int idx = line % 8;
							pt.push_back(y, ((float*)&sub)[idx], totalIndex);
						}
					}
					else
					{
						;
					}

					delete[] conv;
				}
#endif
				store_auto<float>(dstPtr2, totalA);
				store_auto<float>(dstPtr2 + dstStep, totalB);

				dstPtr2 += 8;
				showPtr2 += 8;
			}
			dstPtr += 2 * dstStep;
			showPtr += 2 * dstStep;
		}

#ifdef PLOT_Zk
		pt.setYRange(-pow(10.0, -scale * 0.1), pow(10.0, -scale * 0.1));
		//pt.setKey(cp::Plot::FLOATING);
		pt.plot(wname, false);
		cp::imshowNormalize("order", show);
		//cp::imshowScale("order", 5*show);
#endif
	}

	//32F vfilter O(K) X-YLoop (tend to cause cache slashing)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT3_AVX_32F::verticalFilteringInnerXYK_XYLoop(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
		const int dwidth = dst.cols;//dst.cols

		const int xstart = left;
		const int xend = get_simd_ceil(dwidth - left - right, simdUnrollSize) + xstart;
		//const int simdWidth = (xend - xstart) / simdUnrollSize;

		const int yst = get_loop_end(top + 2, radius + 1, 2);
		const int yed = get_simd_floor(imgSize.height - radius - 1 - yst, 2) + yst;
		const int yend = get_simd_ceil(imgSize.height - (top + bottom), 2) + top - 2 * ((imgSize.height) % 2);
		const bool isylast = (imgSize.height % 2 == 1);
		//print_debug4(height, radius, top, bottom);
		//print_debug4(yst, yed, yend, height- 2 - radius);

		SETVEC C1_2[order + 1];//2*C_1
		SETVEC CR_g[order + 1];//G*C_R
		for (int i = 0; i < order + 1; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 total;
		__m256 Zp[order + 1];
		__m256 Zc[order + 1];
		__m256 delta_inner;//f(x-R-1)-f(x+R+1)

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
				float* Cn_ = GCn + (order + 1) * radius;
				for (int k = 0; k <= order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_mul_ps(Cnk, sumA);
					Zc[k] = _mm256_mul_ps(Cnk, sumB);
				}
			}
			// 1) initilization of Z0 and Z1 (1<=n<radius)
			for (int n = radius - 1; n > 0; --n)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + n)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + n)));
				float* Cn_ = GCn + (order + 1) * n;
				for (int k = 0; k <= order; ++k)
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
				for (int k = 0; k <= order; ++k)
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
					total = _mm256_add_ps(Zp[0], Zp[1]);
					for (int k = 2; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius - 1, swidth, borderType)));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;
				}

				// 3) main loop
				//top part
				for (int y = top + 2; y < yst; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 1 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y - radius - 1, swidth, borderType)));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 2 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y - radius - 0, swidth, borderType)));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;
				}

				//mid part
				float* s0 = (float*)(srcPtr + swidth * (yst - 1 - radius));
				float* s1 = (float*)(srcPtr + swidth * (yst + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (yst - 0 - radius));
				float* s3 = (float*)(srcPtr + swidth * (yst + 2 + radius));
				for (int y = yst; y < yed; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}

				//bottom
				for (int y = yed; y < yend; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_bborder(y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (y - radius - 1)));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_bborder(y + 2 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (y - radius - 0)));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;
				}
				if (isylast)
				{
					const int y = yed - 2;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_bborder(y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (y - radius - 1)));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, total);
				}
			}
			else
			{
				// 2) initial output computing for y=0
				float* s0 = (float*)(srcPtr + swidth * (top - 1 - radius));
				float* s1 = (float*)(srcPtr + swidth * (top + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (top - 0 - radius));
				float* s3 = (float*)(srcPtr + swidth * (top + 2 + radius));
				{
					total = _mm256_add_ps(Zp[0], Zp[1]);
					for (int k = 2; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;
				}

				// 3) main loop
				s0 += 2 * swidth;
				s1 += 2 * swidth;
				s2 += 2 * swidth;
				s3 += 2 * swidth;
				for (int y = top + 2; y < yend; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					store_auto<destT>(dstPtr, total);
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}
			}
		}

		//last loop
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
				float* Cn_ = GCn + (order + 1) * radius;
				for (int k = 0; k <= order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Zp[k] = _mm256_mul_ps(Cnk, sumA);
					Zc[k] = _mm256_mul_ps(Cnk, sumB);
				}
			}
			// 1) initilization of Z0 and Z1 (1<=n<radius)
			for (int n = radius - 1; n > 0; --n)
			{
				const __m256 sumA = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 0 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 0 + n)));
				const __m256 sumB = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - n, swidth, borderType)), _mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + n)));
				float* Cn_ = GCn + (order + 1) * n;
				for (int k = 0; k <= order; ++k)
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
				for (int k = 0; k <= order; ++k)
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
					total = _mm256_add_ps(Zp[0], Zp[1]);
					for (int k = 2; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + swidth * (ystart + 1 + radius + 1)), _mm256_loadu_ps(srcPtr + ref_tborder(ystart + 1 - radius - 1, swidth, borderType)));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;
				}

				// 3) main loop
				//top part
				for (int y = top + 2; y < yst; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 1 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y - radius - 1, swidth, borderType)));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + swidth * (y + 2 + radius)), _mm256_loadu_ps(srcPtr + ref_tborder(y - radius - 0, swidth, borderType)));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;
				}

				//mid part
				float* s0 = (float*)(srcPtr + swidth * (yst - 1 - radius));
				float* s1 = (float*)(srcPtr + swidth * (yst + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (yst - 0 - radius));
				float* s3 = (float*)(srcPtr + swidth * (yst + 2 + radius));
				for (int y = yst; y < yed; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}

				//bottom
				for (int y = yed; y < yend; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_bborder(y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (y - radius - 1)));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_bborder(y + 2 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (y - radius - 0)));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;
				}
				if (isylast)
				{
					const int y = yed - 2;
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(srcPtr + ref_bborder(y + 1 + radius, swidth, imgSize.height, borderType)), _mm256_loadu_ps(srcPtr + swidth * (y - radius - 1)));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
				}
			}
			else
			{
				// 2) initial output computing for y=0
				float* s0 = (float*)(srcPtr + swidth * (top - 1 - radius));
				float* s1 = (float*)(srcPtr + swidth * (top + 1 + radius));
				float* s2 = (float*)(srcPtr + swidth * (top - 0 - radius));
				float* s3 = (float*)(srcPtr + swidth * (top + 2 + radius));
				{
					total = _mm256_add_ps(Zp[0], Zp[1]);
					for (int k = 2; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;
				}

				// 3) main loop
				s0 += 2 * swidth;
				s1 += 2 * swidth;
				s2 += 2 * swidth;
				s3 += 2 * swidth;
				for (int y = top + 2; y < yend; y += 2)
				{
					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s0));
					total = Zp[0];
					Zc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zc[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zp[k]);
						Zc[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zp[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zc[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					delta_inner = _mm256_add_ps(_mm256_loadu_ps(s3), _mm256_loadu_ps(s2));
					total = Zc[0];
					Zp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Zp[0]));
					for (int k = 1; k <= order; ++k)
					{
						total = _mm256_add_ps(total, Zc[k]);
						Zp[k] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[k]), Zc[k], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[k]), delta_inner, Zp[k]));
					}
					_mm256_storescalar_auto(dstPtr, total, xpad);
					dstPtr += dwidth;

					s0 += 2 * swidth;
					s1 += 2 * swidth;
					s2 += 2 * swidth;
					s3 += 2 * swidth;
				}
			}
		}
	}

	//16F
	template<int order>
	void SpatialFilterSlidingDCT3_AVX_32F::horizontalFilteringInnerXKdest16F(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;

		const int dwidth = dst.cols;//not always same as width for cache thrashing
		const int dheight = dst.rows;

		const int xstart = left;//left
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		//print_debug2(left, xend);

		SETVEC C1_2[order + 1];
		SETVEC CR_g[order + 1];
		for (int i = 0; i < order + 1; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 total[8];
		__m256 Fp[order + 1];
		__m256 Fc[order + 1];
		__m256 delta_inner;//f(x-R-1)-f(x+R+1)

		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = 0; y < imgSize.height; y += 8)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : dheight - imgSize.height;
			interleaveVerticalPixel(src, y, borderType, vpad);

			short* dstPtr = dst.ptr<short>(y, xstart);

			// 1) initilization of F0 and F1 (n=0)
			for (int k = 0; k <= order; ++k)
			{
				__m256 Cnk = _mm256_set1_ps(GCn[k]);
				Fp[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 0]);
				Fc[k] = _mm256_mul_ps(Cnk, fn_hfilter[xstart + 1]);
			}

			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
				float* Cn_ = GCn + (order + 1) * n;
				for (int k = 0; k <= order; ++k)
				{
					__m256 Cnk = _mm256_set1_ps(Cn_[k]);
					Fp[k] = _mm256_fmadd_ps(Cnk, sumA, Fp[k]);
					Fc[k] = _mm256_fmadd_ps(Cnk, sumB, Fc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				total[0] = _mm256_add_ps(Fp[0], Fp[1]);
				for (int i = 2; i <= order; ++i)
				{
					total[0] = _mm256_add_ps(total[0], Fp[i]);
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				total[1] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[1] = _mm256_add_ps(total[1], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				total[2] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[2] = _mm256_add_ps(total[2], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				total[3] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[3] = _mm256_add_ps(total[3], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius - 1)]);
				total[4] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[4] = _mm256_add_ps(total[4], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius - 1)]);
				total[5] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[5] = _mm256_add_ps(total[5], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));

				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius - 1)]);
				total[6] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[6] = _mm256_add_ps(total[6], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius - 1)]);
				total[7] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[7] = _mm256_add_ps(total[7], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ph(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			const int SIMDSIZE = (xend - xstart) / simdUnrollSize - 1;
			__m256* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius - 1];

			for (int x = 0; x < SIMDSIZE; x++)
			{
				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[0] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[0] = _mm256_add_ps(total[0], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[1] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[1] = _mm256_add_ps(total[1], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[2] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[2] = _mm256_add_ps(total[2], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[3] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[3] = _mm256_add_ps(total[3], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[4] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[4] = _mm256_add_ps(total[4], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[5] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[5] = _mm256_add_ps(total[5], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[6] = Fp[0];
				Fc[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fp[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[6] = _mm256_add_ps(total[6], Fp[i]);
					Fc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fp[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fc[i]));
				}

				delta_inner = _mm256_add_ps(*buffHR++, *buffHL++);
				total[7] = Fc[0];
				Fp[0] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), Fc[0], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), delta_inner, Fp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[7] = _mm256_add_ps(total[7], Fc[i]);
					Fp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Fc[i], _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), delta_inner, Fp[i]));
				}

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ph(dstPtr, total, dwidth);
				dstPtr += 8;
			}
		}
	}

	template<int order, typename destT>
	void SpatialFilterSlidingDCT3_AVX_32F::verticalFilteringInnerXYKsrc16F(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;
		const int dwidth = dst.cols;
		const int rem = (8 - (swidth - dwidth)) % 8;

		const int xstart = left;
		const int xend = get_simd_ceil(imgSize.width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;
		const int simdend = (rem == 0) ? simdWidth : simdWidth - 1;

		const short* srcPtr = src.ptr<short>();
		destT* dstPtr = dst.ptr<destT>(top, xstart);

		SETVEC C1_2[order + 1];
		SETVEC CR_g[order + 1];
		for (int i = 0; i <= order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
		}

		__m256 totalA, totalB;
		__m256 deltaA, deltaB;

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_cvtph_ps(*(__m128i*) & srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_cvtph_ps(*(__m128i*) & srcPtr[swidth * (top + 1) + x]);

			for (int i = 0; i <= order; ++i)
			{
				*ws++ = _mm256_mul_ps(pA, _mm256_set1_ps(GCn[i]));
				*ws++ = _mm256_mul_ps(pB, _mm256_set1_ps(GCn[i]));
			}
		}

		// 1) initilization of Z0 and F1 (1<=n<=radius)
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

				for (int i = 0; i <= order; ++i)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[(order + 1) * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[(order + 1) * r + i]), *ws);
				}
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart];
			__m128i* pCP = (__m128i*) & srcPtr[swidth * (top + y + radius + 2) + xstart];

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			for (int x = 0; x < simdend; ++x)
			{
				deltaA = _mm256_add_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pBM++));

				totalA = *ws;
				totalB = *(ws + 1);

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;

				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				store_auto<destT>(dstPtr2, totalA);
				store_auto<destT>(dstPtr2 + dwidth, totalB);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				deltaA = _mm256_add_ps(_mm256_cvtph_ps(*pCP++), _mm256_cvtph_ps(*pBM++));

				totalA = *ws;
				totalB = *(ws + 1);

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;

				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
				_mm256_storescalar_auto(dstPtr2 + dwidth, totalB, rem);

				dstPtr2 += 8;
			}
			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		bool isEven = ((imgSize.height - (top + bottom)) % 2 == 0);
		const int hpad = (isEven) ? 0 : -1;
		const int hend = imgSize.height - (top + bottom) + hpad;

		for (int y = 2; y < hend; y += 2)
		{
			__m128i* pAM = (__m128i*) & srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart];
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart];
			__m128i* pBP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart];
			__m128i* pCP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart];

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			for (int x = 0; x < simdend; ++x)
			{
				deltaB = _mm256_add_ps(_mm256_cvtph_ps(*pAM++), _mm256_cvtph_ps(*pBP++));
				deltaA = _mm256_add_ps(_mm256_cvtph_ps(*pBM++), _mm256_cvtph_ps(*pCP++));

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				store_auto<destT>(dstPtr2, totalA);
				store_auto<destT>(dstPtr2 + dwidth, totalB);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				deltaB = _mm256_add_ps(_mm256_cvtph_ps(*pAM++), _mm256_cvtph_ps(*pBP++));
				deltaA = _mm256_add_ps(_mm256_cvtph_ps(*pBM++), _mm256_cvtph_ps(*pCP++));

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
				_mm256_storescalar_auto(dstPtr2 + dwidth, totalB, rem);

				dstPtr2 += 8;
			}
			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend - 2;
			__m128i* pAM = (__m128i*) & srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart];
			__m128i* pBM = (__m128i*) & srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart];
			__m128i* pBP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart];
			__m128i* pCP = (__m128i*) & srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart];

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				deltaB = _mm256_add_ps(_mm256_cvtph_ps(*pAM++), _mm256_cvtph_ps(*pBP++));
				deltaA = _mm256_add_ps(_mm256_cvtph_ps(*pBM++), _mm256_cvtph_ps(*pCP++));

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}

				store_auto<destT>(dstPtr2, totalA);

				dstPtr2 += 8;
			}

			if (rem != 0)
			{
				deltaB = _mm256_add_ps(_mm256_cvtph_ps(*pAM++), _mm256_cvtph_ps(*pBP++));
				deltaA = _mm256_add_ps(_mm256_cvtph_ps(*pBM++), _mm256_cvtph_ps(*pCP++));

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[0]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[0]), deltaA, *ws));
				ws += 2;
				for (int i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_ps(totalA, *ws);
					*(ws + 1) = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *ws, _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaB, *(ws + 1)));
					totalB = _mm256_add_ps(totalB, *(ws + 1));
					*ws = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), *(ws + 1), _mm256_fmsub_ps(_MM256_SET_VEC(CR_g[i]), deltaA, *ws));
					ws += 2;
				}

				_mm256_storescalar_auto(dstPtr2, totalA, rem);

				dstPtr2 += 8;
			}
		}
	}


	void SpatialFilterSlidingDCT3_AVX_32F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		//cout<<"filtering sliding DCT3 GF AVX 32F" << endl;
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

		dst.create(imgSize, src.depth());

		bool isFloat = (dest_depth == CV_32F || (src.depth() == CV_32F && dest_depth < 0)) ? true : false;

		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT3_32F_ORDER_TEMPLATE
			case 1:
				horizontalFilteringInnerXKdest16F<1>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<1, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<1, uchar>(inter, dst, borderType);
				break;
			case 2:
				horizontalFilteringInnerXKdest16F<2>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<2, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<2, uchar>(inter, dst, borderType);
				break;
			case 3:
				horizontalFilteringInnerXKdest16F<3>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<3, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<3, uchar>(inter, dst, borderType);
				break;
			case 4:
				horizontalFilteringInnerXKdest16F<4>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<4, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<4, uchar>(inter, dst, borderType);
				break;
			case 5:
				horizontalFilteringInnerXKdest16F<5>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<5, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<5, uchar>(inter, dst, borderType);
				break;
			case 6:
				horizontalFilteringInnerXKdest16F<6>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<6, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<6, uchar>(inter, dst, borderType);
				break;
			case 7:
				horizontalFilteringInnerXKdest16F<7>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<7, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<7, uchar>(inter, dst, borderType);
				break;
			case 8:
				horizontalFilteringInnerXKdest16F<8>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<8, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<8, uchar>(inter, dst, borderType);
				break;
			case 9:
				horizontalFilteringInnerXKdest16F<9>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<9, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<9, uchar>(inter, dst, borderType);
				break;
			case 10:
				horizontalFilteringInnerXKdest16F<10>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<10, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<10, uchar>(inter, dst, borderType);
				break;
			case 11:
				horizontalFilteringInnerXKdest16F<11>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<11, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<11, uchar>(inter, dst, borderType);
				break;
			case 12:
				horizontalFilteringInnerXKdest16F<12>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<12, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<12, uchar>(inter, dst, borderType);
				break;
			case 13:
				horizontalFilteringInnerXKdest16F<13>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<13, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<13, uchar>(inter, dst, borderType);
				break;
			case 14:
				horizontalFilteringInnerXKdest16F<14>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<14, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<14, uchar>(inter, dst, borderType);
				break;
			case 15:
				horizontalFilteringInnerXKdest16F<15>(src, inter, borderType);
				if (src.depth() == CV_32F)verticalFilteringInnerXYKsrc16F<15, float>(inter, dst, borderType);
				else verticalFilteringInnerXYKsrc16F<15, uchar>(inter, dst, borderType);
				break;
#endif
			default:
				std::cout << "do not support this order (GaussianFilterSlidingDCT3_AVX_32F::SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)" << std::endl;
				//horizontalFilteringInnerXKn(src, inter, borderType);
				//verticalFilteringInnerXYKn(inter, dst, borderType);
				break;
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
		else if (schedule == SLIDING_DCT_SCHEDULE::DEBUG)
		{
			CV_Assert(isFloat);//dest type must be float
			horizontalFilteringNaiveConvolution(src, inter, gf_order, borderType);
			verticalFilteringInnerXYKn_32F_Debug(inter, dst, gf_order, borderType);
			//verticalFilteringInnerXYK<6, float>(inter, dst, borderType);
			//verticalFilteringInnerXYKn_32F_Debug(src, dst, gf_order);
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::V_XY_LOOP)
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT3_32F_ORDER_TEMPLATE
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
				std::cout << "do not support this order (GaussianFilterSlidingDCT3_AVX_32F::SLIDING_DCT_SCHEDULE::V_XY_LOOP)" << std::endl;
				break;
			}
		}
		else if (schedule == SLIDING_DCT_SCHEDULE::DEFAULT)
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT3_32F_ORDER_TEMPLATE
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
				std::cout << "do not support this order (GaussianFilterSlidingDCT3_AVX_32F)" << std::endl;
				break;
			}
		}
	}

	void SpatialFilterSlidingDCT3_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
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

#pragma region SlidingDCT3_AVX_64F

	void SpatialFilterSlidingDCT3_AVX_64F::allocBuffer()
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
		buffVFilter = (__m256d*)_mm_malloc(int((2 * gf_order + 2) * internalWidth / simdUnrollSize) * sizeof(__m256d), AVX_ALIGNMENT);
	}

	int SpatialFilterSlidingDCT3_AVX_64F::getRadius(const double sigma, const int order)
	{
		cv::AutoBuffer<double> Gk(order + 1);
		if (radius == 0)
			return argminR_BruteForce_DCT(sigma, order, 3, Gk, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
		else return radius;
	}

	void SpatialFilterSlidingDCT3_AVX_64F::computeRadius(const int rad, const bool isOptimize)
	{
		if (gf_order == 0)
		{
			radius = rad;
			return;
		}

		cv::AutoBuffer<double> Gk(gf_order + 1);

		if (rad == 0)
		{
			radius = argminR_BruteForce_DCT(sigma, gf_order, 3, Gk, isOptimize);
		}
		else
		{
			radius = rad;
		}

		if (isOptimize)optimizeSpectrum(sigma, gf_order, radius, 3, Gk, 0);
		else computeSpectrumGaussianClosedForm(sigma, gf_order, radius, 3, Gk);

		const double omega = CV_2PI / (2.0 * radius + 2.0);

		_mm_free(GCn);
		const int GCnSize = (gf_order + 1) * (radius + 1);//for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		GCn = (double*)_mm_malloc(GCnSize * sizeof(double), AVX_ALIGN);

		double totalInv = 0.0;
		generateCosKernel(GCn, totalInv, 3, Gk, radius, gf_order);
		for (int i = 0; i < GCnSize; i++)
		{
			GCn[i] *= totalInv;
		}

		_mm_free(shift);
		shift = (double*)_mm_malloc((2 * (gf_order + 1)) * sizeof(double), AVX_ALIGN);
		for (int k = 0; k <= gf_order; k++)
		{
			const double C1 = cos((k + 0.5) * omega * 1);
			const double CR = cos((k + 0.5) * omega * radius);
#ifdef COEFFICIENTS_SMALLEST_FIRST
			shift[2 * (gf_order - k) + 0] = C1 * 2.0;
			shift[2 * (gf_order - k) + 1] = CR * Gk[k] * totalInv;
#else
			shift[2 * k + 0] = cos((k + 0.5) * phase * 1) * 2.0;
			shift[2 * k + 1] = cos((k + 0.5) * phase * radius) * spect[k] * invsum;
#endif
		}
	}

	SpatialFilterSlidingDCT3_AVX_64F::SpatialFilterSlidingDCT3_AVX_64F(cv::Size imgSize, double sigma, int order, const SLIDING_DCT_SCHEDULE schedule)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT3_64_AVX;
		this->gf_order = order;
		this->sigma = sigma;
		this->schedule = schedule;
		this->dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;
		computeRadius(radius, true);

		this->imgSize = imgSize;
		allocBuffer();
	}

	SpatialFilterSlidingDCT3_AVX_64F::SpatialFilterSlidingDCT3_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule, const SpatialKernel skernel)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT3_64_AVX;
		this->schedule = schedule;
		this->depth = CV_64F;
		this->dest_depth = dest_depth;
		this->dct_coeff_method = method;
	}

	SpatialFilterSlidingDCT3_AVX_64F::~SpatialFilterSlidingDCT3_AVX_64F()
	{
		_mm_free(GCn);
		_mm_free(shift);
		_mm_free(buffVFilter);
		_mm_free(fn_hfilter);
		_mm_free(buffHFilter32F);
	}


	void SpatialFilterSlidingDCT3_AVX_64F::interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad)
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
	void SpatialFilterSlidingDCT3_AVX_64F::horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 4;//4

		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;//left
		const int xend = get_simd_ceil(width - (left + right), simdUnrollSize) + xstart;

		__m256d total[4];
		AutoBuffer<__m256d> Z(order + 1);

		for (int y = 0; y < height; y += simdUnrollSize)
		{
			interleaveVerticalPixel(src, y, borderType);

			double* dstPtr = dst.ptr<double>(y, xstart);

			for (int x = xstart; x < xend; x += simdUnrollSize)
			{
				for (int j = 0; j < simdUnrollSize; j++)
				{
					// 1) initilization of Z (1 <= n <= radius) adding small->large
					for (int k = 0; k <= order; ++k)Z[k] = _mm256_setzero_pd();

					for (int n = radius; n >= 1; --n)
					{
						const __m256d sumA = _mm256_add_pd(fn_hfilter[(x + j - n)], fn_hfilter[(x + j + n)]);
						double* Cn_ = GCn + (order + 1) * n;
						for (int k = 0; k <= order; ++k)
						{
							__m256d Cnk = _mm256_set1_pd(Cn_[k]);
							Z[k] = _mm256_fmadd_pd(Cnk, sumA, Z[k]);
						}
					}

					// 1) initilization of Z (n=0) adding small->large
					for (int k = 0; k <= order; ++k)
					{
						__m256d Cnk = _mm256_set1_pd(GCn[k]);
						Z[k] = _mm256_fmadd_pd(Cnk, fn_hfilter[x + j], Z[k]);
					}

					total[j] = _mm256_setzero_pd();
					for (int k = 0; k <= order; k++)
					{
						total[j] = _mm256_add_pd(total[j], Z[k]);
					}
				}
				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, width);
				dstPtr += simdUnrollSize;
			}
		}
	}

#if 0
	template<int order>
	void GaussianFilterSlidingDCT3_AVX_64F::horizontalFilteringInnerXStore32FK(const cv::Mat& src, cv::Mat& dst)
	{
		__m128 patch_sse_32f[4];

		const int simdUnrollSize = 4;

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

		__m256d mshift0[order + 1];
		__m256d mshift1[order + 1];
		for (int i = 0; i <= order; ++i)
		{
			mshift0[i] = _mm256_set1_pd(shift[i * 2 + 0]);
			mshift1[i] = _mm256_set1_pd(shift[i * 2 + 1]);
		}
		__m256d total = _mm256_setzero_pd();

		__m256d a[order + 1];
		__m256d b[order + 1];

		__m256d sumA, sumB;
		__m256d delta;

		int x, y, r, i;
		for (y = 0; y < height; y += simdUnrollSize)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			for (i = 0; i <= order; ++i)
			{
				a[i] = _mm256_mul_pd(_mm256_set1_pd(Cn[i]), fn_hfilter[xstart + 0]);
				b[i] = _mm256_mul_pd(_mm256_set1_pd(Cn[i]), fn_hfilter[xstart + 1]);
			}

			if (left < radius)
			{
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_pd(fn_hfilter[LREF(xstart + 0 - r)], fn_hfilter[(xstart + 0 + r)]);
					sumB = _mm256_add_pd(fn_hfilter[LREF(xstart + 1 - r)], fn_hfilter[(xstart + 1 + r)]);

					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumA, a[i]);
						b[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumB, b[i]);
					}
				}

				//for (x = xstart; x < xstart+simdstep; x += simdstep)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[LREF(xstart + 1 - radius - 1)]);
					total = a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[LREF(xstart + 2 - radius - 1)]);
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[LREF(xstart + 3 - radius - 1)]);
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}

				for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[RREF(x + 0 + radius + 1)], fn_hfilter[LREF(x + 0 - radius - 1)]);
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[RREF(x + 1 + radius + 1)], fn_hfilter[LREF(x + 1 - radius - 1)]);
					a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[RREF(x + 2 + radius + 1)], fn_hfilter[LREF(x + 2 - radius - 1)]);
					b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(fn_hfilter[RREF(x + 3 + radius + 1)], fn_hfilter[LREF(x + 3 - radius - 1)]);
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}
			}
			else
			{
				__m256d* buffHR = &fn_hfilter[xstart + 1];
				__m256d* buffHL = &fn_hfilter[xstart - 1];
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_pd(*buffHL, *buffHR);
					sumB = _mm256_add_pd(*(buffHL + 1), *(buffHR + 1));
					buffHR++;
					buffHL--;
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumA, a[i]);
						b[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumB, b[i]);
					}
				}

				buffHR = &fn_hfilter[(xstart + 1 + radius + 1)];
				buffHL = &fn_hfilter[(xstart + 1 - radius - 1)];
				//for (x = xstart; x < xstart+simdstep; x += simdstep)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					total = a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}

				const int simdWidth = (xend - (xstart + simdUnrollSize)) / simdUnrollSize;
				//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				for (x = 0; x < simdWidth; ++x)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(*buffHR++, *buffHL++);
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}
			}
			/*
			for (x = xstart + simdstep; x < xst; x += simdstep)
			{
				total = a[0];
				for (i = 1; i <= order; i++)
				{
					total = _mm256_add_pd(total, a[i]);
				}
				patch[0] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 0 + radius + 1)], buffHFilter[LREF(x + 0 - radius - 1)]);

				b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
				total = b[0];
				for (i = 1; i <= order; i++)
				{
					b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
					total = _mm256_add_pd(total, b[i]);
				}
				patch[1] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 1 + radius + 1)], buffHFilter[LREF(x + 1 - radius - 1)]);

				a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
				total = a[0];
				for (i = 1; i <= order; i++)
				{
					a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					total = _mm256_add_pd(total, a[i]);
				}
				patch[2] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 2 + radius + 1)], buffHFilter[LREF(x + 2 - radius - 1)]);

				b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
				total = b[0];
				for (i = 1; i <= order; i++)
				{
					b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
					total = _mm256_add_pd(total, b[i]);
				}
				patch[3] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 3 + radius + 1)], buffHFilter[LREF(x + 3 - radius - 1)]);

				for (i = 0; i <= order; i++)
				{
					a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
				}

				_mm256_transpose4_pd(patch, patch);
				_mm256_storepatch_pd(dstPtr, patch, dst.cols);
				dstPtr += simdstep;
			}

			for (x = xst; x < xed; x += simdstep)
			{
				total = a[0];
				for (i = 1; i <= order; i++)
				{
					total = _mm256_add_pd(total, a[i]);
				}
				patch[0] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius - 1)]);

				b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
				total = b[0];
				for (i = 1; i <= order; i++)
				{
					b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
					total = _mm256_add_pd(total, b[i]);
				}
				patch[1] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius - 1)]);

				a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
				total = a[0];
				for (i = 1; i <= order; i++)
				{
					a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					total = _mm256_add_pd(total, a[i]);
				}
				patch[2] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius - 1)]);

				b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
				total = b[0];
				for (i = 1; i <= order; i++)
				{
					b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
					total = _mm256_add_pd(total, b[i]);
				}
				patch[3] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius - 1)]);

				for (i = 0; i <= order; i++)
				{
					a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
				}

				_mm256_transpose4_pd(patch, patch);
				_mm256_storepatch_pd(dstPtr, patch, dst.cols);
				dstPtr += simdstep;
			}

			for (x = xed; x < xend; x += simdstep)
			{
				total = a[0];
				for (i = 1; i <= order; i++)
				{
					total = _mm256_add_pd(total, a[i]);
				}
				patch[0] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[RREF(x + 0 + radius + 1)], buffHFilter[(x + 0 - radius - 1)]);

				b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
				total = b[0];
				for (i = 1; i <= order; i++)
				{
					b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
					total = _mm256_add_pd(total, b[i]);
				}
				patch[1] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[RREF(x + 1 + radius + 1)], buffHFilter[(x + 1 - radius - 1)]);

				a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
				total = a[0];
				for (i = 1; i <= order; i++)
				{
					a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					total = _mm256_add_pd(total, a[i]);
				}
				patch[2] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[RREF(x + 2 + radius + 1)], buffHFilter[(x + 2 - radius - 1)]);

				b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
				total = b[0];
				for (i = 1; i <= order; i++)
				{
					b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
					total = _mm256_add_pd(total, b[i]);
				}
				patch[3] = _mm256_mul_pd(_mm256_set1_pd(norm), total);

				delta = _mm256_add_pd(buffHFilter[RREF(x + 3 + radius + 1)], buffHFilter[(x + 3 - radius - 1)]);

				for (i = 0; i <= order; i++)
				{
					a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
				}

				_mm256_transpose4_pd(patch, patch);
				_mm256_storepatch_pd(dstPtr, patch, dst.cols);
				dstPtr += simdstep;
			}
			*/
		}
	}

	template<int order>
	void GaussianFilterSlidingDCT3_AVX_64F::horizontalFilteringInnerXLoadStore32FK(const cv::Mat& src, cv::Mat& dst)
	{
		__m128 patch_sse_32f[4];

		const int simdUnrollSize = 4;

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

		__m256d mshift0[order + 1];
		__m256d mshift1[order + 1];
		for (int i = 0; i <= order; ++i)
		{
			mshift0[i] = _mm256_set1_pd(shift[i * 2 + 0]);
			mshift1[i] = _mm256_set1_pd(shift[i * 2 + 1]);
		}
		__m256d total = _mm256_setzero_pd();

		__m256d a[order + 1];
		__m256d b[order + 1];

		__m256d sumA, sumB;
		__m256d delta;

		int x, y, r, i;
		for (y = 0; y < height; y += simdUnrollSize)
		{
			copyPatchHorizontalbody(const_cast<Mat&>(src), y);

			float* dstPtr = dst.ptr<float>(y, xstart);

			//r=0
			for (i = 0; i <= order; ++i)
			{
				a[i] = _mm256_mul_pd(_mm256_set1_pd(Cn[i]), _mm256_cvtps_pd(buffHFilter32F[xstart + 0]));
				b[i] = _mm256_mul_pd(_mm256_set1_pd(Cn[i]), _mm256_cvtps_pd(buffHFilter32F[xstart + 1]));
			}

			if (left < radius)
			{
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[LREF(xstart + 0 - r)]), _mm256_cvtps_pd(buffHFilter32F[(xstart + 0 + r)]));
					sumB = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[LREF(xstart + 1 - r)]), _mm256_cvtps_pd(buffHFilter32F[(xstart + 1 + r)]));

					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumA, a[i]);
						b[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumB, b[i]);
					}
				}

				//for (x = xstart; x < xstart+simdstep; x += simdstep)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[(xstart + 1 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(xstart + 1 - radius - 1)]));
					total = a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[(xstart + 2 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(xstart + 2 - radius - 1)]));
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[(xstart + 3 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(xstart + 3 - radius - 1)]));
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}

				for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[RREF(x + 0 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(x + 0 - radius - 1)]));
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[RREF(x + 1 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(x + 1 - radius - 1)]));
					a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[RREF(x + 2 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(x + 2 - radius - 1)]));
					b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(buffHFilter32F[RREF(x + 3 + radius + 1)]), _mm256_cvtps_pd(buffHFilter32F[LREF(x + 3 - radius - 1)]));
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}
			}
			else
			{
				__m128* buffHR = &buffHFilter32F[xstart + 1];
				__m128* buffHL = &buffHFilter32F[xstart - 1];
				for (r = 1; r <= radius; ++r)
				{
					sumA = _mm256_add_pd(_mm256_cvtps_pd(*buffHL), _mm256_cvtps_pd(*buffHR));
					sumB = _mm256_add_pd(_mm256_cvtps_pd(*(buffHL + 1)), _mm256_cvtps_pd(*(buffHR + 1)));
					buffHR++;
					buffHL--;
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumA, a[i]);
						b[i] = _mm256_fmadd_pd(_mm256_set1_pd(Cn[(order + 1) * r + i]), sumB, b[i]);
					}
				}

				buffHR = &buffHFilter32F[(xstart + 1 + radius + 1)];
				buffHL = &buffHFilter32F[(xstart + 1 - radius - 1)];
				//for (x = xstart; x < xstart+simdstep; x += simdstep)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					total = a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}

				const int simdWidth = (xend - (xstart + simdUnrollSize)) / simdUnrollSize;
				//for (x = xstart + simdUnrollSize; x < xend; x += simdUnrollSize)
				for (x = 0; x < simdWidth; ++x)
				{
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[0] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					total = b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[1] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					a[0] = _mm256_fmadd_pd(mshift0[0], b[0], _mm256_fmsub_pd(mshift1[0], delta, a[0]));
					total = a[0];
					for (i = 1; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
						total = _mm256_add_pd(total, a[i]);
					}
					patch_sse_32f[2] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					b[0] = _mm256_fmadd_pd(mshift0[0], a[0], _mm256_fmsub_pd(mshift1[0], delta, b[0]));
					total = b[0];
					for (i = 1; i <= order; ++i)
					{
						b[i] = _mm256_fmadd_pd(mshift0[i], a[i], _mm256_fmsub_pd(mshift1[i], delta, b[i]));
						total = _mm256_add_pd(total, b[i]);
					}
					patch_sse_32f[3] = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), total));

					delta = _mm256_add_pd(_mm256_cvtps_pd(*buffHR++), _mm256_cvtps_pd(*buffHL++));
					for (i = 0; i <= order; ++i)
					{
						a[i] = _mm256_fmadd_pd(mshift0[i], b[i], _mm256_fmsub_pd(mshift1[i], delta, a[i]));
					}

					_MM_TRANSPOSE4_PS(patch_sse_32f[0], patch_sse_32f[1], patch_sse_32f[2], patch_sse_32f[3]);
					_mm_storepatch_ps(dstPtr, patch_sse_32f, dst.cols);
					dstPtr += simdUnrollSize;
				}
			}
		}
	}

	template <int order>
	void GaussianFilterSlidingDCT3_AVX_64F::verticalFilteringInnerXYStore32FK(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdstep = 4;

		const int width = src.cols;
		const int height = imgSize.height;
		const int dstStep = dst.cols / simdstep;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdstep) + xstart;

		const double* srcPtr = src.ptr<double>(0);
		__m128* dstPtr = (__m128*)dst.ptr<float>(top, xstart);

		int x, y, r, i;

		__m256d mshift0[order + 1];
		__m256d mshift1[order + 1];
		for (i = 0; i <= order; i++)
		{
			mshift0[i] = _mm256_set1_pd(shift[i * 2 + 0]);
			mshift1[i] = _mm256_set1_pd(shift[i * 2 + 1]);
		}
		__m256d t[order + 1];

		__m256d totalA, totalB;
		__m256d pA, pB;
		__m256d* pAP, * pAM, * pBP, * pBM, * pCP;
		__m256d deltaB, deltaC;
		__m256d* ws;

		ws = buffVFilter;
		for (x = xstart; x < xend; x += simdstep)
		{
			pA = _mm256_load_pd(&srcPtr[top * width + x]);
			pB = _mm256_load_pd(&srcPtr[top * width + x + width]);

			for (i = 0; i <= order; i++)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(Cn[i]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(Cn[i]));
			}
		}

		for (r = 1; r <= radius; ++r)
		{
			pAM = (__m256d*) & srcPtr[UREF(top + 0 - r) + xstart];
			pAP = (__m256d*) & srcPtr[width * (top + 0 + r) + xstart];
			pBM = (__m256d*) & srcPtr[UREF(top + 1 - r) + xstart];
			pBP = (__m256d*) & srcPtr[width * (top + 1 + r) + xstart];
			ws = buffVFilter;

			for (i = 0; i <= order; i++)
			{
				t[i] = _mm256_set1_pd(Cn[(order + 1) * r + i]);
			}

			for (x = xstart; x < xend; x += simdstep)
			{
				pA = _mm256_add_pd(*pAM++, *pAP++);
				pB = _mm256_add_pd(*pBM++, *pBP++);

				for (i = 0; i <= order; i++)
				{
					*ws++ = _mm256_fmadd_pd(pA, t[i], *ws);
					*ws++ = _mm256_fmadd_pd(pB, t[i], *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m256d*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pCP = (__m256d*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m128* dstPtr2 = dstPtr;
			for (int x = xstart; x < xend; x += simdstep)
			{
				//*ws: alpha
				//*(ws+1): beta
				deltaC = _mm256_add_pd(*pCP++, *pBM++);

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_pd(mshift0[0], *(ws + 1), _mm256_fmsub_pd(mshift1[0], deltaC, *ws));
				ws += 2;
				for (i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(mshift0[i], *(ws + 1), _mm256_fmsub_pd(mshift1[i], deltaC, *ws));

					ws += 2;
				}

				*dstPtr2 = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalA));
				*(dstPtr2 + dstStep) = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalB));

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m256d*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pBM = (__m256d*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m256d*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCP = (__m256d*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m128* dstPtr2 = dstPtr;
			for (int x = xstart; x < xend; x += simdstep)
			{
				//*ws: alpha
				//*(ws+1): beta
				deltaB = _mm256_add_pd(*pAM++, *pBP++);
				deltaC = _mm256_add_pd(*pBM++, *pCP++);

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(mshift0[0], *ws, _mm256_fmsub_pd(mshift1[0], deltaB, *(ws + 1)));
				*ws = _mm256_fmadd_pd(mshift0[0], *(ws + 1), _mm256_fmsub_pd(mshift1[0], deltaC, *ws));
				ws += 2;
				for (i = 1; i <= order; ++i)
				{
					*(ws + 1) = _mm256_fmadd_pd(mshift0[i], *ws, _mm256_fmsub_pd(mshift1[i], deltaB, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(mshift0[i], *(ws + 1), _mm256_fmsub_pd(mshift1[i], deltaC, *ws));

					ws += 2;
				}
				*dstPtr2 = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalA));
				*(dstPtr2 + dstStep) = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalB));

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
	}

	template <int order>
	void GaussianFilterSlidingDCT3_AVX_64F::verticalFilteringInnerXYLoadStore32FK(const cv::Mat& src, cv::Mat& dst)
	{
		const int simdUnrollSize = 4;

		const int width = src.cols;
		const int height = imgSize.height;
		const int dstStep = dst.cols / simdUnrollSize;

		const int xstart = left;
		const int xend = get_simd_ceil(width - left - right, simdUnrollSize) + xstart;
		const int simdWidth = (xend - xstart) / simdUnrollSize;

		const float* srcPtr = src.ptr<float>(0);
		__m128* dstPtr = (__m128*)dst.ptr<float>(top, xstart);

		__m256d mshift0[order + 1];
		__m256d mshift1[order + 1];
		for (int i = 0; i <= order; ++i)
		{
			mshift0[i] = _mm256_set1_pd(shift[i * 2 + 0]);
			mshift1[i] = _mm256_set1_pd(shift[i * 2 + 1]);
		}
		__m256d t[order + 1];

		__m256d totalA, totalB;
		__m256d pA, pB;
		__m128* pAP, * pAM, * pBP, * pBM, * pCP;
		__m256d deltaB, deltaC;
		__m256d* ws = buffVFilter;

		int x, y, r, i;
		for (x = xstart; x < xend; x += simdUnrollSize)
		{
			pA = _mm256_cvtps_pd(_mm_loadu_ps(&srcPtr[top * width + x]));
			pB = _mm256_cvtps_pd(_mm_loadu_ps(&srcPtr[top * width + x + width]));

			for (i = 0; i <= order; ++i)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(Cn[i]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(Cn[i]));
			}
		}

		for (r = 1; r <= radius; ++r)
		{
			pAM = (__m128*) & srcPtr[UREF(top + 0 - r) + xstart];
			pAP = (__m128*) & srcPtr[width * (top + 0 + r) + xstart];
			pBM = (__m128*) & srcPtr[UREF(top + 1 - r) + xstart];
			pBP = (__m128*) & srcPtr[width * (top + 1 + r) + xstart];

			ws = buffVFilter;
			for (i = 0; i <= order; ++i)
			{
				t[i] = _mm256_set1_pd(Cn[(order + 1) * r + i]);
			}

			for (x = 0; x < simdWidth; ++x)
			{
				pA = _mm256_add_pd(_mm256_cvtps_pd(*pAM++), _mm256_cvtps_pd(*pAP++));
				pB = _mm256_add_pd(_mm256_cvtps_pd(*pBM++), _mm256_cvtps_pd(*pBP++));

				for (i = 0; i <= order; ++i)
				{
					*ws++ = _mm256_fmadd_pd(pA, t[i], *ws);
					*ws++ = _mm256_fmadd_pd(pB, t[i], *ws);
				}
			}
		}

		for (y = 0; y < 2; y += 2)
		{
			pBM = (__m128*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pCP = (__m128*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m128* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				deltaC = _mm256_add_pd(_mm256_cvtps_pd(*pCP++), _mm256_cvtps_pd(*pBM++));

				totalA = *ws;
				totalB = *(ws + 1);
				*ws = _mm256_fmadd_pd(mshift0[0], *(ws + 1), _mm256_fmsub_pd(mshift1[0], deltaC, *ws));
				ws += 2;
				for (i = 1; i <= order; ++i)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(mshift0[i], *(ws + 1), _mm256_fmsub_pd(mshift1[i], deltaC, *ws));

					ws += 2;
				}
				*dstPtr2 = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalA));
				*(dstPtr2 + dstStep) = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalB));

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}

		for (y = 2; y < height - (top + bottom); y += 2)
		{
			pAM = (__m128*) & srcPtr[UREF(top + y - radius - 1) + xstart];
			pBM = (__m128*) & srcPtr[UREF(top + y - radius + 0) + xstart];
			pBP = (__m128*) & srcPtr[DREF(top + y + radius + 1) + xstart];
			pCP = (__m128*) & srcPtr[DREF(top + y + radius + 2) + xstart];

			ws = buffVFilter;
			__m128* dstPtr2 = dstPtr;
			for (x = 0; x < simdWidth; ++x)
			{
				deltaB = _mm256_add_pd(_mm256_cvtps_pd(*pAM++), _mm256_cvtps_pd(*pBP++));
				deltaC = _mm256_add_pd(_mm256_cvtps_pd(*pBM++), _mm256_cvtps_pd(*pCP++));

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(mshift0[0], *ws, _mm256_fmsub_pd(mshift1[0], deltaB, *(ws + 1)));
				*ws = _mm256_fmadd_pd(mshift0[0], *(ws + 1), _mm256_fmsub_pd(mshift1[0], deltaC, *ws));
				ws += 2;
				for (i = 1; i <= order; ++i)
				{
					*(ws + 1) = _mm256_fmadd_pd(mshift0[i], *ws, _mm256_fmsub_pd(mshift1[i], deltaB, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(mshift0[i], *(ws + 1), _mm256_fmsub_pd(mshift1[i], deltaC, *ws));

					ws += 2;
				}
				*dstPtr2 = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalA));
				*(dstPtr2 + dstStep) = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_set1_pd(G0), totalB));

				++dstPtr2;
			}
			dstPtr += 2 * dstStep;
		}
	}
#endif

	//64F hfilter O(K)
	template<int order>
	void SpatialFilterSlidingDCT3_AVX_64F::horizontalFilteringInnerXK(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 4;

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVECD C1_2[order + 1];
		SETVECD CR_g[order + 1];
		for (int i = 0; i < order + 1; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
		}

		__m256d total[4];
		__m256d Zp[order + 1];
		__m256d Zc[order + 1];
		__m256d delta_inner;//f(x-R-1)-f(x+R+1)

		__m256d* fn_hfilter = &this->fn_hfilter[radius + 1];

		for (int y = ystart; y < yend; y += 4)
		{
			const int vpad = (y + simdUnrollSize < imgSize.height) ? 0 : imgSize.height - y;
			interleaveVerticalPixel(src, y, borderType, vpad);

			double* dstPtr = dst.ptr<double>(y, xstart);

			// 1) initilization of Z0 and Z1 (n=0)
			for (int k = 0; k <= order; ++k)
			{
				__m256d Cnk = _mm256_set1_pd(GCn[k]);
				Zp[k] = _mm256_mul_pd(Cnk, fn_hfilter[xstart + 0]);
				Zc[k] = _mm256_mul_pd(Cnk, fn_hfilter[xstart + 1]);
			}

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int n = 1; n <= radius; ++n)
			{
				const __m256d sumA = _mm256_add_pd(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256d sumB = _mm256_add_pd(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
				double* Cn_ = GCn + (order + 1) * n;
				for (int k = 0; k <= order; ++k)
				{
					__m256d Cnk = _mm256_set1_pd(Cn_[k]);
					Zp[k] = _mm256_fmadd_pd(Cnk, sumA, Zp[k]);
					Zc[k] = _mm256_fmadd_pd(Cnk, sumB, Zc[k]);
				}
			}

			// 2) initial output computing for x=0
			{
				total[0] = _mm256_add_pd(Zp[0], Zp[1]);
				for (int i = 2; i <= order; ++i)
				{
					total[0] = _mm256_add_pd(total[0], Zp[i]);
				}

				__m256d* buffHR = &fn_hfilter[(xstart + 1 + radius + 1)];
				delta_inner = _mm256_add_pd(*buffHR++, fn_hfilter[(xstart + 1 - radius - 1)]);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_pd(*buffHR++, fn_hfilter[(xstart + 2 - radius - 1)]);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zp[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_pd(*buffHR++, fn_hfilter[(xstart + 3 - radius - 1)]);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zp[i]));
				}

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}

			// 3) main loop
			__m256d* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256d* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius - 1];
			for (int x = 0; x < mainloop_simdsize; x++)
			{
				delta_inner = _mm256_add_pd(*buffHR++, *buffHL++);
				total[0] = Zp[0];
				Zc[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zp[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[0] = _mm256_add_pd(total[0], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_pd(*buffHR++, *buffHL++);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zp[i]));
				}

				delta_inner = _mm256_add_pd(*buffHR++, *buffHL++);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zp[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zc[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zc[i]));
				}

				delta_inner = _mm256_add_pd(*buffHR++, *buffHL++);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), delta_inner, Zp[0]));
				for (int i = 1; i <= order; ++i)
				{
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[i]), delta_inner, Zp[i]));
				}

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}
		}
	}

	//64F vfilter O(K)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT3_AVX_64F::verticalFilteringInnerXYK(const cv::Mat& src, cv::Mat& dst, const int borderType)
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

		SETVECD C1_2[order + 1];
		SETVECD CR_g[order + 1];
		for (int i = 0; i <= order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			CR_g[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
		}

		__m256d totalA, totalB;
		__m256d deltaB, deltaC;

		__m256d* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 4)
		{
			const __m256d pA = _mm256_loadu_pd(&srcPtr[swidth * (top + 0) + x]);
			const __m256d pB = _mm256_loadu_pd(&srcPtr[swidth * (top + 1) + x]);

			for (int k = 0; k <= order; ++k)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(GCn[k]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(GCn[k]));
			}
		}

		// 1) initilization of Z0 and Z1 (1<=n<=radius)
		for (int n = 1; n <= radius; ++n)
		{
			double* pAM = const_cast<double*>(&srcPtr[ref_tborder(top + 0 - n, swidth, borderType) + xstart]);
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + 1 - n, swidth, borderType) + xstart]);
			double* pAP = const_cast<double*>(&srcPtr[swidth * (top + 0 + n) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[swidth * (top + 1 + n) + xstart]);

			ws = buffVFilter;

			for (int x = 0; x < simdWidth; ++x)
			{
				const __m256d pA = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pAP));
				const __m256d pB = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pBP));
				pAP += 4;
				pBP += 4;
				pAM += 4;
				pBM += 4;

				for (int k = 0; k <= order; ++k)
				{
					*ws++ = _mm256_fmadd_pd(pA, _mm256_set1_pd(GCn[(order + 1) * n + k]), *ws);
					*ws++ = _mm256_fmadd_pd(pB, _mm256_set1_pd(GCn[(order + 1) * n + k]), *ws);
				}
			}
		}

		// 2) initial output computing for y=0,1
		for (int y = 0; y < 2; y += 2)
		{
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			double* pCP = const_cast<double*>(&srcPtr[swidth * (top + y + radius + 2) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			for (int x = 0; x < simdend; ++x)
			{
				deltaC = _mm256_add_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pBM));
				pCP += 4;
				pBM += 4;

				totalA = *ws;
				totalB = *(ws + 1);

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaC, *ws));
				ws += 2;
				for (int k = 1; k <= order; ++k)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaC, *ws));
					ws += 2;
				}
				store_auto<destT>(dstPtr2, totalA);
				store_auto<destT>(dstPtr2 + dwidth, totalB);

				dstPtr2 += 4;
			}
			if (rem != 0)
			{
				deltaC = _mm256_add_pd(_mm256_loadu_pd(pCP), _mm256_loadu_pd(pBM));
				pCP += 4;
				pBM += 4;

				totalA = *ws;
				totalB = *(ws + 1);

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaC, *ws));
				ws += 2;
				for (int k = 1; k <= order; ++k)
				{
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[k]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[k]), deltaC, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
				_mm256_storescalar_auto(dstPtr2 + dwidth, totalB, rem);
			}
			dstPtr += 2 * dwidth;
		}

		// 3) main loop
		for (int y = 2; y < hend; y += 2)
		{
			double* pAM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			double* pCP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;
			for (int x = 0; x < simdend; ++x)
			{
				deltaB = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				deltaC = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaC, *ws));
				ws += 2;
				for (int k = 1; k <= order; ++k)
				{
					const __m256d ms0 = _MM256_SET_VECD(C1_2[k]);
					const __m256d ms1 = _MM256_SET_VECD(CR_g[k]);

					*(ws + 1) = _mm256_fmadd_pd(ms0, *ws, _mm256_fmsub_pd(ms1, deltaB, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(ms0, *(ws + 1), _mm256_fmsub_pd(ms1, deltaC, *ws));
					ws += 2;
				}
				store_auto<destT>(dstPtr2, totalA);
				store_auto<destT>(dstPtr2 + dwidth, totalB);

				dstPtr2 += 4;
			}
			if (rem != 0)
			{
				deltaB = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				deltaC = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaC, *ws));
				ws += 2;
				for (int k = 1; k <= order; ++k)
				{
					const __m256d ms0 = _MM256_SET_VECD(C1_2[k]);
					const __m256d ms1 = _MM256_SET_VECD(CR_g[k]);

					*(ws + 1) = _mm256_fmadd_pd(ms0, *ws, _mm256_fmsub_pd(ms1, deltaB, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(ms0, *(ws + 1), _mm256_fmsub_pd(ms1, deltaC, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
				_mm256_storescalar_auto(dstPtr2 + dwidth, totalB, rem);
			}

			dstPtr += 2 * dwidth;
		}

		if (!isEven)
		{
			const int y = hend;

			double* pAM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius - 1, swidth, borderType) + xstart]);
			double* pBM = const_cast<double*>(&srcPtr[ref_tborder(top + y - radius + 0, swidth, borderType) + xstart]);
			double* pBP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 1, swidth, imgSize.height, borderType) + xstart]);
			double* pCP = const_cast<double*>(&srcPtr[ref_bborder(top + y + radius + 2, swidth, imgSize.height, borderType) + xstart]);

			ws = buffVFilter;
			destT* dstPtr2 = dstPtr;

			for (int x = 0; x < simdend; ++x)
			{
				deltaB = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				deltaC = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaC, *ws));
				ws += 2;
				for (int k = 1; k <= order; ++k)
				{
					const __m256d ms0 = _MM256_SET_VECD(C1_2[k]);
					const __m256d ms1 = _MM256_SET_VECD(CR_g[k]);

					*(ws + 1) = _mm256_fmadd_pd(ms0, *ws, _mm256_fmsub_pd(ms1, deltaB, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(ms0, *(ws + 1), _mm256_fmsub_pd(ms1, deltaC, *ws));
					ws += 2;
				}
				store_auto<destT>(dstPtr2, totalA);

				dstPtr2 += 4;
			}
			if (rem != 0)
			{
				deltaB = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				deltaC = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				pBP += 4;
				pCP += 4;
				pAM += 4;
				pBM += 4;

				totalA = *ws;
				totalB = *(ws + 1) = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *ws, _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaB, *(ws + 1)));

				*ws = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[0]), *(ws + 1), _mm256_fmsub_pd(_MM256_SET_VECD(CR_g[0]), deltaC, *ws));
				ws += 2;
				for (int k = 1; k <= order; ++k)
				{
					const __m256d ms0 = _MM256_SET_VECD(C1_2[k]);
					const __m256d ms1 = _MM256_SET_VECD(CR_g[k]);

					*(ws + 1) = _mm256_fmadd_pd(ms0, *ws, _mm256_fmsub_pd(ms1, deltaB, *(ws + 1)));
					totalA = _mm256_add_pd(totalA, *ws);
					totalB = _mm256_add_pd(totalB, *(ws + 1));
					*ws = _mm256_fmadd_pd(ms0, *(ws + 1), _mm256_fmsub_pd(ms1, deltaC, *ws));
					ws += 2;
				}
				_mm256_storescalar_auto(dstPtr2, totalA, rem);
			}
		}
	}


	void SpatialFilterSlidingDCT3_AVX_64F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);

		dst.create(imgSize, (dest_depth < 0) ? src.depth() : dest_depth);

		//input->tr_buffer_h->buffer_v->output

		//8U/32F
		//isInterBuff32F=true&&isHBuff32F=true
		//float/uchar->float->float->float

		//isInterBuff32F=true&&isHBuff32F=false
		//float/uchar->double->float->float

		//isInterBuff32F=false&&isHBuff32F=false
		//float/uchar->double->double->float

		//64F
		//double->double->double->double

		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
#if 0
			if (isHBuff32F)
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT3_64F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXLoadStore32FK<1>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<1>(inter, dst, borderType);
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
#endif
				default:
					cout << "do not support this order(GaussianFilterSlidingDCT3_AVX_64F::filtering, isHBuff32F=true). call 64F filtering." << endl;
					//horizontalFilteringKn(src, inter, gf_order);
					//verticalFilteringStore32FKn(inter, dst, gf_order);
					break;
				}
			}
			else
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT3_64F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXStore32FK<1>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<1>(inter, dst);
					break;
				case 2:
					horizontalFilteringInnerXStore32FK<2>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<2>(inter, dst);
					break;
				case 3:
					horizontalFilteringInnerXStore32FK<3>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<3>(inter, dst);
					break;
				case 4:
					horizontalFilteringInnerXStore32FK<4>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<4>(inter, dst);
					break;
				case 5:
					horizontalFilteringInnerXStore32FK<5>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<5>(inter, dst);
					break;
				case 6:
					horizontalFilteringInnerXStore32FK<6>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<6>(inter, dst);
					break;
				case 7:
					horizontalFilteringInnerXStore32FK<7>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<7>(inter, dst);
					break;
				case 8:
					horizontalFilteringInnerXStore32FK<8>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<8>(inter, dst);
					break;
				case 9:
					horizontalFilteringInnerXStore32FK<9>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<9>(inter, dst);
					break;
				case 10:
					horizontalFilteringInnerXStore32FK<10>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<10>(inter, dst);
					break;
				case 11:
					horizontalFilteringInnerXStore32FK<11>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<11>(inter, dst);
					break;
				case 12:
					horizontalFilteringInnerXStore32FK<12>(src, inter);
					verticalFilteringInnerXYLoadStore32FK<12>(inter, dst);
					break;
#endif
				default:
					cout << "do not support this order(GaussianFilterSlidingDCT3_AVX_64F::filtering, isHBuff32F=false). call 64F filtering." << endl;
					//horizontalFilteringKn(src, inter, gf_order);
					//verticalFilteringStore32FKn(inter, dst, gf_order);
					break;
	}
}
#endif 
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
		else if (schedule == SLIDING_DCT_SCHEDULE::DEFAULT)
		{
			switch (gf_order)
			{
#ifdef COMPILE_GF_DCT3_64F_ORDER_TEMPLATE
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
				cout << "max order is 25. SpatialFilterSlidingDCT3_AVX_64F::body, SLIDING_DCT_SCHEDULE::DEFAULT" << endl;
				break;
			}
		}
	}

	void SpatialFilterSlidingDCT3_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
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