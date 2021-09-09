#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
#pragma region SlidingDCT1_AVX_32F

	void SpatialFilterSlidingDCT1_AVX_32F::allocBuffer()
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

	int SpatialFilterSlidingDCT1_AVX_32F::getRadius(const double sigma, const int order)
	{
		cv::AutoBuffer<double> spect(order + 1);
		if (radius == 0)
			return argminR_BruteForce_DCT(sigma, order, 1, spect, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
		else return radius;
	}

	void SpatialFilterSlidingDCT1_AVX_32F::computeRadius(const int rad, const bool isOptimize)
	{
		if (gf_order == 0)
		{
			radius = rad;
			return;
		}

		cv::AutoBuffer<double> Gk(gf_order + 1);
		if (rad == 0)
		{
			radius = argminR_BruteForce_DCT(sigma, gf_order, 1, Gk, isOptimize);
		}
		else
		{
			radius = rad;
		}

		if (isOptimize) optimizeSpectrum(sigma, gf_order, radius, 1, Gk, 0);
		else computeSpectrumGaussianClosedForm(sigma, gf_order, radius, 1, Gk);

		const double omega = CV_2PI / (2.0 * radius);

		_mm_free(GCn);
		const int GCnSize = (gf_order + 0) * (radius + 1);//for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		GCn = (float*)_mm_malloc(GCnSize * sizeof(float), AVX_ALIGN);
		AutoBuffer<double> GCn64F(GCnSize);

		double totalInv = 0.0;
		generateCosKernel(GCn64F, totalInv, 1, Gk, radius, gf_order);
		G0 = (float)(1.0 / totalInv);
		for (int i = 0; i < gf_order * (radius + 1); i++)
		{
			GCn[i] = float(GCn64F[i] * totalInv);
		}
		G0 = (float)(Gk[0] * totalInv);
#ifdef PLOT_DCT_KERNEL
		plotDCTKernel("DCT1", false, GCn, radius, gf_order, G0, sigma);
#endif

		_mm_free(shift);
		shift = (float*)_mm_malloc((2 * gf_order) * sizeof(float), AVX_ALIGN);
		Gk_dct1 = (float*)_mm_malloc(gf_order * sizeof(float), AVX_ALIGN);
		for (int k = 1; k <= gf_order; ++k)
		{
			const double C1 = cos(k * omega * 1);
			shift[2 * (k - 1) + 0] = float(C1 * 2.0);
			shift[2 * (k - 1) + 1] = float(C1);
			Gk_dct1[k - 1] = float(Gk[k] * totalInv);
		}
	}

	SpatialFilterSlidingDCT1_AVX_32F::SpatialFilterSlidingDCT1_AVX_32F(cv::Size imgSize, float sigma, int order)
		: SpatialFilterBase(imgSize, CV_32F)
	{
		//cout << "init sliding DCT1 GF AVX 32F" << endl;
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT1_AVX;
		this->gf_order = order;
		this->sigma = sigma;
		this->dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;
		computeRadius(radius, true);

		this->imgSize = imgSize;
		allocBuffer();
	}

	SpatialFilterSlidingDCT1_AVX_32F::SpatialFilterSlidingDCT1_AVX_32F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule, const SpatialKernel skernel)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT1_AVX;
		this->schedule = schedule;
		this->depth = CV_32F;
		this->dest_depth = dest_depth;
		this->dct_coeff_method = method;
	}

	SpatialFilterSlidingDCT1_AVX_32F::~SpatialFilterSlidingDCT1_AVX_32F()
	{
		_mm_free(GCn);
		_mm_free(shift);
		_mm_free(Gk_dct1);
		_mm_free(fn_hfilter);
		_mm_free(buffVFilter);
	}


	void SpatialFilterSlidingDCT1_AVX_32F::interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad)
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
	void SpatialFilterSlidingDCT1_AVX_32F::horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int xstart = left;//left
		const int xend = get_simd_ceil(imgSize.width - (left + right), simdUnrollSize) + xstart;

		__m256 total[8];
		__m256 F0;
		AutoBuffer<__m256> Z(order);
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		__m256* fn_hfilter = &this->fn_hfilter[radius + 1];

		const int pad = get_simd_ceil(src.rows, 8) - src.rows;

		for (int y = 0; y < imgSize.height; y += simdUnrollSize)
		{
			interleaveVerticalPixel(src, y, borderType, pad);

			float* dstPtr = dst.ptr<float>(y, xstart);

			for (int x = xstart; x < xend; x += simdUnrollSize)
			{
				for (int j = 0; j < 8; j++)
				{
					// 1) initilization of Z (1 <= n <= radius)
					for (int k = 0; k <= order; ++k)Z[k] = _mm256_setzero_ps();
					F0 = _mm256_setzero_ps();

					for (int n = radius; n >= 1; --n)
					{
						const __m256 sumA = _mm256_add_ps(fn_hfilter[(x + j - n)], fn_hfilter[(x + j + n)]);
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

	//horizontal filtering template (increment)
	template<int order>
	void SpatialFilterSlidingDCT1_AVX_32F::horizontalFilteringInnerXK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;//SIMDSIZE

		SETVEC C1_2[order];
		SETVEC C1[order];
		SETVEC G[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			C1[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
			G[i] = _MM256_SETLUT_VEC(Gk_dct1[i]);
		}

		__m256 total[8];
		__m256 F0;
		__m256 Zp[order];
		__m256 Zc[order];
		__m256 delta_inner;//t1-Ck_1*t2
		__m256 fdelta;//f_x+R+1 + f_x-R: can be removed, but used for chache efficiency of loading src images
		__m256 t1;//f_x+R+1 + f_x-R-1
		__m256 t2;//f_x+R + f_x-R

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

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
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
				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);
				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius)], fn_hfilter[(xstart + 4 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[4] = Zp[0];
				Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius)], fn_hfilter[(xstart + 5 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[5] = Zc[0];
				Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius)], fn_hfilter[(xstart + 6 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[6] = Zp[0];
				Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius)], fn_hfilter[(xstart + 7 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				total[7] = Zc[0];
				Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			__m256* fx_mRm1 = &fn_hfilter[xstart + simdUnrollSize - radius - 1];
			__m256* fx_pRp1 = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256* fx_mR = &fn_hfilter[xstart + simdUnrollSize - radius];
			__m256* fx_pR = &fn_hfilter[xstart + simdUnrollSize + radius];
			//const bool isUseC2 = true;
			const bool isUseC2 = false;
			if (isUseC2)
			{
				for (int x = 0; x < mainloop_simdsize; x++)
				{
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[0] = Zp[0];
					Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[1] = Zc[0];
					Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[2] = Zp[0];
					Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[3] = Zc[0];
					Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[4] = Zp[0];
					Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[5] = Zc[0];
					Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[6] = Zp[0];
					Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[7] = Zc[0];
					Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

					_mm256_transpose8_ps(total);
					_mm256_storeupatch_ps(dstPtr, total, dwidth);
					dstPtr += 8;
				}
			}
			else
			{
				for (int x = 0; x < mainloop_simdsize; x++)
				{
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[0] = Zp[0];
					//Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					Zc[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						//Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						//Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[1] = Zc[0];
					//Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					Zp[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						//Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						//Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[2] = Zp[0];
					//Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					Zc[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						//Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						//Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[3] = Zc[0];
					//Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					Zp[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						//Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						//Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[4] = Zp[0];
					//Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					Zc[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						//Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						//Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[5] = Zc[0];
					//Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					Zp[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						//Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						//Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[6] = Zp[0];
					//Zc[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					Zc[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zp[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zc[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						//Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						//Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
						Zc[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*fx_pRp1, *fx_mR);
					t1 = _mm256_add_ps(*fx_pRp1++, *fx_mRm1++);
					t2 = _mm256_add_ps(*fx_pR++, *fx_mR++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
					total[7] = Zc[0];
					//Zp[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					Zp[0] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[0])), Zc[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), delta_inner, Zp[0]));
					for (int i = 2; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						//Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = 1; i < order; i += 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						//Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
						Zp[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(2.f), _MM256_SET_VEC(C1[i])), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

					_mm256_transpose8_ps(total);
					_mm256_storeupatch_ps(dstPtr, total, dwidth);
					dstPtr += 8;
				}
			}
		}
	}

	//vertical filtering template(increment)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT1_AVX_32F::verticalFilteringInnerXYK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;//8

		const int swidth = src.cols;//src.cols
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

		SETVEC C1_2[order];
		SETVEC C1[order];
		SETVEC G[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[2 * i + 0]);
			C1[i] = _MM256_SETLUT_VEC(shift[2 * i + 1]);
			G[i] = _MM256_SETLUT_VEC(Gk_dct1[i]);
		}

		__m256 totalA, totalB;
		__m256 deltaA;
		__m256 deltaB;
		__m256 dc, dn;
		__m256 t1, t2, t3, t4;

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (top + 1) + x]);

			for (int i = 0; i < order; ++i)
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

				for (int i = 0; i < order; ++i)
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
				t1 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				t2 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = ws[0];
				totalB = ws[1];
				deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				ws[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[1], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaB, ws[0]));

				for (int i = 2; i < order; i += 2)
				{
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 temp = _mm256_add_ps(*ws, dc);
				_mm256_storeu_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), temp, totalB));
				*ws++ = _mm256_add_ps(temp, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				t1 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				t2 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
				pBP += 8;
				pCP += 8;
				pBM += 8;
				pCM += 8;

				totalA = ws[0];
				totalB = ws[1];
				deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				ws[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[1], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaB, ws[0]));

				for (int i = 2; i < order; i += 2)
				{
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

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
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
				t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
				t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
				pAM += 8;
				pAP += 8;
				pBM += 8;
				pBP += 8;
				pCM += 8;
				pCP += 8;

				deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[1], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					/*
					__m256 c1 = _MM256_SET_VEC(C1[i]);
					__m256 c2 = _mm256_mul_ps(_mm256_set1_ps(2.f),c1);
					deltaA = _mm256_fnmadd_ps(c1, t2, t1);
					deltaB = _mm256_fnmadd_ps(c1, t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_ps(c2, ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_ps(c2, ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					*/
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));

				}
				for (int i = 1; i < order; i += 2)
				{
					/*
					__m256 c1 = _MM256_SET_VEC(C1[i]);
					__m256 c2 = _mm256_mul_ps(_mm256_set1_ps(2.f), c1);
					deltaA = _mm256_fnmadd_ps(c1, t2, t1);
					deltaB = _mm256_fnmadd_ps(c1, t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_ps(c2, ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_ps(c2, ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					*/
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storeu_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
				t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
				t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
				pAM += 8;
				pAP += 8;
				pBM += 8;
				pBP += 8;
				pCM += 8;
				pCP += 8;

				deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[1], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
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
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
				t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
				t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
				pAM += 8;
				pAP += 8;
				pBM += 8;
				pBP += 8;
				pCM += 8;
				pCP += 8;

				deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[1], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

				_mm256_storeu_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
			if (rem != 0)
			{
				dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
				dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
				t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
				t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
				t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
				t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
				pAM += 8;
				pAP += 8;
				pBM += 8;
				pBP += 8;
				pCM += 8;
				pCP += 8;

				deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[0], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[0]), ws[1], _mm256_fmadd_ps(_MM256_SET_VEC(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
					totalA = _mm256_add_ps(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
				__m256 F0 = _mm256_add_ps(*ws, dc);
				*ws++ = _mm256_add_ps(F0, dn);

				dstPtr2 += 8;
			}
			dstPtr += 2 * dwidth;
		}
	}

	//horizontal filtering template(decrement)
	template<int order>
	void SpatialFilterSlidingDCT1_AVX_32F::horizontalFilteringInnerXK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 8;

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;//SIMDSIZE

		SETVEC C1_2[order];
		SETVEC C1[order];
		SETVEC G[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[i * 2 + 0]);
			C1[i] = _MM256_SETLUT_VEC(shift[i * 2 + 1]);
			G[i] = _MM256_SETLUT_VEC(Gk_dct1[i]);
		}

		__m256 total[8];
		__m256 F0;
		__m256 Zp[order];
		__m256 Zc[order];
		__m256 delta_inner;//t1-Ck_1*t2
		__m256 fdelta;//f_x+R+1 + f_x-R
		__m256 t1;//f_x+R+1 + f_x-R-1
		__m256 t2;//f_x+R + f_x-R

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

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int n = 1; n <= radius; ++n)
			{
				const __m256 sumA = _mm256_add_ps(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256 sumB = _mm256_add_ps(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
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
			if constexpr (order % 2 == 0)
			{
				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);
				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius)], fn_hfilter[(xstart + 4 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius)], fn_hfilter[(xstart + 5 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius)], fn_hfilter[(xstart + 6 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius)], fn_hfilter[(xstart + 7 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}
			else if constexpr (order % 2 == 1)
			{
				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);
				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_ps(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 1 + radius)], fn_hfilter[(xstart + 1 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[1] = _mm256_add_ps(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 2 + radius)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[2] = _mm256_add_ps(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 3 + radius)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[3] = _mm256_add_ps(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius + 1)], fn_hfilter[(xstart + 4 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 4 + radius)], fn_hfilter[(xstart + 4 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[4] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[4] = _mm256_add_ps(total[4], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius + 1)], fn_hfilter[(xstart + 5 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 5 + radius)], fn_hfilter[(xstart + 5 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[5] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[5] = _mm256_add_ps(total[5], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius + 1)], fn_hfilter[(xstart + 6 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 6 + radius)], fn_hfilter[(xstart + 6 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[6] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[6] = _mm256_add_ps(total[6], Zp[i]);
					Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
				}
				total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
				F0 = _mm256_add_ps(F0, fdelta);

				fdelta = _mm256_sub_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius)]);
				t1 = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius + 1)], fn_hfilter[(xstart + 7 - radius - 1)]);
				t2 = _mm256_add_ps(fn_hfilter[(xstart + 7 + radius)], fn_hfilter[(xstart + 7 - radius)]);
				delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
				total[7] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
					total[7] = _mm256_add_ps(total[7], Zc[i]);
					Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
				}
				total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

				_mm256_transpose8_ps(total);
				_mm256_storeupatch_ps(dstPtr, total, dwidth);
				dstPtr += 8;
			}

			// 3) main loop
			__m256* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius];
			__m256* buffHRR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius];
			__m256* buffHLL = &fn_hfilter[xstart + simdUnrollSize - radius - 1];
			if constexpr (order % 2 == 0)
			{
				for (int x = 0; x < mainloop_simdsize; x++)
				{
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[0] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[1] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[2] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[3] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[4] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[5] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[6] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[7] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

					_mm256_transpose8_ps(total);
					_mm256_storeupatch_ps(dstPtr, total, dwidth);
					dstPtr += 8;
				}
			}
			else if constexpr (order % 2 == 1)
			{
				for (int x = 0; x < mainloop_simdsize; x++)
				{
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[0] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[0] = _mm256_add_ps(total[0], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[0] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[0]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[1] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[1] = _mm256_add_ps(total[1], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[1] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[1]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[2] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[2] = _mm256_add_ps(total[2], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[2] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[2]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[3] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[3] = _mm256_add_ps(total[3], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[3] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[3]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[4] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[4] = _mm256_add_ps(total[4], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[4] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[4]);
					F0 = _mm256_add_ps(F0, fdelta);
					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[5] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[5] = _mm256_add_ps(total[5], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[5] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[5]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[6] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						Zc[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[6] = _mm256_add_ps(total[6], Zp[i]);
						Zc[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zp[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zc[i]));
					}
					total[6] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[6]);
					F0 = _mm256_add_ps(F0, fdelta);

					fdelta = _mm256_sub_ps(*buffHRR, *buffHL);
					t1 = _mm256_add_ps(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_ps(*buffHR++, *buffHL++);
					delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					total[7] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						Zp[i] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						total[7] = _mm256_add_ps(total[7], Zc[i]);
						Zp[i] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), Zc[i], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), delta_inner, Zp[i]));
					}
					total[7] = _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, total[7]);

					_mm256_transpose8_ps(total);
					_mm256_storeupatch_ps(dstPtr, total, dwidth);
					dstPtr += 8;
				}
			}
		}
	}

	//vertical filtering template(decrement)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT1_AVX_32F::verticalFilteringInnerXYK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType)
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
		SETVEC C1[order];
		SETVEC G[order];
		SETVEC mG0 = _MM256_SETLUT_VEC(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VEC(shift[2 * i + 0]);
			C1[i] = _MM256_SETLUT_VEC(shift[2 * i + 1]);
			G[i] = _MM256_SETLUT_VEC(Gk_dct1[i]);
		}

		__m256 totalA, totalB;
		__m256 deltaA;
		__m256 deltaB;
		__m256 dc, dn;
		__m256 t1, t2, t3, t4;

		__m256* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 8)
		{
			const __m256 pA = _mm256_loadu_ps(&srcPtr[swidth * (top + 0) + x]);
			const __m256 pB = _mm256_loadu_ps(&srcPtr[swidth * (top + 1) + x]);

			for (int i = 0; i < order; ++i)
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

				for (int i = 0; i < order; ++i)
				{
					*ws++ = _mm256_fmadd_ps(pA, _mm256_set1_ps(GCn[order * r + i]), *ws);
					*ws++ = _mm256_fmadd_ps(pB, _mm256_set1_ps(GCn[order * r + i]), *ws);
				}
				*ws++ = _mm256_add_ps(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		if constexpr (order % 2 == 0)
		{
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
					t1 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pBP += 8;
					pCP += 8;
					pBM += 8;
					pCM += 8;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 2; i >= 0; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
					__m256 F0 = _mm256_add_ps(*ws, dc);
					store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
					*ws++ = _mm256_add_ps(F0, dn);

					dstPtr2 += 8;
				}
				if (rem != 0)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pBP += 8;
					pCP += 8;
					pBM += 8;
					pCM += 8;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 2; i >= 0; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
					__m256 F0 = _mm256_add_ps(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
					*ws++ = _mm256_add_ps(F0, dn);
				}
				dstPtr += 2 * dwidth;
			}
		}
		else if constexpr (order % 2 == 1)
		{
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
					t1 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pBP += 8;
					pCP += 8;
					pBM += 8;
					pCM += 8;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 3; i >= 0; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
					__m256 F0 = _mm256_add_ps(*ws, dc);
					store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
					*ws++ = _mm256_add_ps(F0, dn);

					dstPtr2 += 8;
				}
				if (rem != 0)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pBP += 8;
					pCP += 8;
					pBM += 8;
					pCM += 8;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 3; i >= 0; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
					__m256 F0 = _mm256_add_ps(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
					*ws++ = _mm256_add_ps(F0, dn);
				}
				dstPtr += 2 * dwidth;
			}
		}

		// 3) main loop
		if constexpr (order % 2 == 0)
		{
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
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
					__m256 F0 = _mm256_add_ps(*ws, dc);
					store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
					*ws++ = _mm256_add_ps(F0, dn);

					dstPtr2 += 8;
				}
				if (rem != 0)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
					__m256 F0 = _mm256_add_ps(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
					*ws++ = _mm256_add_ps(F0, dn);
				}
				dstPtr += 2 * dwidth;
			}
		}
		else if constexpr (order % 2 == 1)
		{
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
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
					__m256 F0 = _mm256_add_ps(*ws, dc);
					store_auto<destT>(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB));
					*ws++ = _mm256_add_ps(F0, dn);

					dstPtr2 += 8;
				}
				if (rem != 0)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
					__m256 F0 = _mm256_add_ps(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), F0, totalB), rem);
					*ws++ = _mm256_add_ps(F0, dn);
				}
				dstPtr += 2 * dwidth;
			}
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

			if constexpr (order % 2 == 0)
			{
				for (int x = 0; x < simdend; ++x)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
					__m256 F0 = _mm256_add_ps(*ws, dc);
					*ws++ = _mm256_add_ps(F0, dn);

					dstPtr2 += 8;
				}
				if (rem != 0)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
					__m256 F0 = _mm256_add_ps(*ws, dc);
					*ws++ = _mm256_add_ps(F0, dn);
				}
			}
			else if constexpr (order % 2 == 1)
			{
				for (int x = 0; x < simdend; ++x)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					store_auto<destT>(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA));
					__m256 F0 = _mm256_add_ps(*ws, dc);
					*ws++ = _mm256_add_ps(F0, dn);

					dstPtr2 += 8;
				}
				if (rem != 0)
				{
					dc = _mm256_sub_ps(_mm256_loadu_ps(pBP), _mm256_loadu_ps(pBM));
					dn = _mm256_sub_ps(_mm256_loadu_ps(pCP), _mm256_loadu_ps(pCM));
					t1 = _mm256_add_ps(_mm256_loadu_ps(pAM), _mm256_loadu_ps(pBP));
					t2 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pAP));
					t3 = _mm256_add_ps(_mm256_loadu_ps(pBM), _mm256_loadu_ps(pCP));
					t4 = _mm256_add_ps(_mm256_loadu_ps(pCM), _mm256_loadu_ps(pBP));
					pAM += 8;
					pAP += 8;
					pBM += 8;
					pBP += 8;
					pCM += 8;
					pCP += 8;

					deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_ps(_MM256_SET_VEC(C1[i]), t4, t3);
						totalA = _mm256_add_ps(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_ps(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_ps(_MM256_SET_VEC(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_ps(_MM256_SET_VEC(G[i]), deltaB, ws[i * 2]));
					}

					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_ps(_MM256_SET_VEC(mG0), *ws, totalA), rem);
					__m256 F0 = _mm256_add_ps(*ws, dc);
					*ws++ = _mm256_add_ps(F0, dn);
				}
			}
		}
	}


	//filtering
	void SpatialFilterSlidingDCT1_AVX_32F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		//cout<<"filtering sliding DCT1 GF AVX 32F" << endl;
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

		dst.create(imgSize, (dest_depth < 0) ? src.depth() : dest_depth);

		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			std::cout << "do not support CV16 (GaussianFilterSlidingDCT1_AVX_32F)" << std::endl;
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
			const bool isInc = true;//inc
			//const bool isInc = false;//dec
			if (isInc)
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT1_32F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXK_inc<1>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<1, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<1, uchar>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXK_inc<2>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<2, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<2, uchar>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXK_inc<3>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<3, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<3, uchar>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXK_inc<4>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<4, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<4, uchar>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXK_inc<5>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<5, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<5, uchar>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXK_inc<6>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<6, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<6, uchar>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXK_inc<7>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<7, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<7, uchar>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXK_inc<8>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<8, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<8, uchar>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXK_inc<9>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<9, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<9, uchar>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXK_inc<10>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<10, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<10, uchar>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXK_inc<11>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<11, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<11, uchar>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXK_inc<12>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<12, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<12, uchar>(inter, dst, borderType);
					break;
				case 13:
					horizontalFilteringInnerXK_inc<13>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<13, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<13, uchar>(inter, dst, borderType);
					break;
				case 14:
					horizontalFilteringInnerXK_inc<14>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<14, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<14, uchar>(inter, dst, borderType);
					break;
				case 15:
					horizontalFilteringInnerXK_inc<15>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_inc<15, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_inc<15, uchar>(inter, dst, borderType);
					break;
#endif
				default:
					std::cout << "do not support this order (GaussianFilterSlidingDCT1_AVX_32F)" << std::endl;
					break;
				}
			}
			else
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT1_32F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXK_dec<1>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<1, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<1, uchar>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXK_dec<2>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<2, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<2, uchar>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXK_dec<3>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<3, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<3, uchar>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXK_dec<4>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<4, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<4, uchar>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXK_dec<5>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<5, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<5, uchar>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXK_dec<6>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<6, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<6, uchar>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXK_dec<7>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<7, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<7, uchar>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXK_dec<8>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<8, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<8, uchar>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXK_dec<9>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<9, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<9, uchar>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXK_dec<10>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<10, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<10, uchar>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXK_dec<11>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<11, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<11, uchar>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXK_dec<12>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<12, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<12, uchar>(inter, dst, borderType);
					break;
				case 13:
					horizontalFilteringInnerXK_dec<13>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<13, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<13, uchar>(inter, dst, borderType);
					break;
				case 14:
					horizontalFilteringInnerXK_dec<14>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<14, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<14, uchar>(inter, dst, borderType);
					break;
				case 15:
					horizontalFilteringInnerXK_dec<15>(src, inter, borderType);
					if (dst.depth() == CV_32F)verticalFilteringInnerXYK_dec<15, float>(inter, dst, borderType);
					else verticalFilteringInnerXYK_dec<15, uchar>(inter, dst, borderType);
					break;
#endif
				default:
					std::cout << "do not support this order (GaussianFilterSlidingDCT1_AVX_32F)" << std::endl;
					break;
				}
			}
		}
	}

	void SpatialFilterSlidingDCT1_AVX_32F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
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

#pragma region SlidingDCT1_AVX_64F

	void SpatialFilterSlidingDCT1_AVX_64F::allocBuffer()
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

	int SpatialFilterSlidingDCT1_AVX_64F::getRadius(const double sigma, const int order)
	{
		cv::AutoBuffer<double> Gk(order + 1);
		if (radius == 0)
			return argminR_BruteForce_DCT(sigma, order, 1, Gk, dct_coeff_method == DCT_COEFFICIENTS::FULL_SEARCH_OPT);
		else return radius;
	}

	void SpatialFilterSlidingDCT1_AVX_64F::computeRadius(const int rad, const bool isOptimize)
	{
		if (gf_order == 0)
		{
			radius = rad;
			return;
		}

		cv::AutoBuffer<double> Gk(gf_order + 1);
		if (rad == 0)
		{
			radius = argminR_BruteForce_DCT(sigma, gf_order, 1, Gk, isOptimize);
		}
		else
		{
			radius = rad;
		}

		if (isOptimize)optimizeSpectrum(sigma, gf_order, radius, 1, Gk, 0);
		else computeSpectrumGaussianClosedForm(sigma, gf_order, radius, 1, Gk);

		const double omega = CV_2PI / (2.0 * radius);

		_mm_free(GCn);
		const int GCnSize = (gf_order + 0) * (radius + 1);//for dct3 and 7, dct5 has DC; thus, we can reduce the size.
		GCn = (double*)_mm_malloc(GCnSize * sizeof(double), AVX_ALIGN);

		double totalInv = 0;
		generateCosKernel(GCn, totalInv, 1, Gk, radius, gf_order);
		for (int i = 0; i < GCnSize; i++)
		{
			GCn[i] *= totalInv;
		}
		G0 = Gk[0] * totalInv;

		_mm_free(shift);
		shift = (double*)_mm_malloc((2 * gf_order) * sizeof(double), AVX_ALIGN);
		_mm_free(Gk_dct1);
		Gk_dct1 = (double*)_mm_malloc((gf_order) * sizeof(double), AVX_ALIGN);
		for (int k = 1; k <= gf_order; ++k)
		{
			const double C1 = cos(k * omega * 1);
			shift[2 * (k - 1) + 0] = C1 * 2.0;
			shift[2 * (k - 1) + 1] = C1;
			Gk_dct1[k - 1] = Gk[k] * totalInv;
		}
	}

	SpatialFilterSlidingDCT1_AVX_64F::SpatialFilterSlidingDCT1_AVX_64F(cv::Size imgSize, double sigma, int order, const SLIDING_DCT_SCHEDULE schedule)
		: SpatialFilterBase(imgSize, CV_64F)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT1_64_AVX;
		this->gf_order = order;
		this->sigma = sigma;
		this->schedule = schedule;
		this->dct_coeff_method = DCT_COEFFICIENTS::FULL_SEARCH_OPT;
		computeRadius(radius, true);

		this->imgSize = imgSize;
		allocBuffer();
	}

	SpatialFilterSlidingDCT1_AVX_64F::SpatialFilterSlidingDCT1_AVX_64F(const DCT_COEFFICIENTS method, const int dest_depth, const SLIDING_DCT_SCHEDULE schedule, const SpatialKernel skernel)
	{
		this->algorithm = SpatialFilterAlgorithm::SlidingDCT1_64_AVX;
		this->schedule = schedule;
		this->depth = CV_64F;
		this->dest_depth = dest_depth;
		this->dct_coeff_method = method;
	}

	SpatialFilterSlidingDCT1_AVX_64F::~SpatialFilterSlidingDCT1_AVX_64F()
	{
		_mm_free(GCn);
		_mm_free(shift);
		_mm_free(Gk_dct1);
		_mm_free(fn_hfilter);
		_mm_free(buffVFilter);
	}


	void SpatialFilterSlidingDCT1_AVX_64F::interleaveVerticalPixel(const cv::Mat& src, const int y, const int borderType, const int vpad)
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
	void SpatialFilterSlidingDCT1_AVX_64F::horizontalFilteringNaiveConvolution(const cv::Mat& src, cv::Mat& dst, const int order, const int borderType)
	{
		const int simdUnrollSize = 4;//4

		const int width = imgSize.width;
		const int height = imgSize.height;

		const int xstart = left;//left
		const int xend = get_simd_ceil(width - (left + right), simdUnrollSize) + xstart;

		__m256d total[4];
		__m256d F0;
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		AutoBuffer<__m256d> Z(order);

		for (int y = 0; y < height; y += simdUnrollSize)
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
						const __m256d sumA = _mm256_add_pd(fn_hfilter[(x + j - n)], fn_hfilter[(x + j + n)]);
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
				_mm256_storeupatch_pd(dstPtr, total, width);
				dstPtr += simdUnrollSize;
			}
		}
	}

	//horizontal filtering template(increment)
	template<int order>
	void SpatialFilterSlidingDCT1_AVX_64F::horizontalFilteringInnerXK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 4;

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVECD C1_2[order];
		SETVECD C1[order];
		SETVECD G[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			C1[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
			G[i] = _MM256_SETLUT_VECD(Gk_dct1[i]);
		}

		__m256d total[4];
		__m256d F0;
		__m256d Zp[order];
		__m256d Zc[order];
		__m256d delta_inner;
		__m256d dc, dp;
		__m256d t1, t2;

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

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int n = 1; n <= radius; ++n)
			{
				const __m256d sumA = _mm256_add_pd(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256d sumB = _mm256_add_pd(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
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
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_pd(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius)], fn_hfilter[(xstart + 1 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius)], fn_hfilter[(xstart + 2 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zp[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zc[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius)], fn_hfilter[(xstart + 3 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}

			// 3) main loop
			__m256d* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius];
			__m256d* buffHRR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256d* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius];
			__m256d* buffHLL = &fn_hfilter[xstart + simdUnrollSize - radius - 1];
			for (int x = 0; x < mainloop_simdsize; x++)
			{
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHRR, *buffHL);
				t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
				t2 = _mm256_add_pd(*buffHR++, *buffHL++);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[0] = Zp[0];
				Zc[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zp[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zc[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[0] = _mm256_add_pd(total[0], Zp[i]);
					Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[0] = _mm256_add_pd(total[0], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHRR, *buffHL);
				t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
				t2 = _mm256_add_pd(*buffHR++, *buffHL++);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[1] = Zc[0];
				Zp[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(*buffHRR, *buffHL);
				t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
				t2 = _mm256_add_pd(*buffHR++, *buffHL++);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[2] = Zp[0];
				Zc[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zp[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zc[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(*buffHRR, *buffHL);
				t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
				t2 = _mm256_add_pd(*buffHR++, *buffHL++);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				total[3] = Zc[0];
				Zp[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), Zc[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), delta_inner, Zp[0]));
				for (int i = 2; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = 1; i < order; i += 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}
		}
	}

	//vertical filtering template(increment)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT1_AVX_64F::verticalFilteringInnerXYK_inc(const cv::Mat& src, cv::Mat& dst, const int borderType)
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
		SETVECD C1[order];
		SETVECD G[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[2 * i + 0]);
			C1[i] = _MM256_SETLUT_VECD(shift[2 * i + 1]);
			G[i] = _MM256_SETLUT_VECD(Gk_dct1[i]);
		}

		__m256d totalA, totalB;
		__m256d deltaA;
		__m256d deltaB;
		__m256d dp, dc, dn;
		__m256d t1, t2, t3, t4;

		__m256d* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 4)
		{
			const __m256d pA = _mm256_loadu_pd(&srcPtr[(top + 0) * swidth + x]);
			const __m256d pB = _mm256_loadu_pd(&srcPtr[(top + 1) * swidth + x]);

			for (int i = 0; i < order; ++i)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(GCn[i]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(GCn[i]));
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

				for (int i = 0; i < order; ++i)
				{
					*ws++ = _mm256_fmadd_pd(pA, _mm256_set1_pd(GCn[order * r + i]), *ws);
					*ws++ = _mm256_fmadd_pd(pB, _mm256_set1_pd(GCn[order * r + i]), *ws);
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
				t1 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				t2 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
				pBP += 4;
				pCP += 4;
				pBM += 4;
				pCM += 4;

				totalA = ws[0];
				totalB = ws[1];
				deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				ws[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[1], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaB, ws[0]));

				for (int i = 2; i < order; i += 2)
				{
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

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
				t1 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				t2 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
				pBP += 4;
				pCP += 4;
				pBM += 4;
				pCM += 4;

				totalA = ws[0];
				totalB = ws[1];
				deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				ws[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[1], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaB, ws[0]));

				for (int i = 2; i < order; i += 2)
				{
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

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
				t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
				t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
				pAM += 4;
				pAP += 4;
				pBM += 4;
				pBP += 4;
				pCM += 4;
				pCP += 4;

				deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[1], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

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
				t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
				t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
				pAM += 4;
				pAP += 4;
				pBM += 4;
				pBP += 4;
				pCM += 4;
				pCP += 4;

				deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[1], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

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
				t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
				t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
				pAM += 4;
				pAP += 4;
				pBM += 4;
				pBP += 4;
				pCM += 4;
				pCP += 4;

				deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[1], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

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
				t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
				t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
				t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
				t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
				pAM += 4;
				pAP += 4;
				pBM += 4;
				pBP += 4;
				pCM += 4;
				pCP += 4;

				deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t2, t1);
				deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[0]), t4, t3);
				totalA = ws[0];
				totalB = ws[1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[0], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaA, ws[1]));
				ws[0] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[0]), ws[1], _mm256_fmadd_pd(_MM256_SET_VECD(G[0]), deltaB, ws[0]));
				for (int i = 2; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}
				for (int i = 1; i < order; i += 2)
				{
					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
					totalA = _mm256_add_pd(totalA, ws[i * 2]);
					ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
					totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
					ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
				}

				ws += 2 * order;

				_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
				__m256d temp = _mm256_add_pd(*ws, dc);
				*ws++ = _mm256_add_pd(temp, dn);

				dstPtr2 += 4;
			}
			dstPtr += 2 * dwidth;
		}
	}

	//horizontal filtering template(decrement)
	template<int order>
	void SpatialFilterSlidingDCT1_AVX_64F::horizontalFilteringInnerXK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const int simdUnrollSize = 4;

		const int dwidth = dst.cols;

		const int ystart = get_hfilterdct_ystart(src.rows, top, bottom, radius, simdUnrollSize);
		const int yend = get_hfilterdct_yend(src.rows, top, bottom, radius, simdUnrollSize);
		const int xstart = left;//left	
		const int xend = get_xend_slidingdct(left, get_simd_ceil(imgSize.width - (left + right), simdUnrollSize), dst.cols, simdUnrollSize);
		const int mainloop_simdsize = (xend - xstart) / simdUnrollSize - 1;

		SETVECD C1_2[order];
		SETVECD C1[order];
		SETVECD G[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; ++i)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[i * 2 + 0]);
			C1[i] = _MM256_SETLUT_VECD(shift[i * 2 + 1]);
			G[i] = _MM256_SETLUT_VECD(Gk_dct1[i]);
		}

		__m256d total[4];
		__m256d F0;
		__m256d Zp[order];
		__m256d Zc[order];
		__m256d delta_inner;
		__m256d dc, dp;
		__m256d t1, t2;

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

			// 1) initilization of Z0 and Z1 (1<=n<=radius)
			for (int n = 1; n <= radius; ++n)
			{
				const __m256d sumA = _mm256_add_pd(fn_hfilter[(xstart + 0 - n)], fn_hfilter[(xstart + 0 + n)]);
				const __m256d sumB = _mm256_add_pd(fn_hfilter[(xstart + 1 - n)], fn_hfilter[(xstart + 1 + n)]);
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
			if constexpr (order % 2 == 0)
			{
				dc = _mm256_sub_pd(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);

				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_pd(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius)], fn_hfilter[(xstart + 1 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius)], fn_hfilter[(xstart + 2 - radius)]);
				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius)], fn_hfilter[(xstart + 3 - radius)]);
				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 2; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 3; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}
			else if constexpr (order % 2 == 1)
			{
				dc = _mm256_sub_pd(fn_hfilter[(xstart + 0 + radius + 1)], fn_hfilter[(xstart + 0 - radius)]);

				total[0] = Zp[order - 1];
				for (int i = order - 2; i >= 0; i--)
				{
					total[0] = _mm256_add_pd(total[0], Zp[i]);
				}
				total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius + 1)], fn_hfilter[(xstart + 1 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 1 + radius)], fn_hfilter[(xstart + 1 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
				total[1] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[1] = _mm256_add_pd(total[1], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
				F0 = _mm256_add_pd(F0, dp);

				dc = _mm256_sub_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius + 1)], fn_hfilter[(xstart + 2 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 2 + radius)], fn_hfilter[(xstart + 2 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
				total[2] = Zp[order - 1];
				Zc[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zc[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[2] = _mm256_add_pd(total[2], Zp[i]);
					Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
				}
				total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
				F0 = _mm256_add_pd(F0, dc);

				dp = _mm256_sub_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius)]);
				t1 = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius + 1)], fn_hfilter[(xstart + 3 - radius - 1)]);
				t2 = _mm256_add_pd(fn_hfilter[(xstart + 3 + radius)], fn_hfilter[(xstart + 3 - radius)]);

				delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
				total[3] = Zc[order - 1];
				Zp[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
				for (int i = order - 3; i >= 0; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				for (int i = order - 2; i >= 1; i -= 2)
				{
					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
					total[3] = _mm256_add_pd(total[3], Zc[i]);
					Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
				}
				total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

				_mm256_transpose4_pd(total);
				_mm256_storeupatch_pd(dstPtr, total, dwidth);
				dstPtr += 4;
			}

			// 3) main loop
			__m256d* buffHR = &fn_hfilter[xstart + simdUnrollSize + radius];
			__m256d* buffHRR = &fn_hfilter[xstart + simdUnrollSize + radius + 1];
			__m256d* buffHL = &fn_hfilter[xstart + simdUnrollSize - radius];
			__m256d* buffHLL = &fn_hfilter[xstart + simdUnrollSize - radius - 1];
			if constexpr (order % 2 == 0)
			{
				for (int x = 0; x < mainloop_simdsize; x++)
				{
					F0 = _mm256_add_pd(F0, dp);

					dc = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[0] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[0] = _mm256_add_pd(total[0], Zp[i]);
						Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[0] = _mm256_add_pd(total[0], Zp[i]);
						Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
					F0 = _mm256_add_pd(F0, dc);

					dp = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[1] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[1] = _mm256_add_pd(total[1], Zc[i]);
						Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[1] = _mm256_add_pd(total[1], Zc[i]);
						Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
					F0 = _mm256_add_pd(F0, dp);

					dc = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[2] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[2] = _mm256_add_pd(total[2], Zp[i]);
						Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[2] = _mm256_add_pd(total[2], Zp[i]);
						Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
					F0 = _mm256_add_pd(F0, dc);

					dp = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[3] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[3] = _mm256_add_pd(total[3], Zc[i]);
						Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[3] = _mm256_add_pd(total[3], Zc[i]);
						Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

					_mm256_transpose4_pd(total);
					_mm256_storeupatch_pd(dstPtr, total, dwidth);
					dstPtr += 4;
				}
			}
			else if constexpr (order % 2 == 1)
			{
				for (int x = 0; x < mainloop_simdsize; x++)
				{
					F0 = _mm256_add_pd(F0, dp);

					dc = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[0] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[0] = _mm256_add_pd(total[0], Zp[i]);
						Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[0] = _mm256_add_pd(total[0], Zp[i]);
						Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					total[0] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[0]);
					F0 = _mm256_add_pd(F0, dc);

					dp = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[1] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[1] = _mm256_add_pd(total[1], Zc[i]);
						Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[1] = _mm256_add_pd(total[1], Zc[i]);
						Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					total[1] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[1]);
					F0 = _mm256_add_pd(F0, dp);

					dc = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[2] = Zp[order - 1];
					Zc[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zp[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zc[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[2] = _mm256_add_pd(total[2], Zp[i]);
						Zc[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[2] = _mm256_add_pd(total[2], Zp[i]);
						Zc[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zp[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zc[i]));
					}
					total[2] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[2]);
					F0 = _mm256_add_pd(F0, dc);

					dp = _mm256_sub_pd(*buffHRR, *buffHL);
					t1 = _mm256_add_pd(*buffHRR++, *buffHLL++);
					t2 = _mm256_add_pd(*buffHR++, *buffHL++);

					delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					total[3] = Zc[order - 1];
					Zp[order - 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), Zc[order - 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), delta_inner, Zp[order - 1]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[3] = _mm256_add_pd(total[3], Zc[i]);
						Zp[i] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						delta_inner = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						total[3] = _mm256_add_pd(total[3], Zc[i]);
						Zp[i] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), Zc[i], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), delta_inner, Zp[i]));
					}
					total[3] = _mm256_fmadd_pd(_MM256_SET_VECD(mG0), F0, total[3]);

					_mm256_transpose4_pd(total);
					_mm256_storeupatch_pd(dstPtr, total, dwidth);
					dstPtr += 4;
				}
			}
		}
	}

	//vertical filtering template(decrement)
	template<int order, typename destT>
	void SpatialFilterSlidingDCT1_AVX_64F::verticalFilteringInnerXYK_dec(const cv::Mat& src, cv::Mat& dst, const int borderType)
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
		SETVECD C1[order];
		SETVECD G[order];
		SETVECD mG0 = _MM256_SETLUT_VECD(G0);
		for (int i = 0; i < order; i++)
		{
			C1_2[i] = _MM256_SETLUT_VECD(shift[2 * i + 0]);
			C1[i] = _MM256_SETLUT_VECD(shift[2 * i + 1]);
			G[i] = _MM256_SETLUT_VECD(Gk_dct1[i]);
		}

		__m256d totalA, totalB;
		__m256d deltaA;
		__m256d deltaB;
		__m256d dp, dc, dn;
		__m256d t1, t2, t3, t4;

		__m256d* ws = buffVFilter;

		// 1) initilization of Z0 and Z1 (n=0)
		for (int x = xstart; x < xend; x += 4)
		{
			const __m256d pA = _mm256_loadu_pd(&srcPtr[(top + 0) * swidth + x]);
			const __m256d pB = _mm256_loadu_pd(&srcPtr[(top + 1) * swidth + x]);

			for (int i = 0; i < order; ++i)
			{
				*ws++ = _mm256_mul_pd(pA, _mm256_set1_pd(GCn[i]));
				*ws++ = _mm256_mul_pd(pB, _mm256_set1_pd(GCn[i]));
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

				for (int i = 0; i < order; ++i)
				{
					*ws++ = _mm256_fmadd_pd(pA, _mm256_set1_pd(GCn[order * r + i]), *ws);
					*ws++ = _mm256_fmadd_pd(pB, _mm256_set1_pd(GCn[order * r + i]), *ws);
				}
				*ws++ = _mm256_add_pd(*ws, pA);
			}
		}

		// 2) initial output computing for y=0,1
		if constexpr (order % 2 == 0)
		{
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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pBP += 4;
					pCP += 4;
					pBM += 4;
					pCM += 4;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 2; i >= 0; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pBP += 4;
					pCP += 4;
					pBM += 4;
					pCM += 4;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 2; i >= 0; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
					__m256d temp = _mm256_add_pd(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB), rem);
					*ws++ = _mm256_add_pd(temp, dn);

					dstPtr2 += 4;
				}
				dstPtr += 2 * dwidth;
			}
		}
		else if constexpr (order % 2 == 1)
		{
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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pBP += 4;
					pCP += 4;
					pBM += 4;
					pCM += 4;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 3; i >= 0; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pBP += 4;
					pCP += 4;
					pBM += 4;
					pCM += 4;

					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1];
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					ws[(order - 1) * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));

					for (int i = order - 3; i >= 0; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
					__m256d temp = _mm256_add_pd(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB), rem);
					*ws++ = _mm256_add_pd(temp, dn);
				}
				dstPtr += 2 * dwidth;
			}
		}

		// 3) main loop
		if constexpr (order % 2 == 0)
		{
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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
					__m256d temp = _mm256_add_pd(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB), rem);
					*ws++ = _mm256_add_pd(temp, dn);
				}
				dstPtr += 2 * dwidth;
			}
		}
		else if constexpr (order % 2 == 1)
		{
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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
					__m256d temp = _mm256_add_pd(*ws, dc);
					_mm256_storescalar_auto(dstPtr2 + dwidth, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), temp, totalB), rem);
					*ws++ = _mm256_add_pd(temp, dn);
				}
				dstPtr += 2 * dwidth;
			}
		}

		if (!isEven)
		{
			if constexpr (order % 2 == 0)
			{
				const int y = hend;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 2; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 3; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
					__m256d temp = _mm256_add_pd(*ws, dc);
					*ws++ = _mm256_add_pd(temp, dn);
				}
			}
			else if constexpr (order % 2 == 1)
			{
				const int y = hend;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

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
					t1 = _mm256_add_pd(_mm256_loadu_pd(pAM), _mm256_loadu_pd(pBP));
					t2 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pAP));
					t3 = _mm256_add_pd(_mm256_loadu_pd(pBM), _mm256_loadu_pd(pCP));
					t4 = _mm256_add_pd(_mm256_loadu_pd(pCM), _mm256_loadu_pd(pBP));
					pAM += 4;
					pAP += 4;
					pBM += 4;
					pBP += 4;
					pCM += 4;
					pCP += 4;

					deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t2, t1);
					deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[order - 1]), t4, t3);
					totalA = ws[(order - 1) * 2];
					totalB = ws[(order - 1) * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaA, ws[(order - 1) * 2 + 1]));
					ws[(order - 1) * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[order - 1]), ws[(order - 1) * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[order - 1]), deltaB, ws[(order - 1) * 2]));
					for (int i = order - 3; i >= 0; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmsub_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmadd_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					for (int i = order - 2; i >= 1; i -= 2)
					{
						deltaA = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t2, t1);
						deltaB = _mm256_fnmadd_pd(_MM256_SET_VECD(C1[i]), t4, t3);
						totalA = _mm256_add_pd(totalA, ws[i * 2]);
						ws[i * 2 + 1] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaA, ws[i * 2 + 1]));
						totalB = _mm256_add_pd(totalB, ws[i * 2 + 1]);
						ws[i * 2] = _mm256_fmadd_pd(_MM256_SET_VECD(C1_2[i]), ws[i * 2 + 1], _mm256_fmsub_pd(_MM256_SET_VECD(G[i]), deltaB, ws[i * 2]));
					}
					ws += 2 * order;

					_mm256_storescalar_auto(dstPtr2, _mm256_fmadd_pd(_MM256_SET_VECD(mG0), *ws, totalA), rem);
					__m256d temp = _mm256_add_pd(*ws, dc);
					*ws++ = _mm256_add_pd(temp, dn);
				}
			}
		}
	}

	//filtering
	void SpatialFilterSlidingDCT1_AVX_64F::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F || src.depth() == CV_64F);

		dst.create(imgSize, (dest_depth < 0) ? src.depth() : dest_depth);

		if (schedule == SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION)
		{
			cout << "not supported" << endl;
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
			//const bool isInc = true;
			const bool isInc = false;
			if (isInc)
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT1_64F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXK_inc<1>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<1, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<1, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<1, uchar>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXK_inc<2>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<2, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<2, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<2, uchar>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXK_inc<3>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<3, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<3, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<3, uchar>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXK_inc<4>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<4, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<4, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<4, uchar>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXK_inc<5>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<5, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<5, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<5, uchar>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXK_inc<6>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<6, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<6, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<6, uchar>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXK_inc<7>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<7, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<7, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<7, uchar>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXK_inc<8>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<8, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<8, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<8, uchar>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXK_inc<9>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<9, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<9, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<9, uchar>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXK_inc<10>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<10, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<10, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<10, uchar>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXK_inc<11>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<11, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<11, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<11, uchar>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXK_inc<12>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<12, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<12, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<12, uchar>(inter, dst, borderType);
					break;
				case 13:
					horizontalFilteringInnerXK_inc<13>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<13, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<13, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<13, uchar>(inter, dst, borderType);
					break;
				case 14:
					horizontalFilteringInnerXK_inc<14>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<14, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<14, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<14, uchar>(inter, dst, borderType);
					break;
				case 15:
					horizontalFilteringInnerXK_inc<15>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<15, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<15, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<15, uchar>(inter, dst, borderType);
					break;
				case 16:
					horizontalFilteringInnerXK_inc<16>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<16, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<16, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<16, uchar>(inter, dst, borderType);
					break;
				case 17:
					horizontalFilteringInnerXK_inc<17>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<17, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<17, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<17, uchar>(inter, dst, borderType);
					break;
				case 18:
					horizontalFilteringInnerXK_inc<18>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<18, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<18, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<18, uchar>(inter, dst, borderType);
					break;
				case 19:
					horizontalFilteringInnerXK_inc<19>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<19, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<19, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<19, uchar>(inter, dst, borderType);
					break;
				case 20:
					horizontalFilteringInnerXK_inc<20>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<20, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<20, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<20, uchar>(inter, dst, borderType);
					break;
				case 21:
					horizontalFilteringInnerXK_inc<21>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<21, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<21, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<21, uchar>(inter, dst, borderType);
					break;
				case 22:
					horizontalFilteringInnerXK_inc<22>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<22, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<22, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<22, uchar>(inter, dst, borderType);
					break;
				case 23:
					horizontalFilteringInnerXK_inc<23>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<23, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<23, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<23, uchar>(inter, dst, borderType);
					break;
				case 24:
					horizontalFilteringInnerXK_inc<24>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<24, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<24, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<24, uchar>(inter, dst, borderType);
					break;
				case 25:
					horizontalFilteringInnerXK_inc<25>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_inc<25, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_inc<25, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_inc<25, uchar>(inter, dst, borderType);
					break;
#endif
				default:
					cout << "max order is 25" << endl;
					break;
				}
			}
			else
			{
				switch (gf_order)
				{
#ifdef COMPILE_GF_DCT1_64F_ORDER_TEMPLATE
				case 1:
					horizontalFilteringInnerXK_dec<1>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<1, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<1, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<1, uchar>(inter, dst, borderType);
					break;
				case 2:
					horizontalFilteringInnerXK_dec<2>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<2, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<2, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<2, uchar>(inter, dst, borderType);
					break;
				case 3:
					horizontalFilteringInnerXK_dec<3>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<3, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<3, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<3, uchar>(inter, dst, borderType);
					break;
				case 4:
					horizontalFilteringInnerXK_dec<4>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<4, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<4, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<4, uchar>(inter, dst, borderType);
					break;
				case 5:
					horizontalFilteringInnerXK_dec<5>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<5, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<5, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<5, uchar>(inter, dst, borderType);
					break;
				case 6:
					horizontalFilteringInnerXK_dec<6>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<6, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<6, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<6, uchar>(inter, dst, borderType);
					break;
				case 7:
					horizontalFilteringInnerXK_dec<7>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<7, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<7, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<7, uchar>(inter, dst, borderType);
					break;
				case 8:
					horizontalFilteringInnerXK_dec<8>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<8, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<8, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<8, uchar>(inter, dst, borderType);
					break;
				case 9:
					horizontalFilteringInnerXK_dec<9>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<9, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<9, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<9, uchar>(inter, dst, borderType);
					break;
				case 10:
					horizontalFilteringInnerXK_dec<10>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<10, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<10, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<10, uchar>(inter, dst, borderType);
					break;
				case 11:
					horizontalFilteringInnerXK_dec<11>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<11, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<11, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<11, uchar>(inter, dst, borderType);
					break;
				case 12:
					horizontalFilteringInnerXK_dec<12>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<12, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<12, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<12, uchar>(inter, dst, borderType);
					break;
				case 13:
					horizontalFilteringInnerXK_dec<13>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<13, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<13, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<13, uchar>(inter, dst, borderType);
					break;
				case 14:
					horizontalFilteringInnerXK_dec<14>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<14, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<14, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<14, uchar>(inter, dst, borderType);
					break;
				case 15:
					horizontalFilteringInnerXK_dec<15>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<15, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<15, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<15, uchar>(inter, dst, borderType);
					break;
				case 16:
					horizontalFilteringInnerXK_dec<16>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<16, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<16, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<16, uchar>(inter, dst, borderType);
					break;
				case 17:
					horizontalFilteringInnerXK_dec<17>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<17, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<17, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<17, uchar>(inter, dst, borderType);
					break;
				case 18:
					horizontalFilteringInnerXK_dec<18>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<18, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<18, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<18, uchar>(inter, dst, borderType);
					break;
				case 19:
					horizontalFilteringInnerXK_dec<19>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<19, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<19, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<19, uchar>(inter, dst, borderType);
					break;
				case 20:
					horizontalFilteringInnerXK_dec<20>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<20, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<20, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<20, uchar>(inter, dst, borderType);
					break;
				case 21:
					horizontalFilteringInnerXK_dec<21>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<21, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<21, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<21, uchar>(inter, dst, borderType);
					break;
				case 22:
					horizontalFilteringInnerXK_dec<22>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<22, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<22, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<22, uchar>(inter, dst, borderType);
					break;
				case 23:
					horizontalFilteringInnerXK_dec<23>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<23, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<23, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<23, uchar>(inter, dst, borderType);
					break;
				case 24:
					horizontalFilteringInnerXK_dec<24>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<24, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<24, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<24, uchar>(inter, dst, borderType);
					break;
				case 25:
					horizontalFilteringInnerXK_dec<25>(src, inter, borderType);
					if (dst.depth() == CV_64F) verticalFilteringInnerXYK_dec<25, double>(inter, dst, borderType);
					else if (dst.depth() == CV_32F) verticalFilteringInnerXYK_dec<25, float>(inter, dst, borderType);
					else if (dst.depth() == CV_8U) verticalFilteringInnerXYK_dec<25, uchar>(inter, dst, borderType);
					break;
#endif
				default:
					cout << "max order is 25" << endl;
					break;
				}
			}
		}
	}

	void SpatialFilterSlidingDCT1_AVX_64F::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
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