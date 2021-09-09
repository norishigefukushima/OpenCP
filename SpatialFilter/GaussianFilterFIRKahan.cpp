#include "stdafx.h"
#include <fpplus/fpplus.h>

using namespace std;
using namespace cv;

namespace cp
{
	static double getOuterWeight(const double gauss_space_coeff, const int radius, const double out_radius)
	{
		double ret = 0.0;
		for (int i = radius + 1; i <= out_radius; i++)
		{
			const double v = exp(i * i * gauss_space_coeff);
			ret += v;
		}
		return ret;
	}

	static int setSpaceKernel1DH(double* space_weight, int* space_ofs, const int radiusH, const double gauss_space_coeff)
	{
		int maxk = 0;
		for (int j = -radiusH; j <= radiusH; j++)
		{
			space_weight[maxk] = exp(j * j * gauss_space_coeff);
			space_ofs[maxk++] = j;
		}
		return maxk;
	}

	static int setSpaceKernel1DV(double* space_weight, int* space_ofs, const int radiusV, const double gauss_space_coeff, const int imstep)
	{
		int maxk = 0;
		for (int i = -radiusV; i <= radiusV; i++)
		{
			space_weight[maxk] = exp(i * i * gauss_space_coeff);
			space_ofs[maxk++] = i * imstep;
		}
		return maxk;
	}

	struct SpaceKernelData
	{
		double r;
		double w;
		int ofs;
		SpaceKernelData(const double r, const double w, const int ofs)
		{
			this->r = r;
			this->w = w;
			this->ofs = ofs;
		}
	};

	static bool cmpSpaceKernelData(const SpaceKernelData& a, const SpaceKernelData& b)
	{
		return a.r > b.r;
	}

	static int setSpaceKernelSort1DH(double* space_weight, int* space_ofs, const int radiusH, const double gauss_space_coeff)
	{
		int maxk = 0;
		vector<SpaceKernelData> data;

		for (int j = -radiusH; j <= radiusH; j++)
		{
			data.push_back(SpaceKernelData(j, exp(double(j * j) * gauss_space_coeff), j));
			maxk++;
		}

		sort(data.begin(), data.end(), cmpSpaceKernelData);
		for (int i = 0; i < maxk; i++)
		{
			//cout << data[i].r << endl;
			space_weight[i] = data[i].w;
			space_ofs[i] = data[i].ofs;
		}
		return maxk;
	}

	static int setSpaceKernelSort1DV(double* space_weight, int* space_ofs, const int radiusV, const double gauss_space_coeff, const int imstep)
	{
		int maxk = 0;
		vector<SpaceKernelData> data;
		for (int i = -radiusV; i <= radiusV; i++)
		{
			data.push_back(SpaceKernelData(i, exp(double(i * i) * gauss_space_coeff), i * imstep));
			maxk++;
		}
		sort(data.begin(), data.end(), cmpSpaceKernelData);
		for (int i = 0; i < maxk; i++)
		{
			//cout << data[i].r << endl;
			space_weight[i] = data[i].w;
			space_ofs[i] = data[i].ofs;
		}
		return maxk;
	}

	void GaussianFilterSepKahan64f(const Mat& src, Mat& dst, const Size kernelSize, double sigma_space, const int borderType, const bool isSort)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }

		CV_Assert(!src.empty());
		CV_Assert(!dst.empty());
		CV_Assert(src.type() == CV_64FC1);
		CV_Assert(src.channels() == 1);
		CV_Assert(src.type() == dst.type());
		CV_Assert(src.size() == dst.size());

		if (sigma_space <= 0) sigma_space = 1;
		const double gauss_space_coeff = -1.0 / (2.0 * sigma_space * sigma_space);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat border;
		const int WIDTH = get_simd_ceil(src.cols, 4);
		const int pad = WIDTH - src.cols;
		const int rem = 4 - pad;
		copyMakeBorder(src, border, radiusV, radiusV, radiusH, radiusH + pad, borderType);

		Mat borderInterH(border.size(), CV_64F);

		AutoBuffer<double> space_weight1(kernelSize.width);
		AutoBuffer<int> space_ofs1(kernelSize.width);

		AutoBuffer<double> space_weight2(kernelSize.height);
		AutoBuffer<int> space_ofs2(kernelSize.height);

		int maxk = 0;
		if (isSort)
		{
			maxk = setSpaceKernelSort1DH(space_weight1, space_ofs1, radiusH, gauss_space_coeff);
			maxk = setSpaceKernelSort1DV(space_weight2, space_ofs2, radiusV, gauss_space_coeff, border.cols);
		}
		else
		{
			maxk = setSpaceKernel1DH(space_weight1, space_ofs1, radiusH, gauss_space_coeff);
			maxk = setSpaceKernel1DV(space_weight2, space_ofs2, radiusV, gauss_space_coeff, border.cols);
		}

		const bool isUseOuterWeight = true;
		if (isUseOuterWeight)
		{
			if (!isSort)
			{
				space_weight1[0] = space_weight1[kernelSize.width - 1] = space_weight1[0] + getOuterWeight(gauss_space_coeff, radiusH, (int)ceil(sigma_space * 10));
				space_weight2[0] = space_weight2[kernelSize.height - 1] = space_weight2[0] + getOuterWeight(gauss_space_coeff, radiusV, (int)ceil(sigma_space * 10));
			}
			else
			{
				space_weight1[0] = space_weight1[1] = space_weight1[1] + getOuterWeight(gauss_space_coeff, radiusH, (int)ceil(sigma_space * 10));;
				space_weight2[0] = space_weight2[1] = space_weight2[1] + getOuterWeight(gauss_space_coeff, radiusV, (int)ceil(sigma_space * 10));;
			}
		}

		__m256dd wvald = _mm256_setzero_pdd();
		for (int k = 0; k < maxk; k++)
		{
			_mm256_addkahan_pdd(_mm256_set1_pd(space_weight1[k]), wvald);
		}
		const __m256d wval = _mm256_mul_pd(wvald.hi, wvald.hi);

		// hfilter
		{
			for (int i = 0; i < border.rows; i++)
			{
				const double* sptr = border.ptr<double>(i, radiusH);
				double* dptr = borderInterH.ptr<double>(i, radiusH);//borderInterH.ptr<double>(i, radiusH)
				int j = 0;

				for (; j < WIDTH; j += 4)//4 pixel unit
				{
					int* ofs = space_ofs1;
					double* spw = space_weight1;

					const double* sptrj = sptr + j;

					/*
					__m256d tval = _mm256_setzero_pd();
					__m256d wval = _mm256_setzero_pd();
					for (k = 0; k < maxk; k++, ofs++, spw++)
					{
						const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
						const __m256d sw = _mm256_set1_pd(*spw);

						tval = _mm256_fmadd_pd(sw, sref, tval);
						wval = _mm256_add_pd(sw, wval);
					}
					_mm256_storeu_pd((dptr + j), _mm256_div_pd(tval, wval));
					*/

					__m256dd tval = _mm256_setzero_pdd();

					for (int k = 0; k < maxk; k++, ofs++, spw++)
					{
						const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
						const __m256d sw = _mm256_set1_pd(*spw);

						_mm256_fmakahan_pdd(sw, sref, tval);
					}
					_mm256_storeu_pd((dptr + j), tval.hi);
				}
			}
		}

		// vfilter
		{
			for (int i = 0; i < src.rows; i++)
			{
				const double* sptr = borderInterH.ptr<double>(i + radiusV, radiusH);
				double* dptr = dst.ptr<double>(i);//dst.ptr<double>(i)
				int j = 0;
				for (; j < WIDTH - 4; j += 4)//4 pixel unit
				{
					const int* ofs = space_ofs2;
					const double* spw = space_weight2;
					const double* sptrj = sptr + j;

					/*
					__m256d tval = _mm256_setzero_pd();
					__m256d wval = _mm256_setzero_pd();
					for (k = 0; k < maxk; k++, ofs++, spw++)
					{
						const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
						const __m256d sw = _mm256_set1_pd(*spw);

						tval = _mm256_fmadd_pd(sw, sref, tval);
						wval = _mm256_add_pd(sw, wval);
					}
					_mm256_storeu_pd((dptr + j), _mm256_div_pd(tval, wval));
					*/

					__m256dd tval = _mm256_setzero_pdd();
					for (int k = 0; k < maxk; k++, ofs++, spw++)
					{
						const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
						const __m256d sw = _mm256_set1_pd(*spw);
						_mm256_fmakahan_pdd(sw, sref, tval);
					}
					_mm256_storeu_pd((dptr + j), _mm256_div_pd(tval.hi, wval));
				}

				for (; j < src.cols; j += 4)//4 pixel unit
				{
					const int* ofs = space_ofs2;
					const double* spw = space_weight2;
					const double* sptrj = sptr + j;

					__m256dd tval = _mm256_setzero_pdd();
					for (int k = 0; k < maxk; k++, ofs++, spw++)
					{
						const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
						const __m256d sw = _mm256_set1_pd(*spw);
						_mm256_fmakahan_pdd(sw, sref, tval);
					}
					_mm256_storescalar_pd((dptr + j), _mm256_div_pd(tval.hi, wval), rem);
				}
			}
		}
	}


	GaussianFilterFIRKahan::GaussianFilterFIRKahan(cv::Size imgSize, double sigma, int trunc, int depth)
		: SpatialFilterBase(imgSize, depth)
	{
		this->algorithm = SpatialFilterAlgorithm::FIR_KAHAN;
		this->gf_order = trunc;
		this->sigma = sigma;
		radius = (int)ceil(trunc * sigma);
		d = 2 * radius + 1;
	}

	GaussianFilterFIRKahan::GaussianFilterFIRKahan(const int dest_depth)
	{
		this->algorithm = SpatialFilterAlgorithm::FIR_KAHAN;
		this->dest_depth = dest_depth;
		this->depth = CV_64F;
	}

	GaussianFilterFIRKahan::~GaussianFilterFIRKahan()
	{
		;
	}

	void GaussianFilterFIRKahan::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		const bool isSort = false;

		dst.create(src.size(), dest_depth);

		if (src.depth() == CV_64F && dst.depth() == CV_64F)
		{
			//cout << "src64f dst64f" << endl;
			GaussianFilterSepKahan64f(src, dst, Size(d, d), sigma, borderType, isSort);
		}
		else
		{
			if (src.depth() != CV_64F)
			{
				//cout << "src!=64f" << endl;
				src.convertTo(src64, CV_64F);//converting type
				internalBuff.create(src64.size(), CV_64F);

				GaussianFilterSepKahan64f(src64, internalBuff, Size(d, d), sigma, borderType, isSort);
				internalBuff.convertTo(dst, dest_depth);
			}
			else
			{
				internalBuff.create(src.size(), CV_64F);
				GaussianFilterSepKahan64f(src, internalBuff, Size(d, d), sigma, borderType, isSort);
				internalBuff.convertTo(dst, dest_depth);
			}
		}
	}

	void GaussianFilterFIRKahan::filter(const cv::Mat& _src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		this->gf_order = order;
		this->sigma = sigma;
		this->radius = (gf_order == 0) ? radius : (int)ceil(gf_order * sigma);
		this->d = 2 * radius + 1;

		body(_src, dst, borderType);
	}
}