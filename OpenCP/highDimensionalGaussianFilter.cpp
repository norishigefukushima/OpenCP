#include "highDimensionalGaussianFilter.hpp"
#include "inlineCVFunctions.hpp"
#include "inlineSIMDFunctions.hpp"
#include "tiling.hpp"
#include "patchPCA.hpp"
#include "color.hpp"
#include "Plot.hpp"//RGB histogram
#include "imshowExtension.hpp"//imshowScale
using namespace std;
using namespace cv;

namespace cp
{
#pragma region highDimensionalGaussianFilter
	inline double get_l2norm_maxval(const int channel)
	{
		return sqrt(1024 * 1024 * channel);
	}

	template<int gchannel, HDGFSchedule schedule>
	void highDimensionalGaussianFilter32F_(const cv::Mat src, const cv::Mat guide, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border)
	{
		const int schannels = src.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);

		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			float* wsptr = weight_space.ptr<float>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = (float)std::exp(distance * coeff_space);
			}
		}
		const float coeff_range = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff_range = _mm256_set1_ps(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<float> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				float* dptr = dst.ptr<float>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer = _mm256_setzero_ps();
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr = srcborder.ptr<float>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							AutoBuffer<__m256> msrc(guide.channels());
							for (int c = 0; c < gchannel; c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								msrc[c] = _mm256_loadu_ps(srcptr + i);
							}
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								__m256 sub = _mm256_sub_ps(msrc[c], _mm256_loadu_ps(gptr + x + i));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_ps(dptr + i - r, _mm256_div_ps(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				float* dptr0 = split_dst[0].ptr<float>(j);
				float* dptr1 = split_dst[1].ptr<float>(j);
				float* dptr2 = split_dst[2].ptr<float>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer0 = _mm256_setzero_ps();
					__m256 mnumer1 = _mm256_setzero_ps();
					__m256 mnumer2 = _mm256_setzero_ps();

					AutoBuffer<__m256> msrc(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						float* srcptr = split_guideborder[c].ptr<float>(j + r, i + r);
						msrc[c] = _mm256_loadu_ps(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j, i);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j, i);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(msrc[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer0 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_ps(dptr0 + i, _mm256_div_ps(mnumer0, mdenom));
					_mm256_storeu_ps(dptr1 + i, _mm256_div_ps(mnumer1, mdenom));
					_mm256_storeu_ps(dptr2 + i, _mm256_div_ps(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<float*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<float>(j);
				}
				AutoBuffer<__m256> mnumer(schannels);
				AutoBuffer<float*> Iptr(schannels);
				AutoBuffer<__m256> msrc(gchannel);

				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_ps();
					}

					for (int c = 0; c < gchannel; c++)
					{
						float* srcptr = split_guideborder[c].ptr<float>(j + r, i + r);
						msrc[c] = _mm256_loadu_ps(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<float>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(msrc[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_ps(dptr[c] + i, _mm256_div_ps(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}

	void highDimensionalGaussianFilter32F_Dim(const cv::Mat src, const cv::Mat guide, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border, HDGFSchedule schedule)
	{
		const int schannels = src.channels();
		const int gchannel = guide.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);

		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			float* wsptr = weight_space.ptr<float>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = (float)std::exp(distance * coeff_space);
			}
		}
		const float coeff_range = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff_range = _mm256_set1_ps(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<float> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				float* dptr = dst.ptr<float>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer = _mm256_setzero_ps();
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr = srcborder.ptr<float>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							AutoBuffer<__m256> msrc(guide.channels());
							for (int c = 0; c < gchannel; c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								msrc[c] = _mm256_loadu_ps(srcptr + i);
							}
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								__m256 sub = _mm256_sub_ps(msrc[c], _mm256_loadu_ps(gptr + x + i));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_ps(dptr + i - r, _mm256_div_ps(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				float* dptr0 = split_dst[0].ptr<float>(j);
				float* dptr1 = split_dst[1].ptr<float>(j);
				float* dptr2 = split_dst[2].ptr<float>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer0 = _mm256_setzero_ps();
					__m256 mnumer1 = _mm256_setzero_ps();
					__m256 mnumer2 = _mm256_setzero_ps();

					AutoBuffer<__m256> msrc(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						float* srcptr = split_guideborder[c].ptr<float>(j + r, i + r);
						msrc[c] = _mm256_loadu_ps(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j, i);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j, i);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(msrc[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer0 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_ps(dptr0 + i, _mm256_div_ps(mnumer0, mdenom));
					_mm256_storeu_ps(dptr1 + i, _mm256_div_ps(mnumer1, mdenom));
					_mm256_storeu_ps(dptr2 + i, _mm256_div_ps(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<float*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<float>(j);
				}
				AutoBuffer<__m256> mnumer(schannels);

				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_ps();
					}

					AutoBuffer<__m256> msrc(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						float* srcptr = split_guideborder[c].ptr<float>(j + r, i + r);
						msrc[c] = _mm256_loadu_ps(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);
						AutoBuffer<float*> Iptr(schannels);
						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<float>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(msrc[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_ps(dptr[c] + i, _mm256_div_ps(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}

	template<int gchannel, HDGFSchedule schedule>
	void highDimensionalGaussianFilter64F_(const cv::Mat src, const cv::Mat guide, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border)
	{
		const int schannels = src.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);

		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			double* wsptr = weight_space.ptr<double>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = std::exp(distance * coeff_space);
			}
		}
		const double coeff_range = 1.0 / (-2.0 * sigma_range * sigma_range);
		const __m256d mcoeff_range = _mm256_set1_pd(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<double> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				double* dptr = dst.ptr<double>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer = _mm256_setzero_pd();
					for (int y = -r; y <= r; y++)
					{
						double* wsptr = weight_space.ptr<double>(y + r);
						double* Iptr = srcborder.ptr<double>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							AutoBuffer<__m256d> msrc(guide.channels());
							for (int c = 0; c < gchannel; c++)
							{
								double* srcptr = split_guideborder[c].ptr<double>(j);
								msrc[c] = _mm256_loadu_pd(srcptr + i);
							}
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j);
								__m256d sub = _mm256_sub_pd(msrc[c], _mm256_loadu_pd(gptr + x + i));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE) mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_pd(dptr + i - r, _mm256_div_pd(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				double* dptr0 = split_dst[0].ptr<double>(j);
				double* dptr1 = split_dst[1].ptr<double>(j);
				double* dptr2 = split_dst[2].ptr<double>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer0 = _mm256_setzero_pd();
					__m256d mnumer1 = _mm256_setzero_pd();
					__m256d mnumer2 = _mm256_setzero_pd();

					AutoBuffer<__m256d> msrc(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						double* srcptr = split_guideborder[c].ptr<double>(j + r, i + r);
						msrc[c] = _mm256_loadu_pd(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);
						double* Iptr0 = split_srcborder[0].ptr<double>(y + j, i);
						double* Iptr1 = split_srcborder[1].ptr<double>(y + j, i);
						double* Iptr2 = split_srcborder[2].ptr<double>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(msrc[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer0 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_pd(dptr0 + i, _mm256_div_pd(mnumer0, mdenom));
					_mm256_storeu_pd(dptr1 + i, _mm256_div_pd(mnumer1, mdenom));
					_mm256_storeu_pd(dptr2 + i, _mm256_div_pd(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<double*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<double>(j);
				}
				AutoBuffer<__m256d> mnumer(schannels);
				AutoBuffer<double*> Iptr(schannels);
				AutoBuffer<__m256d> msrc(gchannel);

				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_pd();
					}

					for (int c = 0; c < gchannel; c++)
					{
						double* srcptr = split_guideborder[c].ptr<double>(j + r, i + r);
						msrc[c] = _mm256_loadu_pd(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<double>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(msrc[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_pd(dptr[c] + i, _mm256_div_pd(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}

	void highDimensionalGaussianFilter64F_Dim(const cv::Mat src, const cv::Mat guide, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border, HDGFSchedule schedule)
	{
		const int schannels = src.channels();
		const int gchannel = guide.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);

		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			double* wsptr = weight_space.ptr<double>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = std::exp(distance * coeff_space);
			}
		}
		const double coeff_range = 1.0 / (-2.0 * sigma_range * sigma_range);
		const __m256d mcoeff_range = _mm256_set1_pd(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<double> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				double* dptr = dst.ptr<double>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer = _mm256_setzero_pd();
					for (int y = -r; y <= r; y++)
					{
						double* wsptr = weight_space.ptr<double>(y + r);
						double* Iptr = srcborder.ptr<double>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							AutoBuffer<__m256d> msrc(guide.channels());
							for (int c = 0; c < gchannel; c++)
							{
								double* srcptr = split_guideborder[c].ptr<double>(j);
								msrc[c] = _mm256_loadu_pd(srcptr + i);
							}
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j);
								__m256d sub = _mm256_sub_pd(msrc[c], _mm256_loadu_pd(gptr + x + i));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if (schedule == HDGFSchedule::COMPUTE) mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_pd(dptr + i - r, _mm256_div_pd(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				double* dptr0 = split_dst[0].ptr<double>(j);
				double* dptr1 = split_dst[1].ptr<double>(j);
				double* dptr2 = split_dst[2].ptr<double>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer0 = _mm256_setzero_pd();
					__m256d mnumer1 = _mm256_setzero_pd();
					__m256d mnumer2 = _mm256_setzero_pd();

					AutoBuffer<__m256d> msrc(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						double* srcptr = split_guideborder[c].ptr<double>(j + r, i + r);
						msrc[c] = _mm256_loadu_pd(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);
						double* Iptr0 = split_srcborder[0].ptr<double>(y + j, i);
						double* Iptr1 = split_srcborder[1].ptr<double>(y + j, i);
						double* Iptr2 = split_srcborder[2].ptr<double>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(msrc[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer0 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_pd(dptr0 + i, _mm256_div_pd(mnumer0, mdenom));
					_mm256_storeu_pd(dptr1 + i, _mm256_div_pd(mnumer1, mdenom));
					_mm256_storeu_pd(dptr2 + i, _mm256_div_pd(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<double*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<double>(j);
				}
				AutoBuffer<__m256d> mnumer(schannels);
				AutoBuffer<double*> Iptr(schannels);
				AutoBuffer<__m256d> msrc(gchannel);

				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_pd();
					}

					for (int c = 0; c < gchannel; c++)
					{
						double* srcptr = split_guideborder[c].ptr<double>(j + r, i + r);
						msrc[c] = _mm256_loadu_pd(srcptr);
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<double>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(msrc[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_pd(dptr[c] + i, _mm256_div_pd(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}


	template<int gchannel, HDGFSchedule schedule>
	void highDimensionalGaussianFilterCenterReplace32F_(const cv::Mat src, const cv::Mat guide, const cv::Mat center, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border)
	{
		const int schannels = src.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);
		vector<Mat> split_center;
		split(center, split_center);

		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			float* wsptr = weight_space.ptr<float>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = (float)std::exp(distance * coeff_space);
			}
		}
		const float coeff_range = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff_range = _mm256_set1_ps(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<float> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				float* dptr = dst.ptr<float>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer = _mm256_setzero_ps();
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr = srcborder.ptr<float>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							AutoBuffer<__m256> mcenter(guide.channels());
							for (int c = 0; c < gchannel; c++)
							{
								mcenter[c] = _mm256_loadu_ps(split_center[c].ptr<float>(j - r, i - r));
							}
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								__m256 sub = _mm256_sub_ps(mcenter[c], _mm256_loadu_ps(gptr + x + i));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_ps(dptr + i - r, _mm256_div_ps(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				float* dptr0 = split_dst[0].ptr<float>(j);
				float* dptr1 = split_dst[1].ptr<float>(j);
				float* dptr2 = split_dst[2].ptr<float>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer0 = _mm256_setzero_ps();
					__m256 mnumer1 = _mm256_setzero_ps();
					__m256 mnumer2 = _mm256_setzero_ps();

					AutoBuffer<__m256> mcenter(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_ps(split_center[c].ptr<float>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j, i);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j, i);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(mcenter[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer0 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_ps(dptr0 + i, _mm256_div_ps(mnumer0, mdenom));
					_mm256_storeu_ps(dptr1 + i, _mm256_div_ps(mnumer1, mdenom));
					_mm256_storeu_ps(dptr2 + i, _mm256_div_ps(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<float*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<float>(j);
				}
				AutoBuffer<__m256> mnumer(schannels);
				AutoBuffer<float*> Iptr(schannels);
				AutoBuffer<__m256> mcenter(gchannel);

				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_ps();
					}

					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_ps(split_center[c].ptr<float>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<float>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(mcenter[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_ps(dptr[c] + i, _mm256_div_ps(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}

	void highDimensionalGaussianFilterCenterReplace32F_Dim(const cv::Mat src, const cv::Mat guide, const cv::Mat center, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border, HDGFSchedule schedule)
	{
		const int schannels = src.channels();
		const int gchannel = guide.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);
		vector<Mat> split_center;
		split(center, split_center);

		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			float* wsptr = weight_space.ptr<float>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = (float)std::exp(distance * coeff_space);
			}
		}
		const float coeff_range = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff_range = _mm256_set1_ps(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<float> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				float* dptr = dst.ptr<float>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer = _mm256_setzero_ps();
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr = srcborder.ptr<float>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							AutoBuffer<__m256> mcenter(guide.channels());
							for (int c = 0; c < gchannel; c++)
							{
								mcenter[c] = _mm256_loadu_ps(split_center[c].ptr<float>(j - r, i - r));
							}
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								__m256 sub = _mm256_sub_ps(mcenter[c], _mm256_loadu_ps(gptr + x + i));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_ps(dptr + i - r, _mm256_div_ps(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				float* dptr0 = split_dst[0].ptr<float>(j);
				float* dptr1 = split_dst[1].ptr<float>(j);
				float* dptr2 = split_dst[2].ptr<float>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					__m256 mnumer0 = _mm256_setzero_ps();
					__m256 mnumer1 = _mm256_setzero_ps();
					__m256 mnumer2 = _mm256_setzero_ps();

					AutoBuffer<__m256> mcenter(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_ps(split_center[c].ptr<float>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j, i);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j, i);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(mcenter[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							mnumer0 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_ps(dptr0 + i, _mm256_div_ps(mnumer0, mdenom));
					_mm256_storeu_ps(dptr1 + i, _mm256_div_ps(mnumer1, mdenom));
					_mm256_storeu_ps(dptr2 + i, _mm256_div_ps(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<float*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<float>(j);
				}
				AutoBuffer<__m256> mnumer(schannels);
				AutoBuffer<float*> Iptr(schannels);
				AutoBuffer<__m256> mcenter(gchannel);

				for (int i = 0; i < src.cols; i += 8)
				{
					__m256 mdenom = _mm256_setzero_ps();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_ps();
					}

					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_ps(split_center[c].ptr<float>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						float* wsptr = weight_space.ptr<float>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<float>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256 mdiff = _mm256_setzero_ps();
							for (int c = 0; c < gchannel; c++)
							{
								float* gptr = split_guideborder[c].ptr<float>(y + j, i);
								const __m256 sub = _mm256_sub_ps(mcenter[c], _mm256_loadu_ps(gptr + x));
								mdiff = _mm256_fmadd_ps(sub, sub, mdiff);
							}
							__m256 mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_ps(lut, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdiff)), 4);

							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(wsptr[x]), mgauss);
							mdenom = _mm256_add_ps(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_ps(mw, _mm256_loadu_ps(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_ps(dptr[c] + i, _mm256_div_ps(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}

	template<int gchannel, HDGFSchedule schedule>
	void highDimensionalGaussianFilterCenterReplace64F_(const cv::Mat src, const cv::Mat guide, const cv::Mat center, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border)
	{
		const int schannels = src.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);
		vector<Mat> split_center;
		split(center, split_center);
		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			double* wsptr = weight_space.ptr<double>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = std::exp(distance * coeff_space);
			}
		}
		const double coeff_range = 1.0 / (-2.0 * sigma_range * sigma_range);
		const __m256d mcoeff_range = _mm256_set1_pd(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<double> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				double* dptr = dst.ptr<double>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer = _mm256_setzero_pd();
					for (int y = -r; y <= r; y++)
					{
						double* wsptr = weight_space.ptr<double>(y + r);
						double* Iptr = srcborder.ptr<double>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							AutoBuffer<__m256d> mcenter(center.channels());
							for (int c = 0; c < gchannel; c++)
							{
								mcenter[c] = _mm256_loadu_pd(split_center[c].ptr<double>(j - r, i - r));
							}
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j);
								__m256d sub = _mm256_sub_pd(mcenter[c], _mm256_loadu_pd(gptr + x + i));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE) mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_pd(dptr + i - r, _mm256_div_pd(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				double* dptr0 = split_dst[0].ptr<double>(j);
				double* dptr1 = split_dst[1].ptr<double>(j);
				double* dptr2 = split_dst[2].ptr<double>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer0 = _mm256_setzero_pd();
					__m256d mnumer1 = _mm256_setzero_pd();
					__m256d mnumer2 = _mm256_setzero_pd();

					AutoBuffer<__m256d> mcenter(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_pd(split_center[c].ptr<double>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);
						double* Iptr0 = split_srcborder[0].ptr<double>(y + j, i);
						double* Iptr1 = split_srcborder[1].ptr<double>(y + j, i);
						double* Iptr2 = split_srcborder[2].ptr<double>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(mcenter[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer0 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_pd(dptr0 + i, _mm256_div_pd(mnumer0, mdenom));
					_mm256_storeu_pd(dptr1 + i, _mm256_div_pd(mnumer1, mdenom));
					_mm256_storeu_pd(dptr2 + i, _mm256_div_pd(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<double*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<double>(j);
				}
				AutoBuffer<__m256d> mnumer(schannels);
				AutoBuffer<double*> Iptr(schannels);
				AutoBuffer<__m256d> mcenter(gchannel);

				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_pd();
					}

					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_pd(split_center[c].ptr<double>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<double>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(mcenter[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if constexpr (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_pd(dptr[c] + i, _mm256_div_pd(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}

	void highDimensionalGaussianFilterCenterReplace64F_Dim(const cv::Mat src, const cv::Mat guide, const cv::Mat center, cv::Mat& dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border, HDGFSchedule schedule)
	{
		const int schannels = src.channels();
		const int gchannel = guide.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);
		vector<Mat> split_center;
		split(center, split_center);
		cv::Mat weight_space(ksize, src.type());
		const double coeff_space = 1.0 / (-2.0 * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			double* wsptr = weight_space.ptr<double>(j);
			for (int i = 0; i < D; i++)
			{
				const double distance = double((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = std::exp(distance * coeff_space);
			}
		}
		const double coeff_range = 1.0 / (-2.0 * sigma_range * sigma_range);
		const __m256d mcoeff_range = _mm256_set1_pd(coeff_range);

		const int lut_size = (int)ceil(get_l2norm_maxval(gchannel));
		cv::AutoBuffer<double> lut(lut_size);
		for (int i = 0; i < lut_size; i++)
		{
			lut[i] = exp(i * i * coeff_range);
		}

		if (src.channels() == 1)
		{
#pragma omp parallel for schedule(dynamic)
			for (int j = r; j < srcborder.rows - r; j++)
			{
				double* dptr = dst.ptr<double>(j - r);
				for (int i = r; i < srcborder.cols - r; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer = _mm256_setzero_pd();
					for (int y = -r; y <= r; y++)
					{
						double* wsptr = weight_space.ptr<double>(y + r);
						double* Iptr = srcborder.ptr<double>(y + j);
						for (int x = -r; x <= r; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							AutoBuffer<__m256d> mcenter(center.channels());
							for (int c = 0; c < gchannel; c++)
							{
								mcenter[c] = _mm256_loadu_pd(split_center[c].ptr<double>(j - r, i - r));
							}
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j);
								__m256d sub = _mm256_sub_pd(mcenter[c], _mm256_loadu_pd(gptr + x + i));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if (schedule == HDGFSchedule::COMPUTE) mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x + r]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr + i + x), mnumer);
						}
					}
					_mm256_store_pd(dptr + i - r, _mm256_div_pd(mnumer, mdenom));
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(3);
			split_dst[0].create(src.size(), src.depth());
			split_dst[1].create(src.size(), src.depth());
			split_dst[2].create(src.size(), src.depth());

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				double* dptr0 = split_dst[0].ptr<double>(j);
				double* dptr1 = split_dst[1].ptr<double>(j);
				double* dptr2 = split_dst[2].ptr<double>(j);
#if 0
				for (int i = r; i < srcborder.cols - r; i++)
				{
					float denom = 0.f;
					float numer0 = 0.f;
					float numer1 = 0.f;
					float numer2 = 0.f;
					for (int y = -r; y <= r; y++)
					{
						float* wsptr = weight_space.ptr<float>(y + r);
						float* Iptr0 = split_srcborder[0].ptr<float>(y + j + r);
						float* Iptr1 = split_srcborder[1].ptr<float>(y + j + r);
						float* Iptr2 = split_srcborder[2].ptr<float>(y + j + r);
						for (int x = -r; x <= r; x++)
						{
							float diff = 0.f;
							for (int c = 0; c < guide.channels(); c++)
							{
								float* srcptr = split_guideborder[c].ptr<float>(j);
								float* gptr = split_guideborder[c].ptr<float>(y + j);
								diff += (gptr[x + i] - srcptr[i]) * (gptr[x + i] - srcptr[i]);
							}
							float w = wsptr[x + r] * exp(diff * coeff_range);
							denom += w;
							numer0 += w * Iptr0[i + x];
							numer1 += w * Iptr1[i + x];
							numer2 += w * Iptr2[i + x];
						}
					}
					dptr0[i - r] = numer0 / denom;
					dptr1[i - r] = numer1 / denom;
					dptr2[i - r] = numer2 / denom;
				}
#else
				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					__m256d mnumer0 = _mm256_setzero_pd();
					__m256d mnumer1 = _mm256_setzero_pd();
					__m256d mnumer2 = _mm256_setzero_pd();

					AutoBuffer<__m256d> mcenter(gchannel);
					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_pd(split_center[c].ptr<double>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);
						double* Iptr0 = split_srcborder[0].ptr<double>(y + j, i);
						double* Iptr1 = split_srcborder[1].ptr<double>(y + j, i);
						double* Iptr2 = split_srcborder[2].ptr<double>(y + j, i);
						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(mcenter[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							mnumer0 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr0 + x), mnumer0);
							mnumer1 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr1 + x), mnumer1);
							mnumer2 = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr2 + x), mnumer2);
						}
					}
					_mm256_storeu_pd(dptr0 + i, _mm256_div_pd(mnumer0, mdenom));
					_mm256_storeu_pd(dptr1 + i, _mm256_div_pd(mnumer1, mdenom));
					_mm256_storeu_pd(dptr2 + i, _mm256_div_pd(mnumer2, mdenom));
				}
#endif
			}
			merge(split_dst, dst);
		}
		else // nchannel
		{
			vector<Mat> split_srcborder; split(srcborder, split_srcborder);
			vector<Mat> split_dst(schannels);
			for (int i = 0; i < schannels; i++)
			{
				split_dst[i].create(src.size(), src.depth());
			}

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				AutoBuffer<double*> dptr(schannels);
				for (int c = 0; c < schannels; c++)
				{
					dptr[c] = split_dst[c].ptr<double>(j);
				}
				AutoBuffer<__m256d> mnumer(schannels);
				AutoBuffer<double*> Iptr(schannels);
				AutoBuffer<__m256d> mcenter(gchannel);

				for (int i = 0; i < src.cols; i += 4)
				{
					__m256d mdenom = _mm256_setzero_pd();
					for (int c = 0; c < schannels; c++)
					{
						mnumer[c] = _mm256_setzero_pd();
					}

					for (int c = 0; c < gchannel; c++)
					{
						mcenter[c] = _mm256_loadu_pd(split_center[c].ptr<double>(j, i));
					}

					for (int y = 0; y < D; y++)
					{
						double* wsptr = weight_space.ptr<double>(y);

						for (int c = 0; c < schannels; c++)
						{
							Iptr[c] = split_srcborder[c].ptr<double>(y + j, i);
						}

						for (int x = 0; x < D; x++)
						{
							__m256d mdiff = _mm256_setzero_pd();
							for (int c = 0; c < gchannel; c++)
							{
								double* gptr = split_guideborder[c].ptr<double>(y + j, i);
								const __m256d sub = _mm256_sub_pd(mcenter[c], _mm256_loadu_pd(gptr + x));
								mdiff = _mm256_fmadd_pd(sub, sub, mdiff);
							}
							__m256d mgauss;
							if (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_pd(_mm256_mul_pd(mdiff, mcoeff_range));
							else mgauss = _mm256_i32gather_pd(lut, _mm256_cvtpd_epi32(_mm256_sqrt_pd(mdiff)), 8);

							__m256d mw = _mm256_mul_pd(_mm256_set1_pd(wsptr[x]), mgauss);
							mdenom = _mm256_add_pd(mdenom, mw);
							for (int c = 0; c < schannels; c++)
							{
								mnumer[c] = _mm256_fmadd_pd(mw, _mm256_loadu_pd(Iptr[c] + x), mnumer[c]);
							}
						}
					}
					for (int c = 0; c < schannels; c++)
					{
						_mm256_storeu_pd(dptr[c] + i, _mm256_div_pd(mnumer[c], mdenom));
					}
				}
			}
			merge(split_dst, dst);
		}
	}


	void highDimensionalGaussianFilter(InputArray src, InputArray guide, OutputArray dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border, HDGFSchedule schedule)
	{
		CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);
		CV_Assert(src.depth() == guide.depth());
		dst.create(src.size(), src.type());
		if (schedule == HDGFSchedule::COMPUTE)
		{
			if (src.depth() == CV_32F)
			{
				switch (guide.channels())
				{
				case 1:highDimensionalGaussianFilter32F_<1, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2:highDimensionalGaussianFilter32F_<2, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3:highDimensionalGaussianFilter32F_<3, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4:highDimensionalGaussianFilter32F_<4, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5:highDimensionalGaussianFilter32F_<5, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6:highDimensionalGaussianFilter32F_<6, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7:highDimensionalGaussianFilter32F_<7, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8:highDimensionalGaussianFilter32F_<8, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9:highDimensionalGaussianFilter32F_<9, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilter32F_<33, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilter32F_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::COMPUTE); break;
				}
			}
			else if (src.depth() == CV_64F)
			{
				switch (guide.channels())
				{
				case 1:highDimensionalGaussianFilter64F_<1, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2:highDimensionalGaussianFilter64F_<2, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3:highDimensionalGaussianFilter64F_<3, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4:highDimensionalGaussianFilter64F_<4, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5:highDimensionalGaussianFilter64F_<5, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6:highDimensionalGaussianFilter64F_<6, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7:highDimensionalGaussianFilter64F_<7, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8:highDimensionalGaussianFilter64F_<8, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9:highDimensionalGaussianFilter64F_<9, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilter64F_<33, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilter64F_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::COMPUTE); break;
				}
			}
			else
			{
				cout << "do not support this depth type " << src.depth() << endl;
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				switch (guide.channels())
				{
				case 1:highDimensionalGaussianFilter32F_<1, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2:highDimensionalGaussianFilter32F_<2, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3:highDimensionalGaussianFilter32F_<3, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4:highDimensionalGaussianFilter32F_<4, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5:highDimensionalGaussianFilter32F_<5, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6:highDimensionalGaussianFilter32F_<6, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7:highDimensionalGaussianFilter32F_<7, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8:highDimensionalGaussianFilter32F_<8, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9:highDimensionalGaussianFilter32F_<9, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilter32F_<33, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilter32F_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::LUT_SQRT); break;
				}
			}
			else if (src.depth() == CV_64F)
			{
				switch (guide.channels())
				{
				case 1:highDimensionalGaussianFilter64F_<1, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2:highDimensionalGaussianFilter64F_<2, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3:highDimensionalGaussianFilter64F_<3, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4:highDimensionalGaussianFilter64F_<4, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5:highDimensionalGaussianFilter64F_<5, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6:highDimensionalGaussianFilter64F_<6, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7:highDimensionalGaussianFilter64F_<7, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8:highDimensionalGaussianFilter64F_<8, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9:highDimensionalGaussianFilter64F_<9, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilter64F_<33, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilter32F_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::LUT_SQRT); break;
				}
			}
			else
			{
				cout << "do not support this depth type " << src.depth() << endl;
			}
		}
	}

	void highDimensionalGaussianFilter(InputArray src, InputArray guide, InputArray center, OutputArray dst, const cv::Size ksize, const double sigma_range, const double sigma_space, const int border, HDGFSchedule schedule)
	{
		CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);
		CV_Assert(src.depth() == guide.depth());
		dst.create(src.size(), src.type());
		if (schedule == HDGFSchedule::COMPUTE)
		{
			if (src.depth() == CV_32F)
			{
				switch (guide.channels())
				{
				case 1:highDimensionalGaussianFilterCenterReplace32F_<1, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2:highDimensionalGaussianFilterCenterReplace32F_<2, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3:highDimensionalGaussianFilterCenterReplace32F_<3, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4:highDimensionalGaussianFilterCenterReplace32F_<4, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5:highDimensionalGaussianFilterCenterReplace32F_<5, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6:highDimensionalGaussianFilterCenterReplace32F_<6, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7:highDimensionalGaussianFilterCenterReplace32F_<7, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8:highDimensionalGaussianFilterCenterReplace32F_<8, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9:highDimensionalGaussianFilterCenterReplace32F_<9, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilterCenterReplace32F_<33, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilter32F_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::COMPUTE); break;
				}
			}
			else if (src.depth() == CV_64F)
			{
				switch (guide.channels())
				{
				case 1:highDimensionalGaussianFilterCenterReplace64F_<1, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2:highDimensionalGaussianFilterCenterReplace64F_<2, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3:highDimensionalGaussianFilterCenterReplace64F_<3, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4:highDimensionalGaussianFilterCenterReplace64F_<4, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5:highDimensionalGaussianFilterCenterReplace64F_<5, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6:highDimensionalGaussianFilterCenterReplace64F_<6, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7:highDimensionalGaussianFilterCenterReplace64F_<7, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8:highDimensionalGaussianFilterCenterReplace64F_<8, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9:highDimensionalGaussianFilterCenterReplace64F_<9, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilterCenterReplace64F_<33, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilter64F_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::COMPUTE); break;
				}
			}
			else
			{
				cout << "do not support this depth type " << src.depth() << endl;
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				switch (guide.channels())
				{
				case 1: highDimensionalGaussianFilterCenterReplace32F_<1, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 2: highDimensionalGaussianFilterCenterReplace32F_<2, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 3: highDimensionalGaussianFilterCenterReplace32F_<3, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 4: highDimensionalGaussianFilterCenterReplace32F_<4, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 5: highDimensionalGaussianFilterCenterReplace32F_<5, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 6: highDimensionalGaussianFilterCenterReplace32F_<6, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 7: highDimensionalGaussianFilterCenterReplace32F_<7, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 8: highDimensionalGaussianFilterCenterReplace32F_<8, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 9: highDimensionalGaussianFilterCenterReplace32F_<9, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilterCenterReplace32F_<33, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilterCenterReplace32F_Dim(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::LUT_SQRT); break;
				}
			}
			else if (src.depth() == CV_64F)
			{
				switch (guide.channels())
				{
				case  1:highDimensionalGaussianFilterCenterReplace64F_<1, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  2:highDimensionalGaussianFilterCenterReplace64F_<2, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  3:highDimensionalGaussianFilterCenterReplace64F_<3, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  4:highDimensionalGaussianFilterCenterReplace64F_<4, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  5:highDimensionalGaussianFilterCenterReplace64F_<5, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  6:highDimensionalGaussianFilterCenterReplace64F_<6, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  7:highDimensionalGaussianFilterCenterReplace64F_<7, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  8:highDimensionalGaussianFilterCenterReplace64F_<8, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case  9:highDimensionalGaussianFilterCenterReplace64F_<9, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				case 33:highDimensionalGaussianFilterCenterReplace64F_<33, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
				default:
					highDimensionalGaussianFilterCenterReplace64F_Dim(src.getMat(), guide.getMat(), center.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::LUT_SQRT); break;
				}
			}
			else
			{
				cout << "do not support this depth type " << src.depth() << endl;
			}
		}
	}
#pragma endregion

#pragma region TileHDGF
	TileHDGF::TileHDGF(cv::Size div_) :
		thread_max(omp_get_max_threads()), div(div_)
	{
		;
	}

	TileHDGF::~TileHDGF()
	{
		;
	}


	void TileHDGF::nlmFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int patch_r, const int reduced_dim, const int pca_method, double truncateBoundary, const int borderType)
	{
		channels = src.channels();
		guide_channels = guide.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
		Size ksize = Size(d, d);
		if (div.area() == 1)
		{
			cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
			tileSize = src.size();
		}
		else
		{
			int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
			divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

			if (split_dst.size() != channels) split_dst.resize(channels);

			for (int c = 0; c < channels; c++)
			{
				split_dst[c].create(tileSize, CV_32FC1);
			}

			if (subImageInput.empty())
			{
				subImageInput.resize(thread_max);
				subImageGuide.resize(thread_max);
				subImageOutput.resize(thread_max);
				for (int n = 0; n < thread_max; n++)
				{
					subImageInput[n].resize(channels);
					subImageGuide[n].resize(guide_channels);
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}
			else
			{
				if (subImageGuide.empty()) subImageGuide.resize(thread_max);
				if (subImageGuide[0].size() != guide_channels)
				{
					for (int n = 0; n < thread_max; n++)
					{
						subImageGuide[n].resize(guide_channels);
					}
				}
			}


			if (src.channels() != 3)split(src, srcSplit);
			if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
			for (int n = 0; n < div.area(); n++)
			{
				const int thread_num = omp_get_thread_num();
				const cv::Point idx = cv::Point(n % div.width, n / div.width);


				if (src.channels() == 3)
				{
					cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < srcSplit.size(); c++)
					{
						cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				if (guide.channels() == 3)
				{
					cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < guideSplit.size(); c++)
					{
						cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				Mat s, g;
				merge(subImageInput[thread_num], s);
				std::vector<Mat> buff;
				cp::DRIM2COL(subImageGuide[thread_num], buff, patch_r, reduced_dim, borderType, pca_method);

				merge(buff, g);
				cp::highDimensionalGaussianFilter(s, g, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
			}
		}
	}

	double getError(Mat& rgb, Mat& gray)
	{
		double ret = 0.0;
		float* s = rgb.ptr<float>();
		float* g = gray.ptr<float>();
		for (int i = 0; i < rgb.size().area(); i++)
		{
			double dist
				= (s[3 * i + 0] - g[i]) * (s[3 * i + 0] - g[i])
				+ (s[3 * i + 1] - g[i]) * (s[3 * i + 1] - g[i])
				+ (s[3 * i + 2] - g[i]) * (s[3 * i + 2] - g[i]);
			ret += sqrt(dist);
		}
		return ret / rgb.size().area();
	}

	cp::RGBHistogram rgbh;
	void TileHDGF::cvtgrayFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int method, double truncateBoundary)
	{
		static int bb = 50; createTrackbar("b", "", &bb, 300);
		static int gg = 50; createTrackbar("g", "", &gg, 300);
		static int rr = 50; createTrackbar("r", "", &rr, 300);
		static int index = 0; createTrackbar("index", "", &index, div.area() - 1);
		channels = src.channels();
		guide_channels = guide.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int borderType = cv::BORDER_REFLECT;
		const int vecsize = sizeof(__m256) / sizeof(float);//8

		int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
		Size ksize = Size(d, d);
		if (div.area() == 1)
		{
			cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
			tileSize = src.size();
		}
		else
		{
			int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
			divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

			if (split_dst.size() != channels) split_dst.resize(channels);

			for (int c = 0; c < channels; c++)
			{
				split_dst[c].create(tileSize, CV_32FC1);
			}

			if (subImageInput.empty())
			{
				subImageInput.resize(thread_max);
				subImageGuide.resize(thread_max);
				subImageOutput.resize(thread_max);
				for (int n = 0; n < thread_max; n++)
				{
					subImageInput[n].resize(channels);
					subImageGuide[n].resize(guide_channels);
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}
			else
			{
				if (subImageGuide.empty()) subImageGuide.resize(thread_max);
				if (subImageGuide[0].size() != guide_channels)
				{
					for (int n = 0; n < thread_max; n++)
					{
						subImageGuide[n].resize(guide_channels);
					}
				}
			}

			std::vector<cv::Mat> srcSplit;
			std::vector<cv::Mat> guideSplit;
			if (src.channels() != 3)split(src, srcSplit);
			if (guide.channels() != 3)split(guide, guideSplit);

			//#pragma omp parallel for schedule(static)
			for (int n = 0; n < div.area(); n++)
			{
				const int thread_num = omp_get_thread_num();
				const cv::Point idx = cv::Point(n % div.width, n / div.width);

				if (src.channels() == 3)
				{
					cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < srcSplit.size(); c++)
					{
						cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				if (guide.channels() == 3)
				{
					cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < guideSplit.size(); c++)
					{
						cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				Mat s, mergedGuide, mergedGuideReduction;
				merge(subImageGuide[thread_num], mergedGuide);
				merge(subImageInput[thread_num], s);
				if (method == 0)
				{
					cp::cvtColorAverageGray(mergedGuide, mergedGuideReduction, true);
				}
				else
				{
					namedWindow("3DPlot");
					moveWindow("3DPlot", 800, 200);
					Scalar a = Scalar(bb, gg, rr, 0);//mean(g);
					Mat evec, eval, mean, temp;
					cp::cvtColorPCA(mergedGuide, temp, 1, evec, eval, mean);
					Scalar ev0 = Scalar(evec.at<double>(0, 0), evec.at<double>(0, 1), evec.at<double>(0, 2), 0);
					Scalar ev1 = Scalar(evec.at<double>(1, 0), evec.at<double>(1, 1), evec.at<double>(1, 2), 0);
					a = eval.at<double>(0) * ev0 + eval.at<double>(1) * ev1;
					Scalar b = Scalar(255.0, 255.0, 255.0, 255.0);
					Scalar m, v;
					meanStdDev(mergedGuide, m, v);

					//double al = 0.95;
					//a = (al * a + (1.0 - al) * b);


					Mat mat(1, 3, CV_32F);
					double norm = 1.f / sqrt(a.val[0] * a.val[0] + a.val[1] * a.val[1] + a.val[2] * a.val[2]);
					mat.at<float>(0) = float(a.val[0] * norm);
					mat.at<float>(1) = float(a.val[1] * norm);
					mat.at<float>(2) = float(a.val[2] * norm);
					transform(mergedGuide, mergedGuideReduction, mat);
					if (n == index)
					{

						Mat evec, eval, mean, temp;
						cp::cvtColorPCA(mergedGuide, temp, 1, evec, eval, mean);
						std::cout << evec << std::endl;
						std::cout << eval << std::endl;
						//std::system("cls");
						//std::cout << "index:" << idx.y * div.width + idx.x << std::endl;
						//std::cout << getError(g, gr) << std::endl;
						//std::cout << mat << std::endl;
						//std::cout << "cave" << m << std::endl;
						//std::cout << "cvar" << v << std::endl;
						//std::cout << "xxxx" << v * sqrt(3) / (v.val[0] + v.val[1] + v.val[2]) << std::endl;
						//meanStdDev(mergedGuideReduction, m, v);
						//std::cout << "gave" << m << std::endl;
						//std::cout << "gvar" << m << std::endl << std::endl;
						//rgbh.push_back(g);

#pragma omp critical
						{
							//rgbh.push_back_line(0, 0, 0, mat.at<float>(0) * 255*sqrt(3.0), mat.at<float>(1) * 255 * sqrt(3.0), mat.at<float>(2) * 255 * sqrt(3.0));
							Point3f s = Point3f((float)mean.at<double>(0), (float)mean.at<double>(1), (float)mean.at<double>(2));
							Point3f d = Point3f(mat.at<float>(0), mat.at<float>(1), mat.at<float>(2)) * 100.f * sqrt(3.f);
							rgbh.push_back_line(s - d, s + d);
							cp::imshowScale("a", mergedGuide);
							rgbh.plot(mergedGuide, false, "3DPlot");
							rgbh.clear();
						}
					}

				}
				cp::highDimensionalGaussianFilter(s, mergedGuideReduction, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
			}
		}
	}

	void TileHDGF::pcaFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int reduced_dim, double truncateBoundary)
	{
		channels = src.channels();
		guide_channels = guide.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int borderType = cv::BORDER_REFLECT;
		const int vecsize = sizeof(__m256) / sizeof(float);//8

		int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
		Size ksize = Size(d, d);
		if (div.area() == 1)
		{
			cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
			tileSize = src.size();
		}
		else
		{
			int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
			divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

			if (split_dst.size() != channels) split_dst.resize(channels);

			for (int c = 0; c < channels; c++)
			{
				split_dst[c].create(tileSize, CV_32FC1);
			}

			if (subImageInput.empty())
			{
				subImageInput.resize(thread_max);
				subImageGuide.resize(thread_max);
				subImageOutput.resize(thread_max);
				for (int n = 0; n < thread_max; n++)
				{
					subImageInput[n].resize(channels);
					subImageGuide[n].resize(guide_channels);
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}
			else
			{
				if (subImageGuide.empty()) subImageGuide.resize(thread_max);
				if (subImageGuide[0].size() != guide_channels)
				{
					for (int n = 0; n < thread_max; n++)
					{
						subImageGuide[n].resize(guide_channels);
					}
				}
			}

			std::vector<cv::Mat> srcSplit;
			std::vector<cv::Mat> guideSplit;
			if (src.channels() != 3)split(src, srcSplit);
			if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
			for (int n = 0; n < div.area(); n++)
			{
				const int thread_num = omp_get_thread_num();
				const cv::Point idx = cv::Point(n % div.width, n / div.width);


				if (src.channels() == 3)
				{
					cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < srcSplit.size(); c++)
					{
						cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				if (guide.channels() == 3)
				{
					cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < guideSplit.size(); c++)
					{
						cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				Mat s, g;
				std::vector<Mat> buff;
				cp::cvtColorPCA(subImageGuide[thread_num], buff, reduced_dim);
				merge(subImageInput[thread_num], s);
				merge(buff, g);
				cp::highDimensionalGaussianFilter(s, g, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
			}
		}
	}

	void TileHDGF::filter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, double truncateBoundary)
	{
		channels = src.channels();
		guide_channels = guide.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int borderType = cv::BORDER_REFLECT;
		const int vecsize = sizeof(__m256) / sizeof(float);//8

		int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
		Size ksize = Size(d, d);
		if (div.area() == 1)
		{
			cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
			tileSize = src.size();
		}
		else
		{
			int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
			divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

			if (split_dst.size() != channels) split_dst.resize(channels);

			for (int c = 0; c < channels; c++)
			{
				split_dst[c].create(tileSize, CV_32FC1);
			}

			if (subImageInput.empty())
			{
				subImageInput.resize(thread_max);
				subImageGuide.resize(thread_max);
				subImageOutput.resize(thread_max);
				for (int n = 0; n < thread_max; n++)
				{
					subImageInput[n].resize(channels);
					subImageGuide[n].resize(guide_channels);
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}
			else
			{
				if (subImageGuide.empty()) subImageGuide.resize(thread_max);
				if (subImageGuide[0].size() != guide_channels)
				{
					for (int n = 0; n < thread_max; n++)
					{
						subImageGuide[n].resize(guide_channels);
					}
				}
			}

			std::vector<cv::Mat> srcSplit;
			std::vector<cv::Mat> guideSplit;
			if (src.channels() != 3)split(src, srcSplit);
			if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
			for (int n = 0; n < div.area(); n++)
			{
				const int thread_num = omp_get_thread_num();
				const cv::Point idx = cv::Point(n % div.width, n / div.width);


				if (src.channels() == 3)
				{
					cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < srcSplit.size(); c++)
					{
						cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				if (guide.channels() == 3)
				{
					cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < guideSplit.size(); c++)
					{
						cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				Mat s, g;
				merge(subImageInput[thread_num], s);
				merge(subImageGuide[thread_num], g);
				cp::highDimensionalGaussianFilter(s, g, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
			}
		}
	}


	cv::Size TileHDGF::getTileSize()
	{
		return tileSize;
	}

	void TileHDGF::getTileInfo()
	{
		print_debug(div);
		print_debug(divImageSize);
		print_debug(tileSize);
		int borderLength = (tileSize.width - divImageSize.width) / 2;
		print_debug(borderLength);

	}
#pragma endregion
}