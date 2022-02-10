#include "highDimensionalGaussianFilter.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	inline double get_l2norm_maxval(const int channel)
	{
		return sqrt(1024 * 1024 * channel);
	}

	template<int gchannel, HDGFSchedule schedule>
	void highDimensionalGaussianFilter_(const cv::Mat src, const cv::Mat guide, cv::Mat& dst, const cv::Size ksize, const float sigma_range, const float sigma_space, const int border)
	{
		const int schannels = src.channels();
		//const int schannels = 33;
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);

		cv::Mat weight_space(ksize, src.type());
		const float coeff_space = 1.f / (-2.f * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			float* wsptr = weight_space.ptr<float>(j);
			for (int i = 0; i < D; i++)
			{
				const float distance = float((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = std::exp(distance * coeff_space);
			}
		}
		const float coeff_range = 1.f / (-2.f * sigma_range * sigma_range);
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

	void highDimensionalGaussianFilter_Dim(const cv::Mat src, const cv::Mat guide, cv::Mat& dst, const cv::Size ksize, const float sigma_range, const float sigma_space, const int border, HDGFSchedule schedule)
	{
		const int schannels = src.channels();
		const int gchannel = guide.channels();
		const int r = ksize.height / 2;
		Mat srcborder; copyMakeBorder(src, srcborder, r, r, r, r, border);
		Mat guideborder; copyMakeBorder(guide, guideborder, r, r, r, r, border);
		vector<Mat> split_guideborder;
		split(guideborder, split_guideborder);

		cv::Mat weight_space(ksize, src.type());
		const float coeff_space = 1.f / (-2.f * sigma_space * sigma_space);
		const int D = 2 * r + 1;
		for (int j = 0; j < D; j++)
		{
			float* wsptr = weight_space.ptr<float>(j);
			for (int i = 0; i < D; i++)
			{
				const float distance = float((i - r) * (i - r) + (j - r) * (j - r));
				wsptr[i] = std::exp(distance * coeff_space);
			}
		}
		const float coeff_range = 1.f / (-2.f * sigma_range * sigma_range);
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
							if  (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
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
							if  (schedule == HDGFSchedule::COMPUTE)mgauss = _mm256_exp_ps(_mm256_mul_ps(mdiff, mcoeff_range));
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

	void highDimensionalGaussianFilter(InputArray src, InputArray guide, OutputArray dst, const cv::Size ksize, const float sigma_range, const float sigma_space, const int border, HDGFSchedule schedule)
	{
		dst.create(src.size(), src.type());
		if (schedule == HDGFSchedule::COMPUTE)
		{
			switch (guide.channels())
			{
			case 1:highDimensionalGaussianFilter_<1, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 2:highDimensionalGaussianFilter_<2, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 3:highDimensionalGaussianFilter_<3, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 4:highDimensionalGaussianFilter_<4, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 5:highDimensionalGaussianFilter_<5, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 6:highDimensionalGaussianFilter_<6, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 7:highDimensionalGaussianFilter_<7, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 8:highDimensionalGaussianFilter_<8, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 9:highDimensionalGaussianFilter_<9, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 33:highDimensionalGaussianFilter_<33, HDGFSchedule::COMPUTE>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			default:
				highDimensionalGaussianFilter_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::COMPUTE); break;
			}
		}
		else
		{
			switch (guide.channels())
			{
			case 1:highDimensionalGaussianFilter_<1, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 2:highDimensionalGaussianFilter_<2, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 3:highDimensionalGaussianFilter_<3, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 4:highDimensionalGaussianFilter_<4, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 5:highDimensionalGaussianFilter_<5, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 6:highDimensionalGaussianFilter_<6, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 7:highDimensionalGaussianFilter_<7, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 8:highDimensionalGaussianFilter_<8, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 9:highDimensionalGaussianFilter_<9, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			case 33:highDimensionalGaussianFilter_<33, HDGFSchedule::LUT_SQRT>(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border); break;
			default:
				highDimensionalGaussianFilter_Dim(src.getMat(), guide.getMat(), dst.getMat(), ksize, sigma_range, sigma_space, border, HDGFSchedule::LUT_SQRT); break;
			}
		}
	}
}