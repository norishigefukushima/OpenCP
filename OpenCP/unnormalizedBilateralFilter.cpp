#include "unnormalizedBilateralFilter.hpp"
#include "inlineSIMDFunctions.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	static inline double getPyramidSigma(double sigma, double level)
	{
		double ret = 0.0;
		for (int l = 1; l <= level; l++)
		{
			double v = pow(2, l - 1);
			ret = ret + (sigma * v) * (sigma * v);
		}
		return sqrt(ret);
	}

	static inline float getGaussianRangeWeight(float v, float sigma_range, bool isEnhance, float boost)
	{
		float ret;
		//ret = (float)exp(v * v / (-2.0 * sigma_range * sigma_range));
		int n = 2;
		//float k = 2.f * detail_param;
		float k = 1.f * boost;
		//ret = (float)exp(pow(abs(v), n) / (-n * pow(sigma_range, n)));
		ret = (float)k * exp(pow(abs(v), n) / (-n * pow(sigma_range, n)));
		//ret = 2*cos(std::min(1.5f	, v / sigma_range) * CV_PI);
		//float alpha = 0.5;
		/*if (v < sigma_range)
			ret = (v + sigma_range * pow(abs(v) / sigma_range, alpha))/(sigma_range*5);
		else ret = 0;
		*/
		if (isEnhance) return -ret;
		else return ret;
	}

	void unnormalizedBilateralFilterGray(const Mat& src, Mat& dest, const int r, const float sigma_range, const float sigma_space, bool isEnhance, int borderType)
	{
		Mat srcf;
		if (src.depth() == CV_32F)srcf = src;
		else src.convertTo(srcf, CV_32F);

		Mat destf(src.size(), CV_32F);

		Mat im;
		copyMakeBorder(srcf, im, r, r, r, r, borderType);

		const int d = (2 * r + 1) * (2 * r + 1);
		vector<float> rangeTable(256);
		float* rweight = &rangeTable[0];
		vector<float> space(d);
		vector<int> offset(d);

		for (int i = 0; i < 256; i++)
		{
			rangeTable[i] = getGaussianRangeWeight(float(i), sigma_range, isEnhance, 1);
		}

		const double coeff_s = -1.0 / (2.0 * sigma_space * sigma_space);
		float wsum = 0.f;
		for (int j = -r, idx = 0; j <= r; j++)
		{
			for (int i = -r; i <= r; i++)
			{
				double dis = double(i * i + j * j);
				offset[idx] = im.cols * j + i;
				float v = (float)exp(dis * coeff_s);
				wsum += v;
				space[idx] = v;
				idx++;
			}
		}

		for (int k = 0; k < d; k++)
		{
			space[k] /= wsum;
		}

		bool isOpt = true;
		int unroll = 4;
		if (isOpt)
		{
			if (unroll == 1)
			{

#pragma omp parallel for schedule(dynamic)
				for (int j = 0; j < src.rows; j++)
				{
					const float* s = im.ptr<float>(j + r) + r;
					float* dst = destf.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 8)
					{
						const float* si = s + i;
						const __m256 mt = _mm256_loadu_ps(si);
						__m256 msum = mt;
						for (int k = 0; k < d; k++)
						{
							__m256 mv = _mm256_sub_ps(_mm256_loadu_ps(si + offset[k]), mt);
							__m256 mw = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(mv)), 4));
							msum = _mm256_fmadd_ps(mw, mv, msum);
						}
						_mm256_storeu_ps(dst + i, msum);
					}
				}
			}
			else
			{

#pragma omp parallel for schedule(dynamic)
				for (int j = 0; j < src.rows; j++)
				{
					const float* s = im.ptr<float>(j + r) + r;
					float* dst = destf.ptr<float>(j);
					for (int i = 0; i < src.cols; i += 32)
					{
						const float* si = s + i;
						const __m256 mt0 = _mm256_lddqu_ps(si);
						const __m256 mt1 = _mm256_lddqu_ps(si + 8);
						const __m256 mt2 = _mm256_lddqu_ps(si + 16);
						const __m256 mt3 = _mm256_lddqu_ps(si + 24);
						__m256 msum0 = mt0;
						__m256 msum1 = mt1;
						__m256 msum2 = mt2;
						__m256 msum3 = mt3;
						for (int k = 0; k < d; k++)
						{
							__m256 mv0 = _mm256_sub_ps(_mm256_lddqu_ps(si + offset[k] + 0), mt0);
							__m256 mv1 = _mm256_sub_ps(_mm256_lddqu_ps(si + offset[k] + 8), mt1);
							__m256 mv2 = _mm256_sub_ps(_mm256_lddqu_ps(si + offset[k] + 16), mt2);
							__m256 mv3 = _mm256_sub_ps(_mm256_lddqu_ps(si + offset[k] + 24), mt3);
							__m256 mw0 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(mv0)), 4));
							__m256 mw1 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(mv1)), 4));
							__m256 mw2 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(mv2)), 4));
							__m256 mw3 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(mv3)), 4));
							msum0 = _mm256_fmadd_ps(mw0, mv0, msum0);
							msum1 = _mm256_fmadd_ps(mw1, mv1, msum1);
							msum2 = _mm256_fmadd_ps(mw2, mv2, msum2);
							msum3 = _mm256_fmadd_ps(mw3, mv3, msum3);
						}
						_mm256_storeu_ps(dst + i + 0, msum0);
						_mm256_storeu_ps(dst + i + 8, msum1);
						_mm256_storeu_ps(dst + i + 16, msum2);
						_mm256_storeu_ps(dst + i + 24, msum3);
					}
				}
			}
		}
		else
		{

#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < src.rows; j++)
			{
				float* s = im.ptr<float>(j + r) + r;
				float* dst = destf.ptr<float>(j);
				for (int i = 0; i < src.cols; i++)
				{
					const float t = s[i];
					float sum = t;
					for (int k = 0; k < d; k++)
					{
						const float v = s[offset[k] + i];
						sum += space[k] * rangeTable[(int)abs(v - t)] * (v - t);
					}
					dst[i] = sum;
				}
			}
		}

		destf.convertTo(dest, src.type());
	}

	void buildGaussianStack(Mat& src, vector<Mat>& GaussianStack, const float sigma_s, const int level)
	{
		GaussianStack.resize(level + 1);

		src.convertTo(GaussianStack[0], CV_32F);

#pragma omp parallel for schedule(dynamic)
		for (int i = 1; i <= level; i++)
		{
			//const float sigma_l = sigma_s * i;
			const float sigma_l = (float)getPyramidSigma(sigma_s, i);
			const int r = (int)ceil(sigma_l * 3.f);
			const Size ksize(2 * r + 1, 2 * r + 1);
#ifdef USE_SLIDING_DCT
			gf::SpatialFilterSlidingDCT5_AVX_32F sf(gf::DCT_COEFFICIENTS::FULL_SEARCH_NOOPT);
			sf.filter(GaussianStack[0], GaussianStack[i], sigma_l, 2);
#else
			GaussianBlur(GaussianStack[0], GaussianStack[i], ksize, sigma_l);
#endif
		}
	}

	void unnormalizedBilateralFilterMultiGray(Mat& src, Mat& dest, const int r, const float sigma_range, const float sigma_space, int level, const bool isEnhance, const int borderType)
	{
		Mat srcf;
		if (src.depth() == CV_8U)src.convertTo(srcf, CV_32F);
		else srcf = src;

		Mat destf(src.size(), CV_32F);

		const int r_max = r * level;
		Mat im;
		copyMakeBorder(srcf, im, r_max, r_max, r_max, r_max, borderType);

		const int d_max = (2 * r_max + 1) * (2 * r_max + 1);
		vector<vector<float>> wg(level + 1);
		vector<vector<float>> ws(level + 1);
		for (int i = 0; i <= level; i++)
		{
			wg[i].resize(d_max);
			ws[i].resize(d_max);
		}
		vector<int> offset(d_max);

		vector<float> wr(256);
		for (int i = 0; i < 256; i++)
		{
			wr[i] = getGaussianRangeWeight(float(i), sigma_range, isEnhance, 1);
		}

		int l = 0;
		for (int j = -r_max, idx = 0; j <= r_max; j++)
		{
			for (int i = -r_max; i <= r_max; i++)
			{
				double dis = double(i * i + j * j);
				offset[idx] = im.cols * j + i;
				wg[l][idx] = 0.f;
				idx++;
			}
		}
		for (int l = 1; l <= level; l++)
		{
			float wsum = 0.f;
			const double coeff_s = -1.0 / (2.0 * sigma_space * l * sigma_space * l);
			for (int j = -r_max, idx = 0; j <= r_max; j++)
			{
				for (int i = -r_max; i <= r_max; i++)
				{
					double dis = double(i * i + j * j);
					float v = (float)exp(dis * coeff_s);
					wsum += v;
					wg[l][idx] = v;
					idx++;
				}
			}
			//normalize
			for (int k = 0; k < d_max; k++)
			{
				wg[l][k] /= wsum;
			}
		}
		//DoG	
		for (int k = 0; k < d_max; k++)
		{
			ws[l][k] = wg[l][k];
		}
		for (int l = 1; l <= level; l++)
		{
			for (int k = 0; k < d_max; k++)
			{
				ws[l][k] = wg[l][k] - wg[l - 1][k];
			}
		}

		vector<Mat> Gauss;
		//must impkement
		buildGaussianStack(srcf, Gauss, sigma_space, level);

		vector<float*> wsv(level + 1);

#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < src.rows; j++)
		{
			float* s = im.ptr<float>(j + r_max) + r_max;
			float* dst = destf.ptr<float>(j);
			vector<float> ref(level + 1);
			for (int i = 0; i < src.cols; i++)
			{
				float* si = s + i;
				const float t = si[0];
				float sum = t;//level 0			
				for (int l = 2; l <= level; l++)
				{
					ref[l] = Gauss[l - 1].at<float>(j, i);
				}
				for (int k = 0; k < d_max; k++)
				{
					const float v = si[offset[k]];

					sum += ws[1][k] * wr[(int)abs(v - t)] * (v - t);//level1
					for (int l = 2; l <= level; l++)
					{
						sum += ws[l][k] * wr[(int)abs(v - ref[l])] * (v - ref[l]);
					}
				}
				dst[i] = sum;
			}
		}

		destf.convertTo(dest, src.type());
	}

	void unnormalizedBilateralFilter(Mat& src, Mat& dest, const int r, const float sigma_range, const float sigma_space, const bool isEnhance, const int borderType)
	{
		if (src.channels() == 1)
		{
			unnormalizedBilateralFilterGray(src, dest, r, sigma_range, sigma_space, isEnhance, borderType);
		}
		else
		{
			vector<Mat> vsrc;
			vector<Mat> vdst(3);
			split(src, vsrc);
			unnormalizedBilateralFilterGray(vsrc[0], vdst[0], r, sigma_range, sigma_space, isEnhance, borderType);
			unnormalizedBilateralFilterGray(vsrc[1], vdst[1], r, sigma_range, sigma_space, isEnhance, borderType);
			unnormalizedBilateralFilterGray(vsrc[2], vdst[2], r, sigma_range, sigma_space, isEnhance, borderType);
			merge(vdst, dest);
		}
	}

	void unnormalizedBilateralFilterMulti(Mat& src, Mat& dest, const int r, const float sigma_range, const float sigma_space, const int level, bool isEnhance, const int borderType)
	{
		if (src.channels() == 1)
		{
			unnormalizedBilateralFilterMultiGray(src, dest, r, sigma_range, sigma_space, level, isEnhance, borderType);
		}
		else
		{
			vector<Mat> vsrc;
			vector<Mat> vdst(3);
			split(src, vsrc);
			unnormalizedBilateralFilterMultiGray(vsrc[0], vdst[0], r, sigma_range, sigma_space, level, isEnhance, borderType);
			unnormalizedBilateralFilterMultiGray(vsrc[1], vdst[1], r, sigma_range, sigma_space, level, isEnhance, borderType);
			unnormalizedBilateralFilterMultiGray(vsrc[2], vdst[2], r, sigma_range, sigma_space, level, isEnhance, borderType);
			merge(vdst, dest);
		}
	}
	}
