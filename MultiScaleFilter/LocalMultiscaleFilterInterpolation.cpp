#include "multiscalefilter/MultiScaleFilter.hpp"
using namespace cv;
using namespace std;
namespace cp
{
#pragma region FastLLFReference

	float FastLLFReference::getTau(const int k)
	{
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
	}

	void FastLLFReference::blendLaplacianLinear(const vector<vector<Mat>>& LaplacianPyramid, vector<Mat>& GaussianPyramid, vector<Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		destPyramid.resize(level);
		AutoBuffer<const float*> lptr(order);
		for (int l = 0; l < level - 1; l++)
		{
			destPyramid[l].create(GaussianPyramid[l].size(), CV_32F);
			float* g = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();
			for (int k = 0; k < order; k++)
			{
				lptr[k] = LaplacianPyramid[k][l].ptr<float>();
			}

			for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
			{
				float alpha;
				int high, low;
				getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
				d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
			}
		}
	}

	void FastLLFReference::pyramid(const Mat& src, Mat& dest)
	{
		pyramidComputeMethod = PyramidComputeMethod::Full;

		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) build Gaussian Pyramid
		buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);

		//(2) build Laplacian Pyramids
		LaplacianPyramidOrder.resize(order);
		for (int n = 0; n < order; n++)
		{
			LaplacianPyramidOrder[n].resize(level + 1);

			//(2)-1 Remap Input Image
			if (adaptiveMethod == AdaptiveMethod::FIX) remap(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n), sigma_range, boost);
			else remapAdaptive(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n), adaptiveSigmaMap[0], adaptiveBoostMap[0]);

			//(2)-2 Build Remapped Laplacian Pyramids
			buildLaplacianPyramid(LaplacianPyramidOrder[n][0], LaplacianPyramidOrder[n], level, sigma_space);
		}

		//(3) interpolate Laplacian Pyramid from Remapped Laplacian Pyramids
		blendLaplacianLinear(LaplacianPyramidOrder, GaussianPyramid, LaplacianPyramid, order);
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) collapse Laplacian Pyramid
		collapseLaplacianPyramid(LaplacianPyramid, dest);
	}

	void FastLLFReference::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod, const int interpolationMethod)
	{
		allocSpaceWeight(sigma_space);

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		this->order = order;
		body(src, dest);

		freeSpaceWeight();
	}

#pragma endregion

	void LocalMultiScaleFilterInterpolation::initRangeTableInteger(const float sigma, const float boost)
	{
		const int intensityRange2 = get_simd_ceil((int)intensityRange, order - 1);
		const int tableSize = intensityRange2 + 1;
		integerSampleTable = (float*)_mm_malloc(sizeof(float) * tableSize, AVX_ALIGN);
		int rem = intensityRange2 - (int)intensityRange;

		intensityRange = float(intensityRange2);
		intensityMax += (float)rem;
		for (int i = 0; i < tableSize; i++)
		{
			integerSampleTable[i] = getGaussianRangeWeight(float(i), sigma_range, boost);
		}
	}

	float LocalMultiScaleFilterInterpolation::getTau(const int k)
	{
#if 1
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
#else
		const float intensityRange = float(intensityMax - intensityMin);
		const float delta = intensityRange / (order - 2);
		return float(k * delta + intensityMin - delta);
#endif
	}

#pragma region pyramid
	template<bool is_use_table, int D>
	void LocalMultiScaleFilterInterpolation::remapGaussDownIgnoreBoundary(const Mat& src, Mat& remapIm, Mat& dest, const float g, const float sigma_range, const float boost)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);
		remapIm.create(size, CV_32F);

		//const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

#pragma region remap top
		const __m256 mg = _mm256_set1_ps(g);
		const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff = _mm256_set1_ps(coeff);
		const __m256 mdetail = _mm256_set1_ps(boost);

		//splat
		{
			const float* sptr = src.ptr<float>();
			float* d = remapIm.ptr<float>();
			const int size = width * (D - 1);
			const int REMAPSIZE32 = get_simd_floor(size, 32);
			const int REMAPSIZE8 = get_simd_ceil(size, 8);
			if constexpr (is_use_table)
			{
				//float* rt = &rangeTable[0];
				float* rt = integerSampleTable;
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
			}
			else
			{
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
			}
		}

#pragma endregion

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		//memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = remapIm.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);

		const int HEND32 = get_simd_floor(hend, 32);
		const int HEND = get_simd_floor(hend, 8);
		const __m128i maskhend = get_storemask1(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//remap line
			{
				const float* sptr = src.ptr<float>(j + D - 1);
				float* d = remapIm.ptr<float>(j + D - 1);
				const int size = 2 * width;
				const int REMAPSIZE32 = get_simd_floor(size, 32);
				const int REMAPSIZE8 = get_simd_ceil(size, 8);
				if constexpr (is_use_table)
				{
					//float* rt = &rangeTable[0];
					float* rt = integerSampleTable;
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
				}
				else
				{
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
				}
			}
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND32; i += 32)
			{
				float* lb0 = linebuff + i;
				float* lb1 = linebuff + i + 8;
				float* lb2 = linebuff + i + 16;
				float* lb3 = linebuff + i + 24;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				__m256 sum1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb1++));
				__m256 sum2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb2++));
				__m256 sum3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb3++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
					sum1 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb1++), sum1);
					sum2 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb2++), sum2);
					sum3 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb3++), sum3);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));

				sum1 = _mm256_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2, 0, 2, 0));
				sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));

				sum2 = _mm256_shuffle_ps(sum2, sum2, _MM_SHUFFLE(2, 0, 2, 0));
				sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));

				sum3 = _mm256_shuffle_ps(sum3, sum3, _MM_SHUFFLE(2, 0, 2, 0));
				sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
			}
			for (int i = HEND32; i < HEND; i += 8)
			{
				float* lb0 = linebuff + i;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));
			}
#ifdef MASKSTORE
			//last
			{
				float* lb0 = linebuff + HEND;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_maskstore_ps(dptr + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum0));
			}
#else
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
#endif
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}

	template<bool is_use_table>
	void LocalMultiScaleFilterInterpolation::remapGaussDownIgnoreBoundary(const Mat& src, Mat& remapIm, Mat& dest, const float g, const float sigma_range, const float boost)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);
		remapIm.create(size, CV_32F);

		const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

#pragma region remap top
		const __m256 mg = _mm256_set1_ps(g);
		const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
		const __m256 mcoeff = _mm256_set1_ps(coeff);
		const __m256 mdetail = _mm256_set1_ps(boost);

		//splat
		{
			const float* sptr = src.ptr<float>();
			float* d = remapIm.ptr<float>();
			const int size = width * (D - 1);
			const int REMAPSIZE32 = get_simd_floor(size, 32);
			const int REMAPSIZE8 = get_simd_ceil(size, 8);
			if constexpr (is_use_table)
			{
				//float* rt = &rangeTable[0];
				float* rt = integerSampleTable;
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
				}
			}
			else
			{
				for (int i = 0; i < REMAPSIZE32; i += 32)
				{
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 8);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 16);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

					ms = _mm256_loadu_ps(sptr + i + 24);
					subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
				for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
				{
					const __m256 ms = _mm256_loadu_ps(sptr + i);
					const __m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
			}
		}

#pragma endregion

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		//memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = remapIm.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);

		const int HEND32 = get_simd_floor(hend, 32);
		const int HEND = get_simd_floor(hend, 8);
		const __m128i maskhend = get_storemask1(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//remap line
			{
				const float* sptr = src.ptr<float>(j + D - 1);
				float* d = remapIm.ptr<float>(j + D - 1);
				const int size = 2 * width;
				const int REMAPSIZE32 = get_simd_floor(size, 32);
				const int REMAPSIZE8 = get_simd_ceil(size, 8);
				if constexpr (is_use_table)
				{
					//float* rt = &rangeTable[0];
					float* rt = integerSampleTable;
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_i32gather_ps(rt, _mm256_cvtps_epi32(_mm256_abs_ps(subsg)), sizeof(float)), ms));
					}
				}
				else
				{
					for (int i = 0; i < REMAPSIZE32; i += 32)
					{
						__m256 ms = _mm256_loadu_ps(sptr + i);
						__m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 8);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 16);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

						ms = _mm256_loadu_ps(sptr + i + 24);
						subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
					for (int i = REMAPSIZE32; i < REMAPSIZE8; i += 8)
					{
						const __m256 ms = _mm256_loadu_ps(sptr + i);
						const __m256 subsg = _mm256_sub_ps(ms, mg);
						_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
					}
				}
			}
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND32; i += 32)
			{
				float* lb0 = linebuff + i;
				float* lb1 = linebuff + i + 8;
				float* lb2 = linebuff + i + 16;
				float* lb3 = linebuff + i + 24;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				__m256 sum1 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb1++));
				__m256 sum2 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb2++));
				__m256 sum3 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb3++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
					sum1 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb1++), sum1);
					sum2 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb2++), sum2);
					sum3 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb3++), sum3);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));

				sum1 = _mm256_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2, 0, 2, 0));
				sum1 = _mm256_permute4x64_ps(sum1, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 8) >> 1), _mm256_castps256_ps128(sum1));

				sum2 = _mm256_shuffle_ps(sum2, sum2, _MM_SHUFFLE(2, 0, 2, 0));
				sum2 = _mm256_permute4x64_ps(sum2, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 16) >> 1), _mm256_castps256_ps128(sum2));

				sum3 = _mm256_shuffle_ps(sum3, sum3, _MM_SHUFFLE(2, 0, 2, 0));
				sum3 = _mm256_permute4x64_ps(sum3, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + ((i + 24) >> 1), _mm256_castps256_ps128(sum3));
			}
			for (int i = HEND32; i < HEND; i += 8)
			{
				float* lb0 = linebuff + i;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum0));
			}
#ifdef MASKSTORE
			//last
			{
				float* lb0 = linebuff + HEND;
				__m256 sum0 = _mm256_mul_ps(W[0], _mm256_loadu_ps(lb0++));
				for (int k = 1; k < D; k++)
				{
					sum0 = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(lb0++), sum0);
				}
				sum0 = _mm256_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 0, 2, 0));
				sum0 = _mm256_permute4x64_ps(sum0, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_maskstore_ps(dptr + (HEND >> 1), maskhend, _mm256_castps256_ps128(sum0));
			}
#else
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
#endif
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}


	void LocalMultiScaleFilterInterpolation::remapAdaptiveGaussDownIgnoreBoundary(const Mat& src, Mat& remapIm, Mat& dest, const float g, const Mat& sigma_range, const Mat& boost)
	{
		CV_Assert(src.depth() == CV_32F);
		const Size size = src.size();
		dest.create(size / 2, CV_32F);
		remapIm.create(size, CV_32F);

		const int D = 2 * radius + 1;
		const int rs = radius >> 1;
		__m256* W = (__m256*)_mm_malloc(sizeof(__m256) * D, AVX_ALIGN);
		for (int k = 0; k < D; k++)
		{
			W[k] = _mm256_set1_ps(GaussWeight[k]);
		}

		const int width = src.cols;
		const int height = src.rows;

#pragma region remap top
		const __m256 mg = _mm256_set1_ps(g);

		//splat
		{
			const float* sptr = src.ptr<float>();
			const float* asmap = sigma_range.ptr<float>();
			const float* abmap = boost.ptr<float>();
			float* d = remapIm.ptr<float>();
			const int SIZE = get_simd_ceil(width * (D - 1), 8);

			for (int i = 0; i < SIZE; i += 8)
			{
				const __m256 msgma = _mm256_loadu_ps(asmap + i);
				__m256 mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
				const __m256 mdetail = _mm256_loadu_ps(abmap + i);
				__m256 ms = _mm256_loadu_ps(sptr + i);
				__m256 subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
			}
		}

#pragma endregion

		const int linesize = src.cols;
		float* linebuff = (float*)_mm_malloc(sizeof(float) * linesize, AVX_ALIGN);
		memset(linebuff, 0, sizeof(float) * linesize);

		const float* sptr = remapIm.ptr<float>();
		float* dptr = dest.ptr<float>(rs, rs);
		const int hend = width - 2 * radius;
		const int vend = height - 2 * radius;
		const int WIDTH = get_simd_floor(width, 8);
		const int HEND = get_simd_floor(hend, 8);

		for (int j = 0; j < vend; j += 2)
		{
			//remap line
			{
				const float* sptr = src.ptr<float>(j + D - 1);
				const float* asmap = sigma_range.ptr<float>(j + D - 1);
				const float* abmap = boost.ptr<float>(j + D - 1);
				float* d = remapIm.ptr<float>(j + D - 1);
				const int SIZE = get_simd_floor(width * 2, 8);
				for (int i = 0; i < SIZE; i += 8)
				{
					const __m256 msgma = _mm256_loadu_ps(asmap + i);
					const __m256 mcoeff = _mm256_rcpnr_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), _mm256_mul_ps(msgma, msgma)));
					const __m256 mdetail = _mm256_loadu_ps(abmap + i);
					__m256 ms = _mm256_loadu_ps(sptr + i);
					__m256 subsg = _mm256_sub_ps(ms, mg);
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
				}
				for (int i = SIZE; i < width * 2; i++)
				{
					const float sigma = asmap[i];
					const float coeff = 1.f / (-2.f * sigma * sigma);
					const float detail = abmap[i];
					float s = sptr[i];
					float subsg = s - g;
					d[i] = subsg * (detail * exp(subsg * subsg * coeff)) + s;
				}
			}
			//v filter
			for (int i = 0; i < WIDTH; i += 8)
			{
				const float* s = sptr + i;
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(s));
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(s), sum); s += width;
				}
				_mm256_storeu_ps(linebuff + i, sum);
			}
			for (int i = WIDTH; i < width; i++)
			{
				const float* s = sptr + i;
				float sum = GaussWeight[0] * *s;
				s += width;
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * *s;
					s += width;
				}
				linebuff[i] = sum;
			}
			sptr += 2 * width;

			//h filter
			for (int i = 0; i < HEND; i += 8)
			{
				__m256 sum = _mm256_mul_ps(W[0], _mm256_loadu_ps(linebuff + i));
				for (int k = 1; k < D; k++)
				{
					sum = _mm256_fmadd_ps(W[k], _mm256_loadu_ps(linebuff + i + k), sum);
				}
				sum = _mm256_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 0, 2, 0));
				sum = _mm256_permute4x64_ps(sum, _MM_SHUFFLE(2, 0, 2, 0));
				_mm_storeu_ps(dptr + (i >> 1), _mm256_castps256_ps128(sum));
			}
			for (int i = HEND; i < hend; i += 2)
			{
				float sum = GaussWeight[0] * linebuff[i];
				for (int k = 1; k < D; k++)
				{
					sum += GaussWeight[k] * linebuff[i + k];
				}
				dptr[i >> 1] = sum;
			}
			dptr += dest.cols;
		}

		_mm_free(linebuff);
		_mm_free(W);
	}


	template<bool isInit, int interpolation, int D2>
	void LocalMultiScaleFilterInterpolation::GaussUpSubProductSumIgnoreBoundary(const Mat& src, const cv::Mat& subsrc, const Mat& GaussianPyramid, Mat& dest, const float g)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D = 2 * rs + 1;
		//const int D2 = 2 * D;

		const int step = src.cols;

		float* linebuff = (float*)_mm_malloc(sizeof(float) * (src.cols * 2 + 8), AVX_ALIGN);
		float* linee = linebuff;
		float* lineo = linebuff + src.cols;

		const int hend = src.cols - 2 * rs;
		const int HEND8 = get_simd_floor(hend, 8);
		const int WIDTH32 = get_simd_floor(src.cols, 32);
		const int WIDTH8 = get_simd_floor(src.cols, 8);
		const __m256i maskwidth = get_simd_residualmask_epi32(src.cols);

		__m256i maskhendL, maskhendR;
		get_storemask2(hend, maskhendL, maskhendR, 8);

		const float delta = intensityRange / (order - 1);
		const float idelta = 1.f / delta;
		const __m256 mg = _mm256_set1_ps(g);
		const __m256 mgmax = _mm256_set1_ps(intensityMax - delta);
		const __m256 mgmin = _mm256_set1_ps(intensityMin + delta);
		const __m256 midelta = _mm256_set1_ps(idelta);
		const __m256 mcubicalpha = _mm256_set1_ps(cubicAlpha);
		const __m256 mone = _mm256_set1_ps(1.f);
		const __m256 mtwo = _mm256_set1_ps(2.f);
		const __m256 mtwoalpha = _mm256_set1_ps(2.f + cubicAlpha);
		const __m256 mnthreealpha = _mm256_set1_ps(-(3.f + cubicAlpha));
		const __m256 mmfouralpha = _mm256_set1_ps(-4.f * cubicAlpha);
		const __m256 mmfivealpha = _mm256_set1_ps(-5.f * cubicAlpha);
		const __m256 meightalpha = _mm256_set1_ps(8.f * cubicAlpha);

		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH32; i += 32)
			{
				const float* si = sptr + i;
				__m256 sume0 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si));
				__m256 sumo0 = _mm256_setzero_ps();
				__m256 sume1 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 8));
				__m256 sumo1 = _mm256_setzero_ps();
				__m256 sume2 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 16));
				__m256 sumo2 = _mm256_setzero_ps();
				__m256 sume3 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 24));
				__m256 sumo3 = _mm256_setzero_ps();
				si += step;
				for (int k = 2; k < D2; k += 2)
				{
					__m256 ms = _mm256_loadu_ps(si);
					sume0 = _mm256_fmadd_ps(GW[k], ms, sume0);
					sumo0 = _mm256_fmadd_ps(GW[k - 1], ms, sumo0);

					ms = _mm256_loadu_ps(si + 8);
					sume1 = _mm256_fmadd_ps(GW[k], ms, sume1);
					sumo1 = _mm256_fmadd_ps(GW[k - 1], ms, sumo1);

					ms = _mm256_loadu_ps(si + 16);
					sume2 = _mm256_fmadd_ps(GW[k], ms, sume2);
					sumo2 = _mm256_fmadd_ps(GW[k - 1], ms, sumo2);

					ms = _mm256_loadu_ps(si + 24);
					sume3 = _mm256_fmadd_ps(GW[k], ms, sume3);
					sumo3 = _mm256_fmadd_ps(GW[k - 1], ms, sumo3);

					si += step;
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume0, mevenratio));
				_mm256_storeu_ps(linee + i + 8, _mm256_mul_ps(sume1, mevenratio));
				_mm256_storeu_ps(linee + i + 16, _mm256_mul_ps(sume2, mevenratio));
				_mm256_storeu_ps(linee + i + 24, _mm256_mul_ps(sume3, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo0, moddratio));
				_mm256_storeu_ps(lineo + i + 8, _mm256_mul_ps(sumo1, moddratio));
				_mm256_storeu_ps(lineo + i + 16, _mm256_mul_ps(sumo2, moddratio));
				_mm256_storeu_ps(lineo + i + 24, _mm256_mul_ps(sumo3, moddratio));
			}
			for (int i = WIDTH32; i < WIDTH8; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
#ifdef MASKSTORE
			{
				const float* si = sptr + WIDTH8;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_maskstore_ps(linee + WIDTH8, maskwidth, _mm256_mul_ps(sume, mevenratio));
				_mm256_maskstore_ps(lineo + WIDTH8, maskwidth, _mm256_mul_ps(sumo, moddratio));
			}
#else
			for (int i = WIDTH8; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}
#endif

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* gpye = GaussianPyramid.ptr<float>(j, radius);
			const float* gpyo = GaussianPyramid.ptr<float>(j + 1, radius);
			const float* daeptr = subsrc.ptr<float>(j, radius);
			const float* daoptr = subsrc.ptr<float>(j + 1, radius);

			for (int i = 0; i < HEND8; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#ifdef MASKSTORELASTLINEAR
			if (HEND8 != hend)
			{
				const int i = HEND8;
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#else
			for (int i = HEND8; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				float w;

				if constexpr (interpolation == 0) w = (abs((gpye[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 0] = w * (daeptr[I + 0] - sumee * evenratio);
				else deptr[I + 0] += w * (daeptr[I + 0] - sumee * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpye[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 1] = w * (daeptr[I + 1] - sumoe * oddratio);
				else deptr[I + 1] += w * (daeptr[I + 1] - sumoe * oddratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 0] = w * (daoptr[I + 0] - sumeo * evenratio);
				else doptr[I + 0] += w * (daoptr[I + 0] - sumeo * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 1] = w * (daoptr[I + 1] - sumoo * oddratio);
				else doptr[I + 1] += w * (daoptr[I + 1] - sumoo * oddratio);
			}
#endif
		}

		_mm_free(linebuff);
		_mm_free(GW);
	}

	template<bool isInit, int interpolation>
	void LocalMultiScaleFilterInterpolation::GaussUpSubProductSumIgnoreBoundary(const Mat& src, const cv::Mat& subsrc, const Mat& GaussianPyramid, Mat& dest, const float g)
	{
		CV_Assert(src.depth() == CV_32F);
		dest.create(src.size() * 2, src.type());

		__m256* GW = (__m256*)_mm_malloc(sizeof(__m256) * (2 * radius + 1), AVX_ALIGN);
		for (int i = 0; i < 2 * radius + 1; i++)
		{
			GW[i] = _mm256_set1_ps(GaussWeight[i]);
		}
		const __m256 mevenoddratio = _mm256_setr_ps(evenratio, oddratio, evenratio, oddratio, evenratio, oddratio, evenratio, oddratio);
		const __m256 mevenratio = _mm256_set1_ps(evenratio);
		const __m256 moddratio = _mm256_set1_ps(oddratio);
		const int rs = radius >> 1;
		const int D = 2 * rs + 1;
		const int D2 = 2 * D;

		const int step = src.cols;

		float* linebuff = (float*)_mm_malloc(sizeof(float) * (src.cols * 2 + 8), AVX_ALIGN);
		float* linee = linebuff;
		float* lineo = linebuff + src.cols;

		const int hend = src.cols - 2 * rs;
		const int HEND8 = get_simd_floor(hend, 8);
		const int WIDTH32 = get_simd_floor(src.cols, 32);
		const int WIDTH8 = get_simd_floor(src.cols, 8);
		const __m256i maskwidth = get_simd_residualmask_epi32(src.cols);

		__m256i maskhendL, maskhendR;
		get_storemask2(hend, maskhendL, maskhendR, 8);

		const float delta = intensityRange / (order - 1);
		const float idelta = 1.f / delta;
		const __m256 mg = _mm256_set1_ps(g);
		const __m256 mgmax = _mm256_set1_ps(intensityMax - delta);
		const __m256 mgmin = _mm256_set1_ps(intensityMin + delta);
		const __m256 midelta = _mm256_set1_ps(idelta);
		const __m256 mcubicalpha = _mm256_set1_ps(cubicAlpha);
		const __m256 mone = _mm256_set1_ps(1.f);
		const __m256 mtwo = _mm256_set1_ps(2.f);
		const __m256 mtwoalpha = _mm256_set1_ps(2.f + cubicAlpha);
		const __m256 mnthreealpha = _mm256_set1_ps(-(3.f + cubicAlpha));
		const __m256 mmfouralpha = _mm256_set1_ps(-4.f * cubicAlpha);
		const __m256 mmfivealpha = _mm256_set1_ps(-5.f * cubicAlpha);
		const __m256 meightalpha = _mm256_set1_ps(8.f * cubicAlpha);

		for (int j = radius; j < dest.rows - radius; j += 2)
		{
			const float* sptr = src.ptr<float>((j - radius) >> 1);
			//v filter
			for (int i = 0; i < WIDTH32; i += 32)
			{
				const float* si = sptr + i;
				__m256 sume0 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si));
				__m256 sumo0 = _mm256_setzero_ps();
				__m256 sume1 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 8));
				__m256 sumo1 = _mm256_setzero_ps();
				__m256 sume2 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 16));
				__m256 sumo2 = _mm256_setzero_ps();
				__m256 sume3 = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si + 24));
				__m256 sumo3 = _mm256_setzero_ps();
				si += step;
				for (int k = 2; k < D2; k += 2)
				{
					__m256 ms = _mm256_loadu_ps(si);
					sume0 = _mm256_fmadd_ps(GW[k], ms, sume0);
					sumo0 = _mm256_fmadd_ps(GW[k - 1], ms, sumo0);

					ms = _mm256_loadu_ps(si + 8);
					sume1 = _mm256_fmadd_ps(GW[k], ms, sume1);
					sumo1 = _mm256_fmadd_ps(GW[k - 1], ms, sumo1);

					ms = _mm256_loadu_ps(si + 16);
					sume2 = _mm256_fmadd_ps(GW[k], ms, sume2);
					sumo2 = _mm256_fmadd_ps(GW[k - 1], ms, sumo2);

					ms = _mm256_loadu_ps(si + 24);
					sume3 = _mm256_fmadd_ps(GW[k], ms, sume3);
					sumo3 = _mm256_fmadd_ps(GW[k - 1], ms, sumo3);

					si += step;
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume0, mevenratio));
				_mm256_storeu_ps(linee + i + 8, _mm256_mul_ps(sume1, mevenratio));
				_mm256_storeu_ps(linee + i + 16, _mm256_mul_ps(sume2, mevenratio));
				_mm256_storeu_ps(linee + i + 24, _mm256_mul_ps(sume3, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo0, moddratio));
				_mm256_storeu_ps(lineo + i + 8, _mm256_mul_ps(sumo1, moddratio));
				_mm256_storeu_ps(lineo + i + 16, _mm256_mul_ps(sumo2, moddratio));
				_mm256_storeu_ps(lineo + i + 24, _mm256_mul_ps(sumo3, moddratio));
			}
			for (int i = WIDTH32; i < WIDTH8; i += 8)
			{
				const float* si = sptr + i;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_storeu_ps(linee + i, _mm256_mul_ps(sume, mevenratio));
				_mm256_storeu_ps(lineo + i, _mm256_mul_ps(sumo, moddratio));
			}
#ifdef MASKSTORE
			{
				const float* si = sptr + WIDTH8;
				__m256 sume = _mm256_mul_ps(GW[0], _mm256_loadu_ps(si)); si += step;
				__m256 sumo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 ms = _mm256_loadu_ps(si); si += step;
					sume = _mm256_fmadd_ps(GW[k], ms, sume);
					sumo = _mm256_fmadd_ps(GW[k - 1], ms, sumo);
				}
				_mm256_maskstore_ps(linee + WIDTH8, maskwidth, _mm256_mul_ps(sume, mevenratio));
				_mm256_maskstore_ps(lineo + WIDTH8, maskwidth, _mm256_mul_ps(sumo, moddratio));
			}
#else
			for (int i = WIDTH8; i < src.cols; i++)
			{
				const float* si = sptr + i;
				float sume = GaussWeight[0] * *si; si += step;
				float sumo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sume += GaussWeight[K] * *si;
					sumo += GaussWeight[K - 1] * *si;
					si += step;
				}
				linee[i] = sume * evenratio;
				lineo[i] = sumo * oddratio;
			}
#endif

			// h filter
			float* deptr = dest.ptr<float>(j, radius);
			float* doptr = dest.ptr<float>(j + 1, radius);
			const float* gpye = GaussianPyramid.ptr<float>(j, radius);
			const float* gpyo = GaussianPyramid.ptr<float>(j + 1, radius);
			const float* daeptr = subsrc.ptr<float>(j, radius);
			const float* daoptr = subsrc.ptr<float>(j + 1, radius);

			for (int i = 0; i < HEND8; i += 8)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_storeu_ps(deptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_storeu_ps(deptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_storeu_ps(doptr + 2 * i + 0, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_storeu_ps(doptr + 2 * i + 8, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#ifdef MASKSTORE0
			//last
			{
				const int i = HEND8;
				float* sie = linee + i;
				float* sio = lineo + i;
				__m256 sumee = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sie++));
				__m256 sumoe = _mm256_setzero_ps();
				__m256 sumeo = _mm256_mul_ps(GW[0], _mm256_loadu_ps(sio++));
				__m256 sumoo = _mm256_setzero_ps();
				for (int k = 2; k < D2; k += 2)
				{
					const __m256 msie = _mm256_loadu_ps(sie++);
					sumee = _mm256_fmadd_ps(GW[k], msie, sumee);
					sumoe = _mm256_fmadd_ps(GW[k - 1], msie, sumoe);
					const __m256 msio = _mm256_loadu_ps(sio++);
					sumeo = _mm256_fmadd_ps(GW[k], msio, sumeo);
					sumoo = _mm256_fmadd_ps(GW[k - 1], msio, sumoo);
				}

				__m256 s1 = _mm256_unpacklo_ps(sumee, sumoe);
				__m256 s2 = _mm256_unpackhi_ps(sumee, sumoe);

				__m256 w;
				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0))));
				else _mm256_maskstore_ps(deptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daeptr + 2 * i + 0)), _mm256_loadu_ps(deptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpye + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpye + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8))));
				else _mm256_maskstore_ps(deptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daeptr + 2 * i + 8)), _mm256_loadu_ps(deptr + 2 * i + 8)));

				s1 = _mm256_unpacklo_ps(sumeo, sumoo);
				s2 = _mm256_unpackhi_ps(sumeo, sumoo);

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 0), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 0);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0))));
				else _mm256_maskstore_ps(doptr + 2 * i + 0, maskhendL, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x20), _mm256_loadu_ps(daoptr + 2 * i + 0)), _mm256_loadu_ps(doptr + 2 * i + 0)));

				if constexpr (interpolation == 0) w = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg))), _mm256_set1_ps(0.5f), 14), mone);
				if constexpr (interpolation == 1) w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpyo + 2 * i + 8), mg)), mone));
				if constexpr (interpolation == 2)
				{
					const __m256 mgpy = _mm256_loadu_ps(gpyo + 2 * i + 8);
					const __m256 x = _mm256_mul_ps(midelta, _mm256_abs_ps(_mm256_sub_ps(mgpy, mg)));
					const __m256 x2 = _mm256_mul_ps(x, x);
					const __m256 x3 = _mm256_mul_ps(x2, x);
					const __m256 m1 = _mm256_fmadd_ps(mtwoalpha, x3, _mm256_fmadd_ps(mnthreealpha, x2, mone));
					const __m256 m2 = _mm256_fmadd_ps(mcubicalpha, x3, _mm256_fmadd_ps(mmfivealpha, x2, _mm256_fmadd_ps(meightalpha, x, mmfouralpha)));
					w = _mm256_andnot_ps(_mm256_cmp_ps(x, mtwo, 14), m2);
					w = _mm256_blendv_ps(m1, w, _mm256_cmp_ps(x, mone, 14));

					__m256 wl = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(mone, x));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmin, 1));
					w = _mm256_blendv_ps(w, wl, _mm256_cmp_ps(mgpy, mgmax, 14));
				}
				if constexpr (isInit) _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_mul_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8))));
				else _mm256_maskstore_ps(doptr + 2 * i + 8, maskhendR, _mm256_fmadd_ps(w, _mm256_fnmadd_ps(mevenoddratio, _mm256_permute2f128_ps(s1, s2, 0x31), _mm256_loadu_ps(daoptr + 2 * i + 8)), _mm256_loadu_ps(doptr + 2 * i + 8)));
			}
#else
			for (int i = HEND8; i < hend; i++)
			{
				float* sie = linee + i;
				float* sio = lineo + i;
				float sumee = GaussWeight[0] * *sie++;
				float sumoe = 0.f;
				float sumeo = GaussWeight[0] * *sio++;
				float sumoo = 0.f;
				for (int k = 1; k < D; k++)
				{
					const int K = k << 1;
					sumee += GaussWeight[K] * *sie;
					sumoe += GaussWeight[K - 1] * *sie++;
					sumeo += GaussWeight[K] * *sio;
					sumoo += GaussWeight[K - 1] * *sio++;
				}
				const int I = i << 1;
				float w;

				if constexpr (interpolation == 0) w = (abs((gpye[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 0] = w * (daeptr[I + 0] - sumee * evenratio);
				else deptr[I + 0] += w * (daeptr[I + 0] - sumee * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpye[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpye[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpye[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) deptr[I + 1] = w * (daeptr[I + 1] - sumoe * oddratio);
				else deptr[I + 1] += w * (daeptr[I + 1] - sumoe * oddratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 0] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 0] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 0];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 0] = w * (daoptr[I + 0] - sumeo * evenratio);
				else doptr[I + 0] += w * (daoptr[I + 0] - sumeo * evenratio);

				if constexpr (interpolation == 0) w = (abs((gpyo[I + 1] - g) * idelta) < 0.5f) ? 1.f : 0.f;
				if constexpr (interpolation == 1) w = max(0.f, 1.f - abs((gpyo[I + 1] - g) * idelta));
				if constexpr (interpolation == 2)
				{
					float gpy = gpyo[I + 1];
					w = getCubicCoeff((gpy - g) * idelta, cubicAlpha);
					if (gpy < intensityMin + delta) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
					if (gpy > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy - g) * idelta);//hat
				}
				if constexpr (isInit) doptr[I + 1] = w * (daoptr[I + 1] - sumoo * oddratio);
				else doptr[I + 1] += w * (daoptr[I + 1] - sumoo * oddratio);
			}
#endif
		}

		_mm_free(linebuff);
		_mm_free(GW);
	}


	//for parallel
	void LocalMultiScaleFilterInterpolation::buildRemapLaplacianPyramidEachOrder(const Mat& src, vector<Mat>& destPyramid, const int level, const float sigma, const float g, const float sigma_range, const float boost)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		//destPyramid[0].create(src.size(), CV_32F);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			remapGaussDownIgnoreBoundary<false>(src, destPyramid[0], destPyramid[1], g, sigma_range, boost);
			GaussUpAddIgnoreBoundary <false>(destPyramid[1], destPyramid[0], destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary(destPyramid[l], destPyramid[l + 1]);
				GaussUpAddIgnoreBoundary <false>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Fast)
		{
			cout << "not supported: buildRemapLaplacianPyramid" << endl;
			GaussDown(src, destPyramid[1]);
			GaussUpAdd<false>(destPyramid[1], src, destPyramid[0]);
			for (int l = 1; l < level; l++)
			{
				GaussDown(destPyramid[l], destPyramid[l + 1]);
				GaussUpAdd <false>(destPyramid[l + 1], destPyramid[l], destPyramid[l]);
			}
		}
		else if (pyramidComputeMethod == Full)
		{
			cout << "not supported: buildRemapLaplacianPyramid" << endl;
			GaussDownFull(src, destPyramid[1], sigma, borderType);
			GaussUpAddFull<false>(destPyramid[1], src, destPyramid[0], sigma, borderType);
			for (int l = 1; l < level; l++)
			{
				GaussDownFull(destPyramid[l], destPyramid[l + 1], sigma, borderType);
				GaussUpAddFull<false>(destPyramid[l + 1], destPyramid[l], destPyramid[l], sigma, borderType);
			}
		}
		else if (pyramidComputeMethod == OpenCV)
		{
			cout << "not supported: buildRemapLaplacianPyramid" << endl;
			buildPyramid(src, destPyramid, level, borderType);
			for (int i = 0; i < level; i++)
			{
				Mat temp;
				pyrUp(destPyramid[i + 1], temp, destPyramid[i].size(), borderType);
				subtract(destPyramid[i], temp, destPyramid[i]);
			}
		}
	}

	//for serial
	template<bool isInit>
	void LocalMultiScaleFilterInterpolation::buildRemapLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, vector<Mat>& destPyramid, const int level, const float sigma, const float g, const float sigma_range, const float boost)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		if (LaplacianPyramid.size() != level + 1) LaplacianPyramid.resize(level + 1);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (isUseTable)
			{
				if (radius == 2) remapGaussDownIgnoreBoundary<true, 5>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else if (radius == 4) remapGaussDownIgnoreBoundary<true, 9>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else remapGaussDownIgnoreBoundary<true>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
			}
			else
			{
				if (radius == 2) remapGaussDownIgnoreBoundary<false, 5>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else if (radius == 4) remapGaussDownIgnoreBoundary<false, 9>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
				else remapGaussDownIgnoreBoundary<false>(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
			}

			//const int rs = radius >> 1;
			//const int D = 2 * rs + 1;
			//const int D2 = 2 * D;
			if (interpolation_method == 0)
			{
				if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 6>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 10>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			}
			if (interpolation_method == 1)
			{
				if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 6>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 10>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			}
			if (interpolation_method == 2)
			{
				if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 6>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 10>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
				else GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			}

			float* linebuff = (float*)_mm_malloc(sizeof(float) * LaplacianPyramid[1].cols, AVX_ALIGN);
			for (int l = 1; l < level; l++)
			{
				if (radius == 2)  GaussDownIgnoreBoundary<5>(LaplacianPyramid[l], LaplacianPyramid[l + 1], linebuff);
				else if (radius == 4) GaussDownIgnoreBoundary<9>(LaplacianPyramid[l], LaplacianPyramid[l + 1], linebuff);
				else GaussDownIgnoreBoundary(LaplacianPyramid[l], LaplacianPyramid[l + 1]);

				if (interpolation_method == 0)
				{
					if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 6>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 0, 10>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else  GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				}
				if (interpolation_method == 1)
				{
					if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 6>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 1, 10>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				}
				if (interpolation_method == 2)
				{
					if (radius == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 6>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else if (radius == 4) GaussUpSubProductSumIgnoreBoundary<isInit, 2, 10>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
					else GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				}
			}
			_mm_free(linebuff);
		}
	}

	template<bool isInit>
	void LocalMultiScaleFilterInterpolation::buildRemapAdaptiveLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& LaplacianPyramid, vector<Mat>& destPyramid, const int level, const float sigma, const float g, const Mat& sigma_range, const Mat& boost)
	{
		if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);
		if (LaplacianPyramid.size() != level + 1) LaplacianPyramid.resize(level + 1);
		//destPyramid[0].create(src.size(), CV_32F);

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			remapAdaptiveGaussDownIgnoreBoundary(GaussianPyramid[0], LaplacianPyramid[0], LaplacianPyramid[1], g, sigma_range, boost);
			if (interpolation_method == 0) GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			if (interpolation_method == 1) GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			if (interpolation_method == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[1], LaplacianPyramid[0], GaussianPyramid[0], destPyramid[0], g);
			for (int l = 1; l < level; l++)
			{
				GaussDownIgnoreBoundary(LaplacianPyramid[l], LaplacianPyramid[l + 1]);
				if (interpolation_method == 0) GaussUpSubProductSumIgnoreBoundary<isInit, 0>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				if (interpolation_method == 1) GaussUpSubProductSumIgnoreBoundary<isInit, 1>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
				if (interpolation_method == 2) GaussUpSubProductSumIgnoreBoundary<isInit, 2>(LaplacianPyramid[l + 1], LaplacianPyramid[l], GaussianPyramid[l], destPyramid[l], g);
			}
		}
	}
#pragma endregion

	void LocalMultiScaleFilterInterpolation::blendDetailStack(const vector<vector<Mat>>& detailStack, const vector<Mat>& approxStack, vector<Mat>& destStack, const int order, const int interpolationMethod)
	{
		const int level = (int)approxStack.size();
		AutoBuffer<const float*> lptr(order);

		if (order == 256)
		{
			for (int l = 0; l < level - 1; l++)
			{
				const float* s = approxStack[l].ptr<float>();
				float* d = destStack[l].ptr<float>();
				for (int i = 0; i < approxStack[l].size().area(); i++)
				{
					const int c = saturate_cast<uchar>(s[i]);
					d[i] = detailStack[c][l].at<float>(i);
				}
			}
		}
		else
		{
			for (int l = 0; l < level - 1; l++)
			{
				const float* g = approxStack[l].ptr<float>();
				float* d = destStack[l].ptr<float>();
				for (int k = 0; k < order; k++)
				{
					lptr[k] = detailStack[k][l].ptr<float>();
				}

				if (!isParallel) omp_set_num_threads(1);

				if (interpolationMethod == INTER_NEAREST)
				{
					const float idelta = (order - 1) / intensityRange;
#pragma omp parallel for //schedule (dynamic)
					for (int i = 0; i < approxStack[l].size().area(); i++)
					{
						const int c = min(order - 1, (int)saturate_cast<uchar>((g[i] - intensityMin) * idelta));
						//const int c = min(order - 1, int(g[i] * istep+0.5));
						d[i] = lptr[c][i];
					}
				}
				else if (interpolationMethod == INTER_LINEAR)
				{
#pragma omp parallel for //schedule (dynamic)
					for (int i = 0; i < approxStack[l].size().area(); i++)
					{
						float alpha;
						int high, low;
						getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
						d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
					}
				}
				else if (interpolationMethod == INTER_CUBIC)
				{
#pragma omp parallel for //schedule (dynamic)
					for (int i = 0; i < approxStack[l].size().area(); i++)
					{
						d[i] = getCubicInterpolation(g[i], order, lptr, i, cubicAlpha, intensityMin, intensityMax);
					}
				}
				if (!isParallel) omp_set_num_threads(omp_get_max_threads());
			}
		}
	}

	template<int interpolation>
	void LocalMultiScaleFilterInterpolation::productSumLaplacianPyramid(const std::vector<cv::Mat>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order, const float g)
	{
		const int level = (int)GaussianPyramid.size();

		for (int l = 0; l < level - 1; l++)
		{
			const float* lpy = LaplacianPyramid[l].ptr<float>();
			float* gpy = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();

			//const float delta = intensityRange / (order - 2);
			const float delta = intensityRange / (order - 1);
			const float idelta = 1.f / delta;
			if (isParallel)
			{
#pragma omp parallel for //schedule (dynamic)
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					;
				}
			}
			else
			{
				//__m256 milinearstep = _mm256_set1_ps(istep);
				//__m256 mlinearstepk = _mm256_set1_ps(g);
				/*for (int i = 0; i < GaussianPyramid[l].size().area(); i += 8)
				{
					//const float w = max(0.f, 1.f - abs(gpy[i] - g) * istep);//hat
					//d[i] += w * lpy[i];
					__m256 w = _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(milinearstep, _mm256_abs_ps(_mm256_sub_ps(_mm256_loadu_ps(gpy + i + 0), mlinearstepk)), _mm256_set1_ps(1.f)));
					_mm256_storeu_ps(d + i, _mm256_fmadd_ps(w, _mm256_loadu_ps(lpy + i), _mm256_loadu_ps(d + i)));
				}*/
				for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
				{
					float w;
					if constexpr (interpolation == 0)
					{
						//const int c = min(order - 1, (int)saturate_cast<uchar>(g[i] * istep));
						w = (abs(gpy[i] - g) * idelta < 0.5f) ? 1.f : 0.f;
					}
					else if constexpr (interpolation == 1)
					{
						w = max(0.f, 1.f - abs(gpy[i] - g) * idelta);//hat
					}
					else //cv::INTER_CUBIC
					{
						w = getCubicCoeff((gpy[i] - g) * idelta, cubicAlpha);
						if (gpy[i] < intensityMin + delta) w = max(0.f, 1.f - abs(gpy[i] - g) * idelta);//hat
						if (gpy[i] > (intensityMax - delta)) w = max(0.f, 1.f - abs(gpy[i] - g) * idelta);//hat
					}
					d[i] += w * lpy[i];
				}
			}
		}
	}


	void LocalMultiScaleFilterInterpolation::pyramidParallel(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);

		remapIm.resize(threadMax);

		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		const int gfRadius = getGaussianRadius(sigma_space);
		const int lowr = 2 * gfRadius + gfRadius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		Mat smap, bmap;
		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (src.depth() == CV_32F)
			{
				copyMakeBorder(src, GaussianPyramid[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
			}
			else
			{
				copyMakeBorder(src, border, r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				border.convertTo(GaussianPyramid[0], CV_32F);
			}
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				adaptiveSigmaBorder.resize(1);
				adaptiveBoostBorder.resize(1);
				cv::copyMakeBorder(adaptiveSigmaMap[0], adaptiveSigmaBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				cv::copyMakeBorder(adaptiveBoostMap[0], adaptiveBoostBorder[0], r_pad0, r_pad0, r_pad0, r_pad0, borderType);
				smap = adaptiveSigmaBorder[0];
				bmap = adaptiveBoostBorder[0];
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				src.copyTo(GaussianPyramid[0]);
			}
			else
			{
				src.convertTo(GaussianPyramid[0], CV_32F);
			}
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				smap = adaptiveSigmaMap[0];
				bmap = adaptiveBoostMap[0];
			}
		}

		//(1) build Gaussian Pyramid
		{
			//cp::Timer t("(1) build Gaussian Pyramid");
			buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);
		}

		//(2) build Laplacian Pyramid
		LaplacianPyramid.resize(order);
		{
			//cp::Timer t("(2) build Laplacian Pyramid");
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
#pragma omp parallel for schedule(dynamic)
				for (int n = 0; n < order; n++)
				{
					const int tidx = omp_get_thread_num();
					remapAdaptive(GaussianPyramid[0], remapIm[tidx], getTau(n), smap, bmap);
					buildLaplacianPyramid(remapIm[tidx], LaplacianPyramid[n], level, sigma_space);
				}
			}
			else
			{
#pragma omp parallel for schedule(dynamic)
				for (int n = 0; n < order; n++)
				{
					const int tidx = omp_get_thread_num();
#if 0
					float* linebuff = (float*)_mm_malloc(sizeof(float) * GaussianPyramid[0].cols, AVX_ALIGN);
					remap(GaussianPyramid[0], remapIm[tidx], (float)(step * n), sigma_range, detail_param);
					if (radius == 2)
					{
						buildLaplacianPyramid<5, 3, 6>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
					}
					else if (radius == 4)
					{
						buildLaplacianPyramid<9, 5, 10>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
					}
					else
					{
						buildLaplacianPyramid(remapIm[tidx], LaplacianPyramid[n], level, sigma_space);
					}
					_mm_free(linebuff);
#else
					buildRemapLaplacianPyramidEachOrder(GaussianPyramid[0], LaplacianPyramid[n], level, sigma_space, getTau(n), sigma_range, boost);
#endif
				}
			}

			blendDetailStack(LaplacianPyramid, GaussianPyramid, GaussianPyramid, order, interpolation_method);//orverride destnation pyramid for saving memory
		}

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			collapseLaplacianPyramid(GaussianPyramid, GaussianPyramid[0]);
			if (src.depth() == CV_32F)
			{
				GaussianPyramid[0](Rect(r_pad0, r_pad0, src.cols, src.rows)).copyTo(dest);
			}
			else
			{
				GaussianPyramid[0](Rect(r_pad0, r_pad0, src.cols, src.rows)).convertTo(dest, src.type());
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				collapseLaplacianPyramid(GaussianPyramid, dest);
			}
			else
			{
				Mat srcf;
				collapseLaplacianPyramid(GaussianPyramid, srcf);//override srcf for saving memory	
				srcf.convertTo(dest, src.type());
			}
		}
		//showPyramid("Laplacian Pyramid fast", GaussianPyramid);
	}

	void LocalMultiScaleFilterInterpolation::pyramidSerial(const Mat& src, Mat& dest)
	{
		layerSize.resize(level + 1);

		//initRangeTable(sigma_range, boost);
		if (isUseTable) initRangeTableInteger(sigma_range, boost);

		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		const int gfRadius = getGaussianRadius(sigma_space);
		const int lowr = 2 * gfRadius + gfRadius;
		const int r_pad0 = lowr * (int)pow(2, level - 1);

		Mat smap, bmap;
		if (pyramidComputeMethod == IgnoreBoundary)
		{
			if (src.depth() == CV_8U)
			{
				src.convertTo(GaussianPyramid[0], CV_32F);
				//src8u = src;
			}
			else
			{
				src.copyTo(GaussianPyramid[0]);
			}

			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				smap = adaptiveSigmaMap[0];
				bmap = adaptiveBoostMap[0];
			}
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				src.copyTo(GaussianPyramid[0]);
			}
			else
			{
				src.convertTo(GaussianPyramid[0], CV_32F);
			}
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				smap = adaptiveSigmaMap[0];
				bmap = adaptiveBoostMap[0];
			}
		}

		//(1) build Gaussian Pyramid
		{
			//cp::Timer t("(1) build Gaussian Pyramid");
			buildGaussianPyramid(GaussianPyramid[0], GaussianPyramid, level, sigma_space);
			ImageStack.resize(GaussianPyramid.size());
			for (int i = 0; i < GaussianPyramid.size() - 1; i++)
			{
				ImageStack[i].create(GaussianPyramid[i].size(), CV_32F);
			}
			ImageStack[level] = GaussianPyramid[level];
		}

		//(2) build Laplacian Pyramid
		LaplacianPyramid.resize(1);
		{
			//cp::Timer t("(2) build Laplacian Pyramid");
			if (adaptiveMethod == AdaptiveMethod::ADAPTIVE)
			{
				buildRemapAdaptiveLaplacianPyramid<true>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(0), smap, bmap);
				for (int n = 1; n < order; n++)
				{
					buildRemapAdaptiveLaplacianPyramid<false>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(n), smap, bmap);
				}
			}
			else
			{
				const bool test = false;
				if (test)
				{
					for (int n = 0; n < order; n++)
					{
						buildRemapLaplacianPyramidEachOrder(GaussianPyramid[0], LaplacianPyramid[0], level, sigma_space, getTau(n), sigma_range, boost);
						if (interpolation_method == 0) productSumLaplacianPyramid<0>(LaplacianPyramid[0], GaussianPyramid, ImageStack, order, getTau(n));
						if (interpolation_method == 1) productSumLaplacianPyramid<1>(LaplacianPyramid[0], GaussianPyramid, ImageStack, order, getTau(n));
						if (interpolation_method == 2) productSumLaplacianPyramid<2>(LaplacianPyramid[0], GaussianPyramid, ImageStack, order, getTau(n));
					}
				}
				else
				{
					buildRemapLaplacianPyramid<true>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(0), sigma_range, boost);
					for (int n = 1; n < order; n++)
					{
#if 0
						float* linebuff = (float*)_mm_malloc(sizeof(float) * GaussianPyramid[0].cols, AVX_ALIGN);
						remap(GaussianPyramid[0], remapIm[tidx], (float)(step * n), sigma_range, detail_param);
						if (radius == 2)
						{
							buildLaplacianPyramid<5, 3, 6>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
						}
						else if (radius == 4)
						{
							buildLaplacianPyramid<9, 5, 10>(remapIm[tidx], LaplacianPyramid[n], level, sigma_space, linebuff);
						}
						else
						{
							buildLaplacianPyramid(remapIm[tidx], LaplacianPyramid[n], level, sigma_space);
						}
						_mm_free(linebuff);
#else
						buildRemapLaplacianPyramid<false>(GaussianPyramid, LaplacianPyramid[0], ImageStack, level, sigma_space, getTau(n), sigma_range, boost);
#endif
					}
				}
			}
		}

		if (pyramidComputeMethod == IgnoreBoundary)
		{
			collapseLaplacianPyramid(ImageStack, dest);
			//collapseLaplacianPyramid(GaussianPyramid, dest);
		}
		else
		{
			if (src.depth() == CV_32F)
			{
				collapseLaplacianPyramid(GaussianPyramid, dest);
			}
			else
			{
				Mat srcf;
				collapseLaplacianPyramid(GaussianPyramid, srcf);//override srcf for saving memory	
				srcf.convertTo(dest, src.type());
			}
		}
		//showPyramid("Laplacian Pyramid fast", GaussianPyramid);
		if (isUseTable)
		{
			_mm_free(integerSampleTable);
			integerSampleTable = nullptr;
		}
	}

	void LocalMultiScaleFilterInterpolation::pyramid(const Mat& src, Mat& dest)
	{
		rangeDescope(src);

		if (isParallel) pyramidParallel(src, dest);
		else pyramidSerial(src, dest);
	}

	void LocalMultiScaleFilterInterpolation::dog(const Mat& src, Mat& dest)
	{
		initRangeTable(sigma_range, boost);
		remapIm.resize(omp_get_max_threads());

		Mat srcf;
		if (src.depth() == CV_32F)
		{
			srcf = src;
		}
		else
		{
			src.convertTo(srcf, CV_32F);
		}

		//(1) build Gaussian stack
		{
			//merged in next step for parallelization
			//cp::Timer t("(1) build DoG");
			//buildGaussianStack(srcf, GaussianStack, sigma_space, level);
		}

		//(2) build DoG stack
		DoGStackLayer.resize(order);
		const float step = 255.f / (order - 1);
		{
			//cp::Timer t("(2) build DoG stack");
#pragma omp parallel for schedule(dynamic)
			for (int n = -1; n < order; n++)
			{
				if (n == -1)
				{
					buildGaussianStack(srcf, GaussianStack, sigma_space, level);
				}
				else
				{
					const int tidx = omp_get_thread_num();
					remap(srcf, remapIm[tidx], (float)(step * n), sigma_range, boost);
					buildDoGStack(remapIm[tidx], DoGStackLayer[n], sigma_space, level);
				}
			}
			blendDetailStack(DoGStackLayer, GaussianStack, GaussianStack, order, interpolation_method);//orverride destnation pyramid for saving memory
		}

		collapseDoGStack(GaussianStack, dest, src.depth());
	}

	void LocalMultiScaleFilterInterpolation::filter(const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const ScaleSpace scaleSpaceMethod, const int interpolationMethod)
	{
		allocSpaceWeight(sigma_space);

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->level = level;
		this->boost = boost;
		this->scalespaceMethod = scaleSpaceMethod;

		this->interpolation_method = interpolationMethod;
		this->order = order;
		body(src, dest);

		freeSpaceWeight();
	}


	void LocalMultiScaleFilterInterpolation::setCubicAlpha(const float alpha)
	{
		this->cubicAlpha = alpha;
	}

	void LocalMultiScaleFilterInterpolation::setComputeScheduleMethod(const bool useTable)
	{
		this->isUseTable = useTable;
	}

	string LocalMultiScaleFilterInterpolation::getComputeScheduleName()
	{
		string ret = "";
		if (this->isUseTable)ret = "IntegerSampledTable";
		else ret = "Compute";
		return ret;
	}

	void LocalMultiScaleFilterInterpolation::setIsParallel(const bool flag)
	{
		isParallel = flag;
	}

#pragma region TileLocalMultiScaleFilterInterpolation
	TileLocalMultiScaleFilterInterpolation::TileLocalMultiScaleFilterInterpolation()
	{
		msf = new LocalMultiScaleFilterInterpolation[threadMax];
		for (int i = 0; i < threadMax; i++)
			msf[i].setIsParallel(false);
	}

	TileLocalMultiScaleFilterInterpolation::~TileLocalMultiScaleFilterInterpolation()
	{
		delete[] msf;
	}

	void TileLocalMultiScaleFilterInterpolation::setComputeScheduleMethod(bool useTable)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setComputeScheduleMethod(useTable);
	}

	string TileLocalMultiScaleFilterInterpolation::getComputeScheduleName()
	{
		return msf[0].getComputeScheduleName();
	}

	void TileLocalMultiScaleFilterInterpolation::setAdaptive(const bool flag, const cv::Size div, cv::Mat& adaptiveSigmaMap, cv::Mat& adaptiveBoostMap)
	{
		if (flag)
		{
			vector<Mat> g{ adaptiveSigmaMap, adaptiveBoostMap };
			initGuide(div, g);
		}
		else
		{
			unsetUseGuide();
		}
	}

	void TileLocalMultiScaleFilterInterpolation::setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod scaleSpaceMethod)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setRangeDescopeMethod(scaleSpaceMethod);
	}

	void TileLocalMultiScaleFilterInterpolation::setCubicAlpha(const float alpha)
	{
		for (int i = 0; i < threadMax; i++)
			msf[i].setCubicAlpha(alpha);
	}

	void TileLocalMultiScaleFilterInterpolation::process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex)
	{
		if (isUseGuide)
		{
			msf[threadIndex].setAdaptive(true, guideTile[0][imageIndex], guideTile[1][imageIndex], level);
		}
		else
		{
			Mat a;
			msf[threadIndex].setAdaptive(false, a, a, 0);
		}

		msf[threadIndex].filter(src, dst, order, sigma_range, sigma_space, boost, level, scaleSpaceMethod, interpolation);
	}

	void TileLocalMultiScaleFilterInterpolation::filter(const Size div, const Mat& src, Mat& dest, const int order, const float sigma_range, const float sigma_space, const float boost, const int level, const MultiScaleFilter::ScaleSpace scaleSpaceMethod, int interpolation)
	{
		this->order = order;
		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
		this->boost = boost;
		this->level = level;

		this->scaleSpaceMethod = scaleSpaceMethod;
		this->interpolation = interpolation;

		const int lowr = 3 * msf[0].getGaussianRadius(sigma_space);
		const int r_pad0 = lowr * (int)pow(2, level - 1);
		invoker(div, src, dest, r_pad0);
	}
#pragma endregion
}