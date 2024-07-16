#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"
#include "simdexp_local.hpp"

using namespace std;
using namespace cv;

//computeIndex(signal, signalRes);//K*imsize
//split_blur<1>(k, isUseFmath, isUseLocalStatisticsPrior);
//merge(k, k == 0);
//normalize(dst);
namespace cp
{
#pragma region setter
	void ClusteringHDKF_Interpolation2Single::setIsUseLocalMu(const bool flag)
	{
		this->isUseLocalMu = flag;
	}

	void ClusteringHDKF_Interpolation2Single::setIsUseLocalStatisticsPrior(const bool flag)
	{
		this->isUseLocalStatisticsPrior = flag;
	}

	void ClusteringHDKF_Interpolation2Single::setDeltaLocalStatisticsPrior(const float delta)
	{
		this->delta = delta;
	}
#pragma endregion

	void ClusteringHDKF_Interpolation2Single::alloc(cv::Mat& dst)
	{
		if (isUseLocalStatisticsPrior)
		{
			blendLSPMask.create(img_size, CV_32F);
			lsp.resize(3);
			lsp[0].create(img_size, CV_32F);
			lsp[1].create(img_size, CV_32F);
			lsp[2].create(img_size, CV_32F);
		}

		if (downSampleImage != 1)
		{
			vsrcRes.resize(channels);
			vguideRes.resize(guide_channels);

			NumerDenomRes.resize(channels + 1);
			for (int c = 0; c < channels + 1; c++)
			{
				NumerDenomRes[c].create(img_size, CV_32FC1);
			}
		}

		if (alpha.size() != K || alpha[0].size() != img_size)
		{
			alpha.resize(K);
			for (int i = 0; i < K; i++)
			{
				alpha[i].create(img_size, CV_32FC1);
			}
		}

		if (denom.size() != img_size)
			denom.create(img_size, CV_32FC1);

		if (numer.size() != channels || numer[0].size() != img_size)
		{
			numer.resize(channels);
			for (int c = 0; c < channels; c++)
			{
				numer[c].create(img_size, CV_32FC1);
			}
		}

		if (split_inter.size() != channels) split_inter.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			if (split_inter[c].size() != img_size / downSampleImage)
				split_inter[c].create(img_size / downSampleImage, CV_32FC1);
		}
		if (vecW.size() != K || vecW[0].size() != img_size / downSampleImage)
		{
			vecW.resize(K);
			for (int i = 0; i < K; i++)
			{
				vecW[i].create(img_size / downSampleImage, CV_32FC1);
			}

			wmap.resize(K);
			for (int i = 0; i < K; i++)
			{
				wmap[i].create(img_size / downSampleImage, CV_32FC1);
			}
		}

		dst.create(img_size, CV_MAKETYPE(CV_32F, channels));
	}

	//isUseLocalMu=false, without local mu
	template<int use_fmath>
	void ClusteringHDKF_Interpolation2Single::computeWandAlpha(const std::vector<cv::Mat>& guide)
	{
		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);
		
		AutoBuffer<float*> w_ptr(K);
		for (int k = 0; k < K; k++)
		{
			w_ptr[k] = vecW[k].ptr<float>();
		}

		const int method = 0;//nk-loop-hard
		//const int method = 1;//nk-loop-soft
		if (method == 0)
		{
			const __m256 margclip = _mm256_set1_ps(float(sigma_range * 6.0 * sigma_range * 6.0));
			const __m256 mone = _mm256_set1_ps(1.f);
			__m256* mk = (__m256*)_mm_malloc(sizeof(__m256) * K, AVX_ALIGN);
			for (int k = 0; k < K; k++)
			{
				mk[k] = _mm256_set1_ps(float(k));
			}

			if (guide_channels == 3)
			{
				const float* im0 = guide[0].ptr<float>();
				const float* im1 = guide[1].ptr<float>();
				const float* im2 = guide[2].ptr<float>();
				cv::AutoBuffer<float*> aptr(K);
				__m256* mc0 = (__m256*)_mm_malloc(sizeof(__m256) * K, AVX_ALIGN);
				__m256* mc1 = (__m256*)_mm_malloc(sizeof(__m256) * K, AVX_ALIGN);
				__m256* mc2 = (__m256*)_mm_malloc(sizeof(__m256) * K, AVX_ALIGN);
				for (int k = 0; k < K; k++)
				{
					aptr[k] = alpha[k].ptr<float>();
					mc0[k] = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
					mc1[k] = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
					mc2[k] = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);
				}

				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);
					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_set1_ps(0.f);

					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(aptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						aptr[k] += 8;
						_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
					}
					for (int k = 0; k < K; k++)
					{
						//_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_set1_ps(FLT_MIN), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
						const __m256 m = _mm256_cmp_ps(argment, mk[k], 0);
						_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(m, mone, m));
					}
				}

				_mm_free(mc0);
				_mm_free(mc1);
				_mm_free(mc2);
			}
			else //n-dimensional signal
			{
				cv::AutoBuffer<float*> gptr(guide_channels);
				cv::AutoBuffer<__m256> mguide(guide_channels);
				for (int c = 0; c < guide_channels; c++) gptr[c] = (float*)guide[c].ptr<float>();

				for (int n = 0; n < img_size.area(); n += 8)
				{
					for (int c = 0; c < guide_channels; c++) mguide[c] = _mm256_load_ps(gptr[c] + n);

					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_setzero_ps();

					for (int k = 0; k < K; k++)
					{
						const float* muPtr = mu.ptr<float>(k);
						__m256 mdiff = _mm256_setzero_ps();
						for (int c = 0; c < guide_channels; c++)
						{
							__m256 msub = _mm256_sub_ps(mguide[c], _mm256_set1_ps(muPtr[c]));
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						}
						mdiff = _mm256_min_ps(margclip, mdiff);

						float* a_ptr = alpha[k].ptr<float>();
						_mm256_store_ps(a_ptr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

						_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
					}
					for (int k = 0; k < K; k++)
					{
						//_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_set1_ps(FLT_MIN), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
						_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(float(k)), 0)));
					}
				}
			}

			_mm_free(mk);
		}
		/*else if (method == 1)
		{
			__m256 margclip = _mm256_set1_ps(float(sigma_range * 6.0 * sigma_range * 6.0));
			float* im0 = vsrc[0].ptr<float>();
			float* im1 = vsrc[1].ptr<float>();
			float* im2 = vsrc[2].ptr<float>();
			for (int n = 0; n < img_size.area(); n += 8)
			{
				const __m256 mimage0 = _mm256_load_ps(im0 + n);
				const __m256 mimage1 = _mm256_load_ps(im1 + n);
				const __m256 mimage2 = _mm256_load_ps(im2 + n);
				__m256 malpha_sum = _mm256_setzero_ps();
				cv::AutoBuffer<__m256> mvalpha(K);
				for (int k = 0; k < K; k++)
				{
					const __m256 mcenter0 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
					const __m256 mcenter1 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
					const __m256 mcenter2 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);

					__m256 msub = _mm256_sub_ps(mimage0, mcenter0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(mimage1, mcenter1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(mimage2, mcenter2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					mdiff = _mm256_min_ps(margclip, mdiff);

					float* a_ptr = alpha[k].ptr<float>();
					_mm256_store_ps(a_ptr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
					//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian
					mvalpha[k] = malpha;
					malpha_sum = _mm256_add_ps(malpha_sum, malpha);
				}
				for (int k = 0; k < K; k++)
				{
					float* w_ptr = vecW[k].ptr<float>();
					//_mm256_store_ps(w_ptr + n, _mm256_blendv_ps(_mm256_set1_ps(FLT_MIN), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
					_mm256_store_ps(w_ptr + n, _mm256_div_ps(mvalpha[k], malpha_sum));
				}
			}
		}*/
	}

	//isUseLocalMu=false, without local mu
	void ClusteringHDKF_Interpolation2Single::computeIndex(const std::vector<cv::Mat>& guide, const std::vector<cv::Mat>& guideRes)
	{
		const Size size = img_size / downSampleImage;
		const int imsize = size.area();
		const int IMSIZE8 = imsize / 8;
		index.create(size, CV_32F);

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);

		if (guide_channels == 1)
		{
			__m256* gptr = (downSampleImage == 1) ? (__m256*)guide[0].ptr<float>() : (__m256*)guideRes[0].ptr<float>();
			__m256* idx = (__m256*)index.ptr<float>();

			AutoBuffer<__m256> mmu(K);
			for (int k = 0; k < K; k++)
			{
				mmu[k] = _mm256_set1_ps(mu.at<float>(k));
			}
			for (int n = 0; n < IMSIZE8; n++)
			{
				__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
				__m256 argment = _mm256_setzero_ps();

				for (int k = 0; k < K; k++)
				{
					const __m256 msub = _mm256_sub_ps(*gptr, mmu[k]);
					const __m256 mdiff = _mm256_mul_ps(msub, msub);

					_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
				}
				*idx++ = argment;
				gptr++;
			}
		}
		else if (guide_channels == 3)
		{
			const __m256* im0 = (downSampleImage == 1) ? (__m256*)guide[0].ptr<float>() : (__m256*)guideRes[0].ptr<float>();
			const __m256* im1 = (downSampleImage == 1) ? (__m256*)guide[1].ptr<float>() : (__m256*)guideRes[1].ptr<float>();
			const __m256* im2 = (downSampleImage == 1) ? (__m256*)guide[2].ptr<float>() : (__m256*)guideRes[2].ptr<float>();
			__m256* idx = (__m256*)index.ptr<float>();

			AutoBuffer<__m256> mc0(K);
			AutoBuffer<__m256> mc1(K);
			AutoBuffer<__m256> mc2(K);
			for (int k = 0; k < K; k++)
			{
				mc0[k] = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
				mc1[k] = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
				mc2[k] = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);
			}

			constexpr int method = 0;
			if constexpr (method == 0)
			{
				AutoBuffer<float*> wptr(K);
				for (int k = 0; k < K; k++)wptr[k] = wmap[k].ptr<float>();

				for (int n = 0; n < IMSIZE8; n++)
				{
					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_setzero_ps();

					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
					}

					mdiffmax = _mm256_setzero_ps();
					__m256 msum = _mm256_setzero_ps();
					__m256 mdiffmin = _mm256_set1_ps(FLT_MAX);
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						mdiffmax = _mm256_max_ps(mdiffmax, mdiff);
						mdiffmin = _mm256_min_ps(mdiffmin, mdiff);

						mdiff = _mm256_div_ps(_mm256_set1_ps(1), mdiff);
						msum = _mm256_add_ps(mdiff, msum);
						//_mm256_store_ps(wptr[k] + 8 * n, mdiff);
					}

					__m256 mone = _mm256_set1_ps(-1.f);
					const float s = 10.f;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						mdiff = _mm256_sub_ps(mdiff, mdiffmin);
						/*mdiff = _mm256_div_ps(_mm256_sub_ps(mdiffmax,mdiff), mdiffmax);
						mdiff = _mm256_mul_ps(mdiff, mdiff);
						mdiff = _mm256_mul_ps(mdiff, mdiff);
						mdiff = _mm256_mul_ps(mdiff, mdiff);
						mdiff = _mm256_mul_ps(mdiff, mdiff);
						mdiff = _mm256_mul_ps(mdiff, mdiff); mdiff = _mm256_mul_ps(mdiff, mdiff);*/
						mdiff = _mm256_exp_ps(_mm256_mul_ps(mdiff, _mm256_set1_ps(-0.5f / (s * s))));
						
						//__m256 v = _mm256_andnot_ps(mone, _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0));
						/*cout << k << endl;
						print_m256(mdiff);
						print_m256(v);
						getchar();*/
						//mdiff = _mm256_div_ps(_mm256_set1_ps(1), mdiff);
						_mm256_store_ps(wptr[k] + 8 * n, mdiff);
						///_mm256_store_ps(wptr[k] + 8 * n, _mm256_div_ps(_mm256_mul_ps(v,mdiff), msum));
						//_mm256_store_ps(wptr[k] + 8 * n, _mm256_sub_ps(mdiffmax, mdiff));
						//_mm256_store_ps(wptr[k] + 8 * n, _mm256_div_ps(_mm256_sub_ps(mdiffmax, mdiff), msum));
					}

					im0++; im1++; im2++;
				}
				/*
				for (int n = 0; n < IMSIZE8; n++)
				{
					__m256 mdiffmax = _mm256_set1_ps(0);
					__m256 msum = _mm256_setzero_ps();
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						mdiffmax = _mm256_max_ps(mdiffmax, mdiff);

						mdiff = _mm256_div_ps(_mm256_set1_ps(1), mdiff);
						msum = _mm256_add_ps(mdiff, msum);
						//_mm256_store_ps(wptr[k] + 8 * n, mdiff);
					}
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						mdiff = _mm256_div_ps(_mm256_set1_ps(1), mdiff);
						_mm256_store_ps(wptr[k] + 8 * n, _mm256_div_ps(mdiff, msum));
						//_mm256_store_ps(wptr[k] + 8 * n, _mm256_sub_ps(mdiffmax, mdiff));
						//_mm256_store_ps(wptr[k] + 8 * n, _mm256_div_ps(_mm256_sub_ps(mdiffmax, mdiff), msum));
					}
					im0++; im1++; im2++;
				}
				*/
			}
			else if constexpr (method == 1)
			{
				AutoBuffer<float*> wptr(K);
				for (int k = 0; k < K; k++)wptr[k] = wmap[k].ptr<float>();

				for (int n = 0; n < IMSIZE8; n++)
				{
					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_setzero_ps();
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
					}

					__m256 mone = _mm256_set1_ps(-0.1f);
					for (int k = 0; k < K; k++)
					{
						__m256 v = _mm256_andnot_ps(mone, _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0));
						_mm256_store_ps(wptr[k] + 8 * n, v);
					}

					im0++; im1++; im2++;
				}
			}
			else
			{
				for (int n = 0; n < IMSIZE8; n++)
				{
					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_setzero_ps();
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(*im0, mc0[k]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1, mc1[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2, mc2[k]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
					}
					*idx++ = argment;
					im0++; im1++; im2++;
				}
			}
		}
		else //n-dimensional signal
		{
			cv::AutoBuffer<__m256*> gptr(guide_channels);
			for (int c = 0; c < guide_channels; c++) gptr[c] = (downSampleImage == 1) ? (__m256*)guide[c].ptr<float>() : (__m256*)guideRes[c].ptr<float>();
			__m256* idx = (__m256*)index.ptr<float>();

			AutoBuffer<__m256*> mmu(K);
			for (int k = 0; k < K; k++)
			{
				mmu[k] = (__m256*)_mm_malloc(guide_channels * sizeof(__m256), AVX_ALIGN);
				const float* muPtr = mu.ptr<float>(k);
				for (int c = 0; c < guide_channels; c++)
				{
					mmu[k][c] = _mm256_set1_ps(muPtr[c]);
				}
			}

			for (int n = 0; n < IMSIZE8; n++)
			{
				__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
				__m256 argment = _mm256_setzero_ps();

				for (int k = 0; k < K; k++)
				{
					__m256 mdiff;
					{
						const __m256 msub = _mm256_sub_ps(*gptr[0], mmu[k][0]);
						mdiff = _mm256_mul_ps(msub, msub);
					}
					for (int c = 1; c < guide_channels; c++)
					{
						const __m256 msub = _mm256_sub_ps(*gptr[c], mmu[k][c]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}

					_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
				}
				*idx++ = argment;
				for (int c = 0; c < guide_channels; c++)gptr[c]++;
			}

			for (int k = 0; k < K; k++)
			{
				_mm_free(mmu[k]);
			}
		}
	}

	//isUseLocalMu=false: guide and mu for alpha
	template<int use_fmath>
	void ClusteringHDKF_Interpolation2Single::computeAlpha(const std::vector<cv::Mat>& guide, const int k)
	{
		bool isROI = true;

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);
		const int imsize = img_size.area() / 8;
		if (guide_channels == 1)
		{
			__m256* gptr = (__m256*)guide[0].ptr<float>();
			__m256* a_ptr = (__m256*)alpha[0].ptr<float>();
			const float* muPtr = mu.ptr<float>(k);
			__m256 mmu = _mm256_set1_ps(muPtr[0]);

			for (int n = 0; n < imsize; n++)
			{
				__m256 msub = _mm256_sub_ps(*gptr++, mmu);
				__m256 mdiff = _mm256_mul_ps(msub, msub);

				*a_ptr++ = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
			}
		}
		else if (guide_channels == 2)
		{
			const __m256* im0 = (__m256*)guide[0].ptr<float>();
			const __m256* im1 = (__m256*)guide[1].ptr<float>();
			__m256* aptr = (__m256*)alpha[0].ptr<float>();

			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec2f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec2f>(k)[1]);

			for (int n = 0; n < imsize; n++)
			{
				__m256 msub = _mm256_sub_ps(*im0++, mc0);
				__m256 mdiff = _mm256_mul_ps(msub, msub);
				msub = _mm256_sub_ps(*im1++, mc1);
				mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

				*aptr++ = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
			}
		}
		else if (guide_channels == 3)
		{
			if (isROI)
			{
				const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
				const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
				const __m256 mc2 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);

				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
					const __m256* im1 = (__m256*)guide[1].ptr<float>(y, boundaryLength);
					const __m256* im2 = (__m256*)guide[2].ptr<float>(y, boundaryLength);
					__m256* aptr = (__m256*)alpha[0].ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						__m256 msub = _mm256_sub_ps(*im0++, mc0);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(*im1++, mc1);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(*im2++, mc2);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						*aptr++ = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					}
				}
			}
			else
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>();
				const __m256* im1 = (__m256*)guide[1].ptr<float>();
				const __m256* im2 = (__m256*)guide[2].ptr<float>();
				__m256* aptr = (__m256*)alpha[0].ptr<float>();

				const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
				const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
				const __m256 mc2 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);

				for (int n = 0; n < imsize; n++)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*im1++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*im2++, mc2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					*aptr++ = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
				}
			}
		}
		else //n-dimensional signal
		{
			cv::AutoBuffer<__m256*> gptr(guide_channels);
			cv::AutoBuffer<__m256> mguide(guide_channels);
			for (int c = 0; c < guide_channels; c++) gptr[c] = (__m256*)guide[c].ptr<float>();
			__m256* a_ptr = (__m256*)alpha[0].ptr<float>();
			const float* muPtr = mu.ptr<float>(k);
			AutoBuffer<__m256> mmu(guide_channels);
			for (int c = 0; c < guide_channels; c++)
			{
				mmu[c] = _mm256_set1_ps(muPtr[c]);
			}

			for (int n = 0; n < imsize; n++)
			{
				__m256 mdiff;
				{
					__m256 msub = _mm256_sub_ps(*gptr[0]++, mmu[0]);
					mdiff = _mm256_mul_ps(msub, msub);
				}
				for (int c = 1; c < guide_channels; c++)
				{
					__m256 msub = _mm256_sub_ps(*gptr[c]++, mmu[c]);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
				}

				*a_ptr++ = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
			}
		}
	}

	template<int use_fmath>
	void ClusteringHDKF_Interpolation2Single::computeW(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& vecW)
	{
		const int size = src[0].size().area();
		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);
		//const __m256 mlambda = _mm256_set1_ps(-lambda);
		cv::AutoBuffer<float*> w_ptr(K);
		for (int k = 0; k < K; k++)
		{
			w_ptr[k] = vecW[k].ptr<float>();
		}

		const int method = 0;//nk-loop-hard
		//const int method = 1;//nk-loop-soft
		if (method == 0)
		{
			__m256 margclip = _mm256_set1_ps(float(sigma_range * 6.0 * sigma_range * 6.0));
			const float* im0 = src[0].ptr<float>();
			const float* im1 = src[1].ptr<float>();
			const float* im2 = src[2].ptr<float>();

			if (guide_channels == 3)
			{
				for (int n = 0; n < size; n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);

					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_setzero_ps();

					for (int k = 0; k < K; k++)
					{
						const float* muPtr = mu.ptr<float>(k);
						const __m256 mcenter0 = _mm256_set1_ps(muPtr[0]);
						const __m256 mcenter1 = _mm256_set1_ps(muPtr[1]);
						const __m256 mcenter2 = _mm256_set1_ps(muPtr[2]);

						__m256 msub = _mm256_sub_ps(mimage0, mcenter0);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mcenter1);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage2, mcenter2);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_argmin_ps(mdiff, mdiffmax, argment, float(k));
					}
					for (int k = 0; k < K; k++)
					{
						//_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_set1_ps(FLT_MIN), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
						//_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
						_mm256_store_ps(w_ptr[k] + n, _mm256_andnot_ps(_mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(float(k)), 0)));
					}
				}
			}
			else
			{
				for (int n = 0; n < size; n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);

					__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
					__m256 argment = _mm256_set1_ps(0.f);

					for (int k = 0; k < K; k++)
					{
						const float* muPtr = mu.ptr<float>(k);
						__m256 mdiff = _mm256_setzero_ps();
						for (int c = 0; c < guide_channels; c++)
						{
							__m256 msub = _mm256_sub_ps(mimage0, _mm256_set1_ps(muPtr[c]));
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						}
						_mm256_argmin_ps(mdiff, mdiffmax, argment, (float)k);
					}
					for (int k = 0; k < K; k++)
					{
						//_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_set1_ps(FLT_MIN), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
						_mm256_store_ps(w_ptr[k] + n, _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(float(k)), 0)));
					}
				}
			}
		}
	}


	void ClusteringHDKF_Interpolation2Single::split_blur(const int k, const bool isUseFmath, const bool isUseLSP)
	{
		const Size size = img_size / downSampleImage;
		const int imsize = size.area();
		const int IMSIZE8 = imsize / 8;

		if (isUsePrecomputedWforeachK)
		{
			float* vecw_ptr = vecW[k].ptr<float>();

			if (isUseLSP)
			{
				float* mask_ptr = blendLSPMask.ptr<float>();

				float* src0 = nullptr;
				float* src1 = nullptr;
				float* src2 = nullptr;
				src0 = vsrc[0].ptr<float>();//for gray and color
				if (channels == 3)//for color
				{
					src1 = vsrc[1].ptr<float>();
					src2 = vsrc[2].ptr<float>();
				}

				float* inter0 = nullptr;
				float* inter1 = nullptr;
				float* inter2 = nullptr;
				inter0 = split_inter[0].ptr<float>();//for gray and color
				if (channels == 3)//for color
				{
					inter1 = split_inter[1].ptr<float>();
					inter2 = split_inter[2].ptr<float>();
				}

				//split
				for (int n = 0; n < img_size.area(); n += 8)
				{
					__m256 mvecw = _mm256_load_ps(vecw_ptr + n);
					__m256 msrc0 = _mm256_load_ps(src0 + n);
					__m256 msrc1 = _mm256_load_ps(src1 + n);
					__m256 msrc2 = _mm256_load_ps(src2 + n);

					_mm256_store_ps(inter0 + n, _mm256_mul_ps(mvecw, msrc0));
					_mm256_store_ps(inter1 + n, _mm256_mul_ps(mvecw, msrc1));
					_mm256_store_ps(inter2 + n, _mm256_mul_ps(mvecw, msrc2));
					_mm256_store_ps(mask_ptr + n, _mm256_cmp_ps(mvecw, _mm256_setzero_ps(), 4));//4: NEQ
				}

				//blur
				GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
				GF->filter(split_inter[0], split_inter[0], sigma_space, spatial_order);
				GF->filter(split_inter[1], split_inter[1], sigma_space, spatial_order);
				GF->filter(split_inter[2], split_inter[2], sigma_space, spatial_order);

				//bilateralFilterLocalStatisticsPriorInternal(vsrc, split_inter, vecW[k], sigma_range, sigma_space, delta, blendLSPMask, BFLSPSchedule::LUTSQRT, &lut_bflsp[0]);
				bilateralFilterLocalStatisticsPriorInternal(vsrc, vecW[k], split_inter, (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
			}
			else
			{
				if (channels == 1)
				{
					float* src0 = nullptr;
					src0 = vsrc[0].ptr<float>();//for gray and color

					float* inter0 = nullptr;
					inter0 = split_inter[0].ptr<float>();//for gray and color

					for (int n = 0; n < imsize; n += 8)
					{
						__m256 mvecw = _mm256_load_ps(vecw_ptr + n);
						_mm256_store_ps(inter0 + n, _mm256_mul_ps(mvecw, _mm256_load_ps(src0 + n)));
					}

					GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
					GF->filter(split_inter[0], split_inter[0], sigma_space, spatial_order);
				}
				else if (channels == 3)
				{
					__m256* idx = (__m256*)index.ptr<float>();
					__m256* vecw_ptr = (__m256*)vecW[k].ptr<float>();
					__m256* src0 = (downSampleImage == 1) ? (__m256*)vsrc[0].ptr<float>() : (__m256*)vsrcRes[0].ptr<float>();
					__m256* src1 = (downSampleImage == 1) ? (__m256*)vsrc[1].ptr<float>() : (__m256*)vsrcRes[1].ptr<float>();
					__m256* src2 = (downSampleImage == 1) ? (__m256*)vsrc[2].ptr<float>() : (__m256*)vsrcRes[2].ptr<float>();
					__m256* inter0 = (__m256*)split_inter[0].ptr<float>();
					__m256* inter1 = (__m256*)split_inter[1].ptr<float>();
					__m256* inter2 = (__m256*)split_inter[2].ptr<float>();

					isWRedunductLoadDecomposition = false;
					if (isWRedunductLoadDecomposition)
					{
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							*inter0++ = _mm256_mul_ps(mvecw, *src0++);
						}
						GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order);

						vecw_ptr = (__m256*)vecW[k].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							*inter1++ = _mm256_mul_ps(mvecw, *src1++);
						}
						GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order);

						vecw_ptr = (__m256*)vecW[k].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							*inter2++ = _mm256_mul_ps(mvecw, *src2++);
						}
						GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order);
						GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
					}
					else
					{
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							//const __m256 mvecw = _mm256_andnot_ps(_mm256_set1_ps(1.f), _mm256_cmp_ps(*idx++, _mm256_set1_ps(float(k)), 0));
							//*vecw_ptr++ = mvecw;
							*inter0++ = _mm256_mul_ps(mvecw, *src0++);
							*inter1++ = _mm256_mul_ps(mvecw, *src1++);
							*inter2++ = _mm256_mul_ps(mvecw, *src2++);
						}
						GF->filter(vecW[k], vecW[k], sigma_space / downSampleImage, spatial_order);
						GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order);
						GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order);
						GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order);
					}
				}
				else
				{
					vector<__m256*> src(channels);
					vector<__m256*> inter(channels);
					__m256* vecw = (__m256*)vecW[k].ptr<float>();
					for (int c = 0; c < channels; c++)
					{
						src[c] = (__m256*)vsrc[c].ptr<float>();
						inter[c] = (__m256*)split_inter[c].ptr<float>();
					}

					if (isWRedunductLoadDecomposition)
					{
						for (int c = 0; c < channels; c++)
						{
							vecw = (__m256*)vecW[k].ptr<float>();
							for (int n = 0; n < img_size.area(); n += 8)
							{
								*inter[c]++ = _mm256_mul_ps(*vecw++, *src[c]++);
							}
							GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order);
						}
					}
					else
					{
						for (int n = 0; n < img_size.area(); n += 8)
						{
							const __m256 mvecw = *vecw++;
							for (int c = 0; c < channels; c++)
							{
								*inter[c]++ = _mm256_mul_ps(mvecw, *src[c]++);
							}
						}
						for (int c = 0; c < channels; c++)
						{
							GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order);
						}
					}
					GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
				}
			}
		}
		else
		{
			//std::cout<<"here"<<std::endl;
			const bool ch1 = true;
			const bool ch3 = true;
			//const bool ch1 = false;
			//const bool ch3 = false;
			const __m256 mone = _mm256_set1_ps(1.f);
			const __m256 mk = _mm256_set1_ps(float(k));
			if (channels == 1 && ch1)
			{
				__m256* idx = (__m256*)index.ptr<float>();
				__m256* vecw = (__m256*)vecW[0].ptr<float>();
				__m256* src0 = (downSampleImage == 1) ? (__m256*)vsrc[0].ptr<float>() : (__m256*)vsrcRes[0].ptr<float>();
				__m256* inter0 = (__m256*)split_inter[0].ptr<float>();
				for (int n = 0; n < IMSIZE8; n++)
				{
					const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
					*vecw++ = mvecw;
					*inter0++ = _mm256_mul_ps(mvecw, *src0++);
				}
				GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
				GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);
			}
			else if (channels == 3 && ch3)
			{
				__m256* idx = (__m256*)index.ptr<float>();
				__m256* vecw = (__m256*)vecW[0].ptr<float>();
				const __m256* src0 = (downSampleImage == 1) ? (__m256*)vsrc[0].ptr<float>() : (__m256*)vsrcRes[0].ptr<float>();
				const __m256* src1 = (downSampleImage == 1) ? (__m256*)vsrc[1].ptr<float>() : (__m256*)vsrcRes[1].ptr<float>();
				const __m256* src2 = (downSampleImage == 1) ? (__m256*)vsrc[2].ptr<float>() : (__m256*)vsrcRes[2].ptr<float>();
				__m256* inter0 = (__m256*)split_inter[0].ptr<float>();
				__m256* inter1 = (__m256*)split_inter[1].ptr<float>();
				__m256* inter2 = (__m256*)split_inter[2].ptr<float>();

				//isWRedunductLoadDecomposition = false;
				if (isWRedunductLoadDecomposition)
				{
					for (int n = 0; n < IMSIZE8; n++)
					{
						const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));

						*vecw++ = mvecw;
						*inter0++ = _mm256_mul_ps(mvecw, *src0++);
					}
					GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);

					vecw = (__m256*)vecW[0].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						*inter1++ = _mm256_mul_ps(*vecw++, *src1++);
					}
					GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order, borderType);

					vecw = (__m256*)vecW[0].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						*inter2++ = _mm256_mul_ps(*vecw++, *src2++);
					}
					GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order, borderType);

					if (isUseLocalStatisticsPrior)
					{
						vecW[0].copyTo(blendLSPMask);
						GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
						if (downSampleImage == 1)
						{
							bilateralFilterLocalStatisticsPriorInternal(vsrc, vecW[0], split_inter, (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
						}
						else
						{
							bilateralFilterLocalStatisticsPriorInternal(vsrcRes, vecW[0], split_inter, (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
						}
						//bilateralFilterLocalStatisticsPriorInternal(vguide, split_inter, vecW[0], (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
					}
					else
					{
						GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
					}
				}
				else
				{
					for (int n = 0; n < IMSIZE8; n++)
					{
						const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
						*vecw++ = mvecw;
						*inter0++ = _mm256_mul_ps(mvecw, *src0++);
						*inter1++ = _mm256_mul_ps(mvecw, *src1++);
						*inter2++ = _mm256_mul_ps(mvecw, *src2++);
					}
					GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
					GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);
					GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order, borderType);
					GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order, borderType);
				}
			}
			else
			{
				__m256* idx = (__m256*)index.ptr<float>();
				__m256* vecw = (__m256*)vecW[0].ptr<float>();
				vector<__m256*> src(channels);
				vector<__m256*> inter(channels);
				for (int c = 0; c < channels; c++)
				{
					src[c] = (downSampleImage == 1) ? (__m256*)vsrc[c].ptr<float>() : (__m256*)vsrcRes[c].ptr<float>();
					inter[c] = (__m256*)split_inter[c].ptr<float>();
				}

				//isWRedunductLoadDecomposition = false;
				if (isWRedunductLoadDecomposition)
				{
					//c=0
					{
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
							*vecw++ = mvecw;
							*inter[0]++ = _mm256_mul_ps(mvecw, *src[0]++);
						}
						GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);
					}
					for (int c = 1; c < channels; c++)
					{
						vecw = (__m256*)vecW[0].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							*inter[c]++ = _mm256_mul_ps(*vecw++, *src[c]++);
						}
						GF->filter(split_inter[c], split_inter[c], sigma_space / downSampleImage, spatial_order, borderType);
					}

					if (isUseLocalStatisticsPrior)
					{
						vecW[0].copyTo(blendLSPMask);
						GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
						bilateralFilterLocalStatisticsPriorInternal(vsrc, vecW[0], split_inter, (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
					}
					else
					{
						GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
					}
				}
				else
				{
					for (int n = 0; n < IMSIZE8; n++)
					{
						const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
						*vecw++ = mvecw;
						for (int c = 0; c < channels; c++)
						{
							*inter[c]++ = _mm256_mul_ps(mvecw, *src[c]++);
						}
					}
					for (int c = 0; c < channels; c++)
					{
						GF->filter(split_inter[c], split_inter[c], sigma_space / downSampleImage, spatial_order, borderType);
					}
					GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
				}
			}
		}
	}

	template<int channels>
	void ClusteringHDKF_Interpolation2Single::split_blur(const int k, const bool isUseFmath, const bool isUseLSP)
	{
		const Size size = img_size / downSampleImage;
		const int imsize = size.area();
		const int IMSIZE8 = imsize / 8;

		if (isUsePrecomputedWforeachK)
		{
			float* vecw_ptr = vecW[k].ptr<float>();

			if (isUseLSP)
			{
				float* mask_ptr = blendLSPMask.ptr<float>();

				float* src0 = nullptr;
				float* src1 = nullptr;
				float* src2 = nullptr;
				src0 = vsrc[0].ptr<float>();//for gray and color
				if (channels == 3)//for color
				{
					src1 = vsrc[1].ptr<float>();
					src2 = vsrc[2].ptr<float>();
				}

				float* inter0 = nullptr;
				float* inter1 = nullptr;
				float* inter2 = nullptr;
				inter0 = split_inter[0].ptr<float>();//for gray and color
				if (channels == 3)//for color
				{
					inter1 = split_inter[1].ptr<float>();
					inter2 = split_inter[2].ptr<float>();
				}

				//split
				for (int n = 0; n < img_size.area(); n += 8)
				{
					__m256 mvecw = _mm256_load_ps(vecw_ptr + n);
					__m256 msrc0 = _mm256_load_ps(src0 + n);
					__m256 msrc1 = _mm256_load_ps(src1 + n);
					__m256 msrc2 = _mm256_load_ps(src2 + n);

					_mm256_store_ps(inter0 + n, _mm256_mul_ps(mvecw, msrc0));
					_mm256_store_ps(inter1 + n, _mm256_mul_ps(mvecw, msrc1));
					_mm256_store_ps(inter2 + n, _mm256_mul_ps(mvecw, msrc2));
					_mm256_store_ps(mask_ptr + n, _mm256_cmp_ps(mvecw, _mm256_setzero_ps(), 4));//4: NEQ
				}

				//blur
				GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
				GF->filter(split_inter[0], split_inter[0], sigma_space, spatial_order);
				GF->filter(split_inter[1], split_inter[1], sigma_space, spatial_order);
				GF->filter(split_inter[2], split_inter[2], sigma_space, spatial_order);

				//bilateralFilterLocalStatisticsPriorInternal(vsrc, split_inter, vecW[k], sigma_range, sigma_space, delta, blendLSPMask, BFLSPSchedule::LUTSQRT, &lut_bflsp[0]);
				bilateralFilterLocalStatisticsPriorInternal(vsrc, vecW[k], split_inter, (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
			}
			else
			{
				if (channels == 1)
				{
					float* src0 = nullptr;
					src0 = vsrc[0].ptr<float>();//for gray and color


					float* inter0 = nullptr;
					inter0 = split_inter[0].ptr<float>();//for gray and color

					for (int n = 0; n < imsize; n += 8)
					{
						__m256 mvecw = _mm256_load_ps(vecw_ptr + n);
						_mm256_store_ps(inter0 + n, _mm256_mul_ps(mvecw, _mm256_load_ps(src0 + n)));
					}

					GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
					GF->filter(split_inter[0], split_inter[0], sigma_space, spatial_order);
				}
				else if (channels == 3)
				{
					__m256* idx = (__m256*)index.ptr<float>();
					__m256* vecw_ptr = (__m256*)vecW[k].ptr<float>();
					__m256* src0 = (downSampleImage == 1) ? (__m256*)vsrc[0].ptr<float>() : (__m256*)vsrcRes[0].ptr<float>();
					__m256* src1 = (downSampleImage == 1) ? (__m256*)vsrc[1].ptr<float>() : (__m256*)vsrcRes[1].ptr<float>();
					__m256* src2 = (downSampleImage == 1) ? (__m256*)vsrc[2].ptr<float>() : (__m256*)vsrcRes[2].ptr<float>();
					__m256* inter0 = (__m256*)split_inter[0].ptr<float>();
					__m256* inter1 = (__m256*)split_inter[1].ptr<float>();
					__m256* inter2 = (__m256*)split_inter[2].ptr<float>();

					isWRedunductLoadDecomposition = false;
					if (isWRedunductLoadDecomposition)
					{
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							*inter0++ = _mm256_mul_ps(mvecw, *src0++);
						}
						GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order);

						vecw_ptr = (__m256*)vecW[k].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							*inter1++ = _mm256_mul_ps(mvecw, *src1++);
						}
						GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order);

						vecw_ptr = (__m256*)vecW[k].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							*inter2++ = _mm256_mul_ps(mvecw, *src2++);
						}
						GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order);
						GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
					}
					else
					{
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = *vecw_ptr++;
							//const __m256 mvecw = _mm256_andnot_ps(_mm256_set1_ps(1.f), _mm256_cmp_ps(*idx++, _mm256_set1_ps(float(k)), 0));
							//*vecw_ptr++ = mvecw;
							*inter0++ = _mm256_mul_ps(mvecw, *src0++);
							*inter1++ = _mm256_mul_ps(mvecw, *src1++);
							*inter2++ = _mm256_mul_ps(mvecw, *src2++);
						}
						GF->filter(vecW[k], vecW[k], sigma_space / downSampleImage, spatial_order);
						GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order);
						GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order);
						GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order);
					}
				}
				else
				{
					vector<__m256*> src(channels);
					vector<__m256*> inter(channels);
					__m256* vecw = (__m256*)vecW[k].ptr<float>();
					for (int c = 0; c < channels; c++)
					{
						src[c] = (__m256*)vsrc[c].ptr<float>();
						inter[c] = (__m256*)split_inter[c].ptr<float>();
					}

					if (isWRedunductLoadDecomposition)
					{
						for (int c = 0; c < channels; c++)
						{
							vecw = (__m256*)vecW[k].ptr<float>();
							for (int n = 0; n < img_size.area(); n += 8)
							{
								*inter[c]++ = _mm256_mul_ps(*vecw++, *src[c]++);
							}
							GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order);
						}
					}
					else
					{
						for (int n = 0; n < img_size.area(); n += 8)
						{
							const __m256 mvecw = *vecw++;
							for (int c = 0; c < channels; c++)
							{
								*inter[c]++ = _mm256_mul_ps(mvecw, *src[c]++);
							}
						}
						for (int c = 0; c < channels; c++)
						{
							GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order);
						}
					}
					GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
				}
			}
		}
		else
		{
			//std::cout<<"here"<<std::endl;
			const bool ch1 = true;
			const bool ch3 = true;
			//const bool ch1 = false;
			//const bool ch3 = false;
			const __m256 mone = _mm256_set1_ps(1.f);
			const __m256 mk = _mm256_set1_ps(float(k));
			if (channels == 1 && ch1)
			{
				__m256* idx = (__m256*)index.ptr<float>();
				__m256* vecw = (__m256*)vecW[0].ptr<float>();
				__m256* src0 = (downSampleImage == 1) ? (__m256*)vsrc[0].ptr<float>() : (__m256*)vsrcRes[0].ptr<float>();
				__m256* inter0 = (__m256*)split_inter[0].ptr<float>();
				for (int n = 0; n < IMSIZE8; n++)
				{
					const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
					*vecw++ = mvecw;
					*inter0++ = _mm256_mul_ps(mvecw, *src0++);
				}
				GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
				GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);
			}
			else if (channels == 3 && ch3)
			{
				//std::cout << "here 3ch" << std::endl;
				//const __m256* idx = (__m256*)index.ptr<float>();
				const __m256* idx = (__m256*)wmap[k].ptr<float>();
				const __m256* src0 = (downSampleImage == 1) ? (__m256*)vsrc[0].ptr<float>() : (__m256*)vsrcRes[0].ptr<float>();
				const __m256* src1 = (downSampleImage == 1) ? (__m256*)vsrc[1].ptr<float>() : (__m256*)vsrcRes[1].ptr<float>();
				const __m256* src2 = (downSampleImage == 1) ? (__m256*)vsrc[2].ptr<float>() : (__m256*)vsrcRes[2].ptr<float>();
				const __m256* guide0 = (downSampleImage == 1) ? (__m256*)vguide[0].ptr<float>() : (__m256*)vguideRes[0].ptr<float>();
				const __m256* guide1 = (downSampleImage == 1) ? (__m256*)vguide[1].ptr<float>() : (__m256*)vguideRes[1].ptr<float>();
				const __m256* guide2 = (downSampleImage == 1) ? (__m256*)vguide[2].ptr<float>() : (__m256*)vguideRes[2].ptr<float>();

				__m256* vecw = (__m256*)vecW[0].ptr<float>();
				__m256* inter0 = (__m256*)split_inter[0].ptr<float>();
				__m256* inter1 = (__m256*)split_inter[1].ptr<float>();
				__m256* inter2 = (__m256*)split_inter[2].ptr<float>();

				//isWRedunductLoadDecomposition = false;
				if (isWRedunductLoadDecomposition)
				{
					for (int n = 0; n < IMSIZE8; n++)
					{
						/*if constexpr (true)
						{
							//test for soft assign (not good)
							const __m256 mlambda = _mm256_set1_ps(-1.0 / (1000.0));

							const float* muPtr = mu.ptr<float>(k);
							__m256 mc0 = _mm256_set1_ps(muPtr[0]);
							__m256 mc1 = _mm256_set1_ps(muPtr[1]);
							__m256 mc2 = _mm256_set1_ps(muPtr[2]);
							__m256 msub = _mm256_sub_ps(guide0[n], mc0);
							__m256 mdiff = _mm256_mul_ps(msub, msub);
							msub = _mm256_sub_ps(guide1[n], mc1);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							msub = _mm256_sub_ps(guide2[n], mc2);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							const __m256 mvecw2 = v_exp_ps<1>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));
							const __m256 mvecw = v_exp_ps<1>(_mm256_mul_ps(mlambda, mdiff));

							*vecw++ = mvecw;
							*inter0++ = _mm256_mul_ps(mvecw, src0[n]);
						}
						else*/
						{
							//const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
							const __m256 mvecw = *idx++;
							*vecw++ = mvecw;
							*inter0++ = _mm256_mul_ps(mvecw, *src0++);
						}
					}
					GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);

					vecw = (__m256*)vecW[0].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						*inter1++ = _mm256_mul_ps(*vecw++, *src1++);
					}
					GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order, borderType);

					vecw = (__m256*)vecW[0].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						*inter2++ = _mm256_mul_ps(*vecw++, *src2++);
					}
					GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order, borderType);

					if (isUseLocalStatisticsPrior)
					{
						vecW[0].copyTo(blendLSPMask);
						GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
						if (downSampleImage == 1)
						{
							bilateralFilterLocalStatisticsPriorInternal(vsrc, vecW[0], split_inter, (float)sigma_range, (float)sigma_space, delta, blendLSPMask, BFLSPSchedule::Compute);
						}
						else
						{
							bilateralFilterLocalStatisticsPriorInternal(vsrcRes, vecW[0], split_inter, (float)sigma_range, (float)sigma_space / downSampleImage, delta, blendLSPMask, BFLSPSchedule::Compute);
						}
					}
					else
					{
						GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
					}
				}
				else
				{
					for (int n = 0; n < IMSIZE8; n++)
					{
						const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
						*vecw++ = mvecw;
						*inter0++ = _mm256_mul_ps(mvecw, *src0++);
						*inter1++ = _mm256_mul_ps(mvecw, *src1++);
						*inter2++ = _mm256_mul_ps(mvecw, *src2++);
					}
					GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
					GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);
					GF->filter(split_inter[1], split_inter[1], sigma_space / downSampleImage, spatial_order, borderType);
					GF->filter(split_inter[2], split_inter[2], sigma_space / downSampleImage, spatial_order, borderType);
				}
			}
			else
			{
				__m256* idx = (__m256*)index.ptr<float>();
				__m256* vecw = (__m256*)vecW[0].ptr<float>();
				vector<__m256*> src(channels);
				vector<__m256*> inter(channels);
				for (int c = 0; c < channels; c++)
				{
					src[c] = (downSampleImage == 1) ? (__m256*)vsrc[c].ptr<float>() : (__m256*)vsrcRes[c].ptr<float>();
					inter[c] = (__m256*)split_inter[c].ptr<float>();
				}

				//isWRedunductLoadDecomposition = false;
				if (isWRedunductLoadDecomposition)
				{
					//c=0
					{
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
							*vecw++ = mvecw;
							*inter[0]++ = _mm256_mul_ps(mvecw, *src[0]++);
						}
						GF->filter(split_inter[0], split_inter[0], sigma_space / downSampleImage, spatial_order, borderType);
					}
					for (int c = 1; c < channels; c++)
					{
						vecw = (__m256*)vecW[0].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							*inter[c]++ = _mm256_mul_ps(*vecw++, *src[c]++);
						}
						GF->filter(split_inter[c], split_inter[c], sigma_space / downSampleImage, spatial_order, borderType);
					}
					GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
				}
				else
				{
					for (int n = 0; n < IMSIZE8; n++)
					{
						const __m256 mvecw = _mm256_andnot_ps(mone, _mm256_cmp_ps(*idx++, mk, 0));
						*vecw++ = mvecw;
						for (int c = 0; c < channels; c++)
						{
							*inter[c]++ = _mm256_mul_ps(mvecw, *src[c]++);
						}
					}
					for (int c = 0; c < channels; c++)
					{
						GF->filter(split_inter[c], split_inter[c], sigma_space / downSampleImage, spatial_order, borderType);
					}
					GF->filter(vecW[0], vecW[0], sigma_space / downSampleImage, spatial_order, borderType);
				}
			}
		}
	}

#pragma region merge
	void ClusteringHDKF_Interpolation2Single::mergeNumerDenomMat(vector<Mat>& dest, const int k, const int upsampleSize)
	{
		const int intermethod = INTER_LINEAR;
		//const int intermethod = INTER_CUBIC;
		const int wk = isWRedunductLoadDecomposition ? 0 : k;

		if (upsampleSize != 1)
		{
			for (int c = 0; c < channels; c++)
			{
				resize(split_inter[c], NumerDenomRes[c], Size(), downSampleImage, downSampleImage, intermethod);
			}
			resize(vecW[wk], NumerDenomRes[channels], Size(), downSampleImage, downSampleImage, intermethod);
			//cp::upsampleLinear(split_inter[0], NumerDenomRes[0], downSampleImage);
			//cp::upsampleLinear(split_inter[1], NumerDenomRes[1], downSampleImage);
			//cp::upsampleLinear(split_inter[2], NumerDenomRes[2], downSampleImage);
			//cp::upsampleLinear(vecW[wk], NumerDenomRes[3], downSampleImage);

			for (int c = 0; c < channels; c++)
			{
				dest[c] = NumerDenomRes[c];
			}
			dest[channels] = NumerDenomRes[channels];
		}
		else
		{
			for (int c = 0; c < channels; c++)
			{
				dest[c] = split_inter[c];
			}
			dest[channels] = vecW[wk];
		}
	}

	template<int use_fmath, const bool isInit>
	void ClusteringHDKF_Interpolation2Single::mergeRecomputeAlphaForUsingMu(std::vector<cv::Mat>& src, const int k)
	{
		if (isJoint)
		{
			//std::cout << "ConstantTimeHDGF_InterpolationSingle::recomputeAlpha must be src==guide" << std::endl;
		//	CV_Assert(!isJoint);
		}

		const int w = src[0].cols;
		const int h = src[0].rows;

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);

		const int wk = isWRedunductLoadDecomposition ? 0 : k;

		vector<Mat> intermat(channels + 1);
		mergeNumerDenomMat(intermat, k, downSampleImage);

		if (channels == 3)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				const __m256* src0 = (const __m256*)src[0].ptr<float>(y, boundaryLength);
				const __m256* src1 = (const __m256*)src[1].ptr<float>(y, boundaryLength);
				const __m256* src2 = (const __m256*)src[2].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* inter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* inter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* interw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *interw);

					__m256 msub = _mm256_fnmadd_ps(*inter0, norm, *src0++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_fnmadd_ps(*inter1, norm, *src1++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_fnmadd_ps(*inter2, norm, *src2++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*numer1++ = _mm256_mul_ps(malpha, *inter1++);
						*numer2++ = _mm256_mul_ps(malpha, *inter2++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else
		{
			AutoBuffer<const __m256*> msrc(channels);
			AutoBuffer<const __m256*> minter(channels);
			AutoBuffer<__m256*> mnumer(channels);

			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < channels; c++) msrc[c] = (const __m256*)src[c].ptr<float>(y, boundaryLength);
				for (int c = 0; c < channels; c++) minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[channels].ptr<float>(y, boundaryLength);

				for (int c = 0; c < channels; c++)  mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);

					__m256 msub = _mm256_fnmadd_ps(*minter[0], norm, *msrc[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					for (int c = 1; c < channels; c++)
					{
						msub = _mm256_fnmadd_ps(*minter[c], norm, *msrc[c]++);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_mul_ps(malpha, *minter[c]++);
						}
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_fmadd_ps(malpha, *minter[c]++, *mnumer[c]);
						}
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
	}

	template<int use_fmath, const bool isInit>
	void ClusteringHDKF_Interpolation2Single::mergeRecomputeAlphaForUsingNLMMu(std::vector<cv::Mat>& src, const int k)
	{
		if (isJoint)
		{
			//std::cout << "ConstantTimeHDGF_InterpolationSingle::recomputeAlpha must be src==guide" << std::endl;
		//	CV_Assert(!isJoint);
		}

		const int w = src[0].cols;
		const int h = src[0].rows;

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);

		const int wk = isWRedunductLoadDecomposition ? 0 : k;

		vector<Mat> intermat(channels + 1);
		mergeNumerDenomMat(intermat, k, downSampleImage);

		if (channels == 3)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				const int step = src[0].cols;
				const float* src0 = src[0].ptr<float>(y, boundaryLength);
				const float* src1 = src[1].ptr<float>(y, boundaryLength);
				const float* src2 = src[2].ptr<float>(y, boundaryLength);
				const float* inter0 = intermat[0].ptr<float>(y, boundaryLength);
				const float* inter1 = intermat[1].ptr<float>(y, boundaryLength);
				const float* inter2 = intermat[2].ptr<float>(y, boundaryLength);
				const float* interw = intermat[3].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);
				const int r = 0;
				const __m256 mones = _mm256_set1_ps(1.f);
				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					__m256 mdiff = _mm256_setzero_ps();
					for (int v = -r; v <= r; v++)
					{
						for (int h = -r; h <= r; h++)
						{
							const int id = v * step + h;
							const __m256 norm = _mm256_div_avoidzerodiv_ps(mones, _mm256_loadu_ps(interw + id));
							//const __m256 norm = _mm256_div_zerodivzero_ps(mones, _mm256_loadu_ps(interw + id));
							//const __m256 norm = _mm256_div_ps(mones, _mm256_loadu_ps(interw + id));

							__m256 msub = _mm256_fnmadd_ps(_mm256_loadu_ps(inter0 + id), norm, _mm256_loadu_ps(src0 + id));
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							msub = _mm256_fnmadd_ps(_mm256_loadu_ps(inter1 + id), norm, _mm256_loadu_ps(src1 + id));
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							msub = _mm256_fnmadd_ps(_mm256_loadu_ps(inter2 + id), norm, _mm256_loadu_ps(src2 + id));
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						}
					}
					mdiff = _mm256_mul_ps(mdiff, _mm256_set1_ps(9.f));

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, _mm256_load_ps(inter0));
						*numer1++ = _mm256_mul_ps(malpha, _mm256_load_ps(inter1));
						*numer2++ = _mm256_mul_ps(malpha, _mm256_load_ps(inter2));
						*denom_++ = _mm256_mul_ps(malpha, _mm256_load_ps(interw));
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, _mm256_load_ps(inter0), *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, _mm256_load_ps(inter1), *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, _mm256_load_ps(inter2), *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, _mm256_load_ps(interw), *denom_);
					}

					src0 += 8;
					src1 += 8;
					src2 += 8;
					inter0 += 8;
					inter1 += 8;
					inter2 += 8;
					interw += 8;
				}
			}
		}
		else
		{
			AutoBuffer<const __m256*> msrc(channels);
			AutoBuffer<const __m256*> minter(channels);
			AutoBuffer<__m256*> mnumer(channels);

			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < channels; c++) msrc[c] = (const __m256*)src[c].ptr<float>(y, boundaryLength);
				for (int c = 0; c < channels; c++) minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[channels].ptr<float>(y, boundaryLength);

				for (int c = 0; c < channels; c++)  mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);

					__m256 msub = _mm256_fnmadd_ps(*minter[0], norm, *msrc[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					for (int c = 1; c < channels; c++)
					{
						msub = _mm256_fnmadd_ps(*minter[c], norm, *msrc[c]++);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_mul_ps(malpha, *minter[c]++);
						}
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_fmadd_ps(malpha, *minter[c]++, *mnumer[c]);
						}
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
	}

	template<int use_fmath, const bool isInit, int channels, int guide_channels>
	void ClusteringHDKF_Interpolation2Single::mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k)
	{
		const int w = guide[0].cols;
		const int h = guide[0].rows;

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);

		const int wk = isWRedunductLoadDecomposition ? 0 : k;
		AutoBuffer<AutoBuffer<__m256>> p(guide_channels);
		for (int c = 0; c < guide_channels; c++)
		{
			p.resize(channels);
			for (int cc = 0; cc < channels; cc++)
			{
				p[c][cc] = _mm256_set1_ps(projectionMatrix.at<float>(c, cc));
			}
		}

		vector<Mat> intermat(channels + 1);
		mergeNumerDenomMat(intermat, k, downSampleImage);

		AutoBuffer<const __m256*> mguide(guide_channels);
		AutoBuffer<const __m256*> minter(channels);
		AutoBuffer<__m256*> mnumer(channels);
		AutoBuffer<__m256> mpca(guide_channels);

		for (int y = boundaryLength; y < h - boundaryLength; y++)
		{
			for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);
			for (int c = 0; c < channels; c++) minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
			const __m256* minterw = (const __m256*)intermat[channels].ptr<float>(y, boundaryLength);
			for (int c = 0; c < channels; c++)  mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
			__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

			for (int x = boundaryLength; x < w - boundaryLength; x += 8)
			{
				const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);

				for (int c = 0; c < guide_channels; c++)
				{
					mpca[c] = _mm256_mul_ps(*minter[0], p[c][0]);
					for (int cc = 1; cc < channels; cc++)
					{
						mpca[c] = _mm256_fmadd_ps(*minter[cc], p[c][cc], mpca[c]);
					}
				}

				__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
				__m256 mdiff = _mm256_mul_ps(msub, msub);
				for (int c = 1; c < guide_channels; c++)
				{
					msub = _mm256_fnmadd_ps(mpca[c], norm, *mguide[c]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
				}

				const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

				if constexpr (isInit)
				{
					for (int c = 0; c < channels; c++)
					{
						*mnumer[c]++ = _mm256_mul_ps(malpha, *minter[c]++);
					}
					*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
				}
				else
				{
					for (int c = 0; c < channels; c++)
					{
						*mnumer[c]++ = _mm256_fmadd_ps(malpha, *minter[c]++, *mnumer[c]);
					}
					*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
				}
			}
		}
	}

	template<int use_fmath, const bool isInit>
	void ClusteringHDKF_Interpolation2Single::mergeRecomputeAlphaForUsingMuPCA(std::vector<cv::Mat>& guide, const int k)
	{
		const int w = guide[0].cols;
		const int h = guide[0].rows;

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);

		const int wk = isWRedunductLoadDecomposition ? 0 : k;
		AutoBuffer<__m256*> p(guide_channels);

		for (int c = 0; c < guide_channels; c++)
		{
			p[c] = (__m256*)_mm_malloc(sizeof(__m256) * channels, AVX_ALIGN);
			for (int cc = 0; cc < channels; cc++)
			{
				p[c][cc] = _mm256_set1_ps(projectionMatrix.at<float>(c, cc));
			}
		}

		vector<Mat> intermat(channels + 1);
		mergeNumerDenomMat(intermat, k, downSampleImage);

		AutoBuffer<const __m256*> mguide(guide_channels);
		AutoBuffer<const __m256*> minter(channels);
		AutoBuffer<__m256*> mnumer(channels);
		AutoBuffer<__m256> mpca(guide_channels);

		if (channels == 1 && guide_channels == 4)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mdenom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);
					mpca[0] = _mm256_mul_ps(*minter0, p[0][0]);
					mpca[1] = _mm256_mul_ps(*minter0, p[1][0]);
					mpca[2] = _mm256_mul_ps(*minter0, p[2][0]);
					mpca[3] = _mm256_mul_ps(*minter0, p[3][0]);

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_fnmadd_ps(mpca[1], norm, *mguide[1]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_fnmadd_ps(mpca[2], norm, *mguide[2]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_fnmadd_ps(mpca[3], norm, *mguide[3]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mdenom_++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mdenom_++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom_);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 1)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* minter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mnumer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* mnumer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);
					mpca[0] = _mm256_mul_ps(*minter0, p[0][0]);
					mpca[0] = _mm256_fmadd_ps(*minter1, p[0][1], mpca[0]);
					mpca[0] = _mm256_fmadd_ps(*minter2, p[0][2], mpca[0]);

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mnumer1++ = _mm256_mul_ps(malpha, *minter1++);
						*mnumer2++ = _mm256_mul_ps(malpha, *minter2++);
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mnumer1++ = _mm256_fmadd_ps(malpha, *minter1++, *mnumer1);
						*mnumer2++ = _mm256_fmadd_ps(malpha, *minter2++, *mnumer2);
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 2)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* minter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mnumer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* mnumer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);
					mpca[0] = _mm256_mul_ps(*minter0, p[0][0]);
					mpca[0] = _mm256_fmadd_ps(*minter1, p[0][1], mpca[0]);
					mpca[0] = _mm256_fmadd_ps(*minter2, p[0][2], mpca[0]);
					mpca[1] = _mm256_mul_ps(*minter0, p[1][0]);
					mpca[1] = _mm256_fmadd_ps(*minter1, p[1][1], mpca[1]);
					mpca[1] = _mm256_fmadd_ps(*minter2, p[1][2], mpca[1]);

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_fnmadd_ps(mpca[1], norm, *mguide[1]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mnumer1++ = _mm256_mul_ps(malpha, *minter1++);
						*mnumer2++ = _mm256_mul_ps(malpha, *minter2++);
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mnumer1++ = _mm256_fmadd_ps(malpha, *minter1++, *mnumer1);
						*mnumer2++ = _mm256_fmadd_ps(malpha, *minter2++, *mnumer2);
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 3)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* minter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mnumer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* mnumer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);
					mpca[0] = _mm256_mul_ps(*minter0, p[0][0]);
					mpca[0] = _mm256_fmadd_ps(*minter1, p[0][1], mpca[0]);
					mpca[0] = _mm256_fmadd_ps(*minter2, p[0][2], mpca[0]);
					mpca[1] = _mm256_mul_ps(*minter0, p[1][0]);
					mpca[1] = _mm256_fmadd_ps(*minter1, p[1][1], mpca[1]);
					mpca[1] = _mm256_fmadd_ps(*minter2, p[1][2], mpca[1]);
					mpca[2] = _mm256_mul_ps(*minter0, p[2][0]);
					mpca[2] = _mm256_fmadd_ps(*minter1, p[2][1], mpca[2]);
					mpca[2] = _mm256_fmadd_ps(*minter2, p[2][2], mpca[2]);

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_fnmadd_ps(mpca[1], norm, *mguide[1]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_fnmadd_ps(mpca[2], norm, *mguide[2]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mnumer1++ = _mm256_mul_ps(malpha, *minter1++);
						*mnumer2++ = _mm256_mul_ps(malpha, *minter2++);
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mnumer1++ = _mm256_fmadd_ps(malpha, *minter1++, *mnumer1);
						*mnumer2++ = _mm256_fmadd_ps(malpha, *minter2++, *mnumer2);
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 4)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* minter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mnumer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* mnumer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);
					mpca[0] = _mm256_mul_ps(*minter0, p[0][0]);
					mpca[0] = _mm256_fmadd_ps(*minter1, p[0][1], mpca[0]);
					mpca[0] = _mm256_fmadd_ps(*minter2, p[0][2], mpca[0]);
					mpca[1] = _mm256_mul_ps(*minter0, p[1][0]);
					mpca[1] = _mm256_fmadd_ps(*minter1, p[1][1], mpca[1]);
					mpca[1] = _mm256_fmadd_ps(*minter2, p[1][2], mpca[1]);
					mpca[2] = _mm256_mul_ps(*minter0, p[2][0]);
					mpca[2] = _mm256_fmadd_ps(*minter1, p[2][1], mpca[2]);
					mpca[2] = _mm256_fmadd_ps(*minter2, p[2][2], mpca[2]);
					mpca[3] = _mm256_mul_ps(*minter0, p[3][0]);
					mpca[3] = _mm256_fmadd_ps(*minter1, p[3][1], mpca[3]);
					mpca[3] = _mm256_fmadd_ps(*minter2, p[3][2], mpca[3]);

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_fnmadd_ps(mpca[1], norm, *mguide[1]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_fnmadd_ps(mpca[2], norm, *mguide[2]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_fnmadd_ps(mpca[3], norm, *mguide[3]++);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mnumer1++ = _mm256_mul_ps(malpha, *minter1++);
						*mnumer2++ = _mm256_mul_ps(malpha, *minter2++);
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mnumer1++ = _mm256_fmadd_ps(malpha, *minter1++, *mnumer1);
						*mnumer2++ = _mm256_fmadd_ps(malpha, *minter2++, *mnumer2);
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
		else if (channels == 3)
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* minter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mnumer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* mnumer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);

					for (int c = 0; c < guide_channels; c++)
					{
						mpca[c] = _mm256_mul_ps(*minter0, p[c][0]);
						mpca[c] = _mm256_fmadd_ps(*minter1, p[c][1], mpca[c]);
						mpca[c] = _mm256_fmadd_ps(*minter2, p[c][2], mpca[c]);
					}

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					for (int c = 1; c < guide_channels; c++)
					{
						msub = _mm256_fnmadd_ps(mpca[c], norm, *mguide[c]++);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mnumer1++ = _mm256_mul_ps(malpha, *minter1++);
						*mnumer2++ = _mm256_mul_ps(malpha, *minter2++);
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mnumer1++ = _mm256_fmadd_ps(malpha, *minter1++, *mnumer1);
						*mnumer2++ = _mm256_fmadd_ps(malpha, *minter2++, *mnumer2);
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}
		else
		{
			for (int y = boundaryLength; y < h - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);
				for (int c = 0; c < channels; c++) minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
				const __m256* minterw = (const __m256*)intermat[channels].ptr<float>(y, boundaryLength);
				for (int c = 0; c < channels; c++)  mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
				__m256* mdenom = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < w - boundaryLength; x += 8)
				{
					const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *minterw);

					for (int c = 0; c < guide_channels; c++)
					{
						mpca[c] = _mm256_mul_ps(*minter[0], p[c][0]);
						for (int cc = 1; cc < channels; cc++)
						{
							mpca[c] = _mm256_fmadd_ps(*minter[cc], p[c][cc], mpca[c]);
						}
					}

					__m256 msub = _mm256_fnmadd_ps(mpca[0], norm, *mguide[0]++);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					for (int c = 1; c < guide_channels; c++)
					{
						msub = _mm256_fnmadd_ps(mpca[c], norm, *mguide[c]++);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}

					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_mul_ps(malpha, *minter[c]++);
						}
						*mdenom++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_fmadd_ps(malpha, *minter[c]++, *mnumer[c]);
						}
						*mdenom++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom);
					}
				}
			}
		}

		for (int c = 0; c < guide_channels; c++) _mm_free(p[c]);
	}

	template<int use_fmath, const bool isInit>
	void ClusteringHDKF_Interpolation2Single::mergeRecomputeAlpha(const std::vector<cv::Mat>& guide, const int k)
	{
		//print_debug2(channels, guide_channels);
		const bool isROI = true;
		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);

		//merge
		const int wk = (isUsePrecomputedWforeachK) ? k : 0;
		vector<Mat> intermat(channels + 1);
		mergeNumerDenomMat(intermat, k, downSampleImage);
		AutoBuffer<const __m256*> mguide(guide_channels);
		if (channels == 1 && guide_channels == 1)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<float>(k));
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[1].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else if (channels == 1 && guide_channels == 2)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec2f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec2f>(k)[1]);
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
				const __m256* im1 = (__m256*)guide[1].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[1].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*im1++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else if (channels == 1 && guide_channels == 3)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
			const __m256 mc2 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*mguide[0]++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*mguide[1]++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*mguide[2]++, mc2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else if (channels == 1 && guide_channels == 4)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[1]);
			const __m256 mc2 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[2]);
			const __m256 mc3 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[3]);
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				for (int c = 0; c < guide_channels; c++) mguide[c] = (const __m256*)guide[c].ptr<float>(y, boundaryLength);

				const __m256* minter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* minterw = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
				__m256* mnumer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* mdenom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*mguide[0]++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*mguide[1]++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*mguide[2]++, mc2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*mguide[3]++, mc3);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						*mnumer0++ = _mm256_mul_ps(malpha, *minter0++);
						*mdenom_++ = _mm256_mul_ps(malpha, *minterw++);
					}
					else
					{
						*mnumer0++ = _mm256_fmadd_ps(malpha, *minter0++, *mnumer0);
						*mdenom_++ = _mm256_fmadd_ps(malpha, *minterw++, *mdenom_);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 1)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<float>(k));
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* inter1 = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* inter2 = (__m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[3].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*numer1++ = _mm256_mul_ps(malpha, *inter1++);
						*numer2++ = _mm256_mul_ps(malpha, *inter2++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 2)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec2f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec2f>(k)[1]);
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
				const __m256* im1 = (__m256*)guide[1].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* inter1 = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* inter2 = (__m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[3].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*im1++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*numer1++ = _mm256_mul_ps(malpha, *inter1++);
						*numer2++ = _mm256_mul_ps(malpha, *inter2++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 3)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
			const __m256 mc2 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
				const __m256* im1 = (__m256*)guide[1].ptr<float>(y, boundaryLength);
				const __m256* im2 = (__m256*)guide[2].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* inter1 = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* inter2 = (__m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[3].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*im1++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*im2++, mc2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*numer1++ = _mm256_mul_ps(malpha, *inter1++);
						*numer2++ = _mm256_mul_ps(malpha, *inter2++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else if (channels == 3 && guide_channels == 4)
		{
			const __m256 mc0 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[0]);
			const __m256 mc1 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[1]);
			const __m256 mc2 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[2]);
			const __m256 mc3 = _mm256_set1_ps(mu.at<cv::Vec4f>(k)[3]);
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const __m256* im0 = (__m256*)guide[0].ptr<float>(y, boundaryLength);
				const __m256* im1 = (__m256*)guide[1].ptr<float>(y, boundaryLength);
				const __m256* im2 = (__m256*)guide[2].ptr<float>(y, boundaryLength);
				const __m256* im3 = (__m256*)guide[3].ptr<float>(y, boundaryLength);
				const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
				const __m256* inter1 = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
				const __m256* inter2 = (__m256*)intermat[2].ptr<float>(y, boundaryLength);
				const __m256* interw = (__m256*)intermat[3].ptr<float>(y, boundaryLength);

				__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
				__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
				__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 msub = _mm256_sub_ps(*im0++, mc0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(*im1++, mc1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*im2++, mc2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(*im3++, mc3);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
					if constexpr (isInit)
					{
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*numer1++ = _mm256_mul_ps(malpha, *inter1++);
						*numer2++ = _mm256_mul_ps(malpha, *inter2++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
		else //n-dimensional signal
		{
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				cv::AutoBuffer<__m256*> gptr(guide_channels);
				cv::AutoBuffer<__m256> mguide(guide_channels);
				for (int c = 0; c < guide_channels; c++) gptr[c] = (__m256*)guide[c].ptr<float>(y, boundaryLength);

				const float* muPtr = mu.ptr<float>(k);
				AutoBuffer<__m256> mmu(guide_channels);
				for (int c = 0; c < guide_channels; c++)
				{
					mmu[c] = _mm256_set1_ps(muPtr[c]);
				}

				AutoBuffer<const __m256*> minter(channels);
				AutoBuffer<__m256*> mnumer(channels);
				for (int c = 0; c < channels; c++)
				{
					minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
					mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
				}
				const __m256* interw = (__m256*)intermat[channels].ptr<float>(y, boundaryLength);
				const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>(y, boundaryLength);
				__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 mdiff;
					{
						__m256 msub = _mm256_sub_ps(*gptr[0]++, mmu[0]);
						mdiff = _mm256_mul_ps(msub, msub);
					}
					for (int c = 1; c < guide_channels; c++)
					{
						__m256 msub = _mm256_sub_ps(*gptr[c]++, mmu[c]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}
					const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

					if constexpr (isInit)
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_mul_ps(malpha, *minter[c]++);
						}
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
					else
					{
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_fmadd_ps(malpha, *minter[c]++, *mnumer[c]);
						}
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
		}
	}

	void ClusteringHDKF_Interpolation2Single::mergePreComputedAlpha(const int k, const bool isInit)
	{
		//merge
		const int wk = (isUsePrecomputedWforeachK) ? k : 0;
		vector<Mat> intermat(channels + 1);
		mergeNumerDenomMat(intermat, k, downSampleImage);

		const bool isROI = true;
		if (isInit)
		{
			if (channels == 1)
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
					const __m256* interw = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
					const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>(y, boundaryLength);

					__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
					__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 malpha = *alpha_ptr++;
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
				}
			}
			else if (channels == 3)
			{
				if (isROI)
				{
					for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
					{
						const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
						const __m256* inter1 = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
						const __m256* inter2 = (__m256*)intermat[2].ptr<float>(y, boundaryLength);
						const __m256* interw = (__m256*)intermat[3].ptr<float>(y, boundaryLength);
						const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>(y, boundaryLength);

						__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
						__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
						__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
						__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

						for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
						{
							const __m256 malpha = *alpha_ptr++;
							*numer0++ = _mm256_mul_ps(malpha, *inter0++);
							*numer1++ = _mm256_mul_ps(malpha, *inter1++);
							*numer2++ = _mm256_mul_ps(malpha, *inter2++);
							*denom_++ = _mm256_mul_ps(malpha, *interw++);
						}
					}
				}
				else
				{
					const int size8 = img_size.area() / 8;
					const __m256* inter0 = (__m256*)intermat[0].ptr<float>();
					const __m256* inter1 = (__m256*)intermat[1].ptr<float>();
					const __m256* inter2 = (__m256*)intermat[2].ptr<float>();
					const __m256* interw = (__m256*)intermat[3].ptr<float>();
					const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>();

					__m256* numer0 = (__m256*)numer[0].ptr<float>();
					__m256* numer1 = (__m256*)numer[1].ptr<float>();
					__m256* numer2 = (__m256*)numer[2].ptr<float>();
					__m256* denom_ = (__m256*)denom.ptr<float>();

					for (int i = 0; i < size8; i++)
					{
						const __m256 malpha = *alpha_ptr++;
						*numer0++ = _mm256_mul_ps(malpha, *inter0++);
						*numer1++ = _mm256_mul_ps(malpha, *inter1++);
						*numer2++ = _mm256_mul_ps(malpha, *inter2++);
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
				}
			}
			else //n-dimensional signal
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					AutoBuffer<const __m256*> minter(channels);
					AutoBuffer<__m256*> mnumer(channels);
					for (int c = 0; c < channels; c++)
					{
						minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
						mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
					}
					const __m256* interw = (__m256*)intermat[channels].ptr<float>(y, boundaryLength);
					const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>(y, boundaryLength);
					__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 malpha = *alpha_ptr++;
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c]++ = _mm256_mul_ps(malpha, *minter[c]++);
						}
						*denom_++ = _mm256_mul_ps(malpha, *interw++);
					}
				}
			}
		}
		else
		{
			if (channels == 1)
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					const __m256* inter0 = (__m256*)intermat[0].ptr<float>(y, boundaryLength);
					const __m256* interw = (__m256*)intermat[1].ptr<float>(y, boundaryLength);
					const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>(y, boundaryLength);

					__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
					__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 malpha = *alpha_ptr++;
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
			else if (channels == 3)
			{
				if (isROI)
				{
					for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
					{
						const __m256* inter0 = (const __m256*)intermat[0].ptr<float>(y, boundaryLength);
						const __m256* inter1 = (const __m256*)intermat[1].ptr<float>(y, boundaryLength);
						const __m256* inter2 = (const __m256*)intermat[2].ptr<float>(y, boundaryLength);
						const __m256* interw = (const __m256*)intermat[3].ptr<float>(y, boundaryLength);
						const __m256* alphap = (const __m256*)alpha[wk].ptr<float>(y, boundaryLength);

						__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
						__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
						__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
						__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

						for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
						{
							const __m256 malpha = *alphap++;
							*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
							*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
							*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
							*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
						}
					}
				}
				else
				{
					const int size8 = img_size.area() / 8;
					const __m256* inter0 = (__m256*)intermat[0].ptr<float>();
					const __m256* inter1 = (__m256*)intermat[1].ptr<float>();
					const __m256* inter2 = (__m256*)intermat[2].ptr<float>();
					const __m256* interw = (__m256*)intermat[3].ptr<float>();
					const __m256* alphap = (__m256*)alpha[wk].ptr<float>();

					__m256* numer0 = (__m256*)numer[0].ptr<float>();
					__m256* numer1 = (__m256*)numer[1].ptr<float>();
					__m256* numer2 = (__m256*)numer[2].ptr<float>();
					__m256* denom_ = (__m256*)denom.ptr<float>();

					for (int i = 0; i < size8; i++)
					{
						const __m256 malpha = *alphap++;
						*numer0++ = _mm256_fmadd_ps(malpha, *inter0++, *numer0);
						*numer1++ = _mm256_fmadd_ps(malpha, *inter1++, *numer1);
						*numer2++ = _mm256_fmadd_ps(malpha, *inter2++, *numer2);
						*denom_++ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
					}
				}
			}
			else //n-dimensional signal
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					AutoBuffer<const __m256*> minter(channels);
					AutoBuffer<__m256*> mnumer(channels);
					for (int c = 0; c < channels; c++)
					{
						minter[c] = (const __m256*)intermat[c].ptr<float>(y, boundaryLength);
						mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
					}
					const __m256* interw = (__m256*)intermat[channels].ptr<float>(y, boundaryLength);
					const __m256* alpha_ptr = (__m256*)alpha[wk].ptr<float>(y, boundaryLength);
					__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 malpha = *alpha_ptr++;
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c] = _mm256_fmadd_ps(malpha, *minter[c]++, *mnumer[c]);
							*mnumer[c]++;
						}
						*denom_ = _mm256_fmadd_ps(malpha, *interw++, *denom_);
						*denom_++;
					}
				}
			}
		}
	}

	void ClusteringHDKF_Interpolation2Single::merge(const int k, const bool isInit)
	{
		//merge
		//if (isUseLocalMu && (!isJoint))
		if (isUseLocalMu && (statePCA != 2))
		{
			//std::cout << "here: using local mu" << std::endl;
			constexpr bool isNLM = false;

			if constexpr (isNLM)
			{
				if (isInit)
				{
					if (isUseFmath) mergeRecomputeAlphaForUsingNLMMu<1, true>(vsrc, k);
					else mergeRecomputeAlphaForUsingNLMMu<0, true>(vsrc, k);
				}
				else
				{
					if (isUseFmath) mergeRecomputeAlphaForUsingNLMMu<1, false>(vsrc, k);
					else mergeRecomputeAlphaForUsingNLMMu<0, false>(vsrc, k);
				}
			}
			else
			{
				if (isInit)
				{
					if (isUseFmath) mergeRecomputeAlphaForUsingMu<1, true>(vsrc, k);
					else mergeRecomputeAlphaForUsingMu<0, true>(vsrc, k);
				}
				else
				{
					if (isUseFmath) mergeRecomputeAlphaForUsingMu<1, false>(vsrc, k);
					else mergeRecomputeAlphaForUsingMu<0, false>(vsrc, k);
				}
			}
		}
		else if (isUseLocalMu && (statePCA == 2))
		{
			if (isInit) mergeRecomputeAlphaForUsingMuPCA<1, true>(vguide, k);
			else mergeRecomputeAlphaForUsingMuPCA<1, false>(vguide, k);
			/*
			//std::cout << "here: using local mu (PCA)" << std::endl;
			if (channels == 1 && guide_channels == 1) mergeRecomputeAlphaForUsingMuPCA<1, 1, 1>(vguide, k, isInit);
			else if (channels == 1 && guide_channels == 2) mergeRecomputeAlphaForUsingMuPCA<1, 1, 2>(vguide, k, isInit);
			else if (channels == 1 && guide_channels == 3) mergeRecomputeAlphaForUsingMuPCA<1, 1, 3>(vguide, k, isInit);
			else if (channels == 3 && guide_channels == 1) mergeRecomputeAlphaForUsingMuPCA<1, 3, 1>(vguide, k, isInit);
			else if (channels == 3 && guide_channels == 2) mergeRecomputeAlphaForUsingMuPCA<1, 3, 2>(vguide, k, isInit);
			else if (channels == 3 && guide_channels == 3) mergeRecomputeAlphaForUsingMuPCA<1, 3, 3>(vguide, k, isInit);
			else if (channels == 33 && guide_channels == 1) mergeRecomputeAlphaForUsingMuPCA<1, 33, 1>(vguide, k, isInit);
			else if (channels == 33 && guide_channels == 2) mergeRecomputeAlphaForUsingMuPCA<1, 33, 2>(vguide, k, isInit);
			else if (channels == 33 && guide_channels == 3) mergeRecomputeAlphaForUsingMuPCA<1, 33, 3>(vguide, k, isInit);
			else
			{
				if (isUseFmath) mergeRecomputeAlphaForUsingMuPCA<1>(vguide, k, isInit);
				else mergeRecomputeAlphaForUsingMuPCA<0>(vguide, k, isInit);
			}*/
		}
		else
		{
			//std::cout << "here: not using local mu" << std::endl;
#if 1
			//std::cout << "here: not using local mu" << std::endl;
			if (isInit)
			{
				if (isUseFmath) mergeRecomputeAlpha<1, true>(vguide, k);
				else mergeRecomputeAlpha<0, true>(vguide, k);
				/*computeAlpha<1>((isJoint) ? vguide : vsrc, k);//K*imsize
				merge(k, isInit);*/

			}
			else
			{
				if (isUseFmath) mergeRecomputeAlpha<1, false>(vguide, k);
				else mergeRecomputeAlpha<0, false>(vguide, k);
			}
#else 
			computeAlpha<1>((isJoint) ? vguide : vsrc, k);//K*imsize
			merge(k, isInit);
#endif
		}
	}
#pragma endregion

	void ClusteringHDKF_Interpolation2Single::normalize(cv::Mat& dst)
	{
		if (channels == 1)
		{
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				float* numer0 = numer[0].ptr<float>(y);
				float* denom_ = denom.ptr<float>(y);
				float* dptr = dst.ptr<float>(y);
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 mnumer_0 = _mm256_load_ps(numer0 + x);
					__m256 mdenom__ = _mm256_load_ps(denom_ + x);

					__m256 mb = _mm256_div_avoidzerodiv_ps(mnumer_0, mdenom__);
					//__m256 mb = _mm256_div_ps(mnumer_0, mdenom__);
					_mm256_store_ps(dptr + x, mb);
				}
			}
		}
		else if (channels == 3)
		{
			//const __m256 m255=_mm256_set1_ps(255.f);
			const bool isROI = true;
			if (isROI)
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
					__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
					__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
					__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);
					float* dptr = dst.ptr<float>(y);
					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 mdenom__ = *denom_++;

						__m256 mb = _mm256_div_avoidzerodiv_ps(*numer0++, mdenom__);
						__m256 mg = _mm256_div_avoidzerodiv_ps(*numer1++, mdenom__);
						__m256 mr = _mm256_div_avoidzerodiv_ps(*numer2++, mdenom__);
						//mb = _mm256_max_ps(_mm256_min_ps(mb, m255), _mm256_setzero_ps());
						//mg = _mm256_max_ps(_mm256_min_ps(mg, m255), _mm256_setzero_ps());
						//mr = _mm256_max_ps(_mm256_min_ps(mr, m255), _mm256_setzero_ps());
						//__m256 mb = _mm256_div_ps(*numer0++, mdenom__);
						//__m256 mg = _mm256_div_ps(*numer1++, mdenom__);
						//__m256 mr = _mm256_div_ps(*numer2++, mdenom__);
						_mm256_store_ps_color(dptr + 3 * x, mb, mg, mr);
					}
				}
			}
			else
			{
				const int size = img_size.area();
				__m256* numer0 = (__m256*)numer[0].ptr<float>();
				__m256* numer1 = (__m256*)numer[1].ptr<float>();
				__m256* numer2 = (__m256*)numer[2].ptr<float>();
				__m256* denom_ = (__m256*)denom.ptr<float>();
				float* dptr = dst.ptr<float>();
				for (int i = 0; i < size; i += 8)
				{
					const __m256 mdenom__ = *denom_++;
					__m256 mb = _mm256_div_avoidzerodiv_ps(*numer0++, mdenom__);
					__m256 mg = _mm256_div_avoidzerodiv_ps(*numer1++, mdenom__);
					__m256 mr = _mm256_div_avoidzerodiv_ps(*numer2++, mdenom__);
					_mm256_store_ps_color(dptr + 3 * i, mb, mg, mr);
				}
			}
		}
		else
		{
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				AutoBuffer<__m256*> mnumer(channels);
				for (int c = 0; c < channels; c++)
				{
					mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
				}
				const __m256* denom_ = (const __m256*)denom.ptr<float>(y, boundaryLength);
				float* dptr = dst.ptr<float>(y);
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					const __m256 mdenom__ = *denom_++;
					for (int c = 0; c < channels; c++)
					{
						*mnumer[c] = _mm256_div_avoidzerodiv_ps(*mnumer[c], mdenom__);
						dptr[channels * (x + 0) + c] = mnumer[c]->m256_f32[0];
						dptr[channels * (x + 1) + c] = mnumer[c]->m256_f32[1];
						dptr[channels * (x + 2) + c] = mnumer[c]->m256_f32[2];
						dptr[channels * (x + 3) + c] = mnumer[c]->m256_f32[3];
						dptr[channels * (x + 4) + c] = mnumer[c]->m256_f32[4];
						dptr[channels * (x + 5) + c] = mnumer[c]->m256_f32[5];
						dptr[channels * (x + 6) + c] = mnumer[c]->m256_f32[6];
						dptr[channels * (x + 7) + c] = mnumer[c]->m256_f32[7];
						mnumer[c]++;
					}
				}
			}
		}
	}


	void ClusteringHDKF_Interpolation2Single::body(const std::vector<cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide)
	{
		{
			//cp::Timer t("alloc");
			alloc(dst);
		}

		{
			//cp::Timer t("clustering");
			clustering();
		}

		{
			downsampleImage(src, vsrcRes, guide, vguideRes, downsampleImageMethod);
		}

		{
			//cp::Timer t("compute alpha");
			vector<Mat> signal = (isJoint) ? guide : src;
			vector<Mat> signalRes = (isJoint) ? vguideRes : vsrcRes;
			if (isUsePrecomputedWforeachK)
			{
				if (isUseLocalMu && !isJoint)
				{
					//std::cout << "isUseLocalMu && !isJoint" << std::endl;
					if (isUseFmath) computeW<1>(src, vecW);//K*imsize
					else computeW<0>(src, vecW);
				}
				else
				{
					//std::cout << "computeIndex" << std::endl;
					if (isUseFmath) computeWandAlpha<1>(signal);//K*imsize
					else computeWandAlpha<0>(signal);
				}
			}
			else
			{
				//std::cout << "computeIndex" << std::endl;
				computeIndex(signal, signalRes);//K*imsize
			}
		}

		{
			//cp::Timer t("blur");
			if (isUseLocalStatisticsPrior)
			{
				/* //for LUT
				const float sqrt2_sr_divpi = float((sqrt(2.0) * sigma_range) / sqrt(CV_PI));
				const float sqrt2_sr_inv = float(1.0 / (sqrt(2.0) * sigma_range));
				const float eps2 = delta * sqrt2_sr_inv;
				const float exp2 = exp(-eps2 * eps2);
				const float erf2 = erf(eps2);
				lut_bflsp.resize(4430 + 1);
				lut_bflsp[0] = 1.f;
				for (int i = 1; i <= 4430; i++)
				{
					float ii = i * 0.1f;
					//float ii = i;
					const float eps1 = (2.f * ii + delta) * sqrt2_sr_inv;
					lut_bflsp[i] = ((exp(-eps1 * eps1) - exp2) / (erf(eps1) + erf2)) * sqrt2_sr_divpi / (ii + FLT_EPSILON);
				}
				*/
			}

			switch (channels)
			{
			case 1:
			{
				for (int k = 0; k < K; k++)
				{
					split_blur<1>(k, isUseFmath, isUseLocalStatisticsPrior);
					merge(k, k == 0);
				}
				break;
			}
			case 3:
			{
				for (int k = 0; k < K; k++)
				{
					split_blur<3>(k, isUseFmath, isUseLocalStatisticsPrior);
					merge(k, k == 0);
				}
				break;
			}
			case 33:
			{
				for (int k = 0; k < K; k++)
				{
					split_blur<33>(k, isUseFmath, isUseLocalStatisticsPrior);
					merge(k, k == 0);
				}
				break;
			}
			default:
			{
				for (int k = 0; k < K; k++)
				{
					split_blur(k, isUseFmath, isUseLocalStatisticsPrior);
					merge(k, k == 0);
				}
				break;
			}
			}
		}
		{
			//cp::Timer t("normal");
			normalize(dst);
		}
	}
}