#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"
#include "simdexp_local.hpp"

namespace cp
{
	void ClusteringHDKF_SoftAssignmentSingle::setLambdaInterpolation(const float lambda)
	{
		this->lambda = lambda;
	}

	void ClusteringHDKF_SoftAssignmentSingle::alloc(cv::Mat& dst)
	{
		downsampleSRC.resize(channels + 1);

		if (vecW.size() != K || vecW[0].size() != img_size)
		{
			vecW.resize(K);
			for (int i = 0; i < K; i++)
			{
				vecW[i].create(img_size, CV_32F);
			}
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
				alpha[i].create(img_size, CV_32F);
			}
		}

		if (alphaSum.size() != img_size)
		{
			alphaSum.create(img_size, CV_32F);
		}

		if (numer.size() != channels || numer[0].size() != img_size)
		{
			numer.resize(channels);
			for (int c = 0; c < channels; c++)
			{
				numer[c].create(img_size, CV_32F);
			}
		}

		if (split_inter.size() != channels) split_inter.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			if (split_inter[c].size() != img_size)
				split_inter[c].create(img_size, CV_32FC1);
		}

		dst.create(img_size, CV_MAKETYPE(CV_32F, channels));
	}

	template<int use_fmath>
	void ClusteringHDKF_SoftAssignmentSingle::computeWandAlpha(const std::vector<cv::Mat>& guide, const std::vector<cv::Mat>& guideRes)
	{
		/*const cv::Size size = img_size / downSampleImage;
		const int imsize = size.area();
		const int IMSIZE8 = imsize / 8;*/
		const cv::Size size = img_size;
		const int imsize = size.area();
		const int IMSIZE8 = imsize / 8;

		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);
		const __m256 mlambda = _mm256_set1_ps(-lambda);
		__m256 margclip = _mm256_set1_ps(float(sigma_range * 5.0 * sigma_range * 5.0));
		cv::AutoBuffer<float*> alpha_ptr(K);
		cv::AutoBuffer<float*> w_ptr(K);
		for (int k = 0; k < K; k++)
		{
			alpha_ptr[k] = alpha[k].ptr<float>();
			w_ptr[k] = vecW[k].ptr<float>();
		}

		const int method = 0;//nk-loop-soft
		//const int method = 1;//nk-loop-hard
		//const int method = 2;//kn-loop-soft (fastest at core i7 6700k)
		if (method == 0)
		{
			cv::AutoBuffer<__m256> mvalpha(K);
			__m256* mc = (__m256*)_mm_malloc(sizeof(__m256) * K * guide_channels, 32);
			for (int k = 0; k < K; k++)
			{
				float* mup = mu.ptr<float>(k);
				__m256* mcp = &mc[guide_channels * k];
				for (int c = 0; c < guide_channels; c++)
				{
					mcp[c] = _mm256_set1_ps(mup[c]);
				}
			}

			if (guide_channels == 1)
			{
				//std::cout << "nk-1" << std::endl;
				//print_matinfo(mu);
				const float* im0 = guide[0].ptr<float>();
				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					__m256 malpha_sum = _mm256_setzero_ps();

					__m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mcp[0]);
						__m256 mdiff = _mm256_min_ps(margclip, _mm256_mul_ps(msub, msub));

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian

						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}

					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}
			else if (guide_channels == 2)
			{
				//std::cout << "nk-2" << std::endl;
				//print_matinfo(mu);
				const float* im0 = guide[0].ptr<float>();
				const float* im1 = guide[1].ptr<float>();
				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					__m256 malpha_sum = _mm256_setzero_ps();

					__m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mcp[0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mcp[1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian

						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}

					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}
			else if (guide_channels == 3)
			{
				/*const float* im0 = (downSampleImage == 1) ? guide[0].ptr<float>() : guideRes[0].ptr<float>();
				const float* im1 = (downSampleImage == 1) ? guide[1].ptr<float>() : guideRes[1].ptr<float>();
				const float* im2 = (downSampleImage == 1) ? guide[2].ptr<float>() : guideRes[2].ptr<float>();*/
				const float* im0 = guide[0].ptr<float>();
				const float* im1 = guide[1].ptr<float>();
				const float* im2 = guide[2].ptr<float>();
				for (int n = 0; n < imsize; n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);
					__m256 malpha_sum = _mm256_setzero_ps();

					__m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mcp[0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mcp[1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage2, mcp[2]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian
						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}
					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}
			else if (guide_channels == 4)
			{
				const float* im0 = guide[0].ptr<float>();
				const float* im1 = guide[1].ptr<float>();
				const float* im2 = guide[2].ptr<float>();
				const float* im3 = guide[3].ptr<float>();
				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);
					const __m256 mimage3 = _mm256_load_ps(im3 + n);
					__m256 malpha_sum = _mm256_setzero_ps();

					__m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mcp[0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mcp[1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage2, mcp[2]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage3, mcp[3]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian
						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}
					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}
			else if (guide_channels == 5)
			{
				const float* im0 = guide[0].ptr<float>();
				const float* im1 = guide[1].ptr<float>();
				const float* im2 = guide[2].ptr<float>();
				const float* im3 = guide[3].ptr<float>();
				const float* im4 = guide[4].ptr<float>();
				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);
					const __m256 mimage3 = _mm256_load_ps(im3 + n);
					const __m256 mimage4 = _mm256_load_ps(im4 + n);
					__m256 malpha_sum = _mm256_setzero_ps();

					__m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mcp[0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mcp[1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage2, mcp[2]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage3, mcp[3]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage4, mcp[4]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian

						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}
					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}
			else if (guide_channels == 6)
			{
				const float* im0 = guide[0].ptr<float>();
				const float* im1 = guide[1].ptr<float>();
				const float* im2 = guide[2].ptr<float>();
				const float* im3 = guide[3].ptr<float>();
				const float* im4 = guide[4].ptr<float>();
				const float* im5 = guide[5].ptr<float>();
				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mimage0 = _mm256_load_ps(im0 + n);
					const __m256 mimage1 = _mm256_load_ps(im1 + n);
					const __m256 mimage2 = _mm256_load_ps(im2 + n);
					const __m256 mimage3 = _mm256_load_ps(im3 + n);
					const __m256 mimage4 = _mm256_load_ps(im4 + n);
					const __m256 mimage5 = _mm256_load_ps(im5 + n);
					__m256 malpha_sum = _mm256_setzero_ps();

					__m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(mimage0, mcp[0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(mimage1, mcp[1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage2, mcp[2]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage3, mcp[3]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage4, mcp[4]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(mimage5, mcp[5]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian

						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}
					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}
			else
			{
				cv::AutoBuffer<const float*> im(guide_channels);
				for (int c = 0; c < guide_channels; c++)
				{
					im[c] = guide[c].ptr<float>();
				}
				cv::AutoBuffer<__m256> mimage(guide_channels);
				for (int n = 0; n < img_size.area(); n += 8)
				{
					for (int c = 0; c < guide_channels; c++)
					{
						mimage[c] = _mm256_load_ps(im[c] + n);
					}
					__m256 malpha_sum = _mm256_setzero_ps();

					const __m256* mcp = mc;
					for (int k = 0; k < K; k++)
					{
						__m256 mdiff = _mm256_setzero_ps();
						for (int c = 0; c < guide_channels; c++)
						{
							__m256 msub = _mm256_sub_ps(mimage[c], mcp[c]);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						}
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr[k], v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian

						mvalpha[k] = malpha;
						malpha_sum = _mm256_add_ps(malpha_sum, malpha);
						w_ptr[k] += 8;
						mcp += guide_channels;
					}

					const __m256 malpha_sum_inv = _mm256_div_ps(_mm256_set1_ps(1.f), malpha_sum);
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(alpha_ptr[k] + n, _mm256_mul_ps(mvalpha[k], malpha_sum_inv));
						//_mm256_store_ps(alpha_ptr[k] + n, _mm256_div_ps(mvalpha[k], malpha_sum));
						//_mm256_store_ps(alpha_ptr[k] + n, mvalpha[k]);
					}
				}
			}

			_mm_free(mc);
		}
		else if (method == 1)
		{
			float* im0 = vsrc[0].ptr<float>();
			float* im1 = vsrc[1].ptr<float>();
			float* im2 = vsrc[2].ptr<float>();
			for (int n = 0; n < img_size.area(); n += 8)
			{
				const __m256 mimage0 = _mm256_load_ps(im0 + n);
				const __m256 mimage1 = _mm256_load_ps(im1 + n);
				const __m256 mimage2 = _mm256_load_ps(im2 + n);

				__m256 mdiffmax = _mm256_set1_ps(FLT_MAX);
				__m256 argment = _mm256_set1_ps(0);

				for (int k = 0; k < K; k++)
				{
					const __m256 mcenter0 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[0]);
					const __m256 mcenter1 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[1]);
					const __m256 mcenter2 = _mm256_set1_ps(mu.at<cv::Vec3f>(k)[2]);
					float* w_ptr = vecW[k].ptr<float>();

					__m256 msub = _mm256_sub_ps(mimage0, mcenter0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(mimage1, mcenter1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(mimage2, mcenter2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					mdiff = _mm256_min_ps(margclip, mdiff);

					_mm256_store_ps(w_ptr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

					_mm256_argmin_ps(mdiff, mdiffmax, argment, (float)k);
				}
				for (int k = 0; k < K; k++)
				{
					_mm256_store_ps(alpha_ptr[k] + n, _mm256_blendv_ps(_mm256_set1_ps(FLT_MIN), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(float(k)), 0)));
					//_mm256_store_ps(a_ptr + n, _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.f), _mm256_cmp_ps(argment, _mm256_set1_ps(k), 0)));
				}
			}
		}
		else //method = 2(kn-loop-soft)
		{
			alphaSum.setTo(0.f);
			if (guide_channels == 3)
			{
				for (int k = 0; k < K; k++)
				{
					__m256 mcenter0 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][0]);
					__m256 mcenter1 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][1]);
					__m256 mcenter2 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][2]);

					const float* im0 = guide[0].ptr<float>();
					const float* im1 = guide[1].ptr<float>();
					const float* im2 = guide[2].ptr<float>();
					float* w_ptr = vecW[k].ptr<float>();
					float* alpha_ptr = alpha[k].ptr<float>();
					float* alpha_denom_ptr = alphaSum.ptr<float>();

					for (int n = 0; n < img_size.area(); n += 8)
					{
						__m256 mimage0 = _mm256_load_ps(im0 + n);
						__m256 mimage1 = _mm256_load_ps(im1 + n);
						__m256 mimage2 = _mm256_load_ps(im2 + n);

						mimage0 = _mm256_sub_ps(mimage0, mcenter0);
						__m256 mdiff = _mm256_mul_ps(mimage0, mimage0);
						mimage1 = _mm256_sub_ps(mimage1, mcenter1);
						mdiff = _mm256_fmadd_ps(mimage1, mimage1, mdiff);
						mimage2 = _mm256_sub_ps(mimage2, mcenter2);
						mdiff = _mm256_fmadd_ps(mimage2, mimage2, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian
						_mm256_store_ps(alpha_ptr + n, malpha);
						_mm256_store_ps(alpha_denom_ptr + n, _mm256_add_ps(malpha, _mm256_load_ps(alpha_denom_ptr + n)));
					}
				}
			}
			else if (guide_channels == 2)
			{
				//std::cout << "kn-2"<<std::endl;
				for (int k = 0; k < K; k++)
				{
					const __m256 mcenter0 = _mm256_set1_ps(mu.ptr<float>(k)[0]);
					const __m256 mcenter1 = _mm256_set1_ps(mu.ptr<float>(k)[1]);

					const float* im0 = guide[0].ptr<float>();
					const float* im1 = guide[1].ptr<float>();
					float* wk_ptr = vecW[k].ptr<float>();
					float* alphak_ptr = alpha[k].ptr<float>();
					float* alpha_denom_ptr = alphaSum.ptr<float>();

					for (int n = 0; n < img_size.area(); n += 8)
					{
						__m256 mimage0 = _mm256_load_ps(im0 + n);
						__m256 mimage1 = _mm256_load_ps(im1 + n);

						mimage0 = _mm256_sub_ps(mimage0, mcenter0);
						__m256 mdiff = _mm256_mul_ps(mimage0, mimage0);
						mimage1 = _mm256_sub_ps(mimage1, mcenter1);
						mdiff = _mm256_fmadd_ps(mimage1, mimage1, mdiff);
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(wk_ptr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian

						_mm256_store_ps(alphak_ptr + n, malpha);
						_mm256_store_ps(alpha_denom_ptr + n, _mm256_add_ps(malpha, _mm256_load_ps(alpha_denom_ptr + n)));
					}
				}
			}
			else
			{
				cv::AutoBuffer<__m256> mcenter(guide_channels);
				cv::AutoBuffer<const float*> gptr;
				for (int c = 0; c < guide_channels; c++)
				{
					gptr[c] = guide[c].ptr<float>();
				}
				for (int k = 0; k < K; k++)
				{
					const float* muPtr = mu.ptr<float>(k);
					for (int c = 0; c < guide_channels; c++) mcenter[c] = _mm256_set1_ps(muPtr[c]);

					float* w_ptr = vecW[k].ptr<float>();
					float* alpha_ptr = alpha[k].ptr<float>();
					float* alpha_denom_ptr = alphaSum.ptr<float>();
					for (int n = 0; n < img_size.area(); n += 8)
					{
						__m256 mdiff = _mm256_setzero_ps();
						for (int c = 0; c < guide_channels; c++)
						{
							__m256 mimage0 = _mm256_load_ps(gptr[c] + n);
							mimage0 = _mm256_sub_ps(mimage0, mcenter[c]);
							mdiff = _mm256_fmadd_ps(mimage0, mimage0, mdiff);
						}
						mdiff = _mm256_min_ps(margclip, mdiff);

						_mm256_store_ps(w_ptr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));

						const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, _mm256_sqrt_ps(mdiff)));//Laplacian
						//const __m256 malpha = v_exp_ps<use_fmath>(_mm256_mul_ps(mlambda, mdiff));//Gaussian
						_mm256_store_ps(alpha_ptr + n, malpha);
						_mm256_store_ps(alpha_denom_ptr + n, _mm256_add_ps(malpha, _mm256_load_ps(alpha_denom_ptr + n)));
					}
				}
			}

			for (int k = 0; k < K; k++)
			{
				float* alphak_ptr = alpha[k].ptr<float>();
				float* alpha_denom_ptr = alphaSum.ptr<float>();
				for (int n = 0; n < img_size.area(); n += 8)
				{
					_mm256_store_ps(alphak_ptr + n, _mm256_div_ps(_mm256_load_ps(alphak_ptr + n), _mm256_load_ps(alpha_denom_ptr + n)));
				}
			}
		}

		/*for (int k = 0; k < K; k++)
		{
			//std::cout << cp::getPSNR(alpha[k], as[k]) << std::endl;;
			std::cout << cp::getPSNR(vecW[k], ws[k]) << std::endl;;
		}*/
	}

	void multiply32f(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst)
	{
		if (dst.size() != src1.size())dst.create(src1.size(), CV_32F);
		const int SIZE = src1.size().area();
		const int SIZE32 = get_simd_floor(SIZE, 32);
		const float* s1 = src1.ptr<float>();
		const float* s2 = src2.ptr<float>();
		float* d = dst.ptr<float>();
		for (int i = 0; i < SIZE32; i += 32)
		{
			_mm256_store_ps(d + i + 0, _mm256_mul_ps(_mm256_load_ps(s1 + i + 0), _mm256_load_ps(s2 + i + 0)));
			_mm256_store_ps(d + i + 8, _mm256_mul_ps(_mm256_load_ps(s1 + i + 8), _mm256_load_ps(s2 + i + 8)));
			_mm256_store_ps(d + i + 16, _mm256_mul_ps(_mm256_load_ps(s1 + i + 16), _mm256_load_ps(s2 + i + 16)));
			_mm256_store_ps(d + i + 24, _mm256_mul_ps(_mm256_load_ps(s1 + i + 24), _mm256_load_ps(s2 + i + 24)));
		}
		for (int i = SIZE32; i < SIZE; i += 8)
		{
			_mm256_store_ps(d + i, _mm256_mul_ps(_mm256_load_ps(s1 + i), _mm256_load_ps(s2 + i)));
		}
	}

	void ClusteringHDKF_SoftAssignmentSingle::split_blur(const int k)
	{
		float* vecw_ptr = vecW[k].ptr<float>();

		if (channels == 1)
		{
			//split
			//blur
			multiply32f(vsrc[0], vecW[k], split_inter[0]);
			GF->filter(vecW[k], vecW[k], sigma_space, spatial_order);
			GF->filter(split_inter[0], split_inter[0], sigma_space, spatial_order);
		}
		else if (channels == 3)
		{
			const bool isSplitBlurFusion = true;
			//cv::Mat blendLSPMask;
			if (isSplitBlurFusion)
			{
				if (downSampleImage != 1)
				{
					const double res = 1.0 / downSampleImage;
					cv::Mat w = downsampleSRC[0];
					cv::Mat conv = downsampleSRC[1];
					const double dss = sigma_space * res;
					resize(vecW[k], w, cv::Size(), res, res, cv::INTER_AREA);
					for (int c = 0; c < channels; c++)
					{
						multiply32f(vsrcRes[c], w, conv);
						GF->filter(conv, conv, dss, spatial_order, borderType);
						resize(conv, split_inter[c], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					}
					GF->filter(w, w, dss, spatial_order, borderType);
					resize(w, vecW[k], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);

					//resize(alpha[k], w, cv::Size(), res, res, cv::INTER_AREA);
					//resize(w, alpha[k], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					//GF->filter(vecW[k], vecW[k], sigma_space, spatial_order, borderType);
				}
				else
				{
					for (int c = 0; c < channels; c++)
					{
						multiply32f(vsrc[c], vecW[k], split_inter[c]);
						GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order, borderType);
					}
					GF->filter(vecW[k], vecW[k], sigma_space, spatial_order, borderType);
				}
				//vecW[k].copyTo(blendLSPMask);
				//bilateralFilterLocalStatisticsPriorInternal(vsrc, split_inter, vecW[k], (float)sigma_range*0.001, (float)sigma_space*0.0001, -10000.f, blendLSPMask, BFLSPSchedule::Compute);
			}
			else
			{
				float* src0 = vsrc[0].ptr<float>();
				float* src1 = vsrc[1].ptr<float>();
				float* src2 = vsrc[2].ptr<float>();
				float* inter0 = split_inter[0].ptr<float>();
				float* inter1 = split_inter[1].ptr<float>();
				float* inter2 = split_inter[2].ptr<float>();
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
				}

				//blur
				GF->filter(vecW[k], vecW[k], sigma_space, spatial_order, borderType);
				GF->filter(split_inter[0], split_inter[0], sigma_space, spatial_order, borderType);
				GF->filter(split_inter[1], split_inter[1], sigma_space, spatial_order, borderType);
				GF->filter(split_inter[2], split_inter[2], sigma_space, spatial_order, borderType);
			}
		}
		else
		{
			const bool isSplitBlurFusion = true;
			cv::AutoBuffer<float*> srcp(channels);
			cv::AutoBuffer<float*> intp(channels);

			for (int c = 0; c < channels; c++)
			{
				srcp[c] = vsrc[c].ptr<float>();
				intp[c] = split_inter[c].ptr<float>();
			}
			if (isSplitBlurFusion)
			{
				if (downSampleImage != 1)
				{
					const double res = 1.0 / downSampleImage;
					cv::Mat w = downsampleSRC[0];
					cv::Mat conv = downsampleSRC[1];

					resize(vecW[k], w, cv::Size(), res, res, cv::INTER_AREA);
					for (int c = 0; c < channels; c++)
					{
						multiply32f(vsrcRes[c], w, conv);
						GF->filter(conv, conv, sigma_space * res, spatial_order, borderType);
						resize(conv, split_inter[c], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					}
					GF->filter(w, w, sigma_space * res, spatial_order, borderType);
					resize(w, vecW[k], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					//GF->filter(vecW[k], vecW[k], sigma_space, spatial_order, borderType);
				}
				else
				{
					for (int c = 0; c < channels; c++)
					{
						multiply32f(vsrc[c], vecW[k], split_inter[c]);
						GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order, borderType);
					}
					GF->filter(vecW[k], vecW[k], sigma_space, spatial_order, borderType);
				}
			}
			else
			{
				//split
				for (int n = 0; n < img_size.area(); n += 8)
				{
					const __m256 mvecw = _mm256_load_ps(vecw_ptr + n);
					for (int c = 0; c < channels; c++)
					{
						_mm256_store_ps(intp[c] + n, _mm256_mul_ps(mvecw, _mm256_load_ps(srcp[c] + n)));
					}
				}
				//blur
				GF->filter(vecW[k], vecW[k], sigma_space, spatial_order, borderType);
				for (int c = 0; c < channels; c++)
				{
					GF->filter(split_inter[c], split_inter[c], sigma_space, spatial_order, borderType);
				}
			}
		}
	}

	//flag 0: init, 1: midd, 2: last
	template<int flag>
	void ClusteringHDKF_SoftAssignmentSingle::merge(cv::Mat& dst, const int k)
	{
		if (channels == 1)
		{
			//merge
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const float* inter0 = split_inter[0].ptr<float>(y);
				const float* vecw_ptr = vecW[k].ptr<float>(y);
				const float* alpha_ptr = alpha[k].ptr<float>(y);

				float* numer0 = numer[0].ptr<float>(y);
				float* dptr = dst.ptr<float>(y);
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					const __m256 malpha = _mm256_loadu_ps(alpha_ptr + x);
					const __m256 minterw = _mm256_loadu_ps(vecw_ptr + x);
					const __m256 minter0 = _mm256_loadu_ps(inter0 + x);

					if constexpr (flag == 0)
					{
						const __m256 mrcpw = _mm256_mul_ps(malpha, _mm256_rcp_ps(minterw));
						_mm256_store_ps(numer0 + x, _mm256_mul_ps(minter0, mrcpw));
					}
					else if constexpr (flag == 1)
					{
						const __m256 mnumer_0 = _mm256_loadu_ps(numer0 + x);
						const __m256 mrcpw = _mm256_mul_ps(malpha, _mm256_rcp_ps(minterw));
						_mm256_store_ps(numer0 + x, _mm256_fmadd_ps(minter0, mrcpw, mnumer_0));
					}
					else if constexpr (flag == 2)
					{
						const __m256 mnumer_0 = _mm256_loadu_ps(numer0 + x);
						const __m256 mrcpw = _mm256_mul_ps(malpha, _mm256_rcp_ps(minterw));
						_mm256_store_ps(dptr + x, _mm256_fmadd_ps(minter0, mrcpw, mnumer_0));
					}
				}
			}
		}
		else if (channels == 3)
		{
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				const float* inter0 = split_inter[0].ptr<float>(y);
				const float* inter1 = split_inter[1].ptr<float>(y);
				const float* inter2 = split_inter[2].ptr<float>(y);
				const float* vecw_ptr = vecW[k].ptr<float>(y);
				const float* alpha_ptr = alpha[k].ptr<float>(y);

				float* numer0 = numer[0].ptr<float>(y);
				float* numer1 = numer[1].ptr<float>(y);
				float* numer2 = numer[2].ptr<float>(y);
				float* dptr = dst.ptr<float>(y);//for flag==2
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					const __m256 mrcpw = _mm256_mul_ps(_mm256_loadu_ps(alpha_ptr + x), _mm256_rcp_ps(_mm256_loadu_ps(vecw_ptr + x)));
					if constexpr (flag == 0)
					{
						_mm256_store_ps(numer0 + x, _mm256_mul_ps(_mm256_loadu_ps(inter0 + x), mrcpw));
						_mm256_store_ps(numer1 + x, _mm256_mul_ps(_mm256_loadu_ps(inter1 + x), mrcpw));
						_mm256_store_ps(numer2 + x, _mm256_mul_ps(_mm256_loadu_ps(inter2 + x), mrcpw));
					}
					else if constexpr (flag == 1)
					{
						_mm256_store_ps(numer0 + x, _mm256_fmadd_ps(_mm256_loadu_ps(inter0 + x), mrcpw, _mm256_loadu_ps(numer0 + x)));
						_mm256_store_ps(numer1 + x, _mm256_fmadd_ps(_mm256_loadu_ps(inter1 + x), mrcpw, _mm256_loadu_ps(numer1 + x)));
						_mm256_store_ps(numer2 + x, _mm256_fmadd_ps(_mm256_loadu_ps(inter2 + x), mrcpw, _mm256_loadu_ps(numer2 + x)));
					}
					else if constexpr (flag == 2)
					{
						const __m256 mb = _mm256_fmadd_ps(_mm256_loadu_ps(inter0 + x), mrcpw, _mm256_loadu_ps(numer0 + x));
						const __m256 mg = _mm256_fmadd_ps(_mm256_loadu_ps(inter1 + x), mrcpw, _mm256_loadu_ps(numer1 + x));
						const __m256 mr = _mm256_fmadd_ps(_mm256_loadu_ps(inter2 + x), mrcpw, _mm256_loadu_ps(numer2 + x));
						_mm256_store_ps_color(dptr + 3 * x, mb, mg, mr);
					}
				}
			}
		}
		else
		{
			cv::AutoBuffer<float*> intp(channels);
			cv::AutoBuffer<float*> nump(channels);
			//merge
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				for (int c = 0; c < channels; c++)
				{
					intp[c] = split_inter[c].ptr<float>(y);
					nump[c] = numer[c].ptr<float>(y);
				}
				const float* vecw_ptr = vecW[k].ptr<float>(y);
				const float* alpha_ptr = alpha[k].ptr<float>(y);

				float* dptr = dst.ptr<float>(y);//for flag==2
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					const __m256 malpha = _mm256_loadu_ps(alpha_ptr + x);
					const __m256 minterw = _mm256_loadu_ps(vecw_ptr + x);
					const __m256 mrcpw = _mm256_mul_ps(malpha, _mm256_rcp_ps(minterw));
					//__m256 mr = _mm256_fmadd_ps(malpha, _mm256_div_ps(minter2, mvecw), mnumer_2);
					//__m256 mb = _mm256_fmadd_ps(malpha, _mm256_div_avoidzerodiv_ps(minter0, minterw), mnumer_0);
					if constexpr (flag == 0)
					{
						for (int c = 0; c < channels; c++)
						{
							_mm256_store_ps(nump[c] + x, _mm256_mul_ps(_mm256_loadu_ps(intp[c] + x), mrcpw));
						}
					}
					else if constexpr (flag == 1)
					{
						for (int c = 0; c < channels; c++)
						{
							_mm256_store_ps(nump[c] + x, _mm256_fmadd_ps(_mm256_loadu_ps(intp[c] + x), mrcpw, _mm256_loadu_ps(nump[c] + x)));
						}
					}
					else if constexpr (flag == 2)
					{
						for (int c = 0; c < channels; c++)
						{
							__m256 dst = _mm256_fmadd_ps(_mm256_loadu_ps(intp[c] + x), mrcpw, _mm256_loadu_ps(nump[c] + x));
							//for (int s = 0; s < 8; s++) dptr[channels * (x + s) + c] = dst.m256_f32[s];
							dptr[channels * (x + 0) + c] = dst.m256_f32[0];
							dptr[channels * (x + 1) + c] = dst.m256_f32[1];
							dptr[channels * (x + 2) + c] = dst.m256_f32[2];
							dptr[channels * (x + 3) + c] = dst.m256_f32[3];
							dptr[channels * (x + 4) + c] = dst.m256_f32[4];
							dptr[channels * (x + 5) + c] = dst.m256_f32[5];
							dptr[channels * (x + 6) + c] = dst.m256_f32[6];
							dptr[channels * (x + 7) + c] = dst.m256_f32[7];
						}
					}
				}
			}
		}
	}

	void ClusteringHDKF_SoftAssignmentSingle::body(const std::vector<cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide)
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
			if (guide.empty())
			{
				if (isUseFmath) computeWandAlpha<1>(src, vsrcRes);//K*imsize
				else computeWandAlpha<0>(src, vsrcRes);
			}
			else
			{
				if (isUseFmath) computeWandAlpha<1>(guide, vguideRes);//K*imsize
				else computeWandAlpha<0>(guide, vguideRes);
			}
		}
		{
			//cp::Timer t("blur");
			for (int k = 0; k < K; k++)
			{
				split_blur(k);
				if (k == 0) merge<0>(dst, k);
				else if (k == K - 1)merge<2>(dst, k);
				else merge<1>(dst, k);
			}
		}
	}
}