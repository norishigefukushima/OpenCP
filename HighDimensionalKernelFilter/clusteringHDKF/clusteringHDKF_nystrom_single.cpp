#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"
#include "simdexp_local.hpp"

#define USE_EIGEN //direct usage C:\eigen
#ifdef USE_EIGEN
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#endif

//constexpr bool isComputeU = false; //only support 3channel
constexpr bool isComputeU = true;
//computeAandEVD
//computeB<int use_fmath, int guide_channels>
//split_blur_merge<int channels>
namespace cp
{
	void ClusteringHDKF_NystromSingle::alloc(cv::Mat& dst)
	{
		downsampleSRC.resize(channels + 1);//signal + w, i.e., RGBW, RGBDW, RGBIRW,...

		if (B.size() != K || B[0].size() != img_size)
		{
			B.resize(K);
			for (int i = 0; i < K; i++)
			{
				B[i].create(img_size, CV_32FC1);
			}
		}

		if (A.size() != cv::Size(K, K))
		{
			A.create(cv::Size(K, K), CV_32FC1);
		}

		if (denom.size() != img_size)
		{
			denom.create(img_size, CV_32FC1);
		}

		if (numer.size() != channels || numer[0].size() != img_size)
		{
			numer.resize(channels);
			for (int c = 0; c < channels; c++)
			{
				numer[c].create(img_size, CV_32F);
			}
		}

		if (Uf.size() != channels) Uf.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			Uf[c].create(img_size, CV_32F);
		}

		U.create(img_size, CV_32F);
		U_Gaussian.create(img_size, CV_32F);

		dst.create(img_size, CV_MAKETYPE(CV_32F, channels));
	}

	void ClusteringHDKF_NystromSingle::computeAandEVD(const cv::Mat& mu, cv::Mat& lambdaA, cv::Mat& eigenvecA)
	{
		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const bool is32F = true;
		if (is32F)
		{
			for (int i = 0; i < K; i++)
			{
				float* a_i = A.ptr<float>(i);
				a_i[i] = 1.f;//exp(0);
				const float* mui = mu.ptr<float>(i);
				for (int j = i + 1; j < K; j++)
				{
					const float* muj = mu.ptr<float>(j);
					float distance = (mui[0] - muj[0]) * (mui[0] - muj[0]);
					for (int c = 1; c < guide_channels; c++)
					{
						distance += (mui[c] - muj[c]) * (mui[c] - muj[c]);
					}
					float* a_j = A.ptr<float>(j);
#ifdef SUBNORMALCLIP 
					a_j[i] = a_i[j] = std::exp(std::max(coeff * distance, EXP_ARGUMENT_CLIP_VALUE_SP));
#else
					a_j[i] = a_i[j] = std::exp(coeff * distance);
#endif
				}
			}
			//eigen value decomposition lambda: large->small, vector: row major 
			cv::eigen(A, lambdaA, eigenvecA);

			/*for (int j = 0; j < eigenvecA.rows; j++)
			{
				cp::Plot pt;
				print_debug2(j,lambdaA.at<float>(j,0));
				for (int i = 0; i < eigenvecA.cols; i++)
				{
					pt.push_back(i, eigenvecA.at<float>(j, i));
				}
				pt.plot();
			}*/
		}
		else
		{
			const double coeff64 = (-1.0 / (2.0 * sigma_range * sigma_range));
			cv::Mat A64;
			A.convertTo(A64, CV_64F);
			for (int i = 0; i < K; i++)
			{
				double* a_i = A64.ptr<double>(i);
				a_i[i] = 1.0;//exp(0);
				const float* mui = mu.ptr<float>(i);
				for (int j = i + 1; j < K; j++)
				{
					const float* muj = mu.ptr<float>(j);
					double distance = (double)(mui[0] - muj[0]) * (double)(mui[0] - muj[0]);
					for (int c = 1; c < guide_channels; c++)
					{
						distance += double(mui[c] - muj[c]) * double(mui[c] - muj[c]);
					}
					double* a_j = A64.ptr<double>(j);
#ifdef SUBNORMALCLIP 
					a_j[i] = a_i[j] = std::exp(std::max(coeff64 * distance, (double)EXP_ARGUMENT_CLIP_VALUE_SP));
#else
					a_j[i] = a_i[j] = std::exp(coeff64 * distance);
#endif
				}
			}
			//A64.convertTo(A, CV_32F);
			//eigen value decomposition lambda: large->small, vector: row major 

#ifdef USE_EIGEN
			Eigen::MatrixXd W4Eigen;
			cv::cv2eigen<double>(A64, W4Eigen);
			Eigen::EigenSolver<Eigen::MatrixXd> esolve(W4Eigen, true);

			Eigen::MatrixXd a = esolve.eigenvalues().real();
			Eigen::MatrixXd b = esolve.eigenvectors().real().transpose();

			cv::Mat lambdaA64;
			cv::Mat eigenvecA64;
			cv::eigen2cv(a, lambdaA64);
			cv::eigen2cv(b, eigenvecA64);
#else
			cv::Mat lambdaA64;
			cv::Mat eigenvecA64;

			cv::eigen(A64, lambdaA64, eigenvecA64);
#endif
			lambdaA64.convertTo(lambdaA, CV_32F);
			eigenvecA64.convertTo(eigenvecA, CV_32F);
		}

		for (int i = 0; i < lambdaA.size().area(); i++)
		{
			float v = lambdaA.at<float>(i);
			if (abs(v) < FLT_EPSILON)lambdaA.at<float>(i) = FLT_EPSILON;
		}
	}

	template<int use_fmath, int guide_channels>
	void ClusteringHDKF_NystromSingle::computeB(const std::vector<cv::Mat>& guide)
	{
		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);
		const int imsize = guide[0].size().area();

		constexpr bool nk_loop = true;
		//k-n loop
		if constexpr (nk_loop)
		{
			if (guide_channels == 1)
			{
				__m256* gptr0 = (__m256*)guide[0].ptr<float>();

				cv::AutoBuffer<float*> BPtr(K);
				for (int k = 0; k < K; k++) BPtr[k] = B[k].ptr<float>();

				cv::AutoBuffer<__m256*> mmu(K);
				for (int k = 0; k < K; k++)
				{
					const float* mu_kPtr = mu.ptr<float>(k);
					mmu[k] = (__m256*)_mm_malloc(sizeof(__m256) * guide_channels, AVX_ALIGN);
					for (int c = 0; c < guide_channels; c++)
					{
						mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
					}
				}

				cv::AutoBuffer<__m256> U(K);
				for (int n = 0; n < imsize; n += 8)
				{
					const __m256 msrc0 = *gptr0++;
					for (int k = 0; k < K; k++) U[k] = _mm256_setzero_ps();

					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(msrc0, mmu[k][0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						__m256 mexp = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));
						const float* eigVecPtr = eigenvecA.ptr<float>(k);
						for (int l = 0; l < K; l++)
						{
							U[l] = _mm256_fmadd_ps(_mm256_set1_ps(eigVecPtr[l]), mexp, U[l]);
						}
					}
					for (int k = 0; k < K; k++)
					{
						_mm256_store_ps(BPtr[k] + n, U[k]);
					}
				}

				for (int k = 0; k < K; k++) _mm_free(mmu[k]);
			}
			else if (guide_channels == 2)
			{
				const __m256* gptr0 = (__m256*)guide[0].ptr<float>();
				const __m256* gptr1 = (__m256*)guide[1].ptr<float>();

				cv::AutoBuffer<float*> BPtr(K);
				for (int k = 0; k < K; k++) BPtr[k] = B[k].ptr<float>();

				cv::AutoBuffer<__m256*> mmu(K);
				for (int k = 0; k < K; k++)
				{
					const float* mu_kPtr = mu.ptr<float>(k);
					mmu[k] = (__m256*)_mm_malloc(sizeof(__m256) * guide_channels, AVX_ALIGN);
					for (int c = 0; c < guide_channels; c++)
					{
						mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
					}
				}

				for (int n = 0; n < imsize; n += 8)
				{
					const __m256 msrc0 = *gptr0++;
					const __m256 msrc1 = *gptr1++;

					for (int k = 0; k < K; k++)
					{
						const float* mu_kPtr = mu.ptr<float>(k);

						__m256 msub = _mm256_sub_ps(msrc0, mmu[k][0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(msrc1, mmu[k][1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						_mm256_store_ps(BPtr[k] + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
					}
				}

				for (int k = 0; k < K; k++) _mm_free(mmu[k]);
			}
			else if (guide_channels == 3)
			{
				if constexpr (isComputeU)
				{
					const __m256* gptr0 = (__m256*)guide[0].ptr<float>();
					const __m256* gptr1 = (__m256*)guide[1].ptr<float>();
					const __m256* gptr2 = (__m256*)guide[2].ptr<float>();

					cv::AutoBuffer<float*> BPtr(K);
					for (int k = 0; k < K; k++) BPtr[k] = B[k].ptr<float>();

					cv::AutoBuffer<__m256*> mmu(K);
					for (int k = 0; k < K; k++)
					{
						const float* mu_kPtr = mu.ptr<float>(k);
						mmu[k] = (__m256*)_mm_malloc(sizeof(__m256) * guide_channels, AVX_ALIGN);
						for (int c = 0; c < guide_channels; c++)
						{
							mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
						}
					}

					for (int n = 0; n < imsize; n += 8)
					{
						const __m256 msrc0 = *gptr0++;
						const __m256 msrc1 = *gptr1++;
						const __m256 msrc2 = *gptr2++;

						for (int k = 0; k < K; k++)
						{
							const float* mu_kPtr = mu.ptr<float>(k);

							__m256 msub = _mm256_sub_ps(msrc0, mmu[k][0]);
							__m256 mdiff = _mm256_mul_ps(msub, msub);
							msub = _mm256_sub_ps(msrc1, mmu[k][1]);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							msub = _mm256_sub_ps(msrc2, mmu[k][2]);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							_mm256_store_ps(BPtr[k] + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
						}
					}

					for (int k = 0; k < K; k++) _mm_free(mmu[k]);
				}
				else
				{
					const __m256* gptr0 = (__m256*)guide[0].ptr<float>();
					const __m256* gptr1 = (__m256*)guide[1].ptr<float>();
					const __m256* gptr2 = (__m256*)guide[2].ptr<float>();

					cv::AutoBuffer<float*> BPtr(K);
					for (int k = 0; k < K; k++) BPtr[k] = B[k].ptr<float>();

					//set mu
					cv::AutoBuffer<__m256*> mmu(K);
					for (int k = 0; k < K; k++)
					{
						const float* mu_kPtr = mu.ptr<float>(k);
						mmu[k] = (__m256*)_mm_malloc(sizeof(__m256) * guide_channels, AVX_ALIGN);
						for (int c = 0; c < guide_channels; c++)
						{
							mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
						}
					}
					//set evec
					cv::AutoBuffer<__m256*> mevec(K);
					for (int k = 0; k < K; k++)
					{
						mevec[k] = (__m256*)_mm_malloc(sizeof(__m256) * K, AVX_ALIGN);
						for (int l = 0; l < K; l++)
						{
							const float* eigVecPtr = eigenvecA.ptr<float>(l);
							mevec[k][l] = _mm256_set1_ps(eigVecPtr[k]);
						}
					}

					cv::AutoBuffer<__m256> mmU(K);
					for (int n = 0; n < imsize; n += 8)
					{
						const __m256 msrc0 = *gptr0++;
						const __m256 msrc1 = *gptr1++;
						const __m256 msrc2 = *gptr2++;

						for (int k = 0; k < K; k++)
						{
							__m256 msub = _mm256_sub_ps(msrc0, mmu[k][0]);
							__m256 mdiff = _mm256_mul_ps(msub, msub);
							msub = _mm256_sub_ps(msrc1, mmu[k][1]);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							msub = _mm256_sub_ps(msrc2, mmu[k][2]);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
							const __m256 mexp = v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff));

							mmU[0] = _mm256_mul_ps(mevec[k][0], mexp);
							for (int l = 1; l < K; l++)
							{
								mmU[l] = _mm256_fmadd_ps(mevec[k][l], mexp, mmU[l]);
							}
						}

						for (int k = 0; k < K; k++)
						{
							_mm256_store_ps(BPtr[k] + n, mmU[k]);
						}
					}

					for (int k = 0; k < K; k++)
					{
						_mm_free(mmu[k]);
						_mm_free(mevec[k]);
					}
				}
			}
			else if (guide_channels == 4)
			{
				const __m256* gptr0 = (__m256*)guide[0].ptr<float>();
				const __m256* gptr1 = (__m256*)guide[1].ptr<float>();
				const __m256* gptr2 = (__m256*)guide[2].ptr<float>();
				const __m256* gptr3 = (__m256*)guide[3].ptr<float>();

				cv::AutoBuffer<float*> BPtr(K);
				for (int k = 0; k < K; k++) BPtr[k] = B[k].ptr<float>();

				cv::AutoBuffer<__m256*> mmu(K);
				for (int k = 0; k < K; k++)
				{
					const float* mu_kPtr = mu.ptr<float>(k);
					mmu[k] = (__m256*)_mm_malloc(sizeof(__m256) * guide_channels, AVX_ALIGN);
					for (int c = 0; c < guide_channels; c++)
					{
						mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
					}
				}

				for (int n = 0; n < imsize; n += 8)
				{
					const __m256 msrc0 = *gptr0++;
					const __m256 msrc1 = *gptr1++;
					const __m256 msrc2 = *gptr2++;
					const __m256 msrc3 = *gptr3++;

					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(msrc0, mmu[k][0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						msub = _mm256_sub_ps(msrc1, mmu[k][1]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(msrc2, mmu[k][2]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						msub = _mm256_sub_ps(msrc3, mmu[k][3]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);

						_mm256_store_ps(BPtr[k] + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
					}
				}

				for (int k = 0; k < K; k++) _mm_free(mmu[k]);
			}
			else
			{
				cv::AutoBuffer<__m256*> gptr(guide_channels);
				for (int c = 0; c < guide_channels; c++) gptr[c] = (__m256*)guide[c].ptr<float>();

				cv::AutoBuffer<float*> BPtr(K);
				for (int k = 0; k < K; k++) BPtr[k] = B[k].ptr<float>();

				cv::AutoBuffer<__m256*> mmu(K);
				for (int k = 0; k < K; k++)
				{
					const float* mu_kPtr = mu.ptr<float>(k);
					mmu[k] = (__m256*)_mm_malloc(sizeof(__m256) * guide_channels, AVX_ALIGN);
					for (int c = 0; c < guide_channels; c++)
					{
						mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
					}
				}

				cv::AutoBuffer<__m256> msrc(guide_channels);

				for (int n = 0; n < imsize; n += 8)
				{
					for (int c = 0; c < guide_channels; c++) msrc[c] = *gptr[c]++;

					for (int k = 0; k < K; k++)
					{
						__m256 msub = _mm256_sub_ps(msrc[0], mmu[k][0]);
						__m256 mdiff = _mm256_mul_ps(msub, msub);
						for (int c = 1; c < guide_channels; c++)
						{
							const __m256 msub = _mm256_sub_ps(msrc[c], mmu[k][c]);
							mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
						}
						_mm256_store_ps(BPtr[k] + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
					}
				}

				for (int k = 0; k < K; k++) _mm_free(mmu[k]);
			}
		}
		else
		{
			std::cout << "should be fixed for any channel" << std::endl;
			const float* im0 = guide[0].ptr<float>();
			const float* im1 = guide[1].ptr<float>();
			const float* im2 = guide[2].ptr<float>();
			for (int k = 0; k < K; k++)
			{
				const __m256 mmu0 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][0]);
				const __m256 mmu1 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][1]);
				const __m256 mmu2 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][2]);
				float* WPtr = B[k].ptr<float>(0);

				for (int n = 0; n < imsize; n += 8)
				{
					const __m256 msrc0 = _mm256_load_ps(im0 + n);
					const __m256 msrc1 = _mm256_load_ps(im1 + n);
					const __m256 msrc2 = _mm256_load_ps(im2 + n);

					__m256 msub = _mm256_sub_ps(msrc0, mmu0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(msrc1, mmu1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(msrc2, mmu2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					_mm256_store_ps(WPtr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
				}
			}
		}
	}

	template<int use_fmath>
	void ClusteringHDKF_NystromSingle::computeBCn(const std::vector<cv::Mat>& guide)
	{
		const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
		const __m256 mcoef = _mm256_set1_ps(coeff);
		const int imsize = guide[0].size().area();

		const bool nk_loop = true;
		//k-n loop
		if (nk_loop)
		{
			cv::AutoBuffer<__m256*> gptr(guide_channels);
			for (int c = 0; c < guide_channels; c++) gptr[c] = (__m256*)guide[c].ptr<float>();

			cv::AutoBuffer<float*> WPtr(K);
			for (int k = 0; k < K; k++) WPtr[k] = B[k].ptr<float>();

			cv::AutoBuffer<__m256> msrc(guide_channels);

			cv::AutoBuffer<cv::AutoBuffer<__m256>> mmu(K);
			for (int k = 0; k < K; k++)
			{
				const float* mu_kPtr = mu.ptr<float>(k);
				mmu[k].resize(guide_channels);
				for (int c = 0; c < guide_channels; c++)
				{
					mmu[k][c] = _mm256_set1_ps(mu_kPtr[c]);
				}
			}

			for (int n = 0; n < imsize; n += 8)
			{
				for (int c = 0; c < guide_channels; c++) msrc[c] = *gptr[c]++;

				for (int k = 0; k < K; k++)
				{
					__m256 msub = _mm256_sub_ps(msrc[0], mmu[k][0]);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					for (int c = 1; c < guide_channels; c++)
					{
						const __m256 msub = _mm256_sub_ps(msrc[c], mmu[k][c]);
						mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					}
					_mm256_store_ps(WPtr[k] + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
				}
			}
		}
		else
		{
			std::cout << "should be fixed for any channel" << std::endl;
			const float* im0 = guide[0].ptr<float>();
			const float* im1 = guide[1].ptr<float>();
			const float* im2 = guide[2].ptr<float>();
			for (int k = 0; k < K; k++)
			{
				const __m256 mmu0 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][0]);
				const __m256 mmu1 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][1]);
				const __m256 mmu2 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][2]);
				float* WPtr = B[k].ptr<float>(0);

				for (int n = 0; n < imsize; n += 8)
				{
					const __m256 msrc0 = _mm256_load_ps(im0 + n);
					const __m256 msrc1 = _mm256_load_ps(im1 + n);
					const __m256 msrc2 = _mm256_load_ps(im2 + n);

					__m256 msub = _mm256_sub_ps(msrc0, mmu0);
					__m256 mdiff = _mm256_mul_ps(msub, msub);
					msub = _mm256_sub_ps(msrc1, mmu1);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					msub = _mm256_sub_ps(msrc2, mmu2);
					mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
					_mm256_store_ps(WPtr + n, v_exp_ps<use_fmath>(_mm256_mul_ps(mcoef, mdiff)));
				}
			}
		}
	}

	template<int channels>
	void ClusteringHDKF_NystromSingle::split_blur_merge()
	{
		const int imsize = img_size.area();
		const int IMSIZE = imsize / 8;

		//std::cout << img_size << "," << R << std::endl;
		cv::AutoBuffer<__m256*> src(channels);
		cv::AutoBuffer<__m256*> inter(channels);
		cv::AutoBuffer<__m256*> xf(channels);
		cv::AutoBuffer<__m256*> mnumer(channels);
		for (int k = 0; k < K; k++)
		{
			if constexpr (false)// computing U
			{
				float* eigVecPtr = eigenvecA.ptr<float>(k);
				//cost consuming part: computing X
				constexpr bool nk_loop = true;

				if constexpr (nk_loop)
				{
					float* UPtr = U.ptr<float>();//must be full sample
					cv::AutoBuffer<float*> Bptr(K);
					for (int k_w = 0; k_w < K; k_w++)
					{
						Bptr[k_w] = B[k_w].ptr<float>();
					}

					for (int n = 0; n < imsize; n += 8)
					{
						__m256 mx;
						{
							//int k_w = 0;
							mx = _mm256_mul_ps(_mm256_set1_ps(eigVecPtr[0]), _mm256_load_ps(Bptr[0]));
							Bptr[0] += 8;
						}
						for (int k_w = 1; k_w < K; k_w++)
						{
							mx = _mm256_fmadd_ps(_mm256_set1_ps(eigVecPtr[k_w]), _mm256_load_ps(Bptr[k_w]), mx);
							Bptr[k_w] += 8;
						}
						_mm256_store_ps(UPtr, mx);
						UPtr += 8;
					}
				}
				else
				{
					__m256* XPtr = (__m256*)U.ptr<float>();
					{
						//k_w = 0
						const __m256 meigvec = _mm256_set1_ps(eigVecPtr[0]);
						__m256* WPtr = (__m256*)B[0].ptr<float>();
						for (int n = 0; n < IMSIZE; n++)
						{
							*XPtr = _mm256_mul_ps(*WPtr, meigvec);
							WPtr++;
							XPtr++;
						}
					}
					for (int k_w = 1; k_w < K; k_w++)
					{
						const __m256 meigvec = _mm256_set1_ps(eigVecPtr[k_w]);
						__m256* XPtr = (__m256*)U.ptr<float>();
						__m256* WPtr = (__m256*)B[k_w].ptr<float>();
						for (int n = 0; n < IMSIZE; n++)
						{
							*XPtr = _mm256_fmadd_ps(*WPtr, meigvec, *XPtr);
							WPtr++;
							XPtr++;
						}
					}
				}

				for (int c = 0; c < channels; c++)
				{
					src[c] = (__m256*)vsrc[c].ptr<float>();
					inter[c] = (__m256*)Uf[c].ptr<float>();
				}
				const __m256* UPtr = (__m256*)U.ptr<float>();
				for (int x = 0; x < IMSIZE; x++)
				{
					__m256 u = *UPtr;
					for (int c = 0; c < channels; c++)
					{
						*inter[c] = _mm256_mul_ps(*src[c], u);
						src[c]++;
						inter[c]++;
					}
					UPtr++;
				}
			}
			else //loading U
			{
				//std::cout << "load u" << std::endl;
				for (int c = 0; c < channels; c++)
				{
					src[c] = (__m256*)vsrc[c].ptr<float>();
					inter[c] = (__m256*)Uf[c].ptr<float>();
				}
				float* UPtr = U.ptr<float>();
				const __m256* BPtr = (__m256*)B[k].ptr<float>();
				for (int x = 0; x < IMSIZE; x++)
				{
					const __m256 b = *BPtr; BPtr++;
					//_mm256_store_ps(UPtr, mx);
					_mm256_store_ps(UPtr + x * 8, b);
					//*UPtr = b; UPtr++;
					for (int c = 0; c < channels; c++)
					{
						*inter[c] = _mm256_mul_ps(*src[c], b);
						src[c]++;
						inter[c]++;
					}
				}
			}

			for (int c = 0; c < channels; c++)
			{
				GF->filter(Uf[c], Uf[c], sigma_space, spatial_order, borderType);
			}
			GF->filter(U, U_Gaussian, sigma_space, spatial_order, borderType);

			const float* lambdalPtr = lambdaA.ptr<float>(k);
			__m256 mlambdainv = (lambdalPtr[0] != 0) ? _mm256_set1_ps((1.f / lambdalPtr[0])) : _mm256_set1_ps((1.f));

			if (k == 0)
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					for (int c = 0; c < channels; c++)
					{
						xf[c] = (__m256*)Uf[c].ptr<float>(y, boundaryLength);
						mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
					}
					__m256* mX = (__m256*)U.ptr<float>(y, boundaryLength);//must be full sample
					__m256* gauss = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);
					__m256* mdenom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *mX);

						*mdenom_ = _mm256_mul_ps(mxlambda, *gauss);
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c] = _mm256_mul_ps(mxlambda, *xf[c]);
							xf[c]++, mnumer[c]++;
						}
						mX++;
						gauss++;
						mdenom_++;
					}
				}
			}
			else
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					for (int c = 0; c < channels; c++)
					{
						xf[c] = (__m256*)Uf[c].ptr<float>(y, boundaryLength);
						mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
					}
					const __m256* mX = (__m256*)U.ptr<float>(y, boundaryLength);//must be full sample
					const __m256* gauss = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);
					__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *mX);

						*denom_ = _mm256_fmadd_ps(mxlambda, *gauss, *denom_);
						for (int c = 0; c < channels; c++)
						{
							*mnumer[c] = _mm256_fmadd_ps(mxlambda, *xf[c], *mnumer[c]);
							xf[c]++;
							mnumer[c]++;
						}
						mX++;
						gauss++;
						denom_++;
					}
				}
			}
		}
	}

	//specialized for 1 and 3 or any channel
	void ClusteringHDKF_NystromSingle::split_blur_merge()
	{
		const int imsize = img_size.area();
		const int IMSIZE8 = imsize / 8;
		const int IMSIZE16 = imsize / 16;

		//std::cout << img_size << "," << R << std::endl;
		for (int k = 0; k < K; k++)
		{
			float* eigVecPtr = eigenvecA.ptr<float>(k);
			float* lambdalPtr = lambdaA.ptr<float>(k);

			//cost consuming part: computing U
			constexpr bool nk_loop = true;
			if constexpr (nk_loop)
			{
				float* XPtr = U.ptr<float>();//must be full sample
				cv::AutoBuffer<__m256*> Uptr(K);
				for (int kk = 0; kk < K; kk++)
				{
					Uptr[kk] = (__m256*)B[kk].ptr<float>();
				}

				cv::AutoBuffer<__m256> ev(K);
				for (int kk = 0; kk < K; kk++) ev[kk] = _mm256_set1_ps(eigVecPtr[kk]);
				if (channels == 1)
				{
					__m256* src0 = (__m256*)vsrc[0].ptr<float>();
					__m256* inter0 = (__m256*)Uf[0].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						__m256 mx;
						{
							//int k_w = 0;
							mx = _mm256_mul_ps(ev[0], *Uptr[0]++);
						}
						for (int k_w = 1; k_w < K; k_w++)
						{
							mx = _mm256_fmadd_ps(ev[k_w], *Uptr[k_w]++, mx);
						}
						*inter0++ = _mm256_mul_ps(*src0++, mx);
						_mm256_store_ps(XPtr, mx); XPtr += 8;
					}
				}
				else if (channels == 3)
				{
					if constexpr (isComputeU) //compute U
					{
						__m256* src0 = (__m256*)vsrc[0].ptr<float>();
						__m256* src1 = (__m256*)vsrc[1].ptr<float>();
						__m256* src2 = (__m256*)vsrc[2].ptr<float>();
						__m256* inter0 = (__m256*)Uf[0].ptr<float>();
						__m256* inter1 = (__m256*)Uf[1].ptr<float>();
						__m256* inter2 = (__m256*)Uf[2].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							__m256 mx;
							{
								//int k_w = 0;
								mx = _mm256_mul_ps(ev[0], *Uptr[0]++);
							}
							for (int k_w = 1; k_w < K; k_w++)
							{
								mx = _mm256_fmadd_ps(ev[k_w], *Uptr[k_w]++, mx);
							}
							*inter0++ = _mm256_mul_ps(*src0++, mx);
							*inter1++ = _mm256_mul_ps(*src1++, mx);
							*inter2++ = _mm256_mul_ps(*src2++, mx);
							_mm256_store_ps(XPtr, mx); XPtr += 8;
						}
					}
					else //load u
					{
						__m256* src0 = (__m256*)vsrc[0].ptr<float>();
						__m256* src1 = (__m256*)vsrc[1].ptr<float>();
						__m256* src2 = (__m256*)vsrc[2].ptr<float>();
						__m256* inter0 = (__m256*)Uf[0].ptr<float>();
						__m256* inter1 = (__m256*)Uf[1].ptr<float>();
						__m256* inter2 = (__m256*)Uf[2].ptr<float>();
						for (int n = 0; n < IMSIZE8; n++)
						{
							const __m256 mx = *Uptr[k]++;
							*inter0++ = _mm256_mul_ps(*src0++, mx);
							*inter1++ = _mm256_mul_ps(*src1++, mx);
							*inter2++ = _mm256_mul_ps(*src2++, mx);
							_mm256_store_ps(XPtr, mx); XPtr += 8;
						}
					}
				}
				else
				{
					cv::AutoBuffer<__m256*> src(channels);
					cv::AutoBuffer<__m256*> inter(channels);

					for (int c = 0; c < channels; c++)
					{
						src[c] = (__m256*)vsrc[c].ptr<float>();
						inter[c] = (__m256*)Uf[c].ptr<float>();
					}

					for (int n = 0; n < IMSIZE8; n++)
					{
						__m256 mx;
						{
							//int k_w = 0;
							mx = _mm256_mul_ps(ev[0], *Uptr[0]++);
						}
						for (int k_w = 1; k_w < K; k_w++)
						{
							mx = _mm256_fmadd_ps(ev[k_w], *Uptr[k_w]++, mx);
						}
						for (int c = 0; c < channels; c++)
						{
							*inter[c]++ = _mm256_mul_ps(*src[c]++, mx);
						}
						_mm256_store_ps(XPtr, mx); XPtr += 8;
					}
				}
				// HERE: no src copy case is required for effective downsampling
			}
			else
			{
				__m256* XPtr = (__m256*)U.ptr<float>();
				{
					//k_w = 0
					const __m256 meigvec = _mm256_set1_ps(eigVecPtr[0]);
					__m256* WPtr = (__m256*)B[0].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						*XPtr = _mm256_mul_ps(*WPtr, meigvec);
						WPtr++;
						XPtr++;
					}
				}
				for (int k_w = 1; k_w < K; k_w++)
				{
					const __m256 meigvec = _mm256_set1_ps(eigVecPtr[k_w]);
					__m256* XPtr = (__m256*)U.ptr<float>();
					__m256* WPtr = (__m256*)B[k_w].ptr<float>();
					for (int n = 0; n < IMSIZE8; n++)
					{
						*XPtr = _mm256_fmadd_ps(*WPtr, meigvec, *XPtr);
						WPtr++;
						XPtr++;
					}
				}
			}

			//blur merge
			if (channels == 1)
			{
				GF->filter(Uf[0], Uf[0], sigma_space, spatial_order, borderType);
				GF->filter(U, U_Gaussian, sigma_space, spatial_order, borderType);

				const __m256 mlambdainv = (lambdalPtr[0] != 0) ? _mm256_set1_ps((1.f / lambdalPtr[0])) : _mm256_set1_ps((1.f));

				if (k == 0)
				{
					for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
					{
						__m256* uf0 = (__m256*)Uf[0].ptr<float>(y, boundaryLength);
						__m256* u__ = (__m256*)U.ptr<float>(y, boundaryLength);
						__m256* ubl = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);

						__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
						__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

						for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
						{
							const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *u__);

							*denom_ = _mm256_mul_ps(mxlambda, *ubl);
							*numer0 = _mm256_mul_ps(mxlambda, *uf0);

							u__++; ubl++; uf0++;
							denom_++, numer0++;
						}
					}
				}
				else
				{
					for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
					{
						__m256* uf0 = (__m256*)Uf[0].ptr<float>(y, boundaryLength);
						__m256* u__ = (__m256*)U.ptr<float>(y, boundaryLength);
						__m256* ubl = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);

						__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
						__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

						for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
						{
							const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *u__);

							*denom_ = _mm256_fmadd_ps(mxlambda, *ubl, *denom_);
							*numer0 = _mm256_fmadd_ps(mxlambda, *uf0, *numer0);

							u__++; ubl++; uf0++;
							denom_++, numer0++;
						}
					}
				}

			}
			else if (channels == 3)
			{
				//downSampleSRC = 2;// not effective
				if (downSampleImage != 1)
				{
					const double res = 1.0 / downSampleImage;
					resize(Uf[0], downsampleSRC[0], cv::Size(), res, res, cv::INTER_NEAREST);
					resize(Uf[1], downsampleSRC[1], cv::Size(), res, res, cv::INTER_NEAREST);
					resize(Uf[2], downsampleSRC[2], cv::Size(), res, res, cv::INTER_NEAREST);
					resize(U, downsampleSRC[3], cv::Size(), res, res, cv::INTER_NEAREST);

					GF->filter(downsampleSRC[0], downsampleSRC[0], sigma_space * res, spatial_order, borderType);
					GF->filter(downsampleSRC[1], downsampleSRC[1], sigma_space * res, spatial_order, borderType);
					GF->filter(downsampleSRC[2], downsampleSRC[2], sigma_space * res, spatial_order, borderType);
					GF->filter(downsampleSRC[3], downsampleSRC[3], sigma_space * res, spatial_order, borderType);

					resize(downsampleSRC[0], Uf[0], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					resize(downsampleSRC[1], Uf[1], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					resize(downsampleSRC[2], Uf[2], cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
					resize(downsampleSRC[3], U_Gaussian, cv::Size(), downSampleImage, downSampleImage, cv::INTER_LINEAR);
				}
				else
				{
					GF->filter(Uf[0], Uf[0], sigma_space, spatial_order, borderType);
					GF->filter(Uf[1], Uf[1], sigma_space, spatial_order, borderType);
					GF->filter(Uf[2], Uf[2], sigma_space, spatial_order, borderType);
					GF->filter(U, U_Gaussian, sigma_space, spatial_order, borderType);
					/*cv::Mat temp;
					std::vector<cv::Mat>a(3);
					divideZeroDivZero(Uf[0], U_Gaussian, a[0],U);
					divideZeroDivZero(Uf[1], U_Gaussian, a[1],U);
					divideZeroDivZero(Uf[2], U_Gaussian, a[2],U);

					cv::merge(Uf, temp);
					//cv::merge(a, temp);
					temp.convertTo(temp, CV_8U);
					//cv::putText(temp, cv::format("k=%d", k), cv::Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 2, COLOR_WHITE);
					//imshow("a", temp);
					cp::imshowNormalize("a", temp);
					cv::waitKey();*/

				}
				const __m256 mlambdainv = (lambdalPtr[0] != 0.f) ? _mm256_set1_ps((1.f / lambdalPtr[0])) : _mm256_set1_ps((1.f));
				//const __m256 mlambdainv = (lambdalPtr[0] != 0.f) ? _mm256_set1_ps((1.f / lambdalPtr[0])) : _mm256_set1_ps((FLT_MAX));

				bool isLocalMu = false;
				if (isLocalMu)
				{
					const float coeff = float(-1.0 / (2.0 * sigma_range * sigma_range));
					const __m256 mcoef = _mm256_set1_ps(coeff);
					const __m256 mmu0 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][0]);
					const __m256 mmu1 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][1]);
					const __m256 mmu2 = _mm256_set1_ps(mu.ptr<cv::Vec3f>(k)[0][2]);

					float* eigVecPtr = eigenvecA.ptr<float>(k);
					cv::AutoBuffer<__m256> ev(K);
					const float lambda = (lambdalPtr[0] != 0.f) ? 1.f / lambdalPtr[0] : 1.f;
					for (int kk = 0; kk < K; kk++) ev[kk] = _mm256_set1_ps(eigVecPtr[kk] * lambda);
					//for (int kk = 0; kk < K; kk++) ev[kk] = _mm256_set1_ps(lambda);
					//for (int kk = 0; kk < K; kk++) ev[kk] = _mm256_set1_ps(1.f);
					if (k == 0)
					{
						for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
						{
							cv::AutoBuffer<__m256*> Bptr(K);
							for (int kk = 0; kk < K; kk++)
							{
								Bptr[kk] = (__m256*)B[kk].ptr<float>(y, boundaryLength);
							}

							const __m256* src0 = (const __m256*)vsrc[0].ptr<float>(y, boundaryLength);
							const __m256* src1 = (const __m256*)vsrc[1].ptr<float>(y, boundaryLength);
							const __m256* src2 = (const __m256*)vsrc[2].ptr<float>(y, boundaryLength);

							const __m256* uf0 = (__m256*)Uf[0].ptr<float>(y, boundaryLength);
							const __m256* uf1 = (__m256*)Uf[1].ptr<float>(y, boundaryLength);
							const __m256* uf2 = (__m256*)Uf[2].ptr<float>(y, boundaryLength);
							const __m256* u__ = (__m256*)U.ptr<float>(y, boundaryLength);
							const __m256* ubl = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);

							__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
							__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
							__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
							__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

							for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
							{
								const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *ubl);

								__m256 msub = _mm256_fnmadd_ps(*uf0, norm, *src0++);
								__m256 mdiff = _mm256_mul_ps(msub, msub);
								msub = _mm256_fnmadd_ps(*uf1, norm, *src1++);
								mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
								msub = _mm256_fnmadd_ps(*uf2, norm, *src2++);
								mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
								/*__m256 mu = _mm256_mul_ps(ev[k], v_exp_ps<true>(_mm256_mul_ps(mcoef, mdiff)));
								for (int kk = 0; kk < k; kk++)
								{
									mu = _mm256_fmadd_ps(ev[kk], *Bptr[kk]++, mu);
								}
								Bptr[k]++;
								for (int kk = k + 1; kk < K; kk++)
								{
									mu = _mm256_fmadd_ps(ev[kk], *Bptr[kk]++, mu);
								}*/
								__m256 mu = _mm256_setzero_ps();
								for (int kk = 0; kk < K; kk++)
								{
									mu = _mm256_fmadd_ps(ev[kk], *Bptr[kk]++, mu);
								}

								*denom_ = _mm256_mul_ps(mu, *ubl);
								*numer0 = _mm256_mul_ps(mu, *uf0);
								*numer1 = _mm256_mul_ps(mu, *uf1);
								*numer2 = _mm256_mul_ps(mu, *uf2);

								u__++; ubl++; uf0++; uf1++; uf2++;
								denom_++, numer0++, numer1++, numer2++;
							}
						}
					}
					else
					{
						for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
						{
							cv::AutoBuffer<__m256*> Bptr(K);
							for (int kk = 0; kk < K; kk++)
							{
								Bptr[kk] = (__m256*)B[kk].ptr<float>(y, boundaryLength);
							}

							const __m256* src0 = (const __m256*)vsrc[0].ptr<float>(y, boundaryLength);
							const __m256* src1 = (const __m256*)vsrc[1].ptr<float>(y, boundaryLength);
							const __m256* src2 = (const __m256*)vsrc[2].ptr<float>(y, boundaryLength);

							const __m256* uf0 = (__m256*)Uf[0].ptr<float>(y, boundaryLength);
							const __m256* uf1 = (__m256*)Uf[1].ptr<float>(y, boundaryLength);
							const __m256* uf2 = (__m256*)Uf[2].ptr<float>(y, boundaryLength);
							const __m256* u__ = (__m256*)U.ptr<float>(y, boundaryLength);
							const __m256* ubl = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);

							__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
							__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
							__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
							__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

							for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
							{
								const __m256 norm = _mm256_div_avoidzerodiv_ps(_mm256_set1_ps(1.f), *ubl);

								__m256 msub = _mm256_fnmadd_ps(*uf0, norm, *src0++);
								//__m256 msub = _mm256_fnmadd_ps(*uf0, norm, mmu0);
								__m256 mdiff = _mm256_mul_ps(msub, msub);
								msub = _mm256_fnmadd_ps(*uf1, norm, *src1++);
								//msub = _mm256_fnmadd_ps(*uf1, norm, mmu1);
								mdiff = _mm256_fmadd_ps(msub, msub, mdiff);
								msub = _mm256_fnmadd_ps(*uf2, norm, *src2++);
								//msub = _mm256_fnmadd_ps(*uf2, norm, mmu2);
								mdiff = _mm256_fmadd_ps(msub, msub, mdiff);


								__m256 mu = _mm256_mul_ps(ev[k], v_exp_ps<true>(_mm256_mul_ps(mcoef, mdiff)));
								//__m256 mu = _mm256_setzero_ps();
								for (int kk = 0; kk < k; kk++)
								{
									mu = _mm256_fmadd_ps(ev[kk], *Bptr[kk]++, mu);
								}
								Bptr[k]++;
								for (int kk = k + 1; kk < K; kk++)
								{
									mu = _mm256_fmadd_ps(ev[kk], *Bptr[kk]++, mu);
								}

								*denom_ = _mm256_fmadd_ps(mu, *ubl, *denom_);
								*numer0 = _mm256_fmadd_ps(mu, *uf0, *numer0);
								*numer1 = _mm256_fmadd_ps(mu, *uf1, *numer1);
								*numer2 = _mm256_fmadd_ps(mu, *uf2, *numer2);

								u__++; ubl++; uf0++; uf1++; uf2++;
								denom_++, numer0++, numer1++, numer2++;
							}
						}
					}
					//cv::Mat v; merge(numer, v); imshowScale("aa", v); cv::waitKey();
				}
				else
				{
					if (k == 0)
					{
						for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
						{
							const __m256* uf0 = (__m256*)Uf[0].ptr<float>(y, boundaryLength);
							const __m256* uf1 = (__m256*)Uf[1].ptr<float>(y, boundaryLength);
							const __m256* uf2 = (__m256*)Uf[2].ptr<float>(y, boundaryLength);
							const __m256* u__ = (__m256*)U.ptr<float>(y, boundaryLength);
							const __m256* ubl = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);

							__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
							__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
							__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
							__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

							for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
							{
								const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *u__);

								*denom_ = _mm256_mul_ps(mxlambda, *ubl);
								*numer0 = _mm256_mul_ps(mxlambda, *uf0);
								*numer1 = _mm256_mul_ps(mxlambda, *uf1);
								*numer2 = _mm256_mul_ps(mxlambda, *uf2);

								u__++; ubl++; uf0++; uf1++; uf2++;
								denom_++, numer0++, numer1++, numer2++;
							}
						}
					}
					else
					{
						for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
						{
							const __m256* uf0 = (__m256*)Uf[0].ptr<float>(y, boundaryLength);
							const __m256* uf1 = (__m256*)Uf[1].ptr<float>(y, boundaryLength);
							const __m256* uf2 = (__m256*)Uf[2].ptr<float>(y, boundaryLength);
							const __m256* u__ = (__m256*)U.ptr<float>(y, boundaryLength);
							const __m256* ubl = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);

							__m256* numer0 = (__m256*)numer[0].ptr<float>(y, boundaryLength);
							__m256* numer1 = (__m256*)numer[1].ptr<float>(y, boundaryLength);
							__m256* numer2 = (__m256*)numer[2].ptr<float>(y, boundaryLength);
							__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

							for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
							{
								const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *u__);

								*denom_ = _mm256_fmadd_ps(mxlambda, *ubl, *denom_);
								*numer0 = _mm256_fmadd_ps(mxlambda, *uf0, *numer0);
								*numer1 = _mm256_fmadd_ps(mxlambda, *uf1, *numer1);
								*numer2 = _mm256_fmadd_ps(mxlambda, *uf2, *numer2);

								u__++; ubl++; uf0++; uf1++; uf2++;
								denom_++, numer0++, numer1++, numer2++;
							}
						}
					}
				}
			}
			else
			{
				cv::AutoBuffer<__m256*> mnumer(channels);
				for (int c = 0; c < channels; c++)
				{
					GF->filter(Uf[c], Uf[c], sigma_space, spatial_order, borderType);
				}
				GF->filter(U, U_Gaussian, sigma_space, spatial_order, borderType);

				const __m256 mlambdainv = (lambdalPtr[0] != 0) ? _mm256_set1_ps((1.f / lambdalPtr[0])) : _mm256_set1_ps((1.f));

				cv::AutoBuffer<__m256*> mmUf(channels);
				if (k == 0)
				{
					for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
					{
						for (int c = 0; c < channels; c++)
						{
							mmUf[c] = (__m256*)Uf[c].ptr<float>(y, boundaryLength);
							mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
						}
						__m256* mmU = (__m256*)U.ptr<float>(y, boundaryLength);
						__m256* mmUG = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);
						__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

						for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
						{
							const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *mmU);

							*denom_ = _mm256_mul_ps(mxlambda, *mmUG);
							for (int c = 0; c < channels; c++)
							{
								*mnumer[c] = _mm256_mul_ps(mxlambda, *mmUf[c]);
								mmUf[c]++;
								mnumer[c]++;
							}
							mmU++;
							mmUG++;
							denom_++;
						}
					}
				}
				else
				{
					for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
					{
						for (int c = 0; c < channels; c++)
						{
							mmUf[c] = (__m256*)Uf[c].ptr<float>(y, boundaryLength);
							mnumer[c] = (__m256*)numer[c].ptr<float>(y, boundaryLength);
						}
						__m256* mX = (__m256*)U.ptr<float>(y, boundaryLength);
						__m256* gauss = (__m256*)U_Gaussian.ptr<float>(y, boundaryLength);
						__m256* denom_ = (__m256*)denom.ptr<float>(y, boundaryLength);

						for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
						{
							const __m256 mxlambda = _mm256_mul_ps(mlambdainv, *mX);

							*denom_ = _mm256_fmadd_ps(mxlambda, *gauss, *denom_);
							for (int c = 0; c < channels; c++)
							{
								*mnumer[c] = _mm256_fmadd_ps(mxlambda, *mmUf[c], *mnumer[c]);
								mmUf[c]++;
								mnumer[c]++;
							}
							mX++;
							gauss++;
							denom_++;
						}
					}
				}
			}
		}
	}

	/*
	static void divideZeroDivZero(cv::Mat& src, cv::Mat& div, cv::Mat& dest, cv::Mat& mask)
	{
		dest.create(src.size(), CV_32F);
		for (int i = 0; i < src.size().area(); i++)
		{
			if (mask.at<float>(i) != 0.f)
				dest.at<float>(i) = (div.at<float>(i) == 0.f) ? 0.f : src.at<float>(i) / div.at<float>(i);
			else dest.at<float>(i) = 0.f;
		}
	}
	*/

	void ClusteringHDKF_NystromSingle::normalize(cv::Mat& dst)
	{
		const bool isSafeDiv = true;
		if (isSafeDiv)
		{
			__m256 m255 = _mm256_set1_ps(255.f);
			if (channels == 1)
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					float* numer0 = numer[0].ptr<float>(y);
					float* src0 = vsrc[0].ptr<float>(y);
					float* dptr = dst.ptr<float>(y);
					float* denom_ptr = denom.ptr<float>(y);
					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						__m256 msrc0 = _mm256_load_ps(src0 + x);
						__m256 mnumer0 = _mm256_load_ps(numer0 + x);
						__m256 mdenom = _mm256_load_ps(denom_ptr + x);

						mnumer0 = _mm256_div_ps(mnumer0, mdenom);

						//mnumer0 = _mm256_div_avoidzerodiv_ps(mnumer0, mdenom);
						//mnumer1 = _mm256_div_avoidzerodiv_ps(mnumer1, mdenom);
						//mnumer2 = _mm256_div_avoidzerodiv_ps(mnumer2, mdenom);

						__m256 mask = _mm256_cmp_ps(mnumer0, _mm256_setzero_ps(), _CMP_GE_OQ);
						mnumer0 = _mm256_blendv_ps(msrc0, mnumer0, mask);
						mask = _mm256_cmp_ps(mnumer0, m255, _CMP_LE_OQ);
						mnumer0 = _mm256_blendv_ps(msrc0, mnumer0, mask);

						_mm256_store_ps(dptr + x, mnumer0);
					}
				}
			}
			else if (channels == 3)
			{
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					float* numer0 = numer[0].ptr<float>(y);
					float* numer1 = numer[1].ptr<float>(y);
					float* numer2 = numer[2].ptr<float>(y);
					float* src0 = vsrc[0].ptr<float>(y);
					float* src1 = vsrc[1].ptr<float>(y);
					float* src2 = vsrc[2].ptr<float>(y);
					float* dptr = dst.ptr<float>(y);
					float* denom_ptr = denom.ptr<float>(y);
					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 mdenom = _mm256_load_ps(denom_ptr + x);

#if 0
						//__m256 mnumer0 = _mm256_div_avoidzerodiv_ps(_mm256_load_ps(numer0 + x), mdenom);
						//__m256 mnumer1 = _mm256_div_avoidzerodiv_ps(_mm256_load_ps(numer1 + x), mdenom);
						//__m256 mnumer2 = _mm256_div_avoidzerodiv_ps(_mm256_load_ps(numer2 + x), mdenom);
						__m256 mnumer0 = _mm256_div_zerodivzero_ps(_mm256_load_ps(numer0 + x), mdenom);
						__m256 mnumer1 = _mm256_div_zerodivzero_ps(_mm256_load_ps(numer1 + x), mdenom);
						__m256 mnumer2 = _mm256_div_zerodivzero_ps(_mm256_load_ps(numer2 + x), mdenom);
#else
						__m256 mnumer0 = _mm256_div_ps(_mm256_load_ps(numer0 + x), mdenom);
						__m256 mnumer1 = _mm256_div_ps(_mm256_load_ps(numer1 + x), mdenom);
						__m256 mnumer2 = _mm256_div_ps(_mm256_load_ps(numer2 + x), mdenom);
#endif
						const __m256 msrc0 = _mm256_load_ps(src0 + x);
						const __m256 msrc1 = _mm256_load_ps(src1 + x);
						const __m256 msrc2 = _mm256_load_ps(src2 + x);

						__m256 mask = _mm256_cmp_ps(mnumer0, _mm256_setzero_ps(), _CMP_GE_OQ);
						mnumer0 = _mm256_blendv_ps(msrc0, mnumer0, mask);
						mask = _mm256_cmp_ps(mnumer0, m255, _CMP_LE_OQ);
						mnumer0 = _mm256_blendv_ps(msrc0, mnumer0, mask);

						mask = _mm256_cmp_ps(mnumer1, _mm256_setzero_ps(), _CMP_GE_OQ);
						mnumer1 = _mm256_blendv_ps(msrc1, mnumer1, mask);
						mask = _mm256_cmp_ps(mnumer1, m255, _CMP_LE_OQ);
						mnumer1 = _mm256_blendv_ps(msrc1, mnumer1, mask);

						mask = _mm256_cmp_ps(mnumer2, _mm256_setzero_ps(), _CMP_GE_OQ);
						mnumer2 = _mm256_blendv_ps(msrc2, mnumer2, mask);
						mask = _mm256_cmp_ps(mnumer2, m255, _CMP_LE_OQ);
						mnumer2 = _mm256_blendv_ps(msrc2, mnumer2, mask);

						_mm256_store_ps_color(dptr + 3 * x, mnumer0, mnumer1, mnumer2);
					}
				}
			}
			else
			{
				cv::AutoBuffer<float*> numerptr(channels);
				cv::AutoBuffer<float*> src(channels);
				cv::AutoBuffer<__m256> mnumer(channels);
				cv::AutoBuffer<__m256> msrc(channels);
				__m256 mask;
				for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
				{
					for (int c = 0; c < channels; c++)
					{
						numerptr[c] = numer[c].ptr<float>(y);
						src[c] = vsrc[c].ptr<float>(y);
					}
					float* dptr = dst.ptr<float>(y);
					float* denom_ptr = denom.ptr<float>(y);
					for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
					{
						const __m256 mdenom = _mm256_load_ps(denom_ptr + x);
						for (int c = 0; c < channels; c++)
						{
							mnumer[c] = _mm256_div_ps(_mm256_load_ps(numerptr[c] + x), mdenom);
							msrc[c] = _mm256_load_ps(src[c] + x);

							mask = _mm256_cmp_ps(mnumer[c], _mm256_setzero_ps(), _CMP_GE_OQ);
							mnumer[c] = _mm256_blendv_ps(msrc[c], mnumer[c], mask);
							mask = _mm256_cmp_ps(mnumer[c], m255, _CMP_LE_OQ);
							mnumer[c] = _mm256_blendv_ps(msrc[c], mnumer[c], mask);

							for (int s = 0; s < 8; s++)
							{
								dptr[channels * (x + s) + c] = mnumer[c].m256_f32[s];
							}
						}
					}
				}
			}
		}
		else
		{
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				float* numer0 = numer[0].ptr<float>(y);
				float* numer1 = numer[1].ptr<float>(y);
				float* numer2 = numer[2].ptr<float>(y);
				float* dptr = dst.ptr<float>(y);
				float* denom_ptr = denom.ptr<float>(y);
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 mnumer0 = _mm256_load_ps(numer0 + x);
					__m256 mnumer1 = _mm256_load_ps(numer1 + x);
					__m256 mnumer2 = _mm256_load_ps(numer2 + x);
					__m256 mdenom = _mm256_load_ps(denom_ptr + x);
					mnumer0 = _mm256_div_ps(mnumer0, mdenom);
					mnumer1 = _mm256_div_ps(mnumer1, mdenom);
					mnumer2 = _mm256_div_ps(mnumer2, mdenom);
					//mnumer0 = _mm256_div_avoidzerodiv_ps(mnumer0, mdenom);
					//mnumer1 = _mm256_div_avoidzerodiv_ps(mnumer1, mdenom);
					//mnumer2 = _mm256_div_avoidzerodiv_ps(mnumer2, mdenom);

					_mm256_store_ps_color(dptr + 3 * x, mnumer0, mnumer1, mnumer2);
				}
			}
		}
	}

	template<int channels>
	void ClusteringHDKF_NystromSingle::normalize(cv::Mat& dst)
	{
		const bool isSafeDiv = true;
		if (isSafeDiv)
		{
			__m256 m255 = _mm256_set1_ps(255.f);
			cv::AutoBuffer<float*> numerptr(channels);
			cv::AutoBuffer<float*> src(channels);
			cv::AutoBuffer<__m256> mnumer(channels);
			cv::AutoBuffer<__m256> msrc(channels);
			__m256 mask;
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				for (int c = 0; c < channels; c++)
				{
					numerptr[c] = numer[c].ptr<float>(y);
					src[c] = vsrc[c].ptr<float>(y);
				}
				float* dptr = dst.ptr<float>(y);
				float* denom_ptr = denom.ptr<float>(y);
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					const __m256 mdenom = _mm256_load_ps(denom_ptr + x);
					for (int c = 0; c < channels; c++)
					{
						mnumer[c] = _mm256_div_ps(_mm256_load_ps(numerptr[c] + x), mdenom);
						msrc[c] = _mm256_load_ps(src[c] + x);

						mask = _mm256_cmp_ps(mnumer[c], _mm256_setzero_ps(), _CMP_GE_OQ);
						mnumer[c] = _mm256_blendv_ps(msrc[c], mnumer[c], mask);
						mask = _mm256_cmp_ps(mnumer[c], m255, _CMP_LE_OQ);
						mnumer[c] = _mm256_blendv_ps(msrc[c], mnumer[c], mask);

						dptr[channels * (x + 0) + c] = mnumer[c].m256_f32[0];
						dptr[channels * (x + 1) + c] = mnumer[c].m256_f32[1];
						dptr[channels * (x + 2) + c] = mnumer[c].m256_f32[2];
						dptr[channels * (x + 3) + c] = mnumer[c].m256_f32[3];
						dptr[channels * (x + 4) + c] = mnumer[c].m256_f32[4];
						dptr[channels * (x + 5) + c] = mnumer[c].m256_f32[5];
						dptr[channels * (x + 6) + c] = mnumer[c].m256_f32[6];
						dptr[channels * (x + 7) + c] = mnumer[c].m256_f32[7];
					}
				}
			}
		}
		else
		{
			for (int y = boundaryLength; y < img_size.height - boundaryLength; y++)
			{
				float* numer0 = numer[0].ptr<float>(y);
				float* numer1 = numer[1].ptr<float>(y);
				float* numer2 = numer[2].ptr<float>(y);
				float* dptr = dst.ptr<float>(y);
				float* denom_ptr = denom.ptr<float>(y);
				for (int x = boundaryLength; x < img_size.width - boundaryLength; x += 8)
				{
					__m256 mnumer0 = _mm256_load_ps(numer0 + x);
					__m256 mnumer1 = _mm256_load_ps(numer1 + x);
					__m256 mnumer2 = _mm256_load_ps(numer2 + x);
					__m256 mdenom = _mm256_load_ps(denom_ptr + x);
					mnumer0 = _mm256_div_ps(mnumer0, mdenom);
					mnumer1 = _mm256_div_ps(mnumer1, mdenom);
					mnumer2 = _mm256_div_ps(mnumer2, mdenom);
					//mnumer0 = _mm256_div_avoidzerodiv_ps(mnumer0, mdenom);
					//mnumer1 = _mm256_div_avoidzerodiv_ps(mnumer1, mdenom);
					//mnumer2 = _mm256_div_avoidzerodiv_ps(mnumer2, mdenom);

					_mm256_store_ps_color(dptr + 3 * x, mnumer0, mnumer1, mnumer2);
				}
			}
		}
	}

	void ClusteringHDKF_NystromSingle::body(const std::vector<cv::Mat>& src, cv::Mat& dst, const std::vector<cv::Mat>& guide)
	{
		{
			//timer[0].start();
			alloc(dst);
			//timer[0].getpushLapTime();
		}

		{
			//timer[1].start();
			clustering();//K*res-imsize*iter
			//timer[1].getpushLapTime();
		}
		/*for (int i = 0; i < K; i++)
		{
			float* p = mu.ptr<float>(i);
			for (int c = 0; c < guide.size(); c++)
			{
				std::cout << p[c] << " ";
			}
			std::cout << std::endl;
		}*/

		{
			//timer[2].start();
			computeAandEVD(mu, lambdaA, eigenvecA);//KxK
			//timer[2].getpushLapTime();
		}
		{
			//timer[3].start();
			{
				std::vector<cv::Mat> signal = (isJoint) ? guide : src;

				//fast math case
				switch (signal.size())
				{
				case 1: computeB<1, 1>(signal); break;
				case 2: computeB<1, 2>(signal); break;
				case 3: computeB<1, 3>(signal); break;
				case 4: computeB<1, 4>(signal); break;
				case 5: computeB<1, 5>(signal); break;
				case 6: computeB<1, 6>(signal); break;
				case 7: computeB<1, 7>(signal); break;
				case 8: computeB<1, 8>(signal); break;
				case 9: computeB<1, 9>(signal); break;
				case 10: computeB<1, 10>(signal); break;
				case 11: computeB<1, 11>(signal); break;
				case 12: computeB<1, 12>(signal); break;
				case 13: computeB<1, 13>(signal); break;
				case 14: computeB<1, 14>(signal); break;
				case 15: computeB<1, 15>(signal); break;
				default:
					if (isUseFmath) computeBCn<1>(signal);
					else computeBCn<0>(signal);
					//std::cout << "do not define " << guide.size() << " channel" << std::endl;
					break;
				}
			}
			//timer[3].getpushLapTime();
		}
		{
			//timer[4].start();
			if (channels == 33) split_blur_merge<33>();
			else split_blur_merge();
			//timer[4].getpushLapTime();
		}

		{
			//timer[5].start();
			if (channels == 33) normalize<33>(dst);
			else normalize(dst);
			//timer[5].getpushLapTime();
		}
	}
}