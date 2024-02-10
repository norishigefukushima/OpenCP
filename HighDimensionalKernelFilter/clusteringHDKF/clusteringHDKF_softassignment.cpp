#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

namespace cp
{
	void ClusteringHDKF_SoftAssignment::init(const cv::Mat& src, cv::Mat& dst)
	{
		//	src.convertTo(input_image32f, CV_32FC3, 1.0 / 255.0);
		src.convertTo(input_image32f, CV_32FC3);
		cv::split(input_image32f, split_image);
		switch (cm)
		{
		case ClusterMethod::mediancut_median:
		case ClusterMethod::mediancut_max:
		case ClusterMethod::mediancut_min:

		case ClusterMethod::quantize_wan:
		case ClusterMethod::kmeans_wan:
		case ClusterMethod::quantize_wu:
		case ClusterMethod::kmeans_wu:
		case ClusterMethod::quantize_neural:
		case ClusterMethod::kmeans_neural:
		case ClusterMethod::quantize_DIV:
		case ClusterMethod::kmeans_DIV:
		case ClusterMethod::quantize_PNN:
		case ClusterMethod::kmeans_PNN:
		case ClusterMethod::quantize_SPA:
		case ClusterMethod::kmeans_SPA:
		case ClusterMethod::quantize_EAS:
		case ClusterMethod::kmeans_EAS:
			src.convertTo(input_image8u, CV_8UC3);
			break;
		default:
			break;
		}

		//	double sr = sigma_range / 255.0;
		double sr = sigma_range;

		coef = float(-0.5 * (1.0 / (sr * sr)));

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_32FC3);

		if (W2_sum.size() != img_size)
		{
			W2_sum.create(img_size, CV_32FC1);
		}

		if (split_inter2.size() != 3 || split_inter2[0].size() != img_size)
		{
			split_inter2.resize(3);
			split_inter2[0].create(img_size, CV_32FC1);
			split_inter2[1].create(img_size, CV_32FC1);
			split_inter2[2].create(img_size, CV_32FC1);
		}

		if (downsampleMethod == DownsampleMethod::IMPORTANCE_MAP) src.convertTo(input_image8u, CV_8UC3);


		if (cm == ClusterMethod::X_means)
		{
			return;
		}

		if (vecW.size() != K || vecW[0].size() != img_size)
		{
			vecW.resize(K);
			for (int i = 0; i < K; i++)
			{
				vecW[i].create(img_size, CV_32FC1);
			}
		}

		if (W2.size() != K || W2[0].size() != img_size)
		{
			W2.resize(K);
			for (int i = 0; i < K; i++)
			{
				W2[i].create(img_size, CV_32FC1);
			}
		}

		if (split_inter.size() != K || split_inter[0].size() != 3 || split_inter[0][0].size() != img_size)
		{
			split_inter.resize(K);
			for (int k = 0; k < K; k++)
			{
				split_inter[k].resize(3);
			}
			for (int k = 0; k < K; k++)
			{
				split_inter[k][0].create(img_size, CV_32FC1);
				split_inter[k][1].create(img_size, CV_32FC1);
				split_inter[k][2].create(img_size, CV_32FC1);
			}
		}
	}

	void ClusteringHDKF_SoftAssignment::calcAlpha()
	{
		__m256 mcoef = _mm256_set1_ps(coef);
		//	__m256 mcoef2 = _mm256_set1_ps(-lambda * 255.f);
		__m256 mcoef2 = _mm256_set1_ps(-lambda / 255.f);


		W2_sum = cv::Mat::zeros(img_size, CV_32FC1);
		omp_set_dynamic(1);

#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			__m256 mcenter0 = _mm256_set1_ps(centers.ptr<cv::Vec3f>(k)[0][0]);
			__m256 mcenter1 = _mm256_set1_ps(centers.ptr<cv::Vec3f>(k)[0][1]);
			__m256 mcenter2 = _mm256_set1_ps(centers.ptr<cv::Vec3f>(k)[0][2]);

			__m256 mimage0, mimage1, mimage2;
			__m256 msum;


			for (int y = 0; y < img_size.height; y++)
			{
				float* im0 = split_image[0].ptr<float>(y);
				float* im1 = split_image[1].ptr<float>(y);
				float* im2 = split_image[2].ptr<float>(y);

				float* vecw_ptr = vecW[k].ptr<float>(y);
				float* w2_ptr = W2[k].ptr<float>(y);
				float* w2_sum = W2_sum.ptr<float>(y);

				for (int x = 0; x < img_size.width; x += 8)
				{
					mimage0 = _mm256_load_ps(im0 + x);
					mimage1 = _mm256_load_ps(im1 + x);
					mimage2 = _mm256_load_ps(im2 + x);

					mimage0 = _mm256_sub_ps(mimage0, mcenter0);
					mimage1 = _mm256_sub_ps(mimage1, mcenter1);
					mimage2 = _mm256_sub_ps(mimage2, mcenter2);

					mimage0 = _mm256_mul_ps(mimage0, mimage0);
					mimage1 = _mm256_mul_ps(mimage1, mimage1);
					mimage2 = _mm256_mul_ps(mimage2, mimage2);

					msum = _mm256_add_ps(mimage2, _mm256_add_ps(mimage0, mimage1));

					_mm256_store_ps(vecw_ptr + x, _mm256_exp_ps(_mm256_mul_ps(mcoef, msum)));
					_mm256_store_ps(w2_ptr + x, _mm256_exp_ps(_mm256_mul_ps(mcoef2, msum)));
				}
			}
		}

#pragma omp parallel for
		for (int y = 0; y < img_size.height; ++y)
		{
			//k=0
			{
				float* w2_ptr = W2[0].ptr<float>(y);
				float* w2_sum = W2_sum.ptr<float>(y);
				for (int x = 0; x < img_size.width; x += 8)
				{
					_mm256_store_ps(w2_sum + x, _mm256_add_ps(_mm256_load_ps(w2_sum + x), _mm256_load_ps(w2_ptr + x)));
				}
			}

			for (int k = 1; k < K; k++)
			{
				float* w2_ptr = W2[k].ptr<float>(y);
				float* w2_sum = W2_sum.ptr<float>(y);
				for (int x = 0; x < img_size.width; x += 8)
				{
					_mm256_store_ps(w2_sum + x, _mm256_add_ps(_mm256_load_ps(w2_sum + x), _mm256_load_ps(w2_ptr + x)));
				}
			}
		}

#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			for (int y = 0; y < img_size.height; y++)
			{
				float* w2_ptr = W2[k].ptr<float>(y);
				float* w2_sum = W2_sum.ptr<float>(y);
				for (int x = 0; x < img_size.width; x += 8)
				{
					_mm256_store_ps(w2_ptr + x, _mm256_div_ps(_mm256_load_ps(w2_ptr + x), _mm256_load_ps(w2_sum + x)));
				}
			}
		}

	}

	void ClusteringHDKF_SoftAssignment::mul_add_gaussian()
	{
		// íÜä‘âÊëú
		for (int i = 0; i < 3; i++)
		{
			split_inter2[i] = cv::Mat::zeros(img_size, CV_32FC1);
		}

#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			__m256 msrc0, msrc1, msrc2;
			__m256 mvecw;

			// inter = src.mul(W);
			for (int y = 0; y < img_size.height; y++)
			{
				float* src0 = split_image[0].ptr<float>(y);
				float* src1 = split_image[1].ptr<float>(y);
				float* src2 = split_image[2].ptr<float>(y);
				float* inter0 = split_inter[k][0].ptr<float>(y);
				float* inter1 = split_inter[k][1].ptr<float>(y);
				float* inter2 = split_inter[k][2].ptr<float>(y);
				float* vecw_ptr = vecW[k].ptr<float>(y);

				for (int x = 0; x < img_size.width; x += 8)
				{
					mvecw = _mm256_load_ps(vecw_ptr + x);
					msrc0 = _mm256_load_ps(src0 + x);
					msrc1 = _mm256_load_ps(src1 + x);
					msrc2 = _mm256_load_ps(src2 + x);

					_mm256_store_ps(inter0 + x, _mm256_mul_ps(mvecw, msrc0));
					_mm256_store_ps(inter1 + x, _mm256_mul_ps(mvecw, msrc1));
					_mm256_store_ps(inter2 + x, _mm256_mul_ps(mvecw, msrc2));
				}
			}

			int thread_num = omp_get_thread_num();

			GF[thread_num]->filter(vecW[k], vecW[k], sigma_space, spatial_order);
			GF[thread_num]->filter(split_inter[k][0], split_inter[k][0], sigma_space, spatial_order);
			GF[thread_num]->filter(split_inter[k][1], split_inter[k][1], sigma_space, spatial_order);
			GF[thread_num]->filter(split_inter[k][2], split_inter[k][2], sigma_space, spatial_order);
		}

		// åWêîÇ©ÇØÇÈ
#pragma omp parallel for
		for (int y = 0; y < img_size.height; y++)
		{
			__m256 mvecw, mw2;
			__m256 minter0, minter1, minter2;
			__m256 minter2_0, minter2_1, minter2_2;

			float* inter2_0 = split_inter2[0].ptr<float>(y);
			float* inter2_1 = split_inter2[1].ptr<float>(y);
			float* inter2_2 = split_inter2[2].ptr<float>(y);

			// k=0
			{
				float* inter0 = split_inter[0][0].ptr<float>(y);
				float* inter1 = split_inter[0][1].ptr<float>(y);
				float* inter2 = split_inter[0][2].ptr<float>(y);
				float* vecw_ptr = vecW[0].ptr<float>(y);
				float* w2_ptr = W2[0].ptr<float>(y);
				for (int x = 0; x < img_size.width; x += 8)
				{
					mvecw = _mm256_load_ps(vecw_ptr + x);
					mw2 = _mm256_load_ps(w2_ptr + x);
					minter0 = _mm256_load_ps(inter0 + x);
					minter1 = _mm256_load_ps(inter1 + x);
					minter2 = _mm256_load_ps(inter2 + x);
					minter2_0 = _mm256_load_ps(inter2_0 + x);
					minter2_1 = _mm256_load_ps(inter2_1 + x);
					minter2_2 = _mm256_load_ps(inter2_2 + x);

					_mm256_store_ps(inter2_0 + x, _mm256_fmadd_ps(mw2, _mm256_div_ps(minter0, mvecw), minter2_0));
					_mm256_store_ps(inter2_1 + x, _mm256_fmadd_ps(mw2, _mm256_div_ps(minter1, mvecw), minter2_1));
					_mm256_store_ps(inter2_2 + x, _mm256_fmadd_ps(mw2, _mm256_div_ps(minter2, mvecw), minter2_2));
				}
			}

			for (int k = 1; k < K; k++)
			{
				float* inter0 = split_inter[k][0].ptr<float>(y);
				float* inter1 = split_inter[k][1].ptr<float>(y);
				float* inter2 = split_inter[k][2].ptr<float>(y);
				float* vecw_ptr = vecW[k].ptr<float>(y);
				float* w2_ptr = W2[k].ptr<float>(y);
				for (int x = 0; x < img_size.width; x += 8)
				{
					mvecw = _mm256_load_ps(vecw_ptr + x);
					mw2 = _mm256_load_ps(w2_ptr + x);
					minter0 = _mm256_load_ps(inter0 + x);
					minter1 = _mm256_load_ps(inter1 + x);
					minter2 = _mm256_load_ps(inter2 + x);
					minter2_0 = _mm256_load_ps(inter2_0 + x);
					minter2_1 = _mm256_load_ps(inter2_1 + x);
					minter2_2 = _mm256_load_ps(inter2_2 + x);

					_mm256_store_ps(inter2_0 + x, _mm256_fmadd_ps(mw2, _mm256_div_ps(minter0, mvecw), minter2_0));
					_mm256_store_ps(inter2_1 + x, _mm256_fmadd_ps(mw2, _mm256_div_ps(minter1, mvecw), minter2_1));
					_mm256_store_ps(inter2_2 + x, _mm256_fmadd_ps(mw2, _mm256_div_ps(minter2, mvecw), minter2_2));
				}
			}
		}
	}

	void ClusteringHDKF_SoftAssignment::xmeans_init(const cv::Mat& src, cv::Mat& dst)
	{
		if (vecW.size() != K || vecW[0].size() != img_size)
		{
			vecW.resize(K);
			for (int i = 0; i < K; i++)
			{
				vecW[i].create(img_size, CV_32FC1);
			}
		}

		if (W2.size() != K || W2[0].size() != img_size)
		{
			W2.resize(K);
			for (int i = 0; i < K; i++)
			{
				W2[i].create(img_size, CV_32FC1);
			}
		}

		if (split_inter.size() != K || split_inter[0].size() != 3 || split_inter[0][0].size() != img_size)
		{
			split_inter.resize(K);
			for (int k = 0; k < K; k++)
			{
				split_inter[k].resize(3);
			}
			for (int k = 0; k < K; k++)
			{
				split_inter[k][0].create(img_size, CV_32FC1);
				split_inter[k][1].create(img_size, CV_32FC1);
				split_inter[k][2].create(img_size, CV_32FC1);
			}
		}
	}


	cv::Mat ClusteringHDKF_SoftAssignment::get_centers()
	{
		return this->centers;
	}

	void ClusteringHDKF_SoftAssignment::set_labels(const cv::Mat& labels)
	{
		this->labels = labels;
	}

	void ClusteringHDKF_SoftAssignment::set_centers(const cv::Mat& centers)
	{
		this->centers = centers;
	}

	void ClusteringHDKF_SoftAssignment::body(const cv::Mat& src, cv::Mat& dst, const cv::Mat& guide)
	{
		init(src, dst);
		clustering();
		if (cm == ClusterMethod::X_means) xmeans_init(src, dst);
		calcAlpha();
		mul_add_gaussian();
		merge(split_inter2, dst);
		//	dst.convertTo(dst, CV_32FC3, 255.0);
		dst.convertTo(dst, CV_32FC3);
	}

	void ClusteringHDKF_SoftAssignment::filtering(const cv::Mat& src, cv::Mat& dst)
	{
		init(src, dst);
		clustering();
		if (cm == ClusterMethod::X_means) xmeans_init(src, dst);
		calcAlpha();
		mul_add_gaussian();
		merge(split_inter2, dst);
		//	dst.convertTo(dst, CV_32FC3, 255.0);
		dst.convertTo(dst, CV_32FC3);
	}

	void ClusteringHDKF_SoftAssignment::filtering(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod)
	{
		if (src.channels() != 3)
		{
			std::cout << "channels is not 3" << std::endl;
			assert(src.channels() == 3);
		}

		setParameter(src.size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod);


		filtering(src, dst);
	}


	void ClusteringHDKF_SoftAssignment::jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod)
	{
		std::cout << "not implemented ConstantTimeHDGF_SoftAssignment::jointfilter" << std::endl;
		if (src.channels() != 3)
		{
			std::cout << "channels is not 3" << std::endl;
			assert(src.channels() == 3);
		}

		setParameter(src.size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod);


		filtering(src, dst);
	}
}