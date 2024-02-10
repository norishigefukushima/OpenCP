#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

namespace cp
{
	void ClusteringHDKF_Nystrom::init(const cv::Mat& src, cv::Mat& dst)
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
		if (split_numer.size() != 3 || split_numer[0].size() != img_size)
		{
			split_numer.resize(3);
			split_numer[0].create(img_size, CV_32FC1);
			split_numer[1].create(img_size, CV_32FC1);
			split_numer[2].create(img_size, CV_32FC1);
		}

		//	double sr = sigma_range / 255.0;
		double sr = sigma_range;
		coef = float(-0.5 * (1.0 / (sr * sr)));

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_32FC3);

		if (denom.size() != img_size)
		{
			denom.create(img_size, CV_32FC1);
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

		if (A.size() != cv::Size(K, K))
		{
			A.create(cv::Size(K, K), CV_32FC1);
		}


		if (inter_numer.size() != 3 || inter_numer[0].size() != K || inter_numer[0][0].size() != img_size)
		{
			inter_numer.resize(3);
			inter_numer[0].resize(K);
			inter_numer[1].resize(K);
			inter_numer[2].resize(K);
			for (int k = 0; k < K; k++)
			{
				inter_numer[0][k].create(img_size, CV_32FC1);
				inter_numer[1][k].create(img_size, CV_32FC1);
				inter_numer[2][k].create(img_size, CV_32FC1);
			}
		}

		if (inter_denom.size() != K || inter_denom[0].size() != img_size)
		{
			inter_denom.resize(K);
			for (int k = 0; k < K; k++)
			{
				inter_denom[k].create(img_size, CV_32FC1);
			}
		}

		for (int k = 0; k < K; k++)
		{
			inter_denom[k] = cv::Mat::zeros(img_size, CV_32FC1);
		}
	}

	void ClusteringHDKF_Nystrom::calcVecW()
	{
		__m256 mcoef = _mm256_set1_ps(coef);

		omp_set_dynamic(1);
#pragma omp parallel for
		for (int i = 0; i < K; i++)
		{
			__m256 mcenter0 = _mm256_set1_ps(centers.ptr<cv::Vec3f>(i)[0][0]);
			__m256 mcenter1 = _mm256_set1_ps(centers.ptr<cv::Vec3f>(i)[0][1]);
			__m256 mcenter2 = _mm256_set1_ps(centers.ptr<cv::Vec3f>(i)[0][2]);

			__m256 mimage0, mimage1, mimage2;
			__m256 msum;


			for (int y = 0; y < img_size.height; y++)
			{
				float* im0 = split_image[0].ptr<float>(y);
				float* im1 = split_image[1].ptr<float>(y);
				float* im2 = split_image[2].ptr<float>(y);

				float* vecw_ptr = vecW[i].ptr<float>(y);

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

					msum = _mm256_mul_ps(mcoef, _mm256_add_ps(mimage2, _mm256_add_ps(mimage0, mimage1)));
					_mm256_store_ps(vecw_ptr + x, _mm256_exp_ps(msum));
				}
			}
		}
	}

	void ClusteringHDKF_Nystrom::calcA()
	{
#pragma omp parallel for
		for (int i = 0; i < K; i++)
		{
			float* a_i = A.ptr<float>(i);
			a_i[i] = 1;
			//A.at<float>(i, i) = 1;
			for (int j = i + 1; j < K; j++)
			{
				float* a_j = A.ptr<float>(j);
				//A.at<float>(i, j)
				a_i[j] = std::exp(coef * ((centers.ptr<cv::Vec3f>(i)[0][0] - centers.ptr<cv::Vec3f>(j)[0][0]) * (centers.ptr<cv::Vec3f>(i)[0][0] - centers.ptr<cv::Vec3f>(j)[0][0]) +
					(centers.ptr<cv::Vec3f>(i)[0][1] - centers.ptr<cv::Vec3f>(j)[0][1]) * (centers.ptr<cv::Vec3f>(i)[0][1] - centers.ptr<cv::Vec3f>(j)[0][1]) +
					(centers.ptr<cv::Vec3f>(i)[0][2] - centers.ptr<cv::Vec3f>(j)[0][2]) * (centers.ptr<cv::Vec3f>(i)[0][2] - centers.ptr<cv::Vec3f>(j)[0][2])));
				//A.at<float>(j, i);
				a_j[i] = a_i[j];
			}
		}
		//std::cout << A << std::endl;
		//std::cout << A.size() << std::endl;
	}

	// inter_numer, inter_denom計算
	void ClusteringHDKF_Nystrom::mul_add_gaussian()
	{
		// 固有値分解
		cv::eigen(A, D, V);
		// 固有値大きい順になってる，固有ベクトルはrow

#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			std::vector<cv::Mat> split_inter;
			split_inter.resize(3);
			split_inter[0].create(img_size, CV_32FC1);
			split_inter[1].create(img_size, CV_32FC1);
			split_inter[2].create(img_size, CV_32FC1);

			int thread_num = omp_get_thread_num();
			//cv::Mat eigmat, eigmat2, eig_gaussian;
			//eigmat.create(img_size, CV_32FC1);
			//eigmat2.create(img_size, CV_32FC1);

			cv::Mat eigmat = cv::Mat::zeros(img_size, CV_32FC1);
			cv::Mat eigmat2 = cv::Mat::zeros(img_size, CV_32FC1);
			cv::Mat eig_gaussian;

			float* eigVec = V.ptr<float>(k);
			float* eigVal = D.ptr<float>(k);

			for (int i = 0; i < K; i++)
			{
				__m256 meigvec = _mm256_set1_ps(eigVec[i]);
				__m256 meigmat, mvecw;
				for (int y = 0; y < img_size.height; y++)
				{
					float* eigMat = eigmat.ptr<float>(y);
					float* vecw = vecW[i].ptr<float>(y);

					for (int x = 0; x < img_size.width; x += 8)
					{
						meigmat = _mm256_load_ps(eigMat + x);
						mvecw = _mm256_load_ps(vecw + x);
						_mm256_store_ps(eigMat + x, _mm256_fmadd_ps(mvecw, meigvec, meigmat));
						//_mm256_store_ps(eigMat + x, _mm256_fmadd_ps(_mm256_load_ps(vecw + x), meigvec, _mm256_load_ps(eigMat + x)));
					}
				}
			}

			GF[thread_num]->filter(eigmat, eig_gaussian, sigma_space, spatial_order, border);

			if (eigVal[0] != 0)
			{
				__m256 meigmat;
				__m256 meigval = _mm256_set1_ps((1.f / eigVal[0]));
				for (int y = 0; y < img_size.height; y++)
				{
					float* eigMat = eigmat.ptr<float>(y);
					float* eigMat2 = eigmat2.ptr<float>(y);

					for (int x = 0; x < img_size.width; x += 8)
					{
						meigmat = _mm256_load_ps(eigMat + x);
						_mm256_store_ps(eigMat2 + x, _mm256_mul_ps(meigmat, meigval));
					}
				}
			}
			else
			{
				;
			}

			__m256 meigmat, meigmat2;
			__m256 meiggauss;
			__m256 mdenom;

			for (int y = 0; y < img_size.height; y++)
			{
				float* denom_ptr = inter_denom[k].ptr<float>(y);
				float* eigMat2 = eigmat2.ptr<float>(y);
				float* eigGauss = eig_gaussian.ptr<float>(y);

				//std::cout << *denom_ptr << std::endl;
				for (int x = 0; x < img_size.width; x += 8)
				{
					meigmat2 = _mm256_load_ps(eigMat2 + x);
					meiggauss = _mm256_load_ps(eigGauss + x);
					mdenom = _mm256_load_ps(denom_ptr + x);
					_mm256_store_ps(denom_ptr + x, _mm256_fmadd_ps(meigmat2, meiggauss, mdenom));
					//_mm256_store_ps(denom_ptr + x, _mm256_fmadd_ps(_mm256_load_ps(eigMat2 + x), _mm256_load_ps(eigGauss + x), _mm256_load_ps(denom_ptr + x)));
				}
			}

			__m256 minter0, minter1, minter2;
			__m256 mnumer0, mnumer1, mnumer2;

			for (int y = 0; y < img_size.height; y++)
			{
				float* inter0 = split_inter[0].ptr<float>(y);
				float* inter1 = split_inter[1].ptr<float>(y);
				float* inter2 = split_inter[2].ptr<float>(y);

				float* s_image0 = split_image[0].ptr<float>(y);
				float* s_image1 = split_image[1].ptr<float>(y);
				float* s_image2 = split_image[2].ptr<float>(y);
				float* eigMat = eigmat.ptr<float>(y);

				for (int x = 0; x < img_size.width; x += 8)
				{
					meigmat = _mm256_load_ps(eigMat + x);
					minter0 = _mm256_load_ps(s_image0 + x);
					minter1 = _mm256_load_ps(s_image1 + x);
					minter2 = _mm256_load_ps(s_image2 + x);

					_mm256_store_ps(inter0 + x, _mm256_mul_ps(minter0, meigmat));
					_mm256_store_ps(inter1 + x, _mm256_mul_ps(minter1, meigmat));
					_mm256_store_ps(inter2 + x, _mm256_mul_ps(minter2, meigmat));
				}
			}

			GF[thread_num]->filter(split_inter[0], split_inter[0], sigma_space, spatial_order, border);
			GF[thread_num]->filter(split_inter[1], split_inter[1], sigma_space, spatial_order, border);
			GF[thread_num]->filter(split_inter[2], split_inter[2], sigma_space, spatial_order, border);


			for (int y = 0; y < img_size.height; y++)
			{
				float* inter0 = split_inter[0].ptr<float>(y);
				float* inter1 = split_inter[1].ptr<float>(y);
				float* inter2 = split_inter[2].ptr<float>(y);
				float* eigMat2 = eigmat2.ptr<float>(y);

				float* numer0 = inter_numer[0][k].ptr<float>(y);
				float* numer1 = inter_numer[1][k].ptr<float>(y);
				float* numer2 = inter_numer[2][k].ptr<float>(y);

				for (int x = 0; x < img_size.width; x += 8)
				{
					meigmat2 = _mm256_load_ps(eigMat2 + x);
					minter0 = _mm256_load_ps(inter0 + x);
					minter1 = _mm256_load_ps(inter1 + x);
					minter2 = _mm256_load_ps(inter2 + x);

					mnumer0 = _mm256_load_ps(numer0 + x);
					mnumer1 = _mm256_load_ps(numer1 + x);
					mnumer2 = _mm256_load_ps(numer2 + x);

					_mm256_store_ps(numer0 + x, _mm256_mul_ps(minter0, meigmat2));
					_mm256_store_ps(numer1 + x, _mm256_mul_ps(minter1, meigmat2));
					_mm256_store_ps(numer2 + x, _mm256_mul_ps(minter2, meigmat2));

					//_mm256_store_ps(numer0 + x, _mm256_fmadd_ps(minter0, meigmat2, mnumer0));
					//_mm256_store_ps(numer1 + x, _mm256_fmadd_ps(minter1, meigmat2, mnumer1));
					//_mm256_store_ps(numer2 + x, _mm256_fmadd_ps(minter2, meigmat2, mnumer2));

					//_mm256_store_ps(numer0 + x, _mm256_fmadd_ps(mnumer0, meigmat, mnumer0));
					//_mm256_store_ps(numer1 + x, _mm256_fmadd_ps(mnumer1, meigmat, mnumer1));
					//_mm256_store_ps(numer2 + x, _mm256_fmadd_ps(mnumer2, meigmat, mnumer2));

					//_mm256_store_ps(numer0 + x, _mm256_fmadd_ps(_mm256_load_ps(inter0 + x), _mm256_load_ps(eigMat2 + x), _mm256_load_ps(numer0 + x)));
					//_mm256_store_ps(numer1 + x, _mm256_fmadd_ps(_mm256_load_ps(inter1 + x), _mm256_load_ps(eigMat2 + x), _mm256_load_ps(numer1 + x)));
					//_mm256_store_ps(numer2 + x, _mm256_fmadd_ps(_mm256_load_ps(inter2 + x), _mm256_load_ps(eigMat2 + x), _mm256_load_ps(numer2 + x)));
				}
			}
		}
	}

	// inter_numer, inter_denom の加算
	void ClusteringHDKF_Nystrom::summing()
	{
		//std::cout << "summing " << std::endl;
#pragma omp parallel for
		for (int y = 0; y < img_size.height; ++y)
		{
			float* numer_ptr0 = split_numer[0].ptr<float>(y);
			float* numer_ptr1 = split_numer[1].ptr<float>(y);
			float* numer_ptr2 = split_numer[2].ptr<float>(y);
			float* denom_ptr = denom.ptr<float>(y);

			//k=0
			{
				float* inter_n0 = inter_numer[0][0].ptr<float>(y);
				float* inter_n1 = inter_numer[1][0].ptr<float>(y);
				float* inter_n2 = inter_numer[2][0].ptr<float>(y);
				float* inter_d = inter_denom[0].ptr<float>(y);

				for (int x = 0; x < img_size.width; x += 8)
				{
					_mm256_store_ps(&numer_ptr0[x], *(__m256*) & inter_n0[x]);
					_mm256_store_ps(&numer_ptr1[x], *(__m256*) & inter_n1[x]);
					_mm256_store_ps(&numer_ptr2[x], *(__m256*) & inter_n2[x]);
					_mm256_store_ps(&denom_ptr[x], *(__m256*) & inter_d[x]);
				}
			}

			//#pragma omp parallel for
			for (int k = 1; k < K; k++)
			{
				float* inter_n0 = inter_numer[0][k].ptr<float>(y);
				float* inter_n1 = inter_numer[1][k].ptr<float>(y);
				float* inter_n2 = inter_numer[2][k].ptr<float>(y);
				float* inter_d = inter_denom[k].ptr<float>(y);
				for (int x = 0; x < img_size.width; x += 8)
				{
					_mm256_store_ps(&numer_ptr0[x], _mm256_add_ps(*(__m256*) & numer_ptr0[x], *(__m256*) & inter_n0[x]));
					_mm256_store_ps(&numer_ptr1[x], _mm256_add_ps(*(__m256*) & numer_ptr1[x], *(__m256*) & inter_n1[x]));
					_mm256_store_ps(&numer_ptr2[x], _mm256_add_ps(*(__m256*) & numer_ptr2[x], *(__m256*) & inter_n2[x]));
					_mm256_store_ps(&denom_ptr[x], _mm256_add_ps(*(__m256*) & denom_ptr[x], *(__m256*) & inter_d[x]));
				}
			}
		}
	}

	// numer / denom
	void ClusteringHDKF_Nystrom::divide()
	{
		__m256 mnumer0, mnumer1, mnumer2, mdenom;
		//	__m256 mones = _mm256_set1_ps(1.f);
		__m256 mones = _mm256_set1_ps(255.f);
		__m256 mzeros = _mm256_set1_ps(0.f);

		//imshow("split_numer" + std::to_string(0), split_numer[0]);
		//imshow("split_numer" + std::to_string(1), split_numer[1]);
		//imshow("split_numer" + std::to_string(2), split_numer[2]);
		//imshow("denom" , denom);


#pragma omp parallel for
		for (int y = 0; y < img_size.height; y++)
		{
			float* numer0 = split_numer[0].ptr<float>(y);
			float* numer1 = split_numer[1].ptr<float>(y);
			float* numer2 = split_numer[2].ptr<float>(y);
			float* denom_ptr = denom.ptr<float>(y);
			__m256 mask1_0, mask1_1, mask1_2, mask2_0, mask2_1, mask2_2;
			for (int x = 0; x < img_size.width; x += 8)
			{
				mnumer0 = _mm256_load_ps(numer0 + x);
				mnumer1 = _mm256_load_ps(numer1 + x);
				mnumer2 = _mm256_load_ps(numer2 + x);
				mdenom = _mm256_load_ps(denom_ptr + x);
				mnumer0 = _mm256_div_ps(mnumer0, mdenom);
				mnumer1 = _mm256_div_ps(mnumer1, mdenom);
				mnumer2 = _mm256_div_ps(mnumer2, mdenom);
				mask1_0 = _mm256_cmp_ps(mnumer0, mones, _CMP_GT_OQ);
				mask1_1 = _mm256_cmp_ps(mnumer1, mones, _CMP_GT_OQ);
				mask1_2 = _mm256_cmp_ps(mnumer2, mones, _CMP_GT_OQ);
				mask2_0 = _mm256_cmp_ps(mnumer0, mzeros, _CMP_LT_OQ);
				mask2_1 = _mm256_cmp_ps(mnumer1, mzeros, _CMP_LT_OQ);
				mask2_2 = _mm256_cmp_ps(mnumer2, mzeros, _CMP_LT_OQ);
				_mm256_store_ps(numer0 + x, _mm256_blendv_ps(_mm256_blendv_ps(mnumer0, mones, mask1_0), mzeros, mask2_0));
				_mm256_store_ps(numer1 + x, _mm256_blendv_ps(_mm256_blendv_ps(mnumer1, mones, mask1_1), mzeros, mask2_1));
				_mm256_store_ps(numer2 + x, _mm256_blendv_ps(_mm256_blendv_ps(mnumer2, mones, mask1_2), mzeros, mask2_2));
				//_mm256_store_ps(numer1 + x, _mm256_div_ps(mnumer1, mdenom));
				//_mm256_store_ps(numer2 + x, _mm256_div_ps(mnumer2, mdenom));
				//if (filterd_ptr[x][0] > 1.f) filterd_ptr[x][0] = 1.f;
				//if (filterd_ptr[x][1] > 1.f) filterd_ptr[x][1] = 1.f;
				//if (filterd_ptr[x][2] > 1.f) filterd_ptr[x][2] = 1.f;
				//if (filterd_ptr[x][0] < 0) filterd_ptr[x][0] = 0;
				//if (filterd_ptr[x][1] < 0) filterd_ptr[x][1] = 0;
				//if (filterd_ptr[x][2] < 0) filterd_ptr[x][2] = 0;
			}
		}

	}

	void ClusteringHDKF_Nystrom::xmeans_init(const cv::Mat& src, cv::Mat& dst)
	{
		if (vecW.size() != K || vecW[0].size() != img_size)
		{
			vecW.resize(K);
			for (int i = 0; i < K; i++)
			{
				vecW[i].create(img_size, CV_32FC1);
			}
		}

		if (A.size() != cv::Size(K, K))
		{
			A.create(cv::Size(K, K), CV_32FC1);
		}

		if (inter_numer.size() != 3 || inter_numer[0].size() != K || inter_numer[0][0].size() != img_size)
		{
			inter_numer.resize(3);
			inter_numer[0].resize(K);
			inter_numer[1].resize(K);
			inter_numer[2].resize(K);
			for (int k = 0; k < K; k++)
			{
				inter_numer[0][k].create(img_size, CV_32FC1);
				inter_numer[1][k].create(img_size, CV_32FC1);
				inter_numer[2][k].create(img_size, CV_32FC1);
			}

		}

		if (inter_denom.size() != K || inter_denom[0].size() != img_size)
		{
			inter_denom.resize(K);
			for (int k = 0; k < K; k++)
			{
				inter_denom[k].create(img_size, CV_32FC1);
			}
		}

		for (int k = 0; k < K; k++)
		{
			inter_denom[k] = cv::Mat::zeros(img_size, CV_32FC1);
		}
	}


	cv::Mat ClusteringHDKF_Nystrom::get_centers()
	{
		return this->centers;
	}

	void ClusteringHDKF_Nystrom::set_labels(const cv::Mat& labels)
	{
		this->labels = labels;
	}

	void ClusteringHDKF_Nystrom::set_centers(const cv::Mat& centers)
	{
		this->centers = centers;
	}


	void ClusteringHDKF_Nystrom::body(const cv::Mat& src, cv::Mat& dst, const cv::Mat& guide)
	{
		init(src, dst);
		clustering();
		if (cm == ClusterMethod::X_means) xmeans_init(src, dst);
		calcVecW();
		calcA();
		mul_add_gaussian();
		summing();
		divide();
		merge(split_numer, dst);
		dst.convertTo(dst, CV_32FC3);
		//	dst.convertTo(dst, CV_32FC3, 255.0);
	}

	void ClusteringHDKF_Nystrom::filtering(const cv::Mat& src, cv::Mat& dst)
	{
		init(src, dst);
		clustering();
		if (cm == ClusterMethod::X_means) xmeans_init(src, dst);
		calcVecW();
		calcA();
		mul_add_gaussian();
		summing();
		divide();
		merge(split_numer, dst);
		dst.convertTo(dst, CV_32FC3);
		//	dst.convertTo(dst, CV_32FC3, 255.0);
	}

	void ClusteringHDKF_Nystrom::filtering(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod)
	{
		isJoint = false;
		//std::cout << "filtering" << std::endl;
		if (src.channels() != 3)
		{
			std::cout << "channels is not 3" << std::endl;
			assert(src.channels() == 3);
		}

		setParameter(src.size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod);


		body(src, dst, cv::Mat());
	}

	void ClusteringHDKF_Nystrom::jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod)
	{
		isJoint = true;

		channels = src.channels();

		setParameter(src.size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod);


		body(src, dst, guide);
	}
}