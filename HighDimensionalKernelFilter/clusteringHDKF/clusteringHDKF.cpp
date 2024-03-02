#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

using namespace std;
using namespace cv;
namespace cp
{
	void randomSet(const int K, const int destChannels, cv::Mat& dest_center, float minv, float maxv)
	{
		dest_center.create(K, destChannels, CV_32F);
		cv::RNG rng(cv::getTickCount());
		for (int i = 0; i < K; i++)
		{
			for (int c = 0; c < dest_center.cols; c++)
			{
				dest_center.at<float>(i, c) = rng.uniform(minv, maxv);
			}
		}
	}

	void randomSample(std::vector<cv::Mat>& vsrc32f, const int K, cv::Mat& dest_center)
	{
		//print_debug3(src32f.cols, src32f.rows, dest_center.cols);

		cv::RNG rng(cv::getTickCount());

		dest_center.create(K, 1, CV_32FC3);
		int size = vsrc32f[0].size().area();
		for (int i = 0; i < K; i++)
		{
			const int idx = rng.uniform(0, size);
			for (int c = 0; c < dest_center.cols; c++)
			{
				dest_center.at<float>(i, c) = vsrc32f[c].at<float>(idx);
			}
		}
	}

	void randomSample(cv::Mat& src32f, const int K, cv::Mat& dest_center)
	{
		//randomSet(K, src32f.rows, dest_center, 0.f, 255.f);
		dest_center.create(K, src32f.rows, CV_32F);
		//print_debug3(src32f.cols, src32f.rows, dest_center.cols);

		cv::Mat a(1, src32f.cols, CV_32F);
		int* aptr = a.ptr<int>();
		for (int i = 0; i < src32f.cols; i++)
		{
			aptr[i] = i;
		}
		cv::RNG rng(cv::getTickCount());
		cv::randShuffle(a, 2, &rng);

		const float* s = src32f.ptr<float>();
		for (int k = 0; k < K; k++)
		{
			const int idx = a.at<int>(k);
			for (int c = 0; c < src32f.rows; c++)
			{
				dest_center.at<float>(k, c) = s[src32f.cols * c + idx];
			}
		}
	}

#pragma region ConstantTimeCBFBase

	cv::Ptr<ClusteringHDKFBase> createClusteringHDKF(ConstantTimeHDGF method)
	{
		switch (method)
		{
		case ConstantTimeHDGF::Interpolation:
			//return cv::Ptr<ConstantTimeHDGFBase>(new ConstantTimeHDGF_Interpolation); break;
			std::cout << "not implemented" << std::endl;
			break;
		case ConstantTimeHDGF::Nystrom:
			return cv::Ptr<ClusteringHDKFBase>(new ClusteringHDKF_Nystrom); break;
		case ConstantTimeHDGF::SoftAssignment:
			return cv::Ptr<ClusteringHDKFBase>(new ClusteringHDKF_SoftAssignment); break;
		default:
			std::cout << "do not support this method in createConstantTimeHDGFSingle" << std::endl;
			std::cout << "retun interpolation, instead" << std::endl;

			break;
		}
		return cv::Ptr<ClusteringHDKFBase>(new ClusteringHDKF_Nystrom);
	}

	void ClusteringHDKFBase::clustering()
	{
		//int64 start, end;
		if (downsampleMethod == DownsampleMethod::IMPORTANCE_MAP)
		{
			input_image32f.convertTo(input_image8u, CV_8U);
		}

		//start = cv::getTickCount();
		if (isDownsampleClustering)
		{
			downsampleForClustering();
		}
		else
		{
			reshaped_image32f = input_image32f.reshape(1, img_size.width * img_size.height);
		}

		//end = cv::getTickCount();
		//std::cout << "Downsample time:" << (end - start)*1000/ (cv::getTickFrequency()) << std::endl;

		if (cm == ClusterMethod::K_means_fast || cm == ClusterMethod::K_means_pp_fast)
		{
			const int vecsize = sizeof(__m256) / sizeof(float);//8
			int remsize = reshaped_image32f.rows % vecsize;
			if (remsize != 0)
			{
				cv::Rect roi(cv::Point(0, 0), cv::Size(reshaped_image32f.cols, reshaped_image32f.rows - remsize));
				reshaped_image32f = reshaped_image32f(roi);
			}
		}

		if (reshaped_image32f.depth() != CV_32F)
			reshaped_image32f.convertTo(reshaped_image32f, CV_32F);
		assert(reshaped_image32f.type() == CV_32F);
		cv::TermCriteria criteria(cv::TermCriteria::COUNT, iterations, 1);
		//cv::setNumThreads(1);

#pragma region clustering
		switch (cm)
		{
		case ClusterMethod::random_sample:
			randomSample(reshaped_image32f, K, centers);
			break;
		case ClusterMethod::K_means:
			//start = cv::getTickCount();
			// K-means Clustering
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
			//end = cv::getTickCount();
			//diff = end - start;
			//time = (diff) * 1000 / (cv::getTickFrequency());
			//cout << time << endl;
			//cout << labels.type() << endl;
			//cout << labels.size() << endl;
			break;
		case ClusterMethod::K_means_pp:
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
			//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);

			break;
		case ClusterMethod::K_means_fast:
			kmcluster.clustering(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
			break;
		case ClusterMethod::K_means_pp_fast:
			kmcluster.clustering(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
			break;
		case ClusterMethod::KGaussInvMeansPPFast:
			kmcluster.setSigma(30);
			//kmcluster.gui(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers, cp::KMeans::MeanFunction::GaussInv);
			kmcluster.clustering(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers, cp::KMeans::MeanFunction::GaussInv);
			break;

		case ClusterMethod::mediancut_median:
			mediancut(input_image8u, K, labels, centers, cp::MedianCutMethod::MEDIAN);
			centers.convertTo(centers, CV_32FC3, 1.0 / 255.0);
			break;
		case ClusterMethod::mediancut_max:
			mediancut(input_image8u, K, labels, centers, cp::MedianCutMethod::MAX);
			centers.convertTo(centers, CV_32FC3, 1.0 / 255.0);
			break;
		case ClusterMethod::mediancut_min:
			mediancut(input_image8u, K, labels, centers, cp::MedianCutMethod::MIN);
			centers.convertTo(centers, CV_32FC3, 1.0 / 255.0);
			break;

		case ClusterMethod::X_means:
			setK(cp::xmeans(reshaped_image32f, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers));
			break;

			//	quantize_wu,
			//	kmeans_wu,
			//	quantize_neural,
			//	kmeans_neural,
		case ClusterMethod::quantize_wan:
			quantization(input_image8u, K, centers, labels, 0);
			break;
		case ClusterMethod::kmeans_wan:
			quantization(input_image8u, K, centers, labels, 0);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;
		case ClusterMethod::quantize_wu:
			quantization(input_image8u, K, centers, labels, 1);
			break;
		case ClusterMethod::kmeans_wu:
			quantization(input_image8u, K, centers, labels, 1);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;
		case ClusterMethod::quantize_neural:
			quantization(input_image8u, K, centers, labels, 2);
			break;
		case ClusterMethod::kmeans_neural:
			quantization(input_image8u, K, centers, labels, 2);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;


		case ClusterMethod::quantize_DIV:
			nQunat(input_image8u, K, centers, labels, cm);
			break;
		case ClusterMethod::kmeans_DIV:
			nQunat(input_image8u, K, centers, labels, cm);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;
		case ClusterMethod::quantize_PNN:
			nQunat(input_image8u, K, centers, labels, cm);
			break;
		case ClusterMethod::kmeans_PNN:
			nQunat(input_image8u, K, centers, labels, cm);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;
		case ClusterMethod::quantize_SPA:
			nQunat(input_image8u, K, centers, labels, cm);
			break;
		case ClusterMethod::kmeans_SPA:
			nQunat(input_image8u, K, centers, labels, cm);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;
		case ClusterMethod::quantize_EAS:
			nQunat(input_image8u, K, centers, labels, cm);
			break;
		case ClusterMethod::kmeans_EAS:
			nQunat(input_image8u, K, centers, labels, cm);
			cv::kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
			break;
		default:
			break;
		}
#pragma endregion
		if (centers.rows != K)
		{
			std::cout << "Clustering is not working" << std::endl;
		}
		assert(centers.rows == K);
	}

	void ClusteringHDKFBase::downsampleForClustering()
	{
		switch (downsampleMethod)
		{
		case NEAREST:
		case LINEAR:
		case CUBIC:
		case AREA:
		case LANCZOS:
			cv::resize(input_image32f, reshaped_image32f,
				cv::Size(img_size.width / downsampleRate, img_size.height / downsampleRate),
				0, 0, downsampleMethod);
			reshaped_image32f = reshaped_image32f.reshape(1, (img_size.width / downsampleRate) * (img_size.height / downsampleRate));
			break;

		case IMPORTANCE_MAP:
			cp::generateSamplingMaskRemappedDitherTexturenessPackedAoS(input_image8u, reshaped_image32f, 1.f / (downsampleRate * downsampleRate));
			//reshaped_image32f.convertTo(reshaped_image32f, CV_32FC3, 1.0 / 255);
			reshaped_image32f = reshaped_image32f.reshape(1, reshaped_image32f.rows);
			break;

		default:
			break;
		}
	}

	ClusteringHDKFBase::ClusteringHDKFBase()
	{
		GF.resize(threadMax);
	}

	//Base class of constant time bilateral filtering for multi thread version
	ClusteringHDKFBase::~ClusteringHDKFBase()
	{
		;
	}

	void ClusteringHDKFBase::setGaussianFilterRadius(const int r)
	{
		this->radius = r;
	}

	void ClusteringHDKFBase::setGaussianFilter(const double sigma_space,
		const cp::SpatialFilterAlgorithm method, const int gf_order)
	{
		bool isCompute = false;
		if (GF[0].empty())
		{
			isCompute = true;
		}
		else
		{
			if (GF[0]->getSigma() != sigma_space ||
				GF[0]->getAlgorithmType() != method ||
				GF[0]->getOrder() != gf_order ||
				GF[0]->getSize() != img_size)
			{
				isCompute = true;
			}
		}

		if (isCompute)
		{
			//cout << "alloc GF" << endl;
			//this->sigma_space = sigma_space;


			for (int i = 0; i < threadMax; ++i)
			{
				GF[i] = cp::createSpatialFilter(method, CV_32F, cp::SpatialKernel::GAUSSIAN);
			}

			if (radius != 0)
			{
				if (radius != 0)
				{
					for (int i = 0; i < threadMax; ++i)
					{
						GF[i]->setFixRadius(radius);
					}
				}
			}
		}
	}


	void ClusteringHDKFBase::setParameter(cv::Size img_size, double sigma_space, double sigma_range, ClusterMethod cm,
		int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		bool isDownsampleClustering, int downsampleRate, int downsampleMethod)
	{
		this->depth = depth;
		this->K = K;

		this->img_size = img_size;
		this->sigma_space = sigma_space;
		this->sigma_range = sigma_range;

		this->cm = cm;
		this->setGaussianFilterRadius((int)ceil(this->sigma_space * 3));
		this->setGaussianFilter(sigma_space, gf_method, gf_order);

		this->isDownsampleClustering = isDownsampleClustering;
		this->downsampleRate = downsampleRate;
		this->downsampleMethod = downsampleMethod;
	}

	void ClusteringHDKFBase::setK(int k)
	{
		this->K = k;
	}

	int ClusteringHDKFBase::getK()
	{
		return this->K;
	}

	void ClusteringHDKFBase::setConcat_offset(int concat_offset)
	{
		this->concat_offset = concat_offset;
	}

	void ClusteringHDKFBase::setPca_r(int pca_r)
	{
		this->pca_r = pca_r;
	}

	void ClusteringHDKFBase::setKmeans_ratio(float kmeans_ratio)
	{
		this->kmeans_ratio = kmeans_ratio;
	}
#pragma endregion

#pragma region ConstantTimeHDGFSingleBase

	cv::Ptr<ClusteringHDKFSingleBase> createClusteringHDKFSingle(ConstantTimeHDGF method)
	{
		switch (method)
		{
		case ConstantTimeHDGF::Interpolation:
			return cv::Ptr<ClusteringHDKFSingleBase>(new ClusteringHDKF_InterpolationSingle); break;
			break;
		case ConstantTimeHDGF::Interpolation2:
			return cv::Ptr<ClusteringHDKFSingleBase>(new ClusteringHDKF_Interpolation2Single); break;
			break;
		case ConstantTimeHDGF::Interpolation3:
			return cv::Ptr<ClusteringHDKFSingleBase>(new ClusteringHDKF_Interpolation3Single); break;
			break;
		case ConstantTimeHDGF::Nystrom:
			return cv::Ptr<ClusteringHDKFSingleBase>(new ClusteringHDKF_NystromSingle); break;
			break;
		case ConstantTimeHDGF::SoftAssignment:
			return cv::Ptr<ClusteringHDKFSingleBase>(new ClusteringHDKF_SoftAssignmentSingle); break;
			break;
		default:
			std::cout << "do not support this method in createConstantTimeHDGFSingle" << std::endl;
			std::cout << "retun interpolation, instead" << std::endl;

			break;
		}
		return cv::Ptr<ClusteringHDKFSingleBase>(new ClusteringHDKF_InterpolationSingle);
	}

	ClusteringHDKFSingleBase::ClusteringHDKFSingleBase()
	{
		timer.resize(num_timerblock_max);
		for (int i = 0; i < num_timerblock_max; i++)timer[i].setIsShow(false);
	}

	ClusteringHDKFSingleBase::~ClusteringHDKFSingleBase()
	{
		;
	}

	void reshapeDownSample(std::vector<cv::Mat> vsrc, cv::Mat& dest, const int downsampleRate, const int boundaryLength)
	{
		//const int size = ((vsrc[0].cols - boundaryLength * 2) / downsampleRate) * ((vsrc[0].rows - boundaryLength * 2) / downsampleRate);
		const int size = ((vsrc[0].cols) / downsampleRate) * ((vsrc[0].rows) / downsampleRate);
		dest.create(cv::Size(size, (int)vsrc.size()), CV_32F);

		int index = 0;
		float* dptr0 = dest.ptr<float>(0);
		float* dptr1 = dest.ptr<float>(1);
		float* dptr2 = dest.ptr<float>(2);

#if 0//random sampling
		cv::RNG rng(cv::getTickCount());
		float* bptr = vsrc[0].ptr<float>(0);
		float* gptr = vsrc[1].ptr<float>(0);
		float* rptr = vsrc[2].ptr<float>(0);
		cv::Mat buff = cv::Mat::zeros(vsrc[0].size().area(), 1, CV_8U);
		for (int i = 0; i < size; i++)
		{
			int idx = rng.uniform(0, (int)vsrc[0].size().area());
			for (;;)
			{
				if (buff.at<uchar>(idx) == 0)
				{
					buff.at<uchar>(idx) = 255;
					break;
				}
				idx = rng.uniform(0, (int)vsrc[0].size().area());
			}

			dptr0[index] = bptr[idx];
			dptr1[index] = gptr[idx];
			dptr2[index++] = rptr[idx];
		}
#elif 1
		for (int j = 0; j < vsrc[0].rows; j += downsampleRate)
		{
			float* bptr = vsrc[0].ptr<float>(j);
			float* gptr = vsrc[1].ptr<float>(j);
			float* rptr = vsrc[2].ptr<float>(j);
			for (int i = 0; i < vsrc[0].cols; i += downsampleRate)
			{
				dptr0[index] = bptr[i];
				dptr1[index] = gptr[i];
				dptr2[index++] = rptr[i];
			}
		}
#else
		for (int j = boundaryLength; j < vsrc[0].rows - boundaryLength; j += downsampleRate)
		{
			float* bptr = vsrc[0].ptr<float>(j);
			float* gptr = vsrc[1].ptr<float>(j);
			float* rptr = vsrc[2].ptr<float>(j);
			for (int i = boundaryLength; i < vsrc[0].cols - boundaryLength; i += downsampleRate)
			{
				dptr0[index] = bptr[i];
				dptr1[index] = gptr[i];
				dptr2[index++] = rptr[i];
			}
		}
#endif

	}

	double ClusteringHDKFSingleBase::testClustering(const std::vector<cv::Mat>& src)
	{
		clusteringErrorMap.create(src[0].size(), CV_32F);
		//mu;
		const int size = src[0].size().area();
		const int ch = (int)src.size();
		double error = 0.0;
		for (int i = 0; i < size; i++)
		{
			float mindiff = FLT_MAX;
			for (int k = 0; k < K; k++)
			{
				float diff = 0.f;
				for (int c = 0; c < ch; c++)
				{
					const float v = src[c].at<float>(i) - mu.at<float>(k, c);
					diff += v * v;
				}
				mindiff = std::min(mindiff, diff);
			}
			clusteringErrorMap.at<float>(i) = mindiff / ch;
			error += mindiff;
		}
		return error / size;
	}

	void ClusteringHDKFSingleBase::getClusteringErrorMap(cv::Mat& dest)
	{
		this->clusteringErrorMap.copyTo(dest);
	}

	AutoBuffer<AutoBuffer<AutoBuffer<AutoBuffer<ushort>>>> histogram(100);
	static void refineClustering(vector<Mat>& guide, Mat& mu_inplace, const int clusterRefineMethod)
	{
		if (clusterRefineMethod == 0) return;
		Mat index1st(guide[0].size(), CV_8U);
		Mat index2nd(guide[0].size(), CV_8U);
		Mat index3rd(guide[0].size(), CV_8U);
		const int K = mu_inplace.rows;
		AutoBuffer<float*> mu(K);
		for (int k = 0; k < K; k++)
		{
			mu[k] = mu_inplace.ptr<float>(k);
		}
		const int size = guide[0].size().area();
		float* g0 = guide[0].ptr<float>();
		float* g1 = guide[1].ptr<float>();
		float* g2 = guide[2].ptr<float>();
		for (int i = 0; i < size; i++)
		{
			float distmax = FLT_MAX;
			int argk1st = 0;
			int argk2nd = 0;
			int argk3rd = 0;
			for (int k = 0; k < K; k++)
			{
				float dist = (g0[i] - mu[k][0]) * (g0[i] - mu[k][0]);
				dist += (g1[i] - mu[k][1]) * (g1[i] - mu[k][1]);
				dist += (g2[i] - mu[k][2]) * (g2[i] - mu[k][2]);
				if (dist < distmax)
				{
					distmax = dist;
					argk3rd = argk2nd;
					argk2nd = argk1st;
					argk1st = k;
				}
			}
			index1st.at<uchar>(i) = argk1st;
			index2nd.at<uchar>(i) = argk2nd;
			index3rd.at<uchar>(i) = argk3rd;
		}
		//imshowScale("a", index * 20); waitKey();
		AutoBuffer<float> munew0(K);
		AutoBuffer<float> munew1(K);
		AutoBuffer<float> munew2(K);
		AutoBuffer<int> count_(K);
		for (int k = 0; k < K; k++)
		{
			munew0[k] = 0.f;
			munew1[k] = 0.f;
			munew2[k] = 0.f;
			count_[k] = 0;
		}

		if (clusterRefineMethod == 2)
		{
			for (int k = 0; k < K; k++)
			{
				if (histogram[k].size() != 256) histogram[k].resize(256);
				for (int b = 0; b < 256; b++)
				{
					if (histogram[k][b].size() != 256) histogram[k][b].resize(256);
					for (int g = 0; g < 256; g++)
					{
						if (histogram[k][b][g].size() != 256) histogram[k][b][g].resize(256);
						for (int r = 0; r < 256; r += 16)
						{
							_mm256_store_si256((__m256i*)(histogram[k][b][g] + r), _mm256_setzero_si256());
						}
					}
				}
			}
			AutoBuffer<ushort> histmax(K);
			AutoBuffer<float> normal(K);
			for (int k = 0; k < K; k++)
			{
				histmax[k] = 0;
				normal[k] = 0.f;
			}
			for (int i = 0; i < size; i++)
			{
				for (int k = 0; k < K; k++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						int b = int(g0[i])/4;
						int g = int(g1[i])/4;
						int r = int(g2[i])/4;
						histogram[k][b][g][r]++;
						histmax[k] = max(histmax[k], histogram[k][b][g][r]);
						count_[k]++;
					}
				}
			}

			for (int i = 0; i < size; i++)
			{
				for (int k = 0; k < K; k++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						int b = int(g0[i])/2;
						int g = int(g1[i])/2;
						int r = int(g2[i])/2;
						const float w = pow(histmax[k] - histogram[k][b][g][r] + 1, 1);

						munew0[k] += w * g0[i];
						munew1[k] += w * g1[i];
						munew2[k] += w * g2[i];
						normal[k] += w;
					}
				}
			}
			for (int k = 0; k < K; k++)
			{
				mu[k][0] = munew0[k] / normal[k];
				mu[k][1] = munew1[k] / normal[k];
				mu[k][2] = munew2[k] / normal[k];
			}
		}
		if (clusterRefineMethod == 1)
		{
			AutoBuffer<AutoBuffer<ushort>> histogram(K);
			for (int k = 0; k < K; k++)
			{
				histogram[k].resize(443);
				for (int i = 0; i < 443; i++) histogram[k][i] = 0;
			}
			for (int i = 0; i < size; i++)
			{
				for (int k = 0; k < K; k++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						int idx = (int)sqrt((g0[i] - mu[k][0]) * (g0[i] - mu[k][0]) + (g1[i] - mu[k][1]) * (g1[i] - mu[k][1]) + (g2[i] - mu[k][2]) * (g2[i] - mu[k][2]));
						histogram[k][idx]++;
						count_[k]++;
					}
				}
			}

			AutoBuffer<ushort> histmax(K);
			AutoBuffer<float> normal(K);
			for (int k = 0; k < K; k++)
			{
				histmax[k] = 0;
				normal[k] = 0;
			}
			for (int k = 0; k < K; k++)
			{
				for (int i = 0; i < 443; i++) histmax[k] = max(histmax[k], histogram[k][i]);

				//	print_debug3(k, histmax[k], count_[k]);
			}
			for (int i = 0; i < size; i++)
			{
				for (int k = 0; k < K; k++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						const int idx = (int)sqrt((g0[i] - mu[k][0]) * (g0[i] - mu[k][0]) + (g1[i] - mu[k][1]) * (g1[i] - mu[k][1]) + (g2[i] - mu[k][2]) * (g2[i] - mu[k][2]));
						const float w = pow(histmax[k] - histogram[k][idx] + 1, 3);


						munew0[k] += w * g0[i];
						munew1[k] += w * g1[i];
						munew2[k] += w * g2[i];
						normal[k] += w;
						/*munew0[k] += g0[i];
						munew1[k] += g1[i];
						munew2[k] += g2[i];*/
					}
				}
			}
			for (int k = 0; k < K; k++)
			{
				mu[k][0] = munew0[k] / normal[k];
				mu[k][1] = munew1[k] / normal[k];
				mu[k][2] = munew2[k] / normal[k];
			}

			/*
			for (int i = 0; i < size; i++)
			{
				for (int k = 0; k < K; k++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						munew0[k] += g0[i];
						munew1[k] += g1[i];
						munew2[k] += g2[i];
						count_[k]++;
					}
				}
			}
			for (int k = 0; k < K; k++)
			{
				mu[k][0] = munew0[k] / count_[k];
				mu[k][1] = munew1[k] / count_[k];
				mu[k][2] = munew2[k] / count_[k];
			}
			*/
		}
		else if (clusterRefineMethod == 11)
		{
			//cout << "clusterRefineMethod == 2" << endl;
			for (int k = 0; k < K; k++)
			{
				float distnin = FLT_MAX;
				float distmax = 0.f;
				for (int i = 0; i < size; i++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						const int k2 = index2nd.at<uchar>(i);
						const int k3 = index3rd.at<uchar>(i);
						const float d0 = abs(g0[i] - mu[k2][0]) + 0.01f * abs(g0[i] - mu[k3][0]);
						const float d1 = abs(g1[i] - mu[k2][1]) + 0.01f * abs(g1[i] - mu[k3][1]);
						const float d2 = abs(g2[i] - mu[k2][2]) + 0.01f * abs(g2[i] - mu[k3][2]);
						float dc0 = abs(g0[i] - mu[k][0]);
						float dc1 = abs(g1[i] - mu[k][1]);
						float dc2 = abs(g2[i] - mu[k][2]);
						const float dc = dc0 + dc1 + dc2;
						const float dn = d0 + d1 + d2;
						const float d = 1.f / dc + dn / (255.f * 3 * 0.9f);
						if (distmax < d)
						{
							distmax = d;
							munew0[k] = g0[i];
							munew1[k] = g1[i];
							munew2[k] = g2[i];
						}
					}
				}
			}

			for (int k = 0; k < K; k++)
			{
				mu[k][0] = munew0[k];
				mu[k][1] = munew1[k];
				mu[k][2] = munew2[k];
			}
		}
		else if (clusterRefineMethod == 3)
		{
			//cout << "clusterRefineMethod == 3" << endl;
			for (int k = 0; k < K; k++)
			{
				float distmin = FLT_MAX;
				for (int i = 0; i < size; i++)
				{
					if (index1st.at<uchar>(i) == k)
					{
						const int k2 = index2nd.at<uchar>(i);
						const int k3 = index3rd.at<uchar>(i);
						/*float d0 = abs(g0[i] - mu[k2][0]) + abs(g0[i] - mu[k3][0]);
						float d1 = abs(g1[i] - mu[k2][1]) + abs(g1[i] - mu[k3][1]);
						float d2 = abs(g2[i] - mu[k2][2]) + abs(g2[i] - mu[k3][2]);*/
						float d0 = abs(g0[i] - mu[k][0]);
						float d1 = abs(g1[i] - mu[k][1]);
						float d2 = abs(g2[i] - mu[k][2]);
						const float d = d0 + d1 + d2;
						if (d < distmin)
						{
							distmin = d;
							munew0[k] = g0[i];
							munew1[k] = g1[i];
							munew2[k] = g2[i];
						}
					}
				}
			}

			for (int k = 0; k < K; k++)
			{
				mu[k][0] = munew0[k];
				mu[k][1] = munew1[k];
				mu[k][2] = munew2[k];
			}
		}
	}

	void ClusteringHDKFSingleBase::clustering()
	{
		switch (cm)
		{
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
		{
			if (isDownsampleClustering)
			{
				if (isJoint) downsampleForClusteringWith8U(vguide, reshaped_image32f, guide_image8u, isCropBoundaryClustering);
				else downsampleForClusteringWith8U(vsrc, reshaped_image32f, guide_image8u, isCropBoundaryClustering);
			}
			else
			{
				if (isJoint) mergeForClustering(vguide, reshaped_image32f, isCropBoundaryClustering);
				else mergeForClustering(vsrc, reshaped_image32f, isCropBoundaryClustering);
			}
			break;
		}

		default:
		{
			if (isDownsampleClustering)
			{
				if (isJoint) downsampleForClustering(vguide, reshaped_image32f, isCropBoundaryClustering);
				else downsampleForClustering(vsrc, reshaped_image32f, isCropBoundaryClustering);
			}
			else
			{
				if (isJoint) mergeForClustering(vguide, reshaped_image32f, isCropBoundaryClustering);
				else mergeForClustering(vsrc, reshaped_image32f, isCropBoundaryClustering);
			}
			break;
		}
		}

		if (reshaped_image32f.depth() != CV_32F || reshaped_image32f.channels() != 1)
		{
#pragma omp critical
			{
				std::cout << "depth(vsrc) " << vsrc[0].depth() << std::endl;
				std::cout << "depth " << reshaped_image32f.depth() << std::endl;
				std::cout << "channel " << reshaped_image32f.channels() << std::endl;
			}
			CV_Assert(reshaped_image32f.type() == CV_32FC1);
		}

		if (cm == ClusterMethod::K_means_fast || cm == ClusterMethod::K_means_pp_fast || cm == ClusterMethod::KGaussInvMeansPPFast)
		{
			const int vecsize = sizeof(__m256) / sizeof(float);//8
			if (reshaped_image32f.cols < reshaped_image32f.rows)
			{
				int remsize = reshaped_image32f.rows % vecsize;
				if (remsize != 0)
				{
					cv::Rect roi(cv::Point(0, 0), cv::Size(reshaped_image32f.cols, reshaped_image32f.rows - remsize));
					reshaped_image32f = reshaped_image32f(roi);
				}
			}
			else
			{
				int remsize = reshaped_image32f.cols % vecsize;
				//print_debug(remsize);
				if (remsize != 0)
				{
					//std::cout << "KMEANSFAST" << std::endl;
					cv::Rect roi(cv::Point(0, 0), cv::Size(reshaped_image32f.cols - remsize, reshaped_image32f.rows));
					reshaped_image32f = reshaped_image32f(roi);
				}
			}
		}

		//print_matinfo(reshaped_image32f);
		if (reshaped_image32f.cols < K)
		{
			std::cout << "K is large for K-means: K, reshaped_image32f.cols (" << reshaped_image32f.cols << ", " << K << ")" << std::endl;
		}

		cv::TermCriteria criteria(cv::TermCriteria::COUNT, iterations, 1);
		cv::setNumThreads(1);

		switch (cm)
		{
		case ClusterMethod::random_sample:
			//if (isJoint) randomSample(vguide, K, mu);
			//else randomSample(vsrc, K, mu);
			randomSample(reshaped_image32f, K, mu);
			//randomSample(K, mu);
			break;

		case ClusterMethod::K_means:
		{
			//start = cv::getTickCount();
			// K-means Clustering
			//kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, mu);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, mu);
			//print_matinfo(reshaped_image32f);
			//print_debug(K);
			//print_matinfo(labels);
			//cv::Mat mu1 = mu;
			//print_matinfo(mu1);
			//end = cv::getTickCount();
			//diff = end - start;
			//time = (diff) * 1000 / (cv::getTickFrequency());
			//cout << time << endl;
			//cout << labels.type() << endl;
			//cout << labels.size() << endl;
			break;
		}
		case ClusterMethod::K_means_pp:
		{
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, mu);
			break;
		}
		case ClusterMethod::K_means_fast:
		{
			kmcluster.clustering(reshaped_image32f, K, labels, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, mu, cp::KMeans::MeanFunction::Mean, cp::KMeans::Schedule::SoA_KND);
			break;
		}
		case ClusterMethod::K_means_pp_fast:
		{
			kmcluster.clustering(reshaped_image32f, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, mu, cp::KMeans::MeanFunction::Mean, cp::KMeans::Schedule::SoA_KND);
			break;
		}

		case ClusterMethod::KGaussInvMeansPPFast:
		{
			kmcluster.setSigma((float)kmeans_sigma);
			kmcluster.clustering(reshaped_image32f, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, mu, cp::KMeans::MeanFunction::GaussInv, cp::KMeans::Schedule::SoA_KND);
			break;
		}
		case ClusterMethod::mediancut_median:
		case ClusterMethod::mediancut_max:
		case ClusterMethod::mediancut_min:
		{
			reshaped_image32f.convertTo(reshaped_image8u, CV_8U);
			//print_matinfo(reshaped_image8u);
			if (cm == ClusterMethod::mediancut_median) mediancut(reshaped_image8u, K, labels, mu, cp::MedianCutMethod::MEDIAN);
			if (cm == ClusterMethod::mediancut_max) mediancut(reshaped_image8u, K, labels, mu, cp::MedianCutMethod::MAX);
			if (cm == ClusterMethod::mediancut_min) mediancut(reshaped_image8u, K, labels, mu, cp::MedianCutMethod::MIN);
			//cout << mu << endl;
			mu.convertTo(mu, CV_32F);
			break;
		}
		case ClusterMethod::quantize_wan:
			quantization(guide_image8u, K, mu, labels, 0);
			break;
		case ClusterMethod::kmeans_wan:
			quantization(guide_image8u, K, mu, labels, 0);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;
		case ClusterMethod::quantize_wu:
			quantization(guide_image8u, K, mu, labels, 1);
			break;
		case ClusterMethod::kmeans_wu:
			quantization(guide_image8u, K, mu, labels, 1);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;
		case ClusterMethod::quantize_neural:
			quantization(guide_image8u, K, mu, labels, 2);
			break;
		case ClusterMethod::kmeans_neural:
			quantization(guide_image8u, K, mu, labels, 2);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;

		case ClusterMethod::quantize_DIV:
			nQunat(guide_image8u, K, mu, labels, cm);
			break;
		case ClusterMethod::kmeans_DIV:
			nQunat(guide_image8u, K, mu, labels, cm);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;
		case ClusterMethod::quantize_PNN:
			nQunat(guide_image8u, K, mu, labels, cm);
			break;
		case ClusterMethod::kmeans_PNN:
			nQunat(guide_image8u, K, mu, labels, cm);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;
		case ClusterMethod::quantize_SPA:
			nQunat(guide_image8u, K, mu, labels, cm);
			break;
		case ClusterMethod::kmeans_SPA:
			nQunat(guide_image8u, K, mu, labels, cm);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;
		case ClusterMethod::quantize_EAS:
			nQunat(guide_image8u, K, mu, labels, cm);
			break;
		case ClusterMethod::kmeans_EAS:
			nQunat(guide_image8u, K, mu, labels, cm);
			cv::kmeans(reshaped_image32f.t(), K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
			break;

		default:
			break;
		}

		//print_matinfo(mu);
		if (mu.rows != K)
		{
			std::cout << "Clustering is not working (mu.rows != K)" << std::endl;
		}
		CV_Assert(mu.rows == K);

		if (isTestClustering)
		{
			Mat v; merge(vguide, v);
			Mat labels2;
			Mat mu2;
			kmcluster.setSigma((float)kmeans_sigma);
			kmcluster.gui(reshaped_image32f, K, labels2, criteria, attempts, cv::KMEANS_PP_CENTERS, mu2, cp::KMeans::MeanFunction::GaussInv, cp::KMeans::Schedule::SoA_KND, v);
		}
		refineClustering(vguide, mu, clusterRefineMethod);
		//testClustering(vguide);
	}

	void ClusteringHDKFSingleBase::downsampleForClustering(cv::Mat& src, cv::Mat& dest)
	{
		switch (downsampleClusteringMethod)
		{
		case NEAREST:
		case LINEAR:
		case CUBIC:
		case AREA:
		case LANCZOS:
			//std::cout << "DOWNSAMPLE" << std::endl;
			cv::resize(src, dest,
				cv::Size(img_size.width / downsampleRate, img_size.height / downsampleRate),
				0, 0, downsampleClusteringMethod);
			//reshaped_image32f.convertTo(input_image8u, CV_8UC3);
			//cv::imshow("test", input_image8u);
			//cv::waitKey();
			dest = dest.reshape(1, (img_size.width / downsampleRate) * (img_size.height / downsampleRate));
			break;

		case IMPORTANCE_MAP:
			//std::cout << "IMPORTANCE MAP" << std::endl;
			cp::generateSamplingMaskRemappedDitherTexturenessPackedAoS(src, dest, 1.f / (downsampleRate * downsampleRate));
			dest = dest.reshape(1, reshaped_image32f.rows);
			break;

		default:
			break;
		}
	}

	void ClusteringHDKFSingleBase::downsampleImage(const std::vector<cv::Mat>& vsrc, std::vector<cv::Mat>& vsrcRes, const std::vector<cv::Mat>& vguide, std::vector<cv::Mat>& vguideRes, const int downsampleImageMethod)
	{
		const double res = 1.0 / downSampleImage;
		if (downSampleImage != 1)
		{
			for (int c = 0; c < channels; c++)
			{
				resize(vsrc[c], vsrcRes[c], cv::Size(), res, res, downsampleImageMethod);
			}

			if (isJoint)
			{
				for (int c = 0; c < guide_channels; c++)
				{
					resize(vguide[c], vguideRes[c], cv::Size(), res, res, downsampleImageMethod);
				}
			}
		}
	}

	void sampling_imgproc(Mat& src_, Mat& dest)
	{
		Mat src = src_.clone();

		double ss1 = 3.0;
		Mat temp;
		GaussianBlur(src, temp, Size((int)ceil(ss1 * 3) * 2 + 1, (int)ceil(ss1 * 3) * 2 + 1), ss1);
		absdiff(temp, src, dest);

		Size ksize = Size(5, 5);
		//cp::minFilter(dest, dest, 1);
		GaussianBlur(dest, dest, ksize, 1);

		/*Mat temp;
		cv::pyrDown(src, temp);
		cv::pyrUp(temp, dest);
		absdiff(dest, src, dest);
		Size ksize = Size(5, 5);
		GaussianBlur(dest, dest, ksize, 2);
		normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);*/
	}

	void generateSamplingMaskRemappedDitherTest(vector<cv::Mat>& guide, cv::Mat& dest, const float sampling_ratio, const bool isUseAverage = false, int ditheringMethod = 0)
	{
		CV_Assert(guide[0].depth() == CV_32F);

		const int channels = (int)guide.size();

		int sample_num = 0;
		cv::Mat mask(guide[0].size(), CV_8U);

		Mat v = guide[0].clone();
		/*for (int i = 1; i < guide.size(); i++)
		{
			add(v, guide[i], v);
		}*/
		v.convertTo(v, CV_32F, 1.0 / (1 * 255));
		sampling_imgproc(v, v);

		sample_num = cp::generateSamplingMaskRemappedDitherWeight(v, mask, sampling_ratio, ditheringMethod, cp::DITHER_SCANORDER::MEANDERING, 0.1, cp::DITHER_POSTPROCESS::NO_POSTPROCESS);
		sample_num = get_simd_floor(sample_num, 8);
		//print_debug(sample_num);
		dest.create(Size(sample_num, channels), CV_32F);

		AutoBuffer<float*> s(channels);
		AutoBuffer<float*> d(channels);
		for (int c = 0; c < channels; c++)
		{
			d[c] = dest.ptr<float>(c);
		}

		for (int y = 0, count = 0; y < mask.rows; y++)
		{
			uchar* mask_ptr = mask.ptr<uchar>(y);
			for (int c = 0; c < channels; c++)
			{
				s[c] = guide[c].ptr<float>(y);
			}

			for (int x = 0; x < mask.cols; x++)
			{
				if (mask_ptr[x] == 255)
				{
					for (int c = 0; c < channels; c++)
					{
						d[c][count] = s[c][x];
					}
					count++;
					if (count == sample_num)return;
				}
			}
		}
	}

	void ClusteringHDKFSingleBase::downsampleForClustering(std::vector<cv::Mat>& src, cv::Mat& dest, const bool isCropBoundary)
	{
		const int channels = (int)src.size();
		//std::vector<cv::Mat> cropBuffer;
		cropBufferForClustering.resize(channels);
		if (isCropBoundary)
		{
			//std::cout << "Crop" << std::endl;
			const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
			for (int c = 0; c < channels; c++)
			{
				cropBufferForClustering[c] = src[c](roi).clone();
			}
		}
		else
		{
			for (int c = 0; c < channels; c++)
			{
				cropBufferForClustering[c] = src[c];
			}
		}

		switch (downsampleClusteringMethod)
		{
		case NEAREST:
		case LINEAR:
		case CUBIC:
		case AREA:
		case LANCZOS:
		{
			//std::cout << "DOWNSAMPLE" << std::endl;
			const cv::Size size = cropBufferForClustering[0].size() / downsampleRate;
			dest.create(cv::Size(size.area(), channels), CV_32F);
			for (int c = 0; c < channels; c++)
			{
				cv::Mat d(size, CV_32F, dest.ptr<float>(c));
				cv::resize(cropBufferForClustering[c], d, size, 0, 0, downsampleClusteringMethod);
			}
			break;
		}
		case IMPORTANCE_MAP:
			//std::cout << "IMPORTANCE MAP" << std::endl;
			//cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false, IMAGE_TEXTURNESS_FLOYD_STEINBERG);
			cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false, IMAGE_TEXTURENESS_OSTRO);
			break;
		case IMPORTANCE_MAP2:
		{
			cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false, -1);
			//generateSamplingMaskRemappedDitherTest(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false);
			/*
			const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
			//std::cout << "IMPORTANCE MAP" << std::endl;
			cropBufferForClustering2.resize(this->vsrc.size());
			for (int c = 0; c < this->vsrc.size(); c++)
			{
				cropBufferForClustering2[c] = this->vsrc[c](roi).clone();
			}
			cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering2, cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate));
			*/
		}
		break;

		default:
			break;
		}
	}

	void ClusteringHDKFSingleBase::downsampleForClusteringWith8U(std::vector<cv::Mat>& src, cv::Mat& dest, cv::Mat& image8u, const bool isCropBoundary)
	{
		const int channels = (int)src.size();
		//std::vector<cv::Mat> cropBuffer;
		cropBufferForClustering.resize(channels);
		if (isCropBoundary)
		{
			//std::cout << "Crop" << std::endl;
			const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
			for (int c = 0; c < channels; c++)
			{
				cropBufferForClustering[c] = src[c](roi).clone();
			}
		}
		else
		{
			for (int c = 0; c < channels; c++)
			{
				cropBufferForClustering[c] = src[c];
			}
		}

		switch (downsampleClusteringMethod)
		{
		case NEAREST:
		case LINEAR:
		case CUBIC:
		case AREA:
		case LANCZOS:
		{
			//std::cout << "DOWNSAMPLE" << std::endl;
			const cv::Size size = cropBufferForClustering[0].size() / downsampleRate;
			dest.create(cv::Size(size.area(), channels), CV_32F);
			for (int c = 0; c < channels; c++)
			{
				cv::Mat d(size, CV_32F, dest.ptr<float>(c));
				cv::resize(cropBufferForClustering[c], d, size, 0, 0, downsampleClusteringMethod);
			}

			vector<Mat> im8u(channels);
			for (int c = 0; c < channels; c++)
			{
				cv::resize(cropBufferForClustering[c], im8u[c], size, 0, 0, downsampleClusteringMethod);
				im8u[c].convertTo(im8u[c], CV_8U);
			}
			merge(im8u, image8u);
			break;
		}
		case IMPORTANCE_MAP:
			//std::cout << "IMPORTANCE MAP" << std::endl;
			cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false);
			break;
		case IMPORTANCE_MAP2:
		{
			generateSamplingMaskRemappedDitherTest(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false);
			/*
			const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
			//std::cout << "IMPORTANCE MAP" << std::endl;
			cropBufferForClustering2.resize(this->vsrc.size());
			for (int c = 0; c < this->vsrc.size(); c++)
			{
				cropBufferForClustering2[c] = this->vsrc[c](roi).clone();
			}
			cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering2, cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate));
			*/
		}
		break;

		default:
			break;
		}
	}

	void ClusteringHDKFSingleBase::mergeForClustering(std::vector<cv::Mat>& src, cv::Mat& dest, const bool isCropBoundary)
	{
		const int channels = (int)src.size();
		//std::vector<cv::Mat> cropBuffer;
		cropBufferForClustering.resize(channels);
		if (isCropBoundary)
		{
			//std::cout << "Crop" << std::endl;
			const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
			for (int c = 0; c < channels; c++)
			{
				cropBufferForClustering[c] = src[c](roi).clone();
			}
		}
		else
		{
			for (int c = 0; c < channels; c++)
			{
				cropBufferForClustering[c] = src[c];
			}
		}

		dest.create(cv::Size(src[0].size().area(), channels), CV_32F);
		for (int i = 0; i < channels; i++)
		{
			src[i].copyTo(dest.row(i));
		}
	}


	void ClusteringHDKFSingleBase::filter(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int border)
	{
		isJoint = false;
		statePCA = 0;

		if (src.channels() != 3)
		{
			std::cout << "channels is not 3" << std::endl;
			assert(src.channels() == 3);
		}

		setParameter(src.size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod);

		if (src.depth() == CV_32F)guide_image32f = src;
		else src.convertTo(guide_image32f, CV_32FC3);
		cv::split(guide_image32f, vsrc);

		body(vsrc, dst, std::vector<cv::Mat>());
	}

	void ClusteringHDKFSingleBase::filter(const std::vector<cv::Mat>& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int boundaryLength, int border)
	{
		isJoint = false;
		statePCA = 0;

		setParameter(src[0].size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

		guide_channels = channels = (int)src.size();
		vsrc.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			vsrc[c] = src[c];
		}

		body(vsrc, dst, std::vector<cv::Mat>());
	}

	void ClusteringHDKFSingleBase::PCAfilter(const std::vector<cv::Mat>& src, const int pca_channels, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int boundaryLength, int border)
	{
		isJoint = true;
		statePCA = 1;

		setParameter(src[0].size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

		channels = (int)src.size();
		vsrc.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			vsrc[c] = src[c];
		}

		guide_channels = pca_channels;
		cp::cvtColorPCA(vsrc, vguide, pca_channels, projectionMatrix);

		body(vsrc, dst, vguide);
	}

	void ClusteringHDKFSingleBase::jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, int border)
	{
		isJoint = true;
		statePCA = 0;

		if (src.channels() != 3)
		{
			std::cout << "channels is not 3" << std::endl;
			assert(src.channels() == 3);
		}

		setParameter(src.size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod, border);
		//src.convertTo(input_image32f, CV_32FC3, 1.0 / 255.0);
		src.convertTo(guide_image32f, CV_32F);
		cv::split(guide_image32f, vsrc);

		guide.convertTo(guide_image32f, CV_32F);
		cv::split(guide_image32f, vguide);
		body(vsrc, dst, vguide);
	}

	void ClusteringHDKFSingleBase::jointfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, const int boundaryLength, int border)
	{
		isJoint = true;
		statePCA = 0;

		setParameter(src[0].size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

		//print_debug(src[0].size());
		//print_debug(GF->getRadius());

		channels = (int)src.size();
		vsrc.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			vsrc[c] = src[c];
		}

		guide_channels = (int)guide.size();
		vguide.resize(guide_channels);
		for (int i = 0; i < guide_channels; i++)
		{
			vguide[i] = guide[i];
		}

		body(vsrc, dst, vguide);
	}

	void ClusteringHDKFSingleBase::jointPCAfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, const int pca_channels, cv::Mat& dst, const double sigma_space, const double sigma_range, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, const int boundaryLength, int border)
	{
		isJoint = true;
		statePCA = 2;

		setParameter(src[0].size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

		channels = (int)src.size();
		vsrc.resize(channels);
		for (int c = 0; c < channels; c++)
		{
			vsrc[c] = src[c];
		}

		guide_channels = pca_channels;
		vguide.resize(guide_channels);
		//std::cout << "jointPCA" << std::endl;
		cp::cvtColorPCA(guide, vguide, pca_channels, projectionMatrix, eigenValue);

		body(vsrc, dst, vguide);
	}

	void ClusteringHDKFSingleBase::nlmfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const int parch_r, const int reduced_dim, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, const int boundaryLength, int border)
	{
		isJoint = true;

		setParameter(src[0].size(), sigma_space, sigma_range, cm,
			K, gf_method, gf_order, depth,
			isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

		channels = (int)src.size();
		vsrc.resize(channels);
		for (int c = 0; c < src.size(); c++)
		{
			vsrc[c] = src[c];
		}

		//DRIM2COL(guide, vguide, parch_r, reduced_dim, border, patchPCAMethod, false, projectionMatrix, eigenValue,);
		DRIM2COL(guide, vguide, parch_r, reduced_dim, border, patchPCAMethod, false);
		guide_channels = (int)vguide.size();
		/*for (int c = 0; c < guide_channels; c++)
		{
			double minv, maxv;
			minMaxLoc(vguide[c], &minv, &maxv);
			//print_debug(patchPCAMethod);
			if (maxv > 20000.0|| minv <-20000.0)
			{
				std::cout << c << " ";
				cp::printMinMax(vguide[c]);
				cv::Mat temp;
				cv::merge(guide, temp);
				cp::imshowScale("a", temp);
				cv::waitKey();
			}
		}*/
		body(vsrc, dst, vguide);
	}

#pragma region setter
	void ClusteringHDKFSingleBase::setGaussianFilterRadius(const int r)
	{
		this->radius = r;
	}

	void ClusteringHDKFSingleBase::setGaussianFilter(const double sigma_space, const cp::SpatialFilterAlgorithm method, const int gf_order)
	{
		bool isCompute = false;
		if (GF.empty())
		{
			isCompute = true;
		}
		else
		{
			if (GF->getSigma() != sigma_space ||
				GF->getAlgorithmType() != method ||
				GF->getOrder() != gf_order ||
				GF->getSize() != img_size / downSampleImage)
			{
				isCompute = true;
			}
		}

		if (isCompute)
		{
			//std::cout << "createcreateSpatialFilter" << std::endl;
			GF = cp::createSpatialFilter(method, CV_32F, cp::SpatialKernel::GAUSSIAN);
			const int boundaryLength = 0;//should be fixed
			GF->setIsInner(boundaryLength, boundaryLength, boundaryLength, boundaryLength);
			//if (radius != 0)
			{
				//GF->setFixRadius(radius);

			}
		}
	}


	void ClusteringHDKFSingleBase::setBoundaryLength(const int length)
	{
		boundaryLength = length;
	}

	void ClusteringHDKFSingleBase::setParameter(cv::Size img_size, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int boundarylength, int borderType)
	{
		this->num_sample_max = cv::Size((img_size.width - 2 * boundarylength) / downsampleRate, (img_size.height - 2 * boundarylength) / downsampleRate).area();
		this->depth = depth;

		this->K = std::min(K, num_sample_max);
		if (this->K == num_sample_max) 	std::cout << "full sample (debug message)" << K << "/" << num_sample_max << std::endl;
		//std::cout << (double)K / num_sample_max<<": K, numsample_max ("<<K<<", "<<num_sample_max<<")" << std::endl;
		//print_debug(img_size);
		this->img_size = img_size;
		this->sigma_space = sigma_space;
		this->spatial_order = gf_order;
		this->sigma_range = sigma_range;

		this->cm = cm;
		this->setBoundaryLength(boundarylength);
		this->borderType = borderType;
		//this->setGaussianFilterRadius((int)ceil(this->sigma_space * 3));
		this->setGaussianFilter(sigma_space / downSampleImage, method, gf_order);

		this->isDownsampleClustering = isDownsampleClustering;
		this->downsampleRate = downsampleRate;
		this->downsampleClusteringMethod = downsampleMethod;
	}

	void ClusteringHDKFSingleBase::setConcat_offset(int concat_offset)
	{
		this->concat_offset = concat_offset;
	}

	void ClusteringHDKFSingleBase::setPca_r(int pca_r)
	{
		this->pca_r = pca_r;
	}

	void ClusteringHDKFSingleBase::setKmeans_ratio(float kmeans_ratio)
	{
		this->kmeans_ratio = kmeans_ratio;
	}

	void ClusteringHDKFSingleBase::setCropClustering(bool isCropClustering)
	{
		this->isCropBoundaryClustering = isCropClustering;
	}

	void ClusteringHDKFSingleBase::setPatchPCAMethod(int method)
	{
		this->patchPCAMethod = method;
	}

	void ClusteringHDKFSingleBase::setTestClustering(bool flag)
	{
		this->isTestClustering = flag;
	}

	cv::Mat ClusteringHDKFSingleBase::getSamplingPoints()
	{
		return this->mu;
	}

	cv::Mat ClusteringHDKFSingleBase::cloneEigenValue()
	{
		return this->eigenValue.clone();
	}

	void ClusteringHDKFSingleBase::printRapTime()
	{
		for (int i = 0; i < num_timerblock_max; i++)
		{
			timer[i].getLapTimeMedian(true, cv::format("%d", i));
		}
	}
#pragma endregion
#pragma endregion
}