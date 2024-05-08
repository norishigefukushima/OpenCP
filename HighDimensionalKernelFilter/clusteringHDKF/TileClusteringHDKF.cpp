#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

#define DEBUG_OMP_DISABLE 
using namespace cv;

namespace cp
{
	TileClusteringHDKF::TileClusteringHDKF(cv::Size div_, ConstantTimeHDGF method) :
		thread_max(omp_get_max_threads()), div(div_)
	{
		this->method = method;
		if (div.area() == 1)
		{
			scbf.resize(1);
			scbf[0] = createClusteringHDKFSingle(method);
		}
		else
		{
			scbf.resize(thread_max);
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i] = createClusteringHDKFSingle(method);
			}
		}

		eigenVectors.resize(div.area());
		if (isDebug)
		{
			mu.resize(div.area());
			ci.setWindowName("tile debug");
			ci.setImageSize(Size(1024, 600));
		}
	}

	TileClusteringHDKF::~TileClusteringHDKF()
	{
		;
	}

#pragma region setter
	void TileClusteringHDKF::setLambdaInterpolation(const float lambda_)
	{
		if (div.area() == 1)
		{
			if (method == ConstantTimeHDGF::SoftAssignment)
				scbf[0].dynamicCast<ClusteringHDKF_SoftAssignmentSingle>()->setLambdaInterpolation(lambda_);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				if (method == ConstantTimeHDGF::SoftAssignment)
					scbf[i].dynamicCast<ClusteringHDKF_SoftAssignmentSingle>()->setLambdaInterpolation(lambda_);
			}
		}
	}

	void TileClusteringHDKF::setDeltaLocalStatisticsPrior(const float delta)
	{
		if (this->method == ConstantTimeHDGF::Interpolation)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setDeltaLocalStatisticsPrior(delta);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setDeltaLocalStatisticsPrior(delta);
				}
			}
		}
		if (this->method == ConstantTimeHDGF::Interpolation2)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setDeltaLocalStatisticsPrior(delta);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_Interpolation2Single>()->setDeltaLocalStatisticsPrior(delta);
				}
			}
		}
		if (this->method == ConstantTimeHDGF::Interpolation3)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_Interpolation3Single>()->setDeltaLocalStatisticsPrior(delta);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_Interpolation3Single>()->setDeltaLocalStatisticsPrior(delta);
				}
			}
		}
	}

	void TileClusteringHDKF::setIsUseLocalMu(const bool flag)
	{
		if (this->method == ConstantTimeHDGF::Interpolation)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setIsUseLocalMu(flag);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setIsUseLocalMu(flag);
				}
			}
		}
		if (this->method == ConstantTimeHDGF::Interpolation2)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_Interpolation2Single>()->setIsUseLocalMu(flag);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_Interpolation2Single>()->setIsUseLocalMu(flag);
				}
			}
		}
		if (this->method == ConstantTimeHDGF::Interpolation3)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_Interpolation3Single>()->setIsUseLocalMu(flag);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_Interpolation3Single>()->setIsUseLocalMu(flag);
				}
			}
		}
	}

	void TileClusteringHDKF::setIsUseLocalStatisticsPrior(const bool flag)
	{
		if (this->method == ConstantTimeHDGF::Interpolation)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setIsUseLocalStatisticsPrior(flag);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_InterpolationSingle>()->setIsUseLocalStatisticsPrior(flag);
				}
			}
		}
		if (this->method == ConstantTimeHDGF::Interpolation2)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_Interpolation2Single>()->setIsUseLocalStatisticsPrior(flag);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_Interpolation2Single>()->setIsUseLocalStatisticsPrior(flag);
				}
			}
		}
		if (this->method == ConstantTimeHDGF::Interpolation3)
		{
			if (div.area() == 1)
			{
				scbf[0].dynamicCast<ClusteringHDKF_Interpolation3Single>()->setIsUseLocalStatisticsPrior(flag);
			}
			else
			{
				for (int i = 0; i < thread_max; i++)
				{
					scbf[i].dynamicCast<ClusteringHDKF_Interpolation3Single>()->setIsUseLocalStatisticsPrior(flag);
				}
			}
		}
	}

	void TileClusteringHDKF::setSampleRate(const float rate)
	{
		if (div.area() == 1)
		{
			scbf[0]->setSampleRate(rate);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setSampleRate(rate);
			}
		}
	}
	void TileClusteringHDKF::setBoundaryLength(const int length)
	{
		if (div.area() == 1)
		{
			scbf[0]->setBoundaryLength(length);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setBoundaryLength(length);
			}
		}
	}

	void TileClusteringHDKF::setDownsampleImageSize(const int val)
	{
		if (div.area() == 1)
		{
			scbf[0]->setDownsampleImageSize(val);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setDownsampleImageSize(val);
			}
		}
	}

	void TileClusteringHDKF::setKMeansAttempts(const int attempts)
	{
		if (div.area() == 1)
		{
			scbf[0]->setKMeansAttempts(attempts);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setKMeansAttempts(attempts);
			}
		}
	}

	void TileClusteringHDKF::setNumIterations(const int iterations)
	{
		if (div.area() == 1)
		{
			scbf[0]->setNumIterations(iterations);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setNumIterations(iterations);
			}
		}
	}

	void TileClusteringHDKF::setKMeansPPTrials(const int num)
	{
		if (div.area() == 1)
		{
			scbf[0]->setKMeansPPTrials(num);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setKMeansPPTrials(num);
			}
		}
	}

	void TileClusteringHDKF::setKMeansREPPTrials(const int num)
	{
		if (div.area() == 1)
		{
			scbf[0]->setKMeansREPPTrials(num);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setKMeansREPPTrials(num);
			}
		}
	}

	void TileClusteringHDKF::setKMeansSigma(const double sigma)
	{
		if (div.area() == 1)
		{
			scbf[0]->setKMeansSigma(sigma);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				//std::cout << "set sigma" << sigma << std::endl;
				scbf[i]->setKMeansSigma(sigma);
			}
		}
	}

	void TileClusteringHDKF::setKMeansSignalMax(const double signal_max)
	{
		if (div.area() == 1)
		{
			scbf[0]->setKMeansSignalMax(signal_max);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setKMeansSignalMax(signal_max);
			}
		}
	}

	void TileClusteringHDKF::setClusterRefine(const int method)
	{
		if (div.area() == 1)
		{
			scbf[0]->setClusterRefine(method);
		}
		else
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setClusterRefine(method);
			}
		}
	}

	void TileClusteringHDKF::setConcat_offset(int concat_offset)
	{
		if (div.area() == 1)
		{
			scbf[0]->setConcat_offset(concat_offset);
		}
		else
		{
			const int nthreads = omp_get_max_threads();
			for (int i = 0; i < nthreads; i++)
			{
				scbf[i]->setConcat_offset(concat_offset);
			}
		}
	}

	void TileClusteringHDKF::setPca_r(int pca_r)
	{
		if (div.area() == 1)
		{
			scbf[0]->setPca_r(pca_r);
		}
		else
		{
			const int nthreads = omp_get_max_threads();
			for (int i = 0; i < nthreads; i++)
			{
				scbf[i]->setPca_r(pca_r);
			}
		}
	}

	void TileClusteringHDKF::setKmeans_ratio(float kmeans_ratio)
	{
		if (div.area() == 1)
		{
			scbf[0]->setKmeans_ratio(kmeans_ratio);
		}
		else
		{
			const int nthreads = omp_get_max_threads();
			for (int i = 0; i < nthreads; i++)
			{
				scbf[i]->setKmeans_ratio(kmeans_ratio);
			}
		}
	}

	void TileClusteringHDKF::setCropClustering(bool isCropClustering)
	{
		if (div.area() > 1)
		{
			for (int i = 0; i < thread_max; i++)
			{
				scbf[i]->setCropClustering(isCropClustering);
			}
		}
	}

	void TileClusteringHDKF::setPatchPCAMethod(int method)
	{
		if (div.area() == 1)
		{
			scbf[0]->setPatchPCAMethod(method);
		}
		else
		{
			const int nthreads = omp_get_max_threads();
			for (int i = 0; i < nthreads; i++)
			{
				scbf[i]->setPatchPCAMethod(method);
			}
		}
	}

	void TileClusteringHDKF::setGUITestIndex(int index)
	{
		this->guiTestIndex = index;
		if (index >= div.area())index = -1;
	}
#pragma endregion

#pragma region filter
	void TileClusteringHDKF::filter(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range,
		ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		double downsampleRate, int downsampleMethod, double truncateBoundary, const int borderType)
	{
		guide_channels = channels = src.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			scbf[0]->filter(src, dst, sigma_space, sigma_range,
				cm, K, gf_method, gf_order, depth,
				downsampleRate, downsampleMethod, borderType);
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
				subImageOutput.resize(thread_max);
				for (int n = 0; n < thread_max; n++)
				{
					subImageInput[n].resize(channels);
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}

			if (src.channels() != 3)split(src, srcSplit);

			const int threadnum = omp_get_max_threads();
			if (guiTestIndex < 0) omp_set_num_threads(1);
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
				if (n == guiTestIndex) scbf[thread_num]->setTestClustering(true);
				else scbf[thread_num]->setTestClustering(false);

				scbf[thread_num]->filter(subImageInput[thread_num], subImageOutput[thread_num], sigma_space, sigma_range,
					cm, K, gf_method, gf_order, depth,
					downsampleRate, downsampleMethod, R, borderType);
				//merge(subImageInput[thread_num], subImageOutput[thread_num]);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, vecsize, vecsize);
			}

			if (guiTestIndex < 0) omp_set_num_threads(threadnum);
		}
	}

	static void sortChannel(cv::Mat& src, cv::Mat& dest, const int index)
	{
		const int ch = src.channels();
		std::vector<cv::Mat> v(src.size().area());
		for (int i = 0; i < src.size().area(); i++)
		{
			v[i] = src.row(i).clone();
		}

		if (index == 0) std::sort(v.begin(), v.end(), [](cv::Mat& a, cv::Mat& b) {return a.at<float>(0) < b.at<float>(0); });
		if (index == 1) std::sort(v.begin(), v.end(), [](cv::Mat& a, cv::Mat& b) {return a.at<float>(1) < b.at<float>(1); });
		if (index == 2) std::sort(v.begin(), v.end(), [](cv::Mat& a, cv::Mat& b) {return a.at<float>(2) < b.at<float>(2); });
		if (index == 3) std::sort(v.begin(), v.end(), [](cv::Mat& a, cv::Mat& b) {return a.at<float>(3) < b.at<float>(3); });

		float* d = dest.ptr<float>();
		for (int i = 0; i < src.size().area(); i++)
		{
			for (int c = 0; c < ch; c++)
			{
				*d++ = v[i].at<float>(c);
			}
		}
	}

	void TileClusteringHDKF::PCAfilter(const cv::Mat& src, const int reduce_dim, cv::Mat& dst, double sigma_space, double sigma_range,
		ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		double downsampleRate, int downsampleMethod, double truncateBoundary, const int borderType)
	{
		guide_channels = channels = src.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			scbf[0]->filter(src, dst, sigma_space, sigma_range,
				cm, K, gf_method, gf_order, depth,
				downsampleRate, downsampleMethod, borderType);
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
				subImageOutput.resize(thread_max);
				for (int n = 0; n < thread_max; n++)
				{
					subImageInput[n].resize(channels);
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}

			if (src.channels() != 3)split(src, srcSplit);

			const int threadnum = omp_get_max_threads();
			if (guiTestIndex < 0) omp_set_num_threads(1);
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

				scbf[thread_num]->PCAfilter(subImageInput[thread_num], reduce_dim, subImageOutput[thread_num], sigma_space, sigma_range,
					cm, K, gf_method, gf_order, depth,
					downsampleRate, downsampleMethod, R, borderType);
				if (isDebug) mu[n] = scbf[thread_num]->getSamplingPoints();
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, vecsize, vecsize);
			}

			if (guiTestIndex < 0) omp_set_num_threads(threadnum);
			if (isDebug)
			{
				cv::Mat p = mu[0].reshape(reduce_dim);
				sortChannel(p, p, 0);

				for (int i = 0; i < mu[0].rows; i++)
				{
					float* v = p.ptr<float>(i);
					if (reduce_dim == 1)ci("%5.2f", v[0]);
					if (reduce_dim == 2)ci("%5.2f %5.2f", v[0], v[1]);
					if (reduce_dim == 3)ci("%5.2f %5.2f %5.2f", v[0], v[1], v[2]);
					if (reduce_dim == 4)ci("%5.2f %5.2f %5.2f %5.2f", v[0], v[1], v[2], v[3]);
				}

				ci.show();
				cv::waitKey();
			}
		}
	}

	void TileClusteringHDKF::jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range,
		ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		double downsampleRate, int downsampleMethod, double truncateBoundary, const int borderType)
	{
		//std::cout << "jointfilter" << std::endl;
		channels = src.channels();
		guide_channels = guide.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, channels));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			const int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			Mat srcB; copyMakeBorder(src, srcB, R, R, R, R, borderType);
			Mat guideB; copyMakeBorder(guide, guideB, R, R, R, R, borderType);
			split(srcB, srcSplit);
			split(guideB, guideSplit);
			Mat dest;
			scbf[0]->jointfilter(srcSplit, guideSplit, dest, sigma_space, sigma_range,
				cm, K, gf_method, gf_order, depth,
				downsampleRate, downsampleMethod, R, borderType);
			tileSize = src.size();
			dest(Rect(R, R, src.cols, src.rows)).copyTo(dst);
		}
		else
		{
			const int r = (int)ceil(truncateBoundary * sigma_space);
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

			if (src.channels() != 3) split(src, srcSplit);
			if (guide.channels() != 3) split(guide, guideSplit);

			const int threadnum = omp_get_max_threads();
			if (guiTestIndex >= 0) omp_set_num_threads(1);
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

				if (n == guiTestIndex)
				{
					scbf[thread_num]->setTestClustering(true);
					Mat debugshow; merge(subImageGuide[thread_num], debugshow);
					imshowScale("clustering input image", debugshow); waitKey(1);
				}
				else scbf[thread_num]->setTestClustering(false);

				scbf[thread_num]->jointfilter(subImageInput[thread_num], subImageGuide[thread_num], subImageOutput[thread_num], sigma_space, sigma_range,
					cm, K, gf_method, gf_order, depth,
					downsampleRate, downsampleMethod, R, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, vecsize, vecsize);
			}

			if (guiTestIndex >= 0) omp_set_num_threads(threadnum);
		}
	}

	void TileClusteringHDKF::jointPCAfilter(const cv::Mat& src, const cv::Mat& guide, const int reduced_dim, cv::Mat& dst, double sigma_space, double sigma_range,
		ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		double downsampleRate, int downsampleMethod, double truncateBoundary, const int borderType)
	{
		CV_Assert(!src.empty());
		CV_Assert(!guide.empty());

		channels = src.channels();
		guide_channels = std::min(reduced_dim, guide.channels());

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			split(src, srcSplit);
			/*
			cv::Mat guidepca;
			cp::cvtColorPCA(guide, guidepca, guide_channels);
			split(guidepca, guideSplit);
			scbf[0]->jointfilter(srcSplit, guideSplit, dst, sigma_space, sigma_range, cm, K, gf_method
				, gf_order, depth, isDownsampleClustering, downsampleRate, downsampleMethod);
			*/
			//std::cout << "cal" << std::endl;
			split(guide, guideSplit);
			scbf[0]->jointPCAfilter(srcSplit, guideSplit, guide_channels, dst, sigma_space, sigma_range,
				cm, K, gf_method, gf_order, depth,
				downsampleRate, downsampleMethod, borderType);
			tileSize = src.size();

			eigenVectors[0] = scbf[0]->cloneEigenValue();
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
					subImageGuide[n].resize(guide.channels());
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}
			else
			{
				//std::cout << "here!!" << std::endl;
				if (subImageGuide.empty()) subImageGuide.resize(thread_max);

				if (subImageGuide[0].size() != guide.channels())
				{
					for (int n = 0; n < thread_max; n++)
					{
						subImageGuide[n].resize(guide.channels());
					}
			}
		}

			if (src.channels() != 3) split(src, srcSplit);
			if (guide.channels() != 3) split(guide, guideSplit);

#if DEBUG_CLUSTERING
			Mat emap(src.size(), CV_32F);
#endif

			const int threadnum = omp_get_max_threads();
			if (guiTestIndex < 0) omp_set_num_threads(1);
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

				if (n == guiTestIndex) scbf[thread_num]->setTestClustering(true);
				else scbf[thread_num]->setTestClustering(false);

				scbf[thread_num]->jointPCAfilter(subImageInput[thread_num], subImageGuide[thread_num], guide_channels, subImageOutput[thread_num],
					sigma_space, sigma_range, cm, K, gf_method, gf_order, depth,
					downsampleRate, downsampleMethod, R, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
				eigenVectors[n] = scbf[thread_num]->cloneEigenValue();
#if DEBUG_CLUSTERING
				Mat e; scbf[thread_num]->getClusteringErrorMap(e); cp::pasteTileAlign(e, emap, div, idx, r, 8, 8);
#endif
				}
#if DEBUG_CLUSTERING
			imshowScale("emap", emap);
#endif
			if (guiTestIndex < 0) omp_set_num_threads(threadnum);
			}
		//scbf[0]->printRapTime();
	}

	void TileClusteringHDKF::jointPCAfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, const int reduced_dim, cv::Mat& dst, double sigma_space, double sigma_range,
		ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		double downsampleRate, int downsampleMethod, double truncateBoundary, const int borderType)
	{
		CV_Assert(!src.empty());
		CV_Assert(!guide.empty());

		channels = (int)src.size();
		guide_channels = std::min(reduced_dim, (int)guide.size());

		if (dst.empty() || dst.size() != src[0].size()) dst.create(src[0].size(), CV_MAKETYPE(CV_32F, channels));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			cp::cvtColorPCA(guide, guideSplit, guide_channels);
			scbf[0]->jointfilter(src, guideSplit, dst, sigma_space, sigma_range,
				cm, K, gf_method, gf_order, depth,
				downsampleRate, downsampleMethod, borderType);
			tileSize = src[0].size();
			eigenVectors[0] = scbf[0]->cloneEigenValue();
		}
		else
		{
			int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			tileSize = cp::getTileAlignSize(src[0].size(), div, r, vecsize, vecsize);
			divImageSize = cv::Size(src[0].cols / div.width, src[0].rows / div.height);

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
					subImageGuide[n].resize((int)guide.size());
					subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
				}
			}
			else
			{
				//std::cout << "here!!" << std::endl;
				if (subImageGuide.empty()) subImageGuide.resize(thread_max);

				if (subImageGuide[0].size() != guide_channels)
				{
					for (int n = 0; n < thread_max; n++)
					{
						subImageGuide[n].resize((int)guide.size());
					}
				}
		}
#if DEBUG_CLUSTERING
			Mat emap(src[0].size(), CV_32F);
#endif
			const int threadnum = omp_get_max_threads();
			if (guiTestIndex < 0) omp_set_num_threads(1);
#pragma omp parallel for schedule(static)
			for (int n = 0; n < div.area(); n++)
			{
				const int thread_num = omp_get_thread_num();
				const cv::Point idx = cv::Point(n % div.width, n / div.width);

				for (int c = 0; c < src.size(); c++)
				{
					cp::cropTileAlign(src[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}

				for (int c = 0; c < guide.size(); c++)
				{
					cp::cropTileAlign(guide[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}

				scbf[thread_num]->jointPCAfilter(subImageInput[thread_num], subImageGuide[thread_num], guide_channels, subImageOutput[thread_num],
					sigma_space, sigma_range, cm, K, gf_method, gf_order, depth,
					downsampleRate, downsampleMethod, R, borderType);
				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, vecsize, vecsize);

				eigenVectors[n] = scbf[thread_num]->cloneEigenValue();
#if DEBUG_CLUSTERING
				Mat e; scbf[thread_num]->getClusteringErrorMap(e); cp::pasteTileAlign(e, emap, div, idx, r, 8, 8);
#endif
				}
#if DEBUG_CLUSTERING
			imshowScale("emap", emap);
#endif
			if (guiTestIndex < 0) omp_set_num_threads(threadnum);
			}
	}

	void TileClusteringHDKF::nlmfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range,
		const int patch_r, const int reduced_dim, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
		double downsampleRate, int downsampleMethod, double truncateBoundary, const int borderType)
	{
		channels = src.channels();
		guide_channels = guide.channels();

		if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			scbf[0]->jointfilter(src, guide, dst, sigma_space, sigma_range,
				cm, K, gf_method, gf_order, depth,
				downsampleRate, downsampleMethod, borderType);
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
#if DEBUG_CLUSTERING
			Mat emap(src.size(), CV_32F);
#endif
			const int threadnum = omp_get_max_threads();
			if (guiTestIndex < 0) omp_set_num_threads(1);
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
				//createSubImageCVAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				scbf[thread_num]->nlmfilter(subImageInput[thread_num], subImageGuide[thread_num], subImageOutput[thread_num], sigma_space, sigma_range,
					patch_r, reduced_dim, cm, K, gf_method, gf_order, depth,
					downsampleRate, downsampleMethod, R, borderType);
				//merge(subImageInput[thread_num], subImageOutput[thread_num]);

				cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
				eigenVectors[n] = scbf[thread_num]->cloneEigenValue();
#if DEBUG_CLUSTERING
				Mat e; scbf[thread_num]->getClusteringErrorMap(e); cp::pasteTileAlign(e, emap, div, idx, r, 8, 8);
#endif
		}
#if DEBUG_CLUSTERING
			imshowScale("emap", emap);
#endif
			if (guiTestIndex < 0) omp_set_num_threads(threadnum);
	}
}
#pragma endregion

	cv::Size TileClusteringHDKF::getTileSize()
	{
		return tileSize;
	}

	void TileClusteringHDKF::getTileInfo()
	{
		print_debug(div);
		print_debug(divImageSize);
		print_debug(tileSize);
		int borderLength = (tileSize.width - divImageSize.width) / 2;
		print_debug(borderLength);
	}

	void TileClusteringHDKF::getEigenValueInfo()
	{
		//static ConsoleImage ci(Size(640, 800), "Average Eigen Vectors", true);
		Mat a = eigenVectors[0];
		for (int i = 1; i < div.area(); i++)
		{
			add(a, eigenVectors[i], a);
		}
		double total = 0.0;
		for (int i = 0; i < a.size().area(); i++)
		{
			total += a.at<double>(i) / div.area();
		}
		for (int i = 0; i < a.size().area(); i++)
		{
			const double ave = a.at<double>(i) / div.area();
			std::cout << i << ": " << ave << " (" << ave / total * 100.0 << " %)" << std::endl;
			//ci("%d: %f (%f) %%", i, ave, ave / total * 100.0);
		}
		//ci.show();
		//ci.clear();
	}
}