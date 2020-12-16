#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT KMeans
	{
		float sigma = 0.f;
		cv::Mat labels_internal;
		cv::AutoBuffer<float> _distance;

		//initialize
		void generateKmeansRandomInitialCentroid(cv::Mat& data_points, cv::Mat& centroids, const int K, cv::RNG& rng);
		void generateKmeansPPInitialCentroid_AVX(const cv::Mat& data, cv::Mat& _out_centers, int K, cv::RNG& rng, int trials);

		//get most outer samples for centroids (not used)
		void getOuterSample(cv::Mat& src_centroids, cv::Mat& dest_centroids, const cv::Mat& data_points, const cv::Mat& labels);

		//computing centroid
		void boxMeanCentroid(cv::Mat& data_points, const int* labels, cv::Mat& dest_centroid, int* counters);//simple average
		void weightedMeanCentroid(cv::Mat& data_points, const int* labels, const cv::Mat& src_centroid, float* Table, cv::Mat& dest_centroid, float* centroid_weight, int* counters);
		void harmonicMeanCentroid(cv::Mat& data_points, const int* labels, const cv::Mat& src_centroid, cv::Mat& dest_centroid, float* centroid_weight, int* counters);
	public:

		enum class MeanFunction
		{
			Mean,//box
			Gauss,
			GaussInv,
			Harmonic
		};
		void setSigma(const float sigma) { this->sigma = sigma; }
		double clustering(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers, MeanFunction function = MeanFunction::Mean);
	};

	CP_EXPORT double kmeans(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers);
}
