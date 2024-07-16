#include "pch.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void randomSetCentorid(const int K, const int destChannels, cv::Mat& dest_center, float minv, float maxv)
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

void randomSampleCentroid(std::vector<cv::Mat>& vsrc32f, const int K, cv::Mat& dest_center)
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

void randomSampleCentroid(cv::Mat& src32f, const int K, cv::Mat& dest_center)
{
	//randomSet(K, src32f.rows, dest_center, 0.f, 255.f);
	dest_center.create(K, src32f.rows, CV_32F);
	//print_debug3(src32f.cols, src32f.rows, dest_center.cols);

	cv::RNG rng(cv::getTickCount());

	const float* s = src32f.ptr<float>();
	for (int k = 0; k < K; k++)
	{
		const int idx = rng.uniform(0, src32f.cols);
		for (int c = 0; c < src32f.rows; c++)
		{
			dest_center.at<float>(k, c) = s[src32f.cols * c + idx];
		}
	}
}

void randomSampleShuffleCentroid(cv::Mat& src32f, const int K, cv::Mat& dest_center)
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