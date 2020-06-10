#pragma once

#include "common.hpp" 

namespace cp
{
	//not parallel processing
	class CP_EXPORT CrossBasedLocalFilter
	{
		//private:
	public:
		int minSearch = 0;

		cv::Size size;
		int r;
		int thresh;

		struct CP_EXPORT cross
		{
			uchar hp;
			uchar hm;
			float divh;
			uchar vp;
			uchar vm;
			float divv;
		};
		cross* crossdata = nullptr;

		enum
		{
			CROSS_BASED_LOCAL_FILTER_ARM_BASIC = 0,
			CROSS_BASED_LOCAL_FILTER_ARM_SAMELENGTH,
			CROSS_BASED_LOCAL_FILTER_ARM_SMOOTH_SAMELANGTH
		};
		void setMinSearch(int val);
		cv::Mat areaMap;
		
		CrossBasedLocalFilter();
		CrossBasedLocalFilter(cv::Mat& guide, const int r, const int thresh_);
		~CrossBasedLocalFilter();

		void getCrossAreaCountMap(cv::Mat& dest, int type = CV_8U);

		void makeKernel(cv::Mat& guide, const int r, const int thresh, const int method = CrossBasedLocalFilter::CROSS_BASED_LOCAL_FILTER_ARM_BASIC);
		void makeKernel(cv::Mat& guide, const int r, const int thresh, const double smoothingrate, const int method = CrossBasedLocalFilter::CROSS_BASED_LOCAL_FILTER_ARM_BASIC);
		void visualizeKernel(cv::Mat& dest, cv::Point& pt);

		void operator()(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, int thresh, int iteration = 1);
		void operator()(cv::Mat& src, cv::Mat& dest);
		void operator()(cv::Mat& src, cv::Mat& weight, cv::Mat& guide, cv::Mat& dest, const int r, int thresh, int iteration = 1);
		void operator()(cv::Mat& src, cv::Mat& weight, cv::Mat& dest);

	};

	void guiCrossBasedLocalFilter(cv::Mat& src);
	void crossBasedAdaptiveBoxFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const cv::Size ksize, const int thresh);
	void crossBasedAdaptiveBoxFilter(cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const int thresh);

}