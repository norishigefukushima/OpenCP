#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
	SpatialFilterBox::SpatialFilterBox(const cp::BoxFilterMethod boxfilter_type, const int dest_depth)
	{
		this->dest_depth = dest_depth;
		this->depth = dest_depth;
	}

	void SpatialFilterBox::body(const cv::Mat& src, cv::Mat& dst, int borderType)
	{
		//boxFilter(src, dst, src.depth(), Size(2 * radius + 1, 2 * radius + 1), Point(-1, -1), true, border);
		//blur(src, dst, Size(2 * radius + 1, 2 * radius + 1), Point(-1, -1), border);
		
		if (src.depth() == dest_depth)
		{
			switch (dest_depth)
			{
				
			case CV_8U: cp::boxFilter_8u(const_cast<Mat&>(src), dst, radius, boxfilter_type, NAIVE);break;
			case CV_32F: cp::boxFilter_32f(const_cast<Mat&>(src), dst, radius, boxfilter_type, NAIVE);break;
			case CV_64F: cp::boxFilter_64f(const_cast<Mat&>(src), dst, radius, boxfilter_type, NAIVE);break;
			default:
				break;
			}
		}
		//
		//box->filter();
	}

	void SpatialFilterBox::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		this->sigma = sigma;
		this->gf_order = order;
		this->radius = (int)ceil(sigma * order);

		body(src, dst, borderType);
	}

}