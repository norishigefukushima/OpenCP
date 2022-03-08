#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
	//===============================================================================
	//FIR
	GaussianFilterFIROpenCV::GaussianFilterFIROpenCV(cv::Size imgSize, double sigma, int trunc, int depth)
		: SpatialFilterBase(imgSize, depth)
	{
		this->algorithm = SpatialFilterAlgorithm::FIR_OPENCV;
		this->sigma = sigma;
		this->gf_order = trunc;
		this->radius = (int)ceil(gf_order * sigma);
	}

	GaussianFilterFIROpenCV::GaussianFilterFIROpenCV(const int dest_depth, const bool isCompute32F)
	{
		this->algorithm = SpatialFilterAlgorithm::FIR_OPENCV;
		this->depth = isCompute32F ? CV_32F : CV_64F;
		this->dest_depth = dest_depth;
	}

	GaussianFilterFIROpenCV::~GaussianFilterFIROpenCV()
	{
		;
	}

	void GaussianFilterFIROpenCV::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		this->imgSize = src.size();
		this->d = 2 * this->radius + 1;

		if (dest_depth == src.depth() && this->depth == src.depth())
		{
			GaussianBlur(src, dst, Size(d, d), sigma, sigma, borderType);
		}
		else
		{
			cv::Mat internalBuff;
			src.convertTo(internalBuff, this->depth);
			GaussianBlur(internalBuff, internalBuff, Size(d, d), sigma, sigma, borderType);
			internalBuff.convertTo(dst, dest_depth);
		}
	}

	void GaussianFilterFIROpenCV::filter(const cv::Mat& _src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		this->sigma = sigma;
		this->gf_order = order;
		this->radius = (gf_order == 0) ? radius : (int)ceil(gf_order * sigma);
		body(_src, dst, borderType);
	}
}