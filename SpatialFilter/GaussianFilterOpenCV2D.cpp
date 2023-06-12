#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
	template<typename T>
	static void generateSepGaussKernel(T* data, const int r, const double sigma)
	{
		const int size = 2 * r + 1;
		double sum = 1.0;
		const double coeff = -1.0 / (2.0 * sigma * sigma);

		data[0] = 1.0;
		for (int i = 1; i <= r; i++)
		{
			const double v = exp(i * i * coeff);
			data[-i] = data[i] = (T)v;
			sum += 2.0 * v;
			//cout << i<<": "<<v << endl;
		}
		const int rend = int(9.0 * sigma);
		double eout = 0.0;
		for (int i = rend; i >= r + 1; i--)
		{
			const double v = exp(i * i * coeff);
			eout += v;
		}
		//static int aaa = 100; createTrackbar("aaa", "", &aaa, 200);

		//const double alpha = 1.0-(ceil(double(r) / sigma) - double(r) / sigma);
		const double alpha = 1.0;
		//const double alpha = 2.0 - aaa * 0.01;
		//const double alpha = 0.0;
		data[r] += (T)(eout * alpha);
		data[-r] += (T)(eout * alpha);
		const double ialpha = 1.0 - alpha;
		data[r - 1] += (T)(eout * ialpha);
		data[-r + 1] += (T)(eout * ialpha);

		sum += 2.0 * eout;

		//cout << "o: "<<eout << endl;
		const double inv = 1.0 / sum;
		T* data2 = &data[-r];
		for (int i = 0; i < size; i++)
		{
			data2[i] = (T)(data2[i] * inv);
		}
	}

	static void conv2D(const Mat& src, Mat& dest, const int r, const double sigma, const int borderType)
	{
		const int size = 2 * r + 1;
		Mat gauss;
		if (src.depth() == CV_8U || src.depth() == CV_32F)
		{
			gauss.create(Size(size, 1), CV_32F);
			float* data = gauss.ptr<float>(0, r);
			generateSepGaussKernel<float>(data, r, sigma);
		}
		else if (src.depth() == CV_64F)
		{
			gauss.create(Size(size, 1), CV_64F);
			double* data = gauss.ptr<double>(0, r);
			generateSepGaussKernel<double>(data, r, sigma);
		}
		else
		{
			cout << "not support this type" << src.depth() << endl;
		}

		Mat kernel = gauss.t() * gauss;
		filter2D(src, dest, dest.depth(), kernel, Point(-1, -1), 0.0, borderType);
	}

	//===============================================================================
	//FIR
	GaussianFilterFIROpenCVFilter2D::GaussianFilterFIROpenCVFilter2D(cv::Size imgSize, double sigma, int trunc, int depth)
		: SpatialFilterBase(imgSize, depth)
	{
		this->algorithm = SpatialFilterAlgorithm::FIR_OPENCV_FILTER2D;
		this->sigma = sigma;
		this->gf_order = trunc;
		this->radius = (int)ceil(gf_order * sigma);
	}

	GaussianFilterFIROpenCVFilter2D::GaussianFilterFIROpenCVFilter2D(const int dest_depth, const bool isCompute32F)
	{
		this->algorithm = SpatialFilterAlgorithm::FIR_OPENCV_FILTER2D;
		this->depth = isCompute32F ? CV_32F : CV_64F;
		this->dest_depth = dest_depth;
	}

	GaussianFilterFIROpenCVFilter2D::~GaussianFilterFIROpenCVFilter2D()
	{
		;
	}

	void GaussianFilterFIROpenCVFilter2D::body(const cv::Mat& src, cv::Mat& dst, const int borderType)
	{
		if (dest_depth == src.depth() && this->depth == src.depth())
		{
			if (src.depth() == CV_8U)
			{
				//cv::Mat internalBuff;
				src.convertTo(internalBuff, CV_32F);
				conv2D(internalBuff, internalBuff, radius, sigma, borderType);
				internalBuff.convertTo(dst, dest_depth);
			}
			else
			{
				conv2D(src, dst, radius, sigma, borderType);
			}
		}
		else
		{
			//cv::Mat internalBuff;
			src.convertTo(internalBuff, this->depth);
			conv2D(internalBuff, internalBuff, radius, sigma, borderType);
			internalBuff.convertTo(dst, dest_depth);
		}
	}

	void GaussianFilterFIROpenCVFilter2D::filter(const cv::Mat& _src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		this->sigma = sigma;
		this->gf_order = order;
		this->radius = (gf_order == 0) ? radius : (int)ceil(gf_order * sigma);

		body(_src, dst, borderType);
	}
}