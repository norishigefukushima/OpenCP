#include "stdafx.h"

using namespace std;
using namespace cv;

namespace cp
{
	std::string getAlgorithmName(SpatialFilterAlgorithm method)
	{
		std::string ret;
		switch (method)
		{
		case SpatialFilterAlgorithm::FIR_OPENCV:			ret = "FIR_OPENCV";			break;
		case SpatialFilterAlgorithm::FIR_Sep2D_OPENCV:		ret = "FIR_Sep2D_OPENCV";	break;
		case SpatialFilterAlgorithm::FIR_OPENCV_64F:		ret = "FIR_OPENCV_64F";		break;
		case SpatialFilterAlgorithm::FIR_SEPARABLE:			ret = "FIR_SEPARABLE";		break;
		case SpatialFilterAlgorithm::FIR_KAHAN:				ret = "FIR_KAHAN";			break;

		case SpatialFilterAlgorithm::IIR_AM:				ret = "IIR_AM";				break;
		case SpatialFilterAlgorithm::IIR_AM_NAIVE:			ret = "IIR_AM_NAIVE";		break;
		case SpatialFilterAlgorithm::IIR_VYV:				ret = "IIR_VYV";			break;
		case SpatialFilterAlgorithm::IIR_VYV_NAIVE:			ret = "IIR_VYV_NAIVE";		break;
		case SpatialFilterAlgorithm::IIR_DERICHE:			ret = "IIR_DERICHE";		break;
		case SpatialFilterAlgorithm::IIR_DERICHE_NAIVE:		ret = "IIR_DERICHE_NAIVE";	break;

		case SpatialFilterAlgorithm::BOX:					ret = "BOX";				break;

		case SpatialFilterAlgorithm::SlidingDCT1_AVX:		ret = "SlidingDCT-1_32F_AVX"; break;
		case SpatialFilterAlgorithm::SlidingDCT1_CONV:		ret = "SlidingDCT-1_CONV"; break;
		case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:	ret = "SlidingDCT-1_64F_AVX"; break;

		case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:	ret = "SlidingDCT-3_16F_AVX"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_AVX:		ret = "SlidingDCT-3_32F_AVX"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_VXY:		ret = "SlidingDCT-3_32F_VYX"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_CONV:		ret = "SlidingDCT-3_CONV"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_DEBUG:		ret = "SlidingDCT-3_DEBUG"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:	ret = "SlidingDCT-3_64F_AVX"; break;

		case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:	ret = "SlidingDCT-5_16F_AVX"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_AVX:		ret = "SlidingDCT-5_32F_AVX"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_VXY:		ret = "SlidingDCT-5_32F_VXY"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_CONV:		ret = "SlidingDCT-5_CONV"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_DEBUG:		ret = "SlidingDCT-5_DEBUG"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:	ret = "SlidingDCT-5_64F_AVX"; break;
#ifdef CP_AVX_512
		case SpatialFilterAlgorithm::SlidingDCT5_AVX512:	ret = "SlidingDCT-5_AVX512";	break;
#endif
		case SpatialFilterAlgorithm::SlidingDCT7_AVX:		ret = "SlidingDCT-7_32F_AVX"; break;
		case SpatialFilterAlgorithm::SlidingDCT7_VXY:		ret = "SlidingDCT-7_32F_VXY"; break;
		case SpatialFilterAlgorithm::SlidingDCT7_CONV:		ret = "SlidingDCT-7_CONV"; break;
		case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:	ret = "SlidingDCT-7_64F_AVX"; break;


		case SpatialFilterAlgorithm::DCT_AVX:				ret = "DCT_AVX";			break;
		default:											ret = "";					break;
		}

		return ret;
	}

	std::string getAlgorithmNameShort(SpatialFilterAlgorithm method)
	{
		std::string ret;
		switch (method)
		{
		case SpatialFilterAlgorithm::FIR_OPENCV:			ret = "FIR";			break;
		case SpatialFilterAlgorithm::FIR_Sep2D_OPENCV:		ret = "FIR";	break;
		case SpatialFilterAlgorithm::FIR_OPENCV_64F:		ret = "FIR";		break;
		case SpatialFilterAlgorithm::FIR_SEPARABLE:			ret = "FIR";		break;
		case SpatialFilterAlgorithm::FIR_KAHAN:				ret = "FIR_KAHAN";			break;

		case SpatialFilterAlgorithm::IIR_AM:				ret = "IIR_AM";				break;
		case SpatialFilterAlgorithm::IIR_AM_NAIVE:			ret = "IIR_AM_NAIVE";		break;
		case SpatialFilterAlgorithm::IIR_VYV:				ret = "IIR_VYV";			break;
		case SpatialFilterAlgorithm::IIR_VYV_NAIVE:			ret = "IIR_VYV_NAIVE";		break;
		case SpatialFilterAlgorithm::IIR_DERICHE:			ret = "IIR_DERICHE";		break;
		case SpatialFilterAlgorithm::IIR_DERICHE_NAIVE:		ret = "IIR_DERICHE_NAIVE";	break;

		case SpatialFilterAlgorithm::BOX:					ret = "BOX";				break;

		case SpatialFilterAlgorithm::SlidingDCT1_AVX:		ret = "SlidingDCT-1"; break;
		case SpatialFilterAlgorithm::SlidingDCT1_CONV:		ret = "SlidingDCT-1"; break;
		case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:	ret = "SlidingDCT-1"; break;

		case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:	ret = "SlidingDCT-3"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_AVX:		ret = "SlidingDCT-3"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_VXY:		ret = "SlidingDCT-3"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_CONV:		ret = "SlidingDCT-3"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_DEBUG:		ret = "SlidingDCT-3"; break;
		case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:	ret = "SlidingDCT-3"; break;

		case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:	ret = "SlidingDCT-5"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_AVX:		ret = "SlidingDCT-5"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_VXY:		ret = "SlidingDCT-5"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_CONV:		ret = "SlidingDCT-5"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_DEBUG:		ret = "SlidingDCT-5"; break;
		case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:	ret = "SlidingDCT-5"; break;
#ifdef CP_AVX_512
		case SpatialFilterAlgorithm::SlidingDCT5_AVX512:	ret = "SlidingDCT5_AVX512";	break;
#endif
		case SpatialFilterAlgorithm::SlidingDCT7_AVX:		ret = "SlidingDCT-7"; break;
		case SpatialFilterAlgorithm::SlidingDCT7_VXY:		ret = "SlidingDCT-7"; break;
		case SpatialFilterAlgorithm::SlidingDCT7_CONV:		ret = "SlidingDCT-7"; break;
		case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:	ret = "SlidingDCT-7"; break;


		case SpatialFilterAlgorithm::DCT_AVX:				ret = "DCT";			break;
		default:											ret = "";					break;
		}

		return ret;
	}

	//===============================================================================
#pragma region base class for Gaussian filter
	SpatialFilterBase::SpatialFilterBase(cv::Size imgSize, int depth)
		: imgSize(imgSize), depth(depth)
	{
		;
	}

	SpatialFilterAlgorithm SpatialFilterBase::getAlgorithmType()
	{
		return algorithm;
	}

	int SpatialFilterBase::getOrder()
	{
		return gf_order;
	}

	double SpatialFilterBase::getSigma()
	{
		return sigma;
	}

	Size SpatialFilterBase::getSize()
	{
		return imgSize;
	}

	void SpatialFilterBase::computeRadius(const int radius)
	{
		this->radius = radius;
	}

	void SpatialFilterBase::setFixRadius(const int radius)
	{
		this->radius = radius;
		this->isUseFixRadius = true;
	}

	void SpatialFilterBase::unsetFixRadius()
	{
		this->isUseFixRadius = false;
	}

	int SpatialFilterBase::getRadius()
	{
		return radius;
	}

	void SpatialFilterBase::setIsInner(const int top, const int bottom, const int left, const int right)
	{
		this->top = top;
		this->bottom = bottom;
		this->left = left;
		this->right = right;
	}
#pragma endregion


	cv::Ptr<cp::SpatialFilterBase> createSpatialFilter(const cp::SpatialFilterAlgorithm method, const int dest_depth, const SpatialKernel skernel, const int option)
	{
		const DCT_COEFFICIENTS dct_coeff = (option == 0) ? DCT_COEFFICIENTS::FULL_SEARCH_OPT : DCT_COEFFICIENTS::FULL_SEARCH_NOOPT;

		if (dest_depth == CV_8U || dest_depth == CV_32F)
		{
			switch (method)
			{
			case SpatialFilterAlgorithm::DCT_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterDCT_AVX_32F(dest_depth)); break;

			case SpatialFilterAlgorithm::FIR_OPENCV:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIROpenCV(dest_depth, true)); break;
			case SpatialFilterAlgorithm::FIR_OPENCV_64F:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIROpenCV(dest_depth, false)); break;
			case SpatialFilterAlgorithm::FIR_Sep2D_OPENCV:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIRSep2DOpenCV(dest_depth, true)); break;
			case SpatialFilterAlgorithm::FIR_KAHAN:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIRKahan(dest_depth)); break;
			case SpatialFilterAlgorithm::FIR_SEPARABLE:
			{
				//int schedule = GaussianFilterSeparableFIR::VH_BorderD;
				//int schedule = GaussianFilterSeparableFIR::VHI_LineBH;
				int schedule = GaussianFilterSeparableFIR::VHI_Border;//OK
				//int schedule = GaussianFilterSeparableFIR::VHI_Image;
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterSeparableFIR(schedule, CV_32F)); break;
			}

			case SpatialFilterAlgorithm::IIR_AM:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterAM_AVX_32F(dest_depth)); break;
			case SpatialFilterAlgorithm::IIR_VYV:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterVYV_AVX_32F(dest_depth)); break;
			case SpatialFilterAlgorithm::IIR_DERICHE:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterDERICHE_AVX_32F(dest_depth)); break;

			case SpatialFilterAlgorithm::BOX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterBox(cp::BoxFilterMethod::OPENCV, dest_depth)); break;

			case SpatialFilterAlgorithm::SlidingDCT1_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT1_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT1_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT1_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT1_AVX_64F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;

			case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT3_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT3_VXY:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::V_XY_LOOP, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT3_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_64F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT3_DEBUG:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEBUG, skernel)); break;

			case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::INNER_LOW_PRECISION, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT5_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT5_VXY:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::V_XY_LOOP, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_64F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT5_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT5_DEBUG:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEBUG)); break;

			case SpatialFilterAlgorithm::SlidingDCT7_AVX:
				//cout << "test createSpatialFilter 7->5 noopt" << endl;
				//return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(DCT_COEFFICIENTS::FULL_SEARCH_NOOPT, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT7_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT7_VXY:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT7_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::V_XY_LOOP, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT7_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT7_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT7_AVX_64F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;

			default:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_32F(dct_coeff, dest_depth, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
				cout << getAlgorithmName(method) + " is not support in GF (class GaussianFilter)." << endl;
				break;
			}
		}
		else if (dest_depth == CV_64F)
		{
			switch (method)
			{
			case SpatialFilterAlgorithm::DCT_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterDCT_AVX_64F(dest_depth)); break;

			case SpatialFilterAlgorithm::FIR_OPENCV:
			case SpatialFilterAlgorithm::FIR_OPENCV_64F:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIROpenCV(dest_depth, false)); break;
			case SpatialFilterAlgorithm::FIR_Sep2D_OPENCV:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIRSep2DOpenCV(dest_depth, false)); break;
			case SpatialFilterAlgorithm::FIR_KAHAN:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterFIRKahan(dest_depth)); break;
			case SpatialFilterAlgorithm::FIR_SEPARABLE:
			{
				int schedule = GaussianFilterSeparableFIR::HV_BorderD;
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterSeparableFIR(schedule, CV_64F)); break;
			}

			case SpatialFilterAlgorithm::IIR_AM:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterAM_AVX_64F(dest_depth)); break;
			case SpatialFilterAlgorithm::IIR_VYV:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterVYV_AVX_64F(dest_depth)); break;
			case SpatialFilterAlgorithm::IIR_DERICHE:
				return cv::Ptr<cp::SpatialFilterBase>(new GaussianFilterDERICHE_AVX_64F(dest_depth)); break;

			case SpatialFilterAlgorithm::BOX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterBox(cp::BoxFilterMethod::OPENCV, dest_depth)); break;

			case SpatialFilterAlgorithm::SlidingDCT1_AVX:
			case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT1_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT1_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT1_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;

			case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT3_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT3_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;

			case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT5_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;

			case SpatialFilterAlgorithm::SlidingDCT7_AVX:
			case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT7_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
			case SpatialFilterAlgorithm::SlidingDCT7_CONV:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT7_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::CONVOLUTION, skernel)); break;

			default:
				return cv::Ptr<cp::SpatialFilterBase>(new SpatialFilterSlidingDCT5_AVX_64F(dct_coeff, CV_64F, SLIDING_DCT_SCHEDULE::DEFAULT, skernel)); break;
				cout << getAlgorithmName(method) + " is not support in GF (class GaussianFilter)." << endl;
			}
		}
		else
		{
			cout << "do not suport this data type in createGaussianFilter. Only support 32F and 64F." << endl;
		}

		return nullptr;
	}

#pragma region implement class (GaussianFilter)

	SpatialFilter::SpatialFilter(const SpatialFilterAlgorithm method, const int dest_depth, const SpatialKernel skernel, const int option)
	{
		gauss = createSpatialFilter(method, dest_depth, skernel, option);
	}

	SpatialFilterAlgorithm SpatialFilter::getMethodType()
	{
		return gauss->getAlgorithmType();
	}

	int SpatialFilter::getOrder()
	{
		return gauss->getOrder();
	}

	double SpatialFilter::getSigma()
	{
		return gauss->getSigma();
	}

	Size SpatialFilter::getSize()
	{
		return gauss->getSize();
	}

	int SpatialFilter::getRadius()
	{
		return gauss->getRadius();
	}

	void SpatialFilter::setFixRadius(const int r)
	{
		gauss->setFixRadius(r);
	}

	void SpatialFilter::setIsInner(const int top, const int bottom, const int left, const int right)
	{
		gauss->setIsInner(top, bottom, left, right);
		//#pragma omp critical
		//		cout << left << "," << right << "," << top << "," << bottom <<","<<getSize().width-left-right<< endl;
	}

	void SpatialFilter::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType)
	{
		gauss->filter(src, dst, sigma, order, borderType);
	}

#pragma endregion

#pragma region GaussianFilterTile
	void SpatialFilterTile::init(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const Size div, const SpatialKernel spatial_kernel)
	{
		this->thread_max = omp_get_max_threads();
		this->depth = dest_depth;
		this->div = div;

		if (div.area() == 1)
		{
			gauss.resize(1);
			gauss[0] = createSpatialFilter(gf_method, dest_depth, spatial_kernel);
		}
		else
		{
			srcTile.resize(div.area());
			dstTile.resize(div.area());
			gauss.resize(thread_max);
#pragma omp parallel for
			for (int i = 0; i < thread_max; i++)
			{
				gauss[i] = createSpatialFilter(gf_method, dest_depth, spatial_kernel);
			}
		}
	}

	SpatialFilterTile::SpatialFilterTile(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const Size div, const SpatialKernel spatial_kernel)
	{
		init(gf_method, dest_depth, div, spatial_kernel);
	}

	void SpatialFilterTile::filter(const cv::Mat& src, cv::Mat& dst, const double sigma, const int order, const int borderType, const float truncateBoundary)
	{
		dst.create(src.size(), (depth < 0) ? src.depth() : depth);
		int vecsize = (dst.depth() == CV_64F) ? 4 : 8;

		TileDivision tdiv(src.size(), div);
		tdiv.compute(vecsize, vecsize);

		if (div.area() == 1)
		{
			gauss[0]->filter(src, dst, sigma, order, borderType);
			tileSize = src.size();
		}
		else
		{
			const int trad = gauss[0]->getRadius(sigma, order);
			int tileBoundary = (int)ceil(truncateBoundary * trad);
			//cout << trad << "," << tileBoundary <<","<<gauss[0]->getRadius()<< endl;

			if (0 < tileBoundary && tileBoundary < vecsize)
			{
				tileBoundary = vecsize;
			}

			bool isCreateSubImage = true;
			switch (gauss[0]->getAlgorithmType())
			{
			case SpatialFilterAlgorithm::SlidingDCT1_AVX:
			case SpatialFilterAlgorithm::SlidingDCT1_CONV:
			case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_VXY:
			case SpatialFilterAlgorithm::SlidingDCT3_DEBUG:
			case SpatialFilterAlgorithm::SlidingDCT3_CONV:
			case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_VXY:
			case SpatialFilterAlgorithm::SlidingDCT5_DEBUG:
			case SpatialFilterAlgorithm::SlidingDCT5_CONV:
			case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:
			case SpatialFilterAlgorithm::SlidingDCT7_AVX:
			case SpatialFilterAlgorithm::SlidingDCT7_VXY:
			case SpatialFilterAlgorithm::SlidingDCT7_CONV:
			case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:
				isCreateSubImage = false; break;
			default:
				isCreateSubImage = true; break;
			}

#pragma omp parallel for schedule (static)
			for (int n = 0; n < div.area(); n++)
			{
				if (isCreateSubImage)
				{
#if 1
					const int threadNumber = omp_get_thread_num();
					Rect roi = tdiv.getROI(n);
					cp::cropTileAlign(src, srcTile[n], roi, tileBoundary, borderType, vecsize, vecsize, 1);

					const int top = tileBoundary;
					const int bottom = srcTile[n].rows - top - roi.height;
					const int left = tileBoundary;
					const int right = srcTile[n].cols - left - roi.width;

					gauss[threadNumber]->setIsInner(top, bottom, left, right);
					if (src.depth() == dst.depth())
					{
						gauss[threadNumber]->filter(srcTile[n], srcTile[n], sigma, order, borderType);
						if (srcTile[n].depth() != dst.depth())
						{
							print_matinfo(src);
							print_matinfo(srcTile[n]);
							print_matinfo(dst);
						}
						cp::pasteTile(srcTile[n], dst, roi, tileBoundary);
					}
					else
					{
						gauss[threadNumber]->filter(srcTile[n], dstTile[n], sigma, order, borderType);
						if (dstTile[n].depth() != dst.depth())
						{
							print_matinfo(dstTile[n]);
							print_matinfo(dst);
						}
						cp::pasteTile(dstTile[n], dst, roi, tileBoundary);
					}
#else

					const Point idx = Point(n % div.width, n / div.width);
					const int threadNumber = omp_get_thread_num();
					cp::cropTileAlign(src, srcTile[n], div, idx, tileBoundary, borderType, vecsize, vecsize, 1);

					const int top = tileBoundary;
					const int bottom = srcTile[n].rows - top - src.rows / div.height;
					const int left = tileBoundary;
					const int right = srcTile[n].cols - left - src.cols / div.width;

					gauss[threadNumber]->setIsInner(top, bottom, left, right);
					if (src.depth() == dst.depth())
					{
						gauss[threadNumber]->filter(srcTile[n], srcTile[n], sigma, order, borderType);
						if (srcTile[n].depth() != dst.depth())
						{
							print_matinfo(src);
							print_matinfo(srcTile[n]);
							print_matinfo(dst);
						}
						cp::pasteTile(srcTile[n], dst, div, idx, tileBoundary);
					}
					else
					{
						gauss[threadNumber]->filter(srcTile[n], dstTile[n], sigma, order, borderType);
						if (dstTile[n].depth() != dst.depth())
						{
							print_matinfo(dstTile[n]);
							print_matinfo(dst);
						}
						cp::pasteTile(dstTile[n], dst, div, idx, tileBoundary);
					}
#endif
				}
				else
				{
					const int threadNumber = omp_get_thread_num();
					Rect roi = tdiv.getROI(n);
					const int top = roi.y;
					const int bottom = roi.y + roi.height;
					const int left = roi.x;
					const int right = roi.x + roi.width;
					//cout << n << endl;
					//cout << top << "," << bottom << "," << left << "," << right << endl;
					gauss[threadNumber]->setIsInner(top, src.rows - bottom, left, src.cols - right);
					//dst.setTo(0);
	//#pragma omp critical
					gauss[threadNumber]->filter(src, dst, sigma, order, borderType);
					//cp::imshowScale("a", dst); waitKey(0);		
				}
			}
			tileSize = srcTile[0].size();
		}
		//Mat show;
		//tdiv.draw(dst, show);
		//imshow("tile", show);
	}

	cv::Size SpatialFilterTile::getTileSize()
	{
		return tileSize;
	}


	class PointOperator
	{
	private:
		bool isContinuous = true;
		int top = 0;
		int bottom = 0;
		int left = 0;
		int right = 0;

		void subtract_32F_8(Mat& src1, Mat& src2, Mat& dest)
		{
			int yst = top;
			int yend = src1.rows - bottom;
			int xst = left;
			int x_simd_end = get_simd_floor(src1.cols - left - right, 8) + left;
			int xend = src1.cols - right;
			for (int j = yst; j < yend; j++)
			{
				float* s1 = src1.ptr<float>(j, xst);
				float* s2 = src2.ptr<float>(j, xst);
				float* d = dest.ptr<float>(j, xst);
				for (int i = xst; i < x_simd_end; i += 8)
				{
					_mm256_storeu_ps(d, _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s2)));
					s1 += 8;
					s2 += 8;
					d += 8;
				}
				for (int i = x_simd_end; i < xend; i++)
				{
					*d = *s1 - *s2;
					s1++;
					s2++;
					d++;
				}
			}
		}

		void subtract_32F_16(Mat& src1, Mat& src2, Mat& dest)
		{
			if (isContinuous)
			{
				int simd_end = get_simd_floor((int)src1.total(), 16);
				int end = (int)src1.total();

				float* s1 = src1.ptr<float>();
				float* s2 = src2.ptr<float>();
				float* d = dest.ptr<float>();
				for (int i = 0; i < simd_end; i += 16)
				{
					_mm256_store_ps(d, _mm256_sub_ps(_mm256_load_ps(s1), _mm256_load_ps(s2)));
					_mm256_store_ps(d + 8, _mm256_sub_ps(_mm256_load_ps(s1 + 8), _mm256_load_ps(s2 + 8)));
					s1 += 16;
					s2 += 16;
					d += 16;
				}
				for (int i = simd_end; i < end; i++)
				{
					*d = *s1 - *s2;
					s1++;
					s2++;
					d++;
				}
			}
			else
			{
				int yst = top;
				int yend = src1.rows - bottom;
				int xst = left;
				int x_simd_end = get_simd_floor(src1.cols - left - right, 16) + left;
				int xend = src1.cols - right;
				for (int j = yst; j < yend; j++)
				{
					float* s1 = src1.ptr<float>(j, xst);
					float* s2 = src2.ptr<float>(j, xst);
					float* d = dest.ptr<float>(j, xst);
					for (int i = xst; i < x_simd_end; i += 16)
					{
						_mm256_storeu_ps(d, _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s2)));
						_mm256_storeu_ps(d + 8, _mm256_sub_ps(_mm256_loadu_ps(s1 + 8), _mm256_loadu_ps(s2 + 8)));
						s1 += 16;
						s2 += 16;
						d += 16;
					}
					for (int i = x_simd_end; i < xend; i++)
					{
						*d = *s1 - *s2;
						s1++;
						s2++;
						d++;
					}
				}
			}
		}

		void subtract_32F_32(Mat& src1, Mat& src2, Mat& dest)
		{
			if (isContinuous)
			{
				int simd_end = get_simd_floor((int)src1.total(), 32);
				int end = (int)src1.total();

				float* s1 = src1.ptr<float>();
				float* s2 = src2.ptr<float>();
				float* d = dest.ptr<float>();
				for (int i = 0; i < simd_end; i += 32)
				{
					_mm256_store_ps(d, _mm256_sub_ps(_mm256_load_ps(s1), _mm256_load_ps(s2)));
					_mm256_store_ps(d + 8, _mm256_sub_ps(_mm256_load_ps(s1 + 8), _mm256_load_ps(s2 + 8)));
					_mm256_store_ps(d + 16, _mm256_sub_ps(_mm256_load_ps(s1 + 16), _mm256_load_ps(s2 + 16)));
					_mm256_store_ps(d + 24, _mm256_sub_ps(_mm256_load_ps(s1 + 24), _mm256_load_ps(s2 + 24)));
					s1 += 32;
					s2 += 32;
					d += 32;
				}
				for (int i = simd_end; i < end; i++)
				{
					*d = *s1 - *s2;
					s1++;
					s2++;
					d++;
				}
			}
			else
			{
				int yst = top;
				int yend = src1.rows - bottom;
				int xst = left;
				int x_simd_end = get_simd_floor(src1.cols - left - right, 32) + left;
				int xend = src1.cols - right;
				for (int j = yst; j < yend; j++)
				{
					float* s1 = src1.ptr<float>(j, xst);
					float* s2 = src2.ptr<float>(j, xst);
					float* d = dest.ptr<float>(j, xst);
					for (int i = xst; i < x_simd_end; i += 32)
					{
						_mm256_storeu_ps(d, _mm256_sub_ps(_mm256_loadu_ps(s1), _mm256_loadu_ps(s2)));
						_mm256_storeu_ps(d + 8, _mm256_sub_ps(_mm256_loadu_ps(s1 + 8), _mm256_loadu_ps(s2 + 8)));
						_mm256_store_ps(d + 16, _mm256_sub_ps(_mm256_load_ps(s1 + 16), _mm256_load_ps(s2 + 16)));
						_mm256_store_ps(d + 24, _mm256_sub_ps(_mm256_load_ps(s1 + 24), _mm256_load_ps(s2 + 24)));
						s1 += 32;
						s2 += 32;
						d += 32;
					}
					for (int i = x_simd_end; i < xend; i++)
					{
						*d = *s1 - *s2;
						s1++;
						s2++;
						d++;
					}
				}
			}
		}

		void subtract_8U_8(Mat& src1, Mat& src2, Mat& dest)
		{
			int yst = top;
			int yend = src1.rows - bottom;
			int xst = left;
			int x_simd_end = get_simd_floor(src1.cols - left - right, 8) + left;
			int xend = src1.cols - right;
			for (int j = yst; j < yend; j++)
			{
				uchar* s1 = src1.ptr<uchar>(j, xst);
				uchar* s2 = src2.ptr<uchar>(j, xst);
				uchar* d = dest.ptr <uchar>(j, xst);
				for (int i = xst; i < x_simd_end; i += 8)
				{

					_mm_storel_epi64((__m128i*)d, _mm_subs_epu8(_mm_loadl_epi64((__m128i*) s1), _mm_loadl_epi64((__m128i*) s2)));
					s1 += 8;
					s2 += 8;
					d += 8;
				}
				for (int i = x_simd_end; i < xend; i++)
				{
					*d = *s1 - *s2;
					s1++;
					s2++;
					d++;
				}
			}
		}

		void subtract_8U_16(Mat& src1, Mat& src2, Mat& dest)
		{
			int yst = top;
			int yend = src1.rows - bottom;
			int xst = left;
			int x_simd_end = get_simd_floor(src1.cols - left - right, 16) + left;
			int xend = src1.cols - right;
			for (int j = yst; j < yend; j++)
			{
				uchar* s1 = src1.ptr<uchar>(j, xst);
				uchar* s2 = src2.ptr<uchar>(j, xst);
				uchar* d = dest.ptr <uchar>(j, xst);
				for (int i = xst; i < x_simd_end; i += 16)
				{
					_mm_storeu_si128((__m128i*)d, _mm_subs_epu8(_mm_loadu_si128((__m128i*) s1), _mm_loadu_si128((__m128i*) s2)));
					s1 += 16;
					s2 += 16;
					d += 16;
				}
				for (int i = x_simd_end; i < xend; i++)
				{
					*d = *s1 - *s2;
					s1++;
					s2++;
					d++;
				}
			}
		}

		void subtractOffset_8U_16(Mat& src1, Mat& src2, const uchar offset, Mat& dest)
		{
			const __m128i moff = _mm_set1_epi8(offset);
			if (isContinuous)
			{
				int simd_end = get_simd_floor((int)src1.total(), 16);
				int end = (int)src1.total();
				uchar* s1 = src1.ptr<uchar>();
				uchar* s2 = src2.ptr<uchar>();
				uchar* d = dest.ptr <uchar>();
				for (int i = 0; i < simd_end; i += 16)
				{
					const __m128i ms1 = _mm_loadu_si128((__m128i*)s1);
					const __m128i ms2 = _mm_loadu_si128((__m128i*)s2);
					__m128i md = _mm_adds_epu8(moff, _mm_subs_epu8(ms1, ms2));
					md = _mm_subs_epu8(md, _mm_subs_epu8(ms2, ms1));
					_mm_storeu_si128((__m128i*)d, md);

					s1 += 16;
					s2 += 16;
					d += 16;
				}
				for (int i = simd_end; i < end; i++)
				{
					*d = saturate_cast<uchar>((*s1 - *s2 + offset));
					s1++;
					s2++;
					d++;
				}
			}
			else
			{
				int yst = top;
				int yend = src1.rows - bottom;
				int xst = left;
				int x_simd_end = get_simd_floor(src1.cols - left - right, 16) + left;
				int xend = src1.cols - right;

				for (int j = yst; j < yend; j++)
				{
					uchar* s1 = src1.ptr<uchar>(j, xst);
					uchar* s2 = src2.ptr<uchar>(j, xst);
					uchar* d = dest.ptr <uchar>(j, xst);
					for (int i = xst; i < x_simd_end; i += 16)
					{
						const __m128i ms1 = _mm_loadu_si128((__m128i*)s1);
						const __m128i ms2 = _mm_loadu_si128((__m128i*)s2);
						__m128i md = _mm_adds_epu8(moff, _mm_subs_epu8(ms1, ms2));
						md = _mm_subs_epu8(md, _mm_subs_epu8(ms2, ms1));
						_mm_storeu_si128((__m128i*)d, md);

						s1 += 16;
						s2 += 16;
						d += 16;
					}
					for (int i = x_simd_end; i < xend; i++)
					{
						*d = saturate_cast<uchar>((*s1 - *s2 + offset));
						s1++;
						s2++;
						d++;
					}
				}
			}
		}

		void absdiff_8U_16(Mat& src1, Mat& src2, Mat& dest)
		{
			const int yst = top;
			const int yend = src1.rows - bottom;
			const int xst = left;
			const int x_simd_end = get_simd_floor(src1.cols - left - right, 16) + left;
			const int xend = src1.cols - right;
			const bool is_8unroll = (xend - x_simd_end >= 8) ? true : false;
			const int x_simd_end2 = (is_8unroll) ? x_simd_end + 8 : x_simd_end;
			for (int j = yst; j < yend; j++)
			{
				uchar* s1 = src1.ptr<uchar>(j, xst);
				uchar* s2 = src2.ptr<uchar>(j, xst);
				uchar* d = dest.ptr <uchar>(j, xst);
				for (int i = xst; i < x_simd_end; i += 16)
				{
					__m128i ms1 = _mm_loadu_si128((__m128i*) s1);
					__m128i ms2 = _mm_loadu_si128((__m128i*) s2);
					_mm_storeu_si128((__m128i*)d, _mm_max_epu8(_mm_subs_epu8(ms1, ms2), _mm_subs_epu8(ms2, ms1)));
					s1 += 16;
					s2 += 16;
					d += 16;
				}
				if (is_8unroll)
				{
					__m128i ms1 = _mm_loadl_epi64((__m128i*) s1);
					__m128i ms2 = _mm_loadl_epi64((__m128i*) s2);
					_mm_storel_epi64((__m128i*)d, _mm_max_epu8(_mm_subs_epu8(ms1, ms2), _mm_subs_epu8(ms2, ms1)));
					s1 += 8;
					s2 += 8;
					d += 8;
				}
				for (int i = x_simd_end2; i < xend; i++)
				{
					*d = abs(*s1 - *s2);
					s1++;
					s2++;
					d++;
				}
			}
		}
	public:
		void setIsInner(const int top, const int bottom, const int left, const int right)
		{
			this->top = top;
			this->bottom = bottom;
			this->left = left;
			this->right = right;

			if (top != 0 || bottom != 0 || left != 0 || right != 0) isContinuous = false;
		}

		void setROI(const cv::Rect rect)
		{
			top = rect.y;
			bottom = rect.height + rect.y;
			left = rect.x;
			right = rect.width + rect.x;

			if (top != 0 || bottom != 0 || left != 0 || right != 0) isContinuous = false;
		}

		void subtract(Mat& src1, Mat& src2, Mat& dest)
		{
			CV_Assert(src1.channels() == 1);
			CV_Assert(src1.depth() == src2.depth());

			if (src1.depth() == CV_8U)
			{
				dest.create(src1.size(), CV_8U);
				//subtract_8U_8(src1, src2, dest);
				absdiff_8U_16(src1, src2, dest);
			}
			else if (src1.depth() == CV_32F)
			{
				dest.create(src1.size(), CV_32F);
				//subtract_32F_8(src1, src2, dest);
				subtract_32F_16(src1, src2, dest);
				//subtract_32F_32(src1, src2, dest);
			}
		}

		void subtractOffset(Mat& src1, Mat& src2, const double offset, Mat& dest)
		{
			CV_Assert(src1.channels() == 1);
			CV_Assert(src1.depth() == src2.depth());

			if (src1.depth() == CV_8U)
			{
				dest.create(src1.size(), CV_8U);
				subtractOffset_8U_16(src1, src2, (char)offset, dest);
			}
			else if (src1.depth() == CV_32F)
			{
				dest.create(src1.size(), CV_32F);
				//subtract_32F_16(src1, src2, dest);
			}
		}
	};

	void SpatialFilterDoGTile::init(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const Size div, const SpatialKernel spatial_kernel)
	{
		this->thread_max = omp_get_max_threads();
		this->depth = dest_depth;
		this->div = div;

		if (div.area() == 1)
		{
			gauss.resize(1);
			gauss[0] = createSpatialFilter(gf_method, dest_depth, spatial_kernel);
			gauss2.resize(1);
			gauss2[0] = createSpatialFilter(gf_method, dest_depth, spatial_kernel);
		}
		else
		{
			srcTile.resize(div.area());
			srcTile2.resize(div.area());
			dstTile.resize(div.area());
			gauss.resize(thread_max);
			gauss2.resize(thread_max);

			for (int i = 0; i < thread_max; i++)
			{
				gauss[i] = createSpatialFilter(gf_method, dest_depth, spatial_kernel);
				gauss2[i] = createSpatialFilter(gf_method, dest_depth, spatial_kernel);
			}
		}
	}

	SpatialFilterDoGTile::SpatialFilterDoGTile(const cp::SpatialFilterAlgorithm gf_method, const int dest_depth, const Size div, const SpatialKernel spatial_kernel)
	{
		init(gf_method, dest_depth, div, spatial_kernel);
	}

	cv::Size SpatialFilterDoGTile::getTileSize()
	{
		return tileSize;
	}

	void SpatialFilterDoGTile::filter(const cv::Mat& src, cv::Mat& dst, const double sigma1, const double sigma2, const int order, const int borderType, const float truncateBoundary)
	{
		dst.create(src.size(), (depth < 0) ? src.depth() : depth);
		buff.create(src.size(), (depth < 0) ? src.depth() : depth);

		int vecsize = (dst.depth() == CV_64F) ? 4 : 8;

		TileDivision tdiv(src.size(), div);
		tdiv.compute(vecsize, vecsize);

		if (div.area() == 1)
		{
			gauss[0]->filter(src, dst, sigma1, order, borderType);
			gauss2[0]->filter(src, buff, sigma1, order, borderType);
			subtract(buff, dst, dst);
			tileSize = src.size();
		}
		else
		{
			const int trad = gauss[0]->getRadius(sigma1, order);
			int tileBoundary = (int)ceil(truncateBoundary * trad);
			//cout << trad << "," << tileBoundary <<","<<gauss[0]->getRadius()<< endl;

			if (0 < tileBoundary && tileBoundary < vecsize)
			{
				tileBoundary = vecsize;
			}

			bool isCreateSubImage = true;
			switch (gauss[0]->getAlgorithmType())
			{
			case SpatialFilterAlgorithm::SlidingDCT1_AVX:
			case SpatialFilterAlgorithm::SlidingDCT1_CONV:
			case SpatialFilterAlgorithm::SlidingDCT1_64_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_16_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_AVX:
			case SpatialFilterAlgorithm::SlidingDCT3_VXY:
			case SpatialFilterAlgorithm::SlidingDCT3_DEBUG:
			case SpatialFilterAlgorithm::SlidingDCT3_CONV:
			case SpatialFilterAlgorithm::SlidingDCT3_64_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_16_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_AVX:
			case SpatialFilterAlgorithm::SlidingDCT5_VXY:
			case SpatialFilterAlgorithm::SlidingDCT5_DEBUG:
			case SpatialFilterAlgorithm::SlidingDCT5_CONV:
			case SpatialFilterAlgorithm::SlidingDCT5_64_AVX:
			case SpatialFilterAlgorithm::SlidingDCT7_AVX:
			case SpatialFilterAlgorithm::SlidingDCT7_VXY:
			case SpatialFilterAlgorithm::SlidingDCT7_CONV:
			case SpatialFilterAlgorithm::SlidingDCT7_64_AVX:
				isCreateSubImage = false; break;
			default:
				isCreateSubImage = true; break;
			}

#pragma omp parallel for schedule (static)
			for (int n = 0; n < div.area(); n++)
			{
				if (isCreateSubImage)
				{
					const int threadNumber = omp_get_thread_num();
					Rect roi = tdiv.getROI(n);
					cp::cropTileAlign(src, srcTile[n], roi, tileBoundary, borderType, vecsize, vecsize, 1);

					const int top = tileBoundary;
					const int bottom = tileBoundary;
					const int left = tileBoundary;
					const int right = tileBoundary;

					gauss[threadNumber]->setIsInner(top, bottom, left, right);
					gauss2[threadNumber]->setIsInner(top, bottom, left, right);
					if (src.depth() == dst.depth())
					{
						gauss2[threadNumber]->filter(srcTile[n], srcTile2[n], sigma2, order, borderType);
						gauss[threadNumber]->filter(srcTile[n], srcTile[n], sigma1, order, borderType);
						PointOperator po;
						if (srcTile[n].depth() == CV_8U)
						{
							po.subtractOffset(srcTile2[n], srcTile[n], 128, srcTile[n]);
						}
						else
						{
							//subtract(srcTile2[n], srcTile[n], srcTile[n]);
							po.subtract(srcTile2[n], srcTile[n], srcTile[n]);
						}
						if (srcTile[n].depth() != dst.depth())
						{
							print_matinfo(src);
							print_matinfo(srcTile[n]);
							print_matinfo(dst);
						}
						cp::pasteTile(srcTile[n], dst, roi, tileBoundary);
					}
					else
					{
						gauss[threadNumber]->filter(srcTile[n], srcTile2[n], sigma1, order, borderType);
						gauss2[threadNumber]->filter(srcTile[n], dstTile[n], sigma2, order, borderType);
						PointOperator po;
						if (srcTile[n].depth() == CV_8U)
						{
							po.subtractOffset(dstTile[n], srcTile2[n], 128, dstTile[n]);
						}
						else
						{
							po.subtract(dstTile[n], srcTile2[n], dstTile[n]);
						}
						if (dstTile[n].depth() != dst.depth())
						{
							print_matinfo(dstTile[n]);
							print_matinfo(dst);
						}
						cp::pasteTile(dstTile[n], dst, roi, tileBoundary);
					}
				}
				else
				{
					const int threadNumber = omp_get_thread_num();
					Rect roi = tdiv.getROI(n);
					const int top = roi.y;
					const int bottom = roi.y + roi.height;
					const int left = roi.x;
					const int right = roi.x + roi.width;
					//cout << n << endl;
					//cout << top << "," << bottom << "," << left << "," << right << endl;
					gauss[threadNumber]->setIsInner(top, src.rows - bottom, left, src.cols - right);
					gauss2[threadNumber]->setIsInner(top, src.rows - bottom, left, src.cols - right);
					PointOperator po;
					po.setIsInner(top, src.rows - bottom, left, src.cols - right);

					gauss[threadNumber]->filter(src, dst, sigma1, order, borderType);
					gauss2[threadNumber]->filter(src, buff, sigma2, order, borderType);
					
					if (dst.depth() == CV_8U)
					{
						po.subtractOffset(buff, dst, 128,dst);
					}
					else
					{
						po.subtract(buff, dst, dst);
					}
					//cp::imshowScale("a", dst); waitKey(0);		
				}
			}
			tileSize = srcTile[0].size();
		}
		//Mat show;
		//tdiv.draw(dst, show);
		//imshow("tile", show);
	}
#pragma endregion
}
