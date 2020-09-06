#include "guidedFilter.hpp"

#include "guidedFilter_Naive.h"
#include "guidedFilter_Naive_Share.h"

#include "guidedFilter_SepVHI.h"
#include "guidedFilter_SepVHI_Share.h"

#include "guidedFilter_Merge.h"
#include "guidedFilter_Merge_Share.h"
#include "guidedFilter_Merge_Share_Transpose.h"
#include "guidedFilter_Merge_Share_Ex.h"
#include "guidedFilter_Merge_Transpose.h"
#include "guidedFilter_Merge_Transpose_Inverse.h"
#include "guidedFilter_Merge_Share_Transpose_Inverse.h"
#include "guidedFilter_Merge_nonSplit.h"

#include "guidedFilter_Merge_OnePass.h"
#include "guidedFilter_Merge_OnePass_2Div.h"

#include "guidedFilter_Merge_OnePath_Fast.h"

#include <opencv2/ximgproc.hpp>

namespace cp
{
	//base class

	cv::Size GuidedFilterBase::size()
	{
		return src.size();
	}

	int GuidedFilterBase::src_channels()
	{
		return src.channels();
	}

	int GuidedFilterBase::guide_channels()
	{
		return guide.channels();
	}

	void GuidedFilterBase::setUpsampleMethod(const int method)
	{
		upsample_method = method;
	}

	void GuidedFilterBase::setDownsampleMethod(const int method)
	{
		downsample_method = method;
	}

	int GuidedFilterBase::getImplementation()
	{
		return implementation;
	}

	void GuidedFilterBase::filter(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps)
	{
		src = _src;
		guide = _guide;
		dest = _dest;
		r = _r;
		eps = _eps;
		/*src = _src.clone();
		guide = _guide.clone();
		dest = _dest.clone();*/

		filter();
	}

	void GuidedFilterBase::filterVector(std::vector<cv::Mat>& _src, std::vector <cv::Mat>& _guide, std::vector <cv::Mat>& _dest, int _r, float _eps)
	{
		vsrc = _src;
		vguide = _guide;
		vdest = _dest;
		r = _r;
		eps = _eps;
		/*src = _src.clone();
		guide = _guide.clone();
		dest = _dest.clone();*/

		filterVector();
	}

	void GuidedFilterBase::filterFast(const int ratio)
	{
		cv::resize(src, src_low, src.size() / ratio, 0, 0, downsample_method);
		cv::resize(guide, guide_low, guide.size() / ratio, 0, 0, downsample_method);

		r = cv::max(r / ratio, 1);

		upsample();
	}

	void GuidedFilterBase::filterFast(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, const int ratio)
	{
		src = _src;
		guide = _guide;
		dest = _dest;
		r = _r;
		eps = _eps;

		filterFast(ratio);
	}

	void GuidedFilterBase::computeVarCov()
	{
		;
	}

	void GuidedFilterBase::filterGuidePrecomputed()
	{
		printf("%s: this type of guide precomputed is not implemented\n", cp::getGuidedType(implementation).c_str());
		guide.copyTo(dest);
	}

	void GuidedFilterBase::filterGuidePrecomputed(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps)
	{
		bool isUpdate = false;
		if (guide.empty())
		{
			cv::Scalar v;
			if (_guide.type() == CV_32FC1)
			{
				v = cv::Scalar(_guide.at<float>(0, 0));
			}
			else if (_guide.type() == CV_32FC3)
			{
				v = cv::Scalar(_guide.at<float>(0, 0), _guide.at<float>(0, 1), _guide.at<float>(0, 2));
			}
			if (_guide.type() == CV_8UC1)
			{
				v = cv::Scalar(_guide.at<uchar>(0, 0));
			}
			else if (_guide.type() == CV_8UC3)
			{
				v = cv::Scalar(_guide.at<uchar>(0, 0), _guide.at<uchar>(0, 1), _guide.at<uchar>(0, 2));
			}
			guide_samples.push_back(v);
			isUpdate = true;
		}
		else
		{
			cv::Scalar v;
			if (_guide.type() == CV_32FC1)
			{
				v = cv::Scalar(_guide.at<float>(0, 0));
			}
			else if (_guide.type() == CV_32FC3)
			{
				v = cv::Scalar(_guide.at<float>(0, 0), _guide.at<float>(0, 1), _guide.at<float>(0, 2));
			}
			if (_guide.type() == CV_8UC1)
			{
				v = cv::Scalar(_guide.at<uchar>(0, 0));
			}
			else if (_guide.type() == CV_8UC3)
			{
				v = cv::Scalar(_guide.at<uchar>(0, 0), _guide.at<uchar>(0, 1), _guide.at<uchar>(0, 2));
			}

			if (guide_samples.size() == 0)
			{
				guide_samples.push_back(v);
				isUpdate = true;
			}
			else if (v == guide_samples[0])
			{
				if (r != _r || eps != _eps)isUpdate = true;
				else isUpdate = false;
			}
			else
			{
				isUpdate = true;
				guide_samples[0] = v;
			}
		}

		src = _src;
		guide = _guide;
		dest = _dest;
		r = _r;
		eps = _eps;
		if (isUpdate)computeVarCov();
		filterGuidePrecomputed();
	}

	void GuidedFilterBase::upsample()
	{
		printf("%s: this type of upsample is not implemented\n", cp::getGuidedType(implementation).c_str());
		guide.copyTo(dest);
	}

	void GuidedFilterBase::upsample(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps)
	{
		src = _src;
		guide = _guide;
		dest = _dest;
		r = _r;
		eps = _eps;

		int ratio = _guide.cols / _src.cols;
		cv::resize(_guide, guide_low, _guide.size() / ratio, 0, 0, downsample_method);

		upsample();
	}

	void GuidedFilterBase::upsample(cv::Mat& _src, cv::Mat& _guide_low, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps)
	{
		src = _src;
		guide = _guide;
		dest = _dest;
		r = _r;
		eps = _eps;

		int ratio = _guide.cols / _src.cols;
		guide_low = _guide_low;

		upsample();
	}

	//end class GuideFilterBase
	//////////////////////////////////////////////////////////////////////



	void guidedImageFilter(cv::InputArray src_, cv::InputArray guide_, cv::OutputArray dest, const int r, const float eps, const GuidedTypes guidedType, const BoxTypes boxType, const ParallelTypes parallelType)
	{
		cv::Mat src = src_.getMat();
		cv::Mat guide = guide_.getMat();

		GuidedImageFilter gf;
		gf.setBoxType(boxType);
		gf.filter(src, guide, dest, r, eps, guidedType, parallelType);
	}

	//for ximg proc function
	class guidedFilter_ximgproc : public GuidedFilterBase
	{
	public:
		guidedFilter_ximgproc(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps)
			: GuidedFilterBase(_src, _guide, _dest, _r, _eps)
		{
			implementation = GUIDED_XIMGPROC;
		}

		void filter()
		{
			cv::ximgproc::guidedFilter(guide, src, dest, r, eps);
		}

		void filterVector()
		{
			cv::merge(vsrc, src);
			cv::merge(vguide, guide);
			cv::ximgproc::guidedFilter(guide, src, dest, r, eps);
			split(dest, vdest);
		}
	};

	void GuidedImageFilter::setDownsampleMethod(const int method)
	{
		downsample_method = method;
		if (!gf[0].empty())
		{
			gf[0]->setDownsampleMethod(method);
		}
		if (!gf[1].empty())
		{
			gf[1]->setDownsampleMethod(method);
		}
		if (!gf[2].empty())
		{
			gf[2]->setDownsampleMethod(method);
		}

	}

	void GuidedImageFilter::setUpsampleMethod(const int method)
	{
		upsample_method = method;
		if (!gf[0].empty())
		{
			gf[0]->setUpsampleMethod(method);
		}
		if (!gf[1].empty())
		{
			gf[1]->setUpsampleMethod(method);
		}
		if (!gf[2].empty())
		{
			gf[2]->setUpsampleMethod(method);
		}
	}

	void GuidedImageFilter::setBoxType(const int type)
	{
		box_type = type;
	}

	cv::Ptr<GuidedFilterBase> GuidedImageFilter::getGuidedFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps, const int guided_type)
	{
		//GuidedFilterBase* ret;
		cv::Ptr<GuidedFilterBase> ret;

		switch (guided_type)
		{
		case GUIDED_XIMGPROC:
			ret = new guidedFilter_ximgproc(src, guide, dest, r, eps); break;
		case GUIDED_NAIVE:
		default:
			ret = new guidedFilter_Naive(src, guide, dest, r, eps, box_type, parallel_type); break;
		case GUIDED_NAIVE_SHARE:
			ret = new guidedFilter_Naive_Share(src, guide, dest, r, eps, box_type, parallel_type); break;
		case GUIDED_NAIVE_ONEPASS:
			ret = new guidedFilter_Naive_OnePass(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_SEP_VHI:
			ret = new guidedFilter_SepVHI(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_SEP_VHI_SHARE:
			ret = new guidedFilter_SepVHI_Share(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE:
			ret = new guidedImageFilter_Merge_Base(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SSE:
			ret = new guidedFilter_Merge_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_AVX:
			ret = new guidedFilter_Merge_AVX(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_TRANSPOSE:
			ret = new guidedFilter_Merge_Transpose_nonVec(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_TRANSPOSE_SSE:
			ret = new guidedFilter_Merge_Transpose_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_TRANSPOSE_AVX:
			ret = new guidedFilter_Merge_Transpose_AVX(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_TRANSPOSE_INVERSE:
			ret = new guidedFilter_Merge_Transpose_Inverse_nonVec(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_TRANSPOSE_INVERSE_SSE:
			ret = new guidedFilter_Merge_Transpose_Inverse_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_TRANSPOSE_INVERSE_AVX:
			ret = new guidedFilter_Merge_Transpose_Inverse_AVX(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE:
			ret = new guidedFilter_Merge_Share_Base(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_SSE:
			ret = new guidedFilter_Merge_Share_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_AVX:
			ret = new guidedFilter_Merge_Share_AVX(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_EX:
			ret = new guidedFilter_Merge_Share_Mixed_nonVec(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_EX_SSE:
			ret = new guidedFilter_Merge_Share_Mixed_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_EX_AVX:
			ret = new guidedFilter_Merge_Share_Mixed_AVX(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_TRANSPOSE:
			ret = new guidedFilter_Merge_Share_Transpose_nonVec(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_SSE:
			ret = new guidedFilter_Merge_Share_Transpose_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_AVX:
			ret = new guidedFilter_Merge_Share_Transpose_AVX(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE:
			ret = new guidedFilter_Merge_Share_Transpose_Inverse_nonVec(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE:
			ret = new guidedFilter_Merge_Share_Transpose_Inverse_SSE(src, guide, dest, r, eps, parallel_type); break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX:
			ret = new guidedFilter_Merge_Share_Transpose_Inverse_AVX(src, guide, dest, r, eps, parallel_type); break;

		case GUIDED_MERGE_ONEPASS:
			ret = new guidedFilter_Merge_OnePass(src, guide, dest, r, eps, parallel_type); break;
			//gf = new guidedFilter_Merge_OnePass_LoopFusion(src, guide, dest, r, eps, parallel_type); break;

		case GUIDED_MERGE_ONEPASS_SIMD:
			ret = new guidedFilter_Merge_OnePass_SIMD(src, guide, dest, r, eps, parallel_type); break;
		}

		ret->setUpsampleMethod(upsample_method);
		ret->setDownsampleMethod(downsample_method);
		return ret;
	}

	bool GuidedImageFilter::initialize(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest)
	{
		if (src.depth() == CV_32F || src.depth() == CV_64F)
		{
			srcImage = src;
		}
		else
		{
			src.convertTo(srcImage, CV_32F);
		}

		if (guide.depth() == CV_32F || guide.depth() == CV_64F)
		{
			guideImage = guide;
		}
		else
		{
			guide.convertTo(guideImage, CV_32F);
		}

		bool ret = false;
		if (dest.depth() == CV_32F || dest.depth() == CV_64F)
		{
			if (src.type() != dest.type() || src.channels() != dest.channels())
			{
				if (src.channels() == CV_64F)
					destImage.create(guide.size(), CV_MAKETYPE(CV_64F, src.channels()));
				else
					destImage.create(guide.size(), CV_MAKETYPE(CV_32F, src.channels()));

				ret = true;
			}
			else
			{
				destImage = dest.getMat();
			}
		}
		else
		{
			if (src.channels() == CV_64F)
				destImage.create(guide.size(), CV_MAKETYPE(CV_64F, src.channels()));
			else
				destImage.create(guide.size(), CV_MAKETYPE(CV_32F, src.channels()));

			ret = true;
		}

		return ret;
	}

	void GuidedImageFilter::filter(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool isdestinit = initialize(src, guide, dest);

		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != src.channels() ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;
			gf[0] = getGuidedFilter(srcImage, guideImage, destImage, r, eps, guided_type);
			gf[0]->filter();
		}
		else
		{
			gf[0]->filter(srcImage, guideImage, destImage, r, eps);
		}

		if (isdestinit) destImage.convertTo(dest, src.depth());
	}

	void GuidedImageFilter::filterGuidePrecomputed(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool isdestinit = initialize(src, guide, dest);

		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != src.channels() ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;
			gf[0] = getGuidedFilter(srcImage, guideImage, destImage, r, eps, guided_type);
			gf[0]->filterGuidePrecomputed(srcImage, guideImage, destImage, r, eps);
		}
		else
		{
			gf[0]->filterGuidePrecomputed(srcImage, guideImage, destImage, r, eps);
		}

		if (isdestinit) destImage.convertTo(dest, src.depth());
	}

	void GuidedImageFilter::filterColorParallel(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		if (src.channels() == 1)
		{
			filter(src, guide, dest, r, eps, guided_type, parallel_type_current);
			return;
		}

		bool init = false;

		if (gf[0].empty() || gf[1].empty() || gf[2].empty())
		{
			init = true;
		}
		else  if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != 1 ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != src.size() ||
			gf[1]->getImplementation() != guided_type ||
			gf[1]->src_channels() != 1 ||
			gf[1]->guide_channels() != guide.channels() ||
			gf[1]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;

			cv::split(src, vsrc);
			cv::split(dest, vdest);

#pragma omp parallel for
			for (int c = 0; c < 3; c++)
			{
				gf[c] = getGuidedFilter(vsrc[c], guide, vdest[c], r, eps, guided_type);
				gf[c]->filter();
			}

			merge(vdest, dest);
		}
		else
		{
			cv::split(src, vsrc);

#pragma omp parallel for
			for (int c = 0; c < 3; c++)
			{
				gf[c]->filter(vsrc[c], guide, vdest[c], r, eps);
			}
			merge(vdest, dest);
		}

	}


	void GuidedImageFilter::filterFast(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int ratio, const int guided_type, const int parallel_type_current)
	{
		bool isdestinit = initialize(src, guide, dest);

		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != src.channels() ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;

			gf[0] = getGuidedFilter(srcImage, guideImage, destImage, r, eps, guided_type);
			gf[0]->filterFast(ratio);
		}
		else
		{
			gf[0]->filterFast(srcImage, guideImage, destImage, r, eps, ratio);
		}

		if (isdestinit) destImage.convertTo(dest, src.depth());
	}

	void GuidedImageFilter::upsample(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool isdestinit = initialize(src, guide, dest);

		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != src.channels() ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;
			gf[0] = getGuidedFilter(srcImage, guideImage, destImage, r, eps, guided_type);
		}

		gf[0]->upsample(srcImage, guideImage, destImage, r, eps);

		if (isdestinit) destImage.convertTo(dest, src.depth());
	}

	void GuidedImageFilter::upsample(cv::Mat& src, cv::Mat& guide_low, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool isdestinit = initialize(src, guide, dest);

		if (guide_low.depth() == CV_32F || guide_low.depth() == CV_64F)
		{
			guidelowImage = guide_low;
		}
		else
		{
			guide_low.convertTo(guidelowImage, CV_32F);
		}

		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != src.channels() ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;
			gf[0] = getGuidedFilter(srcImage, guideImage, destImage, r, eps, guided_type);
		}


		gf[0]->upsample(srcImage, guidelowImage, guideImage, destImage, r, eps);

		if (isdestinit) destImage.convertTo(dest, src.depth());
	}

	void GuidedImageFilter::filter(cv::Mat& src, std::vector<cv::Mat>& vguide, cv::Mat& dest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != src.channels() ||
			gf[0]->guide_channels() != vguide.size() ||
			gf[0]->size() != src.size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;

			cv::Mat guide;
			merge(vguide, guide);

			gf[0] = getGuidedFilter(src, guide, dest, r, eps, guided_type);


			gf[0]->filter();
		}
		else
		{
			std::vector<cv::Mat> s(1);
			std::vector<cv::Mat> d(1);
			s[0] = src;
			d[0] = dest;
			gf[0]->filterVector(s, vguide, d, r, eps);
		}
	}

	void GuidedImageFilter::filter(std::vector<cv::Mat>& vsrc, cv::Mat& guide, std::vector<cv::Mat>& vdest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != vsrc.size() ||
			gf[0]->guide_channels() != guide.channels() ||
			gf[0]->size() != vsrc[0].size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;

			cv::Mat src, dest;
			merge(vsrc, src);
			merge(vdest, dest);
			gf[0] = getGuidedFilter(src, guide, dest, r, eps, guided_type);

			gf[0]->filter();
		}
		else
		{
			std::vector<cv::Mat> g(1);
			g[0] = guide;
			gf[0]->filterVector(vsrc, g, vdest, r, eps);
		}
	}

	void GuidedImageFilter::filter(std::vector<cv::Mat>& vsrc, std::vector<cv::Mat>& vguide, std::vector<cv::Mat>& vdest, const int r, const float eps, const int guided_type, const int parallel_type_current)
	{
		bool init = false;

		if (gf[0].empty())
		{
			init = true;
		}
		else if (
			gf[0]->getImplementation() != guided_type ||
			parallel_type != parallel_type_current ||
			gf[0]->src_channels() != vsrc.size() ||
			gf[0]->guide_channels() != vguide.size() ||
			gf[0]->size() != vsrc[0].size()
			)
		{
			init = true;
		}

		if (init)
		{
			parallel_type = parallel_type_current;

			cv::Mat src, dest, guide;
			merge(vsrc, src);
			merge(vdest, dest);
			merge(vguide, guide);

			gf[0] = getGuidedFilter(src, guide, dest, r, eps, guided_type);

			gf[0]->filter();
		}
		else
		{
			gf[0]->filterVector(vsrc, vguide, vdest, r, eps);
		}
	}
}


#include "guidedFilter.hpp"
#include "stencil.hpp"
#include "bitconvert.hpp"
#include "../filterengine.hpp"
using namespace std;
using namespace cv;

const int BORDER_TYPE = cv::BORDER_REPLICATE;
//static const int BORDER_TYPE = cv::BORDER_DEFAULT;


//#define FLOAT_BOX_FILTER_OFF 1
//#define STATICMAT static Mat
#define STATICMAT Mat

////////////////
#define CV_MALLOC_ALIGN 16

void scalarToRawData(const Scalar& s, void* _buf, int type, int unroll_to)
{
	int i, depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
	CV_Assert(cn <= 4);
	switch (depth)
	{
	case CV_8U:
	{
		uchar* buf = (uchar*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<uchar>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
	}
	break;
	case CV_8S:
	{
		schar* buf = (schar*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<schar>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
	}
	break;
	case CV_16U:
	{
		ushort* buf = (ushort*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<ushort>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
	}
	break;
	case CV_16S:
	{
		short* buf = (short*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<short>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
	}
	break;
	case CV_32S:
	{
		int* buf = (int*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<int>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
	}
	break;
	case CV_32F:
	{
		float* buf = (float*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<float>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
	}
	break;
	case CV_64F:
	{
		double* buf = (double*)_buf;
		for (i = 0; i < cn; i++)
			buf[i] = saturate_cast<double>(s.val[i]);
		for (; i < unroll_to; i++)
			buf[i] = buf[i - cn];
		break;
	}
	default:
		CV_Error(cv::Error::Code::StsUnsupportedFormat, "");
	}
}

BaseRowFilter::BaseRowFilter() { ksize = anchor = -1; }
BaseRowFilter::~BaseRowFilter() {}

BaseColumnFilter::BaseColumnFilter() { ksize = anchor = -1; }
BaseColumnFilter::~BaseColumnFilter() {}
void BaseColumnFilter::reset() {}

BaseFilter::BaseFilter() { ksize = Size(-1, -1); anchor = Point(-1, -1); }
BaseFilter::~BaseFilter() {}
void BaseFilter::reset() {}

FilterEngine::FilterEngine()
{
	srcType = dstType = bufType = -1;
	rowBorderType = columnBorderType = BORDER_REPLICATE;
	bufStep = startY = startY0 = endY = rowCount = dstY = 0;
	maxWidth = 0;

	wholeSize = Size(-1, -1);
}


FilterEngine::FilterEngine(const Ptr<BaseFilter>& _filter2D,
	const Ptr<BaseRowFilter>& _rowFilter,
	const Ptr<BaseColumnFilter>& _columnFilter,
	int _srcType, int _dstType, int _bufType,
	int _rowBorderType, int _columnBorderType,
	const Scalar& _borderValue)
{
	init(_filter2D, _rowFilter, _columnFilter, _srcType, _dstType, _bufType,
		_rowBorderType, _columnBorderType, _borderValue);
}

FilterEngine::~FilterEngine()
{
}


void FilterEngine::init(const Ptr<BaseFilter>& _filter2D,
	const Ptr<BaseRowFilter>& _rowFilter,
	const Ptr<BaseColumnFilter>& _columnFilter,
	int _srcType, int _dstType, int _bufType,
	int _rowBorderType, int _columnBorderType,
	const Scalar& _borderValue)
{
	_srcType = CV_MAT_TYPE(_srcType);
	_bufType = CV_MAT_TYPE(_bufType);
	_dstType = CV_MAT_TYPE(_dstType);

	srcType = _srcType;
	int srcElemSize = (int)getElemSize(srcType);
	dstType = _dstType;
	bufType = _bufType;

	filter2D = _filter2D;
	rowFilter = _rowFilter;
	columnFilter = _columnFilter;

	if (_columnBorderType < 0)
		_columnBorderType = _rowBorderType;

	rowBorderType = _rowBorderType;
	columnBorderType = _columnBorderType;

	CV_Assert(columnBorderType != BORDER_WRAP);

	if (isSeparable())
	{
		CV_Assert(rowFilter && columnFilter);
		ksize = Size(rowFilter->ksize, columnFilter->ksize);
		anchor = Point(rowFilter->anchor, columnFilter->anchor);
	}
	else
	{
		CV_Assert(bufType == srcType);
		ksize = filter2D->ksize;
		anchor = filter2D->anchor;
	}

	CV_Assert(0 <= anchor.x && anchor.x < ksize.width &&
		0 <= anchor.y && anchor.y < ksize.height);

	borderElemSize = srcElemSize / (CV_MAT_DEPTH(srcType) >= CV_32S ? sizeof(int) : 1);
	int borderLength = std::max(ksize.width - 1, 1);
	borderTab.resize(borderLength * borderElemSize);

	maxWidth = bufStep = 0;
	constBorderRow.clear();

	if (rowBorderType == BORDER_CONSTANT || columnBorderType == BORDER_CONSTANT)
	{
		constBorderValue.resize(srcElemSize * borderLength);
		int srcType1 = CV_MAKETYPE(CV_MAT_DEPTH(srcType), MIN(CV_MAT_CN(srcType), 4));
		scalarToRawData(_borderValue, &constBorderValue[0], srcType1,
			borderLength * CV_MAT_CN(srcType));
	}

	wholeSize = Size(-1, -1);
}

#define VEC_ALIGN CV_MALLOC_ALIGN

int FilterEngine::start(Size _wholeSize, Rect _roi, int _maxBufRows)
{
	int i, j;

	wholeSize = _wholeSize;
	roi = _roi;
	CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
		roi.x + roi.width <= wholeSize.width &&
		roi.y + roi.height <= wholeSize.height);

	int esz = (int)getElemSize(srcType);
	int bufElemSize = (int)getElemSize(bufType);
	const uchar* constVal = !constBorderValue.empty() ? &constBorderValue[0] : 0;

	if (_maxBufRows < 0)
		_maxBufRows = ksize.height + 3;
	_maxBufRows = std::max(_maxBufRows, std::max(anchor.y, ksize.height - anchor.y - 1) * 2 + 1);

	if (maxWidth < roi.width || _maxBufRows != (int)rows.size())
	{
		rows.resize(_maxBufRows);
		maxWidth = std::max(maxWidth, roi.width);
		int cn = CV_MAT_CN(srcType);
		srcRow.resize(esz * (maxWidth + ksize.width - 1));
		if (columnBorderType == BORDER_CONSTANT)
		{
			constBorderRow.resize(getElemSize(bufType) * (maxWidth + ksize.width - 1 + VEC_ALIGN));
			uchar* dst = alignPtr(&constBorderRow[0], VEC_ALIGN), * tdst;
			int n = (int)constBorderValue.size(), N;
			N = (maxWidth + ksize.width - 1) * esz;
			tdst = isSeparable() ? &srcRow[0] : dst;

			for (i = 0; i < N; i += n)
			{
				n = std::min(n, N - i);
				for (j = 0; j < n; j++)
					tdst[i + j] = constVal[j];
			}

			if (isSeparable())
				(*rowFilter)(&srcRow[0], dst, maxWidth, cn);
		}

		int maxBufStep = bufElemSize * (int)alignSize(maxWidth +
			(!isSeparable() ? ksize.width - 1 : 0), VEC_ALIGN);
		ringBuf.resize(maxBufStep * rows.size() + VEC_ALIGN);
	}

	// adjust bufstep so that the used part of the ring buffer stays compact in memory
	bufStep = bufElemSize * (int)alignSize(roi.width + (!isSeparable() ? ksize.width - 1 : 0), 16);

	dx1 = std::max(anchor.x - roi.x, 0);
	dx2 = std::max(ksize.width - anchor.x - 1 + roi.x + roi.width - wholeSize.width, 0);

	// recompute border tables
	if (dx1 > 0 || dx2 > 0)
	{
		if (rowBorderType == BORDER_CONSTANT)
		{
			int nr = isSeparable() ? 1 : (int)rows.size();
			for (i = 0; i < nr; i++)
			{
				uchar* dst = isSeparable() ? &srcRow[0] : alignPtr(&ringBuf[0], VEC_ALIGN) + bufStep * i;
				memcpy(dst, constVal, dx1 * esz);
				memcpy(dst + (roi.width + ksize.width - 1 - dx2) * esz, constVal, dx2 * esz);
			}
		}
		else
		{
			int xofs1 = std::min(roi.x, anchor.x) - roi.x;

			int btab_esz = borderElemSize, wholeWidth = wholeSize.width;
			int* btab = (int*)&borderTab[0];

			for (i = 0; i < dx1; i++)
			{
				int p0 = (borderInterpolate(i - dx1, wholeWidth, rowBorderType) + xofs1) * btab_esz;
				for (j = 0; j < btab_esz; j++)
					btab[i * btab_esz + j] = p0 + j;
			}

			for (i = 0; i < dx2; i++)
			{
				int p0 = (borderInterpolate(wholeWidth + i, wholeWidth, rowBorderType) + xofs1) * btab_esz;
				for (j = 0; j < btab_esz; j++)
					btab[(i + dx1) * btab_esz + j] = p0 + j;
			}
		}
	}

	rowCount = dstY = 0;
	startY = startY0 = std::max(roi.y - anchor.y, 0);
	endY = std::min(roi.y + roi.height + ksize.height - anchor.y - 1, wholeSize.height);
	if (columnFilter)
		columnFilter->reset();
	if (filter2D)
		filter2D->reset();

	return startY;
}


int FilterEngine::start(const Mat& src, const Rect& _srcRoi,
	bool isolated, int maxBufRows)
{
	Rect srcRoi = _srcRoi;

	if (srcRoi == Rect(0, 0, -1, -1))
		srcRoi = Rect(0, 0, src.cols, src.rows);

	CV_Assert(srcRoi.x >= 0 && srcRoi.y >= 0 &&
		srcRoi.width >= 0 && srcRoi.height >= 0 &&
		srcRoi.x + srcRoi.width <= src.cols &&
		srcRoi.y + srcRoi.height <= src.rows);

	Point ofs;
	Size wsz(src.cols, src.rows);
	if (!isolated)
		src.locateROI(wsz, ofs);
	start(wsz, srcRoi + ofs, maxBufRows);

	return startY - ofs.y;
}


int FilterEngine::remainingInputRows() const
{
	return endY - startY - rowCount;
}

int FilterEngine::remainingOutputRows() const
{
	return roi.height - dstY;
}

int FilterEngine::proceed(const uchar* src, int srcstep, int count,
	uchar* dst, int dststep)
{
	CV_Assert(wholeSize.width > 0 && wholeSize.height > 0);

	const int* btab = &borderTab[0];
	int esz = (int)getElemSize(srcType), btab_esz = borderElemSize;
	uchar** brows = &rows[0];
	int bufRows = (int)rows.size();
	int cn = CV_MAT_CN(bufType);
	int width = roi.width, kwidth = ksize.width;
	int kheight = ksize.height, ay = anchor.y;
	int _dx1 = dx1, _dx2 = dx2;
	int width1 = roi.width + kwidth - 1;
	int xofs1 = std::min(roi.x, anchor.x);
	bool isSep = isSeparable();
	bool makeBorder = (_dx1 > 0 || _dx2 > 0) && rowBorderType != BORDER_CONSTANT;
	int dy = 0, i = 0;

	src -= xofs1 * esz;
	count = std::min(count, remainingInputRows());

	CV_Assert(src && dst && count > 0);

	for (;; dst += dststep * i, dy += i)
	{
		int dcount = bufRows - ay - startY - rowCount + roi.y;
		dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
		dcount = std::min(dcount, count);
		count -= dcount;
		for (; dcount-- > 0; src += srcstep)
		{
			int bi = (startY - startY0 + rowCount) % bufRows;
			uchar* brow = alignPtr(&ringBuf[0], VEC_ALIGN) + bi * bufStep;
			uchar* row = isSep ? &srcRow[0] : brow;

			if (++rowCount > bufRows)
			{
				--rowCount;
				++startY;
			}

			memcpy(row + _dx1 * esz, src, (width1 - _dx2 - _dx1) * esz);

			if (makeBorder)
			{
				if (btab_esz * (int)sizeof(int) == esz)
				{
					const int* isrc = (const int*)src;
					int* irow = (int*)row;

					for (i = 0; i < _dx1 * btab_esz; i++)
						irow[i] = isrc[btab[i]];
					for (i = 0; i < _dx2 * btab_esz; i++)
						irow[i + (width1 - _dx2) * btab_esz] = isrc[btab[i + _dx1 * btab_esz]];
				}
				else
				{
					for (i = 0; i < _dx1 * esz; i++)
						row[i] = src[btab[i]];
					for (i = 0; i < _dx2 * esz; i++)
						row[i + (width1 - _dx2) * esz] = src[btab[i + _dx1 * esz]];
				}
			}

			if (isSep)
				(*rowFilter)(row, brow, width, CV_MAT_CN(srcType));
		}

		int max_i = std::min(bufRows, roi.height - (dstY + dy) + (kheight - 1));
		for (i = 0; i < max_i; i++)
		{
			int srcY = borderInterpolate(dstY + dy + i + roi.y - ay,
				wholeSize.height, columnBorderType);
			if (srcY < 0) // can happen only with constant border type
				brows[i] = alignPtr(&constBorderRow[0], VEC_ALIGN);
			else
			{
				CV_Assert(srcY >= startY);
				if (srcY >= startY + rowCount)
					break;
				int bi = (srcY - startY0) % bufRows;
				brows[i] = alignPtr(&ringBuf[0], VEC_ALIGN) + bi * bufStep;
			}
		}
		if (i < kheight)
			break;
		i -= kheight - 1;
		if (isSeparable())
			(*columnFilter)((const uchar**)brows, dst, dststep, i, roi.width * cn);
		else
			(*filter2D)((const uchar**)brows, dst, dststep, i, roi.width, cn);
	}

	dstY += dy;
	CV_Assert(dstY <= roi.height);
	return dy;
}


void FilterEngine::apply(const Mat& src, Mat& dst,
	const Rect& _srcRoi, Point dstOfs, bool isolated)
{
	CV_Assert(src.type() == srcType && dst.type() == dstType);

	Rect srcRoi = _srcRoi;
	if (srcRoi == Rect(0, 0, -1, -1))
		srcRoi = Rect(0, 0, src.cols, src.rows);

	if (srcRoi.area() == 0)
		return;

	CV_Assert(dstOfs.x >= 0 && dstOfs.y >= 0 &&
		dstOfs.x + srcRoi.width <= dst.cols &&
		dstOfs.y + srcRoi.height <= dst.rows);

	int y = start(src, srcRoi, isolated);
	proceed(src.ptr() + y * src.step + srcRoi.x * src.elemSize(),
		(int)src.step, endY - startY,
		dst.ptr(dstOfs.y) +
		dstOfs.x * dst.elemSize(), (int)dst.step);
}


////////////////
/****************************************************************************************\
Box Filter
\****************************************************************************************/

template<typename T, typename ST> struct RowSum : public BaseRowFilter
{
	RowSum(int _ksize, int _anchor)
	{
		ksize = _ksize;
		anchor = _anchor;
	}

	void operator()(const uchar* src, uchar* dst, int width, int cn)
	{
		const T* S = (const T*)src;
		ST* D = (ST*)dst;
		int i = 0, k, ksz_cn = ksize * cn;

		width = (width - 1) * cn;
		for (k = 0; k < cn; k++, S++, D++)
		{
			ST s = 0;
			for (i = 0; i < ksz_cn; i += cn)
				s += S[i];
			D[0] = s;
			for (i = 0; i < width; i += cn)
			{
				s += S[i + ksz_cn] - S[i];
				D[i + cn] = s;
			}
		}
	}
};


struct ColumnSumFF : public BaseColumnFilter
{
	ColumnSumFF(int _ksize, int _anchor, double _scale)
	{
		ksize = _ksize;
		anchor = _anchor;
		scale = _scale;
		sumCount = 0;
		sumsize = 0;
	}

	void reset() { sumCount = 0; }

	void operator()(const uchar** src, uchar* dst, int dststep, int count, int width)
	{
		int i;
		float* SUM;
		bool haveScale = scale != 1;
		double _scale = scale;

#if CV_SSE2
		bool haveSSE2 = false; checkHardwareSupport(CV_CPU_SSE2);
#endif

		if (width != sumsize)
		{
			sum = new float[width];
			sumsize = width;
			sumCount = 0;
		}

		SUM = &sum[0];
		if (sumCount == 0)
		{
			memset((void*)SUM, 0, width * sizeof(float));
			for (; sumCount < ksize - 1; sumCount++, src++)
			{
				const float* Sp = (const float*)src[0];
				i = 0;
#if CV_SSE2
				if (haveSSE2)
				{
					for (; i < width - 4; i += 4)
					{
						__m128 _sum = _mm_loadu_ps((SUM + i));
						__m128 _sp = _mm_loadu_ps((Sp + i));
						_mm_storeu_ps((SUM + i), _mm_add_ps(_sum, _sp));
					}
				}
#endif
				for (; i < width; i++)
					SUM[i] += Sp[i];
			}
		}
		else
		{
			CV_Assert(sumCount == ksize - 1);
			src += ksize - 1;
		}


		for (; count--; src++)
		{
			const float* Sp = (const float*)src[0];
			const float* Sm = (const float*)src[1 - ksize];
			float* D = (float*)dst;
			if (haveScale)
			{
				i = 0;
#if CV_SSE2
				if (haveSSE2)
				{
					const __m128 scale4 = _mm_set1_ps((float)_scale);
					for (; i < width - 4; i += 4)
					{
						__m128 _sm = _mm_loadu_ps((Sm + i));
						__m128 _s0 = _mm_add_ps(_mm_loadu_ps((SUM + i)),
							_mm_loadu_ps((Sp + i)));
						_mm_storeu_ps(D + i, _mm_mul_ps(scale4, _s0));
						_mm_storeu_ps(SUM + i, _mm_sub_ps(_s0, _sm));
					}
				}
#endif
				for (; i < width; i++)
				{
					float s0 = SUM[i] + Sp[i];
					D[i] = saturate_cast<float>(s0 * _scale);
					SUM[i] = s0 - Sm[i];
				}
			}
			else
			{
				i = 0;
#if CV_SSE2
				if (haveSSE2)
				{
					for (; i < width - 4; i += 4)
					{
						__m128 _sm = _mm_loadu_ps((Sm + i));
						__m128 _s0 = _mm_add_ps(_mm_loadu_ps((SUM + i)),
							_mm_loadu_ps((Sp + i)));
						_mm_storeu_ps(D + i, _s0);
						_mm_storeu_ps(SUM + i, _mm_sub_ps(_s0, _sm));
					}
				}
#endif

				for (; i < width; i++)
				{
					float s0 = SUM[i] + Sp[i];
					D[i] = saturate_cast<uchar>(s0);
					SUM[i] = s0 - Sm[i];
				}
			}
			dst += dststep;
		}
	}

	double scale;
	int sumCount;
	Ptr<float> sum;
	int sumsize;
};

cv::Ptr<cv::FilterEngine> createBoxFilterFFF(int cn, Size ksize, Point anchor, bool normalize, int borderType)
{
	Ptr<BaseRowFilter> rowFilter = Ptr<BaseRowFilter>(new RowSum<float, float>(ksize.width, anchor.x < 0 ? ksize.width / 2 : anchor.x));

	Ptr<BaseColumnFilter> columnFilter = Ptr<BaseColumnFilter>(new ColumnSumFF(ksize.height, anchor.y < 0 ? ksize.height / 2 : anchor.y, normalize ? 1. / (ksize.width * ksize.height) : 1));
	//??fukushima
	return Ptr<FilterEngine>(new FilterEngine(Ptr<BaseFilter>(), rowFilter, columnFilter,
		CV_32F, CV_32F, CV_32F, borderType));
}


void boxFilter2(InputArray _src, OutputArray _dst, int ddepth,
	Size ksize, Point anchor,
	bool normalize, int borderType)
{

	Mat src = _src.getMat();
	int sdepth = src.depth(), cn = src.channels();
	if (ddepth < 0)
		ddepth = sdepth;
	_dst.create(src.size(), CV_MAKETYPE(ddepth, cn));
	Mat dst = _dst.getMat();
	if (borderType != BORDER_CONSTANT && normalize)
	{
		if (src.rows == 1)
			ksize.height = 1;
		if (src.cols == 1)
			ksize.width = 1;
	}

	Ptr<FilterEngine> f = createBoxFilterFFF(cn, ksize, anchor, normalize, borderType);
	f->apply(src, dst);
}

void boxFilter1x1(InputArray _src, OutputArray _dst, int ddepth,
	Size ksize, Point anchor,
	bool normalize, int borderType)
{

	Mat src = _src.getMat();
	int sdepth = src.depth(), cn = src.channels();
	if (ddepth < 0)
		ddepth = sdepth;
	_dst.create(src.size(), CV_MAKETYPE(ddepth, cn));
	Mat dst = _dst.getMat();
	if (borderType != BORDER_CONSTANT && normalize)
	{
		if (src.rows == 1)
			ksize.height = 1;
		if (src.cols == 1)
			ksize.width = 1;
	}

	src.copyTo(dst);
}

namespace cp
{
	static void multiplySSE_float(Mat& src1, const float amp, Mat& dest)
	{

		float* s1 = src1.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const int size = src1.size().area() / 4;
		const __m128 ms2 = _mm_set_ps1(amp);
		const int nn = src1.size().area() - size * 4;
		if (src1.data == dest.data)
		{
			for (int i = size; i--; s1 += 4)
			{
				__m128 ms1 = _mm_load_ps(s1);
				ms1 = _mm_mul_ps(ms1, ms2);
				_mm_store_ps(s1, ms1);
			}
		}
		else
		{
			for (int i = size; i--;)
			{
				__m128 ms1 = _mm_load_ps(s1);
				ms1 = _mm_mul_ps(ms1, ms2);
				_mm_store_ps(d, ms1);

				s1 += 4, d += 4;
			}
		}
		for (int i = 0; i < nn; i++)
		{
			*d++ = *s1++ * amp;
		}
	}

	static void multiplySSE_float(Mat& src1, Mat& src2, Mat& dest)
	{
		float* s1 = src1.ptr<float>(0);
		float* s2 = src2.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const int size = src1.size().area() / 4;
		const int nn = src1.size().area() - size * 4;
		for (int i = size; i--;)
		{
			__m128 ms1 = _mm_load_ps(s1);
			__m128 ms2 = _mm_load_ps(s2);
			ms1 = _mm_mul_ps(ms1, ms2);
			_mm_store_ps(d, ms1);
			s1 += 4, s2 += 4, d += 4;
		}
		for (int i = 0; i < nn; i++)
		{
			*d++ = *s1++ * *s2++;
		}
	}

	static void multiplySSEStream_float(Mat& src1, Mat& src2, Mat& dest)
	{
		float* s1 = src1.ptr<float>(0);
		float* s2 = src2.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const int size = src1.size().area() / 4;
		const int nn = src1.size().area() - size * 4;
		for (int i = size; i--;)
		{
			__m128 ms1 = _mm_load_ps(s1);
			__m128 ms2 = _mm_load_ps(s2);
			ms1 = _mm_mul_ps(ms1, ms2);
			_mm_stream_ps(d, ms1);

			s1 += 4, s2 += 4, d += 4;
		}
		for (int i = 0; i < nn; i++)
		{
			*d++ = *s1++ * *s2++;
		}
	}

	static void multiplySSE_float(Mat& src1, Mat& dest)
	{
		float* s1 = src1.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const int size = src1.size().area() / 4;
		const int nn = src1.size().area() - size * 4;
		for (int i = size; i--;)
		{
			__m128 ms1 = _mm_load_ps(s1);
			ms1 = _mm_mul_ps(ms1, ms1);
			_mm_store_ps(d, ms1);

			s1 += 4, d += 4;
		}
		for (int i = 0; i < nn; i++)
		{
			*d++ = *s1 * *s1;
			s1++;
		}
	}
	static void divideSSE_float(Mat& src1, Mat& src2, Mat& dest)
	{
		float* s1 = src1.ptr<float>(0);
		float* s2 = src2.ptr<float>(0);
		float* d = dest.ptr<float>(0);
		const int size = src1.size().area() / 4;
		const int nn = src1.size().area() - size * 4;
		for (int i = size; i--;)
		{
			__m128 ms1 = _mm_load_ps(s1);
			__m128 ms2 = _mm_load_ps(s2);
			ms1 = _mm_div_ps(ms1, ms2);
			_mm_store_ps(d, ms1);

			s1 += 4, s2 += 4, d += 4;
		}
		for (int i = 0; i < nn; i++)
		{
			*d++ = *s1++ / *s2++;
		}
	}

#ifdef FLOAT_BOX_FILTER_OFF
#define boxFilter2 boxFilter
#endif

	static void guidedFilterSrc1Guidance3SSE_(const Mat& src, const Mat& guidance, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() != 1 && guidance.channels() != 3)
		{
			cout << "Please input gray scale image as src, and color image as guidance." << endl;
			return;
		}
		vector<Mat> I(3);
		vector<Mat> If(3);
		split(guidance, I);

		Mat temp;

		if (src.type() == CV_8U)
		{
			cvt8u32f(src, temp);
			cvt8u32f(I[0], If[0]);
			cvt8u32f(I[1], If[1]);
			cvt8u32f(I[2], If[2]);
		}
		else
		{
			src.convertTo(temp, CV_32F);
			I[0].convertTo(If[0], CV_32F);
			I[1].convertTo(If[1], CV_32F);
			I[2].convertTo(If[2], CV_32F);
		}


		const Size ksize(2 * radius + 1, 2 * radius + 1);
		const Point PT(-1, -1);
		const float e = eps;
		const Size imsize = src.size();
		const int size = src.size().area();
		const int ssesize = size / 4;
		const int nn = size - ssesize * 4;
		const double div = 1.0 / ksize.area();

		Mat mean_I_r(imsize, CV_32F);
		Mat mean_I_g(imsize, CV_32F);
		Mat mean_I_b(imsize, CV_32F);
		Mat mean_p(imsize, CV_32F);

		//cout<<"mean computation"<<endl;
		boxFilter2(If[0], mean_I_r, CV_32F, ksize, PT, true, BORDER_TYPE);
		boxFilter2(If[1], mean_I_g, CV_32F, ksize, PT, true, BORDER_TYPE);
		boxFilter2(If[2], mean_I_b, CV_32F, ksize, PT, true, BORDER_TYPE);

		boxFilter2(temp, mean_p, CV_32F, ksize, PT, true, BORDER_TYPE);

		Mat mean_Ip_r(imsize, CV_32F);
		Mat mean_Ip_g(imsize, CV_32F);
		Mat mean_Ip_b(imsize, CV_32F);
		{
			float* s0 = temp.ptr<float>(0);
			float* s1 = If[0].ptr<float>(0);
			float* s2 = If[1].ptr<float>(0);
			float* s3 = If[2].ptr<float>(0);
			float* d1 = mean_Ip_r.ptr<float>(0);
			float* d2 = mean_Ip_g.ptr<float>(0);
			float* d3 = mean_Ip_b.ptr<float>(0);
			for (int i = ssesize; i--;)
			{
				const __m128 ms1 = _mm_load_ps(s0);
				__m128 ms2 = _mm_load_ps(s1);
				ms2 = _mm_mul_ps(ms1, ms2);
				_mm_store_ps(d1, ms2);

				ms2 = _mm_load_ps(s2);
				ms2 = _mm_mul_ps(ms1, ms2);
				_mm_store_ps(d2, ms2);

				ms2 = _mm_load_ps(s3);
				ms2 = _mm_mul_ps(ms1, ms2);
				_mm_store_ps(d3, ms2);

				s0 += 4, s1 += 4, s2 += 4, s3 += 4, d1 += 4, d2 += 4, d3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*d1 = *s0 * *s1;
				*d2 = *s0 * *s2;
				*d3 = *s0 * *s3;
				s0++, s1++, s2++, s3++, d1++, d2++, d3++;
			}
		}
		boxFilter2(mean_Ip_r, mean_Ip_r, CV_32F, ksize, PT, true, BORDER_TYPE);
		boxFilter2(mean_Ip_g, mean_Ip_g, CV_32F, ksize, PT, true, BORDER_TYPE);
		boxFilter2(mean_Ip_b, mean_Ip_b, CV_32F, ksize, PT, true, BORDER_TYPE);


		//temp: float src will not use;
		//mean_Ip_r,g,b will not use;
		//cout<<"covariance computation"<<endl;
		Mat cov_Ip_r = mean_Ip_r;
		Mat cov_Ip_g = mean_Ip_g;
		Mat cov_Ip_b = mean_Ip_b;

		{
			float* s0 = mean_p.ptr<float>(0);
			float* s1 = mean_I_r.ptr<float>(0);
			float* s2 = mean_I_g.ptr<float>(0);
			float* s3 = mean_I_b.ptr<float>(0);
			float* d1 = cov_Ip_r.ptr<float>(0);
			float* d2 = cov_Ip_g.ptr<float>(0);
			float* d3 = cov_Ip_b.ptr<float>(0);
			for (int i = ssesize; i--;)
			{
				const __m128 ms1 = _mm_load_ps(s0);

				__m128 ms2 = _mm_load_ps(s1);
				ms2 = _mm_mul_ps(ms1, ms2);
				__m128 ms3 = _mm_load_ps(d1);
				ms3 = _mm_sub_ps(ms3, ms2);
				_mm_store_ps(d1, ms3);

				ms2 = _mm_load_ps(s2);
				ms2 = _mm_mul_ps(ms1, ms2);
				ms3 = _mm_load_ps(d2);
				ms3 = _mm_sub_ps(ms3, ms2);
				_mm_store_ps(d2, ms3);

				ms2 = _mm_load_ps(s3);
				ms2 = _mm_mul_ps(ms1, ms2);
				ms3 = _mm_load_ps(d3);
				ms3 = _mm_sub_ps(ms3, ms2);
				_mm_store_ps(d3, ms3);

				s0 += 4, s1 += 4, s2 += 4, s3 += 4, d1 += 4, d2 += 4, d3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*d1 = *d1 - (*s0 * *s1);
				*d2 = *d2 - (*s0 * *s2);
				*d3 = *d3 - (*s0 * *s3);
				s0++, s1++, s2++, s3++, d1++, d2++, d3++;
			}
		}

		//cout<<"variance computation"<<endl;

		Mat var_I_rr;
		Mat var_I_rg;
		Mat var_I_rb;
		Mat var_I_gg;
		Mat var_I_gb;
		Mat var_I_bb;
		multiplySSE_float(If[0], temp);
		boxFilter2(temp, var_I_rr, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiplySSE_float(If[0], If[1], temp);
		boxFilter2(temp, var_I_rg, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiplySSE_float(If[0], If[2], temp);
		boxFilter2(temp, var_I_rb, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiplySSE_float(If[1], temp);
		boxFilter2(temp, var_I_gg, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiplySSE_float(If[1], If[2], temp);
		boxFilter2(temp, var_I_gb, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiplySSE_float(If[2], temp);
		boxFilter2(temp, var_I_bb, CV_32F, ksize, PT, true, BORDER_TYPE);

		{
			float* s1 = mean_I_r.ptr<float>(0);
			float* s2 = mean_I_g.ptr<float>(0);
			float* s3 = mean_I_b.ptr<float>(0);
			float* d1 = var_I_rr.ptr<float>(0);
			float* d2 = var_I_rg.ptr<float>(0);
			float* d3 = var_I_rb.ptr<float>(0);
			float* d4 = var_I_gg.ptr<float>(0);
			float* d5 = var_I_gb.ptr<float>(0);
			float* d6 = var_I_bb.ptr<float>(0);
			const __m128 me = _mm_set1_ps(e);
			for (int i = ssesize; i--;)
			{
				const __m128 mr = _mm_load_ps(s1);

				__m128 md1 = _mm_load_ps(d1);
				__m128 ms1 = _mm_mul_ps(mr, mr);
				ms1 = _mm_sub_ps(md1, ms1);
				ms1 = _mm_add_ps(ms1, me);
				_mm_store_ps(d1, ms1);

				const __m128 mg = _mm_load_ps(s2);
				md1 = _mm_load_ps(d2);
				ms1 = _mm_mul_ps(mr, mg);
				ms1 = _mm_sub_ps(md1, ms1);
				_mm_store_ps(d2, ms1);

				const __m128 mb = _mm_load_ps(s3);
				md1 = _mm_load_ps(d3);
				ms1 = _mm_mul_ps(mr, mb);
				ms1 = _mm_sub_ps(md1, ms1);
				_mm_store_ps(d3, ms1);

				md1 = _mm_load_ps(d4);
				ms1 = _mm_mul_ps(mg, mg);
				ms1 = _mm_sub_ps(md1, ms1);
				ms1 = _mm_add_ps(ms1, me);
				_mm_store_ps(d4, ms1);

				md1 = _mm_load_ps(d5);
				ms1 = _mm_mul_ps(mg, mb);
				ms1 = _mm_sub_ps(md1, ms1);
				_mm_store_ps(d5, ms1);

				md1 = _mm_load_ps(d6);
				ms1 = _mm_mul_ps(mb, mb);
				ms1 = _mm_sub_ps(md1, ms1);
				ms1 = _mm_add_ps(ms1, me);
				_mm_store_ps(d6, ms1);


				s1 += 4, s2 += 4, s3 += 4,
					d1 += 4, d2 += 4, d3 += 4, d4 += 4, d5 += 4, d6 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*d1 = *d1 - (*s1 * *s1) + e;
				*d2 = *d2 - (*s1 * *s2);
				*d3 = *d3 - (*s1 * *s3);
				*d4 = *d4 - (*s2 * *s2) + e;
				*d5 = *d5 - (*s2 * *s3);
				*d6 = *d6 - (*s3 * *s3) + e;
				s1++, s2++, s3++, d1++, d2++, d3++, d4++, d5++, d6++;
			}
		}

		{
			float* rr = var_I_rr.ptr<float>(0);
			float* rg = var_I_rg.ptr<float>(0);
			float* rb = var_I_rb.ptr<float>(0);
			float* gg = var_I_gg.ptr<float>(0);
			float* gb = var_I_gb.ptr<float>(0);
			float* bb = var_I_bb.ptr<float>(0);

			float* covr = cov_Ip_r.ptr<float>(0);
			float* covg = cov_Ip_g.ptr<float>(0);
			float* covb = cov_Ip_b.ptr<float>(0);


			//float CV_DECL_ALIGNED(16) buf[4];
			//for(int i=ssesize;i--;rr+=4,rg+=4,rb+=4,gg+=4,gb+=4,bb+=4,covr+=4,covg+=4,covb+=4)
			for (int i = size; i--; rr++, rg++, rb++, gg++, gb++, bb++, covr++, covg++, covb++)
			{
				const float c0 = *gg * *bb - *gb * *gb;
				const float c1 = *rb * *gb - *rg * *bb;
				const float c2 = *rg * *gb - *rb * *gg;
				const float c4 = *rr * *bb - *rb * *rb;
				const float c5 = *rb * *rg - *rr * *gb;
				const float c8 = *rr * *gg - *rg * *rg;

				const float det = (*rr * *gg * *bb) + (*rg * *gb * *rb) + (*rb * *rg * *gb) - (*rr * *gb * *gb) - (*rb * *gg * *rb) - (*rg * *rg * *bb);
				const float id = 1.f / det;


				const float r = *covr;
				const float g = *covg;
				const float b = *covb;

				*covr = id * (r * c0 + g * c1 + b * c2);
				*covg = id * (r * c1 + g * c4 + b * c5);
				*covb = id * (r * c2 + g * c5 + b * c8);

				//SSE4 vc2010 make faster code... 
				/*const __m128 v = _mm_set_ps(r,g,b,0.f);
				const __m128 v2 = _mm_set1_ps(id);
				__m128 a = _mm_mul_ps(v2,_mm_add_ps(_mm_add_ps(_mm_dp_ps(v, _mm_set_ps(c0,c1,c2,0.f),225),_mm_dp_ps(v, _mm_set_ps(c1,c4,c5,0.f),226)),_mm_dp_ps(v, _mm_set_ps(c2,c5,c8,0.f),228)));
				_mm_store_ps(buf,a);

				*covr = buf[0];
				*covg = buf[1];
				*covb = buf[2];*/





				//over flow...
				/*
				__m128 mrr = _mm_load_ps(rr);
				__m128 mrg = _mm_load_ps(rg);
				__m128 mrb = _mm_load_ps(rb);
				__m128 mgg = _mm_load_ps(gg);
				__m128 mgb = _mm_load_ps(gb);
				__m128 mbb = _mm_load_ps(bb);

				__m128 ggbb = _mm_mul_ps(mgg,mbb);
				__m128 gbgb = _mm_mul_ps(mgb,mgb);
				//float c0 = *gg * *bb - *gb * *gb;
				__m128 mc0 = _mm_sub_ps(ggbb,gbgb);

				__m128 rbgb = _mm_mul_ps(mrb,mgb);
				__m128 rgbb = _mm_mul_ps(mrg,mbb);
				//float c1 = *rb * *gb - *rg * *bb;
				__m128 mc1 = _mm_sub_ps(rbgb,rgbb);

				__m128 rggb = _mm_mul_ps(mrg,mgb);
				__m128 rbgg = _mm_mul_ps(mrb,mgg);
				//float c2 = *rg * *gb - *rb * *gg;
				__m128 mc2 = _mm_sub_ps(rbgb,rbgg);

				__m128 rrbb = _mm_mul_ps(mrr,mbb);
				__m128 rbrb = _mm_mul_ps(mrb,mrb);
				//float c4 = *rr * *bb - *rb * *rb;
				__m128 mc4 = _mm_sub_ps(rrbb,rbrb);

				__m128 rbrg = _mm_mul_ps(mrb,mrg);
				__m128 rrgb = _mm_mul_ps(mrr,mgb);
				//float c5 = *rb * *rg - *rr * *gb;
				__m128 mc5 = _mm_sub_ps(rbrg,rrgb);

				__m128 rrgg = _mm_mul_ps(mrr,mgg);
				__m128 rgrg = _mm_mul_ps(mrg,mrg);
				//float c8 = *rr * *gg - *rg * *rg;
				__m128 mc8 = _mm_sub_ps(rrgg,rgrg);

				//__m128 m1 = _mm_set1_ps(1.0f);
				const __m128 iv = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rrgg,mbb),_mm_mul_ps(rggb,mrb)),_mm_mul_ps(rbrg,mgb)),
				_mm_add_ps(_mm_add_ps(_mm_mul_ps(rrgb,mgb),_mm_mul_ps(rbgg,mrb)),_mm_mul_ps(rgrg,mbb)));
				//const float det = (*rr * *gg * *bb) + (*rg * *gb * *rb) + (*rb * *rg * *gb)  -(*rr * *gb * *gb) - (*rb * *gg * *rb) - (*rg * *rg* *bb);
				//const float id = 1.f/det;

				//const float r = *covr;
				//const float g = *covg;
				//const float b = *covb;
				const __m128 mcvr = _mm_load_ps(covr);
				const __m128 mcvg = _mm_load_ps(covg);
				const __m128 mcvb = _mm_load_ps(covb);

				mrr = _mm_div_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(mcvr,mc0),_mm_mul_ps(mcvg,mc1)),_mm_mul_ps(mcvb,mc2)),iv);
				mrg = _mm_div_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(mcvr,mc1),_mm_mul_ps(mcvg,mc4)),_mm_mul_ps(mcvb,mc5)),iv);
				mrb = _mm_div_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(mcvr,mc2),_mm_mul_ps(mcvg,mc5)),_mm_mul_ps(mcvb,mc8)),iv);
				//*covr = id* (r*c0 + g*c1 + b*c2);
				//*covg = id* (r*c1 + g*c4 + b*c5);
				//*covb = id* (r*c2 + g*c5 + b*c8);

				_mm_store_ps(covr,mrr);
				_mm_store_ps(covg,mrg);
				_mm_store_ps(covb,mrb);
				*/
			}
		}

		Mat a_r = cov_Ip_r;
		Mat a_g = cov_Ip_g;
		Mat a_b = cov_Ip_b;

		{
			float* s0 = mean_p.ptr<float>(0);
			float* s1 = mean_I_r.ptr<float>(0);
			float* s2 = mean_I_g.ptr<float>(0);
			float* s3 = mean_I_b.ptr<float>(0);
			float* a1 = a_r.ptr<float>(0);
			float* a2 = a_g.ptr<float>(0);
			float* a3 = a_b.ptr<float>(0);
			for (int i = ssesize; i--;)
			{
				__m128 ms3 = _mm_load_ps(s0);

				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(a1);
				ms1 = _mm_mul_ps(ms2, ms1);
				ms3 = _mm_sub_ps(ms3, ms1);

				ms1 = _mm_load_ps(s2);
				ms2 = _mm_load_ps(a2);
				ms1 = _mm_mul_ps(ms2, ms1);
				ms3 = _mm_sub_ps(ms3, ms1);

				ms1 = _mm_load_ps(s3);
				ms2 = _mm_load_ps(a3);
				ms1 = _mm_mul_ps(ms2, ms1);
				ms3 = _mm_sub_ps(ms3, ms1);

				_mm_store_ps(s0, ms3);

				s0 += 4, s1 += 4, s2 += 4, s3 += 4, a1 += 4, a2 += 4, a3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*s0 = *s0 - (*a1 * *s1) - ((*a2 * *s2)) - (*a3 * *s3);
				s0++, s1++, s2++, s3++, a1++, a2++, a3++;
			}
		}

		Mat b = mean_p;
		boxFilter2(a_r, a_r, CV_32F, ksize, PT, true, BORDER_TYPE);//break a_r
		boxFilter2(a_g, a_g, CV_32F, ksize, PT, true, BORDER_TYPE);//break a_g
		boxFilter2(a_b, a_b, CV_32F, ksize, PT, true, BORDER_TYPE);//break a_b
		boxFilter2(b, temp, CV_32F, ksize, PT, true, BORDER_TYPE);
		float amp = 1.f;
		{
			float* s0 = temp.ptr<float>(0);
			float* s1 = If[0].ptr<float>(0);
			float* s2 = If[1].ptr<float>(0);
			float* s3 = If[2].ptr<float>(0);
			float* d1 = a_r.ptr<float>(0);
			float* d2 = a_g.ptr<float>(0);
			float* d3 = a_b.ptr<float>(0);
			const __m128 me = _mm_set1_ps(amp);
			for (int i = ssesize; i--;)
			{
				__m128 ms1 = _mm_load_ps(s0);

				__m128 ms2 = _mm_load_ps(s1);
				__m128 ms3 = _mm_load_ps(d1);
				ms2 = _mm_mul_ps(ms2, ms3);
				ms1 = _mm_add_ps(ms1, ms2);

				ms2 = _mm_load_ps(s2);
				ms3 = _mm_load_ps(d2);
				ms2 = _mm_mul_ps(ms2, ms3);
				ms1 = _mm_add_ps(ms1, ms2);

				ms2 = _mm_load_ps(s3);
				ms3 = _mm_load_ps(d3);
				ms2 = _mm_mul_ps(ms2, ms3);
				ms1 = _mm_add_ps(ms1, ms2);

				ms1 = _mm_mul_ps(ms1, me);
				_mm_store_ps(s0, ms1);

				s0 += 4, s1 += 4, s2 += 4, s3 += 4, d1 += 4, d2 += 4, d3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*s0 = amp * (*s0 + (*s1 * *d1) + (*s2 * *d2) + (*s3 * *d3));
				s0++, s1++, s2++, s3++, d1++, d2++, d3++;
			}
		}
		temp.convertTo(dest, src.type());
	}


	static void guidedFilterSrc1Guidance3_(const Mat& src, const Mat& guidance, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() != 1 && guidance.channels() != 3)
		{
			cout << "Please input gray scale image as src, and color image as guidance." << endl;
			return;
		}
		vector<Mat> I(3);
		vector<Mat> If(3);
		split(guidance, I);

		const Size ksize(2 * radius + 1, 2 * radius + 1);
		const Point PT(-1, -1);
		const float e = eps;
		const Size imsize = src.size();
		const int size = src.size().area();
		const double div = 1.0 / ksize.area();

		Mat temp(imsize, CV_32F);
		If[0].create(imsize, CV_32F);
		If[1].create(imsize, CV_32F);
		If[2].create(imsize, CV_32F);

		if (src.type() == CV_8U)
		{
			cvt8u32f(src, temp, 1.f / 255.f);
			cvt8u32f(I[0], If[0], 1.f / 255.f);
			cvt8u32f(I[1], If[1], 1.f / 255.f);
			cvt8u32f(I[2], If[2], 1.f / 255.f);
		}
		else
		{
			src.convertTo(temp, CV_32F, 1.f / 255.f);
			I[0].convertTo(If[0], CV_32F, 1.f / 255.f);
			I[1].convertTo(If[1], CV_32F, 1.f / 255.f);
			I[2].convertTo(If[2], CV_32F, 1.f / 255.f);
		}
		Mat mean_I_r(imsize, CV_32F);
		Mat mean_I_g(imsize, CV_32F);
		Mat mean_I_b(imsize, CV_32F);
		Mat mean_p(imsize, CV_32F);

		//cout<<"mean computation"<<endl;
		boxFilter(If[0], mean_I_r, CV_32F, ksize, PT, true, BORDER_TYPE);
		boxFilter(If[1], mean_I_g, CV_32F, ksize, PT, true, BORDER_TYPE);
		boxFilter(If[2], mean_I_b, CV_32F, ksize, PT, true, BORDER_TYPE);

		boxFilter(temp, mean_p, CV_32F, ksize, PT, true, BORDER_TYPE);

		Mat mean_Ip_r(imsize, CV_32F);
		Mat mean_Ip_g(imsize, CV_32F);
		Mat mean_Ip_b(imsize, CV_32F);
		multiply(If[0], temp, mean_Ip_r);//Ir*p
		boxFilter(mean_Ip_r, mean_Ip_r, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(If[1], temp, mean_Ip_g);//Ig*p
		boxFilter(mean_Ip_g, mean_Ip_g, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(If[2], temp, mean_Ip_b);//Ib*p
		boxFilter(mean_Ip_b, mean_Ip_b, CV_32F, ksize, PT, true, BORDER_TYPE);

		//temp: float src will not use;
		//mean_Ip_r,g,b will not use;
		//cout<<"covariance computation"<<endl;
		Mat cov_Ip_r = mean_Ip_r;
		Mat cov_Ip_g = mean_Ip_g;
		Mat cov_Ip_b = mean_Ip_b;
		multiply(mean_I_r, mean_p, temp);
		cov_Ip_r -= temp;
		multiply(mean_I_g, mean_p, temp);
		cov_Ip_g -= temp;
		multiply(mean_I_b, mean_p, temp);
		cov_Ip_b -= temp;

		//cout<<"variance computation"<<endl;

		//getCovImage();
		//ココからスプリット
		Mat var_I_rr;
		Mat var_I_rg;
		Mat var_I_rb;
		Mat var_I_gg;
		Mat var_I_gb;
		Mat var_I_bb;
		multiply(If[0], If[0], temp);
		boxFilter(temp, var_I_rr, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(mean_I_r, mean_I_r, temp);
		var_I_rr -= temp;

		multiply(If[0], If[1], temp);
		boxFilter(temp, var_I_rg, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(mean_I_r, mean_I_g, temp);
		var_I_rg -= temp;

		multiply(If[0], If[2], temp);
		boxFilter(temp, var_I_rb, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(mean_I_r, mean_I_b, temp);
		var_I_rb -= temp;

		multiply(If[1], If[1], temp);
		boxFilter(temp, var_I_gg, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(mean_I_g, mean_I_g, temp);
		var_I_gg -= temp;

		multiply(If[1], If[2], temp);
		boxFilter(temp, var_I_gb, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(mean_I_g, mean_I_b, temp);
		var_I_gb -= temp;

		multiply(If[2], If[2], temp);
		boxFilter(temp, var_I_bb, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(mean_I_b, mean_I_b, temp);
		var_I_bb -= temp;

		var_I_rr += e;
		var_I_gg += e;
		var_I_bb += e;

		float* rr = var_I_rr.ptr<float>(0);
		float* rg = var_I_rg.ptr<float>(0);
		float* rb = var_I_rb.ptr<float>(0);
		float* gg = var_I_gg.ptr<float>(0);
		float* gb = var_I_gb.ptr<float>(0);
		float* bb = var_I_bb.ptr<float>(0);

		float* covr = cov_Ip_r.ptr<float>(0);
		float* covg = cov_Ip_g.ptr<float>(0);
		float* covb = cov_Ip_b.ptr<float>(0);

		{
			//CalcTime t("cov");
			for (int i = size; i--; rr++, rg++, rb++, gg++, gb++, bb++, covr++, covg++, covb++)
			{/*
			 Matx33f sigmaEps
			 (
			 *rr,*rg,*rb,
			 *rg,*gg,*gb,
			 *rb,*gb,*bb
			 );
			 Matx33f inv = sigmaEps.inv(cv::DECOMP_LU);


			 const float r = *covr;
			 const float g = *covg;
			 const float b = *covb;
			 *covr = r*inv(0,0) + g*inv(1,0) + b*inv(2,0);
			 *covg = r*inv(0,1) + g*inv(1,1) + b*inv(2,1);
			 *covb = r*inv(0,2) + g*inv(1,2) + b*inv(2,2);*/

				const float det = (*rr * *gg * *bb) + (*rg * *gb * *rb) + (*rb * *rg * *gb) - (*rr * *gb * *gb) - (*rb * *gg * *rb) - (*rg * *rg * *bb);
				const float id = 1.f / det;

				float c0 = *gg * *bb - *gb * *gb;
				float c1 = *rb * *gb - *rg * *bb;
				float c2 = *rg * *gb - *rb * *gg;
				float c4 = *rr * *bb - *rb * *rb;
				float c5 = *rb * *rg - *rr * *gb;
				float c8 = *rr * *gg - *rg * *rg;
				const float r = *covr;
				const float g = *covg;
				const float b = *covb;
				*covr = id * (r * c0 + g * c1 + b * c2);
				*covg = id * (r * c1 + g * c4 + b * c5);
				*covb = id * (r * c2 + g * c5 + b * c8);

				/*cout<<format("%f %f %f \n %f %f %f \n %f %f %f \n",
				id*(*gg * *bb - *gb * *gb),//c0
				id*(*rb * *gb - *rg * *bb),//c1
				id*(*rg * *gb - *rb * *gg),//c2
				id*(*gb * *rb - *rg * *bb),//c3=c1
				id*(*rr * *bb - *rb * *rb),//c4
				id*(*rb * *rg - *rr * *gb),//c5
				id*(*rg * *gb - *rb * *gg),//c6=c2
				id*(*rb * *rg - *rr * *gb),//c7 = c5
				id*(*rr * *gg - *rg * *rg)//c8
				);
				cout<<determinant(sigmaEps)<<endl;
				cout<<det<<endl;
				cout<<Mat(inv)<<endl;
				getchar();*/
			}
		}

		Mat a_r = cov_Ip_r;
		Mat a_g = cov_Ip_g;
		Mat a_b = cov_Ip_b;

		multiply(a_r, mean_I_r, mean_I_r);//break mean_I_r;
		multiply(a_g, mean_I_g, mean_I_g);//break mean_I_g;
		multiply(a_b, mean_I_b, mean_I_b);//break mean_I_b;
		mean_p -= (mean_I_r + mean_I_g + mean_I_b);
		Mat b = mean_p;

		boxFilter(a_r, a_r, CV_32F, ksize, PT, true, BORDER_TYPE);//break a_r
		boxFilter(a_g, a_g, CV_32F, ksize, PT, true, BORDER_TYPE);//break a_g
		boxFilter(a_b, a_b, CV_32F, ksize, PT, true, BORDER_TYPE);//break a_b

		boxFilter(b, temp, CV_32F, ksize, PT, true, BORDER_TYPE);
		multiply(a_r, If[0], a_r);
		multiply(a_g, If[1], a_g);
		multiply(a_b, If[2], a_b);
		temp += (a_r + a_g + a_b);

		temp.convertTo(dest, src.type(), 255.0);
	}

	void guidedFilterSrc1Guidance1SSE_(const Mat& src, const Mat& joint, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() != 1 && joint.channels() != 1)
		{
			cout << "Please input gray scale image." << endl;
			return;
		}
		if (dest.empty()) dest.create(src.size(), src.type());
		const Size ksize(2 * radius + 1, 2 * radius + 1);
		const Size imsize = src.size();
		const int ssesize = imsize.area() / 4; //const int ssesize = 0;
		const int nn = imsize.area() - ssesize * 4;
		const float e = eps;

		STATICMAT x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);
		STATICMAT mJoint(imsize, CV_32F);//mean_I
		STATICMAT mSrc(imsize, CV_32F);//mean_p

		STATICMAT sf(imsize, CV_32F);
		STATICMAT jf(imsize, CV_32F);

		if (src.type() == CV_8U)
		{
			cvt8u32f(src, sf);
			cvt8u32f(joint, jf);
		}
		else
		{
			src.convertTo(sf, CV_32F);
			joint.convertTo(jf, CV_32F);
		}

		{
			float* s1 = jf.ptr<float>(0);
			float* s2 = sf.ptr<float>(0);
			float* d = x2.ptr<float>(0);
			float* d2 = x1.ptr<float>(0);

			for (int i = ssesize; i--;)
			{
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);
				ms2 = _mm_mul_ps(ms1, ms2);
				_mm_store_ps(d, ms2);

				ms1 = _mm_mul_ps(ms1, ms1);
				_mm_store_ps(d2, ms1);
				s1 += 4, s2 += 4, d += 4, d2 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*d2++ = *s1 * *s1;
				*d++ = *s1++ * *s2++;
			}
		}

		boxFilter2(jf, mJoint, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mJoint*K
		boxFilter2(sf, mSrc, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mSrc*K
		boxFilter2(x2, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*K
		boxFilter2(x1, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*K	

		{
			float* s1 = mJoint.ptr<float>(0);
			float* s2 = x2.ptr<float>(0);
			float* s3 = x3.ptr<float>(0);//*s1 = *s3 - *s1 * *s5;
			float* s4 = x1.ptr<float>(0);
			float* s5 = mSrc.ptr<float>(0);
			const __m128 ms4 = _mm_set1_ps(e);
			for (int i = ssesize; i--;)
			{
				//mjoint*mjoint
				const __m128 ms1 = _mm_load_ps(s1);
				const __m128 ms5 = _mm_load_ps(s5);
				__m128 ms2 = _mm_mul_ps(ms1, ms1);
				//x2-x1+e
				__m128 ms3 = _mm_load_ps(s2);
				ms2 = _mm_sub_ps(ms3, ms2);
				ms2 = _mm_add_ps(ms2, ms4);
				//x3/xx
				ms3 = _mm_load_ps(s3);
				ms3 = _mm_sub_ps(ms3, _mm_mul_ps(ms1, ms5));

				ms2 = _mm_div_ps(ms3, ms2);
				_mm_store_ps(s3, ms2);
				//ms
				ms2 = _mm_mul_ps(ms2, ms1);
				ms3 = _mm_sub_ps(ms2, ms5);
				_mm_store_ps(s4, ms3);

				s1 += 4, s2 += 4, s3 += 4, s4 += 4, s5 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*s3 = (*s3 - (*s1 * *s5)) / (*s2 - (*s1 * *s1) + e);
				*s4 = (*s3 * *s1) - *s5;
				s1++, s2++, s3++, s4++, s5++;
			}
		}

		boxFilter2(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
		boxFilter2(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k

		{
			float* s1 = x2.ptr<float>(0);
			float* s2 = jf.ptr<float>(0);
			float* s3 = x3.ptr<float>(0);

			for (int i = ssesize; i--;)
			{
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);
				ms1 = _mm_mul_ps(ms1, ms2);

				ms2 = _mm_load_ps(s3);
				ms1 = _mm_sub_ps(ms1, ms2);

				_mm_store_ps(s1, ms1);

				s1 += 4, s2 += 4, s3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*s1 = ((*s1 * *s2) - *s3);
				s1++, s2++, s3++;
			}
		}
		x2.convertTo(dest, src.type());
	}

	static void guidedFilterSrc1Guidance1_(const Mat& src, const Mat& joint, Mat& dest, const int radius, const float eps)
	{
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;

		//論文と同じにするために正規化．実際は要らない．
		Mat sf; src.convertTo(sf, CV_32F, 1.0 / 255);
		Mat jf; joint.convertTo(jf, CV_32F, 1.0 / 255);

		Mat mJoint(imsize, CV_32F);//mean_I
		Mat mSrc(imsize, CV_32F);//mean_p

		boxFilter(jf, mJoint, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mJoint*K/////////////////////////
		boxFilter(sf, mSrc, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(jf, sf, x1);//x1*1
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//corrI
		multiply(mJoint, mSrc, x1);//;x1*K*K
		x3 -= x1;//x1 div k ->x3*k
		multiply(jf, jf, x1);////////////////////////////////////
		boxFilter(x1, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*K
		multiply(mJoint, mJoint, x1);//x1*K*K

		sf = Mat(x2 - x1) + e;
		divide(x3, sf, x3);//x3->a
		multiply(x3, mJoint, x1);
		x1 -= mSrc;//x1->b
		boxFilter(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k
		multiply(x2, jf, x1);//x1*K
		Mat temp = x1 - x3;//
		temp.convertTo(dest, src.type(), 255);
	}

	void guidedFilterRGBSplit(Mat& src, Mat& guidance, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() == 3 && guidance.channels() == 3)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			vector<Mat> j(3);
			split(src, v);
			split(guidance, j);
			guidedFilterSrc1Guidance1_(v[0], j[0], d[0], radius, eps);
			guidedFilterSrc1Guidance1_(v[1], j[0], d[1], radius, eps);
			guidedFilterSrc1Guidance1_(v[2], j[0], d[2], radius, eps);
			merge(d, dest);
		}
	}

	static void weightedBoxFilter(Mat& src, Mat& weight, Mat& dest, int type, Size ksize, Point pt, int border_type)
	{
		Mat sw;
		Mat wsrc(src.size(), CV_32F);
		boxFilter(weight, sw, CV_32F, ksize, pt, true, border_type);
		multiplySSE_float(src, weight, wsrc);//sf*sf
		boxFilter(wsrc, dest, CV_32F, ksize, pt, true, border_type);
		divideSSE_float(dest, sw, dest);
	}
	void weightedAdaptiveGuidedFilter(Mat& src, Mat& guidance, Mat& guidance2, Mat& weight, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() != 1 && guidance.channels() != 1)
		{
			cout << "Please input gray scale image." << endl;
			return;
		}
		//some opt
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;

		Mat sf; src.convertTo(sf, CV_32F, 1.0 / 255);
		Mat jf; guidance.convertTo(jf, CV_32F, 1.0 / 255);
		Mat mJoint(imsize, CV_32F);//mean_I
		Mat mSrc(imsize, CV_32F);//mean_p

		weightedBoxFilter(jf, weight, mJoint, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//mJoint*K
		weightedBoxFilter(sf, weight, mSrc, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//mSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiplySSE_float(jf, sf, x1);//x1*1
		weightedBoxFilter(x1, weight, x3, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//x3*K
		multiplySSE_float(mJoint, mSrc, x1);//;x1*K*K
		x3 -= x1;//x1 div k ->x3*k
		multiplySSE_float(jf, x1);
		weightedBoxFilter(x1, weight, x2, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//x2*K
		multiplySSE_float(mJoint, x1);//x1*K*K
		sf = Mat(x2 - x1) + e;
		divideSSE_float(x3, sf, x3);
		multiplySSE_float(x3, mJoint, x1);
		x1 -= mSrc;
		boxFilter(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k

		guidance2.convertTo(jf, CV_32F, 1.0 / 255);
		multiplySSE_float(x2, jf, x1);//x1*K
		Mat temp = x1 - x3;//
		temp.convertTo(dest, src.type(), 255);
	}

	void weightedGuidedFilter(Mat& src, Mat& guidance, Mat& weight, Mat& dest, const int radius, const float eps)
	{

		if (src.channels() != 1 && guidance.channels() != 1)
		{
			cout << "Please input gray scale image." << endl;
			return;
		}
		//some opt
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;

		Mat sf; src.convertTo(sf, CV_32F, 1.0 / 255);
		Mat jf; guidance.convertTo(jf, CV_32F, 1.0 / 255);
		Mat mJoint(imsize, CV_32F);//mean_I
		Mat mSrc(imsize, CV_32F);//mean_p

		weightedBoxFilter(jf, weight, mJoint, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//mJoint*K
		weightedBoxFilter(sf, weight, mSrc, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//mSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiplySSE_float(jf, sf, x1);//x1*1
		weightedBoxFilter(x1, weight, x3, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//x3*K
		multiplySSE_float(mJoint, mSrc, x1);//;x1*K*K
		x3 -= x1;//x1 div k ->x3*k
		multiplySSE_float(jf, x1);
		weightedBoxFilter(x1, weight, x2, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//x2*K
		multiplySSE_float(mJoint, x1);//x1*K*K
		sf = Mat(x2 - x1) + e;
		divideSSE_float(x3, sf, x3);
		multiplySSE_float(x3, mJoint, x1);
		x1 -= mSrc;
		boxFilter(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k
		multiplySSE_float(x2, jf, x1);//x1*K
		Mat temp = x1 - x3;//
		temp.convertTo(dest, src.type(), 255);


		/*	if(src.channels()!=1 && guidance.channels()!=1)
		{
		cout<<"Please input gray scale image."<<endl;
		return;
		}
		//some opt
		Size ksize(2*radius+1,2*radius+1);
		Size imsize = src.size();
		const float e=eps*eps;

		Mat sf;src.convertTo(sf,CV_32F,1.0/255);
		Mat jf;guidance.convertTo(jf,CV_32F,1.0/255);
		Mat mJoint(imsize,CV_32F);//mean_I
		Mat mSrc(imsize,CV_32F);//mean_p

		boxFilter(jf,mJoint,CV_32F,ksize,Point(-1,-1),true,BORDER_TYPE);//mJoint*K
		boxFilter(sf,mSrc,CV_32F,ksize,Point(-1,-1),true,BORDER_TYPE);//mSrc*K

		Mat x1(imsize,CV_32F),x2(imsize,CV_32F),x3(imsize,CV_32F);

		multiplySSE_float(jf,sf,x1);//x1*1
		boxFilter(x1,x3,CV_32F,ksize,Point(-1,-1),true,BORDER_TYPE);//x3*K
		multiplySSE_float(mJoint,mSrc,x1);//;x1*K*K
		x3-=x1;//x1 div k ->x3*k
		multiplySSE_float(jf,x1);
		boxFilter(x1,x2,CV_32F,ksize,Point(-1,-1),true,BORDER_TYPE);//x2*K
		multiplySSE_float(mJoint,x1);//x1*K*K
		sf = Mat(x2 - x1)+e;
		divideSSE_float(x3,sf,x3);
		multiplySSE_float(x3,mJoint,x1);
		x1-=mSrc;
		weightedBoxFilter(x3,weight,x2,CV_32F,ksize,Point(-1,-1),BORDER_TYPE);//x2*k
		weightedBoxFilter(x1,weight,x3,CV_32F,ksize,Point(-1,-1),BORDER_TYPE);//x3*k
		multiplySSE_float(x2,jf,x1);//x1*K
		Mat temp = x1-x3;//
		temp.convertTo(dest,src.type(),255);*/
	}

	void weightedGuidedFilter2(Mat& src, Mat& guidance, Mat& weight, Mat& dest, const int radius, const float eps)
	{

		if (src.channels() != 1 && guidance.channels() != 1)
		{
			cout << "Please input gray scale image." << endl;
			return;
		}
		//some opt
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;

		Mat sf; src.convertTo(sf, CV_32F, 1.0 / 255);
		Mat jf; guidance.convertTo(jf, CV_32F, 1.0 / 255);
		Mat mJoint(imsize, CV_32F);//mean_I
		Mat mSrc(imsize, CV_32F);//mean_p

		boxFilter(jf, mJoint, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mJoint*K
		boxFilter(sf, mSrc, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiplySSE_float(jf, sf, x1);//x1*1
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*K
		multiplySSE_float(mJoint, mSrc, x1);//;x1*K*K
		x3 -= x1;//x1 div k ->x3*k
		multiplySSE_float(jf, x1);
		boxFilter(x1, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*K
		multiplySSE_float(mJoint, x1);//x1*K*K
		sf = Mat(x2 - x1) + e;
		divideSSE_float(x3, sf, x3);
		multiplySSE_float(x3, mJoint, x1);
		x1 -= mSrc;
		weightedBoxFilter(x3, weight, x2, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//x2*k
		weightedBoxFilter(x1, weight, x3, CV_32F, ksize, Point(-1, -1), BORDER_TYPE);//x3*k
		multiplySSE_float(x2, jf, x1);//x1*K
		Mat temp = x1 - x3;//
		temp.convertTo(dest, src.type(), 255);
	}

	void guidedFilter(const Mat& src, const Mat& guidance, Mat& dest, const int radius, const float eps)
	{
		if (radius == 0) { src.copyTo(dest); return; }
		bool sse = checkHardwareSupport(CV_CPU_SSE2);

		if (src.channels() == 1 && guidance.channels() == 3)
		{
			if (sse)
				guidedFilterSrc1Guidance3SSE_(src, guidance, dest, radius, eps);
			else
				guidedFilterSrc1Guidance3_(src, guidance, dest, radius, eps);
		}
		else if (src.channels() == 1 && guidance.channels() == 1)
		{
			if (sse)
			{
				guidedFilterSrc1Guidance1SSE_(src, guidance, dest, radius, eps);
			}
			else
			{
				guidedFilterSrc1Guidance1_(src, guidance, dest, radius, eps);
			}
		}
		else if (src.channels() == 3 && guidance.channels() == 3)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			split(src, v);
			if (sse)
			{
#pragma omp parallel for
				for (int i = 0; i < 3; i++)
					guidedFilterSrc1Guidance3SSE_(v[i], guidance, d[i], radius, eps);

			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < 3; i++)
					guidedFilterSrc1Guidance3_(v[i], guidance, d[i], radius, eps);

			}
			merge(d, dest);
		}
		else if (src.channels() == 3 && guidance.channels() == 1)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			split(src, v);
			if (sse)
			{
#pragma omp parallel for
				for (int i = 0; i < 3; i++)
					guidedFilterSrc1Guidance1SSE_(v[i], guidance, d[i], radius, eps);

			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < 3; i++)
					guidedFilterSrc1Guidance1_(v[i], guidance, d[i], radius, eps);

			}
			merge(d, dest);
		}
	}


	void guidedFilterSrc1SSE_(const Mat& src, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() != 1)
		{
			cout << "Please input gray scale image." << endl;
			return;
		}
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const int sseims = imsize.area() / 4;
		//const int sseims = 0;
		const int  nn = imsize.area() - sseims * 4;
		const float e = eps;

		Mat sf;
		if (src.depth() == CV_32F) src.copyTo(sf);
		else if (src.depth() == CV_8U) cvt8u32f(src, sf);
		else src.convertTo(sf, CV_32F);

		Mat mSrc(imsize, CV_32F);//mean_p
		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		boxFilter2(sf, mSrc, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mSrc*K
		multiplySSE_float(sf, x1);//sf*sf
		boxFilter2(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//m*sf*sf
		{
			float* s1 = mSrc.ptr<float>(0);
			float* s2 = x3.ptr<float>(0);
			float* s3 = x1.ptr<float>(0);
			const __m128 ms4 = _mm_set1_ps(e);
			for (int i = sseims; i--;)
			{
				const __m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_mul_ps(ms1, ms1);
				__m128 ms3 = _mm_load_ps(s2);

				ms3 = _mm_sub_ps(ms3, ms2);
				ms2 = _mm_add_ps(ms3, ms4);
				ms3 = _mm_div_ps(ms3, ms2);
				_mm_store_ps(s2, ms3);
				ms3 = _mm_mul_ps(ms3, ms1);
				ms3 = _mm_sub_ps(ms3, ms1);
				_mm_store_ps(s3, ms3);

				s1 += 4, s2 += 4, s3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				const float v = *s2 - (*s1 * *s1);
				*s2 = v / (v + e);
				*s3 = (*s2 * *s1) - *s1;
				s1++, s2++, s3++;
			}

		}
		boxFilter2(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
		boxFilter2(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k
		{

			float* s1 = x2.ptr<float>(0);
			float* s2 = sf.ptr<float>(0);
			float* s3 = x3.ptr<float>(0);
			for (int i = sseims; i--;)
			{
				__m128 ms1 = _mm_load_ps(s1);
				__m128 ms2 = _mm_load_ps(s2);
				ms1 = _mm_mul_ps(ms1, ms2);

				ms2 = _mm_load_ps(s3);
				ms1 = _mm_sub_ps(ms1, ms2);
				_mm_store_ps(s1, ms1);

				s1 += 4, s2 += 4, s3 += 4;
			}
			for (int i = 0; i < nn; i++)
			{
				*s1 = ((*s1 * *s2) - *s3);
				s1++, s2++, s3++;
			}
		}

		if (src.depth() == CV_8U)
		{
			cvt32f8u(x2, dest);
		}
		else
		{
			x2.convertTo(dest, src.type());
		}
	}

	void guidedFilterSrc1_(const Mat& src, Mat& dest, const int radius, const float eps)
	{
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;
		Mat sf; src.convertTo(sf, CV_32F);

		Mat mSrc(imsize, CV_32F);//mean_p
		boxFilter(sf, mSrc, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//meanImSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(sf, sf, x1);//sf*sf
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//corrI:m*sf*sf

		multiply(mSrc, mSrc, x1);//;msf*msf
		x3 -= x1;//x3: m*sf*sf-msf*msf;
		x1 = x3 + e;
		divide(x3, x1, x3);
		multiply(x3, mSrc, x1);
		x1 -= mSrc;
		boxFilter(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
		boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k
		multiply(x2, sf, x1);//x1*K
		x2 = x1 - x3;//
		x2.convertTo(dest, src.type());
	}


	void guidedFilter(const Mat& src, Mat& dest, const int radius, const float eps)
	{
		if (radius == 0) { src.copyTo(dest); return; }
		bool sse = checkHardwareSupport(CV_CPU_SSE2);
		if (radius == 0)
			src.copyTo(dest);

		if (src.channels() == 1)
		{
			if (sse)
			{
				guidedFilterSrc1SSE_(src, dest, radius, eps);
			}
			else
			{
				guidedFilterSrc1_(src, dest, radius, eps);
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			split(src, v);
			if (sse)
			{
				guidedFilterSrc1Guidance3SSE_(v[0], src, d[0], radius, eps);
				guidedFilterSrc1Guidance3SSE_(v[1], src, d[1], radius, eps);
				guidedFilterSrc1Guidance3SSE_(v[2], src, d[2], radius, eps);
			}
			else
			{
				guidedFilterSrc1Guidance3_(v[0], src, d[0], radius, eps);
				guidedFilterSrc1Guidance3_(v[1], src, d[1], radius, eps);
				guidedFilterSrc1Guidance3_(v[2], src, d[2], radius, eps);

			}
			merge(d, dest);
		}
	}

	class GuidedFilterInvoler : public cv::ParallelLoopBody
	{
		Size imsize;
		Size gsize;
		vector<Mat> srcGrid;
		vector<Mat> guideGrid;

		Mat* dest2;
		int r;
		float eps;

		bool isGuide;
		int numthread;

	public:
		~GuidedFilterInvoler()
		{
			mergeFromGrid(srcGrid, imsize, *dest2, gsize, 2 * r);
		}
		GuidedFilterInvoler(const Mat& src_, Mat& dest_, int r_, float eps_, int numthread_) :
			dest2(&dest_), r(r_), eps(eps_), numthread(numthread_)
		{
			imsize = src_.size();
			isGuide = false;

			int th = numthread;
			if (th == 1) gsize = Size(1, 1);
			else if (th <= 2) gsize = Size(1, 2);
			else if (th <= 4) gsize = Size(2, 2);
			else if (th <= 8) gsize = Size(2, 4);
			else if (th <= 16) gsize = Size(4, 4);
			else if (th <= 32) gsize = Size(4, 8);
			else if (th <= 64) gsize = Size(8, 8);
			splitToGrid(src_, srcGrid, gsize, 2 * r);
		}

		GuidedFilterInvoler(const Mat& src_, const Mat& guidance_, Mat& dest_, int r_, float eps_, int numthread_) :
			dest2(&dest_), r(r_), eps(eps_), numthread(numthread_)
		{
			imsize = src_.size();
			isGuide = true;

			int th = numthread;
			if (th == 1) gsize = Size(1, 1);
			else if (th <= 2) gsize = Size(1, 2);
			else if (th <= 4) gsize = Size(2, 2);
			else if (th <= 8) gsize = Size(2, 4);
			else if (th <= 16) gsize = Size(4, 4);
			else if (th <= 32) gsize = Size(4, 8);
			else if (th <= 64) gsize = Size(8, 8);

			splitToGrid(src_, srcGrid, gsize, 2 * r);
			splitToGrid(guidance_, guideGrid, gsize, 2 * r);
		}

		void operator() (const Range& range) const
		{
			for (int i = range.start; i != range.end; i++)
			{
				if (!isGuide)
				{
					Mat s = srcGrid[i];
					guidedFilter(s, s, r, eps);
				}
				else
				{
					Mat s = srcGrid[i];
					Mat g = guideGrid[i];
					guidedFilter(s, g, s, r, eps);
					//guidedFilter(srcGrid[i], guideGrid[i], srcGrid[i], r,eps);
				}
			}
		}
	};

	void guidedFilterMultiCore(const Mat& src, Mat& dest, int r, float eps, int numcore)
	{
		int th = (numcore <= 0) ? cv::getNumThreads() : numcore;

		if (th == 1) th = 1;
		else if (th <= 2) th = 2;
		else if (th <= 4) th = 4;
		else if (th <= 8) th = 8;
		else if (th <= 16) th = 16;
		else if (th <= 32) th = 32;
		else if (th <= 64) th = 64;

		dest.create(src.size(), src.type());
		GuidedFilterInvoler body(src, dest, r, eps, th);
		parallel_for_(Range(0, th), body, th);
	}

	void guidedFilterMultiCore(const Mat& src, const Mat& guidance, Mat& dest, int r, float eps, int numcore)
	{
		int th = (numcore <= 0) ? cv::getNumThreads() : numcore;

		if (th == 1) th = 1;
		else if (th <= 2) th = 2;
		else if (th <= 4) th = 4;
		else if (th <= 8) th = 8;
		else if (th <= 16) th = 16;
		else if (th <= 32) th = 32;
		else if (th <= 64) th = 64;

		dest.create(src.size(), src.type());

		GuidedFilterInvoler body(src, guidance, dest, r, eps, th);
		parallel_for_(Range(0, th), body, th);
	}

	void guidedFilterSrc1GuidanceN_(const Mat& src, const Mat& guidance, Mat& dest, const int radius, const float eps)
	{
		int debug = 0;

		int channels = guidance.channels();
		vector<Mat> I(channels);
		vector<Mat> If(channels);

		split(guidance, I);

		const Size ksize(2 * radius + 1, 2 * radius + 1);
		const Point PT(-1, -1);
		const float e = eps;
		const Size imsize = src.size();
		const int size = src.size().area();
		const double div = 1.0 / ksize.area();

		Mat temp(imsize, CV_32F);

		for (int i = 0; i < channels; i++)
			If[i].create(imsize, CV_32F);


		if (src.type() == CV_8U)
		{
			cvt8u32f(src, temp, 1.f);
			for (int i = 0; i < channels; i++)
				cvt8u32f(I[i], If[i], 1.f);
		}
		else
		{
			src.convertTo(temp, CV_32F, 1.f);
			for (int i = 0; i < channels; i++)
				I[i].convertTo(If[i], CV_32F, 1.f);
		}

		vector<Mat> mean_I_vec(channels);
		for (int i = 0; i < channels; i++)
		{
			mean_I_vec[i].create(imsize, CV_32F);
			boxFilter(If[i], mean_I_vec[i], CV_32F, ksize, PT, true, BORDER_TYPE);
		}

		Mat mean_p(imsize, CV_32F);
		boxFilter(temp, mean_p, CV_32F, ksize, PT, true, BORDER_TYPE);

		vector<Mat> mean_Ip_vec(channels);
		for (int i = 0; i < channels; i++)
		{
			mean_Ip_vec[i].create(imsize, CV_32F);
			multiply(If[i], temp, mean_Ip_vec[i]);//Ir*p
			boxFilter(mean_Ip_vec[i], mean_Ip_vec[i], CV_32F, ksize, PT, true, BORDER_TYPE);
		}

		//cout<<"covariance computation"<<endl;

		vector<Mat> cov_Ip_vec(channels);
		for (int i = 0; i < channels; i++)
		{
			cov_Ip_vec[i] = mean_Ip_vec[i];
			multiply(mean_I_vec[i], mean_p, temp);
			cov_Ip_vec[i] -= temp;
		}

		//cout<<"variance computation"<<endl;

		vector<Mat> var_I_vec(channels * channels);
		for (int j = 0; j < channels; j++)
		{
			for (int i = 0; i < channels; i++)
			{
				int idx = channels * j + i;
				multiply(If[i], If[j], temp);
				boxFilter(temp, var_I_vec[idx], CV_32F, ksize, PT, true, BORDER_TYPE);
				multiply(mean_I_vec[i], mean_I_vec[j], temp);
				var_I_vec[idx] -= temp;
			}
		}

		for (int j = 0; j < channels; j++)
		{
			for (int i = 0; i < channels; i++)
			{
				if (i == j)
				{
					int idx = channels * j + i;
					var_I_vec[idx] += e;
				}
			}
		}

		{
			Mat sigmaEps = Mat::zeros(channels, channels, CV_32F);

			//CalcTime t("cov");
			for (int i = 0; i < size; i++)
			{
				for (int n = 0; n < channels * channels; n++)
				{
					sigmaEps.at<float>(n) = var_I_vec[n].at<float>(i);
				}

				Mat inv = sigmaEps.inv(cv::DECOMP_LU);

				//reuse for buffer
				Mat vec = Mat::zeros(channels, 1, CV_32F);

				for (int m = 0; m < channels; m++)
				{
					for (int n = 0; n < channels; n++)
					{
						int idx = channels * m + n;
						vec.at<float>(n) += inv.at<float>(m, n) * cov_Ip_vec[n].at<float>(i);
					}
				}

				for (int n = 0; n < channels; n++)
					cov_Ip_vec[n].at<float>(i) = vec.at<float>(n);

			}
		}

		vector<Mat> a_vec(channels);
		for (int i = 0; i < channels; i++)
		{
			a_vec[i] = cov_Ip_vec[i];
			multiply(a_vec[i], mean_I_vec[i], mean_I_vec[i]);//break mean_I_r;
		}

		Mat mean_vec = Mat::zeros(mean_p.size(), CV_32F);
		for (int i = 0; i < channels; i++)
		{
			mean_vec += mean_I_vec[i];
		}
		mean_p -= mean_vec;


		Mat b = mean_p;

		boxFilter(b, temp, CV_32F, ksize, PT, true, BORDER_TYPE);
		for (int i = 0; i < channels; i++)
		{
			boxFilter(a_vec[i], a_vec[i], CV_32F, ksize, PT, true, BORDER_TYPE);//break a_r
			multiply(a_vec[i], If[i], a_vec[i]);
			temp += a_vec[i];
		}
		temp.convertTo(dest, src.type());
	}

	void guidedFilterBase(const Mat& src, const Mat& guidance, Mat& dest, const int radius, const float eps)
	{
		if (src.channels() == 1 && guidance.channels() == 3)
		{
			guidedFilterSrc1Guidance3_(src, guidance, dest, radius, eps);
		}
		else if (src.channels() == 1 && guidance.channels() == 1)
		{
			guidedFilterSrc1Guidance1_(src, guidance, dest, radius, eps);
		}
		else if (src.channels() == 3 && guidance.channels() == 3)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			split(src, v);

			guidedFilterSrc1Guidance3_(v[0], guidance, d[0], radius, eps);
			guidedFilterSrc1Guidance3_(v[1], guidance, d[1], radius, eps);
			guidedFilterSrc1Guidance3_(v[2], guidance, d[2], radius, eps);

			merge(d, dest);
		}
		else if (src.channels() == 3 && guidance.channels() == 1)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			split(src, v);

			guidedFilterSrc1Guidance1_(v[0], guidance, d[0], radius, eps);
			guidedFilterSrc1Guidance1_(v[1], guidance, d[1], radius, eps);
			guidedFilterSrc1Guidance1_(v[2], guidance, d[2], radius, eps);

			merge(d, dest);
		}
	}
	void guidedFilterBase(const Mat& src, Mat& dest, const int radius, const float eps)
	{
		if (radius == 0) { src.copyTo(dest); return; }

		if (radius == 0)
			src.copyTo(dest);

		if (src.channels() == 1)
		{
			guidedFilterSrc1_(src, dest, radius, eps);

		}
		else if (src.channels() == 3)
		{
			vector<Mat> v(3);
			vector<Mat> d(3);
			split(src, v);
			guidedFilterSrc1Guidance3_(v[0], src, d[0], radius, eps);
			guidedFilterSrc1Guidance3_(v[1], src, d[1], radius, eps);
			guidedFilterSrc1Guidance3_(v[2], src, d[2], radius, eps);

			merge(d, dest);
		}
	}


	/*
	void guidedFilter_matlabconverted(Mat& src, Mat& joint,Mat& dest,const int radius,const double eps)
	{
	//direct

	if(src.channels()!=1 && joint.channels()!=1)
	{
	cout<<"Please input gray scale image."<<endl;
	return;
	}

	Size ksize(2*radius+1,2*radius+1);

	Mat mJoint;//mean_I
	Mat mSrc;//mean_p
	boxFilter(joint,mJoint,CV_32F,ksize,Point(-1,-1),true);
	boxFilter(src,mSrc,CV_32F,ksize,Point(-1,-1),true);

	Mat SxJ;//I.*p
	multiply(joint,src,SxJ,1.0,CV_32F);
	Mat mSxJ;//mean_Ip
	boxFilter(SxJ,mSxJ,CV_32F,ksize,Point(-1,-1),true);

	Mat mSxmJ;//mean_I.*mean_p
	multiply(mJoint,mSrc,mSxmJ,1.0,CV_32F);
	Mat covSxJ =mSxJ - mSxmJ;//cov_Ip

	Mat joint2;
	Mat mJointpow;
	Mat joint32;
	joint.convertTo(joint32,CV_32F);
	cv::pow(joint32,2.0,joint2);
	cv::pow(mJoint,2.0,mJointpow);

	Mat mJoint2;
	boxFilter(joint2,mJoint2,CV_32F,ksize,Point(-1,-1),true);//convert pow2&boxf32フィルタへ

	Mat vJoint = mJoint2 - mJointpow;

	const double e=eps*eps;
	vJoint = vJoint+e;

	Mat a;
	divide(covSxJ,vJoint,a);

	Mat amJoint;
	multiply(a,mJoint,amJoint,1.0,CV_32F);
	Mat b = mSrc - amJoint;

	Mat ma;
	Mat mb;
	boxFilter(a,ma,CV_32F,ksize,Point(-1,-1),true);
	boxFilter(b,mb,CV_32F,ksize,Point(-1,-1),true);

	Mat maJoint;
	multiply(ma,joint32,maJoint,1.0,CV_32F);
	Mat temp = maJoint+mb;

	temp.convertTo(dest,dest.type());
	}
	*/
	/*
	void guidedFilterSrcGrayGuidanceColorNonop(Mat& src, Mat& guidance, Mat& dest, const int radius,const float eps)
	{
	if(src.channels()!=1 && guidance.channels()!=3)
	{
	cout<<"Please input gray scale image as src, and color image as guidance."<<endl;
	return;
	}
	vector<Mat> I;
	split(guidance,I);
	const Size ksize(2*radius+1,2*radius+1);
	const Point PT(-1,-1);
	const double e=eps*eps;
	const int size = src.size().area();

	Mat mean_I_r;
	Mat mean_I_g;
	Mat mean_I_b;

	Mat mean_p;

	//cout<<"mean computation"<<endl;
	boxFilter(I[0],mean_I_r,CV_32F,ksize,PT,true);
	boxFilter(I[1],mean_I_g,CV_32F,ksize,PT,true);
	boxFilter(I[2],mean_I_b,CV_32F,ksize,PT,true);

	boxFilter(src,mean_p,CV_32F,ksize,PT,true);

	Mat mean_Ip_r;
	Mat mean_Ip_g;
	Mat mean_Ip_b;
	multiply(I[0],src,mean_Ip_r,1.0,CV_32F);//Ir*p
	boxFilter(mean_Ip_r,mean_Ip_r,CV_32F,ksize,PT,true);
	multiply(I[1],src,mean_Ip_g,1.0,CV_32F);//Ig*p
	boxFilter(mean_Ip_g,mean_Ip_g,CV_32F,ksize,PT,true);
	multiply(I[2],src,mean_Ip_b,1.0,CV_32F);//Ib*p
	boxFilter(mean_Ip_b,mean_Ip_b,CV_32F,ksize,PT,true);

	//cout<<"covariance computation"<<endl;
	Mat cov_Ip_r;
	Mat cov_Ip_g;
	Mat cov_Ip_b;
	multiply(mean_I_r,mean_p,cov_Ip_r,-1.0,CV_32F);
	cov_Ip_r += mean_Ip_r;
	multiply(mean_I_g,mean_p,cov_Ip_g,-1.0,CV_32F);
	cov_Ip_g += mean_Ip_g;
	multiply(mean_I_b,mean_p,cov_Ip_b,-1.0,CV_32F);
	cov_Ip_b += mean_Ip_b;



	//cout<<"variance computation"<<endl;
	Mat temp;
	Mat var_I_rr;
	multiply(I[0],I[0],temp,1.0,CV_32F);
	boxFilter(temp,var_I_rr,CV_32F,ksize,PT,true);
	multiply(mean_I_r,mean_I_r,temp,1.0,CV_32F);
	var_I_rr-=temp;

	var_I_rr+=eps;

	Mat var_I_rg;
	multiply(I[0],I[1],temp,1.0,CV_32F);
	boxFilter(temp,var_I_rg,CV_32F,ksize,PT,true);
	multiply(mean_I_r,mean_I_g,temp,1.0,CV_32F);
	var_I_rg-=temp;
	Mat var_I_rb;
	multiply(I[0],I[2],temp,1.0,CV_32F);
	boxFilter(temp,var_I_rb,CV_32F,ksize,PT,true);
	multiply(mean_I_r,mean_I_b,temp,1.0,CV_32F);
	var_I_rb-=temp;
	Mat var_I_gg;
	multiply(I[1],I[1],temp,1.0,CV_32F);
	boxFilter(temp,var_I_gg,CV_32F,ksize,PT,true);
	multiply(mean_I_g,mean_I_g,temp,1.0,CV_32F);
	var_I_gg-=temp;

	var_I_gg+=eps;

	Mat var_I_gb;
	multiply(I[1],I[2],temp,1.0,CV_32F);
	boxFilter(temp,var_I_gb,CV_32F,ksize,PT,true);
	multiply(mean_I_g,mean_I_b,temp,1.0,CV_32F);
	var_I_gb-=temp;
	Mat var_I_bb;
	multiply(I[2],I[2],temp,1.0,CV_32F);
	boxFilter(temp,var_I_bb,CV_32F,ksize,PT,true);
	multiply(mean_I_b,mean_I_b,temp,1.0,CV_32F);
	var_I_bb-=temp;

	var_I_bb+=eps;


	float* rr = var_I_rr.ptr<float>(0);
	float* rg = var_I_rg.ptr<float>(0);
	float* rb = var_I_rb.ptr<float>(0);
	float* gg = var_I_gg.ptr<float>(0);
	float* gb = var_I_gb.ptr<float>(0);
	float* bb = var_I_bb.ptr<float>(0);

	Mat a_r = Mat::zeros(src.size(),CV_32F);
	Mat a_g = Mat::zeros(src.size(),CV_32F);
	Mat a_b = Mat::zeros(src.size(),CV_32F);
	float* ar = a_r.ptr<float>(0);
	float* ag = a_g.ptr<float>(0);
	float* ab = a_b.ptr<float>(0);
	float* covr = cov_Ip_r.ptr<float>(0);
	float* covg = cov_Ip_g.ptr<float>(0);
	float* covb = cov_Ip_b.ptr<float>(0);

	for(int i=0;i<size;i++)
	{
	//逆行列無しで3ms

	Matx33f sigmaEps
	(
	rr[i],rg[i],rb[i],
	rg[i],gg[i],gb[i],
	rb[i],gb[i],bb[i]
	);

	//Matx33f inv = sigmaEps.inv(cv::DECOMP_CHOLESKY);
	Matx33f inv = sigmaEps.inv(cv::DECOMP_LU);
	//Matx33f inv2 = sigmaEps.inv(cv::DECOMP_LU);

	ar[i]= covr[i]*inv(0,0) + covg[i]*inv(1,0) + covb[i]*inv(2,0);
	ag[i]= covr[i]*inv(0,1) + covg[i]*inv(1,1) + covb[i]*inv(2,1);
	ab[i]= covr[i]*inv(0,2) + covg[i]*inv(1,2) + covb[i]*inv(2,2);
	}

	multiply(a_r,mean_I_r,mean_I_r,-1.0,CV_32F);//break mean_I_r;
	multiply(a_g,mean_I_g,mean_I_g,-1.0,CV_32F);//break mean_I_g;
	multiply(a_b,mean_I_b,mean_I_b,-1.0,CV_32F);//break mean_I_b;
	Mat b = mean_p + mean_I_r+ mean_I_g + mean_I_b;

	boxFilter(a_r,a_r,CV_32F,ksize,PT,true);//break a_r
	boxFilter(a_g,a_g,CV_32F,ksize,PT,true);//break a_g
	boxFilter(a_b,a_b,CV_32F,ksize,PT,true);//break a_b

	boxFilter(b,temp,CV_32F,ksize,PT,true);
	multiply(a_r,I[0],a_r,1.0,CV_32F);
	multiply(a_g,I[1],a_g,1.0,CV_32F);
	multiply(a_b,I[2],a_b,1.0,CV_32F);
	temp = temp + a_r + a_g + a_b;

	temp.convertTo(dest,src.type());
	}
	*/
}