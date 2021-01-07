#pragma once

#include "common.hpp"
#include "parallel_type.hpp"
#include "boxFilter.hpp"
#include "checkSameImage.hpp"

namespace cp
{
	enum GuidedTypes
	{
		GUIDED_XIMGPROC,
		//--- Conventional Algorithm---
		GUIDED_NAIVE,
		GUIDED_NAIVE_SHARE,
		GUIDED_NAIVE_ONEPASS,
		GUIDED_SEP_VHI,
		GUIDED_SEP_VHI_SHARE,
		//--- Merge Algorithm --- 
	   // SSAT	
	   GUIDED_MERGE_AVX,
	   GUIDED_MERGE_TRANSPOSE_AVX,
	   GUIDED_MERGE_TRANSPOSE_INVERSE_AVX,
	   // SSAT	(cov reuse)	
	   GUIDED_MERGE_SHARE_AVX,
	   GUIDED_MERGE_SHARE_EX_AVX,
	   GUIDED_MERGE_SHARE_TRANSPOSE_AVX,
	   GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX,

	   GUIDED_MERGE,
	   GUIDED_MERGE_SSE,
	   GUIDED_MERGE_TRANSPOSE,
	   GUIDED_MERGE_TRANSPOSE_SSE,
	   GUIDED_MERGE_TRANSPOSE_INVERSE,
	   GUIDED_MERGE_TRANSPOSE_INVERSE_SSE,
	   GUIDED_MERGE_SHARE,
	   GUIDED_MERGE_SHARE_SSE,
	   GUIDED_MERGE_SHARE_EX,
	   GUIDED_MERGE_SHARE_EX_SSE,
	   GUIDED_MERGE_SHARE_TRANSPOSE,
	   GUIDED_MERGE_SHARE_TRANSPOSE_SSE,
	   GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE,
	   GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE,
	   // SSAT (BGR non-split)
	   GUIDED_NONSPLIT,
	   GUIDED_NONSPLIT_SSE,
	   GUIDED_NONSPLIT_AVX,
	   // OP-SAT
	   GUIDED_MERGE_ONEPASS,
	   GUIDED_MERGE_ONEPASS_2div,
	   GUIDED_MERGE_ONEPASS_SIMD,

	   // --- Fast Guided Filter --- 
	   GUIDED_MERGE_ONEPASS_FAST,

	   NumGuidedTypes	// num of guidedTypes. must be last element
	};
	/*
	enum GuidedTypes
	{
		GUIDED_XIMGPROC,
		 --- Conventional Algorithm ---
		GUIDED_CONV,
		GUIDED_CONV_SHARE,
		GUIDED_CONV_ONEPASS,

		 //--- Merge Algorithm ---
		// SSAT
		GUIDED_MERGE,
		GUIDED_MERGE_SSE,
		GUIDED_MERGE_AVX,
		GUIDED_MERGE_TRANSPOSE,
		GUIDED_MERGE_TRANSPOSE_SSE,
		GUIDED_MERGE_TRANSPOSE_AVX,
		GUIDED_MERGE_TRANSPOSE_INVERSE,
		GUIDED_MERGE_TRANSPOSE_INVERSE_SSE,
		GUIDED_MERGE_TRANSPOSE_INVERSE_AVX,
		// SSAT	(cov reuse)
		GUIDED_MERGE_SHARE,
		GUIDED_MERGE_SHARE_SSE,
		GUIDED_MERGE_SHARE_AVX,
		GUIDED_MERGE_SHARE_EX,
		GUIDED_MERGE_SHARE_EX_SSE,
		GUIDED_MERGE_SHARE_EX_AVX,
		GUIDED_MERGE_SHARE_TRANSPOSE,
		GUIDED_MERGE_SHARE_TRANSPOSE_SSE,
		GUIDED_MERGE_SHARE_TRANSPOSE_AVX,
		GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE,
		GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE,
		GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX,
		// SSAT (BGR non-split)
		GUIDED_NONSPLIT,
		GUIDED_NONSPLIT_SSE,
		GUIDED_NONSPLIT_AVX,
		// OP-SAT
		GUIDED_MERGE_ONEPASS,
		GUIDED_MERGE_ONEPASS_2div,
		GUIDED_MERGE_ONEPASS_SIMD,

		// --- Fast Guided Filter ---
		GUIDED_CONV_FAST,
		GUIDED_MERGE_ONEPASS_FAST,

		NumGuidedTypes	// num of guidedTypes. must be last element
	};*/

	inline std::string getGuidedType(const int guidedType)
	{
		std::string type;
		switch (guidedType)
		{
		case GUIDED_XIMGPROC:		type = "GUIDED_XIMGPROC";	break;
		case GUIDED_NAIVE:			type = "GUIDED_NAIVE";	break;
		case GUIDED_NAIVE_SHARE:	type = "GUIDED_NAIVE_SHARE";	break;

		case GUIDED_SEP_VHI:		type = "GUIDED_SEP_VHI";	break;
		case GUIDED_SEP_VHI_SHARE:		type = "GUIDED_SEP_VHI_SHARE";	break;

		case GUIDED_MERGE:			type = "GUIDED_MERGE";	break;
		case GUIDED_MERGE_SSE:		type = "GUIDED_MERGE_SSE";	break;
		case GUIDED_MERGE_AVX:		type = "GUIDED_MERGE_AVX";	break;
		case GUIDED_MERGE_TRANSPOSE:	type = "GUIDED_MERGE_TRANSPOSE";	break;
		case GUIDED_MERGE_TRANSPOSE_SSE:	type = "GUIDED_MERGE_TRANSPOSE_SSE";	break;
		case GUIDED_MERGE_TRANSPOSE_AVX:	type = "GUIDED_MERGE_TRANSPOSE_AVX";	break;
		case GUIDED_MERGE_TRANSPOSE_INVERSE:	type = "GUIDED_MERGE_TRANSPOSE_INVERSE";	break;
		case GUIDED_MERGE_TRANSPOSE_INVERSE_SSE:	type = "GUIDED_MERGE_TRANSPOSE_INVERSE_SSE";	break;
		case GUIDED_MERGE_TRANSPOSE_INVERSE_AVX:	type = "GUIDED_MERGE_TRANSPOSE_INVERSE_AVX";	break;
		case GUIDED_MERGE_SHARE:	type = "GUIDED_MERGE_SHARE";	break;
		case GUIDED_MERGE_SHARE_SSE:	type = "GUIDED_MERGE_SHARE_SSE";	break;
		case GUIDED_MERGE_SHARE_AVX:	type = "GUIDED_MERGE_SHARE_AVX";	break;
		case GUIDED_MERGE_SHARE_EX:	type = "GUIDED_MERGE_SHARE_EX";	break;
		case GUIDED_MERGE_SHARE_EX_SSE:	type = "GUIDED_MERGE_SHARE_EX_SSE";	break;
		case GUIDED_MERGE_SHARE_EX_AVX:	type = "GUIDED_MERGE_SHARE_EX_AVX";	break;
		case GUIDED_MERGE_SHARE_TRANSPOSE:	type = "GUIDED_MERGE_SHARE_TRANSPOSE";	break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_SSE:	type = "GUIDED_MERGE_SHARE_TRANSPOSE_SSE";	break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_AVX:	type = "GUIDED_MERGE_SHARE_TRANSPOSE_AVX";	break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE:	type = "GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE";	break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE:	type = "GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE";	break;
		case GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX:	type = "GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX";	break;

		case GUIDED_NONSPLIT:	type = "GUIDED_NONSPLIT";	break;
		case GUIDED_NONSPLIT_SSE:	type = "GUIDED_NONSPLIT_SSE";	break;
		case GUIDED_NONSPLIT_AVX:	type = "GUIDED_NONSPLIT_AVX";	break;

		case GUIDED_MERGE_ONEPASS:	type = "GUIDED_MERGE_ONEPATH";	break;
		case GUIDED_MERGE_ONEPASS_SIMD:	type = "GUIDED_MERGE_ONEPATH_SIMD";	break;
		case GUIDED_MERGE_ONEPASS_2div:	type = "GUIDED_MERGE_ONEPATH_2div";	break;

		case GUIDED_NAIVE_ONEPASS:	type = "GUIDED_NAIVE_ONEPATH";	break;
		case GUIDED_MERGE_ONEPASS_FAST:	type = "GUIDED_MERGE_ONEPATH_FAST";	break;

		default:	type = "UNDEFINED_METHOD";	break;
		}
		return type;
	}

	class GuidedFilterBase
	{
		bool isDebug = false;
		cp::CheckSameImage checkimage;//checker for filterGuidePrecomputed
		bool isComputeForReuseGuide = true;
	protected:
		cv::Mat src;
		cv::Mat guide;
		cv::Mat dest;

		std::vector<cv::Mat> vsrc;
		std::vector<cv::Mat> vguide;
		std::vector<cv::Mat> vdest;

		cv::Mat src_low;
		cv::Mat guide_low;
		std::vector<cv::Mat> vsrc_low;
		std::vector<cv::Mat> vguide_low;
		int downsample_method = cv::INTER_LINEAR;
		int upsample_method = cv::INTER_CUBIC;

		int r;
		float eps;

		int implementation = -1;
		virtual void computeVarCov();
	public:
		cv::Size size();
		int src_channels();
		int guide_channels();

		GuidedFilterBase(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps)
			: src(_src), guide(_guide), dest(_dest), r(_r), eps(_eps)
		{
			vsrc.resize(3);
			vdest.resize(3);
		}
		void setIsComputeForReuseGuide(const bool flag);
		void setDownsampleMethod(const int method);
		void setUpsampleMethod(const int method);

		int getImplementation();

		virtual void filter() = 0;
		virtual void filterVector() = 0;
		void filter(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps);
		void filterVector(std::vector<cv::Mat>& _src, std::vector <cv::Mat>& _guide, std::vector <cv::Mat>& _dest, int _r, float _eps);
		void filterFast(const int ratio);
		void filterFast(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, const int ratio);
		virtual void filterGuidePrecomputed();
		void filterGuidePrecomputed(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps);
		virtual void upsample();
		void upsample(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps);
		void upsample(cv::Mat& _src, cv::Mat& _guide_low, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps);
	};


	void CP_EXPORT guidedImageFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int r, const float eps, const GuidedTypes guidedType = GuidedTypes::GUIDED_SEP_VHI, const BoxFilterMethod boxType = BoxFilterMethod::OPENCV, const ParallelTypes parallelType = ParallelTypes::OMP);

	class CP_EXPORT GuidedImageFilter
	{
		cv::Mat srcImage;
		cv::Mat guideImage;
		cv::Mat destImage;

		cv::Mat guidelowImage;

		int downsample_method = cv::INTER_LINEAR;
		int upsample_method = cv::INTER_CUBIC;
		int parallel_type = 0;
		BoxFilterMethod box_type = BoxFilterMethod::OPENCV;

		cv::Size size = cv::Size(1, 1);

		std::vector<cv::Mat> vsrc;
		std::vector<cv::Mat> vdest;
		std::vector<cv::Ptr<GuidedFilterBase>> gf;
		cv::Ptr<GuidedFilterBase> getGuidedFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps, const int guided_type);
		bool initialize(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest);
	public:
		GuidedImageFilter()
		{
			size = cv::Size(1, 1);
			gf.resize(3);
		}

		void setIsComputeForReuseGuide(const bool flag);
		void setDownsampleMethod(const int method);
		void setUpsampleMethod(const int method);

		void setBoxType(const BoxFilterMethod type);

		void filter(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filterGuidePrecomputed(cv::Mat& _src, cv::Mat& _guide, cv::OutputArray _dest, int _r, float _eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filterColorParallel(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filterFast(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int ratio, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void upsample(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void upsample(cv::Mat& src, cv::Mat& guide_low, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);

		//for tiling parallelization
		void filter(cv::Mat& src, std::vector<cv::Mat>& guide, cv::Mat& dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filter(std::vector<cv::Mat>& src, cv::Mat& guide, std::vector<cv::Mat>& dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filter(std::vector<cv::Mat>& src, std::vector<cv::Mat>& guide, std::vector<cv::Mat>& dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void print_parameter();
	};

	class CP_EXPORT GuidedImageFilterTiling
	{
	protected:
		cv::Mat src;
		cv::Mat guide;
		cv::Mat dest;
		int r;
		float eps;
		int parallelType;

		cv::Size div = cv::Size(1, 1);

		std::vector<cv::Mat> vSrc;
		std::vector<cv::Mat> vGuide;
		std::vector<cv::Mat> vDest;

		std::vector<cv::Mat> src_sub_vec;
		std::vector<cv::Mat> guide_sub_vec;

		std::vector<cv::Mat> src_sub_b;
		std::vector<cv::Mat> src_sub_g;
		std::vector<cv::Mat> src_sub_r;

		std::vector<cv::Mat> dest_sub_b;
		std::vector<cv::Mat> dest_sub_g;
		std::vector<cv::Mat> dest_sub_r;

		std::vector<cv::Mat> src_sub_temp;
		std::vector<cv::Mat> guide_sub_temp;
		std::vector<cv::Mat> dest_sub_temp;

		std::vector<GuidedImageFilter> gf;
		std::vector<std::vector<cv::Mat>> sub_src;
		std::vector<std::vector<cv::Mat>> sub_guide;
		std::vector<cv::Mat> sub_guideColor;
		std::vector<std::vector<cv::Mat>> buffer;

	public:
		GuidedImageFilterTiling();
		GuidedImageFilterTiling(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, cv::Size _div);

		void filter_SSAT();
		void filter_OPSAT();
		void filter_SSAT_AVX();
		void filter_func(GuidedTypes guidedType);
		void filter(GuidedTypes guidedType);
		void filter(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, const int _r, const float _eps, const cv::Size _div, const GuidedTypes guidedType);
	};

	//under debugging
	class guidedFilter_tiling_noMakeBorder
	{
	private:
		cv::Mat src;
		cv::Mat guide;
		cv::Mat dest;
		int r;
		float eps;

		cv::Size div = cv::Size(1, 1);
		std::vector<cv::Mat> divSrc;
		std::vector<cv::Mat> divGuide;
		std::vector<cv::Mat> divDest;
		int divRow;
		int divCol;
		int padRow;
		int padCol;

		void splitImage();
		void mergeImage();
	public:
		guidedFilter_tiling_noMakeBorder(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, cv::Size div);

		void init();
		void filter();
	};

	/*
	//old function, do not recomend to use
	CP_EXPORT void guidedFilter(const cv::Mat& src, cv::Mat& dest, const int radius, const float eps);
	CP_EXPORT void guidedFilter(const cv::Mat& src, const cv::Mat& guidance, cv::Mat& dest, const int radius, const float eps);
	CP_EXPORT void guidedFilterMultiCore(const cv::Mat& src, cv::Mat& dest, int r, float eps, int numcore = 0);
	CP_EXPORT void guidedFilterMultiCore(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest, int r, float eps, int numcore = 0);
	*/
}