#pragma once

#include "common.hpp"
#include "boxFilter.hpp"
#include "parallel_type.hpp"
//#include <opencv2/core.hpp>
//#include <string>
namespace cp
{
	enum class BoxFilterMethod
	{
		OPENCV,

		NAIVE,
		NAIVE_SSE,
		NAIVE_AVX,

		SEPARABLE_HV,
		SEPARABLE_HV_SSE,
		SEPARABLE_HV_AVX,
		SEPARABLE_VH_AVX,
		SEPARABLE_VHI,
		SEPARABLE_VHI_SSE,
		SEPARABLE_VHI_AVX,

		INTEGRAL,
		INTEGRAL_SSE,
		INTEGRAL_AVX,
		INTEGRAL_ONEPASS,
		INTEGRAL_ONEPASS_AREA,

		SSAT_HV,
		SSAT_HV_SSE,
		SSAT_HV_AVX,
		SSAT_HV_BLOCKING,
		SSAT_HV_BLOCKING_SSE,
		SSAT_HV_BLOCKING_AVX,
		SSAT_HV_4x4,
		SSAT_HV_8x8,
		SSAT_HV_ROWSUM_GATHER_SSE,
		SSAT_HV_ROWSUM_GATHER_AVX,
		SSAT_HtH,
		SSAT_HtH_SSE,
		SSAT_HtH_AVX,

		SSAT_VH,
		SSAT_VH_SSE,
		SSAT_VH_AVX,
		SSAT_VH_ROWSUM_GATHER_SSE,
		SSAT_VH_ROWSUM_GATHER_AVX,
		SSAT_VtV,
		SSAT_VtV_SSE,
		SSAT_VtV_AVX,

		OPSAT,
		OPSAT_2Div,
		OPSAT_nDiv,

		SIZE	// num of boxtypes. must be last element
	};

	enum BoxMultiTypes
	{
		// --- SSAT Space Parallel --- 
		BOX_MULTI_SSAT_AoS,
		BOX_MULTI_SSAT_AoS_SSE,
		BOX_MULTI_SSAT_AoS_AVX,
		BOX_MULTI_SSAT_SoA,
		BOX_MULTI_SSAT_SoA_SSE,
		BOX_MULTI_SSAT_SoA_AVX,
		// --- SSAT Channel Parallel --- 
		BOX_MULTI_SSAT_AoS_CN,
		BOX_MULTI_SSAT_AoS_SSE_CN,
		BOX_MULTI_SSAT_AoS_AVX_CN,
		BOX_MULTI_SSAT_SoA_CN,
		BOX_MULTI_SSAT_SoA_SSE_CN,
		BOX_MULTI_SSAT_SoA_AVX_CN,
		// --- OPSAT --- 
		BOX_MULTI_OPSAT_AoS,
		BOX_MULTI_OPSAT_AoS_SSE,
		BOX_MULTI_OPSAT_AoS_AVX,
		BOX_MULTI_OPSAT_SoA,
		BOX_MULTI_OPSAT_SoA_SSE,
		BOX_MULTI_OPSAT_SoA_AVX,

		NumBoxMultiTypes
	};

#define BOX_FILTER_BORDER_TYPE cv::BORDER_REPLICATE

	class BoxFilterBase : public cv::ParallelLoopBody
	{
	protected:
		cv::Mat& src;
		cv::Mat& dest;
		int r;
		int parallelType;

		float div;
		int row;
		int col;

		virtual void filter_naive_impl() = 0;
		virtual void filter_omp_impl() = 0;
		void operator()(const cv::Range& range) const override = 0;
	public:
		BoxFilterBase(cv::Mat& _src, cv::Mat& _dest, int _r, int _parallelType) : src(_src), dest(_dest), r(_r), parallelType(_parallelType)
		{
			div = 1.f / ((2 * r + 1) * (2 * r + 1));
			row = src.rows;
			col = src.cols;
		};
		virtual void filter();
	};

	inline void BoxFilterBase::filter()
	{
		if (parallelType == ParallelTypes::NAIVE)
		{
			filter_naive_impl();
		}
		else if (parallelType == ParallelTypes::OMP)
		{
			filter_omp_impl();
		}
		else if (parallelType == ParallelTypes::PARALLEL_FOR_)
		{
			cv::parallel_for_(cv::Range(0, row), *this, cv::getNumThreads() - 1);
		}
		else
		{

		}
	}

	CP_EXPORT cv::Ptr<BoxFilterBase> createBoxFilter(const BoxFilterMethod method, const cv::Mat& src, cv::Mat& dest, int r, int _parallelType);

	CP_EXPORT void boxFilter_64f(cv::Mat& src, cv::Mat& dest, const int r, const BoxFilterMethod boxType, const int parallelType);

	CP_EXPORT void boxFilter_32f(cv::Mat& src, cv::Mat& dest, int r, const BoxFilterMethod boxType, int parallelType);

	CP_EXPORT void boxFilter_8u(cv::Mat& src, cv::Mat& dest, int r, const BoxFilterMethod boxType, int parallelType);

	CP_EXPORT void boxFilter_multiChannel(cv::Mat& src, cv::Mat& dest, int r, int boxMultiType, int parallelType);

	inline std::string getBoxType(const BoxFilterMethod boxType)
	{
		std::string type;
		switch (boxType)
		{
		case BoxFilterMethod::OPENCV:					type = "OPENCV"; break;
		case BoxFilterMethod::NAIVE:					type = "NAIVE"; break;
		case BoxFilterMethod::NAIVE_SSE:				type = "NAIVE_SSE";	break;
		case BoxFilterMethod::NAIVE_AVX:				type = "NAIVE_AVX";	break;
			 
		case BoxFilterMethod::SEPARABLE_HV:				type = "SEPARABLE_HV";	break;
		case BoxFilterMethod::SEPARABLE_HV_SSE:			type = "SEPARABLE_HV_SSE";	break;
		case BoxFilterMethod::SEPARABLE_HV_AVX:			type = "SEPARABLE_HV_AVX";	break;
		case BoxFilterMethod::SEPARABLE_VH_AVX:			type = "SEPARABLE_VH_AVX"; break;
		case BoxFilterMethod::SEPARABLE_VHI:			type = "SEPARABLE_VHI";	break;
		case BoxFilterMethod::SEPARABLE_VHI_SSE:		type = "SEPARABLE_VHI_SSE";	break;
		case BoxFilterMethod::SEPARABLE_VHI_AVX:		type = "SEPARABLE_VHI_AVX";	break;
			 
		case BoxFilterMethod::INTEGRAL:					type = "INTEGRAL";	break;
		case BoxFilterMethod::INTEGRAL_SSE:				type = "INTEGRAL_SSE";	break;
		case BoxFilterMethod::INTEGRAL_AVX:				type = "INTEGRAL_AVX";	break;
		case BoxFilterMethod::INTEGRAL_ONEPASS:			type = "INTEGRAL_ONEPASS";	break;
		case BoxFilterMethod::INTEGRAL_ONEPASS_AREA:	type = "INTEGRAL_ONEPASS_AREA";	break;
			 
		case BoxFilterMethod::SSAT_HV:					type = "SSAT_HV";	break;
		case BoxFilterMethod::SSAT_HV_SSE:				type = "SSAT_HV_SSE";	break;
		case BoxFilterMethod::SSAT_HV_AVX:				type = "SSAT_HV_AVX";	break;
		case BoxFilterMethod::SSAT_HV_BLOCKING:			type = "SSAT_HV_CACHE_BLOCKING";	break;
		case BoxFilterMethod::SSAT_HV_BLOCKING_SSE:		type = "SSAT_HV_CACHE_BLOCKING_SSE";	break;
		case BoxFilterMethod::SSAT_HV_BLOCKING_AVX:		type = "SSAT_HV_CACHE_BLOCKING_AVX";	break;
		case BoxFilterMethod::SSAT_HtH:					type = "SSAT_HtH";	break;
		case BoxFilterMethod::SSAT_HtH_SSE:				type = "SSAT_HtH_SSE";	break;
		case BoxFilterMethod::SSAT_HtH_AVX:				type = "SSAT_HtH_AVX";	break;
		case BoxFilterMethod::SSAT_VH:					type = "SSAT_VH";	break;
		case BoxFilterMethod::SSAT_VH_SSE:				type = "SSAT_VH_SSE";	break;
		case BoxFilterMethod::SSAT_VH_AVX:				type = "SSAT_VH_AVX";	break;
			 
		case BoxFilterMethod::SSAT_HV_4x4:				type = "SSAT_4x4";	break;
		case BoxFilterMethod::SSAT_HV_8x8:				type = "SSAT_8x8";	break;
		case BoxFilterMethod::SSAT_VtV:					type = "SSAT_VtV";	break;
		case BoxFilterMethod::SSAT_VtV_SSE:				type = "SSAT_VtV_SSE";	break;
		case BoxFilterMethod::SSAT_VtV_AVX:				type = "SSAT_VtV_AVX";	break;
		case BoxFilterMethod::SSAT_HV_ROWSUM_GATHER_SSE:type = "SSAT_HV_ROWSUM_GATHER_SSE";	break;
		case BoxFilterMethod::SSAT_HV_ROWSUM_GATHER_AVX:type = "SSAT_HV_ROWSUM_GATHER_AVX";	break;
		case BoxFilterMethod::SSAT_VH_ROWSUM_GATHER_SSE:type = "SSAT_VH_ROWSUM_GATHER_SSE";	break;
		case BoxFilterMethod::SSAT_VH_ROWSUM_GATHER_AVX:type = "SSAT_VH_ROWSUM_GATHER_AVX";	break;
			 
		case BoxFilterMethod::OPSAT:					type = "OPSAT";	break;
		case BoxFilterMethod::OPSAT_2Div:				type = "OPSAT_2div";	break;
		case BoxFilterMethod::OPSAT_nDiv:				type = "OPSAT_ndiv";	break;

		default:										type = "UNDEFINED_BOX_FILTER_METHOD";	break;
		}
		return type;
	}

	inline std::string getBoxMultiType(int boxMultiType)
	{
		std::string type;
		switch (boxMultiType)
		{
		case BOX_MULTI_SSAT_AoS:		type = "BOX_MULTI_SSAT";	break;
		case BOX_MULTI_SSAT_AoS_SSE:	type = "BOX_MULTI_SSAT_SSE";	break;
		case BOX_MULTI_SSAT_AoS_AVX:	type = "BOX_MULTI_SSAT_AVX";	break;
		case BOX_MULTI_SSAT_SoA:		type = "BOX_MULTI_SSAT_SoA";	break;
		case BOX_MULTI_SSAT_SoA_SSE:	type = "BOX_MULTI_SSAT_SoA_SSE";	break;
		case BOX_MULTI_SSAT_SoA_AVX:	type = "BOX_MULTI_SSAT_SoA_AVX";	break;

		case BOX_MULTI_SSAT_AoS_CN:		type = "BOX_MULTI_SSAT_CN";	break;
		case BOX_MULTI_SSAT_AoS_SSE_CN:	type = "BOX_MULTI_SSAT_SSE_CN";	break;
		case BOX_MULTI_SSAT_AoS_AVX_CN:	type = "BOX_MULTI_SSAT_AVX_CN";	break;
		case BOX_MULTI_SSAT_SoA_CN:		type = "BOX_MULTI_SSAT_SoA_CN";	break;
		case BOX_MULTI_SSAT_SoA_SSE_CN:	type = "BOX_MULTI_SSAT_SoA_SSE_CN";	break;
		case BOX_MULTI_SSAT_SoA_AVX_CN:	type = "BOX_MULTI_SSAT_SoA_AVX_CN";	break;

		case BOX_MULTI_OPSAT_AoS:		type = "BOX_MULTI_OP-SAT_AoS";	break;
		case BOX_MULTI_OPSAT_AoS_SSE:	type = "BOX_MULTI_OP-SAT_AoS_SSE";	break;
		case BOX_MULTI_OPSAT_AoS_AVX:	type = "BOX_MULTI_OP-SAT_AoS_AVX";	break;
		case BOX_MULTI_OPSAT_SoA:		type = "BOX_MULTI_OP-SAT_SoA";	break;
		case BOX_MULTI_OPSAT_SoA_SSE:	type = "BOX_MULTI_OP-SAT_SoA_SSE";	break;
		case BOX_MULTI_OPSAT_SoA_AVX:	type = "BOX_MULTI_OP-SAT_SoA_AVX";	break;

		default:						type = "UNDEFINED_METHOD";	break;
		}
		return type;
	}
}