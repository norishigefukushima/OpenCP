#pragma once

#include "common.hpp"
//#include <opencv2/core.hpp>
//#include <string>
namespace cp
{
	enum BoxTypes
	{
		BOX_OPENCV,

		BOX_NAIVE,
		BOX_NAIVE_SSE,
		BOX_NAIVE_AVX,

		BOX_SEPARABLE_HV,
		BOX_SEPARABLE_HV_SSE,
		BOX_SEPARABLE_HV_AVX,
		BOX_SEPARABLE_VH_AVX,
		BOX_SEPARABLE_VHI,
		BOX_SEPARABLE_VHI_SSE,
		BOX_SEPARABLE_VHI_AVX,

		BOX_INTEGRAL,
		BOX_INTEGRAL_SSE,
		BOX_INTEGRAL_AVX,
		BOX_INTEGRAL_ONEPASS,
		BOX_INTEGRAL_ONEPASS_AREA,

		BOX_SSAT_HV,
		BOX_SSAT_HV_SSE,
		BOX_SSAT_HV_AVX,
		BOX_SSAT_HV_BLOCKING,
		BOX_SSAT_HV_BLOCKING_SSE,
		BOX_SSAT_HV_BLOCKING_AVX,
		BOX_SSAT_HV_4x4,
		BOX_SSAT_HV_8x8,
		BOX_SSAT_HV_ROWSUM_GATHER_SSE,
		BOX_SSAT_HV_ROWSUM_GATHER_AVX,
		BOX_SSAT_HtH,
		BOX_SSAT_HtH_SSE,
		BOX_SSAT_HtH_AVX,

		BOX_SSAT_VH,
		BOX_SSAT_VH_SSE,
		BOX_SSAT_VH_AVX,
		BOX_SSAT_VH_ROWSUM_GATHER_SSE,
		BOX_SSAT_VH_ROWSUM_GATHER_AVX,
		BOX_SSAT_VtV,
		BOX_SSAT_VtV_SSE,
		BOX_SSAT_VtV_AVX,

		BOX_OPSAT,
		BOX_OPSAT_2Div,
		BOX_OPSAT_nDiv,

		NumBoxTypes	// num of boxtypes. must be last element
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

	CP_EXPORT void boxFilter_64f(cv::Mat& src, cv::Mat& dest, const int r, const int boxType, const int parallelType);

	CP_EXPORT void boxFilter_32f(cv::Mat& src, cv::Mat& dest, int r, int boxType, int parallelType);

	CP_EXPORT void boxFilter_8u(cv::Mat& src, cv::Mat& dest, int r, int boxType, int parallelType);

	CP_EXPORT void boxFilter_multiChannel(cv::Mat& src, cv::Mat& dest, int r, int boxMultiType, int parallelType);

	inline std::string getBoxType(int boxType)
	{
		std::string type;
		switch (boxType)
		{
		case BOX_OPENCV:					type = "BOX_OPENCV"; break;
		case BOX_NAIVE:						type = "BOX_NAIVE"; break;
		case BOX_NAIVE_SSE:					type = "BOX_NAIVE_SSE";	break;
		case BOX_NAIVE_AVX:					type = "BOX_NAIVE_AVX";	break;

		case BOX_SEPARABLE_HV:				type = "BOX_SEPARABLE_HV";	break;
		case BOX_SEPARABLE_HV_SSE:			type = "BOX_SEPARABLE_HV_SSE";	break;
		case BOX_SEPARABLE_HV_AVX:			type = "BOX_SEPARABLE_HV_AVX";	break;
		case BOX_SEPARABLE_VH_AVX:			type = "BOX_SEPARABLE_VH_AVX"; break;
		case BOX_SEPARABLE_VHI:				type = "BOX_SEPARABLE_VHI";	break;
		case BOX_SEPARABLE_VHI_SSE:			type = "BOX_SEPARABLE_VHI_SSE";	break;
		case BOX_SEPARABLE_VHI_AVX:			type = "BOX_SEPARABLE_VHI_AVX";	break;

		case BOX_INTEGRAL:					type = "BOX_INTEGRAL";	break;
		case BOX_INTEGRAL_SSE:				type = "BOX_INTEGRAL_SSE";	break;
		case BOX_INTEGRAL_AVX:				type = "BOX_INTEGRAL_AVX";	break;
		case BOX_INTEGRAL_ONEPASS:			type = "BOX_INTEGRAL_ONEPASS";	break;
		case BOX_INTEGRAL_ONEPASS_AREA:		type = "BOX_INTEGRAL_ONEPASS_AREA";	break;

		case BOX_SSAT_HV:					type = "BOX_SSAT_HV";	break;
		case BOX_SSAT_HV_SSE:				type = "BOX_SSAT_HV_SSE";	break;
		case BOX_SSAT_HV_AVX:				type = "BOX_SSAT_HV_AVX";	break;
		case BOX_SSAT_HV_BLOCKING:			type = "BOX_SSAT_HV_CACHE_BLOCKING";	break;
		case BOX_SSAT_HV_BLOCKING_SSE:		type = "BOX_SSAT_HV_CACHE_BLOCKING_SSE";	break;
		case BOX_SSAT_HV_BLOCKING_AVX:		type = "BOX_SSAT_HV_CACHE_BLOCKING_AVX";	break;
		case BOX_SSAT_HtH:					type = "BOX_SSAT_HtH";	break;
		case BOX_SSAT_HtH_SSE:				type = "BOX_SSAT_HtH_SSE";	break;
		case BOX_SSAT_HtH_AVX:				type = "BOX_SSAT_HtH_AVX";	break;
		case BOX_SSAT_VH:					type = "BOX_SSAT_VH";	break;
		case BOX_SSAT_VH_SSE:				type = "BOX_SSAT_VH_SSE";	break;
		case BOX_SSAT_VH_AVX:				type = "BOX_SSAT_VH_AVX";	break;

		case BOX_SSAT_HV_4x4:				type = "BOX_SSAT_4x4";	break;
		case BOX_SSAT_HV_8x8:				type = "BOX_SSAT_8x8";	break;
		case BOX_SSAT_VtV:					type = "BOX_SSAT_VtV";	break;
		case BOX_SSAT_VtV_SSE:				type = "BOX_SSAT_VtV_SSE";	break;
		case BOX_SSAT_VtV_AVX:				type = "BOX_SSAT_VtV_AVX";	break;
		case BOX_SSAT_HV_ROWSUM_GATHER_SSE:	type = "BOX_SSAT_HV_ROWSUM_GATHER_SSE";	break;
		case BOX_SSAT_HV_ROWSUM_GATHER_AVX:	type = "BOX_SSAT_HV_ROWSUM_GATHER_AVX";	break;
		case BOX_SSAT_VH_ROWSUM_GATHER_SSE:	type = "BOX_SSAT_VH_ROWSUM_GATHER_SSE";	break;
		case BOX_SSAT_VH_ROWSUM_GATHER_AVX:	type = "BOX_SSAT_VH_ROWSUM_GATHER_AVX";	break;

		case BOX_OPSAT:						type = "BOX_OPSAT";	break;
		case BOX_OPSAT_2Div:				type = "BOX_OPSAT_2div";	break;
		case BOX_OPSAT_nDiv:				type = "BOX_OPSAT_ndiv";	break;

		default:							type = "UNDEFINED_METHOD";	break;
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