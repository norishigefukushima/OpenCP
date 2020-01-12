#include "boxFilter.hpp"

#include "boxFilter_Naive.h"
#include "boxFilter_Integral.h"
#include "boxFilter_SSAT_HV.h"
#include "boxFilter_SSAT_HtH.h"
#include "boxFilter_SSAT_VH.h"
#include "boxFilter_Integral_OnePass.h"
#include "boxFilter_OPSAT_AoS.h"
#include "boxFilter_OPSAT_SoA.h"
#include "boxFilter_OPSAT_AoS_2Div.h"
#include "boxFilter_OPSAT_SoA_2Div.h"

#include "boxFilter_SSAT_HV_4x4.h"
#include "boxFilter_SSAT_HV_8x8.h"
#include "boxFilter_Separable.h"

#include "boxFilter_SSAT_VtV.h"
#include "boxFilter_SSAT_RowSumGather.h"

#include "boxFilter_Separable_Interleave.h"
#include "boxFilter_SSAT_SoA.h"
#include "boxFilter_SSAT_HV_CacheBlocking.h"
#include <iostream>

void boxFilter_64f(cv::Mat& src, cv::Mat& dest, const int r, const int boxType, const int parallelType)
{
	cv::boxFilter(src, dest, CV_64F, cv::Size(2 * r + 1, 2 * r + 1), cv::Point(-1, -1), true, BOX_FILTER_BORDER_TYPE);
}

void boxFilter_32f(cv::Mat& src, cv::Mat& dest, int r, int boxType, int parallelType)
{
	switch (boxType)
	{
	case BOX_OPENCV:
	{
		cv::boxFilter(src, dest, CV_32F, cv::Size(2 * r + 1, 2 * r + 1), cv::Point(-1, -1), true, BOX_FILTER_BORDER_TYPE);
		break;
	}
	/* --- Naive --- */
	case BOX_NAIVE:
	{
		if (src.channels() == 1)
		{
			boxFilter_Naive_nonVec_Gray(src, dest, r, parallelType).filter();
		}
		else if (src.channels() == 3)
		{
			boxFilter_Naive_nonVec_Color(src, dest, r, parallelType).filter();
		}
		break;
	}

	case BOX_NAIVE_SSE:
	{
		if (src.channels() == 1)
		{
			boxFilter_Naive_SSE_Gray(src, dest, r, parallelType).filter();
		}
		else if (src.channels() == 3)
		{
			boxFilter_Naive_SSE_Color(src, dest, r, parallelType).filter();
		}
		break;
	}

	case BOX_NAIVE_AVX:
	{
		if (src.channels() == 1)
		{
			boxFilter_Naive_AVX_Gray(src, dest, r, parallelType).filter();
		}
		else if (src.channels() == 3)
		{
			boxFilter_Naive_AVX_Color(src, dest, r, parallelType).filter();
		}
		break;
	}


	/* --- Separable --- */
	case BOX_SEPARABLE_HV:
	{
		boxFilter_Separable_HV_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SEPARABLE_HV_SSE:
	{
		boxFilter_Separable_HV_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SEPARABLE_HV_AVX:
	{
		boxFilter_Separable_HV_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SEPARABLE_VH_AVX:
	{
		boxFilter_Separable_VH_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SEPARABLE_VHI:
	{
		boxFilter_Separable_VHI_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SEPARABLE_VHI_SSE:
	{
		boxFilter_Separable_VHI_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SEPARABLE_VHI_AVX:
	{
		boxFilter_Separable_VHI_AVX(src, dest, r, parallelType).filter();
		break;
	}


	/* --- Integral --- */
	case BOX_INTEGRAL:
	{
		boxFilter_Integral_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_INTEGRAL_SSE:
	{
		boxFilter_Integral_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_INTEGRAL_AVX:
	{
		boxFilter_Integral_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_INTEGRAL_ONEPASS:
	{
		boxFilter_Integral_OnePass(src, dest, r, NAIVE).filter();
		break;
	}

	case BOX_INTEGRAL_ONEPASS_AREA:
	{
		boxFilter_Integral_OnePass_Area(src, dest, r, NAIVE).filter();
		break;
	}


	/* --- Separable Summed Area Table --- */
	case BOX_SSAT_HV:
	{
		boxFilter_SSAT_HV_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_SSE:
	{
		boxFilter_SSAT_HV_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_AVX:
	{
		boxFilter_SSAT_HV_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_BLOCKING:
	{
		boxFilter_SSAT_HV_CacheBlock_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_BLOCKING_SSE:
	{
		boxFilter_SSAT_HV_CacheBlock_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_BLOCKING_AVX:
	{
		boxFilter_SSAT_HV_CacheBlock_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HtH:
	{
		boxFilter_SSAT_HtH_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HtH_SSE:
	{
		boxFilter_SSAT_HtH_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HtH_AVX:
	{
		boxFilter_SSAT_HtH_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VH:
	{
		boxFilter_SSAT_VH_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VH_SSE:
	{
		boxFilter_SSAT_VH_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VH_AVX:
	{
		boxFilter_SSAT_VH_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_4x4:
	{
		boxFilter_SSAT_HV_4x4(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_8x8:
	{
		boxFilter_SSAT_HV_8x8(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VtV:
	{
		boxFilter_SSAT_VtV_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VtV_SSE:
	{
		boxFilter_SSAT_VtV_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VtV_AVX:
	{
		boxFilter_SSAT_VtV_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_ROWSUM_GATHER_SSE:
	{
		//boxFilter_SSAT_RowSumSIMD_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_HV_ROWSUM_GATHER_AVX:
	{
		boxFilter_SSAT_HV_RowSumGather_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VH_ROWSUM_GATHER_SSE:
	{
		boxFilter_SSAT_VH_RowSumGather_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_SSAT_VH_ROWSUM_GATHER_AVX:
	{
		boxFilter_SSAT_VH_RowSumGather_AVX(src, dest, r, parallelType).filter();
		break;
	}


	/* --- One Pass Summed Area Table --- */
	case BOX_OPSAT:
	{
		boxFilter_OPSAT_AoS(src, dest, r, parallelType).filter();
		//boxFilter_OPSAT_OnePass_BGR(src, dest, r, parallelType).filter();
		//boxFilter_OPSAT_OnePass_BGRA(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_OPSAT_2Div:
	{
		boxFilter_OPSAT_AoS_2Div(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_OPSAT_nDiv:
	{
		boxFilter_OPSAT_AoS_nDiv(src, dest, r, parallelType, OMP_THREADS_MAX).filter();
		break;
	}

	}
}

void boxFilter_8u(cv::Mat& src, cv::Mat& dest, int r, int boxType, int parallelType)
{
	switch (boxType)
	{
	case BOX_INTEGRAL_ONEPASS:
	{
		boxFilter_Integral_OnePass_8u(src, dest, r, parallelType).filter();
		break;
	}
	case BOX_SSAT_HV:
	{
		boxFilter_SSAT_8u_nonVec(src, dest, r, parallelType).filter();
		break;
	}
	default:
	{
		std::cout << "undefined method for 8u" << std::endl;
	}
	}
}

void boxFilter_multiChannel(cv::Mat& src, cv::Mat& dest, int r, int boxMultiType, int parallelType)
{
	switch (boxMultiType)
	{
		// Separable Summed Area Table (Space Parallel)
	case BOX_MULTI_SSAT_AoS:
	{
		boxFilter_SSAT_HV_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_AoS_SSE:
	{
		boxFilter_SSAT_HV_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_AoS_AVX:
	{
		boxFilter_SSAT_HV_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_SoA:
	{
		boxFilter_SSAT_SoA_Space_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_SoA_SSE:
	{
		boxFilter_SSAT_SoA_Space_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_SoA_AVX:
	{
		boxFilter_SSAT_SoA_Space_AVX(src, dest, r, parallelType).filter();
		break;
	}


	// Separable Summed Area Table (Channel Parallel)
	case BOX_MULTI_SSAT_AoS_CN:
	{
		boxFilter_SSAT_Channel_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_AoS_SSE_CN:
	{
		boxFilter_SSAT_Channel_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_AoS_AVX_CN:
	{
		boxFilter_SSAT_Channel_AVX(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_SoA_CN:
	{
		boxFilter_SSAT_SoA_Channel_nonVec(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_SoA_SSE_CN:
	{
		boxFilter_SSAT_SoA_Channel_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_SSAT_SoA_AVX_CN:
	{
		boxFilter_SSAT_SoA_Channel_AVX(src, dest, r, parallelType).filter();
		break;
	}


	// One Pass Summed Area Table
	case BOX_MULTI_OPSAT_AoS:
	{
		boxFilter_OPSAT_AoS(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_OPSAT_AoS_SSE:
	{
		boxFilter_OPSAT_AoS_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_OPSAT_AoS_AVX:
	{
		boxFilter_OPSAT_AoS_AVX(src, dest, r, parallelType).filter();
		//boxFilter_Summed_OnePass_AoS_nDiv_AVX(src, dest, r, parallelType, 4).filter();
		break;
	}

	case BOX_MULTI_OPSAT_SoA:
	{
		boxFilter_OPSAT_SoA(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_OPSAT_SoA_SSE:
	{
		boxFilter_OPSAT_SoA_SSE(src, dest, r, parallelType).filter();
		break;
	}

	case BOX_MULTI_OPSAT_SoA_AVX:
	{
		boxFilter_OPSAT_SoA_AVX(src, dest, r, parallelType).filter();
		break;
	}
	}
}
