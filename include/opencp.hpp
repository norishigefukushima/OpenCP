#pragma once 

#include "common.hpp"

/*************************************************************
 filter
*************************************************************/
#include "boxFilter.hpp"
#include "bilateralFilter.hpp"
#include "binalyWeightedRangeFilter.hpp"
#include "boundaryReconstructionFilter.hpp"
#include "costVolumeFilter.hpp"
#include "crossBasedLocalFilter.hpp"
#include "crossBasedLocalMultipointFilter.hpp"
#include "domainTransformFilter.hpp"
#include "dualBilateralFilter.hpp"
#include "dualExponentialSmoothing.hpp"
#include "dxtDenoise.hpp"
#include "GaussianFilter.hpp"
#include "GaussianFilterSpectralRecursive.hpp"
#include "guidedFilter.hpp"
#include "jointBilateralFilter.hpp"
#include "jointDualBilateralFilter.hpp"
#include "jointNearestFilter.hpp"
#include "jointNonLocalMeans.hpp"
#include "L0Smoothing.hpp"
#include "minmaxfilter.hpp"
#include "nonLocalMeans.hpp"
#include "PermutohedralLattice.hpp"
#include "postFilterSet.hpp"
#include "realtimeO1BilateralFilter.hpp"
#include "recursiveBilateralFilter.hpp"
#include "shockFilter.hpp"
#include "weightedModeFilter.hpp"
#include "Wiener2.hpp"
#include "fftFilter.hpp"
#include "dctFilter.hpp"

//GaussianFilter
#include "GaussianBlurIPOL.hpp"
//upsapmle
#include "jointBilateralUpsample.hpp"


/*************************************************************
 imgproc
*************************************************************/
#include "alphaBlend.hpp"
#include "color.hpp"
#include "detailEnhancement.hpp"
#include "diffPixel.hpp"
#include "hazeRemove.hpp"
#include "iterativeBackProjection.hpp"
#include "metrics.hpp"
#include "ppmx.hpp"
#include "shiftImage.hpp"
#include "slic.hpp"
#include "speckle.hpp"
//qualitymetrics
#include "imqc.hpp"


/*************************************************************
 opticalflow
*************************************************************/
#include "opticalFlow.hpp"


/*************************************************************
 stereo
*************************************************************/
#include "Calibrator.hpp"
#include "depth2disparity.hpp"
#include "depthEval.hpp"
#include "depthfilter.hpp"
#include "disparityFitPlane.hpp"
#include "MultiCameraCalibrator.hpp"
#include "pointcloud.hpp"
#include "rectifyMultiCollinear.hpp"
#include "stereo_core.hpp"
#include "StereoBase.hpp"
#include "StereoBM2.hpp"
#include "StereoEx.hpp"
#include "StereoIterativeBM.hpp"
#include "StereoSGM2.hpp"
#include "stereoDisplay.hpp"
#include "stereoDP.hpp"
//view synthesis
#include "mattingRendering.hpp"
#include "viewsynthesis.hpp"


/*************************************************************
 utilty functions
*************************************************************/
#include "arithmetic.hpp"
#include "bitconvert.hpp"
#include "consoleImage.hpp"
#include "countDenormalizedNumber.hpp"
#include "crop.hpp"
#include "csv.hpp"
#include "draw.hpp"
#include "getContrast.hpp"
#include "histogram.hpp"
#include "imagediff.hpp"
#include "maskoperation.hpp"
#include "matinfo.hpp"
#include "noise.hpp"
#include "plot.hpp"
#include "sse_util.hpp"
#include "stat.hpp"
#include "stencil.hpp"
#include "timer.hpp"
#include "tiling.hpp"
#include "updateCheck.hpp"
#include "video.hpp"
#include "yuvio.hpp"



/*************************************************************
 other
*************************************************************/
#include "fftinfo.hpp"
#include "fitPlane.hpp"

/*************************************************************
 inline functions
*************************************************************/
#include "inlineSIMDFunctions.hpp"
#include "inlineMathFunctions.hpp"


//template for new files
/*

#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{



}
*/


//template for parallel_for_
/*
class Template_ParallelBody : public cv::ParallelLoopBody
{
private:
	const cv::Mat* src;
	cv::Mat* dest;

	int parameter;

public:
	Template_ParallelBody(const cv::Mat& src, cv::Mat& dst, const int parameter)
		: src(&src), dest(&dst), parameter(parameter)
	{
	}

	void operator() (const cv::Range& range) const
	{
		const int width = src->cols;
		for (int y = range.start; y < range.end; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dest->at<uchar>(y, x) = src->at<uchar>(y, x);
			}
		}
	}
};
void parallel_template(Mat& src, Mat& dest, const int parameter)
{
	cv::parallel_for_
	(
		cv::Range(0, dest.rows),
		Template_ParallelBody(src, dest, parameter),
		8
	);
}
*/