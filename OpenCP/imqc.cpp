// iq.cpp : Defines the entry point for the console application.
#include "imqc.hpp"
#include "libimq/imq.h"

using namespace cv;

namespace cp
{
	double calcImageQualityMetric(InputArray src_, InputArray target_, const int metric, const int boundingBox)
	{
		Mat src = src_.getMat();
		Mat target = target_.getMat();

		Mat im1;
		Mat im2;
		if (src.channels() == 3)
		{
			Mat temp;
			cvtColor(src, temp, COLOR_BGR2GRAY);
			Mat v;
			temp.convertTo(v, CV_32S);
			Mat(v(Rect(boundingBox, boundingBox, src.cols - 2 * boundingBox, src.rows - 2 * boundingBox))).copyTo(im1);
		}
		else
		{
			Mat v;
			src.convertTo(v, CV_32S);
			Mat(v(Rect(boundingBox, boundingBox, src.cols - 2 * boundingBox, src.rows - 2 * boundingBox))).copyTo(im1);
		}

		if (target.channels() == 3)
		{
			Mat temp;
			cvtColor(target, temp, COLOR_BGR2GRAY);
			Mat v;
			temp.convertTo(v, CV_32S);
			Mat(v(Rect(boundingBox, boundingBox, src.cols - 2 * boundingBox, src.rows - 2 * boundingBox))).copyTo(im2);
		}
		else
		{
			Mat v;
			target.convertTo(v, CV_32S);
			Mat(v(Rect(boundingBox, boundingBox, src.cols - 2 * boundingBox, src.rows - 2 * boundingBox))).copyTo(im2);
		}

		int* orig_img = im1.ptr<int>(0);
		int* comp_img = im2.ptr<int>(0);
		int PX1 = im1.cols;
		int PY1 = im1.rows;
		int BPP1 = 8;

		double REZ = 0.0;

		switch (metric)
		{
		default:
		case IQM_PSNR:
			REZ = DoPSNRY(orig_img, comp_img, PX1, PY1, BPP1);
			break;

		case IQM_MSE:
			REZ = DoMSEY(orig_img, comp_img, PX1, PY1, BPP1);
			break;

		case IQM_MSAD:
			REZ = DoMSADY(orig_img, comp_img, PX1, PY1, BPP1);
			break;

		case IQM_DELTA:
			REZ = DoDeltaY(orig_img, comp_img, PX1, PY1, BPP1);
			break;

		case IQM_SSIM:
			DoSSIMY(orig_img, comp_img, PX1, PY1, BPP1, false);
			break;

		case IQM_SSIM_FAST:
			DoSSIMY(orig_img, comp_img, PX1, PY1, BPP1, true);
			break;

		case IQM_SSIM_MODIFY:
			mDoSSIMY(orig_img, comp_img, PX1, PY1, BPP1, false);
			break;

		case IQM_SSIM_FASTMODIFY:
			mDoSSIMY(orig_img, comp_img, PX1, PY1, BPP1, true);
			break;

		case IQM_CWSSIM:
			//REZ = DoCW_SSIMY(orig_img, comp_img, PX1, PY1, BPP1, true);
			break;

		case IQM_CWSSIM_FAST:
			REZ = DoMS_SSIMY(orig_img, comp_img, PX1, PY1, BPP1, true);
			break;

		case IQM_MSSSIM:
			REZ = DoMS_SSIMY(orig_img, comp_img, PX1, PY1, BPP1, false);
			break;

		case IQM_MSSSIM_FAST:
			//REZ = DoCW_SSIMY(orig_img, comp_img, PX1, PY1, BPP1, false);
			break;
		}

		return REZ;
	}
}