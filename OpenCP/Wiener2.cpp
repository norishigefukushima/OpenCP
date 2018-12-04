/*
* cvWiener2 -- A Wiener 2D Filter implementation for OpenCV
*  Author: Ray Juang  / rayver {_at_} hkn {/_dot_/} berkeley (_dot_) edu
*  Date: 12.1.2006
*
* Modified 1.5.2007 (bug fix --
*   Forgot to subtract off local mean from local variance estimate.
*   (Credits to Kamal Ranaweera for the find)
*
* Modified 1.21.2007 (bug fix --
*   OpenCV's documentation claims that the default anchor for cvFilter2D is center of kernel.
*   This seems to be a lie -- the center has to be explicitly stated
*/

#include "wiener2.hpp"
using namespace cv;

namespace cp
{
//TODO: support OpenCV4
#if CV_MAJOR_VERSION <= 3
	void cvWiener2(const void* srcArr, void* dstArr, int szWindowX, int szWindowY)
	{
		CV_FUNCNAME("cvWiener2");

		szWindowX = cv::max(szWindowX, 1);
		szWindowY = cv::max(szWindowY, 1);

		int nRows;
		int nCols;
		CvScalar v = 0;
		CvMat *p_kernel = NULL;
		CvMat srcStub, *srcMat = NULL;
		CvMat *p_tmpMat1, *p_tmpMat2, *p_tmpMat3, *p_tmpMat4;
		double noise_power;


		//// DO CHECKING ////

		if (srcArr == NULL) {
			CV_ERROR(CV_StsNullPtr, "Source array null");
		}
		if (dstArr == NULL) {
			CV_ERROR(CV_StsNullPtr, "Dest. array null");
		}

		nRows = szWindowY;
		nCols = szWindowX;


		p_kernel = cvCreateMat(nRows, nCols, CV_32F);
		CV_CALL(cvSet(p_kernel, cvScalar(1.0 / (double)(nRows * nCols))));

		//Convert to matrices
		srcMat = (CvMat*)srcArr;

		if (!CV_IS_MAT(srcArr)) {
			CV_CALL(srcMat = cvGetMat(srcMat, &srcStub, 0, 1));
		}

		//Now create a temporary holding matrix
		p_tmpMat1 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
		p_tmpMat2 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
		p_tmpMat3 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
		p_tmpMat4 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));

		//Local mean of input
		cvFilter2D(srcMat, p_tmpMat1, p_kernel, cvPoint(nCols / 2, nRows / 2)); //localMean

		//Local variance of input
		cvMul(srcMat, srcMat, p_tmpMat2); //in^2
		cvFilter2D(p_tmpMat2, p_tmpMat3, p_kernel, cvPoint(nCols / 2, nRows / 2));

		//Subtract off local_mean^2 from local variance
		cvMul(p_tmpMat1, p_tmpMat1, p_tmpMat4); //localMean^2
		cvSub(p_tmpMat3, p_tmpMat4, p_tmpMat3); //filter(in^2) - localMean^2 ==> localVariance

		//Estimate noise power
		v = cvAvg(p_tmpMat3, 0);
		noise_power = v.val[0];
		// result = local_mean  + ( max(0, localVar - noise) ./ max(localVar, noise)) .* (in - local_mean)

		cvSub(srcMat, p_tmpMat1, dstArr); //in - local_mean
		cvMaxS(p_tmpMat3, noise_power, p_tmpMat2); //max(localVar, noise)

		cvAddS(p_tmpMat3, cvScalar(-noise_power), p_tmpMat3); //localVar - noise
		cvMaxS(p_tmpMat3, 0, p_tmpMat3); // max(0, localVar - noise)

		cvDiv(p_tmpMat3, p_tmpMat2, p_tmpMat3); //max(0, localVar-noise) / max(localVar, noise)

		cvMul(p_tmpMat3, dstArr, dstArr);
		cvAdd(dstArr, p_tmpMat1, dstArr);

		cvReleaseMat(&p_kernel);
		cvReleaseMat(&p_tmpMat1);
		cvReleaseMat(&p_tmpMat2);
		cvReleaseMat(&p_tmpMat3);
		cvReleaseMat(&p_tmpMat4);

	exit:
		;
	}
	void wiener2(cv::Mat&src, cv::Mat& dest, int szWindowX, int szWindowY)
	{
		if (szWindowY < 0)szWindowY = szWindowX;
		if (dest.empty())dest.create(src.size(), src.type());
		cvWiener2(&IplImage(src), &IplImage(dest), szWindowX, szWindowY);
	}
#endif
}