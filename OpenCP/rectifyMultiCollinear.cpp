#include "opencp.hpp"
using namespace std;
using namespace cv;

namespace cp
{

static  void adjustTargetMatrix(const vector<vector<Point2f> >& refimgpt,
	const vector<vector<Point2f> >& destimgpt,
	const Mat& refcameraMatrix, const Mat& refdistCoeffs,
	const Mat& destcameraMatrix, const Mat& destdistCoeffs,
	const Mat& refR, const Mat& destR, const Mat& refP, Mat& destP)
{
	vector<Point2f> imgpt1, imgpt3;

	for (int i = 0; i < (int)min(refimgpt.size(), destimgpt.size()); i++)
	{
		if (!refimgpt[i].empty() && !destimgpt[i].empty())
		{
			std::copy(refimgpt[i].begin(), refimgpt[i].end(), std::back_inserter(imgpt1));
			std::copy(destimgpt[i].begin(), destimgpt[i].end(), std::back_inserter(imgpt3));
		}
	}

	undistortPoints(Mat(imgpt1), imgpt1, refcameraMatrix, refdistCoeffs, refR, refP);
	undistortPoints(Mat(imgpt3), imgpt3, destcameraMatrix, destdistCoeffs, destR, destP);

	double y1_ = 0, y2_ = 0, y1y1_ = 0, y1y2_ = 0;
	size_t n = imgpt1.size();

	for (size_t i = 0; i < n; i++)
	{
		double y1 = imgpt3[i].y, y2 = imgpt1[i].y;

		y1_ += y1; y2_ += y2;
		y1y1_ += y1*y1; y1y2_ += y1*y2;
	}

	y1_ /= n;
	y2_ /= n;
	y1y1_ /= n;
	y1y2_ /= n;

	double a = (y1y2_ - y1_*y2_) / (y1y1_ - y1_*y1_);
	double b = y2_ - a*y1_;

	destP.at<double>(0, 0) *= a;
	destP.at<double>(1, 1) *= a;
	destP.at<double>(0, 2) *= a;
	destP.at<double>(1, 2) = destP.at<double>(1, 2)*a + b;
	destP.at<double>(0, 3) *= a;
	destP.at<double>(1, 3) *= a;



}


float rectifyMultiCollinear(
	const vector<Mat>& cameraMatrix,
	const vector<Mat>& distCoeffs,
	const int anchorView1,
	const int anchorView2,
	const vector<vector<vector<Point2f>> >& anchorpt,
	Size imageSize, const vector<Mat>& relativeR, const vector<Mat>& relativeT,
	vector<Mat>& R, vector<Mat>& P, Mat& Q,
	double alpha, Size /*newImgSize*/,
	Rect* anchorROI1, Rect* anchorROI2, int flags)
{
	// first, rectify the 1-2 stereo pair
	stereoRectify(cameraMatrix[anchorView1], distCoeffs[anchorView1], cameraMatrix[anchorView2], distCoeffs[anchorView2],
		imageSize, relativeR[anchorView2], relativeT[anchorView2],
		R[anchorView1], R[anchorView2], P[anchorView1], P[anchorView2], Q, CALIB_ZERO_DISPARITY, alpha, imageSize, anchorROI1, anchorROI2);

	// recompute rectification transforms for cameras 1 & 2.

	vector<Mat> r_r1(cameraMatrix.size());
	for (int i = 0; i < (int)cameraMatrix.size(); i++)
	{
		if (i != anchorView1 && i != anchorView2)
		{
			if (relativeR[i].size() != Size(3, 3))
				Rodrigues(relativeR[i], r_r1[i]);
			else
				relativeR[i].copyTo(r_r1[i]);
		}
	}

	Mat om;
	if (relativeR[anchorView2].size() == Size(3, 3))
		Rodrigues(relativeR[anchorView2], om);
	else
		relativeR[anchorView2].copyTo(om);

	om *= -0.5;
	Mat r_r;
	Rodrigues(om, r_r); // rotate cameras to same orientation by averaging
	Mat_<double> t12 = r_r * relativeT[anchorView2];

	int idx = fabs(t12(0, 0)) > fabs(t12(1, 0)) ? 0 : 1;
	double c = t12(idx, 0), nt = norm(t12, CV_L2);
	Mat_<double> uu = Mat_<double>::zeros(3, 1);
	uu(idx, 0) = c > 0 ? 1 : -1;

	// calculate global Z rotation
	Mat_<double> ww = t12.cross(uu), wR;
	double nw = norm(ww, CV_L2);
	ww *= acos(fabs(c) / nt) / nw;
	Rodrigues(ww, wR);

	for (int i = 0; i < (int)cameraMatrix.size(); i++)
	{
		if (i != anchorView1 && i != anchorView2)
		{
			// now rotate camera 3 to make its optical axis parallel to cameras 1 and 2.
			R[i] = wR*r_r.t()*r_r1[i].t();
			Mat_<double> t13 = R[i] * relativeT[i];

			P[anchorView2].copyTo(P[i]);
			Mat t = P[i].col(3);
			t13.copyTo(t);
			P[i].at<double>(0, 3) *= P[i].at<double>(0, 0);
			P[i].at<double>(1, 3) *= P[i].at<double>(1, 1);

			//if( !anchorpt1.empty() && anchorpt2.empty() )
			adjustTargetMatrix(anchorpt[anchorView1], anchorpt[i], cameraMatrix[anchorView1], distCoeffs[anchorView1], cameraMatrix[i], distCoeffs[i], R[anchorView1], R[i], P[anchorView1], P[i]);
		}
	}
	return 0;
}
}
