#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void moveXYZ(cv::InputArray xyz, cv::OutputArray dest, cv::InputArray R, cv::InputArray t, const bool isRotationThenTranspose = true);
	CP_EXPORT void reprojectXYZ(cv::InputArray depth, cv::OutputArray xyz, const double focalLength);
	CP_EXPORT void reprojectXYZ(cv::InputArray depth, cv::OutputArray xyz, cv::InputArray intrinsic, cv::InputArray distortion = cv::noArray());

	class CP_EXPORT PointCloudShow
	{
	private:
		bool isRotationThenTranspose;
		cv::Point pt;
		bool isInit;
		int x;
		int y;
		int z;
		int pitch;
		int roll;
		int yaw;

		int loolatx;
		int loolaty;

		int renderOpt;
		int viewSW;

		int br;
		int bth;

		int maxr;
		bool isDrawLine;
		bool isWrite;
		bool isLookat;
		cv::Point3d look;

		void depth2XYZ(cv::Mat& srcDepth, float focal);
		void depth2XYZ(cv::Mat& srcDepth, cv::InputArray srcK, cv::InputArray srcDist);
		void disparity2XYZ(cv::Mat& srcDisparity, float disp_amp, float focal, float baseline);
	public:
		cv::Mat renderingImage;
		cv::Mat xyz;

		PointCloudShow();

		void setIsRotationThenTranspose(bool flag);

		void loop(cv::InputArray image, cv::InputArray srcDisparity, const float disp_amp, const float focal, const float baseline, const int loopcount);
		void loopXYZ(cv::InputArray image, cv::InputArray xyzPoints, cv::InputArray K, cv::InputArray R, cv::InputArray t, int loopcount);
		void loopXYZMulti(cv::InputArray image, cv::InputArray xyzPoints, cv::InputArray K, cv::InputArray R, cv::InputArray t, int loopcount);
		void loopDepth(cv::InputArray image, cv::InputArray srcDepth, cv::InputArray K, cv::InputArray dist, int loopcount);
		void loopDepth(cv::InputArray image, cv::InputArray srcDepth, float focal, int loopcount);
		void loopDepth(cv::InputArray image, cv::InputArray srcDepth, cv::InputArray image2_, cv::InputArray srcDepth2_, float focal, cv::InputArray oR, cv::InputArray ot, cv::InputArray K, int loopcount);

		void renderingFromXYZ(cv::OutputArray dest, cv::InputArray image_, cv::InputArray xyz, cv::InputArray R, cv::InputArray t, cv::InputArray k);
		void renderingFromDepth(cv::OutputArray dest, cv::InputArray image, cv::InputArray srcDepth, cv::InputArray srcK, cv::InputArray srcDist, cv::InputArray R_, cv::InputArray t, cv::InputArray destK, cv::InputArray destDist);
		void renderingFromDepth(cv::OutputArray dest, cv::InputArray image, cv::InputArray srcDepth, const float focal, cv::InputArray R, cv::InputArray t);
		void renderingFromDisparity(cv::OutputArray dest, cv::InputArray image, cv::InputArray srcDisparity, const float disp_amp, const float focal, const float baseline, cv::InputArray R, cv::InputArray t);

	protected:
		std::string wname;
		void filterDepth(cv::InputArray src, cv::OutputArray dest);

	};
}