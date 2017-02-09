#pragma once

#include "common.hpp"
#include "depthEval.hpp"

namespace cp
{
	CP_EXPORT void correctDisparityBoundaryFillOcc(cv::Mat& src, cv::Mat& refimg, const int r, cv::Mat& dest);
	CP_EXPORT void correctDisparityBoundary(cv::Mat& src, cv::Mat& refimg, const int r, const int edgeth, cv::Mat& dest, const int secondr = 0, const int minedge = 0);

	class CP_EXPORT StereoBMSimple
	{
		cv::Mat bufferGray;
		cv::Mat bufferGray1;
		cv::Mat bufferGray2;
		cv::Mat bufferGray3;
		cv::Mat bufferGray4;
		cv::Mat bufferGray5;
		void shiftImage(cv::Mat& src, cv::Mat& dest, const int shift);

		std::vector<cv::Mat> target;
		std::vector<cv::Mat> refference;
		cv::Mat specklebuffer;
	public:
		int border;


		int sobelAlpha;
		int prefSize;
		int prefParam;
		int prefParam2;
		int preFilterCap;

		int uniquenessRatio;
		int SADWindowSize;
		int SADWindowSizeH;

		int numberOfDisparities;
		int minDisparity;
		int error_truncate;
		int disp12diff;

		int speckleWindowSize;
		int speckleRange;
		double eps;
		std::vector<cv::Mat> DSI;

		bool isProcessLBorder;
		bool isMinCostFilter;
		bool isBoxSubpix;
		int subboxRange;
		int subboxWindowR;
		cv::Mat minCostMap;

		bool isBT;
		int P1;
		int P2;

		cv::Mat costMap;
		cv::Mat weightMap;

		StereoBMSimple(int blockSize, int minDisp, int disparityRange);
		void StereoBMSimple::imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::Mat& output, int mindis, int range);

		void prefilter(cv::Mat& src1, cv::Mat& src2);
		void preFilter(cv::Mat& src, cv::Mat& dest, int param);

		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::OutputArray output = cv::noArray());

		//void getMatchingCostSADandSobel(vector<Mat>& target, vector<Mat>& refference, const int d,Mat& dest);
		void textureAlpha(cv::Mat& src, cv::Mat& dest, const int th1, const int th2, const int r);
		void getMatchingCostSADAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);
		void getMatchingCostSAD(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, const int d, cv::Mat& dest);
		void halfPixel(cv::Mat& src, cv::Mat& srcp, cv::Mat& srcm);
		void getMatchingCostBT(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, const int d, cv::Mat& dest);
		void getMatchingCostBTAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);

		void getOptScanline();
		void getMatchingCost(const int d, cv::Mat& dest);
		void getCostAggregationBM(cv::Mat& src, cv::Mat& dest, int d);
		void getCostAggregation(cv::Mat& src, cv::Mat& dest, cv::InputArray joint = cv::noArray());
		void getWTA(std::vector<cv::Mat>& dsi, cv::Mat& dest);

		void refineFromCost(cv::Mat& src, cv::Mat& dest);
		void getWeightUniqness(cv::Mat& disp);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest);
		void check(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, StereoEval& eval);

		//post filter
		void uniquenessFilter(cv::Mat& costMap, cv::Mat& dest);
		enum
		{
			SUBPIXEL_NONE = 0,
			SUBPIXEL_QUAD,
			SUBPIXEL_LINEAR
		};
		int subpixMethod;
		void subpixelInterpolation(cv::Mat& dest, int method);

		void fastLRCheck(cv::Mat& costMap, cv::Mat& dest);
		void fastLRCheck(cv::Mat& dest);
		void minCostFilter(cv::Mat& costMap, cv::Mat& dest);
	};
}