#pragma once

#include "common.hpp"
#include "stereoEval.hpp"
#include "crossBasedLocalFilter.hpp"
#include "guidedFilter.hpp"

namespace cp
{
	CP_EXPORT void correctDisparityBoundaryFillOcc(cv::Mat& src, cv::Mat& refimg, const int r, cv::Mat& dest);
	CP_EXPORT void correctDisparityBoundary(cv::Mat& src, cv::Mat& refimg, const int r, const int edgeth, cv::Mat& dest, const int secondr = 0, const int minedge = 0);

	class CP_EXPORT StereoBase
	{
		std::vector<cv::Mat> target;//0: image, 1: Sobel
		std::vector<cv::Mat> reference;//0: image, 1: Sobel
		cv::Mat specklebuffer;

		CrossBasedLocalFilter clf;//filtering function is thread safe.
		//cv::Ptr<GuidedImageFilter> gif;
		GuidedImageFilter* gif;
		const int thread_max;
	public:
		StereoBase(int blockSize, int minDisp, int disparityRange);
		~StereoBase();
		int border;
		int numberOfDisparities;
		int minDisparity;
		std::vector<cv::Mat> DSI;


		//PixelMatchingCost
		enum
		{
			Pixel_Matching_SD,
			Pixel_Matching_SDColor,
			Pixel_Matching_SDSobel,
			Pixel_Matching_SDSobelColor,
			Pixel_Matching_SDSobelBlend,
			Pixel_Matching_SDSobelBlendColor,
			Pixel_Matching_AD,
			Pixel_Matching_ADColor,
			Pixel_Matching_ADSobel,
			Pixel_Matching_ADSobelColor,
			Pixel_Matching_ADSobelBlend,
			Pixel_Matching_ADSobelBlendColor,
			Pixel_Matching_BT,
			Pixel_Matching_BTColor,
			Pixel_Matching_BTSobel,
			Pixel_Matching_BTSobelColor,
			Pixel_Matching_BTSobelBlend,
			Pixel_Matching_BTSobelBlendColor,
			Pixel_Matching_BTFull,
			Pixel_Matching_BTFullColor,
			Pixel_Matching_BTFullSobel,
			Pixel_Matching_BTFullSobelColor,
			Pixel_Matching_BTFullSobelBlend,
			Pixel_Matching_BTFullSobelBlendColor,
			Pixel_Matching_CENSUS3x3,
			Pixel_Matching_CENSUS3x3Color,
			Pixel_Matching_CENSUS5x5,
			Pixel_Matching_CENSUS5x5Color,
			Pixel_Matching_CENSUS7x5,
			Pixel_Matching_CENSUS7x5Color,
			Pixel_Matching_CENSUS9x1,
			Pixel_Matching_CENSUS9x1Color,
			//Pixel_Matching_SAD_TextureBlend,
			//Pixel_Matching_BT_TextureBlend,

			Pixel_Matching_Method_Size
		};
		int PixelMatchingMethod = Pixel_Matching_BT;

		int preFilterCap;//cap for prefilter
		int pixelMatchErrorCap;
		int costAlphaImageSobel;//0-100 alpha*image_err+(1-alpha)*Sobel_err
		//adhoc parameters for pixel error blending
		int sobelBlendMapParam_Size;
		int sobelBlendMapParam1;
		int sobelBlendMapParam2;
		std::string getPixelMatchingMethodName(int method);
		void prefilter(cv::Mat& targetImage, cv::Mat& referenceImage);

		void addCostIterativeFeedback(cv::Mat& cost, const int current_disparity, const cv::Mat& disparity, const int functionType, const int clip, float amp);
		//under debug
		void textureAlpha(cv::Mat& src, cv::Mat& dest, const int th1, const int th2, const int r);
		void getPixelMatchingCostADAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);
		void getPixelMatchingCostBTAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);

		enum
		{
			Aggregation_Box,
			Aggregation_BoxShiftable,
			Aggregation_Gauss,
			Aggregation_GaussShiftable,
			Aggregation_Guided,
			Aggregation_CrossBasedBox,
			Aggregation_Bilateral,

			Aggregation_Method_Size
		};
		std::string getAggregationMethodName(int method);
		int AggregationMethod = Aggregation_Box;

		cv::Size aggregationShiftableKernel = cv::Size(3, 3);
		double aggregationGuidedfilterEps;
		double aggregationSigmaSpace;

		int aggregationRadiusH;
		int aggregationRadiusV;

		int P1;
		int P2;

		bool isUniquenessFilter = true;
		int uniquenessRatio;
		void uniquenessFilter(cv::Mat& costMap, cv::Mat& dest);

		enum
		{
			SUBPIXEL_NONE = 0,
			SUBPIXEL_QUAD,
			SUBPIXEL_LINEAR,

			SUBPIXEL_METHOD_SIZE
		};
		void subpixelInterpolation(cv::Mat& dest, const int method);
		std::string getSubpixelInterpolationMethodName(const int method);
		int subpixelInterpolationMethod;
		bool isRangeFilterSubpix = true;
		int subpixelRangeFilterCap;
		int subpixelRangeFilterWindow;

		bool isLRCheck = true;
		bool isProcessLBorder = false;
		int disp12diff = 1;
		void fastLRCheck(cv::Mat& costMap, cv::Mat& dest);
		void fastLRCheck(cv::Mat& dest);

		bool isSpeckleFilter = true;
		int speckleWindowSize;
		int speckleRange;


		bool isMinCostFilter = false;
		void minCostFilter(cv::Mat& costMap, cv::Mat& dest);


		cv::Mat minCostMap;
		cv::Mat costMap;
		cv::Mat weightMap;
		void refineFromCost(cv::Mat& src, cv::Mat& dest);
		void getWeightUniqness(cv::Mat& disp);


		//internal of matching
		void getPixelMatchingCost(const int d, cv::Mat& dest);
		void getCostAggregation(cv::Mat& src, cv::Mat& dest, cv::InputArray joint = cv::noArray());
		void getWTA(std::vector<cv::Mat>& dsi, cv::Mat& dest, cv::Mat& minimumCostMap);
		void getOptScanline();


		//body
		void matching(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, const bool isFeedback = false);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest);
		void gui(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, StereoEval& eval);

		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::Mat& output, int mindis, int range);
		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::OutputArray output = cv::noArray());
	};
}