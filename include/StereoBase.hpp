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
	protected:
		cv::Mat guideImage;//for aggregation
		std::vector<cv::Mat> target;//0: image, 1: Sobel
		std::vector<cv::Mat> reference;//0: image, 1: Sobel
		cv::Mat specklebuffer;

		CrossBasedLocalFilter clf;//filtering function is thread safe.
		//cv::Ptr<GuidedImageFilter> gif;
		GuidedImageFilter* gif;
		const int thread_max;
		int color_distance = 2;
	
		int border;
		int numberOfDisparities;
		int minDisparity;
		std::vector<cv::Mat> DSI;
		//pre filter
		void computeGuideImageForAggregation(cv::Mat& input);
		void prefilter(cv::Mat& targetImage, cv::Mat& referenceImage);
		//pixel matching
		int PixelMatchingMethod = BT;

		int preFilterCap;//cap for prefilter
		int pixelMatchErrorCap;
		int costAlphaImageSobel;//0-100 alpha*image_err+(1-alpha)*Sobel_err
		void getPixelMatchingCost(const int d, cv::Mat& dest);

		int feedbackFunction = 0;
		int feedbackClip = 2;
		float feedbackAmp = 0.5;
		void addCostIterativeFeedback(cv::Mat& cost, const int current_disparity, const cv::Mat& disparity, const int functionType, const int clip, float amp);

		//under debug pixel matching
		//adhoc parameters for pixel error blending
		int sobelBlendMapParam_Size;
		int sobelBlendMapParam1;
		int sobelBlendMapParam2;
		void textureAlpha(cv::Mat& src, cv::Mat& dest, const int th1, const int th2, const int r);
		void getPixelMatchingCostADAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);
		void getPixelMatchingCostBTAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);
		
		//aggregation
		int AggregationMethod = Box;
		cv::Size aggregationShiftableKernel = cv::Size(3, 3);
		double aggregationGuidedfilterEps;
		double aggregationSigmaSpace;
		int aggregationRadiusH;
		int aggregationRadiusV;
		void getCostAggregation(cv::Mat& src, cv::Mat& dest, cv::InputArray joint = cv::noArray());

		//wta and optimization
		void getWTA(std::vector<cv::Mat>& dsi, cv::Mat& dest, cv::Mat& minimumCostMap);
		int P1;
		int P2;
		void getOptScanline();

		//post filters
		bool isUniquenessFilter = true;
		int uniquenessRatio;
		void uniquenessFilter(cv::Mat& costMap, cv::Mat& dest);
		void subpixelInterpolation(cv::Mat& dest, const int method);
		
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

	public:
		StereoBase(int blockSize, int minDisp, int disparityRange);
		~StereoBase();
		
		//PixelMatchingCost
		enum PixelMatching
		{
			SD,
			SDColor,
			SDEdge,
			SDEdgeColor,
			SDEdgeBlend,
			SDEdgeBlendColor,
			AD,
			ADColor,
			ADEdge,
			ADEdgeColor,
			ADEdgeBlend,
			ADEdgeBlendColor,
			BT,
			BTColor,
			BTEdge,
			BTEdgeColor,
			BTEdgeBlend,
			BTEdgeBlendColor,
			BTFull,
			BTFullColor,
			BTFullEdge,
			BTFullEdgeColor,
			BTFullEdgeBlend,
			BTFullEdgeBlendColor,
			CENSUS3x3,
			CENSUS3x3Color,
			CENSUS5x5,
			CENSUS5x5Color,
			CENSUS7x5,
			CENSUS7x5Color,
			CENSUS9x1,
			CENSUS9x1Color,
			//Pixel_Matching_SAD_TextureBlend,
			//Pixel_Matching_BT_TextureBlend,

			Pixel_Matching_Method_Size
		};
		std::string getPixelMatchingMethodName(int method);

		enum ColorDistance
		{
			ADD,
			AVG,
			MIN,
			MAX,
			ColorDistance_Size
		};
		std::string getColorDistanceName(ColorDistance method);
		void setPixelColorDistance(const ColorDistance method);

		enum Aggregation
		{
			Box,
			BoxShiftable,
			Gaussian,
			GaussShiftable,
			Guided,
			CrossBasedBox,
			Bilateral,

			Aggregation_Method_Size
		};
		std::string getAggregationMethodName(int method);

		enum
		{
			SUBPIXEL_NONE = 0,
			SUBPIXEL_QUAD,
			SUBPIXEL_LINEAR,

			SUBPIXEL_METHOD_SIZE
		};
		std::string getSubpixelInterpolationMethodName(const int method);

		//body
		void matching(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, const bool isFeedback = false);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest);
		void gui(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, StereoEval& eval);

		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::Mat& output, int mindis, int range);
		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::OutputArray output = cv::noArray());
	};
}