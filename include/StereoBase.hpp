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
	public:
		StereoBase(int blockSize, int minDisp, int disparityRange);
		virtual ~StereoBase();

		//PixelMatchingCost
		enum Cost
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
			CENSUS13x3,
			CENSUS13x3Color,
			CENSUS7x3,
			CENSUS7x3Color,
			CENSUSTEST,
			CENSUSTESTColor,
			//Pixel_Matching_SAD_TextureBlend,
			//Pixel_Matching_BT_TextureBlend,

			Pixel_Matching_Method_Size
		};
		std::string getCostMethodName(const Cost method);
		void setCostMethod(const Cost method);

		enum ColorDistance
		{
			ADD,
			AVG,
			MIN,
			MAX,
			ColorDistance_Size
		};
		std::string getCostColorDistanceName(ColorDistance method);
		void setCostColorDistance(const ColorDistance method);

		enum Aggregation
		{
			Box,
			BoxShiftable,
			Gaussian,
			GaussianShiftable,
			Guided,
			CrossBasedBox,
			Bilateral,

			Aggregation_Method_Size
		};
		std::string getAggregationMethodName(const Aggregation method);
		void setAggregationMethod(const Aggregation method);

		enum class SUBPIXEL
		{
			NONE,
			QUAD,
			LINEAR,

			SUBPIXEL_METHOD_SIZE
		};
		std::string getSubpixelInterpolationMethodName(const SUBPIXEL method);
		void setSubpixelInterpolationMethodName(const SUBPIXEL method);
		//for internal or external: subpixel refiment using DSI outer postfilterd disparity map.
		void subpixelInterpolation(cv::Mat& disparity16, const SUBPIXEL method);

		enum HOLE_FILL
		{
			NONE,
			NEAREST_MIN_SCANLINE,
			METHOD2,
			METHOD3,
			METHOD4,

			FILL_OCCLUSION_SIZE
		};
		std::string getHollFillingMethodName(const HOLE_FILL method);
		void setHoleFiillingMethodName(const HOLE_FILL method);
		double getValidRatio();//get valid pixel ratio before hole filling

		enum class REFINEMENT
		{
			NONE,
			GIF_JNF,
			WGIF_GAUSS_JNF,
			WGIF_BFSUB_JNF,//under debug
			WGIF_BFW_JNF,//under debug
			WGIF_DUALBFW_JNF,//under debug
			JBF_JNF,
			WJBF_GAUSS_JNF,
			WMF,
			WWMF_GAUSS,

			REFINEMENT_SIZE
		};
		std::string getRefinementMethodName(const REFINEMENT method);
		//param<0: unchange parameter
		void setRefinementMethod(const REFINEMENT refinementMethod, const int refinementR = -1, const float refinementSigmaRange = -1.f, const float refinementSigmaSpace = -1.f, const int jointNearestR = -1);

		//body
		void matching(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, const bool isFeedback = false);
		void operator()(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest);
		void gui(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, StereoEval& eval);
		void gui(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& dest, StereoEval& eval, cv::Mat& dmapC);

		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::Mat& output, int mindis, int range);
		void imshowDisparity(std::string wname, cv::Mat& disp, int option, cv::OutputArray output = cv::noArray());
		void showWeightMap(std::string wname);
	protected:
		cv::Mat guideImage;//for aggregation
		std::vector<cv::Mat> target;//0: image, 1: Sobel
		std::vector<cv::Mat> reference;//0: image, 1: Sobel
		cv::Mat specklebuffer;

		CrossBasedLocalFilter clf;//filtering function is thread safe.
		GuidedImageFilter* gif;
		const int thread_max;
		int color_distance = 2;

		int border;
		int numberOfDisparities;
		int minDisparity;
		std::vector<cv::Mat> DSI;
		cv::Mat minCostMap;

		//pre filter
		int preFilterCap;//cap for prefilter
		void computeGuideImageForAggregation(cv::Mat& input);
		void computePrefilter(cv::Mat& targetImage, cv::Mat& referenceImage);

		//pixel matching
		int pixelMatchingMethod = Cost::SDEdgeBlend;
		int pixelMatchErrorCap;
		int costAlphaImageSobel;//0-100 alpha*image_err+(1-alpha)*Sobel_err
		void computePixelMatchingCost(const int d, cv::Mat& dest);

		int feedbackFunction = 2;
		int feedbackClip = 2;
		float feedbackAmp = 1.0;
		void addCostIterativeFeedback(cv::Mat& cost, const int current_disparity, const cv::Mat& disparity, const int functionType, const int clip, float amp);

		//under debug pixel matching
		//adhoc parameters for pixel error blending
		int sobelBlendMapParam_Size;
		int sobelBlendMapParam1;
		int sobelBlendMapParam2;
		void computetextureAlpha(cv::Mat& src, cv::Mat& dest, const int th1, const int th2, const int r);
		void computePixelMatchingCostADAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);
		void computePixelMatchingCostBTAlpha(std::vector<cv::Mat>& target, std::vector<cv::Mat>& refference, cv::Mat& alpha, const int d, cv::Mat& dest);

		//aggregation
		int aggregationMethod = Aggregation::Guided;
		cv::Size aggregationShiftableKernel = cv::Size(3, 3);
		double aggregationGuidedfilterEps;
		double aggregationSigmaSpace;
		int aggregationRadiusH;
		int aggregationRadiusV;
		void computeCostAggregation(cv::Mat& src, cv::Mat& dest, cv::InputArray joint = cv::noArray());

		//wta and optimization
		void computeWTA(std::vector<cv::Mat>& dsi, cv::Mat& dest, cv::Mat& minimumCostMap);
		int P1;
		int P2;
		void computeOptimizeScanline();

		//post filters
		bool isUniquenessFilter = true;
		int uniquenessRatio;
		void uniquenessFilter(cv::Mat& costMap, cv::Mat& dest);

		int subpixelInterpolationMethod;
		bool isRangeFilterSubpix = true;
		int subpixelRangeFilterCap;
		int subpixelRangeFilterWindow;
		enum class LRCHECK
		{
			NONE,
			WITH_MINCOST,
			WITHOUT_MINCOST,

			LRCHECK_SIZE
		};
		std::string getLRCheckMethodName(const LRCHECK method);
		void setLRCheckMethod(const LRCHECK method);
		int LRCheckMethod = (int)LRCHECK::WITH_MINCOST;
		bool isProcessLBorder = true;
		int disp12diff = 1;
		void fastLRCheck(cv::Mat& costMap, cv::Mat& dest);
		void fastLRCheck(cv::Mat& dest);

		bool isSpeckleFilter = true;
		int speckleWindowSize;
		int speckleRange;

		bool isMinCostFilter = false;
		uchar minCostThreshold = 4;
		void minCostSwapFilter(const cv::Mat& costMap, cv::Mat& dest);
		void minCostThresholdFilter(const cv::Mat& costMap, cv::Mat& dest, const uchar threshold);

		int holeFillingMethod = 1;
		void computeValidRatio(const cv::Mat& disparityMap);
		double valid_ratio = 0.0;

		int refinementMethod = (int)REFINEMENT::WGIF_GAUSS_JNF;
		int refinementR = 9;
		float refinementSigmaRange = 1.f;
		float refinementSigmaSpace = 255.f;
		float refinementSigmaHistogram = 32.f;
		int jointNearestR = 2;
		int refinementWeightMethod = 0;
		int refinementWeightR = 5;
		float refinementWeightSigma = 1.f;

		//under debug
		cv::Mat weightMap;
		void getWeightUniqness(cv::Mat& disp);
	};
}