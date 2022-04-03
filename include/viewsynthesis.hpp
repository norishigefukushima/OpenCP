#pragma once

#include "common.hpp"

namespace cp
{
	enum
	{
		FILL_OCCLUSION_LINE = 0,
		FILL_OCCLUSION_REFLECT = 1,
		FILL_OCCLUSION_STRETCH = -1,
		FILL_OCCLUSION_HV = 2,
		FILL_OCCLUSION_INPAINT_NS = 3, // OpenCV Navier-Stokes algorithm
		FILL_OCCLUSION_INPAINT_TELEA = 4, // OpenCV A. Telea algorithm
	};
	//depth map hole filling
	enum
	{
		FILL_OCCLUSION_IM_LINE = FILL_OCCLUSION_LINE,
		FILL_OCCLUSION_IM_REFLECT = FILL_OCCLUSION_REFLECT,
		FILL_OCCLUSION_IM_STRETCH = FILL_OCCLUSION_STRETCH,
		FILL_OCCLUSION_IM_HV = FILL_OCCLUSION_HV,
	};
	CP_EXPORT void fillOcclusionImDisp(cv::InputOutputArray im_, cv::InputOutputArray disp_, int invalidvalue, int mode = FILL_OCCLUSION_IM_LINE);
	CP_EXPORT void shiftDisp(const cv::Mat& srcdisp, cv::Mat& destdisp, float amp, float sub_gap, const int large_jump, cv::Mat& mask);
	CP_EXPORT void generateCenterDisparity(cv::InputArray disparityMapL, cv::InputArray disparityMapR, cv::OutputArray dest, const float amp, int depth = CV_32F, const int D = 7, const float thresh = 2);

	class CP_EXPORT DepthMapSubpixelRefinment
	{
		cv::Mat pslice;
		cv::Mat cslice;
		cv::Mat mslice;
		double calcReprojectionError(const cv::Mat& leftim, const cv::Mat& rightim, const cv::Mat& leftdisp, const cv::Mat& rightdisp, int disp_amp, bool left2right = true);

		template <class S, class srcType>
		void getDisparitySubPixel_Integer(cv::Mat& src, cv::Mat& dest, int disp_amp);
		void bluidCostSlice(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dest, int metric, int truncate);
	public:
		DepthMapSubpixelRefinment();
		void operator()(const cv::Mat& leftim, const cv::Mat& rightim, const cv::Mat& leftdisp, const cv::Mat& rightdisp, int disp_amp, cv::Mat& leftdest, cv::Mat& rightdest);
		void naive(const cv::Mat& leftim, const cv::Mat& rightim, const cv::Mat& leftdisp, const cv::Mat& rightdisp, int disp_amp, cv::Mat& leftdest, cv::Mat& rightdest);
	};

	class CP_EXPORT StereoViewSynthesis
	{

	private:
		void depthfilter(cv::Mat& depth, cv::Mat& depth2, cv::Mat& mask2, int viewstep, double disp_amp);
		template <class srcType>
		void analyzeSynthesizedViewDetail_(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp, cv::Mat& srcsynth, cv::Mat& ref);
		template <class srcType>
		void viewsynth(const cv::Mat& srcL, const cv::Mat& srcR, const cv::Mat& dispL, const cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
		template <class srcType>
		void makeMask_(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp);
		template <class srcType>
		void viewsynthSingle(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);

	public:
		//warping parameters
		enum
		{
			WAPR_IMG_INV = 0,//Mori et al.
			WAPR_IMG_FWD_SUB_INV, //Zenger et al.
		};
		int warpMethod;

		int warpInterpolationMethod;//Nearest, Linear or Cubic
		bool warpSputtering;
		int large_jump;

		//warped depth filtering parameters
		enum
		{
			DEPTH_FILTER_SPECKLE = 0,
			DEPTH_FILTER_MEDIAN,
			DEPTH_FILTER_MEDIAN_ERODE,
			DEPTH_FILTER_CRACK,
			DEPTH_FILTER_MEDIAN_BILATERAL,
			DEPTH_FILTER_NONE
		};
		int depthfiltermode;
		int warpedMedianKernel;

		int warpedSpeckesWindow;
		int warpedSpeckesRange;

		int bilateral_r;
		float bilateral_sigma_space;
		float bilateral_sigma_color;

		//blending parameter

		int blendMethod;
		double blend_z_thresh;

		//post filtering parameters
		enum
		{
			POST_GAUSSIAN_FILL = 0,
			POST_FILL,
			POST_NONE
		};
		int postFilterMethod;
		int inpaintMethod;

		double inpaintr;//parameter for opencv inpaint 
		int canny_t1;
		int canny_t2;

		cv::Size occBlurSize;

		cv::Size boundaryKernelSize;
		double boundarySigma;
		double boundaryGaussianRatio;

		//preset
		enum
		{
			PRESET_FASTEST = 0,
			PRESET_SLOWEST,
		};

		StereoViewSynthesis();
		StereoViewSynthesis(int preset);
		void init(int preset);

		void operator()(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
		void operator()(const cv::Mat& srcL, const cv::Mat& srcR, const cv::Mat& dispL, const cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);

		cv::Mat diskMask;
		cv::Mat allMask;//all mask
		cv::Mat boundaryMask;//disparity boundary
		cv::Mat nonOcclusionMask;
		cv::Mat occlusionMask;//half and full occlusion
		cv::Mat fullOcclusionMask;//full occlusion
		cv::Mat nonFullOcclusionMask; //bar of full occlusion
		cv::Mat halfOcclusionMask;//left and right half ooclusion

		void viewsynthSingleAlphaMap(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
		void alphaSynth(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
		void noFilter(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
		void analyzeSynthesizedViewDetail(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp, cv::Mat& srcsynth, cv::Mat& ref);
		void analyzeSynthesizedView(cv::Mat& srcsynth, cv::Mat& ref);
		void makeMask(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp);
		void makeMask(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, double alpha, int invalidvalue, double disp_amp, cv::Mat& srcsynth, cv::Mat& ref);

		void check(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, cv::Mat& ref);
		void check(cv::Mat& src, cv::Mat& disp, cv::Mat& dest, cv::Mat& destdisp, double alpha, int invalidvalue, double disp_amp, cv::Mat& ref);
		void preview(cv::Mat& srcL, cv::Mat& srcR, cv::Mat& dispL, cv::Mat& dispR, int invalidvalue, double disp_amp);
		void preview(cv::Mat& src, cv::Mat& disp, int invalidvalue, double disp_amp);
	};

}