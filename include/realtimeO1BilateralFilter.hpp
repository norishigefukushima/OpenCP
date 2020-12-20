#pragma once

#include "common.hpp"
#if 0
namespace cp
{
	class CP_EXPORT RealtimeO1BilateralFilter
	{
	protected:

		std::vector<cv::Mat> bgrid;//for presubsampling

		std::vector<cv::Mat> sub_range;
		std::vector<cv::Mat> normalize_sub_range;

		std::vector<uchar> bin2num;
		std::vector<uchar> idx;
		std::vector<float> a;

		int num_bin;
		int bin_depth;
		void createBin(cv::Size imsize, int num_bin, int channles);
		void disposeBin(int number_of_bin);

		double sigma_color;
		float CV_DECL_ALIGNED(16) color_weight_32F[256 * 3];
		double CV_DECL_ALIGNED(16) color_weight_64F[256 * 3];
		void setColorLUT(double sigma_color, int channlels);

		int normType;
		template <typename srcType, typename S>
		void splatting(const srcType* s, S* su, S* sd, const uchar* j, const uchar v, const int imageSize, const int channels);
		template <typename srcType, typename S>
		void splattingColor(const srcType* s, S* su, S* sd, const uchar* j, const uchar* v, const int imageSize, const int channels, const int type);

		double sigma_space;
		int radius;
		int filterK;
		int filter_type;
		void blurring(const cv::Mat& src, cv::Mat& dest);

		template <typename srcType, typename S>
		void bodySaveMemorySize_(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dest);
		template <typename srcType, typename S>
		void body_(const cv::Mat& src, const cv::Mat& joint, cv::Mat& dest);

		void body(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, bool save_memorySize);
	public:
		RealtimeO1BilateralFilter();
		~RealtimeO1BilateralFilter();
		void showBinIndex();//show bin index for debug
		void setBinDepth(int depth = CV_32F);
		enum
		{
			L1SQR,//norm for OpenCV's native Bilateral filter
			L1,
			L2
		};

		void setColorNorm(int norm = L1SQR);
		int downsampleSizeSplatting;
		int downsampleSizeBlurring;
		int downsampleMethod;
		int upsampleMethod;
		bool isSaveMemory;

		enum
		{
			FIR_SEPARABLE,
			IIR_AM,
			IIR_SR,
			IIR_Deriche,
			IIR_YVY,
		};

		void gaussIIR(cv::InputArray src, cv::OutputArray dest, float sigma_color, float sigma_space, int num_bin, int method, int K);
		void gaussIIR(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, float sigma_color, float sigma_space, int num_bin, int method, int K);
		void gaussFIR(cv::InputArray src, cv::OutputArray dest, int r, float sigma_color, float sigma_space, int num_bin);
		void gaussFIR(cv::InputArray src, cv::InputArray joint, cv::OutputArray dest, int r, float sigma_color, float sigma_space, int num_bin);
	};
}
#endif