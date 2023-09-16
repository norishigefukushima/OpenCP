#pragma once
#include "common.hpp"

namespace cp
{
	class CP_EXPORT LocalLUTUpsample
	{
	public:
		enum class BUILD_LUT
		{
			L2_MIN,
			L1_MIN,
			LInf_MIN,
			LInf2_MIN,
			FREQUENCY_MAX_WTA,
			FREQUENCY_MAX_DP,

			SIZE
		};

		enum class BOUNDARY
		{
			REPLICATE,
			MINMAX_OUTPUT,
			MINMAX0_255,
			LINEAR,
			LINEAR_LAST2,
			NO_INTERPOLATION,
			EXPERIMENT1,
			EXPERIMENT2,

			SIZE
		};

		enum class UPTENSOR
		{
			NEAREST,//box1
			BOX4,
			BOX16,
			BOX64,
			GAUSS4,
			GAUSS16,
			GAUSS64,
			LINEAR,
			CUBIC,
			BILATERAL16,
			BILATERAL64,
			BoxNxN,
			GaussNxN,
			LaplaceNxN,

			SIZE
		};

		void setBoundaryReplicateOffset(const int offset) { boundary_replicate_offset = offset; }
		void setTensorUpSigmaSpace(const float sigma) { tensor_up_sigma_space = sigma; }
		void setTensorUpSigmaRange(const float sigma) { tensor_up_sigma_range = sigma; }
		void setTensorUpCubic(const float alpha) { tensor_up_cubic_alpha = alpha; }
		void setTensorUpKernelSize(const int d) { tensor_up_kernel_size = d; }

		std::string getBuildingLUTMethod(const BUILD_LUT method);
		std::string getTensorUpsamplingMethod(const UPTENSOR method);
		std::string getBoundaryMethod(const BOUNDARY method);
		void upsample(cv::Mat& src_low, cv::Mat& dst_low, cv::Mat& src, cv::Mat& dst, const int r, const int lut_num = 128, const int lut_filter_radius = 0, const BUILD_LUT build_method = BUILD_LUT::L1_MIN, const UPTENSOR tensorup_method = UPTENSOR::BOX16, const BOUNDARY lut_boundary_method = BOUNDARY::LINEAR, const bool isUseOffsetMap = true);
		void guiLUT(cv::Mat& lowres_src, cv::Mat& highres_src, cv::Mat& highres_out, bool isWait = true, std::string wname = "LocalLUT");

	private:
		const bool useSoA = false;//false; AoS is faster
		cv::Size lowres_size;
		cv::Mat src_low_border;
		cv::Mat dst_low_border;

		cv::Mat LUT_TensorAoS_B;//AoS: Size(width*height), channels=lut_num (or gray case)
		cv::Mat LUT_TensorAoS_G;//AoS: Size(width*height), channels=lut_num
		cv::Mat LUT_TensorAoS_R;//AoS: Size(width*height), channels=lut_num
		std::vector<cv::Mat> LUT_TensorSoA_B;//AoS: Size(width*height), channels=lut_num (or gray case)
		std::vector<cv::Mat> LUT_TensorSoA_G;//AoS: Size(width*height), channels=lut_num
		std::vector<cv::Mat> LUT_TensorSoA_R;//AoS: Size(width*height), channels=lut_num

		cv::Mat offset_map;
		cv::Mat offset_map_buffer;
		int patch_radius = 0;//used for guiLUT
		int tensor_up_kernel_size = 8;
		float tensor_up_sigma_space = 2.f;
		float tensor_up_sigma_range = 30.f;
		float tensor_up_cubic_alpha = 1.f;//from -0.5 to 2
		float up_sampling_ratio_resolution;//up_sampling_ratio
		int boundary_replicate_offset = 0;
		// Write the graph of LUT with gnuplot
		void LUTgraph(const uchar* array_lut, const int lut_num, std::string gnuplotpath = "C:/gnuplot/bin/pgnuplot.exe");
		void createLUTTensor(const int width, const int height, const int lut_num);

		template<int lut_boundary_method, bool isSoA> void buildLocalLUTTensorDistanceMINInvoker(const int distance, const int lut_num, const int r, const int range_div, const int lut_filter_radius);
		void buildLocalLUTTensorDistanceMIN(const int distance, const int lut_num, const int r, const int range_div, const int lut_filter_radius, const BOUNDARY lut_boundary_method);
		void buildLocalLUTTensorFrequencyMaxWTA16U(const int lut_num, const int r, const int ratio);
		void buildLocalLUTTensorFrequencyMaxWTA8U(const int lut_num, const int r, const int range_div, const int lut_filter_radius, const BOUNDARY lut_boundary_method);
		void buildLocalLUTTensorFrequencyMaxDP(const int lut_num, const int r, const int ratio, const short dpcost = 1);

		void boxBlurLUT(uchar* srcdst_lut, uchar* lut_buff, const int lut_num, const int r);

		template<bool quantization> void _tensorUpNearestLinear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const bool isOffset = true);
		template<bool quantization> void _tensorUpConv4Linear(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const int lut_num, const bool isOffset = true);
		template<bool quantization> void _tensorUpConv16Linear(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const int lut_num, const bool isOffset = true);
		template<bool quantization> void _tensorUpConv64Linear(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const int lut_num, const bool isOffset = true);
		template<bool quantization, int scale> void _tensorUpConv64LinearScale(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const int lut_num, const bool isOffset = true);
		template<bool quantization, int scale> void _tensorUpConv64LinearLoadScale(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const int lut_num, const bool isOffset = true);
		template<bool quantization> void _tensorUpConv64LinearSoA(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const int lut_num, const bool isOffset = true);

		template<bool quantization> void _tensorUpBilateralConv16Linear(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const float sigma_range, const int lut_num, const bool isOffset = true);
		template<bool quantization> void _tensorUpBilateralConv64Linear(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& weightmap, const float sigma_range, const int lut_num, const bool isOffset = true);
		void _tensorUpConvNxNLinearNaive(const cv::Mat& src_highres, cv::Mat& dst, const cv::Mat& spaceweight);

		void tensorUpNearestLinear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const bool isOffset = true);

		void tensorUpBox4Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const bool isOffset = true);
		void tensorUpBox16Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const bool isOffset = true);
		void tensorUpBox64Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const bool isOffset = true);

		void tensorUpGauss4Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const float sigma, const bool isOffset = true);
		void tensorUpGauss16Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const float sigma, const bool isOffset = true);
		void tensorUpGauss64Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const float sigma, const bool isOffset = true);

		void tensorUpTriLinear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const bool isOffset = true);
		void tensorUpBiCubicLinear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const float alpha, const bool isOffset = true);

		void tensorUpBoxNxNLinear(const cv::Mat& src_highres, cv::Mat& dst, const int d);
		void tensorUpGaussNxNLinear(const cv::Mat& src_highres, cv::Mat& dst, const int d, const float sigma);
		void tensorUpLaplaceNxNLinear(const cv::Mat& src_highres, cv::Mat& dst, const int d, const float sigma);

		void tensorUpBilateral16Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const float sigma_space, const float sigma_range, const bool isOffset = true);
		void tensorUpBilateral64Linear(const cv::Mat& src_highres, cv::Mat& dst, const int lut_num, const float sigma_space, const float sigma_range, const bool isOffset = true);
	};
}