#pragma once
#include "common.hpp"

namespace cp
{
	class CP_EXPORT LocalLUTUpsample //no constructor
	{
	public:
		enum class BUILD_LUT
		{
			L2_MIN,
			L1_MIN,
			LInf_MIN,
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
			NO_INTERPOLATION,
			//experimental
			LINEAR_LAST2,
			EXPERIMENT1,
			EXPERIMENT2,

			SIZE
		};

		enum class UPTENSOR
		{
			NEAREST,//box1
			BOX4,////2x2
			BOX16,//4x4
			BOX64,//8x8
			GAUSS4,//2x2
			GAUSS16,//4x4
			GAUSS64,//8x8
			LINEAR,//2x2
			CUBIC,//4x4
			BoxNxN,
			GaussNxN,
			//experimental
			BILATERAL16,
			BILATERAL64,
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
		/// <summary>
		/// local LUT upsampling
		/// </summary>
		/// <param name="src_low">source low resolution image</param>
		/// <param name="prc_low">processed low resolution image</param>
		/// <param name="src_high">source high resolution image</param>
		/// <param name="dst_high">processed (destination) high resolution image</param>
		/// <param name="r">radius of subsampled spatial domain</param>
		/// <param name="lut_num">number of LUT per pixel</param>
		/// <param name="lut_filter_radius">radius of LUT filter for range domain</param>
		/// <param name="build_method">building LUT method</param>
		/// <param name="tensorup_method">tensor upsampling method</param>
		/// <param name="lut_boundary_method">LUT boundary condition</param>
		/// <param name="isUseOffsetMap">with/without offset map</param>
		void upsample(cv::InputArray src_low, cv::InputArray prc_low, cv::InputArray src_high, cv::OutputArray prc_high,
			const int r, const int lut_num = 256, const int lut_filter_radius = 2,
			const BUILD_LUT build_method = BUILD_LUT::L1_MIN,
			const UPTENSOR tensorup_method = UPTENSOR::BOX16,
			const BOUNDARY lut_boundary_method = BOUNDARY::LINEAR,
			const bool isUseOffsetMap = true);

		/// <summary>
		/// GUI viewing for LUT per pixel
		/// </summary>
		/// <param name="lowres_src">show image </param>
		/// <param name="lowres_prc">show image </param>
		/// <param name="highres_src">for additional scatter plot</param>
		/// <param name="highres_groundtruth">for additional bold circle plot</param>
		/// <param name="isWait">loop waiting</param>
		/// <param name="wname">window name</param>
		void guiLUT(cv::Mat& lowres_src, cv::Mat& lowres_prc, cv::Mat& highres_src, cv::Mat& highres_out, bool isWait = true, std::string wname = "LocalLUT");

	private:
		const bool useSoA = false;//false; AoS is faster
		cv::Size lowres_size;
		cv::Mat src_low_border;
		cv::Mat prc_low_border;

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
		float up_sampling_ratio_resolution = 0.f;//up_sampling_ratio
		int boundary_replicate_offset = 0;

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