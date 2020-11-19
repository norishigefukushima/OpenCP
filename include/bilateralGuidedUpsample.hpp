#pragma once
#include "common.hpp"

namespace cp
{
	// Size of each luma bin in the grid. Typically 1/8.
// Size of each spatial bin in the grid. Typically 16.
	CP_EXPORT void bilateralGuidedUpsample(cv::Mat& low_res_in, cv::Mat& low_res_out, cv::Mat& high_res_in, cv::Mat& high_res_out, const int s_sigma = 16, const float r_sigma = 1.f / 8.f);

	class CP_EXPORT BilateralGuidedUpsample
	{
		inline float lerp(const float pre_c, const float pre_v, const float x, const float next_c, const float next_v);
		inline float clamp(const float val, const float min_, const float max_);

		void color2gray(cv::Mat& input, cv::Mat& output);

		cv::Mat low_in;// The low resolution input	
		cv::Mat low_out;// The low resolution output	
		cv::Mat high_in;// The high resolution input

		cv::Mat clamped_low_in;// Add a boundary condition to the input.
		cv::Mat clamped_low_out;// Add a boundary condition to the outputs.

		cv::Mat gray_low_in;//clamped gray low-res input
		cv::Mat gray_high_in;//clamped gray high-res input

		cv::Mat bilateralGrid;//grid_width_border*grid_height_border*22 * grid_range_border
		cv::Mat bilateralGridBlur;//grid_width_border*grid_height_border*22 * grid_range_border
		cv::Mat bilateralGridOpt;//grid_width*grid_height*12*grid_range_border;

		cv::Mat output32f;

		const int num_range_coeffs = 22;
		const int num_optimized_coeffs = 12;

		void constructBilateralGrid(const cv::Mat& low_in_border, const cv::Mat& low_out_border, const cv::Mat& gray_low_in_border, cv::Mat& dest, const int num_spatial_blocks, const int num_bin);
		void blur7tap(cv::Mat& src, cv::Mat& dest, const int grid_width_border, const int grid_height_border, const int grid_range_border);
		void optimize(const cv::Mat& src, cv::Mat& dest, const float lambda, const float epsilon, const int grid_width_border, const int grid_height_border, const int grid_range_border);
		void linearInterpolation(const cv::Mat& high_src, const cv::Mat& high_src_gray, const cv::Mat& opt_bgrid, cv::Mat& dest, const int grid_width, const int grid_height, const int grid_range, const float upsample_size);

	public:
		void upsample(cv::Mat& low_res_in, cv::Mat& low_res_out, cv::Mat& high_res_in, cv::Mat& high_res_out, const int num_spatial_blocks, const int num_bin, float lambda = 1e-6f, float epsilon = 1e-6f, const int border = cv::BORDER_REPLICATE);
	};
}