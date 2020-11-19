#include "bilateralGuidedUpsample.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	static void color2gray(Mat& input, Mat& output)
	{
		const float inv = 1.f / 3.f;
		for (int y = 0; y < input.rows; y++)
		{
			float* input_y = input.ptr<float>(y);
			float* output_y = output.ptr<float>(y);

			for (int x = 0; x < input.cols; x++)
			{
				//output_y[x] = (0.25f*input_y[3 * x + 0] + 0.5f*input_y[3 * x + 1] + 0.25f*input_y[3 * x + 2]);
				output_y[x] = (input_y[3 * x + 0] + input_y[3 * x + 1] + input_y[3 * x + 2]) * inv;
			}
		}
	}

	inline float lerp(const float pre_c, const float pre_v, const float x, const float next_c, const float next_v)
	{
		return((next_c - x) * pre_v + (x - pre_c) * next_v) / (next_c - pre_c);
	}

	inline float clamp(const float val, const float min_, const float max_)
	{
		return (min(max(val, min_), max_));
	}

	float pack_channels(int c, vector<float> exprs) {
		float e = exprs.back();
		for (int i = (int)exprs.size() - 2; i >= 0; i--) {
			if (c == i)	e = exprs[i];
		}
		return e;
	}

	// Size of each luma bin in the grid. Typically 1/8.
	// Size of each spatial bin in the grid. Typically 16.
	void bilateralGuidedUpsample(Mat& low_res_in, Mat& low_res_out, Mat& high_res_in, Mat& high_res_out, const int s_sigma, const float r_sigma)
	{
		static int l = 100; createTrackbar("l", "", &l, 100000);
		static int e = 100; createTrackbar("e", "", &e, 100000);
		const float lambda = 1e-6f * l * 0.01;
		const float epsilon = 1e-6f * e * 0.01;

		int height = low_res_in.rows;
		int width = low_res_in.cols;

		// The low resolution input
		Mat low_in;
		low_res_in.convertTo(low_in, CV_32FC3, 1.f / 255.f);

		// The low resolution output
		Mat low_out;
		low_res_out.convertTo(low_out, CV_32FC3, 1.f / 255.f);

		// The high resolution input
		Mat high_in;
		high_res_in.convertTo(high_in, CV_32FC3, 1.f / 255.f);

		// Add a boundary condition to the inputs.
		Mat clamped_low_in, clamped_low_out;

		copyMakeBorder(low_in, clamped_low_in, 7 * s_sigma / 2, 7 * s_sigma / 2, 7 * s_sigma / 2, 7 * s_sigma / 2, BORDER_REPLICATE);
		copyMakeBorder(low_out, clamped_low_out, 7 * s_sigma / 2, 7 * s_sigma / 2, 7 * s_sigma / 2, 7 * s_sigma / 2, BORDER_REPLICATE);

		// Figure out how much we're upsampling by. Not relevant if we're
		// just fitting curves.
		int upsample_factor_x = ceil((float)(high_in.cols / low_in.cols));		// factor = highres / rowres
		int upsample_factor_y = ceil((float)(high_in.rows / low_in.rows));
		int upsample_factor = max(upsample_factor_x, upsample_factor_y);

		Mat gray_low_in(clamped_low_in.rows, clamped_low_in.cols, CV_32FC1);
		Mat gray_high_in(high_in.rows, high_in.cols, CV_32FC1);

		color2gray(clamped_low_in, gray_low_in);
		color2gray(high_in, gray_high_in);

		// Construct the bilateral grid
		const int clamped_height = clamped_low_in.rows;
		const int clamped_width = clamped_low_in.cols;
		int grid_width = clamped_width / s_sigma;
		int grid_height = clamped_height / s_sigma;
		const int grid_range = 1.f / r_sigma + 1;		// 8
		Mat histogram = Mat::zeros(grid_width * grid_height, 22 * grid_range, CV_32FC1);

		//#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < grid_height; y++)
		{
			for (int x = 0; x < grid_width; x++)
			{
				float* histogram_ptr = histogram.ptr<float>(grid_width * y + x);

				for (int i = 0; i < s_sigma; i++)
				{
					float sy = y * s_sigma + i;
					float* clamped_low_out_ptr = clamped_low_out.ptr<float>(sy);
					float* clamped_low_in_ptr = clamped_low_in.ptr<float>(sy);

					for (int j = 0; j < s_sigma; j++)
					{
						int sx = x * s_sigma + j;
						float pos = gray_low_in.at<float>(sy, sx);
						pos = clamp(pos, 0.0f, 1.0f);
						int zi = (int)(round(pos * (1.0f / r_sigma)));

						// Sum all the terms we need to fit a line relating
						// low-res input to low-res output within this bilateral grid cell
						float vb = clamped_low_out_ptr[sx * 3 + 0];
						float vg = clamped_low_out_ptr[sx * 3 + 1];
						float vr = clamped_low_out_ptr[sx * 3 + 2];
						float sb = clamped_low_in_ptr[sx * 3 + 0];
						float sg = clamped_low_in_ptr[sx * 3 + 1];
						float sr = clamped_low_in_ptr[sx * 3 + 2];

						histogram_ptr[22 * zi + 0] += sr * sr;
						histogram_ptr[22 * zi + 1] += sr * sg;
						histogram_ptr[22 * zi + 2] += sr * sb;
						histogram_ptr[22 * zi + 3] += sr;
						histogram_ptr[22 * zi + 4] += sg * sg;
						histogram_ptr[22 * zi + 5] += sg * sb;
						histogram_ptr[22 * zi + 6] += sg;
						histogram_ptr[22 * zi + 7] += sb * sb;
						histogram_ptr[22 * zi + 8] += sb;
						histogram_ptr[22 * zi + 9] += 1.f;
						histogram_ptr[22 * zi + 10] += vr * sr;
						histogram_ptr[22 * zi + 11] += vr * sg;
						histogram_ptr[22 * zi + 12] += vr * sb;
						histogram_ptr[22 * zi + 13] += vr;
						histogram_ptr[22 * zi + 14] += vg * sr;
						histogram_ptr[22 * zi + 15] += vg * sg;
						histogram_ptr[22 * zi + 16] += vg * sb;
						histogram_ptr[22 * zi + 17] += vg;
						histogram_ptr[22 * zi + 18] += vb * sr;
						histogram_ptr[22 * zi + 19] += vb * sg;
						histogram_ptr[22 * zi + 20] += vb * sb;
						histogram_ptr[22 * zi + 21] += vb;
					}
				}
			}
		}

		// Blur the grid using a seven-tap filter	
		const float t0 = 1.f / 64.f;
		const float t1 = 1.f / 27.f;
		const float t2 = 1.f / 8.f;
		const float t3 = 1.f;

		Mat blurz = Mat::zeros(grid_width * grid_height, 22 * grid_range, CV_32FC1);
		for (int y = 0; y < grid_height; y++)
		{
			for (int x = 0; x < grid_width; x++)
			{
				float* blurz_ptr = blurz.ptr<float>(grid_width * y + x);
				float* histogram_ptr = histogram.ptr<float>(grid_width * y + x);

				for (int c = 0; c < 22; c++)
				{
					Mat tmp = Mat::zeros(1, grid_range + 6, CV_32FC1);
					float* tmp_ptr = tmp.ptr<float>(0);

					for (int i = 0; i < grid_range; i++)
					{
						tmp_ptr[i + 3] = histogram_ptr[22 * i + c];
					}

					// blur
					for (int z = 0; z < grid_range; z++)
					{
						blurz_ptr[22 * z + c] =
							tmp_ptr[z + 3 - 3] * t0 +
							tmp_ptr[z + 3 - 2] * t1 +
							tmp_ptr[z + 3 - 1] * t2 +
							tmp_ptr[z + 3 - 0] * t3 +
							tmp_ptr[z + 3 + 1] * t2 +
							tmp_ptr[z + 3 + 2] * t1 +
							tmp_ptr[z + 3 + 3] * t0;
					}
				}
			}
		}

		Mat blury = Mat::zeros(grid_width * (grid_height - 6), 22 * grid_range, CV_32FC1);
		for (int x = 0; x < grid_width; x++)
		{
			for (int z = 0; z < grid_range; z++)
			{
				for (int c = 0; c < 22; c++)
				{
					Mat tmp = Mat::zeros(1, grid_height, CV_32FC1);
					float* tmp_ptr = tmp.ptr<float>(0);

					for (int i = 0; i < grid_height; i++)
					{
						tmp_ptr[i] = blurz.at<float>(grid_width * i + x, 22 * z + c);
					}

					// blur
					for (int y = 0; y < grid_height - 6; y++)
					{
						blury.at<float>(grid_width * y + x, 22 * z + c) =
							tmp_ptr[y + 3 - 3] * t0 +
							tmp_ptr[y + 3 - 2] * t1 +
							tmp_ptr[y + 3 - 1] * t2 +
							tmp_ptr[y + 3 - 0] * t3 +
							tmp_ptr[y + 3 + 1] * t2 +
							tmp_ptr[y + 3 + 2] * t1 +
							tmp_ptr[y + 3 + 3] * t0;
					}
				}
			}
		}
		grid_height -= 6;

		Mat blurx = Mat::zeros((grid_width - 6) * grid_height, 22 * grid_range, CV_32FC1);
		for (int y = 0; y < grid_height; y++)
		{
			for (int z = 0; z < grid_range; z++)
			{
				for (int c = 0; c < 22; c++)
				{
					Mat tmp = Mat::zeros(1, grid_width, CV_32FC1);
					float* tmp_ptr = tmp.ptr<float>(0);

					for (int i = 0; i < grid_width; i++)
					{
						tmp_ptr[i] = blury.at<float>(grid_width * y + i, 22 * z + c);
					}
					// blur
					for (int x = 0; x < grid_width - 6; x++)
					{
						blurx.at<float>((grid_width - 6) * y + x, 22 * z + c) =
							tmp_ptr[x + 3 - 3] * t0 +
							tmp_ptr[x + 3 - 2] * t1 +
							tmp_ptr[x + 3 - 1] * t2 +
							tmp_ptr[x + 3 - 0] * t3 +
							tmp_ptr[x + 3 + 1] * t2 +
							tmp_ptr[x + 3 + 2] * t1 +
							tmp_ptr[x + 3 + 3] * t0;
					}
				}
			}
		}
		grid_width -= 6;
		// Do the solve, to convert the accumulated values to the affine metrices.
		Mat line(grid_width * grid_height, 12 * grid_range, CV_32F);

		// Regularize by pushing the solution towards the average gain
					// in this cell = (average output luma + eps) / (average input luma + eps).
		// Regularize by pushing the solution towards the average gain
					// in this cell = (average output luma + eps) / (average input luma + eps).

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < grid_height; y++)
		{
			Mat A(4, 4, CV_32F);
			Mat b(4, 3, CV_32F);
			Mat result(3, 4, CV_32FC1);
			//Mat tmp(4, 3, CV_32FC1);
			for (int x = 0; x < grid_width; x++)
			{
				float* blurx_ptr = blurx.ptr<float>(grid_width * y + x);
				float* line_ptr = line.ptr<float>(grid_width * y + x);

				for (int z = 0; z < grid_range; z++)
				{
					// Pull out the 4x4 symmetric matrix from the values we've acuumulated.
					A.at<float>(0, 0) = blurx_ptr[22 * z + 0];
					A.at<float>(0, 1) = blurx_ptr[22 * z + 1];
					A.at<float>(0, 2) = blurx_ptr[22 * z + 2];
					A.at<float>(0, 3) = blurx_ptr[22 * z + 3];
					A.at<float>(1, 0) = A.at<float>(0, 1);
					A.at<float>(1, 1) = blurx_ptr[22 * z + 4];
					A.at<float>(1, 2) = blurx_ptr[22 * z + 5];
					A.at<float>(1, 3) = blurx_ptr[22 * z + 6];
					A.at<float>(2, 0) = A.at<float>(0, 2);
					A.at<float>(2, 1) = A.at<float>(1, 2);
					A.at<float>(2, 2) = blurx_ptr[22 * z + 7];
					A.at<float>(2, 3) = blurx_ptr[22 * z + 8];
					A.at<float>(3, 0) = A.at<float>(0, 3);
					A.at<float>(3, 1) = A.at<float>(1, 3);
					A.at<float>(3, 2) = A.at<float>(2, 3);
					A.at<float>(3, 3) = blurx_ptr[22 * z + 9];

					// Pull out the rhs				
					b.at<float>(0, 0) = blurx_ptr[22 * z + 10];
					b.at<float>(1, 0) = blurx_ptr[22 * z + 11];
					b.at<float>(2, 0) = blurx_ptr[22 * z + 12];
					b.at<float>(3, 0) = blurx_ptr[22 * z + 13];
					b.at<float>(0, 1) = blurx_ptr[22 * z + 14];
					b.at<float>(1, 1) = blurx_ptr[22 * z + 15];
					b.at<float>(2, 1) = blurx_ptr[22 * z + 16];
					b.at<float>(3, 1) = blurx_ptr[22 * z + 17];
					b.at<float>(0, 2) = blurx_ptr[22 * z + 18];
					b.at<float>(1, 2) = blurx_ptr[22 * z + 19];
					b.at<float>(2, 2) = blurx_ptr[22 * z + 20];
					b.at<float>(3, 2) = blurx_ptr[22 * z + 21];


					// The bottom right entry of A is a count of the number of
					// constraints affecting this cell.
					const float N = A.at<float>(3, 3);

					const float output_luma = b.at<float>(3, 0) + 2 * b.at<float>(3, 1) + b.at<float>(3, 2) + epsilon * (N + 1);
					const float input_luma = A.at<float>(3, 0) + 2 * A.at<float>(3, 1) + A.at<float>(3, 2) + epsilon * (N + 1);
					const float gain = output_luma / input_luma;

					const float weighted_lambda = lambda * (N + 1);
					A.at<float>(0, 0) += weighted_lambda;
					A.at<float>(1, 1) += weighted_lambda;
					A.at<float>(2, 2) += weighted_lambda;
					A.at<float>(3, 3) += weighted_lambda;

					b.at<float>(0, 0) += weighted_lambda * gain;
					b.at<float>(1, 1) += weighted_lambda * gain;
					b.at<float>(2, 2) += weighted_lambda * gain;

					// Now solve Ax = b							
					solve(A, b, result, DECOMP_LU);
					//transpose(tmp, result);

					// Pack the resulting matrix into the output Func.
					line_ptr[12 * z + 0] = result.at<float>(0, 0);
					line_ptr[12 * z + 1] = result.at<float>(1, 0);
					line_ptr[12 * z + 2] = result.at<float>(2, 0);
					line_ptr[12 * z + 3] = result.at<float>(3, 0);
					line_ptr[12 * z + 4] = result.at<float>(0, 1);
					line_ptr[12 * z + 5] = result.at<float>(1, 1);
					line_ptr[12 * z + 6] = result.at<float>(2, 1);
					line_ptr[12 * z + 7] = result.at<float>(3, 1);
					line_ptr[12 * z + 8] = result.at<float>(0, 2);
					line_ptr[12 * z + 9] = result.at<float>(1, 2);
					line_ptr[12 * z + 10] = result.at<float>(2, 2);
					line_ptr[12 * z + 11] = result.at<float>(3, 2);
				}
			}
		}

		// line‚ðy,x•ûŒü‚É1‚¸‚ÂŠg‘å
		Mat line_((grid_width + 1) * (grid_height + 1), 12 * grid_range, CV_32F);
		for (int y = 0; y < grid_height; y++)
		{
			for (int x = 0; x < grid_width; x++)
			{
				float* line_ptr = line.ptr<float>(grid_width * y + x);
				float* line_ptr_ = line_.ptr<float>((grid_width + 1) * y + x);

				for (int i = 0; i < 12 * grid_range; i++)
				{
					line_ptr_[i] = line_ptr[i];
				}
				if (x == grid_width - 1)
				{
					for (int i = 0; i < 12 * grid_range; i++)
						line_.at<float>((grid_width + 1) * y + x + 1, i) = line_ptr[i];
				}
			}
			if (y == grid_height - 1)
			{
				for (int x = 0; x < grid_width; x++)
				{
					float* line_ptr_ = line_.ptr<float>((grid_width + 1) * (y + 1) + x);
					float* line_ptr = line.ptr<float>(grid_width * y + x);

					for (int i = 0; i < 12 * grid_range; i++)
					{
						line_ptr_[i] = line_ptr[i];
					}
					if (x == grid_width - 1)
					{
						for (int i = 0; i < 12 * grid_range; i++)
							line_.at<float>((grid_width + 1) * (y + 1) + x + 1, i) = line.at<float>(grid_width * y + x, i);
					}
				}
			}
		}

		// Spatial bin size in the high-res image.
		float big_sigma = s_sigma * upsample_factor;
		Mat interpolated_z(high_in.rows * high_in.cols, 12, CV_32FC1);
		Mat interpolated(high_in.rows, high_in.cols, CV_32FC3);
		int num_intensity_bins = (int)(1.0f / r_sigma);

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < high_in.rows; y++)
		{
			float yf = (float)y / big_sigma;
			int yi = (int)yf;
			yf -= yi;

			for (int x = 0; x < high_in.cols; x++)
			{
				float xf = (float)x / big_sigma;
				int xi = (int)xf;
				xf -= xi;

				float val = gray_high_in.at<float>(y, x);
				val = clamp(val, 0.0f, 1.0f);
				float zv = val * num_intensity_bins;

				int zi = (int)zv;
				float zf = zv - zi;

				float* interpolated_z_ptr = interpolated_z.ptr<float>(high_in.cols * y + x);
				float* interpolated_ptr = interpolated.ptr<float>(y) + 3 * x;
				float* high_in_ptr = high_in.ptr<float>(y) + 3 * x;

				for (int c = 0; c < 12; c++)
				{
					float pre_y, next_y, pre_x, next_x, pre_z, next_z;
					pre_y = line_.at<float>((grid_width + 1) * yi + xi, 12 * zi + c);
					next_y = line_.at<float>((grid_width + 1) * (yi + 1) + xi, 12 * zi + c);
					pre_x = lerp(0.f, pre_y, yf, 1.f, next_y);
					//if (yi == grid_height)
					//	pre_x = line_.at<float>(grid_width*yi + xi, 12 * zi + c);
					pre_y = line_.at<float>((grid_width + 1) * yi + (xi + 1), 12 * zi + c);
					next_y = line_.at<float>((grid_width + 1) * (yi + 1) + (xi + 1), 12 * zi + c);
					next_x = lerp(0.f, pre_y, yf, 1.f, next_y);
					//if (yi == grid_height)
					//	next_x = line_.at<float>(grid_width*yi + (xi + 1), 12 * zi + c);
					pre_z = lerp(0.f, pre_x, xf, 1.f, next_x);

					pre_y = line_.at<float>((grid_width + 1) * yi + xi, 12 * (zi + 1) + c);
					next_y = line_.at<float>((grid_width + 1) * (yi + 1) + xi, 12 * (zi + 1) + c);
					pre_x = lerp(0.f, pre_y, yf, 1.f, next_y);
					pre_y = line_.at<float>((grid_width + 1) * yi + (xi + 1), 12 * (zi + 1) + c);
					next_y = line_.at<float>((grid_width + 1) * (yi + 1) + (xi + 1), 12 * (zi + 1) + c);
					next_x = lerp(0.f, pre_y, yf, 1.f, next_y);
					next_z = lerp(0.f, pre_x, xf, 1.f, next_x);

					interpolated_z_ptr[c] = lerp(0.f, pre_z, zf, 1.f, next_z);
				}
				// Multiply by 3x4 by 4x1.
				for (int c = 0; c < 3; c++)
				{
					interpolated_ptr[2 - c] =
						interpolated_z_ptr[4 * c + 0] * high_in_ptr[2] +
						interpolated_z_ptr[4 * c + 1] * high_in_ptr[1] +
						interpolated_z_ptr[4 * c + 2] * high_in_ptr[0] +
						interpolated_z_ptr[4 * c + 3];
				}
				// Normalize
				interpolated_ptr[0] = clamp(interpolated_ptr[0], 0.f, 1.f);
				interpolated_ptr[1] = clamp(interpolated_ptr[1], 0.f, 1.f);
				interpolated_ptr[2] = clamp(interpolated_ptr[2], 0.f, 1.f);
			}
		}

		interpolated.convertTo(high_res_out, CV_8UC3, 255);
	}

	inline float BilateralGuidedUpsample::lerp(const float pre_c, const float pre_v, const float x, const float next_c, const float next_v)
	{
		return((next_c - x) * pre_v + (x - pre_c) * next_v) / (next_c - pre_c);
	}

	inline float BilateralGuidedUpsample::clamp(const float val, const float min_, const float max_)
	{
		return (min(max(val, min_), max_));
	}

	void BilateralGuidedUpsample::color2gray(Mat& input, Mat& output)
	{
		output.create(input.rows, input.cols, CV_32FC1);

		const float inv = 1.f / 3.f;
		//cvtColor(input, output, COLOR_BGR2GRAY);

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < input.rows; y++)
		{
			float* input_y = input.ptr<float>(y);
			float* output_y = output.ptr<float>(y);

			for (int x = 0; x < input.cols; x++)
			{
				output_y[x] = (0.25f * input_y[3 * x + 0] + 0.5f * input_y[3 * x + 1] + 0.25f * input_y[3 * x + 2]);
				//output_y[x] = (input_y[3 * x + 0] + input_y[3 * x + 1] + input_y[3 * x + 2])*inv;
			}
		}
	}

	void BilateralGuidedUpsample::constructBilateralGrid(const Mat& low_in_border, const Mat& low_out_border, const Mat& gray_low_in_border, Mat& dest, const int num_spatial_blocks, const int num_bin)
	{
		const int grid_width_border = low_in_border.cols / num_spatial_blocks;
		const int grid_height_border = low_in_border.rows / num_spatial_blocks;
		const int grid_range_border = num_bin + 1;
		const float r_sigma = 1.f / num_bin;

		dest.create(grid_width_border * grid_height_border, num_range_coeffs * grid_range_border, CV_32FC1);
		dest.setTo(0);

		//#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < grid_height_border; y++)
		{
			for (int x = 0; x < grid_width_border; x++)
			{
				float* bgrid_ptr = dest.ptr<float>(grid_width_border * y + x);

				for (int i = 0; i < num_spatial_blocks; i++)
				{
					const int sy = y * num_spatial_blocks + i;
					const float* clamped_low_in_ptr = low_in_border.ptr<float>(sy);
					const float* clamped_low_out_ptr = low_out_border.ptr<float>(sy);

					for (int j = 0; j < num_spatial_blocks; j++)
					{
						const int sx = x * num_spatial_blocks + j;
						const float pos = gray_low_in_border.at<float>(sy, sx);
						const int zi = (int)(round(pos * (1.0f / r_sigma)));

						// Sum all the terms we need to fit a line relating low-res input to low-res output within this bilateral grid cell
						const float vb = clamped_low_out_ptr[sx * 3 + 0];
						const float vg = clamped_low_out_ptr[sx * 3 + 1];
						const float vr = clamped_low_out_ptr[sx * 3 + 2];
						const float sb = clamped_low_in_ptr[sx * 3 + 0];
						const float sg = clamped_low_in_ptr[sx * 3 + 1];
						const float sr = clamped_low_in_ptr[sx * 3 + 2];

						float* h = &bgrid_ptr[num_range_coeffs * zi];
						h[0] += sr * sr;
						h[1] += sr * sg;
						h[2] += sr * sb;
						h[3] += sr;
						h[4] += sg * sg;
						h[5] += sg * sb;
						h[6] += sg;
						h[7] += sb * sb;
						h[8] += sb;
						h[9] += 1.f;
						h[10] += vr * sr;
						h[11] += vr * sg;
						h[12] += vr * sb;
						h[13] += vr;
						h[14] += vg * sr;
						h[15] += vg * sg;
						h[16] += vg * sb;
						h[17] += vg;
						h[18] += vb * sr;
						h[19] += vb * sg;
						h[20] += vb * sb;
						h[21] += vb;
					}
				}
			}
		}
	}

	void BilateralGuidedUpsample::blur7tap(Mat& src, Mat& dest, const int grid_width_border, const int grid_height_border, const int grid_range_border)
	{
		const int grid_width = grid_width_border - 6;
		const int grid_height = grid_height_border - 6;

		float wr[4];
		float ws[4];
		ws[0] = 1.f / 64.f;
		ws[1] = 1.f / 27.f;
		ws[2] = 1.f / 8.f;
		ws[3] = 1.f / 1.f;

		wr[0] = 1.f / 64.f;
		wr[1] = 1.f / 27.f;
		wr[2] = 1.f / 8.f;
		wr[3] = 1.f / 1.f;

		/*const float t0 = 1.f / 64.f;
		const float t1 = 1.f / 27.f;
		const float t2 = 1.f / 8.f;
		const float t3 = 1.f;*/

		dest.create(Size(src.size()), CV_32FC1);

		//blur z
#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < grid_height_border; y++)
		{
			//Mat seven_tap_filter_buffz = Mat::zeros(1, grid_range + 6, CV_32FC1);
			Mat seven_tap_filter_buffz(1, grid_range_border + 6, CV_32FC1);
			for (int x = 0; x < grid_width_border; x++)
			{
				const float* bgrid_ptr = src.ptr<float>(grid_width_border * y + x);
				float* blurz_ptr = dest.ptr<float>(grid_width_border * y + x);

				for (int c = 0; c < num_range_coeffs; c++)
				{
					float* tap7_buff = seven_tap_filter_buffz.ptr<float>();

					for (int i = 0; i < grid_range_border; i++)
					{
						tap7_buff[i + 3] = bgrid_ptr[num_range_coeffs * i + c];
					}
					tap7_buff[0] = tap7_buff[1] = tap7_buff[2] = tap7_buff[3];
					tap7_buff[grid_range_border + 5] = tap7_buff[grid_range_border + 4] = tap7_buff[grid_range_border + 3] = tap7_buff[grid_range_border + 2];

					// blur
					for (int z = 0; z < grid_range_border; z++)
					{
						blurz_ptr[num_range_coeffs * z + c] =
							tap7_buff[z + 3 - 3] * wr[0] +
							tap7_buff[z + 3 - 2] * wr[1] +
							tap7_buff[z + 3 - 1] * wr[2] +
							tap7_buff[z + 3 - 0] * wr[3] +
							tap7_buff[z + 3 + 1] * wr[2] +
							tap7_buff[z + 3 + 2] * wr[1] +
							tap7_buff[z + 3 + 3] * wr[0];
					}
				}
			}
		}

		//blur y
#pragma omp parallel for schedule(dynamic)
		for (int x = 0; x < grid_width_border; x++)
		{
			Mat seven_tap_filter_buffy(1, grid_height_border, CV_32FC1);
			for (int z = 0; z < grid_range_border; z++)
			{
				for (int c = 0; c < num_range_coeffs; c++)
				{
					float* tap7_buff = seven_tap_filter_buffy.ptr<float>(0);

					for (int i = 0; i < grid_height_border; i++)
					{
						tap7_buff[i] = dest.at<float>(grid_width_border * i + x, num_range_coeffs * z + c);
					}

					// blur
					for (int y = 0; y < grid_height; y++)
					{
						src.at<float>(grid_width_border * y + x, num_range_coeffs * z + c) =
							tap7_buff[y + 3 - 3] * ws[0] +
							tap7_buff[y + 3 - 2] * ws[1] +
							tap7_buff[y + 3 - 1] * ws[2] +
							tap7_buff[y + 3 - 0] * ws[3] +
							tap7_buff[y + 3 + 1] * ws[2] +
							tap7_buff[y + 3 + 2] * ws[1] +
							tap7_buff[y + 3 + 3] * ws[0];
					}
				}
			}
		}

		//blur x
#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < grid_height; y++)
		{
			Mat seven_tap_filter_buffx(1, grid_width_border, CV_32FC1);
			for (int z = 0; z < grid_range_border; z++)
			{
				for (int c = 0; c < num_range_coeffs; c++)
				{
					float* tap7_buff = seven_tap_filter_buffx.ptr<float>(0);

					for (int i = 0; i < grid_width_border; i++)
					{
						tap7_buff[i] = src.at<float>(grid_width_border * y + i, num_range_coeffs * z + c);
					}
					// blur
					for (int x = 0; x < grid_width; x++)
					{
						dest.at<float>(grid_width_border * y + x, num_range_coeffs * z + c) =
							tap7_buff[x + 3 - 3] * ws[0] +
							tap7_buff[x + 3 - 2] * ws[1] +
							tap7_buff[x + 3 - 1] * ws[2] +
							tap7_buff[x + 3 - 0] * ws[3] +
							tap7_buff[x + 3 + 1] * ws[2] +
							tap7_buff[x + 3 + 2] * ws[1] +
							tap7_buff[x + 3 + 3] * ws[0];
					}
				}
			}
		}
	}

	void BilateralGuidedUpsample::optimize(const Mat& src, Mat& dest, const float lambda, const float epsilon, const int grid_width_border, const int grid_height_border, const int grid_range_border)
	{
		const int grid_width = grid_width_border - 6;
		const int grid_height = grid_height_border - 6;
		dest.create(grid_width * grid_height, num_optimized_coeffs * grid_range_border, CV_32F);

		// Regularize by pushing the solution towards the average gain
					// in this cell = (average output luma + eps) / (average input luma + eps).
		// Regularize by pushing the solution towards the average gain
					// in this cell = (average output luma + eps) / (average input luma + eps).

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < grid_height; y++)
		{
			Mat A(4, 4, CV_32F);
			Mat b(4, 3, CV_32F);
			Mat result(3, 4, CV_32F);
			for (int x = 0; x < grid_width; x++)
			{
				const float* bgrid_ptr = src.ptr<float>(grid_width_border * y + x);
				float* dest_ptr = dest.ptr<float>(grid_width * y + x);

				for (int z = 0; z < grid_range_border; z++)
				{
					// Pull out the 4x4 symmetric matrix from the values we've acuumulated.
					const float* bx = &bgrid_ptr[num_range_coeffs * z];
					A.at<float>(0, 0) = bx[0];
					A.at<float>(0, 1) = bx[1];
					A.at<float>(0, 2) = bx[2];
					A.at<float>(0, 3) = bx[3];
					A.at<float>(1, 0) = A.at<float>(0, 1);
					A.at<float>(1, 1) = bx[4];
					A.at<float>(1, 2) = bx[5];
					A.at<float>(1, 3) = bx[6];
					A.at<float>(2, 0) = A.at<float>(0, 2);
					A.at<float>(2, 1) = A.at<float>(1, 2);
					A.at<float>(2, 2) = bx[7];
					A.at<float>(2, 3) = bx[8];
					A.at<float>(3, 0) = A.at<float>(0, 3);
					A.at<float>(3, 1) = A.at<float>(1, 3);
					A.at<float>(3, 2) = A.at<float>(2, 3);
					A.at<float>(3, 3) = bx[9];

					// Pull out the rhs				
					b.at<float>(0, 0) = bx[10];
					b.at<float>(1, 0) = bx[11];
					b.at<float>(2, 0) = bx[12];
					b.at<float>(3, 0) = bx[13];
					b.at<float>(0, 1) = bx[14];
					b.at<float>(1, 1) = bx[15];
					b.at<float>(2, 1) = bx[16];
					b.at<float>(3, 1) = bx[17];
					b.at<float>(0, 2) = bx[18];
					b.at<float>(1, 2) = bx[19];
					b.at<float>(2, 2) = bx[20];
					b.at<float>(3, 2) = bx[21];

					// The bottom right entry of A is a count of the number of
					// constraints affecting this cell.
					const float N = A.at<float>(3, 3);

					const float output_luma = b.at<float>(3, 0) + 2.f * b.at<float>(3, 1) + b.at<float>(3, 2) + epsilon * (N + 1);
					const float input_luma = A.at<float>(3, 0) + 2.f * A.at<float>(3, 1) + A.at<float>(3, 2) + epsilon * (N + 1);
					const float gain = output_luma / input_luma;

					const float weighted_lambda = lambda * (N + 1.f);
					A.at<float>(0, 0) += weighted_lambda;
					A.at<float>(1, 1) += weighted_lambda;
					A.at<float>(2, 2) += weighted_lambda;
					A.at<float>(3, 3) += weighted_lambda;

					b.at<float>(0, 0) += weighted_lambda * gain;
					b.at<float>(1, 1) += weighted_lambda * gain;
					b.at<float>(2, 2) += weighted_lambda * gain;

					// Now solve Ax = b							
					cv::solve(A, b, result, DECOMP_LU);
					//transpose(tmp, result);

					// Pack the resulting matrix into the output Func.
					float* lp = &dest_ptr[num_optimized_coeffs * z];
					lp[0] = result.at<float>(0, 0);
					lp[1] = result.at<float>(1, 0);
					lp[2] = result.at<float>(2, 0);
					lp[3] = result.at<float>(3, 0);
					lp[4] = result.at<float>(0, 1);
					lp[5] = result.at<float>(1, 1);
					lp[6] = result.at<float>(2, 1);
					lp[7] = result.at<float>(3, 1);
					lp[8] = result.at<float>(0, 2);
					lp[9] = result.at<float>(1, 2);
					lp[10] = result.at<float>(2, 2);
					lp[11] = result.at<float>(3, 2);
				}
			}
		}
	}











	void BilateralGuidedUpsample::linearInterpolation(const Mat& high_src, const Mat& high_src_gray, const Mat& opt_bgrid, Mat& dest, const int grid_width, const int grid_height, const int grid_range, const float upsample_size)
	{
		const int num_bins = grid_range;
		dest.create(high_src.rows, high_src.cols, CV_32FC3);

#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < high_src.rows; y++)
		{
			float interpolated_z[12];
			float yf = (float)y / upsample_size;
			const int yi = (int)yf;
			const int yinext = min(yi + 1, grid_height - 1);
			yf -= yi;

			for (int x = 0; x < high_src.cols; x++)
			{
				float xf = (float)x / upsample_size;
				const int xi = (int)xf;
				const int xinext = min(xi + 1, grid_width - 1);
				xf -= xi;

				float val = high_src_gray.at<float>(y, x);
				val = clamp(val, 0.0f, 1.0f);

				const float zv = val * num_bins;
				const int zi = (int)zv;
				float zf = zv - zi;

				const float* high_in_ptr = high_src.ptr<float>(y) + 3 * x;
				float* interpolated_ptr = dest.ptr<float>(y) + 3 * x;

				for (int c = 0; c < num_optimized_coeffs; c++)
				{
					float pre_y, next_y, pre_x, next_x, pre_z, next_z;

					pre_y = opt_bgrid.at<float>(grid_width * yi + xi, num_optimized_coeffs * zi + c);
					next_y = opt_bgrid.at<float>(grid_width * yinext + xi, num_optimized_coeffs * zi + c);
					pre_x = lerp(0.f, pre_y, yf, 1.f, next_y);

					pre_y = opt_bgrid.at<float>(grid_width * yi + xinext, num_optimized_coeffs * zi + c);
					next_y = opt_bgrid.at<float>(grid_width * yinext + xinext, num_optimized_coeffs * zi + c);
					next_x = lerp(0.f, pre_y, yf, 1.f, next_y);
					pre_z = lerp(0.f, pre_x, xf, 1.f, next_x);

					pre_y = opt_bgrid.at<float>(grid_width * yi + xi, num_optimized_coeffs * (zi + 1) + c);
					next_y = opt_bgrid.at<float>(grid_width * yinext + xi, num_optimized_coeffs * (zi + 1) + c);
					pre_x = lerp(0.f, pre_y, yf, 1.f, next_y);

					pre_y = opt_bgrid.at<float>(grid_width * yi + xinext, num_optimized_coeffs * (zi + 1) + c);
					next_y = opt_bgrid.at<float>(grid_width * yinext + xinext, num_optimized_coeffs * (zi + 1) + c);
					next_x = lerp(0.f, pre_y, yf, 1.f, next_y);
					next_z = lerp(0.f, pre_x, xf, 1.f, next_x);

					interpolated_z[c] = lerp(0.f, pre_z, zf, 1.f, next_z);
				}
				// Multiply by 3x4 by 4x1.
				for (int c = 0; c < 3; c++)
				{
					interpolated_ptr[2 - c] =
						interpolated_z[4 * c + 0] * high_in_ptr[2] +
						interpolated_z[4 * c + 1] * high_in_ptr[1] +
						interpolated_z[4 * c + 2] * high_in_ptr[0] +
						interpolated_z[4 * c + 3];
				}

				interpolated_ptr[0] = clamp(interpolated_ptr[0], 0.f, 1.f);
				interpolated_ptr[1] = clamp(interpolated_ptr[1], 0.f, 1.f);
				interpolated_ptr[2] = clamp(interpolated_ptr[2], 0.f, 1.f);
			}
		}
	}

	void BilateralGuidedUpsample::upsample(Mat& low_res_in, Mat& low_res_out, Mat& high_res_in, Mat& high_res_out, const int num_spatial_blocks, const int num_bin, float lambda, float epsilon, const int border)
	{
		CV_Assert(low_res_in.channels() == 3);

		const int num_spatial_div = max(num_spatial_blocks, 1);

		low_res_in.convertTo(low_in, CV_32FC3, 1.f / 255.f);
		low_res_out.convertTo(low_out, CV_32FC3, 1.f / 255.f);
		high_res_in.convertTo(high_in, CV_32FC3, 1.f / 255.f);

		const int tapSize = 7;
		const int bb = tapSize * num_spatial_blocks / 2;
		copyMakeBorder(low_in, clamped_low_in, bb, bb, bb, bb, border);
		copyMakeBorder(low_out, clamped_low_out, bb, bb, bb, bb, border);

		// Figure out how much we're upsampling by. Not relevant if we're just fitting curves.
		const int upsample_factor_x = (int)ceil(((float)high_in.cols / low_in.cols));// factor = highres / rowres
		const int upsample_factor_y = (int)ceil(((float)high_in.rows / low_in.rows));
		const int upsampleFactor = max(upsample_factor_x, upsample_factor_y);

		const int grid_width_border = clamped_low_in.cols / num_spatial_blocks;
		const int grid_height_border = clamped_low_in.rows / num_spatial_blocks;
		const int grid_width = grid_width_border - 6;
		const int grid_height = grid_height_border - 6;
		const int grid_range_border = num_bin + 1;
		const int grid_range = num_bin;

		//cout << gridWidthB << "," << gridHeightB << "," << gridRange << endl;

		//construct bilateral grid
		//cp::Timer t("", 0, false);
		//t.start();
		color2gray(clamped_low_in, gray_low_in);
		constructBilateralGrid(clamped_low_in, clamped_low_out, gray_low_in, bilateralGrid, num_spatial_blocks, num_bin);
		//t.getTime(true);

		//Bluring the grid using a seven-tap filter.
		//src input has inplace oparations; thus, the input of the bilateralGrid's memory is changed.
		//t.start();
		blur7tap(bilateralGrid, bilateralGridBlur, grid_width_border, grid_height_border, grid_range_border);
		//t.getTime(true);

		// Do the solve, to convert the accumulated values to the affine metrices.
		//t.start();
		optimize(bilateralGridBlur, bilateralGridOpt, lambda, epsilon, grid_width_border, grid_height_border, grid_range_border);
		//t.getTime(true);

		//interpolating bilateral grid for full resolution.
		//t.start();
		color2gray(high_in, gray_high_in);
		const float upsampleFactorBG2Highres = upsampleFactor * num_spatial_blocks;
		linearInterpolation(high_in, gray_high_in, bilateralGridOpt, output32f, grid_width, grid_height, grid_range, upsampleFactorBG2Highres);
		//t.getTime(true);cout << endl;

		output32f.convertTo(high_res_out, CV_8UC3, 255);
	}
}