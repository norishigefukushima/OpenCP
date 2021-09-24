#pragma once
#include "common.hpp"
#include "inlineMathFunctions.hpp"
namespace cp
{
	CP_EXPORT void upsampleNearest(cv::InputArray src, cv::OutputArray dest, const int scale);
	CP_EXPORT void upsampleLinear(cv::InputArray src, cv::OutputArray dest, const int scale);

	CP_EXPORT void upsampleCubic(cv::InputArray src, cv::OutputArray dest, const int scale, const double a = -1.5);
	CP_EXPORT void upsampleCubic_parallel(const cv::Mat& src, cv::Mat& dest, const int scale, const double a = 1.5);
	CP_EXPORT void upsampleWeightedCubic(cv::InputArray src_, cv::InputArray guide_, cv::OutputArray dest_, const int scale, const double a = -1.5);

	CP_EXPORT void setUpsampleMask(cv::InputArray src, cv::OutputArray dst);

	CP_EXPORT void resizeShift(cv::InputArray src, cv::OutputArray dest, const int scale, const int resize_method, const double shiftx, const double shifty);

	inline void setGaussianLnWeight(cv::Mat& weightmap, const float sigma, const int pown)
	{
		const int scale = (int)sqrt(weightmap.rows);
		const int d = (int)sqrt(weightmap.cols);
		const int r = d / 2;
		for (int n = 0; n < weightmap.rows; n++)
		{
			const int x_ = n % scale;
			const int y_ = n / scale;
			float x = float(x_) / (scale);
			float y = float(y_) / (scale);

			float* w = weightmap.ptr<float>(n);
			int idx = 0;
			float wsum = 0.f;
			for (int j = 0; j < d; j++)
			{
				for (int i = 0; i < d; i++)
				{
					float dist = hypot(x + r - 1 - i, y + r - 1 - j);
					//float dist = hypot(x - r + 1 + i, y - r + 1 + j);
					float w_ = exp(float(pow(dist, pown) / (-pown * sigma * sigma)));
					//float w_ = exp(dist / (-sigma));//Laplacian
					wsum += w_;
					w[idx++] = w_;
				}
			}
			for (int i = 0; i < weightmap.cols; i++)
			{
				w[i] /= wsum;
			}
		}
	}

	inline void setGaussianWeight8x8(cv::Mat& weightmap, const float sigma)
	{
		const int scale = (int)sqrt(weightmap.rows);

		for (int i = 0; i < scale * scale; i++)
		{
			const int x_ = i % scale;
			const int y_ = i / scale;
			float x = float(x_) / (scale);
			float y = float(y_) / (scale);

			float* weight = weightmap.ptr<float>(i);

			for (int n = 0, idx = 0; n < 8; n++)
			{
				for (int m = 0; m < 8; m++)
				{
					weight[idx++] = hypot(x + 3 - m, y + 3 - n);
				}
			}

			float wsum = 0.f;
			for (int j = 0; j < 64; j++)
			{
				const float dist = weight[j];
				const float w = exp(dist * dist / (-2.f * sigma * sigma));
				weight[j] = w;
				wsum += w;
			}

			wsum = 1.f / wsum;
			for (int j = 0; j < 64; j++)
			{
				weight[j] *= wsum;
			}
		}
	}

	inline void setGaussianWeight4x4(cv::Mat& weightmap, const float sigma)
	{
		const int scale = (int)sqrt(weightmap.rows);

		for (int i = 0; i < scale * scale; i++)
		{
			const int x_ = i % scale;
			const int y_ = i / scale;
			float x = float(x_) / (scale);
			float y = float(y_) / (scale);

			float* weight = weightmap.ptr<float>(i);

			weight[0] = hypot(x + 1, y + 1);
			weight[1] = hypot(x + 0, y + 1);
			weight[2] = hypot(x - 1, y + 1);
			weight[3] = hypot(x - 2, y + 1);

			weight[4] = hypot(x + 1, y + 0);
			weight[5] = hypot(x + 0, y + 0);
			weight[6] = hypot(x - 1, y + 0);
			weight[7] = hypot(x - 2, y + 0);

			weight[8] = hypot(x + 1, y - 1);
			weight[9] = hypot(x + 0, y - 1);
			weight[10] = hypot(x - 1, y - 1);
			weight[11] = hypot(x - 2, y - 1);

			weight[12] = hypot(x + 1, y - 2);
			weight[13] = hypot(x + 0, y - 2);
			weight[14] = hypot(x - 1, y - 2);
			weight[15] = hypot(x - 2, y - 2);

			float wsum = 0.f;
			for (int j = 0; j < 16; j++)
			{
				const float dist = weight[j];
				const float w = exp(dist * dist / (-2.f * sigma * sigma));
				weight[j] = w;
				wsum += w;
			}

			wsum = 1.f / wsum;
			for (int j = 0; j < 16; j++)
			{
				weight[j] *= wsum;
			}
		}
	}


	inline void setCubicWeight4x4(cv::Mat& weightmap, const float alpha)
	{
		const int scale = (int)sqrt(weightmap.rows);

		for (int i = 0; i < scale * scale; i++)
		{
			const int x_ = i % scale;
			const int y_ = i / scale;
			float x = float(x_) / (scale);
			float y = float(y_) / (scale);

			float* weight = weightmap.ptr<float>(i);

			weight[0] = cp::cubic(x + 1, alpha) * cp::cubic(y + 1, alpha);
			weight[1] = cp::cubic(x + 0, alpha) * cp::cubic(y + 1, alpha);
			weight[2] = cp::cubic(x - 1, alpha) * cp::cubic(y + 1, alpha);
			weight[3] = cp::cubic(x - 2, alpha) * cp::cubic(y + 1, alpha);
			weight[4] = cp::cubic(x + 1, alpha) * cp::cubic(y + 0, alpha);
			weight[5] = cp::cubic(x + 0, alpha) * cp::cubic(y + 0, alpha);
			weight[6] = cp::cubic(x - 1, alpha) * cp::cubic(y + 0, alpha);
			weight[7] = cp::cubic(x - 2, alpha) * cp::cubic(y + 0, alpha);

			weight[8] = cp::cubic(x + 1, alpha) * cp::cubic(y - 1, alpha);
			weight[9] = cp::cubic(x + 0, alpha) * cp::cubic(y - 1, alpha);
			weight[10] = cp::cubic(x - 1, alpha) * cp::cubic(y - 1, alpha);
			weight[11] = cp::cubic(x - 2, alpha) * cp::cubic(y - 1, alpha);

			weight[12] = cp::cubic(x + 1, alpha) * cp::cubic(y - 2, alpha);
			weight[13] = cp::cubic(x + 0, alpha) * cp::cubic(y - 2, alpha);
			weight[14] = cp::cubic(x - 1, alpha) * cp::cubic(y - 2, alpha);
			weight[15] = cp::cubic(x - 2, alpha) * cp::cubic(y - 2, alpha);

			float wsum = 0.f;
			for (int j = 0; j < 16; j++)
			{
				wsum += weight[j];
			}

			wsum = 1.f / wsum;
			for (int j = 0; j < 16; j++)
			{
				weight[j] *= wsum;
			}
		}
	}

	inline void setCubicWeightNonSep4x4(cv::Mat& weightmap, const float alpha)
	{
		const int scale = (int)sqrt(weightmap.rows);

		for (int i = 0; i < scale * scale; i++)
		{
			const int x_ = i % scale;
			const int y_ = i / scale;
			float x = float(x_) / (scale);
			float y = float(y_) / (scale);

			float* weight = weightmap.ptr<float>(i);

			weight[0] = hypot(x + 1, y + 1);
			weight[1] = hypot(x + 0, y + 1);
			weight[2] = hypot(x - 1, y + 1);
			weight[3] = hypot(x - 2, y + 1);

			weight[4] = hypot(x + 1, y + 0);
			weight[5] = hypot(x + 0, y + 0);
			weight[6] = hypot(x - 1, y + 0);
			weight[7] = hypot(x - 2, y + 0);

			weight[8] = hypot(x + 1, y - 1);
			weight[9] = hypot(x + 0, y - 1);
			weight[10] = hypot(x - 1, y - 1);
			weight[11] = hypot(x - 2, y - 1);

			weight[12] = hypot(x + 1, y - 2);
			weight[13] = hypot(x + 0, y - 2);
			weight[14] = hypot(x - 1, y - 2);
			weight[15] = hypot(x - 2, y - 2);

			float wsum = 0.f;
			for (int j = 0; j < 16; j++)
			{
				const float dist = weight[j];
				const float w = cp::cubic(dist, alpha);
				weight[j] = w;
				wsum += w;
			}

			wsum = 1.f / wsum;
			for (int j = 0; j < 16; j++)
			{
				weight[j] *= wsum;
			}
		}
	}

	inline void setGaussianWeight2x2(cv::Mat& weightmap, const float sigma)
	{
		const int scale = (int)sqrt(weightmap.rows);

		for (int i = 0; i < scale * scale; i++)
		{
			const int x_ = i % scale;
			const int y_ = i / scale;
			float x = float(x_) / (scale);
			float y = float(y_) / (scale);

			float* weight = weightmap.ptr<float>(i);

			weight[0] = hypot(x + 0, y + 0);
			weight[1] = hypot(x + 1, y + 0);
			weight[2] = hypot(x + 0, y + 1);
			weight[3] = hypot(x + 1, y + 1);

			float wsum = 0.f;
			for (int j = 0; j < 4; j++)
			{
				const float dist = weight[j];
				const float w = exp(dist * dist / (-2.f * sigma * sigma));
				weight[j] = w;
				wsum += w;
			}

			wsum = 1.f / wsum;
			for (int j = 0; j < 4; j++)
			{
				weight[j] *= wsum;
			}
		}
	}

	inline void setLinearWeight2x2(cv::Mat& weightmap)
	{
		const int scale = (int)sqrt(weightmap.rows);

		for (int i = 0; i < scale * scale; i++)
		{
			float x = (i % scale) / float(scale);
			float y = (i / scale) / float(scale);
			float* w = weightmap.ptr<float>(i);
			w[0] = (1.f - x) * (1.f - y);
			w[1] = (0.f + x) * (1.f - y);
			w[2] = (1.f - x) * (0.f + y);
			w[3] = (0.f + x) * (0.f + y);
		}
	}
}
