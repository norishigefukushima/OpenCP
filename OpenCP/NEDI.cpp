#include "upsample.hpp"
#include "NEDI.hpp"
#include <inlineSIMDFunctions.hpp>
using namespace std;
using namespace cv;

namespace cp
{
	inline bool isPow2(const int number)
	{
		if (number < 0)return false;
		if (_mm_popcnt_u32(number) == 1)return true;
		else return false;
	}

	void nearestUpsample2(const Mat& src, Mat& dest)
	{
		for (int y = 0; y < src.rows; ++y)
		{
			const float* s = src.ptr<float>(y);
			float* dst = dest.ptr<float>(y * 2);

			for (int x = 0; x < src.cols; ++x)
			{
				dst[x * 2] = s[x];
			}
		}
	}

	//X. Li and M.T.Orchard
	//New edge-directed interpolation
	//IEEE Transactions on Image Processing, vol. 10, issue 10, 2001.

	void NewEdgeDirectedInterpolation::upsampleGrayDoubleLUOpt(const cv::Mat& sim, cv::Mat& dim, const float threshold, const int window_size, const float eps)
	{
		//copy src to dest(2y,2x)
		cp::upsampleCubic(sim, dim, 2, -1.0);
		//nearestUpsample2(sim, dim);

		const int width = dim.cols;
		const int width2 = width * 2;
		const int height = dim.rows;

		const int window_area = window_size * window_size;
#pragma omp parallel for schedule(dynamic)
		for (int y = window_size + 1; y < height - (window_size + 1); y += 2)//(2y+1,2x+1)
		{
			Mat alpha_coeff(1, 4, CV_32F);
			Mat offset(1, 4, CV_32F);
			Mat vectorY(window_area, 1, CV_32F);
			Mat matrixC(window_area, 4, CV_32F);
			Mat CtC(4, 4, CV_32F);
			Mat CInv(4, 4, CV_32F);
			Mat c(4, 1, CV_32F);
			Mat colsum(4, 1, CV_32F);
			Mat rowsum(1, 4, CV_32F);
			float* dst = dim.ptr<float>(y);

			for (int x = window_size + 1; x < width - (window_size + 1); x += 2)
			{
				float sum = 0, ave = 0, var = 0;
				float nn4[4] = { dst[x - 1 - width], dst[x + 1 - width], dst[x - 1 + width], dst[x + 1 + width] };

				for (int i = 0; i < 4; i++)
				{
					sum += nn4[i];
				}

				ave = sum * 0.25f;

				for (int i = 0; i < 4; i++)
				{
					var += (ave - nn4[i]) * (ave - nn4[i]);
				}

				var *= 0.25f;

				if (var >= threshold)
				{
					float* matC = matrixC.ptr<float>();
					float* vecY = vectorY.ptr<float>();

					float sumy = 0.f;
					for (int Y = 0; Y < window_size * 2; Y += 2)
					{
						float* window = dim.ptr<float>(y - (window_size - 1) + Y, x - (window_size - 1));

						for (int X = 0; X < window_size * 2; X += 2)
						{
							float v = window[X];
							float r0 = window[X - 2 - width2];
							float r1 = window[X + 2 - width2];
							float r2 = window[X - 2 + width2];
							float r3 = window[X + 2 + width2];
							float ra = (r0 + r1 + r2 + r3) * 0.25f;

							matC[0] = r0 - ra;
							matC[1] = r1 - ra;
							matC[2] = r2 - ra;
							matC[3] = r3 - ra;

							*vecY++ = v;
							sumy += v;
							matC += 4;
						}
					}

					vectorY -= sumy / window_area;

					mulTransposed(matrixC, CtC, true);
					CtC.at<float>(0, 0) += eps;
					CtC.at<float>(1, 1) += eps;
					CtC.at<float>(2, 2) += eps;
					CtC.at<float>(3, 3) += eps;
					gemm(matrixC, vectorY, 1.0, noArray(), 0.0, c, GEMM_1_T);//matrixC.t()*vectorY;
					solve(CtC, c, alpha_coeff, DECOMP_LU);//MMSE

#if 0
					invert(CtC, CInv, DECOMP_LU);
					Mat onesVector = Mat::ones(4, 1, CV_32FC1);
					Mat temp1 = (onesVector.t() * CInv * onesVector);
					Mat temp2 = (onesVector.t() * CInv * c);
					Mat offset = (CInv * onesVector) * (1.f - temp2.at<float>(0)) / (temp1.at<float>(0) + FLT_EPSILON);
#else
					invert(CtC, CInv, DECOMP_LU);
					float v1 = float(cv::sum(CInv).val[0] + FLT_EPSILON);//(onesVector.t()*CInv*onesVector);//sum all element;
					reduce(CInv, rowsum, 0, REDUCE_SUM, CV_32F);
					float v2 = Mat(rowsum * c).at<float>(0);//Mat temp2 = (onesVector.t()*CInv*c);//ones.t()*C: col sum 
					reduce(CInv, colsum, 1, REDUCE_SUM, CV_32F);//C*one: col sum
					multiply(colsum, (1.f - v2) / (v1), offset);
#endif		
					alpha_coeff += offset;

					float* alpha = alpha_coeff.ptr<float>(0);
					//if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0)
					if (alpha[0] == 0)
					{
						dst[x] = (dst[x - 1 - width] + dst[x + 1 - width] + dst[x - 1 + width] + dst[x + 1 + width]) * 0.25f;
					}
					else
					{
						dst[x] = alpha[0] * dst[x - 1 - width] + alpha[1] * dst[x + 1 - width] + alpha[2] * dst[x - 1 + width] + alpha[3] * dst[x + 1 + width];
					}
				}
			}
		}

		//second step of edge-directed interpolation
		for (int C = 0; C < 2; ++C)// C=0->(2y,2x+1)?CC=1 ->(2y+1,2x)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = window_size * 2 + C; y < dim.rows - window_size * 2 - C; y += 2)
			{
				Mat alpha_coeff(1, 4, CV_32F);
				Mat offset(1, 4, CV_32F);
				Mat vectorY(window_area, 1, CV_32F);
				Mat matrixC(window_area, 4, CV_32F);
				Mat CtC(4, 4, CV_32F);
				Mat CInv(4, 4, CV_32F);
				Mat c(4, 1, CV_32F);
				Mat colsum(4, 1, CV_32F);
				Mat rowsum(1, 4, CV_32F);
				float* dst = dim.ptr<float>(y);

				for (int x = window_size * 2 + 1 - C; x < dim.cols - (window_size * 2 + 1 - C); x += 2)
				{
					//compute ver
					float sum = 0, ave = 0, var = 0;
					{
						float nn4[4] = { dst[x - 1], dst[x - width], dst[x + width], dst[x + 1] };
						for (int i = 0; i < 4; ++i) { sum += nn4[i]; }
						ave = sum / 4;
						for (int i = 0; i < 4; ++i) { var += (ave - nn4[i]) * (ave - nn4[i]); }
						var /= 4;
					}

					if (var >= threshold)
					{
						float* vecY = vectorY.ptr<float>();
						float* matC = matrixC.ptr<float>();

						float sumy = 0.f;
						for (int Y = 0; Y < window_size; ++Y)
						{
							float* window = dim.ptr<float>(y + Y, x - (window_size - 1) + Y);

							for (int X = 0; X < window_size; ++X)
							{
								const int point = (1 - width) * X;

								float v = window[point];

								float r0 = window[point - 2];
								float r1 = window[point - width2];
								float r2 = window[point + width2];
								float r3 = window[point + 2];
								float ra = (r0 + r1 + r2 + r3) * 0.25f;
								matC[0] = r0 - ra;
								matC[1] = r1 - ra;
								matC[2] = r2 - ra;
								matC[3] = r3 - ra;

								*vecY++ = v;
								sumy += v;
								matC += 4;
							}
						}

						vectorY -= sumy / window_area;

						mulTransposed(matrixC, CtC, true);
						CtC.at<float>(0, 0) += eps;
						CtC.at<float>(1, 1) += eps;
						CtC.at<float>(2, 2) += eps;
						CtC.at<float>(3, 3) += eps;
						gemm(matrixC, vectorY, 1.0, noArray(), 0.0, c, GEMM_1_T);//matrixC.t()*vectorY;
						solve(CtC, c, alpha_coeff, DECOMP_LU);//MMSE

#if 0
						Mat CInv = CtC.inv();
						Mat onesVector = Mat::ones(4, 1, CV_32FC1);
						Mat temp1 = (onesVector.t() * CInv * onesVector);
						Mat temp2 = (onesVector.t() * CInv * c);
						Mat offset = (CInv * onesVector) * (1.f - temp2.at<float>(0)) / (temp1.at<float>(0) + FLT_EPSILON);
#else
						invert(CtC, CInv, DECOMP_LU);
						float v1 = float(cv::sum(CInv).val[0] + FLT_EPSILON);//(onesVector.t()*CInv*onesVector);//sum all element;
						reduce(CInv, rowsum, 0, REDUCE_SUM, CV_32F);
						float v2 = Mat(rowsum * c).at<float>(0);//Mat temp2 = (onesVector.t()*CInv*c);//ones.t()*C: col sum 
						reduce(CInv, colsum, 1, REDUCE_SUM, CV_32F);//C*one: col sum
						multiply(colsum, (1.f - v2) / (v1), offset);
#endif
						alpha_coeff += offset;

						float* alpha = alpha_coeff.ptr<float>(0);

						//if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0)
						if (alpha[0] == 0)
						{
							dst[x] = (dst[x - 1] + dst[x - width] + dst[x + width] + dst[x + 1]) * 0.25f;
						}
						else
						{
							dst[x] = alpha[0] * dst[x - 1] + alpha[1] * dst[x - width] + alpha[2] * dst[x + width] + alpha[3] * dst[x + 1];
						}
					}
				}
			}
		}
		return;
	}

	void NewEdgeDirectedInterpolation::upsampleGrayDoubleLU(const cv::Mat& sim, cv::Mat& dim, const float threshold, const int window_size, const float eps)
	{
		//copy src to dest(2y,2x)
		cp::upsampleCubic_parallel(sim, dim, 2, -1.5);
		//cp::nnUpsample(sim ,dim);

		const int width = dim.cols;
		const int width2 = width * 2;
		const int height = dim.rows;

		const bool isnormalize = true;
#pragma omp parallel for schedule(dynamic)
		for (int y = window_size + 1; y < height - (window_size + 1); y += 2)//(2y+1,2x+1)
		{
			Mat alpha_coeff(1, 4, CV_32F);
			Mat matrixC(window_size * window_size, 4, CV_32F);
			Mat CtC(4, 4, CV_32F);
			Mat c(4, 1, CV_32F);
			float* dst = dim.ptr<float>(y);

			for (int x = window_size + 1; x < width - (window_size + 1); x += 2)
			{
				float sum = 0, ave = 0, var = 0;
				float nn4[4] = { dst[x - 1 - width], dst[x + 1 - width], dst[x - 1 + width], dst[x + 1 + width] };

				for (int i = 0; i < 4; i++)
				{
					sum += nn4[i];
				}

				ave = sum * 0.25f;

				for (int i = 0; i < 4; i++)
				{
					var += (ave - nn4[i]) * (ave - nn4[i]);
				}

				var *= 0.25f;

				if (var >= threshold)
				{
					float* matC = matrixC.ptr<float>();
					float* cptr = c.ptr<float>();
					c.setTo(0.f);
					for (int Y = 0; Y < window_size * 2; Y += 2)
					{
						float* window = dim.ptr<float>(y - (window_size - 1) + Y, x - (window_size - 1));

						for (int X = 0; X < window_size * 2; X += 2)
						{
							matC[0] = window[X - 2 - width2];
							matC[1] = window[X + 2 - width2];
							matC[2] = window[X - 2 + width2];
							matC[3] = window[X + 2 + width2];

							float v = window[X];
							cptr[0] += v * matC[0];
							cptr[1] += v * matC[1];
							cptr[2] += v * matC[2];
							cptr[3] += v * matC[3];

							matC += 4;
						}
					}

					mulTransposed(matrixC, CtC, true);
					CtC.at<float>(0, 0) += eps;
					CtC.at<float>(1, 1) += eps;
					CtC.at<float>(2, 2) += eps;
					CtC.at<float>(3, 3) += eps;
					solve(CtC, c, alpha_coeff, DECOMP_LU);//MMSE

					if (isnormalize)
					{
						float asum = 1.f / (alpha_coeff.at<float>(0) + alpha_coeff.at<float>(1) + alpha_coeff.at<float>(2) + alpha_coeff.at<float>(3));
						alpha_coeff *= asum;
					}
					float* alpha = alpha_coeff.ptr<float>(0);

					//if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0)
					if (alpha[0] == 0)
					{
						dst[x] = (dst[x - 1 - width] + dst[x + 1 - width] + dst[x - 1 + width] + dst[x + 1 + width]) * 0.25f;
					}
					else
					{
						dst[x] = alpha[0] * dst[x - 1 - width] + alpha[1] * dst[x + 1 - width] + alpha[2] * dst[x - 1 + width] + alpha[3] * dst[x + 1 + width];
					}
				}
			}
		}

		//second step of edge-directed interpolation
		for (int C = 0; C < 2; ++C)// C=0->(2y,2x+1)?CC=1 ->(2y+1,2x)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = window_size * 2 + C; y < dim.rows - window_size * 2 - C; y += 2)
			{
				Mat alpha_coeff(1, 4, CV_32F);
				Mat matrixC(window_size * window_size, 4, CV_32F);
				Mat CtC(4, 4, CV_32F);
				Mat c(4, 1, CV_32F);

				float* dst = dim.ptr<float>(y);

				for (int x = window_size * 2 + 1 - C; x < dim.cols - (window_size * 2 + 1 - C); x += 2)
				{
					//compute ver
					float sum = 0, ave = 0, var = 0;
					{
						float nn4[4] = { dst[x - 1], dst[x - width], dst[x + width], dst[x + 1] };
						for (int i = 0; i < 4; ++i) { sum += nn4[i]; }
						ave = sum / 4;
						for (int i = 0; i < 4; ++i) { var += (ave - nn4[i]) * (ave - nn4[i]); }
						var /= 4;
					}

					if (var >= threshold)
					{
						float* matC = matrixC.ptr<float>();
						float* cptr = c.ptr<float>();
						c.setTo(0.f);
						for (int Y = 0; Y < window_size; ++Y)
						{
							float* window = dim.ptr<float>(y + Y, x - (window_size - 1) + Y);

							for (int X = 0; X < window_size; ++X)
							{
								const int point = (1 - width) * X;

								matC[0] = window[point - 2];
								matC[1] = window[point - width2];
								matC[2] = window[point + width2];
								matC[3] = window[point + 2];

								float v = window[point];
								cptr[0] += v * matC[0];
								cptr[1] += v * matC[1];
								cptr[2] += v * matC[2];
								cptr[3] += v * matC[3];

								matC += 4;
							}
						}

						mulTransposed(matrixC, CtC, true);
						CtC.at<float>(0, 0) += eps;
						CtC.at<float>(1, 1) += eps;
						CtC.at<float>(2, 2) += eps;
						CtC.at<float>(3, 3) += eps;
						solve(CtC, c, alpha_coeff, DECOMP_LU);//MMSE

						if (isnormalize)
						{
							float asum = 1.f / (alpha_coeff.at<float>(0) + alpha_coeff.at<float>(1) + alpha_coeff.at<float>(2) + alpha_coeff.at<float>(3));
							alpha_coeff *= asum;
						}

						float* alpha = alpha_coeff.ptr<float>(0);

						//if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0)
						if (alpha[0] == 0)
						{
							dst[x] = (dst[x - 1] + dst[x - width] + dst[x + width] + dst[x + 1]) * 0.25f;
						}
						else
						{
							dst[x] = alpha[0] * dst[x - 1] + alpha[1] * dst[x - width] + alpha[2] * dst[x + width] + alpha[3] * dst[x + 1];
						}
					}
				}
			}
		}
		return;
	}

	void NewEdgeDirectedInterpolation::upsampleGrayDoubleQR(const cv::Mat& sim, cv::Mat& dim, const float threshold, const int window_size)
	{
		//copy src to dest(2y,2x)

		cp::upsampleCubic_parallel(sim, dim, 2, -0.5);
		//nearestUpsample2(sim, dim);

		const int width = dim.cols;
		const int width2 = width * 2;
		const int height = dim.rows;

#pragma omp parallel for schedule(dynamic)
		for (int y = window_size + 1; y < height - (window_size + 1); y += 2)//(2y+1,2x+1)
		{
			Mat alpha_coeff(1, 4, CV_32F);
			Mat vectorY(window_size * window_size, 1, CV_32F);
			Mat matrixC(window_size * window_size, 4, CV_32F);

			float* dst = dim.ptr<float>(y);

			for (int x = window_size + 1; x < width - (window_size + 1); x += 2)
			{
				float sum = 0, ave = 0, var = 0;
				float nn4[4] = { dst[x - 1 - width], dst[x + 1 - width], dst[x - 1 + width], dst[x + 1 + width] };

				for (int i = 0; i < 4; i++)
				{
					sum += nn4[i];
				}

				ave = sum * 0.25f;

				for (int i = 0; i < 4; i++)
				{
					var += (ave - nn4[i]) * (ave - nn4[i]);
				}

				var *= 0.25f;

				if (var >= threshold)
				{
					float* matC = matrixC.ptr<float>();
					float* vecY = vectorY.ptr<float>();
					for (int Y = 0; Y < window_size * 2; Y += 2)
					{
						float* window = dim.ptr<float>(y - (window_size - 1) + Y, x - (window_size - 1));

						for (int X = 0; X < window_size * 2; X += 2)
						{
							matC[0] = window[X - 2 - width2];
							matC[1] = window[X + 2 - width2];
							matC[2] = window[X - 2 + width2];
							matC[3] = window[X + 2 + width2];
							*vecY++ = window[X];
							matC += 4;
						}
					}

					solve(matrixC, vectorY, alpha_coeff, DECOMP_QR);//MMSE
					float* alpha = alpha_coeff.ptr<float>(0);

					//if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0)
					if (alpha[0] == 0)
					{
						dst[x] = (dst[x - 1 - width] + dst[x + 1 - width] + dst[x - 1 + width] + dst[x + 1 + width]) * 0.25f;
					}
					else
					{
						dst[x] = alpha[0] * dst[x - 1 - width] + alpha[1] * dst[x + 1 - width] + alpha[2] * dst[x - 1 + width] + alpha[3] * dst[x + 1 + width];
					}
				}
			}
		}

		//second step of edge-directed interpolation
		for (int C = 0; C < 2; ++C)// C=0->(2y,2x+1)?CC=1 ->(2y+1,2x)
		{
#pragma omp parallel for schedule(dynamic)
			for (int y = window_size * 2 + C; y < dim.rows - window_size * 2 - C; y += 2)
			{
				Mat alpha_coeff(1, 4, CV_32F);
				Mat vectorY(window_size * window_size, 1, CV_32F);
				Mat matrixC(window_size * window_size, 4, CV_32F);

				float* dst = dim.ptr<float>(y);

				for (int x = window_size * 2 + 1 - C; x < dim.cols - (window_size * 2 + 1 - C); x += 2)
				{
					//compute ver
					float sum = 0, ave = 0, var = 0;
					{
						float nn4[4] = { dst[x - 1], dst[x - width], dst[x + width], dst[x + 1] };
						for (int i = 0; i < 4; ++i) { sum += nn4[i]; }
						ave = sum / 4;
						for (int i = 0; i < 4; ++i) { var += (ave - nn4[i]) * (ave - nn4[i]); }
						var /= 4;
					}

					if (var >= threshold)
					{
						int count = 0, point = 0;
						float* vecY = vectorY.ptr<float>();
						float* matC = matrixC.ptr<float>();

						for (int Y = 0; Y < window_size; ++Y)
						{
							float* window = dim.ptr<float>(y + Y, x - (window_size - 1) + Y);

							for (int X = 0; X < window_size; ++X)
							{
								point = (1 - width) * X;

								matC[0] = window[point - 2];
								matC[1] = window[point - width2];
								matC[2] = window[point + width2];
								matC[3] = window[point + 2];

								matC += 4;
								*vecY++ = window[point];
							}
						}

						solve(matrixC, vectorY, alpha_coeff, DECOMP_QR);//MMSE

						float* alpha = alpha_coeff.ptr<float>(0);

						//if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0)
						if (alpha[0] == 0)
						{
							dst[x] = (dst[x - 1] + dst[x - width] + dst[x + width] + dst[x + 1]) * 0.25f;
						}
						else
						{
							dst[x] = alpha[0] * dst[x - 1] + alpha[1] * dst[x - width] + alpha[2] * dst[x + width] + alpha[3] * dst[x + 1];
						}
					}
				}
			}
		}
		return;
	}

	void NewEdgeDirectedInterpolation::upsample(InputArray src, OutputArray dest, const int scale, const float threshold, const int WindowSize, int method)
	{
		CV_Assert(!src.empty());
		CV_Assert(isPow2(scale));

		CV_Assert(src.channels() == 1 || src.channels() == 3);
		CV_Assert(WindowSize % 2 == 0 && WindowSize != 0);

		const int border = BORDER_REFLECT101;

		static int eps = 1; createTrackbar("epsss", "", &eps, 1000);

		if (scale == 1)
		{
			src.copyTo(dest);
			return;
		}

		int level = 0;
		int v = scale;
		while (v != 1)
		{
			level++;
			v /= 2;
		}

		if (dest_border.size() != level)
		{
			dest_border.resize(level);
			image_border.resize(level);
		}

		Mat srcfImg;
		Mat destf;
		if (src.channels() == 1)
		{
			src.getMat().convertTo(srcfImg, CV_32FC1, 1.0f / 255);

			for (int i = 0; i < level; i++)
			{
				if (i == 0)
				{
					copyMakeBorder(srcfImg, image_border[i], WindowSize + 1, WindowSize + 1, WindowSize + 1, WindowSize + 1, border);
					dest_border[i].create(image_border[i].size() * 2, CV_32FC1);
					dest_border[i].setTo(0);
				}
				else
				{
					Mat crop = dest_border[i - 1](Rect((WindowSize + 1) * 2, (WindowSize + 1) * 2, dest_border[i - 1].cols - (WindowSize + 1) * 4, dest_border[i - 1].rows - (WindowSize + 1) * 4));

					copyMakeBorder(crop, image_border[i], WindowSize + 1, WindowSize + 1, WindowSize + 1, WindowSize + 1, border);
					dest_border[i].create(image_border[i].size() * 2, CV_32FC1);
					dest_border[i].setTo(0);
				}

				if (method == 0)upsampleGrayDoubleLU(image_border[i], dest_border[i], threshold, WindowSize, float(eps * 1.0 / 2560.0));
				else if (method == 1)
				{
					//upsampleGrayDoubleLU(image_border[i], dest_border[i], threshold, WindowSize);
					upsampleGrayDoubleLUOpt(image_border[i], dest_border[i], threshold, WindowSize, float(eps * 1.0 / 2560.0));
				}
			}

			int i = level - 1;
			dest_border[i](Rect((WindowSize + 1) * 2, (WindowSize + 1) * 2, dest_border[i].cols - (WindowSize + 1) * 4, dest_border[i].rows - (WindowSize + 1) * 4)).copyTo(destf);
			cout<<destf.size() << endl;
			destf.convertTo(dest, src.depth(), 255);
		}
		else if (src.channels() == 3)
		{
			vector<Mat> planes(3), dst_rgb(3);

			split(src, planes);
			for (int c = 0; c < 3; c++)
			{
				planes[c].convertTo(srcfImg, CV_32FC1, 1.f / 255.f);
				for (int i = 0; i < level; i++)
				{
					if (i == 0)
					{
						copyMakeBorder(srcfImg, image_border[i], WindowSize + 1, WindowSize + 1, WindowSize + 1, WindowSize + 1, border);
						dest_border[i].create(image_border[i].size() * 2, CV_32FC1);
						dest_border[i].setTo(0);
					}
					else
					{
						Mat crop = dest_border[i - 1](Rect((WindowSize + 1) * 2, (WindowSize + 1) * 2, dest_border[i - 1].cols - (WindowSize + 1) * 4, dest_border[i - 1].rows - (WindowSize + 1) * 4));

						copyMakeBorder(crop, image_border[i], WindowSize + 1, WindowSize + 1, WindowSize + 1, WindowSize + 1, border);
						dest_border[i].create(image_border[i].size() * 2, CV_32FC1);
						dest_border[i].setTo(0);
					}

					if (method == 0)upsampleGrayDoubleLU(image_border[i], dest_border[i], threshold, WindowSize, float(eps * 1.0 / 2560.0));
					else if (method == 1)
					{
						//upsampleGrayDoubleLU(image_border[i], dest_border[i], threshold, WindowSize);
						upsampleGrayDoubleLUOpt(image_border[i], dest_border[i], threshold, WindowSize, float(eps * 1.0 / 2560.0));
					}

				}

				int i = level - 1;
				dest_border[i](Rect((WindowSize + 1) * 2, (WindowSize + 1) * 2, dest_border[i].cols - (WindowSize + 1) * 4, dest_border[i].rows - (WindowSize + 1) * 4)).copyTo(dst_rgb[c]);
			}

			merge(dst_rgb, destf);
		}

		destf.convertTo(dest, src.depth(), 255);
	}
}

#if 0
// old
void NEDI16(const Mat& srcImg, Mat& dstImg_, int n_pow, float threshold, int WindowSize)
{
	CV_Assert(!srcImg.empty());
	//1?{
	if (n_pow == 0)
	{
		dstImg_ = srcImg.clone();
		return;
	}

	CV_Assert(n_pow > 0);
	CV_Assert(srcImg.channels() == 1 || srcImg.channels() == 3);
	CV_Assert(WindowSize % 2 == 0 && WindowSize != 0);

	Mat dstImg;
	/*if (srcImg.channels() == 3)//?J???[
	{
		vector<Mat> planes(3), dst_rgb(3);

		split(srcImg, planes);

		NEDI(planes[0], dst_rgb[0], n_pow);
		NEDI(planes[1], dst_rgb[1], n_pow);
		NEDI(planes[2], dst_rgb[2], n_pow);

		merge(dst_rgb, dstImg);
		return;
	}*/

	int srcType = srcImg.type();
	Mat srcfImg;
	for (int iteration = n_pow; iteration > 0; iteration--)
	{
		if (iteration == n_pow)
		{
			copyMakeBorder(srcImg, srcfImg, WindowSize + 4, WindowSize + 4, WindowSize + 4, WindowSize + 4, BORDER_REPLICATE);
			srcfImg.convertTo(srcfImg, CV_32FC1, 1.0f / 255);

			dstImg = Mat::zeros(srcfImg.size() * 2, CV_32FC1);
		}
		else
		{
			copyMakeBorder(dstImg, srcfImg, WindowSize + 4, WindowSize + 4, WindowSize + 4, WindowSize + 4, BORDER_REPLICATE);
			dstImg = Mat::zeros(srcfImg.size() * 2, CV_32FC1);
		}

		//#pragma omp parallel for
		for (int y = 0; y < srcfImg.rows; ++y)
		{
			float* src = srcfImg.ptr<float>(y);
			float* dst = dstImg.ptr<float>(y * 2);

			for (int x = 0; x < srcfImg.cols; ++x)
			{
				dst[x * 2] = src[x];
			}
		}

		const int width = dstImg.cols;
		const int b = WindowSize + 4;
		//first step of edge-directed interpolation
#pragma omp parallel for
		for (int y = b; y < dstImg.rows - b; y += 2)//(2y+1,2x+1)
		{
			Mat alpha_coeff = Mat::zeros(Size(16, 1), dstImg.type());
			Mat vectorY = Mat::zeros(Size(WindowSize * WindowSize, 1), dstImg.type());
			Mat matrixC = Mat::zeros(Size(16, WindowSize * WindowSize), dstImg.type());

			float* dst = dstImg.ptr<float>(y + 1, 1);
			float* vecY = vectorY.ptr<float>(0);

			for (int x = b; x < dstImg.cols - b; x += 2)
			{
				int count = 0;

				for (int Y = 0; Y < WindowSize * 2; Y += 2)
				{
					float* window = dstImg.ptr<float>(y - (WindowSize + 0) + Y, x - (WindowSize + 0));

					for (int X = 0; X < WindowSize * 2; X += 2)
					{
						float* matC = matrixC.ptr<float>(count);
						/*
						matC[0] = window[X - 2 - width * 2];
						matC[1] = window[X + 2 - width * 2];
						matC[2] = window[X - 2 + width * 2];
						matC[3] = window[X + 2 + width * 2];
						*/

						matC[0] = window[X - 4 - width * 4];
						matC[1] = window[X - 2 - width * 4];
						matC[2] = window[X + 2 - width * 4];
						matC[3] = window[X + 4 - width * 4];

						matC[4] = window[X - 4 - width * 2];
						matC[5] = window[X - 2 - width * 2];
						matC[6] = window[X + 2 - width * 2];
						matC[7] = window[X + 4 - width * 2];

						matC[8] = window[X - 4 + width * 2];
						matC[9] = window[X - 2 + width * 2];
						matC[10] = window[X + 2 + width * 2];
						matC[11] = window[X + 4 + width * 2];

						matC[12] = window[X - 4 + width * 4];
						matC[13] = window[X - 2 + width * 4];
						matC[14] = window[X + 2 + width * 4];
						matC[15] = window[X + 4 + width * 4];

						vecY[count++] = window[X];
					}
				}

				solve(matrixC.t() * matrixC, matrixC.t() * vectorY.t(), alpha_coeff, DECOMP_LU);//MMSE

				float* a = alpha_coeff.ptr<float>(0);

				if (a[0] == 0.f)
				{
					dst[x] = (
						+1.f * dst[x - 3 - 3 * width] - 9.f * dst[x - 1 - 3 * width] - 9.f * dst[x + 1 - 3 * width] + 1.f * dst[x + 3 - 3 * width]
						- 9.f * dst[x - 3 - 1 * width] + 81.f * dst[x - 1 - 1 * width] + 81.f * dst[x + 1 - 1 * width] - 9.f * dst[x + 3 - 1 * width]
						- 9.f * dst[x - 3 + 1 * width] + 81.f * dst[x - 1 + 1 * width] + 81.f * dst[x + 1 + 1 * width] - 9.f * dst[x + 3 + 1 * width]
						+ 1.f * dst[x - 3 + 3 * width] - 9.f * dst[x - 1 + 3 * width] - 9.f * dst[x + 1 + 3 * width] + 1.f * dst[x + 3 + 3 * width]
						) / 256.0f;

					//dst[x] = (dst[x - 1 - width] + dst[x + 1 - width] + dst[x - 1 + width] + dst[x + 1 + width]) / 4.0f;
				}
				else
				{


					//dst[x] = a[0] * dst[x - 1 - width] + a[1] * dst[x + 1 - width] + a[2] * dst[x - 1 + width] + a[3] * dst[x + 1 + width];

					dst[x] = (
						+a[0] * dst[x - 3 - 3 * width] + a[1] * dst[x - 1 - 3 * width] + a[2] * dst[x + 1 - 3 * width] + a[3] * dst[x + 3 - 3 * width]
						+ a[4] * dst[x - 3 - 1 * width] + a[5] * dst[x - 1 - 1 * width] + a[6] * dst[x + 1 - 1 * width] + a[7] * dst[x + 3 - 1 * width]
						+ a[8] * dst[x - 3 + 1 * width] + a[9] * dst[x - 1 + 1 * width] + a[10] * dst[x + 1 + 1 * width] + a[11] * dst[x + 3 + 1 * width]
						+ a[12] * dst[x - 3 + 3 * width] + a[13] * dst[x - 1 + 3 * width] + a[14] * dst[x + 1 + 3 * width] + a[15] * dst[x + 3 + 3 * width]
						);

				}
			}
		}


		//second step of edge-directed interpolation
		for (int C = 0; C < 2; ++C)// C=0->(2y,2x+1)?CC=1 ->(2y+1,2x)
		{
#pragma omp parallel for
			for (int y = WindowSize * 2 + C; y < dstImg.rows - WindowSize * 2 - C; y += 2)
			{
				Mat alpha_coeff(Size(4, 1), dstImg.type());
				Mat vectorY(Size(WindowSize * WindowSize, 1), dstImg.type());
				Mat matrixC(Size(4, WindowSize * WindowSize), dstImg.type());

				float* vecY = vectorY.ptr<float>(0);
				float* dst = dstImg.ptr<float>(y + 1);

				for (int x = WindowSize * 2 + 4 - C; x < dstImg.cols - (WindowSize * 2 + 4 - C); x += 2)
				{
					int count = 0, point = 0;

					for (int Y = 0; Y < WindowSize; ++Y)
					{
						float* window = dstImg.ptr<float>(y + Y, x);

						for (int X = 0; X < WindowSize; ++X)
						{
							float* matC = matrixC.ptr<float>(count);

							point = -(WindowSize - 4) - width * X + X + Y;

							matC[0] = window[point - 2];
							matC[1] = window[point - width * 2];
							matC[2] = window[point + width * 2];
							matC[3] = window[point + 2];

							vecY[count++] = window[point];
					}
				}

					solve(matrixC.t() * matrixC, matrixC.t() * vectorY.t(), alpha_coeff, DECOMP_LU);//MMSE

					float* alpha = alpha_coeff.ptr<float>(0);

					if (alpha[0] == alpha[1] && alpha[1] == alpha[2] && alpha[2] == alpha[3] && alpha[3] == 0.f)
					{
						dst[x] = (dst[x - 1] + dst[x - width] + dst[x + width] + dst[x + 1]) / 4.0f;
					}
					else
					{
						dst[x] = alpha[0] * dst[x - 1] + alpha[1] * dst[x - width] + alpha[2] * dst[x + width] + alpha[3] * dst[x + 1];
					}
			}
		}
	}

		dstImg = dstImg(Rect((WindowSize + 4) * 2, (WindowSize + 4) * 2, dstImg.cols - (WindowSize + 4) * 4, dstImg.rows - (WindowSize + 4) * 4));

}

	dstImg.convertTo(dstImg_, srcType, 255);

	return;
}
#endif

