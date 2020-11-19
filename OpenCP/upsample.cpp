#include "upsample.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	template <class srcType>
	static void nnUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols);
		const int dh = dest.rows / (src.rows);

		Mat sim; copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		for (int j = 0; j < src.rows; j++)
		{
			int n = j * dh;
			srcType* s = sim.ptr<srcType>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType ltd = s[i];
				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						d[m + k] = ltd;
					}
				}
			}
		}
	}

	void nnUpsample(InputArray src_, OutputArray dest_)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) nnUpsample_<uchar>(src, dest);
		else if (src.depth() == CV_16S) nnUpsample_<short>(src, dest);
		else if (src.depth() == CV_16U) nnUpsample_<ushort>(src, dest);
		else if (src.depth() == CV_32S) nnUpsample_<int>(src, dest);
		else if (src.depth() == CV_32F) nnUpsample_<float>(src, dest);
		else if (src.depth() == CV_64F) nnUpsample_<double>(src, dest);
	}

	inline int linearinterpolate_(int lt, int rt, int lb, int rb, double a, double b)
	{
		return (int)((b * a * lt + b * (1.0 - a) * rt + (1.0 - b) * a * lb + (1.0 - b) * (1.0 - a) * rb) + 0.5);
	}

	template <class srcType>
	inline double linearinterpolate_(srcType lt, srcType rt, srcType lb, srcType rb, double a, double b)
	{
		return (b * a * lt + b * (1.0 - a) * rt + (1.0 - b) * a * lb + (1.0 - b) * (1.0 - a) * rb);
	}

	template <class srcType>
	static void linearUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		Mat sim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, BORDER_REPLICATE);
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * dh;
			srcType* s = sim.ptr<srcType>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType ltd = s[i];
				const srcType rtd = s[i + 1];
				const srcType lbd = s[i + sim.cols];
				const srcType rbd = s[i + 1 + sim.cols];
				for (int l = 0; l < dh; l++)
				{
					double beta = 1.0 - (double)l / dh;
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;
						d[m + k] = saturate_cast<srcType> (linearinterpolate_<srcType>(ltd, rtd, lbd, rbd, alpha, beta));
					}
				}
			}
		}
	}

	void linearUpsample(InputArray src_, OutputArray dest_)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) linearUpsample_<uchar>(src, dest);
		else if (src.depth() == CV_16S) linearUpsample_<short>(src, dest);
		else if (src.depth() == CV_16U) linearUpsample_<ushort>(src, dest);
		else if (src.depth() == CV_32S) linearUpsample_<int>(src, dest);
		else if (src.depth() == CV_32F) linearUpsample_<float>(src, dest);
		else if (src.depth() == CV_64F) linearUpsample_<double>(src, dest);
	}

	inline double cubicfunc(double x, double a = -1.0)
	{
		double X = abs(x);
		if (X <= 1)
			return ((a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0);
		else if (X <= 2)
			return (a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a);
		else
			return 0.0;
	}

	template <class srcType>
	static void cubicUpsample_(Mat& src, Mat& dest, double a)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		vector<vector<double>> weight(dh * dw);
		for (int i = 0; i < weight.size(); i++)weight[i].resize(16);

		int idx = 0;

		for (int l = 0; l < dh; l++)
		{
			const double y = (double)l / (double)dh;
			for (int k = 0; k < dw; k++)
			{
				const double x = (double)k / (double)dw;

				weight[idx][0] = cubicfunc(1.0 + x, a) * cubicfunc(1.0 + y, a);
				weight[idx][1] = cubicfunc(0.0 + x, a) * cubicfunc(1.0 + y, a);
				weight[idx][2] = cubicfunc(1.0 - x, a) * cubicfunc(1.0 + y, a);
				weight[idx][3] = cubicfunc(2.0 - x, a) * cubicfunc(1.0 + y, a);

				weight[idx][4] = cubicfunc(1.0 + x, a) * cubicfunc(0.0 + y, a);
				weight[idx][5] = cubicfunc(0.0 + x, a) * cubicfunc(0.0 + y, a);
				weight[idx][6] = cubicfunc(1.0 - x, a) * cubicfunc(0.0 + y, a);
				weight[idx][7] = cubicfunc(2.0 - x, a) * cubicfunc(0.0 + y, a);

				weight[idx][8] = cubicfunc(1.0 + x, a) * cubicfunc(1.0 - y, a);
				weight[idx][9] = cubicfunc(0.0 + x, a) * cubicfunc(1.0 - y, a);
				weight[idx][10] = cubicfunc(1.0 - x, a) * cubicfunc(1.0 - y, a);
				weight[idx][11] = cubicfunc(2.0 - x, a) * cubicfunc(1.0 - y, a);

				weight[idx][12] = cubicfunc(1.0 + x, a) * cubicfunc(2.0 - y, a);
				weight[idx][13] = cubicfunc(0.0 + x, a) * cubicfunc(2.0 - y, a);
				weight[idx][14] = cubicfunc(1.0 - x, a) * cubicfunc(2.0 - y, a);
				weight[idx][15] = cubicfunc(2.0 - x, a) * cubicfunc(2.0 - y, a);

				double wsum = 0.0;
				for (int i = 0; i < 16; i++)wsum += weight[idx][i];
				for (int i = 0; i < 16; i++)weight[idx][i] /= wsum;

				idx++;
			}
		}

		Mat sim;
		copyMakeBorder(src, sim, 1, 2, 1, 2, BORDER_REPLICATE);
		for (int j = 0; j < src.rows; j++)
		{
			int n = j * dh;
			srcType* s = sim.ptr<srcType>(j + 1) + 1;

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const srcType v00 = s[i - 1 - sim.cols];
				const srcType v01 = s[i - 0 - sim.cols];
				const srcType v02 = s[i + 1 - sim.cols];
				const srcType v03 = s[i + 2 - sim.cols];
				const srcType v10 = s[i - 1];
				const srcType v11 = s[i - 0];
				const srcType v12 = s[i + 1];
				const srcType v13 = s[i + 2];
				const srcType v20 = s[i - 1 + sim.cols];
				const srcType v21 = s[i - 0 + sim.cols];
				const srcType v22 = s[i + 1 + sim.cols];
				const srcType v23 = s[i + 2 + sim.cols];
				const srcType v30 = s[i - 1 + 2 * sim.cols];
				const srcType v31 = s[i - 0 + 2 * sim.cols];
				const srcType v32 = s[i + 1 + 2 * sim.cols];
				const srcType v33 = s[i + 2 + 2 * sim.cols];

				int idx = 0;
				for (int l = 0; l < dh; l++)
				{
					srcType* d = dest.ptr<srcType>(n + l);
					for (int k = 0; k < dw; k++)
					{

						d[m + k] = saturate_cast<srcType>(
							weight[idx][0] * v00 + weight[idx][1] * v01 + weight[idx][2] * v02 + weight[idx][3] * v03
							+ weight[idx][4] * v10 + weight[idx][5] * v11 + weight[idx][6] * v12 + weight[idx][7] * v13
							+ weight[idx][8] * v20 + weight[idx][9] * v21 + weight[idx][10] * v22 + weight[idx][11] * v23
							+ weight[idx][12] * v30 + weight[idx][13] * v31 + weight[idx][14] * v32 + weight[idx][15] * v33
							);
						idx++;
					}
				}
			}
		}
	}

	void cubicUpsample(InputArray src_, OutputArray dest_, double a)
	{
		Mat dest = dest_.getMat();
		Mat src = src_.getMat();

		if (src.depth() == CV_8U) cubicUpsample_<uchar>(src, dest, a);
		else if (src.depth() == CV_16S) cubicUpsample_<short>(src, dest, a);
		else if (src.depth() == CV_16U) cubicUpsample_<ushort>(src, dest, a);
		else if (src.depth() == CV_32S) cubicUpsample_<int>(src, dest, a);
		else if (src.depth() == CV_32F) cubicUpsample_<float>(src, dest, a);
		else if (src.depth() == CV_64F) cubicUpsample_<double>(src, dest, a);
	}

	void setUpsampleMask(InputArray src, OutputArray dst)
	{
		Mat dest = dst.getMat();
		if (dest.empty())
		{
			cout << "please alloc dest Mat" << endl;
			return;
		}
		dest.setTo(0);
		const int dw = dest.cols / (src.size().width);
		const int dh = dest.rows / (src.size().height);

		for (int j = 0; j < src.size().height; j++)
		{
			int n = j * dh;
			uchar* d = dest.ptr<uchar>(n);
			for (int i = 0, m = 0; i < src.size().width; i++, m += dw)
			{
				d[m] = 255;
			}
		}
	}
}