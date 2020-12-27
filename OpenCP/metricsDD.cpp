#include "metricsDD.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	double PSNR_DD(const doubledouble* src1, const doubledouble* src2, const int size)
	{
		doubledouble ret = doubledouble{ 0.0,0.0 };
		doubledouble m = doubledouble{ -1.0,0.0 };
		for (int i = 0; i < size; i++)
		{
			doubledouble v = ddmul(src1[i], m);
			v = ddadd(src2[i], v);
			v = ddmul(v, v);
			ret = ddadd(ret, v);
		}
		double mse = ret.hi / (double)size;

		if (mse == 0.0)
		{
			return 0;
		}
		else if (cvIsNaN(mse) || cvIsInf(mse))
		{
			cout << "mse = NaN" << endl;
			return 0;
		}
		else if (cvIsInf(mse))
		{
			cout << "mse = Inf" << endl;
			return 0;
		}

		return 10.0 * log10(255.0 * 255.0 / mse);
	}

	double PSNR_DD(doubledouble* src1, cv::InputArray& src2_)
	{
		Mat src2 = src2_.getMat();
		const int size = src2.size().area();
		doubledouble ret = doubledouble{ 0.0, 0.0 };

		if (src2.depth() == CV_64F)
		{
			for (int i = 0; i < size; i++)
			{
				doubledouble v = ddaddw(src1[i], -src2.at<double>(i));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
		}
		else if (src2.depth() == CV_32F)
		{
			for (int i = 0; i < size; i++)
			{
				doubledouble v = ddaddw(src1[i], -src2.at<float>(i));
				v = ddmul(v, v);
				ret = ddadd(ret, v);
			}
		}
		double mse = ret.hi / (double)size;

		if (mse == 0.0)
		{
			return 0.0;
		}
		else if (cvIsNaN(mse) || cvIsInf(mse))
		{
			cout << "mse = NaN" << endl;
			return 0.0;
		}
		else if (cvIsInf(mse))
		{
			cout << "mse = Inf" << endl;
			return 0.0;
		}

		return 10.0 * log10(255.0 * 255.0 / mse);
	}
}