#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	double PSNR64F(InputArray I1_, InputArray I2_)
	{

		Mat I1, I2;
		if (I1_.channels() == 1 && I2_.channels() == 1)
		{
			I1_.getMat().convertTo(I1, CV_64F);
			I2_.getMat().convertTo(I2, CV_64F);
		}
		if (I1_.channels() == 3 && I2_.channels() == 3)
		{
			Mat temp;
			cvtColor(I1_, temp, COLOR_BGR2GRAY);
			temp.convertTo(I1, CV_64F);
			cvtColor(I2_, temp, COLOR_BGR2GRAY);
			temp.convertTo(I2, CV_64F);
		}

		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1 = s1.mul(s1);           // |I1 - I2|^2

		Scalar s = sum(s1);        // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if (sse <= 1e-10) // for small values return zero
			return 0;
		else
		{
			double mse = sse / (double)(I1.channels() * I1.total());
			double psnr = 10.0 * log10((255.0 * 255.0) / mse);
			return psnr;
		}
	}

	double MSE(InputArray I1_, InputArray I2_, InputArray mask)
	{
		Mat I1, I2;
		I1_.getMat().convertTo(I1, CV_64F);
		I2_.getMat().convertTo(I2, CV_64F);

		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1 = s1.mul(s1);           // |I1 - I2|^2
		Mat data = Mat::zeros(I1.size(), I1.type());
		int count = countNonZero(mask);
		s1.copyTo(data, mask);
		Scalar s = sum(s1);        // sum elements per channel

		double sse;
		if (I1.channels() == 1) sse = s.val[0];
		else sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		return sse / (double)(I1.channels() * count);
	}

	double MSE(InputArray I1_, InputArray I2_)
	{
		Mat I1, I2;
		I1_.getMat().convertTo(I1, CV_64F);
		I2_.getMat().convertTo(I2, CV_64F);

		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1 = s1.mul(s1);           // |I1 - I2|^2
		Scalar s = sum(s1);        // sum elements per channel
		double sse;
		if (I1.channels() == 1) sse = s.val[0];
		else sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		return sse / (double)(I1.channels() * I1.total());
	}

	double YPSNR(InputArray src1, InputArray src2)
	{
		Mat g1, g2;
		if (src1.channels() == 1) g1 = src1.getMat();
		else cvtColor(src1, g1, COLOR_BGR2GRAY);
		if (src2.channels() == 1) g2 = src2.getMat();
		else cvtColor(src2, g2, COLOR_BGR2GRAY);

		return PSNR(g1, g2);
	}

	double calcBadPixel(const Mat& src, const Mat& ref, int threshold)
	{
		Mat g1, g2;
		if (src.channels() == 3)
		{
			cvtColor(src, g1, CV_BGR2GRAY);
			cvtColor(ref, g2, CV_BGR2GRAY);
		}
		else
		{
			g1 = src;
			g2 = ref;
		}
		Mat temp;
		absdiff(g1, g2, temp);
		Mat mask;
		compare(temp, threshold, mask, CMP_GE);
		return 100.0*countNonZero(mask) / src.size().area();
	}

	Scalar getMSSIM(const Mat& i1, const Mat& i2, double sigma = 1.5)
	{
		int r = cvRound(sigma*3.0);
		Size kernel = Size(2 * r + 1, 2 * r + 1);

		const double C1 = 6.5025, C2 = 58.5225;
		/***************************** INITS **********************************/
		int d = CV_32F;

		Mat I1, I2;
		i1.convertTo(I1, d);           // cannot calculate on one byte large values
		i2.convertTo(I2, d);

		Mat I2_2 = I2.mul(I2);        // I2^2
		Mat I1_2 = I1.mul(I1);        // I1^2
		Mat I1_I2 = I1.mul(I2);        // I1 * I2

		/*************************** END INITS **********************************/

		Mat mu1, mu2;   // PRELIMINARY COMPUTING
		GaussianBlur(I1, mu1, kernel, sigma);
		GaussianBlur(I2, mu2, kernel, sigma);

		Mat mu1_2 = mu1.mul(mu1);
		Mat mu2_2 = mu2.mul(mu2);
		Mat mu1_mu2 = mu1.mul(mu2);

		Mat sigma1_2, sigma2_2, sigma12;

		GaussianBlur(I1_2, sigma1_2, kernel, sigma);
		sigma1_2 -= mu1_2;

		GaussianBlur(I2_2, sigma2_2, kernel, sigma);
		sigma2_2 -= mu2_2;

		GaussianBlur(I1_I2, sigma12, kernel, sigma);
		sigma12 -= mu1_mu2;

		///////////////////////////////// FORMULA ////////////////////////////////
		Mat t1, t2, t3;

		t1 = 2 * mu1_mu2 + C1;
		t2 = 2 * sigma12 + C2;
		t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		t1 = mu1_2 + mu2_2 + C1;
		t2 = sigma1_2 + sigma2_2 + C2;
		t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

		Mat ssim_map;
		divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

		Scalar mssim = mean(ssim_map); // mssim = average of ssim map
		imshow("ssim", ssim_map);
		return mssim;
	}

	double SSIM(Mat& src, Mat& ref, double sigma)
	{
		Mat gray1, gray2;
		cvtColor(src, gray1, CV_BGR2GRAY);
		cvtColor(ref, gray2, CV_BGR2GRAY);

		Scalar v = getMSSIM(gray1, gray2, sigma);
		return v.val[0];
	}

	inline int norm_l(int a, int b, int norm)
	{
		if (norm == 0)
		{
			int v = (a == b) ? 0 : 1;
			return v;
		}
		else if (norm == 1)
		{
			return abs(a - b);
		}
		else
		{
			return 0;
		}
	}

	double calcTV(Mat& src)
	{
		Mat gray;
		cvtColor(src, gray, CV_BGR2GRAY);
		Mat bb;
		copyMakeBorder(gray, bb, 0, 1, 0, 1, BORDER_REFLECT);

		int sum = 0;
		int count = 0;

		int NRM = 0;
		for (int j = 0; j < src.rows; j++)
		{
			uchar* pb = bb.ptr(j);
			uchar* b = bb.ptr(j + 1);
			for (int i = 0; i < src.rows; i++)
			{
				sum += norm_l(pb[i], b[i], NRM);
				sum += norm_l(b[i], b[i + 1], NRM);
				count++;
			}
		}
		return (double)sum / (double)count;
	}

	double calcEntropy16S(Mat& src, Mat& mask)
	{
		//mask function is not implemented.
		vector<Mat> im;
		cv::split(src, im);

		double ret = 0.0;
		for (int i = 0; i < src.channels(); i++)
		{
			vector<int>hist(512);
			for (int n = 0; n < 512; n++)hist[n] = 0;
			for (int n = 0; n < src.rows; n++)
			{
				short* v = im[i].ptr<short>(n);
				for (int m = 0; m < src.cols; m++)
				{
					hist[255 + v[m]]++;
				}
			}

			float sum = 0.0f;
			for (int j = 0; j < 512; ++j)
			{
				sum += hist[j];
			}

			double invsum = 1.0 / sum;
			for (int j = 0; j < 512; ++j)
			{
				const double v = (double)hist[j] * invsum;
				if (v != 0) ret -= v*log2(v);
			}
		}
		return ret;
	}

	double calcEntropy8U(Mat& src, Mat& mask)
	{
		vector<Mat> im;
		cv::split(src, im);
		
		double ret = 0.0;
		for (int i = 0; i < src.channels(); i++)
		{
			cv::Mat hist;
			const int hdims[] = { 256 }; 
			const float hranges[] = { 0, 256 };
			//const float hranges[] = {-minv,maxv};
			const float* ranges[] = { hranges }; 

			cv::calcHist(&im[i], 1, 0, mask, hist, 1, hdims, ranges);

			double sum = 0.0;
			for (int j = 0; j < hdims[0]; ++j)
			{
				sum += hist.at<float>(j);
			}

			double invsum = 1.0 / sum;
			
			for (int j = 0; j < hdims[0]; ++j)
			{
				const double v = (double)hist.at<float>(j)*invsum;
				if (v != 0) ret -= v*log2(v);	
			}
		}
		return ret;
	}

	double calcEntropy(cv::InputArray src, cv::InputArray mask_)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_16S);
		Mat mask = mask_.getMat();
		Mat src_ = src.getMat();
		double ret = 0.0;
		if (src.depth() == CV_8U)
		{
			ret = calcEntropy8U(src_, mask);
		}
		else if (src.depth() == CV_16S)
		{
			ret = calcEntropy16S(src_, mask);
		}
		else
		{
			;
		}
		return ret;
	}

}