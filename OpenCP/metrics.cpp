#include "metrics.hpp"
#include "inlineSIMDFunctions.hpp"
#include "debugcp.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	std::string getPSNR_PRECISION(const int precision)
	{
		string ret;
		switch (precision)
		{
		case PSNR_8U:
			ret = "PSNR_8U"; break;
		case PSNR_32F:
			ret = "PSNR_32F"; break;
		case PSNR_64F:
			ret = "PSNR_64F"; break;
		case PSNR_KAHAN_64F:
			ret = "PSNR_KAHAN_64F"; break;
			break;
		}
		return ret;
	}

	double MSE_64F(cv::Mat& src, cv::Mat& reference, bool isKahan = true)
	{
		CV_Assert(!src.empty());
		CV_Assert(!reference.empty());
		CV_Assert(src.size() == reference.size());
		CV_Assert(src.channels() == reference.channels());
		CV_Assert(src.depth() == CV_64F);
		CV_Assert(reference.depth() == CV_64F);

		int pixels = src.size().area()*src.channels();

		double MSE = 0.0;

		double* ptr1 = src.ptr<double>(0);
		double* ptr2 = reference.ptr<double>(0);
		const int simdsize = (pixels / 4) * 4;
		const int rem = pixels - simdsize;

		__m256d mmse = _mm256_setzero_pd();

		if (isKahan)
		{
			__m256d c = _mm256_setzero_pd();
			for (int i = 0; i < simdsize; i += 4)
			{
				__m256d m1 = _mm256_load_pd(ptr1 + i);
				__m256d m2 = _mm256_load_pd(ptr2 + i);
				__m256d v = _mm256_sub_pd(m1, m2);

				__m256d y = _mm256_fmsub_pd(v, v, c);
				__m256d t = _mm256_add_pd(mmse, y);
				c = _mm256_sub_pd(_mm256_sub_pd(t, mmse), y);
				mmse = t;
			}
		}
		else
		{
			for (int i = 0; i < simdsize; i += 4)
			{
				__m256d m1 = _mm256_load_pd(ptr1 + i);
				__m256d m2 = _mm256_load_pd(ptr2 + i);
				__m256d t = _mm256_sub_pd(m1, m2);
				mmse = _mm256_fmadd_pd(t, t, mmse);
			}
		}

		MSE += mmse.m256d_f64[0];
		MSE += mmse.m256d_f64[1];
		MSE += mmse.m256d_f64[2];
		MSE += mmse.m256d_f64[3];

		//for (int i = 0; i < pixels; ++i)
		for (int i = simdsize; i < pixels; ++i)
		{
			MSE += (ptr1[i] - ptr2[i])*(ptr1[i] - ptr2[i]);
		}
		MSE /= (double)pixels;

		return MSE;
	}

	double MSE_32F(cv::Mat& src, cv::Mat& reference)
	{
		CV_Assert(!src.empty());
		CV_Assert(!reference.empty());
		CV_Assert(src.size() == reference.size());
		CV_Assert(src.channels() == reference.channels());
		CV_Assert(src.depth() == CV_32F);
		CV_Assert(reference.depth() == CV_32F);

		int pixels = src.size().area()*src.channels();

		double MSE = 0.0;

		float* ptr1 = src.ptr<float>(0);
		float* ptr2 = reference.ptr<float>(0);
		const int simdsize = (pixels / 8) * 8;
		const int rem = pixels - simdsize;

		__m256 mmse = _mm256_setzero_ps();

		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 m1 = _mm256_load_ps(ptr1 + i);
			__m256 m2 = _mm256_load_ps(ptr2 + i);
			__m256 t = _mm256_sub_ps(m1, m2);
			mmse = _mm256_fmadd_ps(t, t, mmse);
		}

		for (int i = 0; i < 8; i++)
		{
			MSE += (double)mmse.m256_f32[i];
		}
		//for (int i = 0; i < pixels; ++i)
		for (int i = simdsize; i < pixels; ++i)
		{
			MSE += (ptr1[i] - ptr2[i])*(ptr1[i] - ptr2[i]);
		}
		MSE /= (double)pixels;

		return MSE;
	}

	double MSE_8U(cv::Mat& src, cv::Mat& reference)
	{
		CV_Assert(!src.empty());
		CV_Assert(!reference.empty());
		CV_Assert(src.size() == reference.size());
		CV_Assert(src.channels() == reference.channels());
		CV_Assert(src.depth() == CV_8U);
		CV_Assert(reference.depth() == CV_8U);

		int pixels = src.size().area()*src.channels();

		double MSE = 0.0;

		uchar* ptr1 = src.ptr<uchar>(0);
		uchar* ptr2 = reference.ptr<uchar>(0);
		const int simdsize = (pixels / 8) * 8;
		const int rem = pixels - simdsize;

		__m256 mmse = _mm256_setzero_ps();

		for (int i = 0; i < simdsize; i += 8)
		{
			__m256 m1 = _mm256_load_epu8cvtps((__m128i*)(ptr1 + i));
			__m256 m2 = _mm256_load_epu8cvtps((__m128i*)(ptr2 + i));
			__m256 t = _mm256_sub_ps(m1, m2);
			mmse = _mm256_fmadd_ps(t, t, mmse);
		}

		for (int i = 0; i < 8; i++)
		{
			MSE += (double)mmse.m256_f32[i];
		}
		//for (int i = 0; i < pixels; ++i)
		for (int i = simdsize; i < pixels; ++i)
		{
			MSE += (ptr1[i] - ptr2[i])*(ptr1[i] - ptr2[i]);
		}
		MSE /= (double)pixels;

		return MSE;
	}


	void PSNRMetrics::cvtImageForPSNR64F(const Mat& src, Mat& dest, const int cmethod)
	{
		if (src.depth() == CV_64F)
		{
			switch (cmethod)
			{
			case PSNR_ALL:
			default:
			{
				//src.copyTo(dest); break;
				dest = src; break;
			}
			case PSNR_Y:
			{
				//cvtColor do not support 64F 
				cv::Matx13d mt(0.114, 0.587, 0.299);
				cv::transform(src, dest, mt); 
				break;
			}
			case PSNR_B:
			{
				cv::split(src, vtemp);
				vtemp[0].copyTo(dest); break;
			}
			case PSNR_G:
			{
				cv::split(src, vtemp);
				vtemp[1].copyTo(dest); break;
			}
			case PSNR_R:
			{
				cv::split(src, vtemp);
				vtemp[2].copyTo(dest); break;
			}
			}
		}
		else
		{
			switch (cmethod)
			{
			case PSNR_ALL:
			default:
			{
				src.convertTo(dest, CV_64F); break;
			}
			case PSNR_Y:
			{
				cvtColor(src, temp, COLOR_BGR2GRAY);
				temp.convertTo(dest, CV_64F); break;
			}
			case PSNR_B:
			{
				split(src, vtemp);
				vtemp[0].convertTo(dest, CV_64F); break;
			}
			case PSNR_G:
			{
				cv::split(src, vtemp);
				vtemp[1].convertTo(dest, CV_64F); break;
			}
			case PSNR_R:
			{
				cv::split(src, vtemp);
				vtemp[2].convertTo(dest, CV_64F); break;
			}
			}
		}
	}

	void PSNRMetrics::cvtImageForPSNR32F(const Mat& src, Mat& dest, const int cmethod)
	{
		if (src.depth() == CV_32F)
		{
			switch (cmethod)
			{
			case PSNR_ALL:
			default:
			{
				//src.copyTo(dest); break;
				dest = src; break;
			}
			case PSNR_Y:
			{
				cvtColor(src, dest, COLOR_BGR2GRAY); break;
			}
			case PSNR_B:
			{
				cv::split(src, vtemp);
				vtemp[0].copyTo(dest); break;
			}
			case PSNR_G:
			{
				cv::split(src, vtemp);
				vtemp[1].copyTo(dest); break;
			}
			case PSNR_R:
			{
				cv::split(src, vtemp);
				vtemp[2].copyTo(dest); break;
			}
			}
		}
		else
		{
			switch (cmethod)
			{
			case PSNR_ALL:
			default:
			{
				src.convertTo(dest, CV_32F); break;
			}
			case PSNR_Y:
			{
				src.convertTo(temp, CV_32F);
				cvtColor(temp, dest, COLOR_BGR2GRAY);
				break;
			}
			case PSNR_B:
			{
				split(src, vtemp);
				vtemp[0].convertTo(dest, CV_32F); break;
			}
			case PSNR_G:
			{
				cv::split(src, vtemp);
				vtemp[1].convertTo(dest, CV_32F); break;
			}
			case PSNR_R:
			{
				cv::split(src, vtemp);
				vtemp[2].convertTo(dest, CV_32F); break;
			}
			}
		}
	}

	void PSNRMetrics::cvtImageForPSNR8U(const Mat& src, Mat& dest, const int cmethod)
	{
		if (src.depth() == CV_8U)
		{
			switch (cmethod)
			{
			case PSNR_ALL:
			default:
			{
				//src.copyTo(dest); break;
				dest = src; break;
			}
			case PSNR_Y:
			{
				cvtColor(src, dest, COLOR_BGR2GRAY); break;
			}
			case PSNR_B:
			{
				cv::split(src, vtemp);
				vtemp[0].copyTo(dest); break;
			}
			case PSNR_G:
			{
				cv::split(src, vtemp);
				vtemp[1].copyTo(dest); break;
			}
			case PSNR_R:
			{
				cv::split(src, vtemp);
				vtemp[2].copyTo(dest); break;
			}
			}
		}
		else
		{
			switch (cmethod)
			{
			case PSNR_ALL:
			default:
			{
				src.convertTo(dest, CV_8U); break;
			}
			case PSNR_Y:
			{
				src.convertTo(temp, CV_8U);
				cvtColor(temp, dest, COLOR_BGR2GRAY);
				break;
			}
			case PSNR_B:
			{
				split(src, vtemp);
				vtemp[0].convertTo(dest, CV_8U); break;
			}
			case PSNR_G:
			{
				cv::split(src, vtemp);
				vtemp[1].convertTo(dest, CV_8U); break;
			}
			case PSNR_R:
			{
				cv::split(src, vtemp);
				vtemp[2].convertTo(dest, CV_8U); break;
			}
			}
		}
	}

	double PSNRMetrics::getPSNR(cv::InputArray src, cv::InputArray ref, const int boundingBox, const int precision, const int compare_method)
	{
		CV_Assert(!ref.empty());
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == ref.channels());
		CV_Assert(src.size() == ref.size());

		setReference(ref, boundingBox, precision, compare_method);
		return getPSNRPreset(src, boundingBox, precision, compare_method);
	}

	double PSNRMetrics::operator()(cv::InputArray src, cv::InputArray ref, const int boundingBox, const int precision, const int compare_method)
	{
		return getPSNR(src, ref, boundingBox, precision, compare_method);
	}

	//set reference image for acceleration
	void PSNRMetrics::setReference(cv::InputArray src, const int boundingBox, const int precision, const int compare_method)
	{
		const int cmethod = (src.channels() == 1) ? PSNR_ALL : compare_method;

		Mat r = src.getMat();

		if (boundingBox != 0)
		{
			r(Rect(boundingBox, boundingBox, r.cols - 2 * boundingBox, r.rows - 2 * boundingBox)).copyTo(cropr);
		}
		else
		{
			cropr = r;
		}

		if (precision == PSNR_32F)
		{
			cvtImageForPSNR32F(cropr, reference, cmethod);
		}
		else if (precision == PSNR_8U)
		{
			cvtImageForPSNR8U(cropr, reference, cmethod);
		}
		else
		{
			cvtImageForPSNR64F(cropr, reference, cmethod);
		}
	}

	//using preset reference image for acceleration
	double PSNRMetrics::getPSNRPreset(cv::InputArray src, const int boundingBox, const int precision, const int compare_method)
	{
		CV_Assert(!reference.empty());
		CV_Assert(!src.empty());
		Mat s = src.getMat();

		const int cmethod = (src.channels() == 1) ? PSNR_ALL : compare_method;

		if (boundingBox != 0)
		{
			s(Rect(boundingBox, boundingBox, s.cols - 2 * boundingBox, s.rows - 2 * boundingBox)).copyTo(crops);
		}
		else
		{
			crops = s;
		}

		
		double MSE = 0.0;
		if (precision == PSNR_32F)
		{
			cvtImageForPSNR32F(crops, source, cmethod);

			if (source.depth() != reference.depth())
			{
				cout << "argments are boundingBox, precision, compare_method." << endl;
				cout << "do not forget boundingBox argment" << endl;
			}
			MSE = MSE_32F(source, reference);
		}
		else if (precision == PSNR_8U)
		{
			cvtImageForPSNR8U(crops, source, cmethod);

			if (source.depth() != reference.depth())
			{
				cout << "argments are boundingBox, precision, compare_method." << endl;
				cout << "do not forget boundingBox argment" << endl;
			}
			MSE = MSE_8U(source, reference);
		}
		else
		{
			cvtImageForPSNR64F(crops, source, cmethod);

			if (source.depth() != reference.depth())
			{
				cout << "argments are boundingBox, precision, compare_method." << endl;
				cout << "do not forget boundingBox argment" << endl;
			}

			MSE = MSE_64F(source, reference, precision == PSNR_KAHAN_64F);
		}

		if (MSE == 0.0)
		{
			return 0;
		}
		else if (cvIsNaN(MSE) || cvIsInf(MSE))
		{
			return -1.0;
		}
		else
		{
			return 10.0 * log10(255.0*255.0 / MSE);
		}
	}

	double getPSNR(cv::InputArray src, cv::InputArray ref, const int boundingBox , const int precision, const int compare_method)
	{
		PSNRMetrics psnr;
		return psnr(src, ref, boundingBox, precision, compare_method);
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

	double calcBadPixel(const Mat& src, const Mat& ref, int threshold)
	{
		Mat g1, g2;
		if (src.channels() == 3)
		{
			cvtColor(src, g1, COLOR_BGR2GRAY);
			cvtColor(ref, g2, COLOR_BGR2GRAY);
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
		cvtColor(src, gray1, COLOR_BGR2GRAY);
		cvtColor(ref, gray2, COLOR_BGR2GRAY);

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
		cvtColor(src, gray, COLOR_BGR2GRAY);
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
				if (v != 0) ret -= v * log2(v);
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
				if (v != 0) ret -= v * log2(v);
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