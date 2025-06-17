#include "metrics.hpp"
#include "inlineMathFunctions.hpp"
#include "inlineSIMDFunctions.hpp"
#include "onelineCVFunctions.hpp"
#include "arithmetic.hpp"
#include "updateCheck.hpp"
#include "color.hpp"
#include "debugcp.hpp"
#include "statistic.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	std::string getPSNR_CHANNEL(const int channel)
	{
		string ret;
		switch (channel)
		{
		case PSNR_ALL:
			ret = "PSNR_ALL"; break;
		case PSNR_Y:
			ret = "PSNR_Y"; break;
		case PSNR_B:
			ret = "PSNR_B"; break;
		case PSNR_G:
			ret = "PSNR_G"; break;
		case PSNR_R:
			ret = "PSNR_R"; break;
		case PSNR_Y_INTEGER:
			ret = "PSNR_Y_INTEGER"; break;
		default:
			ret = "NO SUPPORT"; break;
		}
		return ret;
	}

	std::string getPSNR_PRECISION(const int precision)
	{
		string ret;
		switch (precision)
		{
		case PSNR_UP_CAST:
			ret = "PSNR_UP_CAST"; break;
		case PSNR_8U:
			ret = "PSNR_8U"; break;
		case PSNR_32F:
			ret = "PSNR_32F"; break;
		case PSNR_64F:
			ret = "PSNR_64F"; break;
		case PSNR_KAHAN_64F:
			ret = "PSNR_KAHAN_64F"; break;
		default:
			ret = "NO SUPPORT"; break;
			break;
		}
		return ret;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//class PSNRMetrics
	///////////////////////////////////////////////////////////////////////////////////////////////////

	double PSNRMetrics::MSE_64F(cv::Mat& src, cv::Mat& reference, bool isKahan)
	{
		CV_Assert(!src.empty());
		CV_Assert(!reference.empty());
		CV_Assert(src.size() == reference.size());
		CV_Assert(src.channels() == reference.channels());
		CV_Assert(src.depth() == CV_64F);
		CV_Assert(reference.depth() == CV_64F);

		int pixels = src.size().area() * src.channels();

		double MSE = 0.0;

		double* ptr1 = src.ptr<double>();
		double* ptr2 = reference.ptr<double>();
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
		MSE += _mm256_reduceadd_pd(mmse);


		//for (int i = 0; i < pixels; ++i)
		for (int i = simdsize; i < pixels; ++i)
		{
			MSE += (ptr1[i] - ptr2[i]) * (ptr1[i] - ptr2[i]);
		}
		MSE /= (double)pixels;

		return MSE;
	}

	double PSNRMetrics::MSE_32F(cv::Mat& src, cv::Mat& reference)
	{
		CV_Assert(!src.empty());
		CV_Assert(!reference.empty());
		CV_Assert(src.size() == reference.size());
		CV_Assert(src.channels() == reference.channels());
		CV_Assert(src.depth() == CV_32F);
		CV_Assert(reference.depth() == CV_32F);

		int pixels = src.size().area() * src.channels();

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

		MSE = (double)_mm256_reduceadd_ps(mmse);

		//for (int i = 0; i < pixels; ++i)
		for (int i = simdsize; i < pixels; ++i)
		{
			MSE += (ptr1[i] - ptr2[i]) * (ptr1[i] - ptr2[i]);
		}
		MSE /= (double)pixels;

		return MSE;
	}

	double PSNRMetrics::MSE_8U(cv::Mat& src, cv::Mat& reference)
	{
		CV_Assert(!src.empty());
		CV_Assert(!reference.empty());
		CV_Assert(src.size() == reference.size());
		CV_Assert(src.channels() == reference.channels());
		CV_Assert(src.depth() == CV_8U);
		CV_Assert(reference.depth() == CV_8U);

		int pixels = src.size().area() * src.channels();

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
		MSE = (double)_mm256_reduceadd_ps(mmse);

		//for (int i = 0; i < pixels; ++i)
		for (int i = simdsize; i < pixels; ++i)
		{
			MSE += (ptr1[i] - ptr2[i]) * (ptr1[i] - ptr2[i]);
		}
		MSE /= (double)pixels;

		return MSE;
	}

	void PSNRMetrics::cvtImageForMSE64F(const Mat& src, Mat& dest, const int cmethod)
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
			case PSNR_Y_INTEGER:
			{
				cp::cvtColorIntegerY(src, dest); break;
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
			case PSNR_Y_INTEGER:
			{
				cp::cvtColorIntegerY(src, dest);
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

	void PSNRMetrics::cvtImageForMSE32F(const Mat& src, Mat& dest, const int cmethod)
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
			case PSNR_Y_INTEGER:
			{
				cp::cvtColorIntegerY(src, dest); break;
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
				cvtColor(temp, dest, COLOR_BGR2GRAY); break;
			}
			case PSNR_Y_INTEGER:
			{
				src.convertTo(temp, CV_32F);
				cp::cvtColorIntegerY(temp, dest); break;
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

	void PSNRMetrics::cvtImageForMSE8U(const Mat& src, Mat& dest, const int cmethod)
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
			case PSNR_Y_INTEGER:
			{
				cp::cvtColorIntegerY(src, dest); break;
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
			case PSNR_Y_INTEGER:
			{
				src.convertTo(temp, CV_8U);
				cp::cvtColorIntegerY(temp, dest); break;
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

	inline int PSNRMetrics::getPrecisionUpCast(cv::InputArray src, cv::InputArray ref)
	{
		int prec = PSNR_8U;
		const int depth = max(ref.depth(), src.depth());
		if (depth == CV_8U)
		{
			prec = PSNR_8U;
		}
		else if (depth == CV_32F)
		{
			prec = PSNR_32F;
		}
		else
		{
			prec = PSNR_64F;
		}

		return prec;
	}


	double PSNRMetrics::getMSE(cv::InputArray src, cv::InputArray ref, const int boundingBox, const int precision, const int compare_channel)
	{
		CV_Assert(!ref.empty());
		CV_Assert(!src.empty());
		CV_Assert(src.channels() == ref.channels());
		CV_Assert(src.size() == ref.size());

		int prec = precision;

		if (precision == PSNR_UP_CAST)
		{
			prec = getPrecisionUpCast(src, ref);
		}

		setReference(ref, boundingBox, prec, compare_channel);
		return getMSEPreset(src, boundingBox, prec, compare_channel);
	}

	double PSNRMetrics::getPSNR(cv::InputArray src, cv::InputArray ref, const int boundingBox, const int precision, const int compare_channel)
	{
		double mse = getMSE(src, ref, boundingBox, precision, compare_channel);

		return MSEtoPSNR(mse);
	}

	double PSNRMetrics::operator()(cv::InputArray src, cv::InputArray ref, const int boundingBox, const int precision, const int compare_channel)
	{
		return getPSNR(src, ref, boundingBox, precision, compare_channel);
	}

	//set reference image for acceleration
	void PSNRMetrics::setReference(cv::InputArray src, const int boundingBox, const int precision, const int compare_method)
	{
		CV_Assert(precision != PSNR_UP_CAST);

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
			cvtImageForMSE32F(cropr, reference, cmethod);
		}
		else if (precision == PSNR_8U)
		{
			cvtImageForMSE8U(cropr, reference, cmethod);
		}
		else
		{
			cvtImageForMSE64F(cropr, reference, cmethod);
		}
	}

	//using preset reference image for acceleration
	double PSNRMetrics::getMSEPreset(cv::InputArray src, const int boundingBox, const int precision, const int compare_method)
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
			cvtImageForMSE32F(crops, source, cmethod);

			if (source.depth() != reference.depth())
			{
				cout << "argments are boundingBox, precision, compare_method." << endl;
				cout << "do not forget boundingBox argment" << endl;
			}
			MSE = MSE_32F(source, reference);
		}
		else if (precision == PSNR_8U)
		{
			cvtImageForMSE8U(crops, source, cmethod);

			if (source.depth() != reference.depth())
			{
				cout << "argments are boundingBox, precision, compare_method." << endl;
				cout << "do not forget boundingBox argment" << endl;
			}
			MSE = MSE_8U(source, reference);
		}
		else
		{
			cvtImageForMSE64F(crops, source, cmethod);

			if (source.depth() != reference.depth())
			{
				cout << "argments are boundingBox, precision, compare_method." << endl;
				cout << "do not forget boundingBox argment" << endl;
			}

			MSE = MSE_64F(source, reference, precision == PSNR_KAHAN_64F);
		}

		return MSE;
	}

	double PSNRMetrics::getPSNRPreset(cv::InputArray src, const int boundingBox, const int precision, const int compare_channel)
	{
		double mse = getMSEPreset(src, boundingBox, precision, compare_channel);

		return MSEtoPSNR(mse);
	}
	//////////////////////////////////////////////////////////////////////////////

	double getPSNR(cv::InputArray src, cv::InputArray ref, const int boundingBox, const int precision, const int compare_channel)
	{
		PSNRMetrics psnr;
		return psnr(src, ref, boundingBox, precision, compare_channel);
	}

	double getPSNRClip(cv::InputArray src, cv::InputArray ref, const double minval, const double maxval, const int boundingBox, const int precision, const int compare_channel)
	{
		Mat s;
		Mat r;
		cp::clip(src, s, minval, maxval);
		cp::clip(ref, r, minval, maxval);

		PSNRMetrics psnr;
		return psnr(s, r, boundingBox, precision, compare_channel);
	}

	//internal
	void localPSNRMapColor_64F(vector<Mat>& s1, vector<Mat>& s2, Mat& dest, const int r, const double psnr_inf)
	{
		CV_Assert(s1[0].depth() == CV_64F);
		CV_Assert(s2[0].depth() == CV_64F);

		Size kernel = Size(2 * r + 1, 2 * r + 1);

		subtract(s1[0], s2[0], s1[0]);
		multiply(s1[0], s1[0], s1[0]);
		for (int c = 1; c < s1.size(); c++)
		{
			subtract(s1[c], s2[c], s1[c]);
			fmadd(s1[c], s1[c], s1[0], s1[0]);
		}
		divide(s1[0], 3.0, s1[0]);
		blur(s1[0], s1[0], kernel);

		for (int j = 0; j < s1[0].rows; j++)
		{
			double* sptr1 = s1[0].ptr<double>(j);
			double* dptr = dest.ptr<double>(j);
			for (int i = 0; i < s1[0].cols; i++)
			{
				double mse = sptr1[i];

				double psnr;
				if (mse == 0.0)
				{
					psnr = psnr_inf;
				}
				else if (cvIsNaN(mse))
				{
					psnr = -1.0;
				}
				else if (cvIsInf(mse))
				{
					psnr = -2.0;
				}
				else
				{
					psnr = 10.0 * log10(255.0 * 255.0 / mse);
				}

				dptr[i] = psnr;
			}
		}
	}

	//internal
	void localPSNRMapGray_64F(Mat& s1, Mat& s2, Mat& dest, const int r, const double psnr_inf)
	{
		CV_Assert(s1.depth() == CV_64F);
		CV_Assert(s2.depth() == CV_64F);

		Size kernel = Size(2 * r + 1, 2 * r + 1);
		subtract(s1, s2, s1);
		multiply(s1, s1, s1);
		blur(s1, s1, kernel, Point(-1, -1), BORDER_REFLECT101);

		for (int j = 0; j < s1.rows; j++)
		{
			double* sptr1 = s1.ptr<double>(j);
			double* dptr = dest.ptr<double>(j);
			for (int i = 0; i < s1.cols; i++)
			{
				double mse = sptr1[i];

				double psnr;
				if (mse == 0.0)
				{
					psnr = psnr_inf;
				}
				else if (cvIsNaN(mse))
				{
					psnr = -1.0;
				}
				else if (cvIsInf(mse))
				{
					psnr = -2.0;
				}
				else
				{
					psnr = 10.0 * log10(255.0 * 255.0 / mse);
				}

				dptr[i] = psnr;
			}
		}
	}

	void localPSNRMap(InputArray src1, InputArray src2, OutputArray dest, const int r, const int channel, const double psnr_infinity_value)
	{
		dest.create(src1.size(), CV_64F);
		Mat dst = dest.getMat();
		Mat s1, s2;
		src1.getMat().convertTo(s1, CV_64F);
		src2.getMat().convertTo(s2, CV_64F);

		if (src1.channels() == 1)
		{
			localPSNRMapGray_64F(s1, s2, dst, r, psnr_infinity_value);
		}
		else if (src1.channels() == 3)
		{
			Mat g1, g2;
			if (channel == PSNR_ALL)
			{
				vector<Mat> vs1;
				vector<Mat> vs2;
				split(s1, vs1);
				split(s2, vs2);
				localPSNRMapColor_64F(vs1, vs2, dst, r, psnr_infinity_value);
			}
			else
			{
				if (channel == PSNR_Y)
				{
					if (s1.depth() == CV_64F)
					{
						//cvtColor do not support 64F 
						cv::Matx13d mt(0.114, 0.587, 0.299);
						cv::transform(s1, g1, mt);
						cv::transform(s2, g2, mt);
					}
					else
					{
						cvtColor(s1, g1, COLOR_BGR2GRAY);
						cvtColor(s2, g2, COLOR_BGR2GRAY);
					}
				}
				else
				{
					vector<Mat> vs1;
					vector<Mat> vs2;
					split(s1, vs1);
					split(s2, vs2);
					if (channel == PSNR_B)
					{
						g1 = vs1[0];
						g2 = vs2[0];
					}
					else if (channel == PSNR_G)
					{
						g1 = vs1[1];
						g2 = vs2[1];
					}
					else if (channel == PSNR_R)
					{
						g1 = vs1[2];
						g2 = vs2[2];
					}
				}

				localPSNRMapGray_64F(g1, g2, dst, r, psnr_infinity_value);
			}
		}
	}

	void guiLocalPSNRMap(InputArray src1, InputArray src2, const bool isWait, string wname)
	{
		namedWindow(wname);
		static int is_normalize = 1; createTrackbar("l_psnr_is_norm", wname, &is_normalize, 1);
		static int local_psnr_inf = 255; createTrackbar("l_psnr_inf", wname, &local_psnr_inf, 255);
		static int local_psnr_r = 5; createTrackbar("l_psnr_r", wname, &local_psnr_r, 50);
		static int local_psnr_vamp = 10; createTrackbar("l_psnr_vamp*0.1", wname, &local_psnr_vamp, 30);
		static int local_psnr_chananel = 0; createTrackbar("l_psnr_channel", wname, &local_psnr_chananel, PSNR_CHANNEL_SIZE - 1);
		int key = 0;
		Mat show;
		Mat dest;

		cp::UpdateCheck uc(local_psnr_chananel);
		while (key != 'q')
		{
			if (uc.isUpdate(local_psnr_chananel))
			{
				if (isWait)
					displayOverlay(wname, getPSNR_CHANNEL(local_psnr_chananel) + format(": %f dB", getPSNR(src1, src2, 0, 0, local_psnr_chananel)), 3000);
			}

			localPSNRMap(src1, src2, dest, local_psnr_r, local_psnr_chananel, local_psnr_inf);

			if (local_psnr_vamp == 0 || is_normalize == 1)
			{
				normalize(dest, show, 255, 0, NORM_MINMAX, CV_8U);
			}
			else
			{
				dest.convertTo(show, CV_8U, local_psnr_vamp * 0.1);
			}

			imshow(wname, show);

			if (isWait) key = waitKey(1);
			else key = 'q';

			if (key == 'p')
			{
				displayOverlay(wname, format("%f dB", getPSNR(src1, src2, 0, 0, local_psnr_chananel)), 3000);
			}
			if (key == 'f')
			{
				is_normalize = (is_normalize == 1) ? 0 : 1;
				setTrackbarPos("l_psnr_is_norm", wname, is_normalize);
			}
			if (key == '?')
			{
				cout << "f: flip normalize flag" << endl;
				cout << "p: show normal PSNR" << endl;
				cout << "q: quit" << endl;
			}
		}
		if (isWait) destroyWindow(wname);
	}


	double getMSE(InputArray I1_, InputArray I2_, InputArray mask)
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

	double getMSE(InputArray src, InputArray ref, const int boundingBox, const int precision, const int compare_channel)
	{
		PSNRMetrics mse;
		return mse.getMSE(src, ref, boundingBox, precision, compare_channel);
	}

	double getInacceptableRatio(InputArray src, InputArray ref, const int threshold)
	{
		Mat g1, g2;
		if (src.channels() == 3)
		{
			cvtColor(src, g1, COLOR_BGR2GRAY);
			cvtColor(ref, g2, COLOR_BGR2GRAY);
		}
		else
		{
			g1 = src.getMat();
			g2 = ref.getMat();
		}
		Mat temp;
		absdiff(g1, g2, temp);
		Mat mask;
		compare(temp, threshold, mask, CMP_GE);
		return 100.0 * countNonZero(mask) / src.size().area();
	}

	Scalar getMSSIM(const Mat& i1, const Mat& i2, double sigma = 1.5)
	{
		int r = cvRound(sigma * 3.0);
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

	/*double SSIM(Mat& src, Mat& ref, double sigma)
	{
		Mat gray1, gray2;
		cvtColor(src, gray1, COLOR_BGR2GRAY);
		cvtColor(ref, gray2, COLOR_BGR2GRAY);

		Scalar v = getMSSIM(gray1, gray2, sigma);
		return v.val[0];
	}*/

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

	double getTotalVariation(cv::InputArray src)
	{
		Mat gray;
		cvtColor(src, gray, COLOR_BGR2GRAY);
		Mat bb;
		copyMakeBorder(gray, bb, 0, 1, 0, 1, BORDER_REFLECT);

		int sum = 0;
		int count = 0;

		int NRM = 0;
		for (int j = 0; j < gray.rows; j++)
		{
			uchar* pb = bb.ptr(j);
			uchar* b = bb.ptr(j + 1);
			for (int i = 0; i < gray.cols; i++)
			{
				sum += norm_l(pb[i], b[i], NRM);
				sum += norm_l(b[i], b[i + 1], NRM);
				count++;
			}
		}
		return (double)sum / (double)count;
	}

	static double calcEntropy16S(Mat& src, Mat& mask)
	{
		double ret = 0.0;

		double minv, maxv;
		cv::minMaxLoc(src, &minv, &maxv);
		int minvi = (int)minv;
		const int range = (int)ceil(maxv - minv) + 1;
		AutoBuffer<int> hist(range);
		for (int i = 0; i < range; i++) hist[i] = 0;

		const short* s = src.ptr<short>();
		for (int i = 0; i < src.size().area(); i++)
		{
			hist[s[i] - minvi]++;
		}
		const double invsum = 1.0 / src.size().area();

		for (int i = 0; i < range; i++)
		{
			const double v = (double)hist[i] * invsum;
			if (v != 0) ret -= v * log2(v);
		}
		return ret;
	}

	static double calcEntropy16U(Mat& src, Mat& mask)
	{
		double ret = 0.0;
		cv::Mat hist;
		const int hdims[] = { USHRT_MAX + 1 };
		const float hranges[] = { 0, USHRT_MAX };
		const float* ranges[] = { hranges };

		cv::calcHist(&src, 1, 0, mask, hist, 1, hdims, ranges);

		double sum = 0.0;
		for (int j = 0; j < hdims[0]; ++j)
		{
			sum += hist.at<float>(j);
		}

		double invsum = 1.0 / sum;

		for (int j = 0; j < hdims[0]; ++j)
		{
			const double v = (double)hist.at<float>(j) * invsum;
			if (v != 0) ret -= v * log2(v);
		}
		return ret;
	}

	static double calcEntropy8U(Mat& src, Mat& mask)
	{
		double ret = 0.0;
		cv::Mat hist;
		const int hdims[] = { 256 };
		const float hranges[] = { 0, 256 };
		//const float hranges[] = {-minv,maxv};
		const float* ranges[] = { hranges };

		cv::calcHist(&src, 1, 0, mask, hist, 1, hdims, ranges);

		double sum = 0.0;
		for (int j = 0; j < hdims[0]; ++j)
		{
			sum += hist.at<float>(j);
		}

		double invsum = 1.0 / sum;

		for (int j = 0; j < hdims[0]; ++j)
		{
			const double v = (double)hist.at<float>(j) * invsum;
			if (v != 0) ret -= v * log2(v);
		}
		return ret;
	}

	double getEntropyWeight(cv::InputArray src, const vector<double>& weight, cv::InputArray mask_)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U);

		Mat mask = mask_.getMat();
		Mat src_ = src.getMat();
		int channels = src_.channels();
		vector<Mat> im(channels);
		if (channels != 1)cv::split(src, im);
		else im[0] = src.getMat();

		double ret = 0.0;
		for (int c = 0; c < channels; c++)
		{
			if (src.depth() == CV_8U)
			{
				ret += weight[c] * calcEntropy8U(im[c], mask);
			}
			else if (src.depth() == CV_16S)
			{
				ret += weight[c] * calcEntropy16S(im[c], mask);
			}
			else if (src.depth() == CV_16U)
			{
				ret += weight[c] * calcEntropy16U(im[c], mask);
			}
			else
			{
				;
			}
		}
		return ret;
	}

	double getEntropy(cv::InputArray src, cv::InputArray mask_)
	{
		CV_Assert(src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U);
		int channels = src.channels();
		vector<double> weight;
		for (int i = 0; i < channels; i++) weight.push_back(1.0);
		return getEntropyWeight(src, weight, mask_);
	}



#pragma region GMSD
	static void gradientSquareEdge32F(InputArray ref, OutputArray dst)
	{
		dst.create(ref.size(), ref.type());

		Mat border;
		copyMakeBorder(ref, border, 1, 1, 1, 1, BORDER_DEFAULT);
		Mat s = ref.getMat();
		Mat d = dst.getMat();
		const int step = s.size().width;
		const int simdwidth = get_simd_floor(step, 8);
		__m256 norm = _mm256_set1_ps(1.f / 9.f);
		for (int j = 0; j < ref.size().height; j++)
		{
			float* s = border.ptr<float>(j);
			float* dst = d.ptr<float>(j);
			for (int i = 0; i < simdwidth; i += 8)
			{
				__m256 mx = _mm256_sub_ps(_mm256_loadu_ps(s + i + 0), _mm256_loadu_ps(s + i + 2 + 2 * step));
				__m256 my = _mm256_sub_ps(_mm256_loadu_ps(s + i + 2), _mm256_loadu_ps(s + i + 0 + 2 * step));
				//__m256 mx = _mm256_sub_ps(_mm256_loadu_ps(s + i + step), _mm256_loadu_ps(s + i + 2 + step));
				//__m256 my = _mm256_sub_ps(_mm256_loadu_ps(s + i + 1), _mm256_loadu_ps(s + i + 1 +2*step));
				_mm256_storeu_ps(dst + i, _mm256_mul_ps(norm, _mm256_fmadd_ps(mx, mx, _mm256_mul_ps(my, my))));
			}
			for (int i = simdwidth; i < step; i++)
			{
				float x = (s[i + step] - s[i + 2 + step]);
				float y = (s[i + 1] - s[i + 1 + 2 * step]);
				dst[i] = x * x + y * y;
			}
		}
	}

	//dx*dx+dy*dy
	//[+1 0 -1]*1/3
	void gradientSquarePrewitt32F(InputArray ref, OutputArray dst)
	{
		dst.create(ref.size(), ref.type());

		Mat border;
		copyMakeBorder(ref, border, 1, 1, 1, 1, BORDER_DEFAULT);
		Mat s = ref.getMat();
		Mat d = dst.getMat();
		const int step = s.size().width;
		const int simdwidth = get_simd_floor(step, 8);
		const float normal = 1.f / 9.f;
		__m256 mnormal = _mm256_set1_ps(normal);
		for (int j = 0; j < ref.size().height; j++)
		{
			float* s = border.ptr<float>(j);
			float* dst = d.ptr<float>(j);
			for (int i = 0; i < simdwidth; i += 8)
			{
				__m256 mx = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(s + i + step)), _mm256_loadu_ps(s + i + 2 * step));
				mx = _mm256_sub_ps(mx, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i + 2), _mm256_loadu_ps(s + i + 2 + step)), _mm256_loadu_ps(s + i + 2 + 2 * step)));
				__m256 my = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(s + i + 1)), _mm256_loadu_ps(s + i + 2));
				my = _mm256_sub_ps(my, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i + 2 * step), _mm256_loadu_ps(s + i + 1 + 2 * step)), _mm256_loadu_ps(s + i + 2 + 2 * step)));
				_mm256_storeu_ps(dst + i, _mm256_mul_ps(mnormal, _mm256_fmadd_ps(mx, mx, _mm256_mul_ps(my, my))));
			}
			for (int i = simdwidth; i < step; i++)
			{
				float x = (s[i] + s[i + step] + s[i + 2 * step] - s[i + 2] - s[i + 2 + step] - s[i + 2 + 2 * step]);
				float y = (s[i] + s[i + 1] + s[i + 2] - s[i + 2 * step] - s[i + 1 + 2 * step] - s[i + 2 + 2 * step]);
				dst[i] = (x * x + y * y) * normal;
			}
		}
	}

	void gradientSquareRootPrewitt32F(InputArray ref, OutputArray dst)
	{
		dst.create(ref.size(), ref.type());

		Mat border;
		copyMakeBorder(ref, border, 1, 1, 1, 1, BORDER_DEFAULT);
		Mat s = ref.getMat();
		Mat d = dst.getMat();
		const int step = s.size().width;
		const int simdwidth = get_simd_floor(step, 8);
		const float normal = 1.f / 9.f;
		__m256 mnormal = _mm256_set1_ps(normal);
		for (int j = 0; j < ref.size().height; j++)
		{
			float* s = border.ptr<float>(j);
			float* dst = d.ptr<float>(j);
			for (int i = 0; i < simdwidth; i += 8)
			{
				__m256 mx = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(s + i + step)), _mm256_loadu_ps(s + i + 2 * step));
				mx = _mm256_sub_ps(mx, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i + 2), _mm256_loadu_ps(s + i + 2 + step)), _mm256_loadu_ps(s + i + 2 + 2 * step)));
				__m256 my = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(s + i + 1)), _mm256_loadu_ps(s + i + 2));
				my = _mm256_sub_ps(my, _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(s + i + 2 * step), _mm256_loadu_ps(s + i + 1 + 2 * step)), _mm256_loadu_ps(s + i + 2 + 2 * step)));
				_mm256_storeu_ps(dst + i, _mm256_sqrt_ps(_mm256_mul_ps(mnormal, _mm256_fmadd_ps(mx, mx, _mm256_mul_ps(my, my)))));
			}
			for (int i = simdwidth; i < step; i++)
			{
				float x = (s[i] + s[i + step] + s[i + 2 * step] - s[i + 2] - s[i + 2 + step] - s[i + 2 + 2 * step]);
				float y = (s[i] + s[i + 1] + s[i + 2] - s[i + 2 * step] - s[i + 1 + 2 * step] - s[i + 2 + 2 * step]);
				dst[i] = sqrt((x * x + y * y) * normal);
			}
		}
	}

	template<typename T>
	static void getGradient(InputArray ref, OutputArray dst)
	{
		dst.create(ref.size(), ref.type());

		Mat border;
		copyMakeBorder(ref, border, 1, 1, 1, 1, BORDER_DEFAULT);
		Mat s = ref.getMat();
		Mat d = dst.getMat();
		const int step = s.size().width;

		for (int j = 0; j < ref.size().height; j++)
		{
			T* s = border.ptr<T>(j);
			for (int i = 0; i < ref.size().width; i++)
			{
				double x = (s[i] + s[i + step] + s[i + 2 * step] - s[i + 2] - s[i + 2 + step] - s[i + 2 + 2 * step]) / 3.0;
				double y = (s[i] + s[i + 1] + s[i + 2] - s[i + 2 * step] - s[i + 1 + 2 * step] - s[i + 2 + 2 * step]) / 3.0;
				T v = T(x * x + y * y);
				d.at<T>(j, i) = v;
			}
		}
	}

	double stdpool(Mat& src)
	{
		const int N = src.size().area();
		float* s = src.ptr<float>();
		double x = 0.0;
		double xx = 0.0;
		for (int i = 0; i < N; i++)
		{
			x += s[i];
			xx += s[i] * s[i];
		}
		return xx / N - (x / N) * (x / N);
	}

	double meanMinkowskiDistance(Mat& src)
	{
		return 0;
	}

	Mat GMSDMap(InputArray ref, InputArray src, const double c, const bool isDownsample)
	{
		const double alpha = 0.5;
		const bool isUseAdditonalMasking = true;
		//const bool isUseAdditonalMasking = false;

		Mat gradientRef;
		Mat gradientSrc;
		Mat ref32 = cp::convert(ref, CV_32F);
		Mat src32 = cp::convert(src, CV_32F);
		if (isDownsample)
		{
			resize(ref32, ref32, Size(), 0.5, 0.5, INTER_AREA);
			resize(src32, src32, Size(), 0.5, 0.5, INTER_AREA);
		}
		//getGradient<float>(ref32, gradientRef);
		//getGradient<float>(src32, gradientSrc);
		//gradientSquareEdge32F(ref32, gradientRef);
		//gradientSquareEdge32F(src32, gradientSrc);
		gradientSquarePrewitt32F(ref32, gradientRef);
		gradientSquarePrewitt32F(src32, gradientSrc);
		/*bilateralFilter(ref32, gradientRef, 5, 5, 2);
		bilateralFilter(ref32, gradientSrc, 5, 5, 2);
		subtract(gradientRef, ref32, gradientRef);
		multiply(gradientRef, gradientRef, gradientRef);
		subtract(gradientSrc, src32, gradientSrc);
		multiply(gradientSrc, gradientSrc, gradientSrc);*/


		const int size = src32.size().area();
		float* r = gradientRef.ptr<float>();
		float* s = gradientSrc.ptr<float>();
		Mat qm(src32.size(), CV_32F);
		float* dst = qm.ptr<float>();

		if (isUseAdditonalMasking)
		{
			const int simdsize = get_simd_floor(size, 8);

			const __m256 m2a = _mm256_set1_ps(2.f - alpha);
			const __m256 ma = _mm256_set1_ps(-alpha);
			const __m256 mc = _mm256_set1_ps(max(c, (double)FLT_MIN));
			for (int i = 0; i < simdsize; i += 8)
			{
				const __m256 mr = _mm256_load_ps(r + i);
				const __m256 ms = _mm256_load_ps(s + i);
				const __m256 mv = _mm256_sqrt_ps(_mm256_mul_ps(mr, ms));
				const __m256 mn = _mm256_fmadd_ps(m2a, mv, mc);
				const __m256 md = _mm256_add_ps(_mm256_add_ps(mr, ms), _mm256_fmadd_ps(ma, mv, mc));
				_mm256_store_ps(dst + i, _mm256_div_ps(mn, md));
			}
			for (int i = simdsize; i < size; i++)
			{
				const float v = sqrt(r[i] * s[i]);
				float n = (2.f - alpha) * v + c;
				float d = r[i] + s[i] - alpha * v + c;
				dst[i] = n / d;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				float n = 2.f * sqrt(r[i] * s[i]) + c;
				float d = r[i] + s[i] + c;
				dst[i] = n / d;
			}
		}
		return qm;
	}

	double getGMSD(InputArray ref, InputArray src, const double c, const bool isDownsample)
	{
		CV_Assert(ref.channels() == src.channels());
		const double alpha = 0.5;
		const bool isUseAdditonalMasking = true;
		//const bool isUseAdditonalMasking = false;

		Mat gradientRef;
		Mat gradientSrc;
		Mat ref32;
		Mat src32;
		if (ref.channels() == 1)
		{
			ref32 = cp::convert(ref, CV_32F);
			src32 = cp::convert(src, CV_32F);
		}
		else
		{
			Mat tmp;
			cvtColor(ref, tmp, COLOR_BGR2GRAY);
			ref32 = cp::convert(tmp, CV_32F).clone();
			cvtColor(src, tmp, COLOR_BGR2GRAY);
			src32 = cp::convert(tmp, CV_32F).clone();
		}

		if (isDownsample)
		{
			resize(ref32, ref32, Size(), 0.5, 0.5, INTER_AREA);
			resize(src32, src32, Size(), 0.5, 0.5, INTER_AREA);
		}
		//getGradient<float>(ref32, gradientRef);
		//getGradient<float>(src32, gradientSrc);
		//gradientSquareEdge32F(ref32, gradientRef);
		//gradientSquareEdge32F(src32, gradientSrc);
		gradientSquarePrewitt32F(ref32, gradientRef);
		gradientSquarePrewitt32F(src32, gradientSrc);
		/*bilateralFilter(ref32, gradientRef, 5, 5, 2);
		bilateralFilter(ref32, gradientSrc, 5, 5, 2);
		subtract(gradientRef, ref32, gradientRef);
		multiply(gradientRef, gradientRef, gradientRef);
		subtract(gradientSrc, src32, gradientSrc);
		multiply(gradientSrc, gradientSrc, gradientSrc);*/


		const int size = src32.size().area();
		float* r = gradientRef.ptr<float>();
		float* s = gradientSrc.ptr<float>();
		Mat qm(src32.size(), CV_32F);
		float* dst = qm.ptr<float>();

		if (isUseAdditonalMasking)
		{
			const int simdsize = get_simd_floor(size, 8);

			const __m256 m2a = _mm256_set1_ps(2.f - alpha);
			const __m256 ma = _mm256_set1_ps(-alpha);
			const __m256 mc = _mm256_set1_ps(max(c, (double)FLT_MIN));
			for (int i = 0; i < simdsize; i += 8)
			{
				const __m256 mr = _mm256_load_ps(r + i);
				const __m256 ms = _mm256_load_ps(s + i);
				const __m256 mv = _mm256_sqrt_ps(_mm256_mul_ps(mr, ms));
				const __m256 mn = _mm256_fmadd_ps(m2a, mv, mc);
				const __m256 md = _mm256_add_ps(_mm256_add_ps(mr, ms), _mm256_fmadd_ps(ma, mv, mc));
				_mm256_store_ps(dst + i, _mm256_div_ps(mn, md));
			}
			for (int i = simdsize; i < size; i++)
			{
				const float v = sqrt(r[i] * s[i]);
				float n = (2.f - alpha) * v + c;
				float d = r[i] + s[i] - alpha * v + c;
				dst[i] = n / d;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				float n = 2.f * sqrt(r[i] * s[i]) + c;
				float d = r[i] + s[i] + c;
				dst[i] = n / d;
			}
		}
		cv::Scalar result;
		cv::meanStdDev(qm, cv::noArray(), result);
		return result[0];
	}

	class GMSDclass
	{
		double getIndex(InputArray ref, InputArray src);
	};

	cv::Scalar GMSD2(InputArray ref, InputArray src, const double c)
	{
		const double alpha = 0.5;
		const bool isUseAdditonalMasking = true;
		//const bool isUseAdditonalMasking = false;

		Mat gradientRef;
		Mat gradientSrc;
		Mat ref32 = cp::convert(ref, CV_32F);
		Mat src32 = cp::convert(src, CV_32F);
		//resize(ref32, ref32, Size(), 0.5, 0.5, INTER_AREA);
		//resize(src32, src32, Size(), 0.5, 0.5, INTER_AREA);
		//getGradient<float>(ref32, gradientRef);
		//getGradient<float>(src32, gradientSrc);
		gradientSquareEdge32F(ref32, gradientRef);
		gradientSquareEdge32F(src32, gradientSrc);
		//gradientSquarePrewitt32F(ref32, gradientRef);
		//gradientSquarePrewitt32F(src32, gradientSrc);

		const int size = src32.size().area();
		float* r = gradientRef.ptr<float>();
		float* s = gradientSrc.ptr<float>();
		Mat qm(src32.size(), CV_32F);
		float* dst = qm.ptr<float>();

		if (isUseAdditonalMasking)
		{
			const int simdsize = get_simd_floor(size, 8);

			const __m256 m2a = _mm256_set1_ps(2.f - alpha);
			const __m256 ma = _mm256_set1_ps(-alpha);
			const __m256 mc = _mm256_set1_ps(max(c, (double)FLT_MIN));
			for (int i = 0; i < simdsize; i += 8)
			{
				const __m256 mr = _mm256_load_ps(r + i);
				const __m256 ms = _mm256_load_ps(s + i);
				const __m256 mv = _mm256_sqrt_ps(_mm256_mul_ps(mr, ms));
				const __m256 mn = _mm256_fmadd_ps(m2a, mv, mc);
				const __m256 md = _mm256_add_ps(_mm256_add_ps(mr, ms), _mm256_fmadd_ps(ma, mv, mc));
				_mm256_store_ps(dst + i, _mm256_div_ps(mn, md));
			}
			for (int i = simdsize; i < size; i++)
			{
				const float v = sqrt(r[i] * s[i]);
				float n = (2.f - alpha) * v + c;
				float d = r[i] + s[i] - alpha * v + c;
				dst[i] = n / d;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				float n = 2.f * sqrt(r[i] * s[i]) + c;
				float d = r[i] + s[i] + c;
				dst[i] = n / d;
			}
		}
		cv::Scalar result;
		cv::meanStdDev(qm, cv::noArray(), result);
		return result;
	}

	cv::Scalar MSGMSD(InputArray ref, InputArray src, const double c)
	{
		vector<double> w = { 0.096, 0.596, 0.289, 0.019 };
		//vector<double> w = { 0.0, 0.8, 0.0, 1.0 };
		double wsum = 0.0;
		for (int i = 0; i < w.size(); i++) wsum += w[i];
		for (int i = 0; i < w.size(); i++)  w[i] /= wsum;

		const int M = w.size();
		double index = getGMSD(ref, src, c, false);
		double ret = w[0] * index * index;
		//double ret = w[0] * index;
		Mat r = ref.getMat();
		Mat s = src.getMat();
		for (int i = 1; i < M; i++)
		{
			pyrDown(r, r);
			pyrDown(s, s);
			index = getGMSD(r, s, c, false);
			ret += w[i] * index * index;
			//ret *= w[i] * index;
		}
		Scalar v;
		v.val[0] = sqrt(ret);
		//v.val[0] = ret;
		return v;
	}

	//mean deviation similarity index(MDSI)
	void cvtColorOpponentSplit32F(Mat& src, vector<Mat>& vdst)
	{
		// color space
		cv::Mat mat(3, 3, CV_32F);
		mat.at<float>(0, 2) = +0.2989f;
		mat.at<float>(0, 1) = +0.5870f;
		mat.at<float>(0, 0) = +0.1140f;
		mat.at<float>(1, 2) = +0.30f;
		mat.at<float>(1, 1) = +0.04f;
		mat.at<float>(1, 0) = -0.35f;
		mat.at<float>(2, 2) = +0.34f;
		mat.at<float>(2, 1) = -0.60f;
		mat.at<float>(2, 0) = +0.17f;
		Mat sf;
		if (src.depth() == CV_32F)
		{
			transform(src, sf, mat);
		}
		else
		{
			src.convertTo(sf, CV_32F);
			transform(sf, sf, mat);
		}
		split(sf, vdst);
	}

	double getMADPoolongQuarter(Mat& map)
	{
		Mat tmp(map.size(), CV_32F);
		double ret = 0.0;
		const float* s = map.ptr<float>();
		float* t = tmp.ptr<float>();
		for (int i = 0; i < map.size().area(); i++)
		{
			float v = sqrt(max(0.f, s[i]));
			v = sqrt(v);
			t[i] = v;
			ret += v;
		}
		const float m = ret / map.size().area();

		ret = 0.0;
		for (int i = 0; i < map.size().area(); i++)
		{
			ret += abs(t[i] - m);
		}

		ret = ret / map.size().area();
		ret = sqrt(max(0.0, ret));
		ret = sqrt(ret);
		return ret;
	}

	double getMDSI(InputArray ref, InputArray deg, const bool isDownsample)
	{
		//almost c_3 = 4c_1 = 10c_2
		const float C1 = 140.f;
		const float C2 = 55.f;
		const float C3 = 550.f;
		const float alpha = 0.6f;
		Mat R, D;
		if (isDownsample)
		{
			const int length = max(ref.size().width, ref.size().height);
			if (length > 256)
			{
				/*
				if (ref.size().width > ref.size().height)
				{
					const int h = ref.size().height * 256.0 / double(ref.size().width);
					resize(ref, R, Size(256, h), 0.0, 0.0, INTER_AREA);
					resize(deg, D, Size(256, h), 0.0, 0.0, INTER_AREA);
				}
				else
				{
					const int w = ref.size().width * 256.0 / double(ref.size().height);
					resize(ref, R, Size(w, 256), 0.0, 0.0, INTER_AREA);
					resize(deg, D, Size(w, 256), 0.0, 0.0, INTER_AREA);
				}*/
				resize(ref, R, Size(), 0.5, 0.5, INTER_AREA);
				resize(deg, D, Size(), 0.5, 0.5, INTER_AREA);
			}
			else
			{
				R = ref.getMat();
				D = deg.getMat();
			}
		}
		else
		{
			R = ref.getMat();
			D = deg.getMat();
		}
		vector<Mat> vLHM_R, vLHM_D;
		cvtColorOpponentSplit32F(R, vLHM_R);
		cvtColorOpponentSplit32F(D, vLHM_D);

		Mat gradientRef;
		Mat gradientDeg;
		Mat refL = vLHM_R[0];
		Mat degL = vLHM_D[0];
		Mat aveL, gradientAve;
		addWeighted(refL, 0.5f, degL, 0.5f, 0.f, aveL);

		gradientSquarePrewitt32F(refL, gradientRef);
		gradientSquarePrewitt32F(degL, gradientDeg);
		gradientSquarePrewitt32F(aveL, gradientAve);
		//getGradient<float>(ref32, gradientRef);
		//getGradient<float>(deg32, gradientDeg);
		//getGradient<float>(ave32, gradientAve);

		const int size = degL.size().area();
		const float* rl = gradientRef.ptr<float>();
		const float* dl = gradientDeg.ptr<float>();
		const float* al = gradientAve.ptr<float>();
		const float* rh = vLHM_R[1].ptr<float>();
		const float* dh = vLHM_D[1].ptr<float>();
		const float* rm = vLHM_R[2].ptr<float>();
		const float* dm = vLHM_D[2].ptr<float>();
		Mat gcs(degL.size(), CV_32F);
		float* dst = gcs.ptr<float>();
		for (int i = 0; i < size; i++)
		{
			//L
			float numnumerator = 2.f * sqrt(rl[i] * dl[i]) + C1;
			float denominator = rl[i] + dl[i] + C1;
			float ret = numnumerator / denominator;
			numnumerator = 2.f * sqrt(dl[i] * al[i]) + C2;
			denominator = al[i] + dl[i] + C2;
			ret += numnumerator / max(denominator, FLT_EPSILON);
			numnumerator = 2.f * sqrt(rl[i] * al[i]) + C2;
			denominator = al[i] + rl[i] + C2;
			ret -= numnumerator / max(denominator, FLT_EPSILON);

			//HM
			ret *= alpha;
			numnumerator = 2.f * (rh[i] * dh[i] + rm[i] * dm[i]) + C3;
			denominator = rh[i] * rh[i] + dh[i] * dh[i] + rm[i] * rm[i] + dm[i] * dm[i] + C3;
			ret += (1.f - alpha) * numnumerator / max(denominator, FLT_EPSILON);
			dst[i] = ret;
		}

		double v = getMADPoolongQuarter(gcs);
		return v;
		/*cv::Scalar result;
		cv::meanStdDev(gcs, cv::noArray(), result);
		return result;*/
	}
#pragma endregion

#pragma region SSIM
	double getSSIM(const cv::Mat& src1, const cv::Mat& src2, const double sigma, const bool isDownsample)
	{
		constexpr int depth = CV_32F;
		cv::Mat I1, I2;
		src1.convertTo(I1, depth);
		src2.convertTo(I2, depth);
		if (src1.channels() == 3) cvtColor(I1, I1, COLOR_BGR2GRAY);
		if (src2.channels() == 3) cvtColor(I2, I2, COLOR_BGR2GRAY);

		if (isDownsample)
		{
			resize(I1, I1, Size(), 0.5, 0.5, INTER_AREA);
			resize(I2, I2, Size(), 0.5, 0.5, INTER_AREA);
		}
		const int D = (int)ceil(sigma * 3.0) * 2 + 1;
		const Size kernelSize = Size(D, D);
		constexpr float C1 = 6.5025f, C2 = 58.5225f;
		/***************************** INITS **********************************/


		cv::Mat I2_2 = I2.mul(I2);//1
		cv::Mat I1_2 = I1.mul(I1);//2
		cv::Mat I1_I2 = I1.mul(I2);//3
		/*************************** END INITS **********************************/
		cv::Mat mu1, mu2, sigma1_2, sigma2_2, sigma12;

		cv::GaussianBlur(I1, mu1, kernelSize, sigma);//4,5
		cv::GaussianBlur(I2, mu2, kernelSize, sigma);//6,7
		cv::GaussianBlur(I1_2, sigma1_2, kernelSize, sigma);//8,9
		cv::GaussianBlur(I2_2, sigma2_2, kernelSize, sigma);//10,11
		cv::GaussianBlur(I1_I2, sigma12, kernelSize, sigma);//12,13
		cv::Mat mu1_2 = mu1.mul(mu1);//14
		cv::Mat mu2_2 = mu2.mul(mu2);//15
		cv::Mat mu1_mu2 = mu1.mul(mu2);//16
		sigma1_2 -= mu1_2;//17
		sigma2_2 -= mu2_2;//18
		sigma12 -= mu1_mu2;//19
		///////////////////////////////// FORMULA ////////////////////////////////
		cv::Mat t1, t2, t3;
		t1 = 2 * mu1_mu2 + C1;//20
		t2 = 2 * sigma12 + C2;//21
		t3 = t1.mul(t2);//22
		t1 = mu1_2 + mu2_2 + C1;//23
		t2 = sigma1_2 + sigma2_2 + C2;//24
		t1 = t1.mul(t2);//25  
		cv::Mat ssim_map;//26
		divide(t3, t1, ssim_map);//27

		const int r = D / 2;
		return cp::average(ssim_map, r, r, r, r);
	}
#pragma endregion
}