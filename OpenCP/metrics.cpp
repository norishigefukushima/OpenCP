#include "metrics.hpp"
#include "inlineMathFunctions.hpp"
#include "inlineSIMDFunctions.hpp"
#include "arithmetic.hpp"
#include "updateCheck.hpp"
#include "debugcp.hpp"

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
		MSE +=_mm256_reduceadd_pd(mmse);
		

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
				const double v = (double)hist.at<float>(j) * invsum;
				if (v != 0) ret -= v * log2(v);
			}
		}
		return ret;
	}

	double getEntropy(cv::InputArray src, cv::InputArray mask_)
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