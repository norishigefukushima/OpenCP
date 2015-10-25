#include "libGaussian\gaussian_conv.h"
#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void GaussianBlurSR(cv::InputArray src, cv::OutputArray dest, const double sigma);
	void GaussianBlurAM(cv::InputArray src, cv::OutputArray dest, float sigma, int iteration);

	void GaussianBlurIPOLDCT(InputArray src_, OutputArray dest, const double sigma_space)
	{
		Mat src = src_.getMat();
		Mat srcf, destf;

		if (src.depth() == CV_64F)
		{
			src.convertTo(srcf, CV_64F);
			destf = Mat::zeros(src.size(), CV_64F);
			dct_coeffs < double > c;
			if (src_.channels() == 3)
			{
				Mat temp = Mat::zeros(Size(src.cols, src.rows * 3), CV_64F);
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);

				dct_precomp_image(&c, temp.ptr<double>(0), plane.ptr<double>(0), src.cols, src.rows, 3, sigma_space);
				dct_gaussian_conv(c);
				dct_free(&c);
				cvtColorPLANE2BGR(temp, destf);

			}
			else if (src_.channels() == 1)
			{
				dct_precomp_image(&c, destf.ptr<double>(0), srcf.ptr<double>(0), src.cols, src.rows, 1, sigma_space);
				dct_gaussian_conv(c);
				dct_free(&c);
			}
		}
		else
		{
			src.convertTo(srcf, CV_32F);
			destf = Mat::zeros(src.size(), CV_32F);
			dct_coeffs<float> c;
			if (src_.channels() == 3)
			{
				Mat temp = Mat::zeros(Size(src.cols, src.rows * 3), CV_32F);
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);

				dct_precomp_image(&c, temp.ptr<float>(0), plane.ptr<float>(0), src.cols, src.rows, 3, (float)sigma_space);
				dct_gaussian_conv(c);
				dct_free(&c);
				cvtColorPLANE2BGR(temp, destf);

			}
			else if (src_.channels() == 1)
			{
				dct_precomp_image(&c, destf.ptr<float>(0), srcf.ptr<float>(0), src.cols, src.rows, 1, (float)sigma_space);
				dct_gaussian_conv(c);
				dct_free(&c);
			}
		}


		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			destf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			destf.copyTo(dest);
	}

	void GaussianBlurIPOLFIR(InputArray src_, OutputArray dest, const double sigma_space, const double tol)
	{
		if (src_.channels() == 3) printf("color is not support");
		Mat src = src_.getMat();
		Mat srcf, destf;
		if (src.depth() == CV_64F)
		{
			src.convertTo(srcf, CV_64F);
			destf = Mat::zeros(src.size(), CV_64F);
			Mat buff(Size(max(src.cols, src.rows), 1), CV_64F);

			fir_coeffs<double> c;
			fir_precomp(&c, sigma_space, tol);
			fir_gaussian_conv_image(c, destf.ptr<double>(0), buff.ptr<double>(0), srcf.ptr<double>(0), src.cols, src.rows, src.channels());
			fir_free(&c);
		}
		else
		{
			src.convertTo(srcf, CV_32F);
			destf = Mat::zeros(src.size(), CV_32F);
			Mat buff(Size(max(src.cols, src.rows), 1), CV_32F);

			fir_coeffs<float> c;
			fir_precomp(&c, (float)sigma_space, (float)tol);
			fir_gaussian_conv_image(c, destf.ptr<float>(0), buff.ptr<float>(0), srcf.ptr<float>(0), src.cols, src.rows, src.channels());
			fir_free(&c);
		}

		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			destf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			destf.copyTo(dest);
	}

	void GaussianBlurIPOLBox(InputArray src_, OutputArray dest, const double sigma_space, const int K)
	{
		if (src_.channels() == 3) printf("color is not support");
		Mat src = src_.getMat();
		Mat srcf;
		src.convertTo(srcf, CV_32F);
		Mat destf = Mat::zeros(src.size(), CV_32F);
		Mat buff(Size(max(src.cols, src.rows), 1), CV_32F);
		box_gaussian_conv_image(destf.ptr<float>(0), buff.ptr<float>(0), srcf.ptr<float>(0), src.cols, src.rows, src.channels(), (float)sigma_space, K);

		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			destf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			destf.copyTo(dest);
	}

	void GaussianBlurIPOLEBox(InputArray src_, OutputArray dest, const double sigma_space, const int K)
	{
		if (src_.channels() == 3) printf("color is not support");
		Mat src = src_.getMat();
		Mat srcf;
		src.convertTo(srcf, CV_32F);
		Mat destf = Mat::zeros(src.size(), CV_32F);
		Mat buff(Size(max(src.cols, src.rows), 1), CV_32F);

		ebox_coeffs<float> c;
		ebox_precomp(&c, (float)sigma_space, K);
		ebox_gaussian_conv_image(c, destf.ptr<float>(0), buff.ptr<float>(0), srcf.ptr<float>(0), src.cols, src.rows, src.channels());

		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			destf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			destf.copyTo(dest);
	}

	void GaussianBlurIPOLSII(InputArray src_, OutputArray dest, const double sigma_space, const int K)
	{
		if (src_.channels() == 3) printf("color is not support");
		Mat src = src_.getMat();
		Mat srcf;

		if (!SII_VALID_K(K))
		{
			fprintf(stderr, "Error: K=%d is invalid for SII\n", K);
		}

		if (src.depth() == CV_64F)
		{
			//srcf.create(src.size(), CV_64F);
			srcf = src.clone();

			sii_coeffs<double> c;
			sii_precomp(c, sigma_space, K);
			Mat buff(Size(sii_buffer_size(c, max(src.cols, src.rows)), 1), CV_64F);

			sii_gaussian_conv_image(c, srcf.ptr<double>(0), buff.ptr < double >(0), src.ptr<double>(0), src.cols, src.rows, src.channels());
		}
		else
		{
			src.convertTo(srcf, CV_32F);

			sii_coeffs<float> c;
			sii_precomp(c, (float)sigma_space, K);

			Mat buff(Size(sii_buffer_size(c, max(src.cols, src.rows)), 1), CV_32F);

			sii_gaussian_conv_image(c, srcf.ptr<float>(0), buff.ptr <float>(0), srcf.ptr<float>(0), src.cols, src.rows, src.channels());
		}

		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			srcf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			srcf.copyTo(dest);
	}

	void GaussianBlurIPOLAM(InputArray src_, OutputArray dest, const double sigma, const int K, const double tol)
	{
		Mat src = src_.getMat();
		Mat srcf;

		if (src.depth() == CV_64F)
		{
			if (src.channels() == 1)
			{
				srcf = src.clone();
				am_gaussian_conv_image(srcf.ptr<double>(0), srcf.ptr<double>(0), src.cols, src.rows, 1, sigma, K, tol, true);
			}
			else if (src.channels() == 3)
			{
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);
				am_gaussian_conv_image(plane.ptr<double>(0), plane.ptr<double>(0), src.cols, src.rows, 3, sigma, K, tol, true);
				cvtColorPLANE2BGR(plane, srcf);
			}
		}
		else
		{
			src.convertTo(srcf, CV_32F);
			if (src.channels() == 1)
			{
				srcf = src.clone();
				am_gaussian_conv_image(srcf.ptr<float>(0), srcf.ptr<float>(0), src.cols, src.rows, 1, (float)sigma, K, (float)tol, true);
			}
			else if (src.channels() == 3)
			{
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);
				am_gaussian_conv_image(plane.ptr<float>(0), plane.ptr<float>(0), src.cols, src.rows, 3, (float)sigma, K, (float)tol, true);
				cvtColorPLANE2BGR(plane, srcf);
			}
		}

		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			srcf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			srcf.copyTo(dest);
	}

	void GaussianBlurIPOLDeriche(InputArray src_, OutputArray dest, const double sigma, const int K)
	{
		Mat src = src_.getMat();
		Mat srcf;
		int clip_k = K;
		if (!DERICHE_VALID_K(K))
		{
			fprintf(stderr, "Error: K=%d is invalid for Deriche\n", K);
			clip_k = max(DERICHE_MIN_K, min(DERICHE_MAX_K, K));
		}

		if (src.depth() == CV_64F)
		{
			srcf = src.clone();
			deriche_coeffs<double> c;
			deriche_precomp(&c, sigma, clip_k, 1e-6);

			Mat buffer(src.cols * 2 * src.rows, 1, CV_64F);

			if (src_.channels() == 3)
			{
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);
				deriche_gaussian_conv_image(c, plane.ptr<double>(0), buffer.ptr<double>(0), plane.ptr<double>(0), src.cols, src.rows, 3);
				cvtColorPLANE2BGR(plane, srcf);
			}
			else if (src_.channels() == 1)
			{
				deriche_gaussian_conv_image(c, srcf.ptr<double>(0), buffer.ptr<double>(0), srcf.ptr<double>(0), src.cols, src.rows, 1);
			}
		}
		else
		{
			src.convertTo(srcf, CV_32F);
			deriche_coeffs<float> c;
			deriche_precomp(&c, (float)sigma, clip_k, (float)1e-6);

			Mat buffer(src.cols * 2 * src.rows, 1, CV_32F);

			if (src_.channels() == 3)
			{
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);
				deriche_gaussian_conv_image(c, plane.ptr<float>(0), buffer.ptr<float>(0), plane.ptr<float>(0), src.cols, src.rows, 3);
				cvtColorPLANE2BGR(plane, srcf);
			}
			else if (src_.channels() == 1)
			{
				deriche_gaussian_conv_image(c, srcf.ptr<float>(0), buffer.ptr<float>(0), srcf.ptr<float>(0), src.cols, src.rows, 1);
			}
		}


		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			srcf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			srcf.copyTo(dest);
	}

	void GaussianBlurIPOLVYV(InputArray src_, OutputArray dest, const double sigma_space, const int K, const double tol)
	{
		Mat src = src_.getMat();
		Mat srcf;
		int clip_k = K;
		if (!VYV_VALID_K(K))
		{
			fprintf(stderr, "Error: K=%d is invalid for VYV\n", K);
			clip_k = max(VYV_MIN_K, min(VYV_MAX_K, K));
		}

		if (src.depth() == CV_64F)
		{
			srcf = src.clone();
			vyv_coeffs<double> c;
			vyv_precomp(&c, sigma_space, clip_k, tol);
			if (src_.channels() == 3)
			{
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);
				vyv_gaussian_conv_image(c, plane.ptr<double>(0), plane.ptr<double>(0), src.cols, src.rows, 3);
				cvtColorPLANE2BGR(plane, srcf);
			}
			else if (src_.channels() == 1)
			{
				vyv_gaussian_conv_image(c, srcf.ptr<double>(0), src.ptr<double>(0), src.cols, src.rows, 1);
			}
		}
		else
		{
			vyv_coeffs<float> c;
			vyv_precomp(&c, (float)sigma_space, clip_k, (float)tol);
			if (src.depth() == CV_32F) srcf = src.clone();
			else  src.convertTo(srcf, CV_32F);
			if (src_.channels() == 3)
			{
				Mat plane;
				cvtColorBGR2PLANE(srcf, plane);
				vyv_gaussian_conv_image(c, plane.ptr<float>(0), plane.ptr<float>(0), src.cols, src.rows, 3);
				cvtColorPLANE2BGR(plane, srcf);
			}
			else if (src_.channels() == 1)
			{
				vyv_gaussian_conv_image(c, srcf.ptr<float>(0), src.ptr<float>(0), src.cols, src.rows, 1);
			}
		}

		if (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32S)
			srcf.convertTo(dest, src.depth(), 1.0, 0.5);
		else
			srcf.copyTo(dest);
	}

	void GaussianFilter(InputArray src, OutputArray dest, const double sigma_space, const int filter_method, const int K_, const double tol)
	{
		int K = K_;
		switch (filter_method)
		{
		case GAUSSIAN_FILTER_DCT:
			GaussianBlurIPOLDCT(src, dest, sigma_space);
			break;
		case GAUSSIAN_FILTER_FIR:
			GaussianBlurIPOLFIR(src, dest, sigma_space, tol);
			break;
		case GAUSSIAN_FILTER_BOX:
			K = (K != 0) ? K_ : 3;
			GaussianBlurIPOLBox(src, dest, sigma_space, K);
			break;
		case GAUSSIAN_FILTER_EBOX:
			K = (K != 0) ? K_ : 4;
			GaussianBlurIPOLEBox(src, dest, sigma_space, K);
			break;
		case GAUSSIAN_FILTER_SII:
			K = (K != 0) ? K_ : 3;
			GaussianBlurIPOLSII(src, dest, sigma_space, K);
			break;
		case GAUSSIAN_FILTER_AM:
			K = (K != 0) ? K_ : 5;
			GaussianBlurIPOLAM(src, dest, sigma_space, K, tol);
			break;
		case GAUSSIAN_FILTER_AM2:
			K = (K != 0) ? K_ : 5;
			GaussianBlurAM(src, dest, (float)sigma_space, K);
			break;
		case GAUSSIAN_FILTER_DERICHE:
			K = (K != 0) ? K_ : 3;
			GaussianBlurIPOLDeriche(src, dest, sigma_space, K);
			break;
		case GAUSSIAN_FILTER_VYV:
			K = (K != 0) ? K_ : 4;
			GaussianBlurIPOLVYV(src, dest, sigma_space, K, tol);
			break;
		default:
		case GAUSSIAN_FILTER_SR:
			GaussianBlurSR(src, dest, sigma_space);
			break;
		}
	}
}