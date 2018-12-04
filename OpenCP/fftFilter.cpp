#include "fftFilter.hpp"

using namespace cv;
using namespace std;

namespace cp
{
	void psf2otf(const cv::Mat & psf, cv::Mat & otf, const cv::Size & s)
	{
		copyMakeBorder(psf, otf, 0, s.height - psf.rows, 0, s.width - psf.cols, cv::BORDER_CONSTANT, 0);
		//copyMakeBorder(psf, otf, 0, s.height - psf.rows, 0, s.width - psf.cols, cv::BORDER_REPLICATE);
		cv::dft(otf, otf, cv::DFT_COMPLEX_OUTPUT);
	}

	void GaussianFilterFFT64F(const cv::Mat& src_, cv::Mat& dest_, const cv::Mat & kernel)
	{
		Mat src, dest;
		src_.convertTo(src, CV_64F);
		double minDenom = std::sqrt(std::numeric_limits<double>::epsilon());
		// Minimize border effects : size = 2 * size with mirror constraints to obtain periodic image
		cv::copyMakeBorder(src, dest, src.rows / 2, src.rows / 2, src.cols / 2, src.cols / 2, cv::BORDER_REFLECT);

		// Transform image to frequency space
		cv::dft(dest, dest, cv::DFT_COMPLEX_OUTPUT);

		// transform kernel to frequency space (Optical Transfer Fuction)
		cv::Mat otf;
		psf2otf(kernel, otf, dest.size());

		// Actual filtering, with regularization if frequencies amplitude is too small
		std::complex<double> * otf_pnt = (std::complex<double> *) otf.ptr();
		std::complex<double> * tmp_pnt = (std::complex<double> *) dest.ptr();
		{
			//CalcTime t;
			for (int i = 0; i < otf.rows * otf.cols; i++)
			{
				std::complex<double> conjO(otf_pnt[i].real(), -otf_pnt[i].imag());
				std::complex<double> denom = otf_pnt[i];

				tmp_pnt[i] *= conjO;
			}
		}

		// Back in image space
		//cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
		cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);

		// Crop useless data
		cv::Rect outROI(src.cols / 2 - kernel.cols / 2, src.rows / 2 - kernel.rows / 2, src.cols, src.rows);

		Mat(dest(outROI)).convertTo(dest_, src_.depth());
	}

	void GaussianFilterFFT32F(const cv::Mat& src_, cv::Mat& dest_, const cv::Mat & kernel)
	{
		Mat src, dest;
		src_.convertTo(src, CV_32F);
		float minDenom = std::sqrt(std::numeric_limits<float>::epsilon());
		// Minimize border effects : size = 2 * size with mirror constraints to obtain periodic image
		cv::copyMakeBorder(src, dest, src.rows / 2, src.rows / 2, src.cols / 2, src.cols / 2, cv::BORDER_REFLECT);

		// Transform image to frequency space
		cv::dft(dest, dest, cv::DFT_COMPLEX_OUTPUT);

		// transform kernel to frequency space (Optical Transfer Fuction)
		cv::Mat otf;
		psf2otf(kernel, otf, dest.size());

		// Actual filtering, with regularization if frequencies amplitude is too small
		std::complex<float> * otf_pnt = (std::complex<float> *) otf.ptr();
		std::complex<float> * tmp_pnt = (std::complex<float> *) dest.ptr();
		{
			//CalcTime t;
			for (int i = 0; i < otf.rows * otf.cols; i++)
			{
				std::complex<float> conjO(otf_pnt[i].real(), -otf_pnt[i].imag());
				std::complex<float> denom = otf_pnt[i];

				tmp_pnt[i] *= conjO;
			}
		}

		// Back in image space
		//cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
		cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);

		// Crop useless data
		cv::Rect outROI(src.cols / 2 - kernel.cols / 2, src.rows / 2 - kernel.rows / 2, src.cols, src.rows);

		Mat(dest(outROI)).convertTo(dest_, src_.depth());
	}

	void GaussianFilterFFT(const cv::Mat& src, cv::Mat& dest, const cv::Size ksize, const double sigma, int depth)
	{
		int r = (ksize.width / 2);
		int d = 2 * r + 1;

		cv::Mat kernelX = cv::getGaussianKernel(d, sigma, depth);
		cv::Mat kernelY = cv::getGaussianKernel(d, sigma, depth);

		Mat kernel = kernelX * kernelY.t();

		if (depth == CV_32F)
		{
			GaussianFilterFFT32F(src, dest, kernel);
		}
		else if (depth == CV_64F)
		{
			GaussianFilterFFT64F(src, dest, kernel);
		}
	}

	void wienerDeconvolution32F(const cv::Mat& src_, cv::Mat& dest_, const cv::Mat & kernel, float mu)
	{
		Mat src, dest;
		src_.convertTo(src, CV_32F);
		float minDenom = std::sqrt(std::numeric_limits<float>::epsilon());
		// Minimize border effects : size = 2 * size with mirror constraints to obtain periodic image
		cv::copyMakeBorder(src, dest, src.rows / 2, src.rows / 2, src.cols / 2, src.cols / 2, cv::BORDER_REFLECT);

		// Transform image to frequency space
		cv::dft(dest, dest, cv::DFT_COMPLEX_OUTPUT);

		// transform kernel to frequency space (Optical Transfer Fuction)
		cv::Mat otf;
		psf2otf(kernel, otf, dest.size());

		// Actual filtering, with regularization if frequencies amplitude is too small
		std::complex<float> * otf_pnt = (std::complex<float> *) otf.ptr();
		std::complex<float> * tmp_pnt = (std::complex<float> *) dest.ptr();
		{
			//CalcTime t;
			for (int i = 0; i < otf.rows * otf.cols; i++)
			{
				std::complex<float> conjO(otf_pnt[i].real(), -otf_pnt[i].imag());
				std::complex<float> denom = conjO * otf_pnt[i] + mu;

				tmp_pnt[i] *= conjO / denom;
			}
		}

		// Back in image space
		//cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
		cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);

		// Crop useless data
		cv::Rect outROI(src.cols / 2 - kernel.cols / 2, src.rows / 2 - kernel.rows / 2, src.cols, src.rows);

		Mat(dest(outROI)).convertTo(dest_, src_.depth(), 1.0 + mu);
	}

	void wienerDeconvolution64F(const cv::Mat& src_, cv::Mat& dest_, const cv::Mat & kernel, double mu)
	{
		Mat src, dest;
		src_.convertTo(src, CV_64F);
		double minDenom = std::sqrt(std::numeric_limits<double>::epsilon());
		// Minimize border effects : size = 2 * size with mirror constraints to obtain periodic image
		cv::copyMakeBorder(src, dest, src.rows / 2, src.rows / 2, src.cols / 2, src.cols / 2, cv::BORDER_REFLECT);

		// Transform image to frequency space
		cv::dft(dest, dest, cv::DFT_COMPLEX_OUTPUT);

		// transform kernel to frequency space (Optical Transfer Fuction)
		cv::Mat otf;
		psf2otf(kernel, otf, dest.size());

		// Actual filtering, with regularization if frequencies amplitude is too small
		std::complex<double>* otf_pnt = (std::complex<double> *) otf.ptr();
		std::complex<double>* tmp_pnt = (std::complex<double> *) dest.ptr();
		
		for (int i = 0; i < otf.rows * otf.cols; i++)
		{
			std::complex<double> conjO(otf_pnt[i].real(), -otf_pnt[i].imag());
			std::complex<double> denom = conjO * otf_pnt[i] + mu;
			/*if (std::abs(denom) < minDenom)
			{
			tmp_pnt[i] *= conjO / minDenom;
			}
			else*/
			{
				tmp_pnt[i] *= conjO / denom;
			}
		}

		// Back in image space
		//cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
		cv::dft(dest, dest, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);

		// Crop useless data
		cv::Rect outROI(src.cols / 2 - kernel.cols / 2, src.rows / 2 - kernel.rows / 2, src.cols, src.rows);

		Mat(dest(outROI)).convertTo(dest_, src_.depth(), 1.0 + mu);
	}

	void wienerDeconvolution(const cv::Mat& src_, cv::Mat& dest_, const cv::Mat & kernel, double mu)
	{
		wienerDeconvolution32F(src_, dest_, kernel, mu);
		//wienerDeconvolution64F(src_, dest_, kernel, mu);
	}

	void wienerDeconvolutionGauss(const cv::Mat& src, cv::Mat& dest, const Size ksize, const double sigma, const double eps, const int depth)
	{
		int r = (ksize.width / 2) * 2;
		int d = 2 * r + 1;

		cv::Mat kernelX = cv::getGaussianKernel(d, sigma, depth);
		cv::Mat kernelY = cv::getGaussianKernel(d, sigma, depth);

		Mat kernel = kernelX * kernelY.t();

		if (depth == CV_32F) wienerDeconvolution32F(src, dest, kernel, eps);
		else if (depth == CV_64F) wienerDeconvolution64F(src, dest, kernel, eps);
	}
}