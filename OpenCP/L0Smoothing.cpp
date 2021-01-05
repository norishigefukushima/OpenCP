//https://github.com/cache-tlb/L0Smoothing
#include "L0Smoothing.hpp"
using namespace std;
using namespace cv;

namespace cp
{


	void circshift(cv::Mat &A, int shitf_row, int shift_col, Mat& temp)
	{
		int row = A.rows, col = A.cols;
		shitf_row = (row + (shitf_row % row)) % row;
		shift_col = (col + (shift_col % col)) % col;

		A.copyTo(temp);
		if (shitf_row)
		{
			temp.rowRange(row - shitf_row, row).copyTo(A.rowRange(0, shitf_row));
			temp.rowRange(0, row - shitf_row).copyTo(A.rowRange(shitf_row, row));
		}
		if (shift_col)
		{
			temp.colRange(col - shift_col, col).copyTo(A.colRange(0, shift_col));
			temp.colRange(0, col - shift_col).copyTo(A.colRange(shift_col, col));
		}
		return;
	}

	cv::Mat psf2otf(const cv::Mat &psf, const cv::Size &outSize)
	{
		cv::Size psfSize = psf.size();
		cv::Mat new_psf = cv::Mat(outSize, CV_32FC2);
		new_psf.setTo(0);
		//new_psf(cv::Rect(0,0,psfSize.width, psfSize.height)).setTo(psf);
		for (int i = 0; i < psfSize.height; i++)
		{
			for (int j = 0; j < psfSize.width; j++)
			{
				new_psf.at<cv::Vec2f>(i, j)[0] = psf.at<float>(i, j);
			}
		}

		Mat buff;
		circshift(new_psf, -1 * int(floor(psfSize.height*0.5)), -1 * int(floor(psfSize.width*0.5)), buff);

		cv::Mat otf;
		cv::dft(new_psf, otf, cv::DFT_COMPLEX_OUTPUT);

		return otf;
	}

	template<typename srcType>
	srcType sqr(const srcType x) { return x*x; }

	cv::Mat psf2otf_64F(const cv::Mat &psf, const cv::Size &outSize) {
		cv::Size psfSize = psf.size();
		cv::Mat new_psf = cv::Mat(outSize, CV_64FC2);
		new_psf.setTo(0);
		//new_psf(cv::Rect(0,0,psfSize.width, psfSize.height)).setTo(psf);
		for (int i = 0; i < psfSize.height; i++) {
			for (int j = 0; j < psfSize.width; j++) {
				new_psf.at<cv::Vec2d>(i, j)[0] = psf.at<double>(i, j);
			}
		}
		Mat buff;
		circshift(new_psf, -1 * int(floor(psfSize.height*0.5)), -1 * int(floor(psfSize.width*0.5)), buff);

		cv::Mat otf;
		cv::dft(new_psf, otf, cv::DFT_COMPLEX_OUTPUT);

		return otf;
	}
	//base filter
	void L0Smoothing_64F(cv::Mat &im8uc3, cv::Mat& dest, double lambda = 2e-2, double kappa = 2.0)
	{
		// convert the image to double format
		int row = im8uc3.rows, col = im8uc3.cols;
		cv::Mat S;
		im8uc3.convertTo(S, CV_64FC3, 1. / 255.);

		cv::Mat fx(1, 2, CV_64FC1);
		cv::Mat fy(2, 1, CV_64FC1);
		fx.at<double>(0) = 1; fx.at<double>(1) = -1;
		fy.at<double>(0) = 1; fy.at<double>(1) = -1;

		cv::Size sizeI2D = im8uc3.size();
		cv::Mat otfFx = psf2otf_64F(fx, sizeI2D);
		cv::Mat otfFy = psf2otf_64F(fy, sizeI2D);

		cv::Mat Normin1[3];
		cv::Mat single_channel[3];
		cv::split(S, single_channel);
		for (int k = 0; k < 3; k++) {
			cv::dft(single_channel[k], Normin1[k], cv::DFT_COMPLEX_OUTPUT);
		}
		cv::Mat Denormin2(row, col, CV_64FC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				cv::Vec2d &c1 = otfFx.at<cv::Vec2d>(i, j), &c2 = otfFy.at<cv::Vec2d>(i, j);
				Denormin2.at<double>(i, j) = sqr(c1[0]) + sqr(c1[1]) + sqr(c2[0]) + sqr(c2[1]);
			}
		}

		double beta = 2.0*lambda;
		double betamax = 1e5;

		while (beta < betamax) {
			cv::Mat Denormin = 1.0 + beta*Denormin2;

			// h-v subproblem
			cv::Mat dx[3], dy[3];
			for (int k = 0; k < 3; k++) {
				cv::Mat shifted_x = single_channel[k].clone();
				Mat buff;
				circshift(shifted_x, 0, -1, buff);
				dx[k] = shifted_x - single_channel[k];

				cv::Mat shifted_y = single_channel[k].clone();
				circshift(shifted_y, -1, 0, buff);
				dy[k] = shifted_y - single_channel[k];
			}
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					double val =
						sqr(dx[0].at<double>(i, j)) + sqr(dy[0].at<double>(i, j)) +
						sqr(dx[1].at<double>(i, j)) + sqr(dy[1].at<double>(i, j)) +
						sqr(dx[2].at<double>(i, j)) + sqr(dy[2].at<double>(i, j));

					if (val < lambda / beta) {
						dx[0].at<double>(i, j) = dx[1].at<double>(i, j) = dx[2].at<double>(i, j) = 0.0;
						dy[0].at<double>(i, j) = dy[1].at<double>(i, j) = dy[2].at<double>(i, j) = 0.0;
					}
				}
			}

			// S subproblem
			for (int k = 0; k < 3; k++) {
				cv::Mat shift_dx = dx[k].clone();
				Mat buff;
				circshift(shift_dx, 0, 1, buff);
				cv::Mat ddx = shift_dx - dx[k];

				cv::Mat shift_dy = dy[k].clone();
				circshift(shift_dy, 1, 0, buff);
				cv::Mat ddy = shift_dy - dy[k];
				cv::Mat Normin2 = ddx + ddy;
				cv::Mat FNormin2;
				cv::dft(Normin2, FNormin2, cv::DFT_COMPLEX_OUTPUT);
				cv::Mat FS = Normin1[k] + beta*FNormin2;
				for (int i = 0; i < row; i++) {
					for (int j = 0; j < col; j++) {
						FS.at<cv::Vec2d>(i, j)[0] /= Denormin.at<double>(i, j);
						FS.at<cv::Vec2d>(i, j)[1] /= Denormin.at<double>(i, j);
					}
				}
				cv::Mat ifft;
				cv::idft(FS, ifft, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
				for (int i = 0; i < row; i++) {
					for (int j = 0; j < col; j++) {
						single_channel[k].at<double>(i, j) = ifft.at<cv::Vec2d>(i, j)[0];
					}
				}
			}
			beta *= kappa;
		}
		cv::merge(single_channel, 3, dest);
	}

#define SQR(x) ((x)*(x))

	void L0Smoothing(cv::Mat &im8uc3, cv::Mat& dest, const float lambda, const float kappa)
	{
		// convert the image to double format
		int row = im8uc3.rows, col = im8uc3.cols;
		int size = row*col;
		cv::Mat S;
		im8uc3.convertTo(S, CV_32FC3, 1. / 255.);

		cv::Mat fx(1, 2, CV_32FC1);
		cv::Mat fy(2, 1, CV_32FC1);
		fx.at<float>(0) = 1; fx.at<float>(1) = -1;
		fy.at<float>(0) = 1; fy.at<float>(1) = -1;

		cv::Size sizeI2D = im8uc3.size();
		cv::Mat otfFx = psf2otf(fx, sizeI2D);
		cv::Mat otfFy = psf2otf(fy, sizeI2D);

		cv::Mat Normin1[3];
		cv::Mat single_channel[3];

		cv::split(S, single_channel);

		cv::Mat buffer(S.size(), CV_32F);

		for (int k = 0; k < 3; k++)
		{
			cv::dft(single_channel[k], Normin1[k], cv::DFT_COMPLEX_OUTPUT);
		}

		cv::Mat Denormin2(row, col, CV_32FC1);


		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				cv::Vec2f &c1 = otfFx.at<cv::Vec2f>(i, j), &c2 = otfFy.at<cv::Vec2f>(i, j);
				Denormin2.at<float>(i, j) = SQR(c1[0]) + SQR(c1[1]) + SQR(c2[0]) + SQR(c2[1]);
			}
		}


		// the bigger beta the more time iteration
		float beta = 4.f*lambda;
		// the smaller betamax the less segmentation count
		double betamax = 1e5;
		//float betamax = 3e1;

		cv::Mat Denormin;
		cv::Mat shifted_x;
		cv::Mat shifted_y;
		cv::Mat dx[3], dy[3];

		cv::Mat FNormin2;

		while (beta < betamax)
		{
			addWeighted(Mat::ones(Denormin2.size(), Denormin2.type()), 1.0, Denormin2, beta, 0.0, Denormin);

			Denormin = 1.f / Denormin;
			// h-v subproblem
			for (int k = 0; k < 3; k++)
			{
				single_channel[k].copyTo(shifted_x);
				circshift(shifted_x, 0, -1, buffer);
				dx[k] = shifted_x - single_channel[k];

				single_channel[k].copyTo(shifted_y);
				circshift(shifted_y, -1, 0, buffer);
				dy[k] = shifted_y - single_channel[k];
			}

			const float lb = lambda / beta;

			float* dx0 = dx[0].ptr<float>(0);
			float* dx1 = dx[1].ptr<float>(0);
			float* dx2 = dx[2].ptr<float>(0);
			float* dy0 = dy[0].ptr<float>(0);
			float* dy1 = dy[1].ptr<float>(0);
			float* dy2 = dy[2].ptr<float>(0);
			const __m128 mlb = _mm_set1_ps(lb);
			cv::Mat buff(4, 1, CV_32F);
			float* b = (float*)buff.ptr<float>(0);
			int i = 0;
			for (; i <= size - 4; i += 4)
			{
				__m128 x = _mm_load_ps(dx0 + i);
				__m128 v = _mm_mul_ps(x, x);
				x = _mm_load_ps(dx1 + i);
				v = _mm_add_ps(v, _mm_mul_ps(x, x));
				x = _mm_load_ps(dx2 + i);
				v = _mm_add_ps(v, _mm_mul_ps(x, x));
				x = _mm_load_ps(dy0 + i);
				v = _mm_add_ps(v, _mm_mul_ps(x, x));
				x = _mm_load_ps(dy1 + i);
				v = _mm_add_ps(v, _mm_mul_ps(x, x));
				x = _mm_load_ps(dy2 + i);
				v = _mm_add_ps(v, _mm_mul_ps(x, x));

				_mm_store_ps(b, v);
				if (b[0] < lb)
				{
					dx0[i] = dx1[i] = dx2[i] = dy0[i] = dy1[i] = dy2[i] = 0.f;
				}
				if (b[1] < lb)
				{
					dx0[i + 1] = dx1[i + 1] = dx2[i + 1] = dy0[i + 1] = dy1[i + 1] = dy2[i + 1] = 0.f;
				}
				if (b[2] < lb)
				{
					dx0[i + 2] = dx1[i + 2] = dx2[i + 2] = dy0[i + 2] = dy1[i + 2] = dy2[i + 2] = 0.f;
				}
				if (b[3] < lb)
				{
					dx0[i + 3] = dx1[i + 3] = dx2[i + 3] = dy0[i + 3] = dy1[i + 3] = dy2[i + 3] = 0.f;
				}
			}
			for (; i < size; i++)
			{
				float v = dx0[i] * dx0[i] + dx1[i] * dx1[i] + dx2[i] * dx2[i] + dy0[i] * dy0[i] + dy1[i] * dy1[i] + dy2[i] * dy2[i];
				if (v < lb)
				{
					dx0[i] = dx1[i] = dx2[i] = dy0[i] = dy1[i] = dy2[i] = 0.f;
				}
			}

			// S subproblem
			for (int k = 0; k < 3; k++)
			{
				dx[k].copyTo(shifted_x);
				circshift(shifted_x, 0, 1, buffer);
				dy[k].copyTo(shifted_y);
				circshift(shifted_y, 1, 0, buffer);

				cv::Mat Normin2 = shifted_x - dx[k] + shifted_y - dy[k];

				cv::dft(Normin2, FNormin2, cv::DFT_COMPLEX_OUTPUT);

				//cv::Mat FS = Normin1[k] + beta*FNormin2;
				//FS*=real(Denormin);
				float* n1 = (float*)Normin1[k].ptr<Vec2f>(0);
				float* fn2 = (float*)FNormin2.ptr<Vec2f>(0);
				float* D = Denormin.ptr<float>(0);
				const __m128 mbeta = _mm_set1_ps(beta);
				int i = 0;
				for (; i <= size * 2 - 4; i += 4)
				{
					__m128 mfn2 = _mm_add_ps(_mm_loadu_ps(n1 + i), _mm_mul_ps(mbeta, _mm_loadu_ps(fn2 + i)));
					__m128 mn1 = _mm_loadu_ps(D + (i >> 1));
					mn1 = _mm_shuffle_ps(mn1, mn1, _MM_SHUFFLE(1, 1, 0, 0));
					mfn2 = _mm_mul_ps(mn1, mfn2);
					_mm_storeu_ps(fn2 + i, mfn2);
				}
				for (; i < size * 2; i += 2)
				{
					const float dd = D[(i >> 1)];
					fn2[i] = dd*(n1[i] + beta*fn2[i]);
					fn2[i + 1] = dd*(n1[i + 1] + beta*fn2[i + 1]);
				}

				cv::idft(FNormin2, single_channel[k], cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
			}
			beta *= kappa;
		}

		cv::merge(single_channel, 3, S);
		S.convertTo(dest, CV_8UC3, 255.f);
	}
}