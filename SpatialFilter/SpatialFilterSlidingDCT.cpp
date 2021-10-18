#include "stdafx.h"

#ifdef USE_EIGEN
//#define EIGEN_DONT_VECTORIZE
//#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>
#endif

using namespace std;
using namespace cv;

namespace cp
{
	void plotDCTKernel(string wname, bool isWait, const double* GCn, const int radius, const int order, const double G0, const double sigma)
	{
		const int size = (order) * (radius + 1);
		AutoBuffer<float> G(size);
		for (int i = 0; i < size; i++)G[i] = (float)GCn[i];
		plotDCTKernel(wname, isWait, G, radius, order, (float)G0, sigma);
	}

	void plotDCTKernel(string wname, bool isWait, const float* GCn, const int radius, const int order, const float G0, const double sigma)
	{
		Mat gauss(radius + 1, 1, CV_64F);
		cp::setGaussKernelHalf(gauss, radius, sigma, true);

		static int pre_order = 0;
		cp::Plot p;
		cv::namedWindow(wname);
		const int allIndex = order + 2;
		const int totalIndex = allIndex - 1;
		static int fk = allIndex; createTrackbar("k", wname, &fk, allIndex);
		if (pre_order != order)
		{
			pre_order = order;
			fk = allIndex;
			setTrackbarMax("k", wname, allIndex);
			setTrackbarPos("k", wname, fk);
		}

		int key = 0;
		for (int k = 0; k <= order; k++)
		{
			p.setPlotTitle(k, format("order %d", k));
		}
		p.setPlotTitle(order + 1, "total");

		if (G0 == 0.f)
		{
			while (key != 'q')
			{
				double error = 0.0;
				if (fk == allIndex)
				{
					for (int k = 0; k <= order; k++)
					{
						for (int n = -radius; n <= radius; n++)
						{
							//p.push_back(n, GCn[(order + 1) * abs(n) + order - k], k);
							p.push_back(n, GCn[(order + 1) * abs(n) + k], k);
						}
					}
					//total
					for (int n = -radius; n <= radius; n++)
					{
						float sum = 0.f;
						for (int k = 0; k <= order; k++)
						{
							sum += GCn[(order + 1) * abs(n) + k];
						}
						p.push_back(n, sum, totalIndex);
						double v = gauss.at<double>(abs(n)) - sum;
						error += v * v;
					}
				}
				else
				{
					if (fk == totalIndex)
					{
						for (int n = -radius; n <= radius; n++)
						{
							float sum = 0.f;
							for (int k = 0; k <= order; k++)
							{
								sum += GCn[(order + 1) * abs(n) + k];
							}
							p.push_back(n, sum, totalIndex);
							double v = gauss.at<double>(abs(n)) - sum;
							error += v * v;
						}
					}
					else
					{
						for (int n = -radius; n <= radius; n++)
						{
							p.push_back(n, GCn[(order + 1) * abs(n) + order - fk], fk);
						}
					}
				}
				p.plot(wname, false, "", format("10log10(1/mse) %f", 10 * log10(1 / error)));
				p.clear();
				key = waitKey(1);
				if (!isWait)break;
			}
		}
		else //DCT1,5
		{
			while (key != 'q')
			{
				double error = 0.0;
				if (fk == allIndex)
				{
					for (int n = -radius; n <= radius; n++)
					{
						p.push_back(n, G0, 0);
					}
					for (int k = 0; k < order; k++)
					{
						for (int n = -radius; n <= radius; n++)
						{
							p.push_back(n, GCn[order * abs(n) + k], k + 1);
						}
					}
					//total
					for (int n = -radius; n <= radius; n++)
					{
						float sum = G0;
						for (int k = 0; k < order; k++)
						{
							sum += GCn[order * abs(n) + k];
						}
						p.push_back(n, sum, totalIndex);
						double v = gauss.at<double>(abs(n)) - sum;
						error += v * v;
					}
				}
				else
				{
					if (fk == totalIndex)
					{
						for (int n = -radius; n <= radius; n++)
						{
							float sum = G0;
							for (int k = 0; k < order; k++)
							{
								sum += GCn[(order)*abs(n) + k];
							}
							p.push_back(n, sum, totalIndex);
							double v = gauss.at<double>(abs(n)) - sum;
							error += v * v;
						}
					}
					else if (fk == 0)
					{
						for (int n = -radius; n <= radius; n++)
						{
							p.push_back(n, G0, 0);
						}
					}
					else
					{
						for (int n = -radius; n <= radius; n++)
						{
							p.push_back(n, GCn[(order)*abs(n) + fk - 1], fk);
						}
					}
				}
				p.plot(wname, false, "", format("10log10(1/mse) %f", 10 * log10(1 / error)));

				p.clear();
				key = waitKey(1);
				if (!isWait)break;
			}
		}
	}

	void computeSpectrumGaussianClosedForm(const double sigma, const int K, const int R, const int dcttype, double* destSpect)
	{
		double omega = 0.0;
		double k0 = 0.0;
		switch (dcttype)
		{
		case 1:	omega = CV_2PI / (2.0 * R + 0.0); k0 = 0.0; break;
		case 3:	omega = CV_2PI / (2.0 * R + 2.0); k0 = 0.5; break;
		case 5:	omega = CV_2PI / (2.0 * R + 1.0); k0 = 0.0; break;
		case 7:	omega = CV_2PI / (2.0 * R + 1.0); k0 = 0.5; break;
			//default: throw "Unsupported DCT type"; break;
		}

		for (int k = 0; k <= K; k++)
		{
			double coeff = sigma * omega * (k + k0);
			destSpect[k] = 2.0 * exp(-0.5 * coeff * coeff);
		}

		if (dcttype == 1 || dcttype == 5)
			destSpect[0] = 1.0;
	}

	void computeCtWCinv(Mat& dest, const int K, const int R, const int dctType)
	{
		Mat Minv = Mat::eye(K, K, CV_64F);

		double T = 0.0;
		switch (dctType)
		{
		case 1:
		{
			Minv.at<double>(0, 0) = 0.5;
			//Minv.at<double>(K - 1, K - 1) = 0.5;
			T = (2.0 * R + 0.0); break;
		}
		case 3:
		{
			//Minv.at<double>(0, 0) = 0.5;
			T = (2.0 * R + 2.0); break;
		}
		case 5:
		{
			Minv.at<double>(0, 0) = 0.5;
			T = (2.0 * R + 1.0); break;
		}
		case 7:
		{
			//Minv.at<double>(0, 0) = 0.5;
			//Minv.at<double>(K - 1, K - 1) = 0.5;
			T = (2.0 * R + 1.0); break;
		}
		default: throw "Unsupported DCT type"; break;
		}

		if (dctType == 1)
		{
			Mat s = Mat::ones(K, 1, CV_64F);
			for (int i = 1; i < K; i += 2)s.at<double>(i) = -1.0;

			Mat Minvs = Minv * s;
			//cv::trace(Minv).val[0]=K-2+0.5+0.5
			Mat((Minv - (Minvs * Minvs.t()) / (T + (double)K - 1.0))).copyTo(dest);
		}
		else
		{
			//Mat(4/T * Minv).copyTo(dest);
			Minv.copyTo(dest);//scaling (4/T) is not required due to other rescaling.
		}
	}

	void setGaussKernelHalf(Mat& dest, const int R, const double sigma, bool isNormalize)
	{
		CV_Assert(dest.depth() == CV_64F);
		double sum = 0.0;
		const double coeff = 1.0 / (-2.0 * sigma * sigma);
		for (int n = 1; n <= R; ++n)
		{
			const double v = exp((double)(n * n) * coeff);
			//const double v = exp(-(double)(abs(u)) / (sigma));
			//int n = 1;
			//const double v = exp(-(double)pow(abs(u),(double)n) / (n*pow(sigma,double(n))));
			dest.at<double>(n, 0) = v;
			sum += v;
		}
		sum *= 2.0;
		dest.at<double>(0, 0) = 1.0;//u=0
		sum += 1.0;//u=0

		const int rend = int(10.0 * sigma);
		double eout = 0.0;
		for (int i = R + 1; i <= rend; i++)
		{
			const double v = exp(i * i * coeff);
			eout += v;
		}
		dest.at<double>(R, 0) += eout;

		if (isNormalize)
		{
			for (int n = 0; n <= R; ++n)
			{
				dest.at<double>(n, 0) /= sum;
			}
		}
	}

	//without normalize
	void generateCosKernel(double* dest, double& totalInv, const int dctType, const double* Gk, const int radius, const int order)
	{
		double k0;
		double omega;
		switch (dctType)
		{
		case 1:	omega = CV_2PI / (2.0 * radius + 0.0); k0 = 0.0; break;
		case 3:	omega = CV_2PI / (2.0 * radius + 2.0); k0 = 0.5; break;
		case 5:	omega = CV_2PI / (2.0 * radius + 1.0); k0 = 0.0; break;
		case 7:	omega = CV_2PI / (2.0 * radius + 1.0); k0 = 0.5; break;
		default: throw "Unsupported DCT type"; break;
		}

		double sum = 0.0;
		if (dctType == 1 || dctType == 5)
		{
			sum = Gk[0] * double(2 * radius + 1);//k=0
			for (int r = 0; r <= radius; ++r)
			{
				for (int k = 1; k <= order; ++k)
				{
					const double coeff = cos((k + k0) * omega * r) * Gk[k];
					dest[order * r + k - 1] = coeff;

					if (r == 0) sum += coeff;
					else sum += 2.0 * coeff;
				}
			}
		}
		else
		{
			for (int r = 0; r <= radius; ++r)
			{
				for (int k = 0; k <= order; ++k)
				{
					const double coeff = cos((k + k0) * omega * r) * Gk[k];

#ifdef COEFFICIENTS_SMALLEST_FIRST
					dest[(order + 1) * r + (order)-k] = coeff;
#else
					dest[(order + 1) * r + k] = coeff;
#endif

					if (r == 0) sum += coeff;
					else sum += 2.0 * coeff;
				}
			}
		}

		totalInv = (1.0 / sum);
	}

	bool optimizeSpectrum(const double sigma, const int K, const int R, const int dcttype, double* destSpect, const int M)
	{
		/*if (K > R)
		{
			computeSpectrumGaussianClosedForm(sigma, K, R, dcttype, destSpect);
			return false;
		}*/

		double omega, ratio, k0, n0 = 0.0;
		switch (dcttype)
		{
		case 1:	omega = CV_2PI / (2.0 * R + 0.0); ratio = 2.0 / sqrt(2.0 * R + 0.0); k0 = 0.0; break;
		case 3:	omega = CV_2PI / (2.0 * R + 2.0); ratio = 2.0 / sqrt(2.0 * R + 2.0); k0 = 0.5; break;
		case 5:	omega = CV_2PI / (2.0 * R + 1.0); ratio = 2.0 / sqrt(2.0 * R + 1.0); k0 = 0.0; break;
		case 7:	omega = CV_2PI / (2.0 * R + 1.0); ratio = 2.0 / sqrt(2.0 * R + 1.0); k0 = 0.5; break;
		default: throw "Unsupported DCT type"; break;
		}

		//kernel
		cv::Mat1d h0(R + 1, 1);//R+1
		setGaussKernelHalf(h0, R, sigma, false);

		//weight matrix
		cv::Mat1d W = cv::Mat1d::eye(R + 1, R + 1);
		W(0, 0) = 0.5;

		// DCT matrix
		cv::Mat1d C(R + 1, K + 1);
		for (int n = 0; n <= R; ++n)
		{
			for (int k = 0; k <= K; ++k)
			{
				C(n, k) = cos(omega * (k + k0) * (n + n0));
			}
		}
		C *= ratio;

		cv::Mat1d CWCinv;//K+1 x K+1
		cv::Mat1d a_ls;

#ifdef USE_OPTIMIZE_DCT_SWICH
		static int method = 1;
#ifdef USE_EIGEN
		createTrackbar("opt method", "", &method, 4);
#else
		createTrackbar("opt method", "", &method, 2);
#endif
#else 
		int method = 0;//OpenCV direct
		//int method = 1;//fast method ICASSP2018 universal(but DCT1 and 7 has some bugs?)
		//int method = 2;//OpenCV SVD
		//int method = 3;//Eigen direct
		//int method = 4;//Eigen SVD
#endif
		//print_debug3(K, R, dcttype);
		switch (method)
		{
		default:
		case 0:
		{
			// (CWC)^-1
			CWCinv = (C.t() * W * C).inv(DECOMP_CHOLESKY);
			//cv::Mat1d CWCinv = (C.t() * W * C).inv(DECOMP_LU);
			// L2 minimization with moment preservation
			a_ls = CWCinv * C.t() * W * h0;
			//showMat64F(CWCinv, true);

			break;
		}
		case 1:
		{
			computeCtWCinv(CWCinv, K + 1, R + 1, dcttype);
			//showMat64F(CWCinv, true);
			a_ls = CWCinv * C.t() * W * h0;

			break;
		}
		case 2:
		{
			Mat w, u, vt;
			W(0, 0) = sqrt(0.5);
			//with weight W
			cv::SVDecomp(W * C, w, u, vt, SVD::FULL_UV);
			SVD::backSubst(w, u, vt, W * h0, a_ls);

			//without weight W
			//cv::SVDecomp(C, w, u, vt, SVD::FULL_UV);
			//SVD::backSubst(w, u, vt, h0, a_ls);
			break;
		}
		case 3:
		{
#ifdef USE_EIGEN
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CE;
			cv::cv2eigen(C, CE);
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> WE;
			cv::cv2eigen(W, WE);
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h0E;
			cv::cv2eigen(h0, h0E);

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CtE;
			cv::cv2eigen(C.t(), CtE);

			Eigen::MatrixXd temp = CtE * WE * CE;
			Eigen::MatrixXd temp2 = temp.inverse() * CtE * WE * h0E;
			cv::eigen2cv(temp2, a_ls);
#endif
			break;
		}
		case 4:
		{

#ifdef USE_EIGEN
			W(0, 0) = sqrt(0.5);
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CE;
			cv::cv2eigen(C, CE);
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> WE;
			cv::cv2eigen(W, WE);
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h0E;
			cv::cv2eigen(h0, h0E);

			Eigen::BDCSVD<Eigen::MatrixXd> svd(WE * CE, Eigen::ComputeThinU | Eigen::ComputeThinV);
			Eigen::MatrixXd a = svd.solve(WE * h0E);
			cv::eigen2cv(a, a_ls);
#endif
			break;
		}
		}

		const double sp0 = (dcttype == 1 || dcttype == 5) ? 1.0 : 2.0;
		//const double v = sp0 / a_ls(0, 0);

#ifdef PRINT_RANK_DEFICIENT
		if (a_ls(0, 0) == 0)
		{
			cout << "PRINT_RANK_DEFICIENT" << endl;
			print_debug4(sigma, K, R, dcttype);
		}
#endif
		if (a_ls(0, 0) == 0)
		{
			computeSpectrumGaussianClosedForm(sigma, K, R, dcttype, destSpect);
			return false;
		}

		const double v = sp0 / a_ls(0, 0);

		for (int k = 0; k <= K; ++k)
		{
			//cout << k << ": " << spect[k] << ": ";
			destSpect[k] = a_ls(k, 0) * v;
			//destSpect[k] = a_ls(k, 0)*ratio;
			//cout << spect[k] << endl;		
		}

		if (1 <= M)
		{
			cv::Mat1d a = a_ls.clone();

			//ideal gaussian moment vector
			cv::Mat1d mu(M, 1, 1.0);
			for (auto m = 1; m < M; ++m)
				mu(m, 0) = mu(m - 1, 0) * (2 * m - 1) * sigma * sigma;

			// Vandermonde
			cv::Mat1d V(R + 1, M);
			for (auto m = 0; m < M; ++m)
			{
				for (auto u = 0; u <= R; ++u)
					V(u, m) = pow(u, 2 * m);
				//V(0, m) /= 2.0; // moved from W
			}

			// Moment constraint
			cv::Mat1d U = V.t() * W * C;
			cv::Mat1d S = U * CWCinv * U.t();
			cv::Mat1d Sinv = S.inv();
			a -= CWCinv * U.t() * Sinv * (U * a - 0.5 * mu);

			//print_matrix(U, "[U]");
			//print_matrix(S, "[S]");
			//print_matrix(Sinv, "[S^-1]");
			//print_matrix(U*a - 0.5*mu, "[U*a - 0.5*mu]");

			for (auto k = 0; k <= K; ++k)
			{
				//cout <<k<<": "<< spect[k] << ": ";
				destSpect[k] = a(k, 0);
				//cout << spect[k] << endl;
			}

			if (dcttype == 1 || dcttype == 5)
				destSpect[0] = 1.0;
		}

		//std::cerr << "[Spectra]" << std::endl;
		//for (auto k = 0; k < K; ++k)
		//	std::cerr << cv::format("k=%d:  %10.8f  %10.8f", k, a_ls(k, 0), a(k, 0)) << std::endl;
		return true;
	}

	class SearchFullDCTRadius :public cp::Search1DInt
	{
		double sigma;
		int K;
		int dcttype;
		double* spect;

		bool isOptimize;

		double getErrorDCT(const double sigma, const int K, const int R, const int dcttype, const double* spect)
		{
			const int ROut = (int)ceil(9.0 * sigma);
			//const int ROut = (int)ceil(7.5 * sigma);
			//const int ROut = (int)ceil(6.0 * sigma);
			AutoBuffer<double> ans(ROut + 1);
			AutoBuffer<double> approx(ROut + 1);
			AutoBuffer<double> e2(ROut + 1);

			double sum = 1.0;
			ans[0] = 1.0;
			for (int i = ROut; i >= 1; --i)
			{
				double v = exp(i * i / (-2.0 * sigma * sigma));
				ans[i] = v;
				sum += 2.0 * v;
			}
			for (int i = 0; i <= ROut; i++)
			{
				ans[i] /= sum;
			}

			double errorTruncation = 0.0;
			for (int i = ROut; i >= R + 1; --i)
			{
				double e = ans[i];
				errorTruncation = fma(e, e, errorTruncation);
			}

			double phi, k0;//n0=0
			switch (dcttype)
			{
			case 1:	phi = CV_2PI / (2.0 * R + 0.0); k0 = 0.0; break;
			case 3:	phi = CV_2PI / (2.0 * R + 2.0); k0 = 0.5; break;
			case 5:	phi = CV_2PI / (2.0 * R + 1.0); k0 = 0.0; break;
			case 7:	phi = CV_2PI / (2.0 * R + 1.0); k0 = 0.5; break;

			default: throw "Unsupported DCT type"; break;
			}

			sum = 0.0;
			for (int i = R; i >= 1; --i)
			{
				double s = 0.0;
				for (int k = K; k >= 0; --k)
				{
					s = fma(cos(phi * (k + k0) * (i)), spect[k], s);//DCT1,3,5,7: n0 = 0.
				}
				sum += 2.0 * s;
				approx[i] = s;
			}
			{
				double s = 0.0;
				for (int k = K; k >= 0; --k)
				{
					s += spect[k];
				}
				sum += s;
				approx[0] = s;
			}
			for (int i = 0; i <= R; i++)
			{
				approx[i] /= sum;
			}

			double errorFrec = 0.0;
			for (int i = R; i >= 1; --i)
			{
				errorFrec = fma((ans[i] - approx[i]), (ans[i] - approx[i]), errorFrec);
			}
			errorFrec *= 2.0;
			errorFrec += (ans[0] - approx[0]) * (ans[0] - approx[0]);

			return errorFrec + errorTruncation;
		}

		double getError(int x)
		{
			if (isOptimize)optimizeSpectrum(sigma, K, x, dcttype, spect);
			else computeSpectrumGaussianClosedForm(sigma, K, x, dcttype, spect);

			return getErrorDCT(sigma, K, x, dcttype, spect);
		}

	public:
		SearchFullDCTRadius(const double sigma, const int K, const int dcttype, const double* spect, bool isOptimize)
		{
			this->sigma = sigma;
			this->K = K;
			this->dcttype = dcttype;
			this->spect = (double*)spect;
			this->isOptimize = isOptimize;
		}
	};

	int test_ratio(const double sigma, const int K, const int dcttype)
	{
		double a, b, c_1, c_2, c_3, c_4;
		int dest;
		if (dcttype == 1)
			b = 0.467347, c_1 = 0.0007, c_2 = -0.0277, c_3 = 0.6053, c_4 = 1.8088;
		else if (dcttype == 3)
			b = -0.16509, c_1 = 0.0005, c_2 = -0.0221, c_3 = 0.5458, c_4 = 2.1403;
		else if (dcttype == 5)
			b = 0.00696, c_1 = 0.0008, c_2 = -0.0289, c_3 = 0.6159, c_4 = 1.7929;
		else if (dcttype == 7)
			b = -0.029, c_1 = 0.0005, c_2 = -0.0217, c_3 = 0.5412, c_4 = 2.1912;
		else
		{
			cout << "This is not Sliding DCT" << endl;
			return 0;
		}
		a = c_1 * K * K * K + c_2 * K * K + c_3 * K + c_4;
		return dest = int(a * sigma + b);
	}

	int argminR_BruteForce_DCT(const double sigma, const int K, const int dcttype, const double* spect, const bool isOptimize, const bool isGoldenSelectionSearch)
	{
		//case K<=R:OK
		//case K>R:NG
		int r = argminR_ContinuousForm_DCT(sigma, K, dcttype, isGoldenSelectionSearch);
		
		const int rmin = max(1, (int)floor(0.6 * r));
		const int rmax = (int)ceil(1.4 * r);
		int argmin_r = 0;

		SearchFullDCTRadius s(sigma, K, dcttype, spect, isOptimize);

		//		s.plotError("test", rmin, rmax);
				//debug
				/*{
					int r_gs = s.goldenSearch(rmin, rmax);
					int r_li = s.linearSearch(rmin, rmax);
					if (r_li != r_gs)
					{
						cout << "NG: (sigma, k) = " << sigma << "," << K << endl;
						print_debug2(r_li, r_gs);
						print_debug2(rmin, rmax);
						s.plotError("error", rmin, rmax);
					}
					//cout << "dct" << dcttype << ": K=" << K << ", r=" << argmin_r << ", sigma=" << argmin_r / sigma << endl;
				}*/


		if (isGoldenSelectionSearch)
		{
			argmin_r = s.goldenSearch(rmin, rmax);
			//cout << "dct" << dcttype << ": K=" << K << ", r=" << argmin_r << ", sigma=" << argmin_r / sigma << endl;
		}
		else
		{
			argmin_r = s.linearSearch(rmin, rmax);
			//cout << "dct" << dcttype << ": K=" << K << ", r=" << argmin_r << ", sigma=" << argmin_r / sigma << endl;
		}

		//cout << "Linear search radius is" << test_ratio(sigma, K, dcttype) << endl;
		//cout << "True number radius is" << argmin_r << endl;
		return argmin_r;
	}

#pragma region ContinuousForm
	class SearchDCTRadiusContinuousForm :public cp::Search1DInt
	{
		double sigma;
		int K;
		double* spect = nullptr;

		double getError(int x)
		{
			double T = 2.0 * x + 1.0;
			double Es = erfc(T / (2.0 * sigma));
			double Ef = erfc(sigma * CV_PI * (2.0 * K + 1.0) / T);
			return Es + Ef;
		}

	public:
		SearchDCTRadiusContinuousForm(const double sigma, const int K)
		{
			this->sigma = sigma;
			this->K = K;
		}
	};

	int argminR_ContinuousForm_DCT(const double sigma, const int K, const int dcttype, const bool isGoldenSelectionSearch)
	{
		int argmin_r = 0;

		int rmin = max(K, (int)floor(1.0 * sigma));
		int rmax = (int)ceil(10.0 * sigma);
		if (rmin > rmax)return rmin;
		SearchDCTRadiusContinuousForm s(sigma, K);

		if (isGoldenSelectionSearch)
		{
			argmin_r = s.goldenSearch(rmin, rmax);
			//cout << "dct" << dcttype << ": K=" << K << ", r=" << argmin_r << ", sigma=" << argmin_r / sigma << endl;
		}
		else
		{
			argmin_r = s.linearSearch(rmin, rmax);
			//cout << "dct" << dcttype << ": K=" << K << ", r=" << argmin_r << ", sigma=" << argmin_r / sigma << endl;
		}
		return argmin_r;
	}
#pragma endregion
}