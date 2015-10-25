#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	template <class T>
	static void nnUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols);
		const int dh = dest.rows / (src.rows);

		Mat dst = Mat::zeros(Size(dw*src.cols, dh*src.rows), src.type());
		Mat sim; copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;
			T* s = sim.ptr<T>(j);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				for (int l = 0; l < dh; l++)
				{
					T* d = dst.ptr<T>(n + l);
					for (int k = 0; k < dw; k++)
					{
						if (k < hdw && l < hdh)
							d[m + k] = s[i];
						else if (k >= hdw&&l < hdh)
							d[m + k] = s[i + 1];
						else if (k < hdw&&l >= hdh)
							d[m + k] = s[i + sim.cols];
						else
							d[m + k] = s[i + 1 + sim.cols];
					}

				}
			}
		}
		copyMakeBorder(dst, dest, 0, dest.rows % (src.rows), 0, dest.cols % (src.cols), cv::BORDER_REPLICATE);
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

	inline int linearinterpolate(int lt, int rt, int lb, int rb, double a, double b)
	{
		return (int)((b*a*lt + b*(1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb) + 0.5);
	}

	template <class T>
	inline T linearinterpolate_(T lt, T rt, T lb, T rb, double a, double b)
	{
		return (T)((b*a*lt + b*(1.0 - a)*rt + (1.0 - b)*a*lb + (1.0 - b)*(1.0 - a)*rb) + 0.5);
	}

	template <class T>
	static void linearUpsample_(Mat& src, Mat& dest)
	{
		const int dw = dest.cols / (src.cols - 1);
		const int dh = dest.rows / (src.rows - 1);
		const int hdw = dw >> 1;
		const int hdh = dh >> 1;

		for (int j = 0; j < src.rows - 1; j++)
		{
			int n = j*dh;
			T* s = src.ptr<T>(j);

			for (int i = 0, m = 0; i < src.cols - 1; i++, m += dw)
			{
				const T ltd = s[i];
				const T rtd = s[i + 1];
				const T lbd = s[i + src.cols];
				const T rbd = s[i + 1 + src.cols];
				for (int l = 0; l < dh; l++)
				{
					double beta = 1.0 - (double)l / dh;
					T* d = dest.ptr<T>(n + l);
					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;
						d[m + k] = (T)linearinterpolate_<T>(ltd, rtd, lbd, rbd, alpha, beta);
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
			int n = j*dh;
			uchar* d = dest.ptr<uchar>(n);
			for (int i = 0, m = 0; i < src.size().width; i++, m += dw)
			{
				d[m] = 255;
			}
		}
	}

	template<class T>
	inline T weightedinterpolation_(T lt, T rt, T lb, T rb, double wlt, double wrt, double wlb, double  wrb)
	{
		return (T)((wlt*lt + wrt*rt + wlb*lb + wrb*rb) / (wlt + wrt + wlb + wrb) + 0.5);
	}

	template<class T>
	inline T weightedlinearinterpolate_(T lt, T rt, T lb, T rb, double wlt, double wrt, double wlb, double  wrb, double a, double b)
	{
		return (T)((b*a*wlt*lt + b*(1.0 - a)*wrt*rt + (1.0 - b)*a*wlb*lb + (1.0 - b)*(1.0 - a)*wrb*rb) / (b*a*wlt + b*(1.0 - a)*wrt + (1.0 - b)*a*wlb + (1.0 - b)*(1.0 - a)*wrb) + 0.5);
		//return (int)((b*a*lt + b*(1.0-a)*rt + (1.0-b)*a*lb + (1.0-b)*(1.0-a)*rb)+0.5);
	}

	//NAFDU:
	template<class T>
	static void noiseAwareFilterDepthUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_d, double sigma_s, double eps, double tau)
	{
		Mat maxf;
		Mat minf;
		maxFilter(src, maxf, Size(3, 3));
		minFilter(src, minf, Size(3, 3));
		Mat diff = (maxf - minf);
		Mat ups;

		resize(src, ups, joint.size(), 0, 0, INTER_LINEAR);
		warpShift(ups, ups, -4, -4);

		if (dest.empty())dest.create(joint.size(), src.type());
		Mat sim, jim, eim, uim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		copyMakeBorder(ups, uim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut_sig[256];
		for (int i = 0; i < 256; i++)
		{
			lut_sig[i] = 1.0 / (1.0 + (double)std::exp(-eps*(i - tau)));
		}

		double lut_d[256];
		double gauss_d_coeff = -0.5 / (sigma_d*sigma_d);
		for (int i = 0; i < 256; i++)
		{
			lut_d[i] = (double)std::exp(i*i*gauss_d_coeff);
		}

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}
		vector<double> lut_(dw*dh);
		double* lut_s = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw*dh; i++)
			{
				lut_s[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
			for (int i = 0; i < dw*dh; i++)
			{
				lut_s[i] = (double)std::exp(i*i*gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			T* s = sim.ptr<T>(j);
			T* minmaxdiff = diff.ptr<T>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];


				const T ltd = s[i];
				const T rtd = s[i + 1];
				const T lbd = s[i + sim.cols];
				const T rbd = s[i + 1 + sim.cols];

				double a = lut_sig[(int)minmaxdiff[i]];
				double ia = 1.0 - a;

				for (int l = 0; l < dh; l++)
				{
					T* d = dest.ptr<T>(n + l);
					uchar* jnt = jim.ptr<uchar>(n + l);
					T* ud = uim.ptr<T>(n + l);
					//double beta = 1.0-(double)l/dh;

					for (int k = 0; k < dw; k++)
					{
						//double alpha = 1.0-(double)k/dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];
						const int z = (int)ud[(m + k)];

						double wlt = lut_s[k + l] * (a*lut[abs(ltr - r) + abs(ltg - g) + abs(ltb - b)] + ia*lut_d[abs(z - (int)ltd)]);
						double wrt = lut_s[dw - k + l] * (a*lut[abs(rtr - r) + abs(rtg - g) + abs(rtb - b)] + ia*lut_d[abs(z - (int)rtd)]);
						double wlb = lut_s[k + dh - l] * (a*lut[abs(lbr - r) + abs(lbg - g) + abs(lbb - b)] + ia*lut_d[abs(z - (int)lbd)]);
						double wrb = lut_s[dw - k + dh - l] * (a*lut[abs(rbr - r) + abs(rbg - g) + abs(rbb - b)] + ia*lut_d[abs(z - (int)rbd)]);


						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = weightedinterpolation_<T>(ltd, rtd, lbd, rbd, wlt, wrt, wlb, wrb);
					}
				}
			}
		}
	}

	void noiseAwareFilterDepthUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c, double sigma_d, double sigma_s, double eps, double tau)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U) noiseAwareFilterDepthUpsample_<uchar>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_16S) noiseAwareFilterDepthUpsample_<short>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_16U) noiseAwareFilterDepthUpsample_<ushort>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_32S) noiseAwareFilterDepthUpsample_<int>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_32F) noiseAwareFilterDepthUpsample_<float>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
		else if (src.depth() == CV_64F) noiseAwareFilterDepthUpsample_<double>(src, joint, dest, sigma_c, sigma_d, sigma_s, eps, tau);
	}

	template<class T>
	static void jointBilateralLinearUpsample_(Mat& src, const Mat& joint, Mat& dest, double sigma_c)
	{
		if (dest.empty())dest.create(joint.size(), src.type());
		Mat sim, jim, eim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			T* s = sim.ptr<T>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];

				const T ltd = s[i];
				const T rtd = s[i + 1];
				const T lbd = s[i + sim.cols];
				const T rbd = s[i + 1 + sim.cols];

				for (int l = 0; l < dh; l++)
				{
					T* d = dest.ptr<T>(n + l);
					uchar* jnt = jim.ptr(n + l);
					double beta = 1.0 - (double)l / dh;

					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];

						int minE = abs(ltr - r) + abs(ltg - g) + abs(ltb - b);
						//T mind=ltd;
						//int minp=0;

						double wlt = lut[minE];

						int e = abs(rtr - r) + abs(rtg - g) + abs(rtb - b);
						double wrt = lut[e];

						if (minE > e)
						{
							minE = e;
							//mind = rtd;
							//minp=1;
						}

						e = abs(lbr - r) + abs(lbg - g) + abs(lbb - b);
						double wlb = lut[e];

						if (minE > e)
						{
							minE = e;
							//mind = lbd;
							//minp=2;
						}

						e = abs(rbr - r) + abs(rbg - g) + abs(rbb - b);
						double wrb = lut[e];

						if (minE > e)
						{
							minE = e;
							//mind = rbd;
							//minp=3;
						}

						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = weightedlinearinterpolate_<T>(ltd, rtd, lbd, rbd, wlt, wrt, wlb, wrb, alpha, beta);
					}
				}
			}
		}
	}

	void jointBilateralLinearUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U)
			jointBilateralLinearUpsample_<uchar>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_16S)
			jointBilateralLinearUpsample_<short>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_16U)
			jointBilateralLinearUpsample_<ushort>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_32F)
			jointBilateralLinearUpsample_<float>(src, joint, dest, sigma_c);
		else if (src.depth() == CV_64F)
			jointBilateralLinearUpsample_<double>(src, joint, dest, sigma_c);
	}


	template<class T>
	static void jointBilateralUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_s)
	{
		if (dest.empty())dest.create(joint.size(), src.type());
		Mat sim, jim, eim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}
		vector<double> lut_(dw*dh);
		double* lut2 = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			T* s = sim.ptr<T>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];

				const T ltd = s[i];
				const T rtd = s[i + 1];
				const T lbd = s[i + sim.cols];
				const T rbd = s[i + 1 + sim.cols];

				for (int l = 0; l < dh; l++)
				{
					T* d = dest.ptr<T>(n + l);
					uchar* jnt = jim.ptr(n + l);
					//double beta = 1.0-(double)l/dh;

					for (int k = 0; k < dw; k++)
					{
						//double alpha = 1.0-(double)k/dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];



						double wlt = lut2[k + l] * lut[abs(ltr - r) + abs(ltg - g) + abs(ltb - b)];
						double wrt = lut2[dw - k + l] * lut[abs(rtr - r) + abs(rtg - g) + abs(rtb - b)];
						double wlb = lut2[k + dh - l] * lut[abs(lbr - r) + abs(lbg - g) + abs(lbb - b)];
						double wrb = lut2[dw - k + dh - l] * lut[abs(rbr - r) + abs(rbg - g) + abs(rbb - b)];



						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = weightedinterpolation_<T>(ltd, rtd, lbd, rbd, wlt, wrt, wlb, wrb);
					}
				}
			}
		}
	}

	void jointBilateralUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c, double sigma_s)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U)
			jointBilateralUpsample_<uchar>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16S)
			jointBilateralUpsample_<short>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16U)
			jointBilateralUpsample_<ushort>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_32F)
			jointBilateralUpsample_<float>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_64F)
			jointBilateralUpsample_<double>(src, joint, dest, sigma_c, sigma_s);
	}


	template<class T>
	static void jointBilateralNNUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_s)
	{
		Mat sim, jim, eim;
		copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
		const int dw = (joint.cols) / (src.cols);
		const int dh = (joint.rows) / (src.rows);

		copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

		double lut[256 * 3];
		double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
		for (int i = 0; i < 256 * 3; i++)
		{
			lut[i] = (double)std::exp(i*i*gauss_c_coeff);
		}
		vector<double> lut_(dw*dh);
		double* lut2 = &lut_[0];
		if (sigma_s <= 0.0)
		{
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = 1.0;
			}
		}
		else
		{
			double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
			for (int i = 0; i < dw*dh; i++)
			{
				lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
			}
		}

		for (int j = 0; j < src.rows; j++)
		{
			int n = j*dh;

			T* s = sim.ptr<T>(j);
			uchar* jnt_ = jim.ptr(n);

			for (int i = 0, m = 0; i < src.cols; i++, m += dw)
			{
				const uchar ltr = jnt_[3 * m + 0];
				const uchar ltg = jnt_[3 * m + 1];
				const uchar ltb = jnt_[3 * m + 2];

				const uchar rtr = jnt_[3 * (m + dw) + 0];
				const uchar rtg = jnt_[3 * (m + dw) + 1];
				const uchar rtb = jnt_[3 * (m + dw) + 2];

				const uchar lbr = jnt_[3 * (m + jim.cols*dh) + 0];
				const uchar lbg = jnt_[3 * (m + jim.cols*dh) + 1];
				const uchar lbb = jnt_[3 * (m + jim.cols*dh) + 2];

				const uchar rbr = jnt_[3 * (m + dw + jim.cols*dh) + 0];
				const uchar rbg = jnt_[3 * (m + dw + jim.cols*dh) + 1];
				const uchar rbb = jnt_[3 * (m + dw + jim.cols*dh) + 2];

				const T ltd = s[i];
				const T rtd = s[i + 1];
				const T lbd = s[i + sim.cols];
				const T rbd = s[i + 1 + sim.cols];

				for (int l = 0; l < dh; l++)
				{
					T* d = dest.ptr<T>(n + l);
					uchar* jnt = jim.ptr(n + l);
					double beta = 1.0 - (double)l / dh;

					for (int k = 0; k < dw; k++)
					{
						double alpha = 1.0 - (double)k / dw;

						const uchar r = jnt[3 * (m + k) + 0];
						const uchar g = jnt[3 * (m + k) + 1];
						const uchar b = jnt[3 * (m + k) + 2];

						int minE = abs(ltr - r) + abs(ltg - g) + abs(ltb - b);
						T mind = ltd;
						//int minp=0;

						double wlt = lut2[k + l] * lut[minE];
						double maxE = wlt;

						int e = abs(rtr - r) + abs(rtg - g) + abs(rtb - b);
						double wrt = lut2[dw - k + l] * lut[e];

						if (maxE < wrt)
						{
							maxE = wrt;
							mind = rtd;
							//minp=1;
						}

						e = abs(lbr - r) + abs(lbg - g) + abs(lbb - b);
						double wlb = lut2[k + dh - l] * lut[e];

						if (maxE < wlb)
						{
							maxE = wlb;
							mind = lbd;
							//minp=2;
						}

						e = abs(rbr - r) + abs(rbg - g) + abs(rbb - b);
						double wrb = lut2[dw - k + dh - l] * lut[e];

						if (maxE < wrb)
						{
							maxE = wrb;
							mind = rbd;
							//minp=3;
						}

						/*wlt = (abs(mind-ltd)<dsubth2)?wlt:0.00001;
						wrt = (abs(mind-rtd)<dsubth2)?wrt:0.00001;
						wlb = (abs(mind-lbd)<dsubth2)?wlb:0.00001;
						wrb = (abs(mind-rbd)<dsubth2)?wrb:0.00001;*/
						//if(wlt==0.0 && wlb==0.0 && wrt==0.0 && wrb==0.0)
						//	d[m+k] = weiterdlinearinterpolate(ltd,rtd,lbd,rbd,0.5,0.5,0.5,0.5,alpha,beta);
						///else
						d[m + k] = mind;//weightedlinearinterpolate_<T>(ltd,rtd,lbd,rbd,wlt,wrt,wlb,wrb,alpha,beta);
					}
				}
			}
		}
	}

	void jointBilateralNNUpsample(InputArray src_, InputArray joint_, OutputArray dst, double sigma_c, double sigma_s)
	{
		if (dst.empty() || dst.type() != src_.type() || joint_.size() != dst.size()) dst.create(joint_.size(), src_.type());
		Mat src = src_.getMat();
		Mat joint = joint_.getMat();
		Mat dest = dst.getMat();

		if (src.depth() == CV_8U)
			jointBilateralNNUpsample_<uchar>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16S)
			jointBilateralNNUpsample_<short>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_16U)
			jointBilateralNNUpsample_<ushort>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_32F)
			jointBilateralNNUpsample_<float>(src, joint, dest, sigma_c, sigma_s);
		else if (src.depth() == CV_64F)
			jointBilateralNNUpsample_<double>(src, joint, dest, sigma_c, sigma_s);
	}

}