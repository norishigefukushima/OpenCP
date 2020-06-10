#include "crossBasedLocalFilter.hpp"
#include "blend.hpp"
#include "timer.hpp"
using namespace std;
using namespace cv;

namespace cp
{

	template <class T>
	void orthogonalIntegralImageFilterF(Mat& src, Mat& dest, CrossBasedLocalFilter::cross* crossdata)
	{
		T* s = src.ptr<T>(0);
		T* d = dest.ptr<T>(0);
		CrossBasedLocalFilter::cross* cd = crossdata;

		double* integral;
		const int step = src.cols * src.channels();
		if (src.channels() == 1)
		{
			integral = new double[max(src.cols, src.rows) + 1];
			double* IH;
			int i, j;
			T* ds = dest.ptr<T>(0);
			CrossBasedLocalFilter::cross* cds = crossdata;

			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				IH = integral;
				for (i = src.cols; i--;)
				{
					IH[1] = *IH + *s;
					s++, IH++;
				}
				IH = integral;
				for (i = src.cols; i--;)
				{
					*d = (T)((IH[1 + cd->hp] - IH[-cd->hm]) * cd->divh);
					d++, cd++, IH++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				T* dd = ds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					IH[1] = *IH + *dd;
					dd += step, IH++;
				}

				dd = ds;
				cd = cds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					*dd = (T)((IH[1 + cd->vp] - IH[-cd->vm]) * cd->divv);
					IH++, dd += step, cd += dest.cols;
				}
				ds++;
				cds++;
			}
		}
		else if (src.channels() == 3)
		{
			integral = new double[(max(src.cols, src.rows) + 1) * 3];
			double* IH;
			int i, j;
			T* ds = dest.ptr<T>(0);
			CrossBasedLocalFilter::cross* cds = crossdata;

			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				IH = integral;
				for (i = src.cols; i--;)
				{
					IH[3] = IH[0] + s[0];
					IH[4] = IH[1] + s[1];
					IH[5] = IH[2] + s[2];
					IH += 3, s += 3;
				}
				IH = integral;
				for (i = src.cols; i--;)
				{
					d[0] = (T)((IH[3 + cd->hp * 3] - IH[-cd->hm * 3]) * cd->divh);
					d[1] = (T)((IH[3 + cd->hp * 3 + 1] - IH[-cd->hm * 3 + 1]) * cd->divh);
					d[2] = (T)((IH[3 + cd->hp * 3 + 2] - IH[-cd->hm * 3 + 2]) * cd->divh);
					d += 3, IH += 3, cd++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				T* dd = ds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					IH[3] = IH[0] + dd[0];
					IH[4] = IH[1] + dd[1];
					IH[5] = IH[2] + dd[2];
					dd += step, IH += 3;
				}

				dd = ds;
				cd = cds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					dd[0] = (T)((IH[3 + cd->vp * 3] - IH[-cd->vm * 3]) * cd->divv);
					dd[1] = (T)((IH[3 + cd->vp * 3 + 1] - IH[-cd->vm * 3 + 1]) * cd->divv);
					dd[2] = (T)((IH[3 + cd->vp * 3 + 2] - IH[-cd->vm * 3 + 2]) * cd->divv);
					IH += 3;
					dd += step, cd += dest.cols;
				}
				ds += 3;
				cds++;
			}

		}
		delete[] integral;
	}

	template <class T>
	void orthogonalIntegralImageFilterF(Mat& src, Mat& weight, Mat& dest, CrossBasedLocalFilter::cross* crossdata)
	{
		T* s = src.ptr<T>(0);
		T* w = weight.ptr<T>(0);
		T* ws = w;
		T* d = dest.ptr<T>(0);
		T* ds = d;
		CrossBasedLocalFilter::cross* cd = crossdata;

		double* integral;
		double* integralW;
		const int step = src.cols * src.channels();
		const int stepW = weight.cols;
		if (src.channels() == 1)
		{
			integral = new double[max(src.cols, src.rows) + 1];
			integralW = new double[max(src.cols, src.rows) + 1];
			double* IH;
			double* IHW;
			int i, j;
			CrossBasedLocalFilter::cross* cds = crossdata;
			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				integralW[0] = 0;
				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					IH[1] = *IH + (*s * *w);
					IHW[1] = *IHW + *w;
					s++, w++, IH++, IHW++;
				}
				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					*d = (T)((IH[1 + cd->hp] - IH[-cd->hm]) / (T)(IHW[1 + cd->hp] - IHW[-cd->hm]));
					d++, cd++, IH++, IHW++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				integralW[0] = 0;
				IH = integral;
				IHW = integralW;
				T* dd = ds;
				T* ww = ws;
				for (j = src.rows; j--;)
				{
					IH[1] = *IH + (*dd * *ww);
					IHW[1] = *IHW + *ww;
					dd += step, ww += stepW, IH++, IHW++;
				}
				dd = ds;
				cd = cds;
				IH = integral;
				IHW = integralW;
				for (j = src.rows; j--;)
				{
					*dd = (T)((IH[1 + cd->vp] - IH[-cd->vm]) / (T)(IHW[1 + cd->vp] - IHW[-cd->vm]));
					IH++, IHW++, dd += step, cd += dest.cols;
				}
				ws++;
				ds++;
				cds++;
			}
		}
		else if (src.channels() == 3)
		{
			integral = new double[(max(src.cols, src.rows) + 1) * 3];
			integralW = new double[(max(src.cols, src.rows) + 1)];
			double* IH;
			double* IHW;
			int i, j;
			CrossBasedLocalFilter::cross* cds = crossdata;
			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				integralW[0] = 0;

				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					IH[3] = IH[0] + (s[0] * *w);
					IH[4] = IH[1] + (s[1] * *w);
					IH[5] = IH[2] + (s[2] * *w);
					IHW[1] = *IHW + *w;
					IH += 3, IHW++, s += 3, w++;
				}
				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					T div = 1.0f / (T)(IHW[1 + cd->hp] - IHW[-cd->hm]);
					d[0] = (T)((IH[3 + cd->hp * 3] - IH[-cd->hm * 3]) * div);
					d[1] = (T)((IH[3 + cd->hp * 3 + 1] - IH[-cd->hm * 3 + 1]) * div);
					d[2] = (T)((IH[3 + cd->hp * 3 + 2] - IH[-cd->hm * 3 + 2]) * div);
					d += 3, IH += 3, IHW++, cd++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				integralW[0] = 0;
				T* dd = ds;
				T* ww = ws;
				IH = integral;
				IHW = integralW;
				for (j = src.rows; j--;)
				{
					IH[3] = IH[0] + dd[0] * *ww;
					IH[4] = IH[1] + dd[1] * *ww;
					IH[5] = IH[2] + dd[2] * *ww;
					IHW[1] = *IHW + *ww;
					dd += step, ww += stepW, IH += 3, IHW++;
				}

				dd = ds;
				cd = cds;
				IH = integral;
				IHW = integralW;
				for (j = src.rows; j--;)
				{
					T div = 1.0f / (T)(IHW[1 + cd->vp] - IHW[-cd->vm]);
					dd[0] = (T)((IH[3 + cd->vp * 3] - IH[-cd->vm * 3]) * div);
					dd[1] = (T)((IH[3 + cd->vp * 3 + 1] - IH[-cd->vm * 3 + 1]) * div);
					dd[2] = (T)((IH[3 + cd->vp * 3 + 2] - IH[-cd->vm * 3 + 2]) * div);
					IH += 3, IHW++;
					dd += step, cd += dest.cols;
				}
				ws++;
				ds += 3;
				cds++;
			}
		}
		delete[] integral;
		delete[] integralW;
	}

	template <class T>
	void orthogonalIntegralImageFilterI(Mat& src, Mat& dest, CrossBasedLocalFilter::cross* crossdata)
	{
		T* s = src.ptr<T>(0);
		T* d = dest.ptr<T>(0);
		Mat vfilter = Mat::zeros(src.size(), CV_MAKE_TYPE(CV_32F, src.channels()));
		float* v1 = vfilter.ptr<float>(0);
		CrossBasedLocalFilter::cross* cd = crossdata;

		float* integral;
		const int step = src.cols * src.channels();
		if (src.channels() == 1)
		{
			integral = new float[max(src.cols, src.rows) + 1];
			float* IH;
			int i, j;
			CrossBasedLocalFilter::cross* cds = crossdata;

			for (j = src.rows; j--;)
			{
				integral[0] = 0.f;
				IH = integral;
				for (i = src.cols; i--;)
				{
					IH[1] = *IH + (float)*s;
					s++, IH++;
				}
				IH = integral;
				for (i = src.cols; i--;)
				{
					*v1 = (float)((IH[1 + (int)cd->hp] - IH[-(int)cd->hm]) * cd->divh);
					v1++, cd++, IH++;
				}
			}
			//v
			float* vs = vfilter.ptr<float>(0);
			T* ds = dest.ptr<T>(0);
			for (i = src.cols; i--;)
			{
				v1 = vs;
				integral[0] = 0.f;
				IH = integral;
				for (j = src.rows; j--;)
				{
					IH[1] = *IH + *v1;
					v1 += step, IH++;
				}

				T* dd = ds;
				cd = cds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					*dd = (T)((IH[1 + cd->vp] - IH[-cd->vm]) * cd->divv + 0.5f);
					IH++, dd += step, cd += dest.cols;
				}
				ds++;
				cds++;
				vs++;
			}
		}
		else if (src.channels() == 3)//!under constrution: summing table convertion is not implimented uchar 2 float is not implimented
		{
			integral = new float[(max(src.cols, src.rows) + 1) * 3];
			float* IH;
			int i, j;
			T* ds = dest.ptr<T>(0);
			CrossBasedLocalFilter::cross* cds = crossdata;

			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				IH = integral;
				for (i = src.cols; i--;)
				{
					IH[3] = IH[0] + s[0];
					IH[4] = IH[1] + s[1];
					IH[5] = IH[2] + s[2];
					IH += 3, s += 3;
				}
				IH = integral;
				for (i = src.cols; i--;)
				{
					d[0] = (T)((IH[3 + cd->hp * 3] - IH[-cd->hm * 3]) * cd->divh + 0.5f);
					d[1] = (T)((IH[3 + cd->hp * 3 + 1] - IH[-cd->hm * 3 + 1]) * cd->divh + 0.5f);
					d[2] = (T)((IH[3 + cd->hp * 3 + 2] - IH[-cd->hm * 3 + 2]) * cd->divh + 0.5f);
					d += 3, IH += 3, cd++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				T* dd = ds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					IH[3] = IH[0] + dd[0];
					IH[4] = IH[1] + dd[1];
					IH[5] = IH[2] + dd[2];
					dd += step, IH += 3;
				}

				dd = ds;
				cd = cds;
				IH = integral;
				for (j = src.rows; j--;)
				{
					dd[0] = (T)((IH[3 + cd->vp * 3] - IH[-cd->vm * 3]) * cd->divv + 0.5f);
					dd[1] = (T)((IH[3 + cd->vp * 3 + 1] - IH[-cd->vm * 3 + 1]) * cd->divv + 0.5f);
					dd[2] = (T)((IH[3 + cd->vp * 3 + 2] - IH[-cd->vm * 3 + 2]) * cd->divv + 0.5f);
					IH += 3;
					dd += step, cd += dest.cols;
				}
				ds += 3;
				cds++;
			}
		}
		delete[] integral;
	}

	template <class T>
	void orthogonalIntegralImageFilterI(Mat& src, Mat& weight, Mat& dest, CrossBasedLocalFilter::cross* cd)
	{
		T* s = src.ptr<T>(0);
		T* w = weight.ptr<T>(0);
		T* ws = w;
		T* d = dest.ptr<T>(0);
		T* ds = d;
		CrossBasedLocalFilter::cross* crossdata = cd;

		unsigned int* integral = 0;
		unsigned int* integralW = 0;
		const int step = src.cols * src.channels();
		const int stepW = weight.cols;
		if (src.channels() == 1)
		{
			integral = new unsigned int[max(src.cols, src.rows) + 1];
			integralW = new unsigned int[max(src.cols, src.rows) + 1];
			unsigned int* IH;
			unsigned int* IHW;
			int i, j;
			CrossBasedLocalFilter::cross* cds = crossdata;
			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				integralW[0] = 0;
				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					IH[1] = *IH + (*s * *w);
					IHW[1] = *IHW + *w;
					s++, w++, IH++, IHW++;
				}
				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					*d = (T)((IH[1 + cd->hp] - IH[-cd->hm]) / (float)(IHW[1 + cd->hp] - IHW[-cd->hm]) + 0.5f);
					d++, cd++, IH++, IHW++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				integralW[0] = 0;
				IH = integral;
				IHW = integralW;
				T* dd = ds;
				T* ww = ws;
				for (j = src.rows; j--;)
				{
					IH[1] = *IH + (*dd * *ww);
					IHW[1] = *IHW + *ww;
					dd += step, ww += stepW, IH++, IHW++;
				}
				dd = ds;
				cd = cds;
				IH = integral;
				IHW = integralW;
				for (j = src.rows; j--;)
				{
					*dd = (T)((IH[1 + cd->vp] - IH[-cd->vm]) / (float)(IHW[1 + cd->vp] - IHW[-cd->vm]) + 0.5f);
					IH++, IHW++, dd += step, cd += dest.cols;
				}
				ws++;
				ds++;
				cds++;
			}
		}
		else if (src.channels() == 3)
		{
			integral = new unsigned int[(max(src.cols, src.rows) + 1) * 3];
			integralW = new unsigned int[(max(src.cols, src.rows) + 1)];
			unsigned int* IH;
			unsigned int* IHW;
			int i, j;
			CrossBasedLocalFilter::cross* cds = crossdata;
			for (j = src.rows; j--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				integralW[0] = 0;

				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					IH[3] = IH[0] + (s[0] * *w);
					IH[4] = IH[1] + (s[1] * *w);
					IH[5] = IH[2] + (s[2] * *w);
					IHW[1] = *IHW + *w;
					IH += 3, IHW++, s += 3, w++;
				}
				IH = integral;
				IHW = integralW;
				for (i = src.cols; i--;)
				{
					float div = 1.0f / (float)(IHW[1 + cd->hp] - IHW[-cd->hm]);
					d[0] = (T)((IH[3 + cd->hp * 3] - IH[-cd->hm * 3]) * div + 0.5f);
					d[1] = (T)((IH[3 + cd->hp * 3 + 1] - IH[-cd->hm * 3 + 1]) * div + 0.5f);
					d[2] = (T)((IH[3 + cd->hp * 3 + 2] - IH[-cd->hm * 3 + 2]) * div + 0.5f);
					d += 3, IH += 3, IHW++, cd++;
				}
			}
			//v
			for (i = src.cols; i--;)
			{
				integral[0] = 0;
				integral[1] = 0;
				integral[2] = 0;
				integralW[0] = 0;
				T* dd = ds;
				T* ww = ws;
				IH = integral;
				IHW = integralW;
				for (j = src.rows; j--;)
				{
					IH[3] = IH[0] + dd[0] * *ww;
					IH[4] = IH[1] + dd[1] * *ww;
					IH[5] = IH[2] + dd[2] * *ww;
					IHW[1] = *IHW + *ww;
					dd += step, ww += stepW, IH += 3, IHW++;
				}

				dd = ds;
				cd = cds;
				IH = integral;
				IHW = integralW;
				for (j = src.rows; j--;)
				{
					float div = 1.0f / (float)(IHW[1 + cd->vp] - IHW[-cd->vm]);
					dd[0] = (T)((IH[3 + cd->vp * 3] - IH[-cd->vm * 3]) * div + 0.5f);
					dd[1] = (T)((IH[3 + cd->vp * 3 + 1] - IH[-cd->vm * 3 + 1]) * div + 0.5f);
					dd[2] = (T)((IH[3 + cd->vp * 3 + 2] - IH[-cd->vm * 3 + 2]) * div + 0.5f);
					IH += 3, IHW++;
					dd += step, cd += dest.cols;
				}
				ws++;
				ds += 3;
				cds++;
			}
		}
		delete[] integral;
		delete[] integralW;
	}

	CrossBasedLocalFilter::CrossBasedLocalFilter(Mat& guide, const int r_, const int thresh_)
	{
		makeKernel(guide, r_, thresh_);
	}

	CrossBasedLocalFilter::CrossBasedLocalFilter()
	{
		r = 1;
		thresh = 20;
	}

	CrossBasedLocalFilter::~CrossBasedLocalFilter()
	{
		delete[] crossdata;
	}

	void CrossBasedLocalFilter::setMinSearch(int val)
	{
		minSearch = val;
	}

	void CrossBasedLocalFilter::makeKernel(Mat& guide, const int r_, const int thresh_, const int method)
	{
		r = r_;
		thresh = thresh_;

		size = guide.size();
		if (crossdata == nullptr)
		{
			crossdata = new cross[guide.size().area()];
		}
		else
		{
			delete[]crossdata;
			crossdata = new cross[guide.size().area()];
		}
		const int hw = r;
		const int c = guide.channels();

		Mat gim; copyMakeBorder(guide, gim, r, r, r, r, cv::BORDER_CONSTANT, 0);
		uchar* g = gim.ptr<uchar>(r); g += r * c;
		const int gstep = gim.cols * c;
		cross* cdata = crossdata;
		const int cstep = guide.cols;
		int arm_threshold = c*thresh;

		int i, j;
		if (method == CrossBasedLocalFilter::CROSS_BASED_LOCAL_FILTER_ARM_BASIC)
		{
			if (c == 1)
			{
				for (j = 0; j < guide.rows; j++)
				{
					for (i = 0; i < guide.cols; i++, g++, cdata++)
					{
						const uchar v = g[0];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > arm_threshold)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n]) > arm_threshold)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > arm_threshold)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n * gstep]) > arm_threshold)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
					g += 2 * r;
				}
			}
			else if (c == 3)
			{
				for (j = 0; j < guide.rows; j++)
				{
					for (i = 0; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > arm_threshold)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > arm_threshold)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-gstep * n + 0]) + abs(gv - g[-gstep * n + 1]) + abs(bv - g[-gstep * n + 2]) > arm_threshold)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[gstep * n + 0]) + abs(gv - g[gstep * n + 1]) + abs(bv - g[gstep * n + 2]) > arm_threshold)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
					g += 6 * r;
				}
			}
		}
		if (method == CrossBasedLocalFilter::CROSS_BASED_LOCAL_FILTER_ARM_SAMELENGTH)
		{
			if (c == 1)
			{
				for (j = 0; j < guide.rows; j++)
				{
					for (i = 0; i < guide.cols; i++, g++, cdata++)
					{
						const uchar v = g[0];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > arm_threshold)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n]) > arm_threshold)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > arm_threshold)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n * gstep]) > arm_threshold)
								break;
							cdata->vp = n;
						}
						cdata->vm = cdata->vp = min(cdata->vm, cdata->vp);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					g += 2 * r;
				}
			}
			else if (c == 3)
			{
				for (j = 0; j < guide.rows; j++)
				{
					for (i = 0; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > arm_threshold)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > arm_threshold)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-gstep * n + 0]) + abs(gv - g[-gstep * n + 1]) + abs(bv - g[-gstep * n + 2]) > arm_threshold)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[gstep * n + 0]) + abs(gv - g[gstep * n + 1]) + abs(bv - g[gstep * n + 2]) > arm_threshold)
								break;
							cdata->vp = n;
						}
						cdata->vm = cdata->vp = min(cdata->vm, cdata->vp);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					g += 6 * r;
				}
			}
		}

		//clipping
		cdata = crossdata;
		for (j = 0; j <= r; j++)
		{
			for (i = 0; i <= r; i++)
			{
				cdata[cstep * j + i].hm = min(cdata[cstep * j + i].hm, (uchar)i);
				cdata[cstep * j + i].vm = min(cdata[cstep * j + i].vm, (uchar)j);
				cdata->divh = 1.0f / (float)(cdata[cstep * j + i].hp + cdata[cstep * j + i].hm + 1);
				cdata->divv = 1.0f / (float)(cdata[cstep * j + i].vp + cdata[cstep * j + i].vm + 1);
			}
			for (; i < guide.cols - r - 1; i++)
			{
				cdata[cstep * j + i].vm = min(cdata[cstep * j + i].vm, (uchar)j);
				cdata->divv = 1.0f / (float)(cdata[cstep * j + i].vp + cdata[cstep * j + i].vm + 1);
			}
			for (; i < guide.cols; i++)
			{
				cdata[cstep * j + i].hp = min(cdata[cstep * j + i].hp, (uchar)(guide.cols - 1 - i));
				cdata[cstep * j + i].vm = min(cdata[cstep * j + i].vm, (uchar)j);
				cdata->divh = 1.0f / (float)(cdata[cstep * j + i].hp + cdata[cstep * j + i].hm + 1);
				cdata->divv = 1.0f / (float)(cdata[cstep * j + i].vp + cdata[cstep * j + i].vm + 1);
			}
		}
		for (; j < guide.rows - r - 1; j++)
		{
			for (i = 0; i <= r; i++)
			{
				cdata[cstep * j + i].hm = min(cdata[cstep * j + i].hm, (uchar)i);

				cdata->divh = 1.0f / (float)(cdata[cstep * j + i].hp + cdata[cstep * j + i].hm + 1);
			}
			for (i = guide.cols - r - 1; i < guide.cols; i++)
			{
				cdata[cstep * j + i].hp = min(cdata[cstep * j + i].hp, (uchar)(guide.cols - 1 - i));
				cdata->divh = 1.0f / (float)(cdata[cstep * j + i].hp + cdata[cstep * j + i].hm + 1);
			}
		}
		for (; j < guide.rows; j++)
		{
			for (i = 0; i <= r; i++)
			{
				cdata[cstep * j + i].hm = min(cdata[cstep * j + i].hm, (uchar)i);
				cdata[cstep * j + i].vp = min(cdata[cstep * j + i].vp, (uchar)(guide.rows - 1 - j));
				cdata->divh = 1.0f / (float)(cdata[cstep * j + i].hp + cdata[cstep * j + i].hm + 1);
				cdata->divv = 1.0f / (float)(cdata[cstep * j + i].vp + cdata[cstep * j + i].vm + 1);
			}
			for (; i < guide.cols - r - 1; i++)
			{
				cdata[cstep * j + i].vp = min(cdata[cstep * j + i].vp, (uchar)(guide.rows - 1 - j));
				cdata->divv = 1.0f / (float)(cdata[cstep * j + i].vp + cdata[cstep * j + i].vm + 1);
			}
			for (; i < guide.cols; i++)
			{
				cdata[cstep * j + i].hp = min(cdata[cstep * j + i].hp, (uchar)(guide.cols - 1 - i));
				cdata[cstep * j + i].vp = min(cdata[cstep * j + i].vp, (uchar)(guide.rows - 1 - j));
				cdata->divh = 1.0f / (float)(cdata[cstep * j + i].hp + cdata[cstep * j + i].hm + 1);
				cdata->divv = 1.0f / (float)(cdata[cstep * j + i].vp + cdata[cstep * j + i].vm + 1);
			}
		}
	}

	/*
	void CrossBasedLocalFilter::makeKernel(Mat& guide, const int r, int thresh, int method)
	{
	size = guide.size();
	if(crossdata==0)
	crossdata = new cross[guide.size().area()];
	else
	{
	delete[]crossdata;
	crossdata = new cross[guide.size().area()];
	}

	const int hw = r;
	const int c = guide.channels();

	uchar* g = guide.ptr(0);
	const int gstep = guide.cols*c;
	cross* cdata = crossdata;
	const int cstep = guide.cols;
	thresh*=c;

	if(method==CROSS_BASED_LOCAL_FILTER_ARM_BASIC)
	{
	if(c==1)
	{
	int j=0;
	for(;j<r;j++)
	{
	int i=0;
	for(;i<r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}
	for(i=r;i<guide.cols-r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}

	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}

	for(;i<guide.cols;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}
	}
	for(j=r;j<guide.rows-r;j++)
	{
	int i=0;
	for(;i<r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}
	for(i=r;i<guide.cols-r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}

	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}

	for(;i<guide.cols;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}
	}
	for(;j<guide.rows;j++)
	{
	int i=0;
	for(;i<r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hp=hp;
	cdata->hm=hm;
	cdata->divh=1.0f/(float)(hp+hm+1);

	int vp=minSearch;;
	int vm=minSearch;;

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=vp;
	cdata->vm=vm;
	cdata->divv=1.0f/(float)(vp+vm+1);
	}
	for(i=r;i<guide.cols-r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	cdata->hp=n;
	}

	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);


	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}

	for(;i<guide.cols;i++,g++,cdata++)
	{
	const uchar v = g[0];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	}
	}
	else if(c==3)
	{
	int j=0;
	for(;j<r;j++)
	{
	int i=0;
	for(;i<r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	cdata->vm=cdata->hp=cdata->vp=cdata->vm=0;
	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	for(i=r;i<guide.cols-r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}

	for(;i<guide.cols;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	}
	for(j=r;j<guide.rows-r;j++)
	{
	int i=0;
	for(;i<r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	//main body
	for(i=r;i<guide.cols-r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}

	for(;i<guide.cols;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	}

	for(;j<guide.rows;j++)
	{
	int i=0;
	for(;i<r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	for(i=r;i<guide.cols-r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}

	for(;i<guide.cols;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->divh=1.0f/(float)(cdata->hp+cdata->hm+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->divv=1.0f/(float)(cdata->vp+cdata->vm+1);
	}
	}
	}
	}
	if(method==CROSS_BASED_LOCAL_FILTER_ARM_SAMELENGTH)
	{
	if(c==1)
	{
	int j=0;
	for(;j<r;j++)
	{
	int i=0;
	for(;i<r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vp=cdata->vm=min(vp,vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	for(i=r;i<guide.cols-r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}

	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vm=cdata->vp=min(vp,vm);
	cdata->divv=1.0f/(float)((cdata->vm<<1)+1);
	}

	for(;i<guide.cols;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vm=cdata->vp=min(vp,vm);
	cdata->divv=1.0f/(float)((cdata->vm<<1)+1);
	}
	}
	for(j=r;j<guide.rows-r;j++)
	{
	int i=0;
	for(;i<r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vm=cdata->vp=min(vp,vm);
	cdata->divv=1.0f/(float)((cdata->vm<<1)+1);
	}
	for(i=r;i<guide.cols-r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}

	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vm=cdata->vp=min(vp,vm);
	cdata->divv=1.0f/(float)((cdata->vm<<1)+1);
	}

	for(;i<guide.cols;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hp;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}
	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;
	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vm=cdata->vp=min(vp,vm);

	cdata->divv=1.0f/(float)((cdata->vm<<1)+1);
	}
	}
	for(;j<guide.rows;j++)
	{
	int i=0;
	for(;i<r;i++,g++,cdata++)
	{
	const uchar v = g[0];
	int hm=minSearch;
	for(int n=1;n<=i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	hm=n;
	}
	int hp=minSearch;
	for(int n=1;n<=hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	hp=n;
	}

	cdata->hm=cdata->hp=min(hp,hm);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	int vp=minSearch;;
	int vm=minSearch;;

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	vm=n;
	}
	for(int n=1;n<=vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	vp=n;
	}
	cdata->vm=cdata->vp=min(vp,vm);
	cdata->divv=1.0f/(float)((cdata->vm<<1)+1);
	}
	for(i=r;i<guide.cols-r;i++,g++,cdata++)
	{
	const uchar v = g[0];

	for(int n=1;n<=hw;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm = cdata->hp = min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hp<<1)+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vm = cdata->vp = min(cdata->vm,cdata->vp);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}

	for(;i<guide.cols;i++,g++,cdata++)
	{
	const uchar v = g[0];

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(v-g[-n])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(v-g[n])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm = cdata->hp = min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hp<<1)+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(v-g[-n*gstep])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(v-g[n*gstep])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vm = cdata->vp = min(cdata->vm,cdata->vp);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	}
	}
	else if(c==3)
	{
	int j=0;
	//upper
	for(;j<r;j++)
	{
	int i=0;
	//left
	for(;i<r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	//center
	for(i=r;i<guide.cols-r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	//right
	for(;i<guide.cols;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	}
	//middle
	for(j=r;j<guide.rows-r;j++)
	{
	int i=0;
	//left
	for(;i<r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	//main body center
	for(i=r;i<guide.cols-r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	//right
	for(;i<guide.cols;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	}
	//down
	for(;j<guide.rows;j++)
	{
	//left
	int i=0;
	for(;i<r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	//center
	for(i=r;i<guide.cols-r;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=hw;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	//right
	for(;i<guide.cols;i++,g+=3,cdata++)
	{
	const uchar rv = g[0];
	const uchar gv = g[1];
	const uchar bv = g[2];
	cdata->hm=cdata->hp=cdata->vp=cdata->vm=0;

	for(int n=1;n<=guide.cols-1-i;n++)
	{
	if (abs(rv-g[-3*n+0]) + abs(gv-g[-3*n+1]) + abs(bv-g[-3*n+2])>thresh)
	break;
	cdata->hm=n;
	}
	for(int n=1;n<=cdata->hm;n++)
	{
	if (abs(rv-g[3*n+0]) + abs(gv-g[3*n+1]) + abs(bv-g[3*n+2])>thresh)
	break;
	cdata->hp=n;
	}
	cdata->hm=cdata->hp=min(cdata->hm,cdata->hp);
	cdata->divh=1.0f/(float)((cdata->hm<<1)+1);

	for(int n=1;n<=guide.rows-1-j;n++)
	{
	if (abs(rv-g[-n*gstep+0]) + abs(gv-g[-n*gstep+1]) + abs(bv-g[-n*gstep+2])>thresh)
	break;
	cdata->vm=n;
	}
	for(int n=1;n<=cdata->vm;n++)
	{
	if (abs(rv-g[n*gstep+0]) + abs(gv-g[n*gstep+1]) + abs(bv-g[n*gstep+2])>thresh)
	break;
	cdata->vp=n;
	}
	cdata->vp=cdata->vm=min(cdata->vp,cdata->vm);
	cdata->divv=1.0f/(float)((cdata->vp<<1)+1);
	}
	}
	}
	}
	}
	*/

	void CrossBasedLocalFilter::makeKernel(Mat& guide, const int r, const int thresh, const double smoothingrate, const int method)
	{
		size = guide.size();
		if (crossdata == nullptr)
		{
			crossdata = new cross[guide.size().area()];
		}
		else
		{
			delete[]crossdata;
			crossdata = new cross[guide.size().area()];
		}

		const int hw = r;
		const int c = guide.channels();

		uchar* g = guide.ptr(0);
		const int gstep = guide.cols * c;
		cross* cdata = crossdata;
		const int cstep = guide.cols;
		int arm_threshold = thresh * c;

		if (method == CROSS_BASED_LOCAL_FILTER_ARM_BASIC)
		{
			//upper
			if (c == 1)
			{
				int j = 0;
				for (; j < r; j++)
				{
					int i = 0;
					for (; i < r; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						int hp = minSearch;
						v = g[0];
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						v = g[0];
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}
					for (i = r; i < guide.cols - r; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						int hp = minSearch;
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						v = g[0];
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}

					for (; i < guide.cols; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						int hp = minSearch;
						v = g[0];
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						v = g[0];
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}
				}
				//middle
				for (j = r; j < guide.rows - r; j++)
				{
					int i = 0;
					for (; i < r; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						v = g[0];
						int hp = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}
					for (i = r; i < guide.cols - r; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						v = g[0];
						int hp = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}

					for (; i < guide.cols; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						v = g[0];
						int hp = minSearch;
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);
						v = g[0];
						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}
				}
				for (; j < guide.rows; j++)
				{
					int i = 0;
					for (; i < r; i++, g++, cdata++)
					{
						uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							hm = n;
						}
						v = g[0];
						int hp = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							hp = n;
						}

						cdata->hp = hp;
						cdata->hm = hm;
						cdata->divh = 1.0f / (float)(hp + hm + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						v = g[0];
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							vm = n;
						}
						v = g[0];
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							vp = n;
						}
						cdata->vp = vp;
						cdata->vm = vm;
						cdata->divv = 1.0f / (float)(vp + vm + 1);
					}
					for (i = r; i < guide.cols - r; i++, g++, cdata++)
					{
						uchar v = g[0];

						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							cdata->hm = n;
						}
						v = g[0];
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							cdata->hp = n;
						}

						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);
						v = g[0];
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							cdata->vm = n;
						}
						v = g[0];
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}

					for (; i < guide.cols; i++, g++, cdata++)
					{
						uchar v = g[0];

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							v = (v + g[-n]) >> 1;
							cdata->hm = n;
						}
						v = g[0];
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							v = (v + g[n]) >> 1;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);
						v = g[0];
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							v = (v + g[-n * gstep]) >> 1;
							cdata->vm = n;
						}
						v = g[0];
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							v = (v + g[n * gstep]) >> 1;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
				}
			}
			else if (c == 3)
			{
				int j = 0;
				for (; j < r; j++)
				{
					int i = 0;
					for (; i < r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
					for (i = r; i < guide.cols - r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}

					for (; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
				}
				for (j = r; j < guide.rows - r; j++)
				{
					int i = 0;
					for (; i < r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
					//main body
					for (i = r; i < guide.cols - r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}

					for (; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
				}

				for (; j < guide.rows; j++)
				{
					int i = 0;
					for (; i < r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
					for (i = r; i < guide.cols - r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}

					for (; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->divh = 1.0f / (float)(cdata->hp + cdata->hm + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->divv = 1.0f / (float)(cdata->vp + cdata->vm + 1);
					}
				}
			}
		}
		if (method == CROSS_BASED_LOCAL_FILTER_ARM_SAMELENGTH)
		{
			if (c == 1)
			{
				int j = 0;
				for (; j < r; j++)
				{
					int i = 0;
					for (; i < r; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}

						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}
						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vp = cdata->vm = min(vp, vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					for (i = r; i < guide.cols - r; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}

						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}

						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vm = cdata->vp = min(vp, vm);
						cdata->divv = 1.0f / (float)((cdata->vm << 1) + 1);
					}

					for (; i < guide.cols; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}

						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}
						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vm = cdata->vp = min(vp, vm);
						cdata->divv = 1.0f / (float)((cdata->vm << 1) + 1);
					}
				}
				for (j = r; j < guide.rows - r; j++)
				{
					int i = 0;
					for (; i < r; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}

						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}
						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vm = cdata->vp = min(vp, vm);
						cdata->divv = 1.0f / (float)((cdata->vm << 1) + 1);
					}
					for (i = r; i < guide.cols - r; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}

						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}

						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vm = cdata->vp = min(vp, vm);
						cdata->divv = 1.0f / (float)((cdata->vm << 1) + 1);
					}

					for (; i < guide.cols; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hp; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}
						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;
						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}
						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vm = cdata->vp = min(vp, vm);

						cdata->divv = 1.0f / (float)((cdata->vm << 1) + 1);
					}
				}
				for (; j < guide.rows; j++)
				{
					int i = 0;
					for (; i < r; i++, g++, cdata++)
					{
						const uchar v = g[0];
						int hm = minSearch;
						for (int n = 1; n <= i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							hm = n;
						}
						int hp = minSearch;
						for (int n = 1; n <= hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							hp = n;
						}

						cdata->hm = cdata->hp = min(hp, hm);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						int vp = minSearch;;
						int vm = minSearch;;

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							vm = n;
						}
						for (int n = 1; n <= vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							vp = n;
						}
						cdata->vm = cdata->vp = min(vp, vm);
						cdata->divv = 1.0f / (float)((cdata->vm << 1) + 1);
					}
					for (i = r; i < guide.cols - r; i++, g++, cdata++)
					{
						const uchar v = g[0];

						for (int n = 1; n <= hw; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hp << 1) + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vm = cdata->vp = min(cdata->vm, cdata->vp);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}

					for (; i < guide.cols; i++, g++, cdata++)
					{
						const uchar v = g[0];

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(v - g[-n]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(v - g[n]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hp << 1) + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(v - g[-n * gstep]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(v - g[n * gstep]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vm = cdata->vp = min(cdata->vm, cdata->vp);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
				}
			}
			else if (c == 3)
			{
				int j = 0;
				//upper
				for (; j < r; j++)
				{
					int i = 0;
					//left
					for (; i < r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					//center
					for (i = r; i < guide.cols - r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					//right
					for (; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
				}
				//middle
				for (j = r; j < guide.rows - r; j++)
				{
					int i = 0;
					//left
					for (; i < r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					//main body center
					for (i = r; i < guide.cols - r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					//right
					for (; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
				}
				//down
				for (; j < guide.rows; j++)
				{
					//left
					int i = 0;
					for (; i < r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					//center
					for (i = r; i < guide.cols - r; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= hw; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
					//right
					for (; i < guide.cols; i++, g += 3, cdata++)
					{
						const uchar rv = g[0];
						const uchar gv = g[1];
						const uchar bv = g[2];
						cdata->hm = cdata->hp = cdata->vp = cdata->vm = 0;

						for (int n = 1; n <= guide.cols - 1 - i; n++)
						{
							if (abs(rv - g[-3 * n + 0]) + abs(gv - g[-3 * n + 1]) + abs(bv - g[-3 * n + 2]) > thresh)
								break;
							cdata->hm = n;
						}
						for (int n = 1; n <= cdata->hm; n++)
						{
							if (abs(rv - g[3 * n + 0]) + abs(gv - g[3 * n + 1]) + abs(bv - g[3 * n + 2]) > thresh)
								break;
							cdata->hp = n;
						}
						cdata->hm = cdata->hp = min(cdata->hm, cdata->hp);
						cdata->divh = 1.0f / (float)((cdata->hm << 1) + 1);

						for (int n = 1; n <= guide.rows - 1 - j; n++)
						{
							if (abs(rv - g[-n * gstep + 0]) + abs(gv - g[-n * gstep + 1]) + abs(bv - g[-n * gstep + 2]) > thresh)
								break;
							cdata->vm = n;
						}
						for (int n = 1; n <= cdata->vm; n++)
						{
							if (abs(rv - g[n * gstep + 0]) + abs(gv - g[n * gstep + 1]) + abs(bv - g[n * gstep + 2]) > thresh)
								break;
							cdata->vp = n;
						}
						cdata->vp = cdata->vm = min(cdata->vp, cdata->vm);
						cdata->divv = 1.0f / (float)((cdata->vp << 1) + 1);
					}
				}
			}
		}
	}

	void CrossBasedLocalFilter::visualizeKernel(Mat& dest, Point& pt)
	{
		cross* cr = &crossdata[pt.x + pt.y * dest.cols];

		dest.setTo(0);
		int sty = -cr->vm + pt.y;
		int edy = cr->vp + pt.y;
		for (int j = sty; j <= edy; j++)
		{
			if (j >= 0 && j < dest.cols)
			{
				line(dest, Point(pt.x - crossdata[pt.x + j * dest.cols].hm, j), Point(pt.x + crossdata[pt.x + j * dest.cols].hp, j), CV_RGB(255, 255, 255));
				if (j == pt.y)line(dest, Point(pt.x - crossdata[pt.x + pt.y * dest.cols].hm, j), Point(pt.x + crossdata[pt.x + pt.y * dest.cols].hp, j), CV_RGB(255, 0, 0));
			}
		}
		line(dest, Point(pt.x, sty), Point(pt.x, edy), CV_RGB(255, 0, 0));
	}

	void CrossBasedLocalFilter::getCrossAreaCountMap(Mat& dest, int type)
	{
		if (dest.empty())dest.create(size, type);
		if (dest.type() != type)dest.create(size, type);

		const int step = size.width;
		int i, j;

		cross* cds = crossdata;
		if (type == CV_8U)
		{
			dest.setTo(0);
			uchar* d = dest.ptr<uchar>(0);
			for (j = size.height; j--;)
			{
				for (i = size.width; i--;)
				{
					cross* cd = cds - cds->vm * step;
					const int v = cds->vp + cds->vm + 1;
					int val = 1;
					for (int n = 0; n < v; n++)
					{
						val += cd->hm + cd->hp;
						cd += step;
					}
					*d = min(val, 255);
					d++, cds++;
				}
			}
		}
		else if (type == CV_16S)
		{
			dest.setTo(1);
			short* d = dest.ptr<short>(0);
			for (j = size.height; j--;)
			{
				for (i = size.width; i--;)
				{
					cross* cd = cds - cds->vm * step;
					const int v = cds->vp + cds->vm + 1;
					for (int n = 0; n < v; n++)
					{
						*d += cd->hm + cd->hp;
						cd += step;
					}
					d++, cds++;
				}
			}
		}
		else if (type == CV_16U)
		{
			dest.setTo(1);
			unsigned short* d = dest.ptr<unsigned short>(0);
			for (j = size.height; j--;)
			{
				for (i = size.width; i--;)
				{
					cross* cd = cds - cds->vm * step;
					const int v = cds->vp + cds->vm + 1;
					for (int n = 0; n < v; n++)
					{
						*d += cd->hm + cd->hp;
						cd += step;
					}
					d++, cds++;
				}
			}
		}
		else if (type == CV_32F)
		{
			float* d = dest.ptr<float>(0);
			for (j = size.height; j--;)
			{
				dest.setTo(1.f);
				for (i = size.width; i--;)
				{
					cross* cd = cds - cds->vm * step;
					const int v = cds->vp + cds->vm + 1;
					for (int n = 0; n < v; n++)
					{
						*d += cd->hm + cd->hp;
						cd += step;
					}
					d++, cds++;
				}
			}
		}
		else if (type == CV_64F)
		{
			double* d = dest.ptr<double>(0);
			for (j = size.height; j--;)
			{
				dest.setTo(1.0);
				for (i = size.width; i--;)
				{
					cross* cd = cds - cds->vm * step;
					const int v = cds->vp + cds->vm + 1;
					for (int n = 0; n < v; n++)
					{
						*d += cd->hm + cd->hp;
						cd += step;
					}
					d++, cds++;
				}
			}
		}

		dest.copyTo(areaMap);
	}

	void CrossBasedLocalFilter::operator()(Mat& src, Mat& weight, Mat& guide, Mat& dest, const int r, int thresh, int iteration)
	{
		if (dest.empty())dest.create(src.size(), src.type());
		if (src.channels() != dest.channels())dest.create(src.size(), src.type());
		{
			//		CalcTime t("kernel");
			makeKernel(guide, r, thresh, 1);
		}
		iteration = max(iteration, 1);
		{
			//	CalcTime t("filter");
			for (int i = 0; i < iteration; i++)
				operator()(src, weight, dest);
		}
	}

	void CrossBasedLocalFilter::operator()(Mat& src, Mat& guide, Mat& dest, const int r, int thresh, int iteration)
	{
		if (dest.empty())dest.create(src.size(), src.type());
		if (src.channels() != dest.channels())dest.create(src.size(), src.type());
		{
			//	CalcTime t("kernel");
			makeKernel(guide, r, thresh);
		}
		iteration = max(iteration, 1);
		{
			//	CalcTime t("filter");
			for (int i = 0; i < iteration; i++)
				operator()(src, dest);
		}
	}

	void CrossBasedLocalFilter::operator()(Mat& src, Mat& dest)
	{
		const int cn = src.channels();
		if (dest.empty())dest.create(src.size(), src.type());
		if (src.channels() != dest.channels())dest.create(src.size(), src.type());
		Mat src_ = src;
		if (src.data == dest.data)src_ = src.clone();

		if (src.type() == CV_MAKETYPE(CV_8U, cn))
		{
			orthogonalIntegralImageFilterI<uchar>(src_, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_16S, cn))
		{
			orthogonalIntegralImageFilterI<short>(src_, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_16U, cn))
		{
			orthogonalIntegralImageFilterI<unsigned short>(src_, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_32F, cn))
		{
			orthogonalIntegralImageFilterF<float>(src_, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_64F, cn))
		{
			orthogonalIntegralImageFilterF<double>(src_, dest, crossdata);
		}
	}

	void CrossBasedLocalFilter::operator()(Mat& src, Mat& weight, Mat& dest)
	{
		const int cn = src.channels();
		if (dest.empty())dest.create(src.size(), src.type());
		if (src.channels() != dest.channels())dest.create(src.size(), src.type());
		Mat src_ = src;
		Mat weight_;
		if (src.data == dest.data)src_ = src.clone();

		if (src.type() == CV_MAKETYPE(CV_8U, cn))
		{
			if (weight.type() == CV_8U)weight_ = weight;
			else weight.convertTo(weight_, CV_8U);
			orthogonalIntegralImageFilterI<uchar>(src_, weight, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_16S, cn))
		{
			if (weight.type() == CV_16S)weight_ = weight;
			else weight.convertTo(weight_, CV_16S);
			orthogonalIntegralImageFilterI<short>(src_, weight, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_16U, cn))
		{
			if (weight.type() == CV_16U)weight_ = weight;
			else weight.convertTo(weight_, CV_16U);
			orthogonalIntegralImageFilterI<unsigned short>(src_, weight, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_32F, cn))
		{
			if (weight.type() == CV_32F)weight_ = weight;
			else weight.convertTo(weight_, CV_32F);
			orthogonalIntegralImageFilterF<float>(src_, weight, dest, crossdata);
		}
		else if (src.type() == CV_MAKETYPE(CV_64F, cn))
		{
			if (weight.type() == CV_64F)weight_ = weight;
			else weight.convertTo(weight_, CV_64F);
			orthogonalIntegralImageFilterF<double>(src_, weight, dest, crossdata);
		}
	}


	template <class S, class T>
	void crossBasedAdaptiveBoxFilterSP_(Mat& src, Mat& guide, Mat& dest, int r, int thresh)
	{
		if (dest.empty())dest.create(src.size(), src.type());

		const int hw = r;
		Mat sim, gim;
		copyMakeBorder(src, sim, 0, 0, hw, hw, cv::BORDER_REPLICATE);
		copyMakeBorder(guide, gim, 0, 0, hw, hw, cv::BORDER_REPLICATE);

		unsigned int* integral = new unsigned int[(sim.cols) + 1];
		unsigned int* iH = integral + hw;

		const int c = guide.channels();
		const int gstep = gim.cols * c;

		T* is = sim.ptr<T>(0);
		T* s = is + hw;
		S* g = gim.ptr<S>(0); g += hw * c;
		T* d = dest.ptr<T>(0);
		if (c == 1)
		{
			for (int j = 0; j < src.rows; j++)
			{
				integral[0] = 0;
				for (int i = 0; i < sim.cols; i++)
					integral[i + 1] = integral[i] + is[i];

				for (int i = 0; i < src.cols; i++)
				{
					const S v = g[i];
					int rl = 0;
					for (int n = 1; n <= hw; n++)
					{
						if (abs(v - g[i - n]) > thresh)
							break;
						rl = -n;
					}
					int rr = 0;
					for (int n = 1; n <= hw; n++)
					{
						if (abs(v - g[i + n]) > thresh)
							break;
						rr = n;
					}
					d[i] = (T)((iH[i + 1 + rr] - iH[i + rl]) / (float)(rr - rl + 1) + 0.5f);
				}
				s += sim.cols;
				is += sim.cols;
				g += gstep;
				d += dest.cols;
			}
		}
		else if (c == 3)
		{
			for (int j = 0; j < src.rows; j++)
			{
				integral[0] = 0;
				for (int i = 0; i < sim.cols; i++)
					integral[i + 1] = integral[i] + is[i];

				for (int i = 0; i < src.cols; i++)
				{
					const T rv = g[3 * i];
					const T gv = g[3 * i + 1];
					const T bv = g[3 * i + 2];
					int rl = 0;
					for (int n = 1; n <= hw; n++)
					{
						if (abs(rv - g[3 * (i - n) + 0]) + abs(gv - g[3 * (i - n) + 1]) + abs(bv - g[3 * (i - n) + 2]) > 3 * thresh)
							break;
						rl = -n;
					}
					int rr = 0;
					for (int n = 1; n <= hw; n++)
					{
						if (abs(rv - g[3 * (i + n) + 0]) + abs(gv - g[3 * (i + n) + 1]) + abs(bv - g[3 * (i + n) + 2]) > 3 * thresh)
							break;
						rr = n;
					}
					d[i] = (T)((iH[i + 1 + rr] - iH[i + rl]) / (float)(rr - rl + 1) + 0.5f);
				}
				s += sim.cols;
				is += sim.cols;
				g += gstep;
				d += dest.cols;
			}
		}

		delete[] integral;
	}

	template <class T>
	void crossBasedAdaptiveBoxFilter_(Mat& src, Mat& guide, Mat& dest, Size ksize, int thresh)
	{
		crossBasedAdaptiveBoxFilterSP_<uchar, T>(src, guide, dest, ksize.width / 2, thresh);
		//crossBasedAdaptiveBoxFilterSP_<T>(src,guide,dest,0,thresh);
		Mat st, gt;
		transpose(dest, st);
		transpose(guide, gt);
		crossBasedAdaptiveBoxFilterSP_<uchar, T>(st, gt, st, ksize.height / 2, thresh);
		transpose(st, dest);
	}

	template <class T>
	void crossBasedAdaptiveBoxFilter_(Mat& src, Mat& dest, Size ksize, int thresh)
	{
		crossBasedAdaptiveBoxFilterSP_<T, T>(src, src, dest, ksize.width / 2, thresh);
		//crossBasedAdaptiveBoxFilterSP_<T>(src,guide,dest,0,thresh);
		Mat st, gt;
		transpose(dest, st);
		crossBasedAdaptiveBoxFilterSP_<T, T>(st, st, st, ksize.height / 2, thresh);
		transpose(st, dest);
	}

	void crossBasedAdaptiveBoxFilter(Mat& src, Mat& dest, const Size ksize, const int thresh)
	{
		Mat guidance;

		if (src.type() == CV_8U)
			crossBasedAdaptiveBoxFilter_<uchar>(src, dest, ksize, thresh);
		else if (src.type() == CV_16S)
			crossBasedAdaptiveBoxFilter_<short>(src, dest, ksize, thresh);
		else if (src.type() == CV_16U)
			crossBasedAdaptiveBoxFilter_<unsigned short>(src, dest, ksize, thresh);
		else if (src.type() == CV_32F)
			crossBasedAdaptiveBoxFilter_<float>(src, dest, ksize, thresh);
		else if (src.type() == CV_64F)
			crossBasedAdaptiveBoxFilter_<double>(src, dest, ksize, thresh);
	}

	void crossBasedAdaptiveBoxFilter(Mat& src, Mat& guide, Mat& dest, const Size ksize, const int thresh)
	{
		Mat guidance;

		if (src.type() == CV_8U)
			crossBasedAdaptiveBoxFilter_<uchar>(src, guide, dest, ksize, thresh);
		else if (src.type() == CV_16S)
			crossBasedAdaptiveBoxFilter_<short>(src, guide, dest, ksize, thresh);
		else if (src.type() == CV_16U)
			crossBasedAdaptiveBoxFilter_<unsigned short>(src, guide, dest, ksize, thresh);
		else if (src.type() == CV_32F)
			crossBasedAdaptiveBoxFilter_<float>(src, guide, dest, ksize, thresh);
		else if (src.type() == CV_64F)
			crossBasedAdaptiveBoxFilter_<double>(src, guide, dest, ksize, thresh);
	}

	static void guiCrossBasedLocalFilter(int events, int x, int y, int flags, void* param)
	{
		Point* pt = (Point*)param;
		//if(events==CV_EVENT_LBUTTONDOWN)
		if (flags & EVENT_FLAG_LBUTTON)
		{
			pt->x = x;
			pt->y = y;
		}
	}

	void guiCrossBasedLocalFilter(Mat& src)
	{
		CrossBasedLocalFilter cbabf(src, 5, 30);
		string wname = "Kernel";
		namedWindow(wname);
		Point mpt = Point(100, 100);
		setMouseCallback(wname, guiCrossBasedLocalFilter, &mpt);

		int alpha = 70;

		createTrackbar("alpha", wname, &alpha, 100);
		int thresh = 50;
		createTrackbar("thresh", wname, &thresh, 255);
		int r = 50;
		createTrackbar("r", wname, &r, src.cols / 2);

		int iter = 1;
		createTrackbar("iter", wname, &iter, 100);

		int key = 0;
		Mat kernel;
		Mat show(src.size(), CV_8UC3);
		Mat psf = Mat::zeros(src.size(), CV_16S);
		bool isGrid = false;
		bool isColor = true;
		bool isWeight = true;
		Mat dest;
		Mat dest2;
		Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);

		Mat weight;
		while (key != 'q')
		{
			cbabf.getCrossAreaCountMap(weight);
			//weight.setTo(1);

			//cbabf.makeKernel(src,r,thresh);
			if (isWeight)
			{
				/*if(isColor)
				{
				cbabf.makeKernel(src,r,thresh,0.0,0);
				cbabf(src,dest);
				}
				else
				{
				cbabf.makeKernel(gray,r,thresh,0.0,0);
				cbabf(gray,dest);
				}*/

				if (isColor)
				{
					Timer t("time");
					cbabf.makeKernel(src, r, thresh, 1);
					cbabf(src, dest);
				}
				else
				{
					Timer t("time");
					cbabf.makeKernel(gray, r, thresh, 1);
					cbabf(gray, dest);
				}

			}
			else
			{
				if (isColor)
				{
					Timer t("time");
					cbabf.makeKernel(src, r, thresh, 0);
					cbabf(src, dest);
				}
				else
				{
					Timer t("time");
					cbabf.makeKernel(gray, r, thresh, 0);
					cbabf(gray, dest);
				}
			}

			cbabf.visualizeKernel(show, Point(mpt.x, mpt.y));
			crossBasedAdaptiveBoxFilter(gray, src, dest2, Size(2 * r + 1, 2 * r + 1), thresh);
			if (!isColor) guiAlphaBlend(dest, dest2);
			imshow("2", dest2);
			/*{
			CalcTime t("single");

			}*/

			/*psf.setTo(0);
			psf.at<short>(y,x)=SHRT_MAX;
			crossBasedAdaptiveBoxFilter(psf,src,psf,Size(2*r+1,2*rh+1),thresh);
			threshold(psf,psf,1,255,cv::THRESH_BINARY);

			psf.convertTo(kernel,CV_8U);*/

			if (isColor)
				alphaBlend(show, src, alpha / 100.0, show);
			else
			{
				Mat gs; cvtColor(gray, gs, COLOR_GRAY2BGR);
				alphaBlend(show, gs, alpha / 100.0, show);
			}
			imshow("filtered", dest);
			if (isGrid)
			{
				line(show, Point(mpt.x, 0), Point(mpt.x, src.rows - 1), CV_RGB(255, 0, 0));
				line(show, Point(0, mpt.y), Point(src.cols - 1, mpt.y), CV_RGB(255, 0, 0));
			}
			imshow(wname, show);
			key = waitKey(1);
			if (key == 'g')isGrid = (isGrid) ? false : true;
			if (key == 'c')isColor = (isColor) ? false : true;
			if (key == 'w')isWeight = (isWeight) ? false : true;
		}
	}
}
