#include "opencp.hpp"
using namespace std;
using namespace cv;

namespace cp
{

	//point cloud rendering
#if CV_SSE4_1
	static void myProjectPoint_SSE(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
	{
		float r[3][3];
		Mat kr = K*R;
		r[0][0] = (float)kr.at<double>(0, 0);
		r[0][1] = (float)kr.at<double>(0, 1);
		r[0][2] = (float)kr.at<double>(0, 2);

		r[1][0] = (float)kr.at<double>(1, 0);
		r[1][1] = (float)kr.at<double>(1, 1);
		r[1][2] = (float)kr.at<double>(1, 2);

		r[2][0] = (float)kr.at<double>(2, 0);
		r[2][1] = (float)kr.at<double>(2, 1);
		r[2][2] = (float)kr.at<double>(2, 2);

		float tt[3];
		tt[0] = (float)t.at<double>(0);
		tt[1] = (float)t.at<double>(1);
		tt[2] = (float)t.at<double>(2);

		float* data = (float*)xyz.ptr<float>(0);
		Point2f* dst = &dest[0];

		int size1 = (xyz.size().area() / 4);
		int size2 = xyz.size().area() % 4;

		int i;

		const __m128 addx = _mm_set_ps1(tt[0]);
		const __m128 addy = _mm_set_ps1(tt[1]);
		const __m128 addz = _mm_set_ps1(tt[2]);

		const __m128 r00 = _mm_set_ps1(r[0][0]);
		const __m128 r01 = _mm_set_ps1(r[0][1]);
		const __m128 r02 = _mm_set_ps1(r[0][2]);

		const __m128 r10 = _mm_set_ps1(r[1][0]);
		const __m128 r11 = _mm_set_ps1(r[1][1]);
		const __m128 r12 = _mm_set_ps1(r[1][2]);

		const __m128 r20 = _mm_set_ps1(r[2][0]);
		const __m128 r21 = _mm_set_ps1(r[2][1]);
		const __m128 r22 = _mm_set_ps1(r[2][2]);


		for (i = 0; i < size1; i++)
		{
			__m128 a = _mm_load_ps((data));
			__m128 b = _mm_load_ps((data + 4));
			__m128 c = _mm_load_ps((data + 8));

			__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
			aa = _mm_blend_ps(aa, b, 4);
			__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
			__m128  mx = _mm_add_ps(addx, _mm_blend_ps(aa, cc, 8));

			aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
			__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
			bb = _mm_blend_ps(bb, aa, 1);
			cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
			__m128 my = _mm_add_ps(addy, _mm_blend_ps(bb, cc, 8));

			aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
			bb = _mm_blend_ps(aa, b, 2);
			cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
			__m128 mz = _mm_add_ps(addz, _mm_blend_ps(bb, cc, 12));

			const __m128 div = _mm_rcp_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(r20, mx), _mm_mul_ps(r21, my)), _mm_mul_ps(r22, mz)));

			a = _mm_mul_ps(div, _mm_add_ps(_mm_add_ps(_mm_mul_ps(r00, mx), _mm_mul_ps(r01, my)), _mm_mul_ps(r02, mz)));
			b = _mm_mul_ps(div, _mm_add_ps(_mm_add_ps(_mm_mul_ps(r10, mx), _mm_mul_ps(r11, my)), _mm_mul_ps(r12, mz)));

			_mm_stream_ps((float*)dst + 4, _mm_unpackhi_ps(a, b));
			_mm_stream_ps((float*)dst, _mm_unpacklo_ps(a, b));

			data += 12;
			dst += 4;
		}
		for (i = 0; i < size2; i++)
		{
			const float x = data[0] + tt[0];
			const float y = data[1] + tt[1];
			const float z = data[2] + tt[2];

			const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);

			dst->x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
			dst->y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;

			data += 3;
			dst++;
		}
	}
#endif
	static void myProjectPoint_BF(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
	{
		float r[3][3];
		Mat kr = K*R;

		r[0][0] = (float)kr.at<double>(0, 0);
		r[0][1] = (float)kr.at<double>(0, 1);
		r[0][2] = (float)kr.at<double>(0, 2);

		r[1][0] = (float)kr.at<double>(1, 0);
		r[1][1] = (float)kr.at<double>(1, 1);
		r[1][2] = (float)kr.at<double>(1, 2);

		r[2][0] = (float)kr.at<double>(2, 0);
		r[2][1] = (float)kr.at<double>(2, 1);
		r[2][2] = (float)kr.at<double>(2, 2);

		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		float* data = (float*)xyz.ptr<float>(0);
		Point2f* dst = &dest[0];

		int size2 = xyz.size().area();

		int i;
		for (i = 0; i < size2; i++)
		{
			const float x = data[0] + tt[0];
			const float y = data[1] + tt[1];
			const float z = data[2] + tt[2];

			const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);

			dst->x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
			dst->y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;

			data += 3;
			dst++;
		}
	}


	void moveXYZ(const Mat& xyz, Mat& dest, const Mat& R, const Mat& t)
	{
		float r[3][3];
		Mat kr = R.clone();

		r[0][0] = (float)kr.at<double>(0, 0);
		r[0][1] = (float)kr.at<double>(0, 1);
		r[0][2] = (float)kr.at<double>(0, 2);

		r[1][0] = (float)kr.at<double>(1, 0);
		r[1][1] = (float)kr.at<double>(1, 1);
		r[1][2] = (float)kr.at<double>(1, 2);

		r[2][0] = (float)kr.at<double>(2, 0);
		r[2][1] = (float)kr.at<double>(2, 1);
		r[2][2] = (float)kr.at<double>(2, 2);

		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		float* data = (float*)xyz.ptr<float>(0);
		float* dst = (float*)dest.ptr<float>(0);

		int size2 = xyz.size().area();

		int i;
		for (i = 0; i < size2; i++)
		{
			const float x = data[0] + tt[0];
			const float y = data[1] + tt[1];
			const float z = data[2] + tt[2];

			dst[0] = (r[0][0] * x + r[0][1] * y + r[0][2] * z);
			dst[1] = (r[1][0] * x + r[1][1] * y + r[1][2] * z);
			dst[2] = (r[2][0] * x + r[2][1] * y + r[2][2] * z);
			data += 3;
			dst += 3;
		}
	}

	void projectPointsSimple(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
	{
#ifdef CV_SSE4_1
		myProjectPoint_SSE(xyz, R, t, K, dest);//SSE implimentation
#else
		myProjectPoint_BF(xyz, R, t, K, dest);//normal implementation
#endif
	}

	void projectPointSimple(Point3d& xyz, const Mat& R, const Mat& t, const Mat& K, Point2d& dest)
	{
		float r[3][3];
		Mat kr = K*R;
		r[0][0] = (float)kr.at<double>(0, 0);
		r[0][1] = (float)kr.at<double>(0, 1);
		r[0][2] = (float)kr.at<double>(0, 2);

		r[1][0] = (float)kr.at<double>(1, 0);
		r[1][1] = (float)kr.at<double>(1, 1);
		r[1][2] = (float)kr.at<double>(1, 2);

		r[2][0] = (float)kr.at<double>(2, 0);
		r[2][1] = (float)kr.at<double>(2, 1);
		r[2][2] = (float)kr.at<double>(2, 2);

		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		const float x = (float)xyz.x + tt[0];
		const float y = (float)xyz.y + tt[1];
		const float z = (float)xyz.z + tt[2];

		const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);
		dest.x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
		dest.y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;
	}

	void projectPointsSimple(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, vector<Point2f>& dest)
	{
		float r[3][3];

		r[0][0] = (float)R.at<double>(0, 0);
		r[0][1] = (float)R.at<double>(0, 1);
		r[0][2] = (float)R.at<double>(0, 2);

		r[1][0] = (float)R.at<double>(1, 0);
		r[1][1] = (float)R.at<double>(1, 1);
		r[1][2] = (float)R.at<double>(1, 2);

		r[2][0] = (float)R.at<double>(2, 0);
		r[2][1] = (float)R.at<double>(2, 1);
		r[2][2] = (float)R.at<double>(2, 2);

		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		float* data = (float*)xyz.ptr<float>(0);
		Point2f* dst = &dest[0];

		double fx = K.at<double>(0, 0);
		double fy = K.at<double>(1, 1);
		double cx = K.at<double>(0, 2);
		double cy = K.at<double>(1, 2);
		double k0 = dist.at<double>(0);
		double k1 = dist.at<double>(1);
		double k2 = dist.at<double>(4);

		int size2 = xyz.size().area();

		int i;
		for (i = 0; i < size2; i++)
		{
			const float x = data[0] + tt[0];
			const float y = data[1] + tt[1];
			const float z = data[2] + tt[2];

			const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);

			double X = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
			double Y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;

			double r2 = X*X + Y*Y;
			double r4 = r2*r2;
			double r6 = r4*r2;

			double cdist = 1 + k0*r2 + k1*r4 + k2*r6;

			dst->x = X*cdist*fx + cx;
			dst->y = Y*cdist*fy + cy;

			data += 3;
			dst++;
		}
	}

	void fillSmallHole(const Mat& src, Mat& dest)
	{
		Mat src_;
		if (src.data == dest.data)
			src.copyTo(src_);
		else
			src_ = src;

		uchar* s = (uchar*)src_.ptr<uchar>(1);
		uchar* d = dest.ptr<uchar>(1);
		int step = src.cols * 3;
		for (int j = 1; j < src.rows - 1; j++)
		{
			s += 3, d += 3;
			for (int i = 1; i < src.cols - 1; i++)
			{
				if (s[1] == 0)
				{
					int count = 0;
					int b = 0, g = 0, r = 0;

					int lstep;

					lstep = -step - 3;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}
					lstep = -step;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}
					lstep = -step + 3;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}
					lstep = -3;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}
					lstep = 3;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
				}
					lstep = step - 3;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}
					lstep = step;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}
					lstep = step + 3;
					if (s[lstep + 1 - 1] != 0)
					{
						b += s[lstep + 0];
						g += s[lstep + 1];
						r += s[lstep + 2];
						count++;
					}

					d[0] = (count == 0) ? 0 : (uchar)cvRound((double)b / (double)count);
					d[1] = (count == 0) ? 0 : (uchar)cvRound((double)g / (double)count);
					d[2] = (count == 0) ? 0 : (uchar)cvRound((double)r / (double)count);
			}
				s += 3, d += 3;
		}
			s += 3, d += 3;
	}
}


	void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R_, const Mat& t_, const Mat& K_, const Mat& dist_, Mat& mask, const bool isSub, vector<Point2f>& pt, Mat& depth)
	{
		if (destimage.empty())destimage = Mat::zeros(Size(image.size()), image.type());
		else destimage.setTo(0);

		Mat K, R, t;
		K_.convertTo(K, CV_64F);
		R_.convertTo(R, CV_64F);
		t_.convertTo(t, CV_64F);

		{
#ifdef _CALC_TIME_
			CalcTime t1("depth projection to other viewpoint");
#endif
			if (dist_.empty())
			{
				//no distortionj
				projectPointsSimple(xyz, R, t, K, pt);
			}
			else
			{
				Mat dist; dist_.convertTo(dist, CV_64F);
				projectPointsSimple(xyz, R, t, K, dist, pt);
				//projectPoints(xyz, R, t, K, dist, pt);
			}
		}
#ifdef _CALC_TIME_
		CalcTime tm("rendering");
#endif
		depth.setTo(10000.f);
		Point2f* ptxy = &pt[0];
		float* xyzdata = (float*)xyz.ptr<float>(0);
		uchar* img = (uchar*)image.ptr<uchar>(0);

		float* zbuff;
		const int step1 = image.cols;
		const int step3 = image.cols * 3;
		const int wstep = destimage.cols * 3;

		ptxy += step1;
		xyzdata += step3;
		img += step3;

		for (int j = 1; j < image.rows - 1; j++)
		{
			ptxy++, xyzdata += 3, img += 3;
			for (int i = 1; i < image.cols - 1; i++)
			{
				int x = (int)(ptxy->x);
				int y = (int)(ptxy->y);

				//if(m[i]==255)continue;
				if (x >= 1 && x < image.cols - 1 && y >= 1 && y < image.rows - 1)
				{
					zbuff = depth.ptr<float>(y)+x;
					const float z = xyzdata[2];

					//cout<<format("%d %d %d %d %d %d \n",j,y, (int)ptxy[image.cols].y,i,x,(int)ptxy[1].x);
					//	getchar();
					if (*zbuff > z)
					{
						uchar* dst = destimage.data + wstep*y + 3 * x;
						dst[0] = img[0];
						dst[1] = img[1];
						dst[2] = img[2];
						*zbuff = z;

						if (isSub)
						{
							if ((int)ptxy[image.cols].y - y>1 && (int)ptxy[1].x - x > 1)
							{
								if (zbuff[1] > z)
								{
									dst[3] = img[0];
									dst[4] = img[1];
									dst[5] = img[2];
									zbuff[1] = z;
								}
								if (zbuff[step1 + 1] > z)
								{
									dst[wstep + 0] = img[0];
									dst[wstep + 1] = img[1];
									dst[wstep + 2] = img[2];
									zbuff[step1 + 1] = z;
								}
								if (zbuff[step1] > z)
								{
									dst[wstep + 3] = img[0];
									dst[wstep + 4] = img[1];
									dst[wstep + 5] = img[2];
									zbuff[step1] = z;
								}
							}
							else if ((int)ptxy[1].x - x > 1)
							{
								if (zbuff[1] > z)
								{
									dst[3] = img[0];
									dst[4] = img[1];
									dst[5] = img[2];
									zbuff[1] = z;
								}
							}
							else if ((int)ptxy[image.cols].y - y > 1)
							{
								if (zbuff[step1] > z)
								{
									dst[wstep + 0] = img[0];
									dst[wstep + 1] = img[1];
									dst[wstep + 2] = img[2];
									zbuff[step1] = z;
								}
							}

							if ((int)ptxy[-image.cols].y - y < -1 && (int)ptxy[-1].x - x < -1)
							{
								if (zbuff[-1] > z)
								{
									dst[-3] = img[0];
									dst[-2] = img[1];
									dst[-1] = img[2];
									zbuff[-1] = z;
								}
								if (zbuff[-step1 - 1] > z)
								{
									dst[-wstep + 0] = img[0];
									dst[-wstep + 1] = img[1];
									dst[-wstep + 2] = img[2];
									zbuff[-step1 - 1] = z;
								}
								if (zbuff[-step1] > z)
								{
									dst[-wstep - 3] = img[0];
									dst[-wstep - 2] = img[1];
									dst[-wstep - 1] = img[2];
									zbuff[-step1] = z;
								}
							}
							else if ((int)ptxy[-1].x - x < -1)
							{
								if (zbuff[-1] > z)
								{
									dst[-3] = img[0];
									dst[-2] = img[1];
									dst[-1] = img[2];
									zbuff[-1] = z;
								}
							}
							else if ((int)ptxy[-image.cols].y - y < -1)
							{
								if (zbuff[-step1] > z)
								{
									dst[-wstep + 0] = img[0];
									dst[-wstep + 1] = img[1];
									dst[-wstep + 2] = img[2];
									zbuff[-step1] = z;
								}
							}
						}
					}
				}
				ptxy++, xyzdata += 3, img += 3;
			}
			ptxy++, xyzdata += 3, img += 3;
		}
	}

	void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub)
	{
		vector<Point2f> pt(image.size().area());
		Mat depth = 10000.f*Mat::ones(image.size(), CV_32F);

		projectImagefromXYZ(image, destimage, xyz, R, t, K, dist, mask, isSub, pt, depth);

	}

	template <class T>
	static void reprojectXYZ_(const Mat& depth, Mat& xyz, double f)
	{
		if (xyz.empty())xyz = Mat::zeros(depth.size().area(), 1, CV_32FC3);

		const float bigZ = 10000.f;
		const float fxinv = (float)(1.0 / f);
		const float fyinv = (float)(1.0 / f);
		const float cw = (depth.cols - 1)*0.5f;
		const float ch = (depth.rows - 1)*0.5f;

		T* dep = (T*)depth.ptr<T>(0);
		float* data = xyz.ptr<float>(0);
		//#pragma omp parallel for
		for (int j = 0; j < depth.rows; j++)
		{
			// add 1
			float b = (float)(j - ch + 1);
			const float y = b*fyinv;
			//add 1
			float x = (-cw + 1)*fxinv;
			for (int i = 0; i < depth.cols; i++)
			{
				float z = (float)*dep;
				data[0] = x*z;
				data[1] = y*z;
				data[2] = (z == 0) ? bigZ : z;

				data += 3, dep++;
				x += fxinv;
			}
		}
	}

	void reprojectXYZ(const Mat& depth, Mat& xyz, double f)
	{
		if (depth.type() == CV_8U)
		{
			reprojectXYZ_<uchar>(depth, xyz, f);
		}
		else if (depth.type() == CV_16S)
		{
			reprojectXYZ_<short>(depth, xyz, f);
		}
		else if (depth.type() == CV_16U)
		{
			reprojectXYZ_<unsigned short>(depth, xyz, f);
		}
		else if (depth.type() == CV_32S)
		{
			reprojectXYZ_<int>(depth, xyz, f);
		}
		else if (depth.type() == CV_32F)
		{
			reprojectXYZ_<float>(depth, xyz, f);
		}
		else if (depth.type() == CV_64F)
		{
			reprojectXYZ_<double>(depth, xyz, f);
		}
	}

	//template <class T>
	//void reprojectXYZ(const Mat& depth, Mat& xyz, Mat& intrinsic, Mat& distortion, float a, float b)
	//{
	//	if(xyz.empty())xyz=Mat::zeros(depth.size().area(),1,CV_32FC3);
	//
	//	const float bigZ = 10000.f;
	//	const float fxinv = (float)(1.0/intrinsic.at<double>(0,0));
	//	const float fyinv = (float)(1.0/intrinsic.at<double>(1,1));
	//	const float cw = (float)intrinsic.at<double>(0,2);
	//	const float ch = (float)intrinsic.at<double>(1,2);
	//	const float k0 = (float)distortion.at<double>(0,0);
	//	const float k1 = (float)distortion.at<double>(1,0);
	//	//#pragma omp parallel for
	//	for(int j=0;j<depth.rows;j++)
	//	{
	//		const float y = (j-ch)*fyinv;
	//		const float yy=y*y;
	//		T* dep = (unsigned short*)depth.ptr<T>(j);
	//		float* data=xyz.ptr<float>(j*depth.cols);
	//		for(int i=0;i<depth.cols;i++,dep++,data+=3)
	//		{
	//			const float x = (i-cw)*fxinv;
	//			const float rr = x*x+yy;//r^2
	//
	//			float i2= (k0*rr + k1*rr*rr+1)*i;
	//			float j2= (k0*rr + k1*rr*rr+1)*j;
	//
	//			float z = a* *dep+b;
	//			data[0]=(i2-cw)*fxinv*z;
	//			data[1]=(j2-ch)*fyinv*z;
	//			data[2]= (z==0) ?bigZ:z;
	//		}
	//	}
	//}

	template <class T>
	void reprojectXYZ_(InputArray depth_, OutputArray xyz_, InputArray intrinsic_, InputArray distortion_)
	{
		if (xyz_.empty())xyz_.create(depth_.size().area(), 1, CV_32FC3);
		Mat depth = depth_.getMat();
		Mat xyz = xyz_.getMat();
		Mat intrinsic = intrinsic_.getMat();
		Mat distortion = distortion_.getMat();

		const float bigZ = 100000.f;

		const float fx = (float)(intrinsic.at<double>(0, 0));
		const float fy = (float)(intrinsic.at<double>(1, 1));
		const float fxinv = (float)(1.0 / intrinsic.at<double>(0, 0));
		const float fyinv = (float)(1.0 / intrinsic.at<double>(1, 1));
		const float cw = (float)intrinsic.at<double>(0, 2);
		const float ch = (float)intrinsic.at<double>(1, 2);

		if (distortion_.empty())
		{
			//#pragma omp parallel for
			for (int j = 0; j < depth.rows; j++)
			{
				T* dep = (T*)depth.ptr<T>(j);
				float* data = xyz.ptr<float>(j*depth.cols);
				for (int i = 0; i < depth.cols; i++, dep++, data += 3)
				{
					float z = (float)*dep;
					data[0] = (i - cw + 1)*fxinv*z;
					data[1] = (j - ch + 1)*fyinv*z;
					data[2] = (z == 0) ? bigZ : z;
				}
			}
		}
		else
		{
			CV_Assert(distortion.size() == Size(1, 4) || distortion.size() == Size(4, 1) ||
				distortion.size() == Size(1, 5) || distortion.size() == Size(5, 1) ||
				distortion.size() == Size(1, 8) || distortion.size() == Size(8, 1));

			const float k1 = (float)distortion.at<double>(0);
			const float k2 = (float)distortion.at<double>(1);
			const float k3 = (distortion.size().area()>3) ? (float)distortion.at<double>(4) : 0.f;

			//#pragma omp parallel for
			for (int j = 0; j < depth.rows; j++)
			{
				const float y = (j - ch)*fyinv;
				const float yy = y*y;

				T* dep = (T*)depth.ptr<T>(j);
				float* data = xyz.ptr<float>(j*depth.cols);
				for (int i = 0; i < depth.cols; i++, dep++, data += 3)
				{
					const float x = (i - cw)*fxinv;
					const float rr = x*x + yy;//r^2

					const float kr = 1.f + (k1 + (k2 + k3*rr)*rr)*rr;

					float z = (float)*dep;

					data[0] = x / kr*z;
					data[1] = y / kr*z;
					data[2] = (z == 0) ? bigZ : z;

					/*
					const float x = (i-cw+1)*fxinv;
					const float rr = x*x+yy;//r^2

					const float kr = 1.f+(k1 + (k2 + k3*rr)*rr)*rr;
					float i2= kr*i;
					float j2= kr*j;

					float z = (float)*dep;
					data[0]=(i2-cw+1)*fxinv*z;
					data[1]=(j2-ch+1)*fyinv*z;
					data[2]= (z==0) ?bigZ:z;
					*/
				}
			}
		}
	}

	/*
	else
	{
	CV_Assert( distortion.size() == Size(1, 4) || distortion.size() == Size(4, 1) ||
	distortion.size() == Size(1, 5) || distortion.size() == Size(5, 1) ||
	distortion.size() == Size(1, 8) || distortion.size() == Size(8, 1));

	const float k1 = (float)distortion.at<double>(0);
	const float k2 = (float)distortion.at<double>(1);
	const float k3 =  (distortion.size().area()>3) ?  (float)distortion.at<double>(4):0.f;

	//#pragma omp parallel for
	for(int j=0;j<depth.rows;j++)
	{
	const float v = (j-ch+1);
	const float vv=v*v;

	T* dep = (T*)depth.ptr<T>(j);
	float* data=xyz.ptr<float>(j*depth.cols);
	for(int i=0;i<depth.cols;i++,dep++,data+=3)
	{
	const float u = (i-cw+1);
	const float uu = u*u;

	const float rr = uu+vv;//r^2

	const float kr = 1.f+(k1 + (k2 + k3*rr)*rr)*rr;
	float i2= kr*u+cw;
	float i2= kr*v+ch;
	float j2= (1.f+k1*rr + k2*rr*rr + k3*rr*rr*rr)*j;

	float z = (float)*dep;
	data[0]=(i2-cw+1)*fxinv*z;
	data[1]=(j2-ch+1)*fyinv*z;
	data[2]= (z==0) ?bigZ:z;
	}
	}
	}
	*/
	void reprojectXYZ(InputArray depth, OutputArray xyz, InputArray intrinsic, InputArray distortion)
	{
		if (depth.depth() == CV_8U)
		{
			reprojectXYZ_<uchar>(depth, xyz, intrinsic, distortion);
		}
		else if (depth.depth() == CV_16S)
		{
			reprojectXYZ_<short>(depth, xyz, intrinsic, distortion);
		}
		else if (depth.depth() == CV_16U)
		{
			reprojectXYZ_<unsigned short>(depth, xyz, intrinsic, distortion);
		}
		else if (depth.depth() == CV_32S)
		{
			reprojectXYZ_<int>(depth, xyz, intrinsic, distortion);
		}
		else if (depth.depth() == CV_32F)
		{
			reprojectXYZ_<float>(depth, xyz, intrinsic, distortion);
		}
		else if (depth.depth() == CV_64F)
		{
			reprojectXYZ_<double>(depth, xyz, intrinsic, distortion);
		}
	}

	Point3d get3DPointfromXYZ(Mat& xyz, Size& imsize, Point& pt)
	{
		Point3d ret;
		ret.x = xyz.at<float>(imsize.width * 3 * pt.y + 3 * pt.x + 0);
		ret.y = xyz.at<float>(imsize.width * 3 * pt.y + 3 * pt.x + 1);
		ret.z = xyz.at<float>(imsize.width * 3 * pt.y + 3 * pt.x + 2);

		return ret;
	}



	static void onMouse(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == CV_EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	PointCloudShow::PointCloudShow()
	{
		wname = "Point Cloud Show";
		isInit = false;
	}

	void PointCloudShow::disparity2XYZ(Mat& srcDisparity, float disp_amp, float focal, float baseline)
	{
		Mat dst;
		Mat elemnt = Mat::ones(Size(maxr * 2 + 1, maxr * 2 + 1), CV_8U);
		dilate(srcDisparity, dst, elemnt);

		Mat depthF;
		float fl = focal*baseline*disp_amp;
		if (srcDisparity.depth() == CV_8U)
			disp8U2depth32F(dst, depthF, fl);
		else if (srcDisparity.depth() == CV_16S)
			disp16S2depth32F(dst, depthF, fl);

		//binalyWeightedRangeFilter(depthF,depthF,Size(2*br+1,2*br+1),bth);
		reprojectXYZ(depthF, xyz, focal);
	}

	void PointCloudShow::depth2XYZ(Mat& srcDepth, float focal)
	{
		//filter case
		/*
		Mat dst;
		Mat elemnt = Mat::ones(Size(maxr*2+1,maxr*2+1),CV_8U);
		//mminFilter(srcDepth,dst,Size(maxr*2+1,maxr*2+1));
		//dilate(srcDepth,dst,elemnt);
		erode(srcDepth,dst,elemnt);//depth
		Mat depthF;
		dst.convertTo(depthF,CV_32F);
		binalyWeightedRangeFilter(depthF,depthF,Size(2*br+1,2*br+1),bth);
		*/
		//no filter case
		Mat depthF;
		srcDepth.convertTo(depthF, CV_32F);
		reprojectXYZ(depthF, xyz, focal);
	}

	void PointCloudShow::depth2XYZ(Mat& srcDepth, InputArray K, InputArray Dist)
	{
		Mat depthF;
		srcDepth.convertTo(depthF, CV_32F);
		reprojectXYZ(depthF, xyz, K, Dist);
	}

	void PointCloudShow::warp_xyz(Mat& dest, Mat& image_, InputArray xyz_, InputArray R_, InputArray t_, InputArray k_)
	{
		Mat R = R_.getMat();
		Mat t = t_.getMat();
		Mat k = k_.getMat();

		Mat image;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);

		Mat dshow;

		renderOpt = 2;
		viewSW = 0;

		br = 1;
		bth = 10;
		maxr = 0;

		xyz_.getMat().copyTo(xyz);

		Mat destImage(image.size(), CV_8UC3);	//rendered image

		//project 3D point image
		if (viewSW == 0)//image view
		{
			if (renderOpt > 0)
				projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), true);
			else
				projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), false);
		}
		else//depth map view
		{

			Mat dispC;
			if (viewSW == 1)
				cvtColor(dshow, dispC, CV_GRAY2BGR);
			else
				cv::applyColorMap(dshow, dispC, 2);

			if (renderOpt > 0)
				projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
			else
				projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
		}

		//post filter for rendering image
		if (renderOpt > 2)fillSmallHole(destImage, destImage);

		if (renderOpt > 1)
		{
			Mat gray, mask;
			cvtColor(destImage, gray, CV_BGR2GRAY);
			compare(gray, 0, mask, cv::CMP_EQ);
			medianBlur(destImage, dest, 2 + 1);
			destImage.copyTo(dest, ~mask);
		}
		else destImage.copyTo(dest);
	}

	void PointCloudShow::warp_depth(Mat& dest, Mat& image_, Mat& srcDepth_, InputArray srcK_, InputArray srcDist_, InputArray R_, InputArray t_, InputArray destK_, InputArray destDist_)
	{
		Mat R = R_.getMat();
		Mat t = t_.getMat();
		Mat srcK = srcK_.getMat();
		Mat destK = destK_.getMat();
		Mat srcDist = srcDist_.getMat();
		Mat destDist = destDist_.getMat();

		Mat srcDepth = srcDepth_.clone();
		Mat image;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);

		Mat dshow;

		//the disparity maps is used just for visualization
		Mat disp;
		divide(1.0, srcDepth, disp, srcK.at<double>(0, 0), CV_32F);
		double maxv, minv;
		minMaxLoc(disp, &minv, &maxv);
		disp.convertTo(dshow, CV_8U, 255 / maxv);
		///////////////////////////////////////////

		renderOpt = 1;
		viewSW = 0;

		fillOcclusion(srcDepth, 0);
		Mat tr = srcDepth.t();
		fillOcclusion(tr, 0);
		transpose(tr, srcDepth);

		depth2XYZ(srcDepth, srcK_, srcDist_);

		Mat destImage(image.size(), CV_8UC3);	//rendered image

		//project 3D point image
		if (viewSW == 0)//image view
		{
			if (renderOpt > 0)
				projectImagefromXYZ(image, destImage, xyz, R, t, destK, destDist, Mat(), true);
			else
				projectImagefromXYZ(image, destImage, xyz, R, t, destK, destDist, Mat(), false);
		}
		else//depth map view
		{

			Mat dispC;
			if (viewSW == 1)
				cvtColor(dshow, dispC, CV_GRAY2BGR);
			else
				cv::applyColorMap(dshow, dispC, 2);

			if (renderOpt > 0)
				projectImagefromXYZ(dispC, destImage, xyz, R, t, destK, destDist, Mat(), true);
			else
				projectImagefromXYZ(dispC, destImage, xyz, R, t, destK, destDist, Mat(), false);
		}

		//post filter for rendering image
		if (renderOpt > 2)fillSmallHole(destImage, destImage);

		if (renderOpt > 1)
		{
			Mat gray, mask;
			cvtColor(destImage, gray, CV_BGR2GRAY);
			compare(gray, 0, mask, cv::CMP_EQ);
			medianBlur(destImage, dest, 2 + 1);
			destImage.copyTo(dest, ~mask);
		}
		else destImage.copyTo(dest);
	}

	void PointCloudShow::warp_depth(Mat& dest, Mat& image_, Mat& srcDepth_, float focal, InputArray R_, InputArray t_, InputArray k_)
	{
		Mat R = R_.getMat();
		Mat t = t_.getMat();
		Mat k = k_.getMat();

		cout << R << endl;
		cout << t << endl;
		cout << k << endl;
		Mat srcDepth = srcDepth_.clone();
		Mat image;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);

		Mat dshow;

		//the disparity maps is used just for visualization
		Mat disp;
		divide(1.0, srcDepth, disp, focal, CV_32F);
		double maxv, minv;
		minMaxLoc(disp, &minv, &maxv);
		disp.convertTo(dshow, CV_8U, 255 / maxv);
		///////////////////////////////////////////

		renderOpt = 0;
		viewSW = 0;

		br = 1;
		bth = 10;
		maxr = 0;

		fillOcclusion(srcDepth, 0);
		Mat tr = srcDepth.t();
		fillOcclusion(tr, 0);
		transpose(tr, srcDepth);

		depth2XYZ(srcDepth, focal);

		Mat destImage(image.size(), CV_8UC3);	//rendered image

		if (k.empty())
		{
			k = Mat::eye(3, 3, CV_64F)*focal;
			k.at<double>(0, 2) = (image.cols - 1)*0.5;
			k.at<double>(1, 2) = (image.rows - 1)*0.5;
			k.at<double>(2, 2) = 1.0;
		}

		//project 3D point image
		if (viewSW == 0)//image view
		{
			if (renderOpt > 0)
				projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), true);
			else
				projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), false);
		}
		else//depth map view
		{

			Mat dispC;
			if (viewSW == 1)
				cvtColor(dshow, dispC, CV_GRAY2BGR);
			else
				cv::applyColorMap(dshow, dispC, 2);

			if (renderOpt > 0)
				projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
			else
				projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
		}

		//post filter for rendering image
		if (renderOpt > 2)fillSmallHole(destImage, destImage);

		if (renderOpt > 1)
		{
			Mat gray, mask;
			cvtColor(destImage, gray, CV_BGR2GRAY);
			compare(gray, 0, mask, cv::CMP_EQ);
			medianBlur(destImage, dest, 2 + 1);
			destImage.copyTo(dest, ~mask);
		}
		else destImage.copyTo(dest);
	}

	void PointCloudShow::warp_disparity(Mat& dest, Mat& image_, Mat& srcDisparity_, float disp_amp, float focal, float baseline, Mat& R, Mat& t, Mat& k)
	{
		Mat srcDisparity = srcDisparity_.clone();
		Mat image;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);

		Mat dshow;
		srcDisparity.convertTo(dshow, CV_8U, 1 / disp_amp);

		renderOpt = 2;
		viewSW = 0;

		br = 5;
		bth = 10;
		maxr = 1;

		fillOcclusion(srcDisparity, 0);
		Mat tr = srcDisparity.t();
		fillOcclusion(tr, 0);
		transpose(tr, srcDisparity);

		disparity2XYZ(srcDisparity, disp_amp, focal, baseline);

		Mat destImage(image.size(), CV_8UC3);	//rendered image

		if (k.empty())
		{
			k = Mat::eye(3, 3, CV_64F)*focal;
			k.at<double>(0, 2) = (image.cols - 1)*0.5;
			k.at<double>(1, 2) = (image.rows - 1)*0.5;
			k.at<double>(2, 2) = 1.0;
		}

		//project 3D point image
		if (viewSW == 0)//image view
		{
			if (renderOpt > 0)
				projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), true);
			else
				projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), false);
		}
		else//depth map view
		{

			Mat dispC;
			if (viewSW == 1)
				cvtColor(dshow, dispC, CV_GRAY2BGR);
			else
				cv::applyColorMap(dshow, dispC, 2);

			if (renderOpt > 0)
				projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
			else
				projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
		}

		//post filter for rendering image
		if (renderOpt > 2)fillSmallHole(destImage, destImage);

		if (renderOpt > 1)
		{
			Mat gray, mask;
			cvtColor(destImage, gray, CV_BGR2GRAY);
			compare(gray, 0, mask, cv::CMP_EQ);
			medianBlur(destImage, dest, 2 + 1);
			destImage.copyTo(dest, ~mask);
		}
		else destImage.copyTo(dest);
	}

	void PointCloudShow::loop_depth(Mat& image_, Mat& srcDepth_, float focal, int loopcount)
	{
		Mat srcDepth = srcDepth_.clone();
		Mat image;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);

		Mat dshow;

		Mat disp;
		divide(1.0, srcDepth, disp, focal, CV_32F);
		double maxv, minv;
		minMaxLoc(disp, &minv, &maxv);
		disp.convertTo(dshow, CV_8U, 255 / maxv);

		namedWindow(wname);
		int baseline = 1000;
		const int xmax = baseline * 16;
		const int ymax = baseline * 16;
		const int initx = xmax / 2;
		const int inity = ymax / 2;
		const int initz = baseline * 16;
		const int initpitch = 90;
		const int inityaw = 90;

		if (!isInit)
		{
			pt = Point((image.cols - 1) / 2, (image.rows - 1) / 2);
			x = initx;
			y = initx;
			z = initz;

			pitch = initpitch;
			yaw = inityaw;

			loolatx = (image.cols - 1) / 2;
			loolaty = (image.rows - 1) / 2;

			renderOpt = 2;
			viewSW = 0;

			br = 5;
			bth = 10;
			maxr = 1;

			isDrawLine = true;
			isWrite = false;
			isLookat = false;

			fillOcclusion(srcDepth, 0);
			Mat tr = srcDepth.t();
			fillOcclusion(tr, 0);
			transpose(tr, srcDepth);

			depth2XYZ(srcDepth, focal);
			look = get3DPointfromXYZ(xyz, image.size(), pt);

			isInit = true;
		}

		cv::setMouseCallback(wname, (MouseCallback)onMouse, (void*)&pt);

		createTrackbar("x", wname, &x, xmax);
		createTrackbar("y", wname, &y, ymax);
		createTrackbar("z", wname, &z, 8000);
		createTrackbar("pitch", wname, &pitch, 180);
		createTrackbar("yaw", wname, &yaw, 180);
		createTrackbar("look at x", wname, &loolatx, image.cols - 1);
		createTrackbar("look at y", wname, &loolaty, image.rows - 1);

		createTrackbar("render Opt", wname, &renderOpt, 3);
		createTrackbar("sw", wname, &viewSW, 2);

		createTrackbar("brad", wname, &br, 20);
		createTrackbar("bth", wname, &bth, 200);
		createTrackbar("maxr", wname, &maxr, 5);


		Mat destImage(image.size(), CV_8UC3);	//rendered image
		Mat view;//drawed image

		Mat k = Mat::eye(3, 3, CV_64F)*focal;
		k.at<double>(0, 2) = (image.cols - 1)*0.5;
		k.at<double>(1, 2) = (image.rows - 1)*0.5;
		k.at<double>(2, 2) = 1.0;


		fillOcclusion(srcDepth, 0);
		Mat tr = srcDepth.t();
		fillOcclusion(tr, 0);
		transpose(tr, srcDepth);

		depth2XYZ(srcDepth, focal);

		int count = 0;
		CalcTime tm("total");
		int key = 0;

		int num_loop = 0;
		while (key != 'q')
		{
			double bps = 0.0;
			//from mouse input
			x = (int)(xmax*(double)pt.x / (double)(image.cols - 1) + 0.5);
			y = (int)(ymax*(double)pt.y / (double)(image.rows - 1) + 0.5);
			setTrackbarPos("x", wname, x);
			setTrackbarPos("y", wname, y);

			tm.start();

			//{
			//	//CalcTime t(" depth2xyz projection");
			//	reprojectXYZ(depthF,xyz,focal_length);
			//}

			Mat R = Mat::eye(3, 3, CV_64F);
			Mat t = Mat::zeros(3, 1, CV_64F);
			t.at<double>(0, 0) = x - initx;
			t.at<double>(1, 0) = y - inity;
			t.at<double>(2, 0) = -z + initz;

			Point3d srcview = Point3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
			if (isLookat)
			{
				look = get3DPointfromXYZ(xyz, image.size(), Point(loolatx, loolaty));
			}
			lookat(look, srcview, R);
			Mat r;
			eular2rot(pitch - 90.0, 0.0, yaw - 90, r);
			R = r*R;

			//project 3D point image
			if (viewSW == 0)//image view
			{
				if (renderOpt > 0)
					projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}
			else//depth map view
			{

				Mat dispC;
				if (viewSW == 1)
					cvtColor(dshow, dispC, CV_GRAY2BGR);
				else
					cv::applyColorMap(dshow, dispC, 2);

				if (renderOpt > 0)
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}

			//post filter for rendering image
			if (renderOpt > 2)fillSmallHole(destImage, destImage);
			if (renderOpt > 1)
			{
				Mat gray, mask;
				cvtColor(destImage, gray, CV_BGR2GRAY);
				compare(gray, 0, mask, cv::CMP_EQ);
				medianBlur(destImage, view, 2 + 1);
				destImage.copyTo(view, ~mask);
			}
			else destImage.copyTo(view);

			if (isWrite)imwrite(format("out%4d.jpg", count++), view);

			if (isDrawLine)
			{
				Point2d ptf;
				projectPointSimple(look, R, t, k, ptf);
				circle(view, Point(ptf), 7, CV_RGB(0, 255, 0), CV_FILLED);
				line(view, Point(0, image.rows / 2), Point(image.cols, image.rows / 2), CV_RGB(255, 0, 0));
				line(view, Point(image.cols / 2, 0), Point(image.cols / 2, image.rows), CV_RGB(255, 0, 0));
			}
			double fps = 1000.0 / tm.getTime();
			putText(view, format("%.02f fps", fps), Point(30, 30), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));

			if (isLookat)
				putText(view, format("Look at: Free", bps), Point(30, 60), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));
			else
				putText(view, format("Look at: Fix", bps), Point(30, 60), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));

			//show image
			imshow(wname, view);
			key = waitKey(1);

			if (key == 'o')
			{
				renderOpt++;
				if (renderOpt > 3)renderOpt = 0;

			}
			if (key == 'g')
			{
				isDrawLine = isDrawLine ? false : true;
			}
			if (key == 'l')
			{
				isLookat = (isLookat) ? false : true;
			}
			if (key == 's')
			{
				isWrite = true;
			}
			if (key == 'd')
			{
				depth2XYZ(srcDepth, focal);
			}
			if (key == 'r')
			{
				pt.x = image.cols / 2;
				pt.y = image.rows / 2;

				z = initz;
				pitch = initpitch;
				yaw = inityaw;

				setTrackbarPos("z", wname, z);
				setTrackbarPos("pitch", wname, pitch);
				setTrackbarPos("yaw", wname, yaw);
			}

			if (key == 'v')
			{
				viewSW++;
				if (viewSW > 2) viewSW = 0;
				setTrackbarPos("sw", wname, viewSW);

			}
			num_loop++;
			if (loopcount > 0 && num_loop > loopcount) break;
		}
	}

	void PointCloudShow::loop_depth(Mat& image_, Mat& srcDepth_, Mat& image2_, Mat& srcDepth2_, float focal, Mat& oR, Mat& ot, Mat& K, int loopcount)
	{
		Mat srcDepth = srcDepth_.clone();
		Mat srcDepth2 = srcDepth2_.clone();
		Mat image;
		Mat image2;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);
		if (image2_.channels() == 3)image2 = image2_;
		else cvtColor(image2_, image2, CV_GRAY2BGR);

		Mat dshow, dshow2;

		Mat disp;
		divide(1.0, srcDepth, disp, focal, CV_32F);
		double maxv, minv;
		minMaxLoc(disp, &minv, &maxv);
		disp.convertTo(dshow, CV_8U, 255 / maxv);

		disp;
		divide(1.0, srcDepth2, disp, focal, CV_32F);
		minMaxLoc(disp, &minv, &maxv);
		disp.convertTo(dshow2, CV_8U, 255 / maxv);
		//guiAlphaBlend(dshow,dshow2);

		namedWindow(wname);
		moveWindow(wname, 0, 0);
		int baseline = 1000;
		const int xmax = baseline * 16;
		const int ymax = baseline * 16;
		const int initx = xmax / 2;
		const int inity = ymax / 2;
		const int initz = baseline * 16;
		const int initpitch = 90;
		const int inityaw = 90;

		if (!isInit)
		{
			pt = Point((image.cols - 1) / 2, (image.rows - 1) / 2);
			x = initx;
			y = initx;
			z = initz;

			pitch = initpitch;
			yaw = inityaw;

			loolatx = (image.cols - 1) / 2;
			loolaty = (image.rows - 1) / 2;

			renderOpt = 2;
			viewSW = 0;

			br = 5;
			bth = 10;
			maxr = 1;

			isDrawLine = true;
			isWrite = false;
			isLookat = false;

			fillOcclusion(srcDepth, 0, FILL_DEPTH);
			Mat tr = srcDepth.t();
			fillOcclusion(tr, 0, FILL_DEPTH);
			transpose(tr, srcDepth);

			depth2XYZ(srcDepth, focal);
			look = get3DPointfromXYZ(xyz, image.size(), pt);

			isInit = true;
		}

		cv::setMouseCallback(wname, (MouseCallback)onMouse, (void*)&pt);

		string wname2 = "pos";
		namedWindow(wname2);
		createTrackbar("x", wname2, &x, xmax);
		createTrackbar("y", wname2, &y, ymax);
		createTrackbar("z", wname2, &z, 80000);
		createTrackbar("pitch", wname2, &pitch, 180);
		createTrackbar("yaw", wname2, &yaw, 180);
		createTrackbar("look at x", wname2, &loolatx, image.cols - 1);
		createTrackbar("look at y", wname2, &loolaty, image.rows - 1);

		int va = 50; createTrackbar("alpha", wname, &va, 100);
		createTrackbar("render Opt", wname, &renderOpt, 3);
		createTrackbar("sw", wname, &viewSW, 2);

		createTrackbar("brad", wname, &br, 20);
		createTrackbar("bth", wname, &bth, 200);
		createTrackbar("maxr", wname, &maxr, 5);


		Mat destImage(image.size(), CV_8UC3);	//rendered image
		Mat view;//drawed image

		Mat k = Mat::eye(3, 3, CV_64F)*focal;
		k.at<double>(0, 2) = (image.cols - 1)*0.5;
		k.at<double>(1, 2) = (image.rows - 1)*0.5;
		k.at<double>(2, 2) = 1.0;

		fillOcclusion(srcDepth, 0, FILL_DEPTH);
		Mat tr = srcDepth.t();
		fillOcclusion(tr, 0, FILL_DEPTH);
		transpose(tr, srcDepth);

		depth2XYZ(srcDepth, focal);

		int count = 0;
		CalcTime tm("total");
		int key = 0;

		int num_loop = 0;
		while (key != 'q')
		{
			double bps;
			//from mouse input
			x = (int)(xmax*(double)pt.x / (double)(image.cols - 1) + 0.5);
			y = (int)(ymax*(double)pt.y / (double)(image.rows - 1) + 0.5);
			setTrackbarPos("x", wname2, x);
			setTrackbarPos("y", wname2, y);

			tm.start();

			//{
			//	//CalcTime t(" depth2xyz projection");
			//	reprojectXYZ(depthF,xyz,focal_length);
			//}

			Mat R = Mat::eye(3, 3, CV_64F);
			Mat t = Mat::zeros(3, 1, CV_64F);
			t.at<double>(0, 0) = x - initx;
			t.at<double>(1, 0) = y - inity;
			t.at<double>(2, 0) = -z + initz;

			Point3d srcview = Point3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
			if (isLookat)
			{
				look = get3DPointfromXYZ(xyz, image.size(), Point(loolatx, loolaty));
			}
			lookat(look, srcview, R);
			Mat r;
			eular2rot(pitch - 90.0, 0.0, yaw - 90, r);
			R = r*R;

			//*********************************** left

			depth2XYZ(srcDepth, focal);
			//project 3D point image
			if (viewSW == 0)//image view
			{
				if (renderOpt > 0)
					projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}
			else//depth map view
			{

				Mat dispC;
				if (viewSW == 1)
					cvtColor(dshow, dispC, CV_GRAY2BGR);
				else
					cv::applyColorMap(dshow, dispC, 2);

				if (renderOpt > 0)
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}

			//post filter for rendering image
			if (renderOpt > 2)fillSmallHole(destImage, destImage);
			if (renderOpt > 1)
			{
				Mat gray, mask;
				cvtColor(destImage, gray, CV_BGR2GRAY);
				compare(gray, 0, mask, cv::CMP_EQ);
				medianBlur(destImage, view, 2 + 1);
				destImage.copyTo(view, ~mask);
			}
			else destImage.copyTo(view);

			Mat view2 = view.clone();
			//*********************************** left
			/*t= (t-oR.t()*ot);
			R = R*oR.t();*/
			depth2XYZ(srcDepth2, focal);
			//moveXYZ(xyz,xyz,oR,Mat(-ot));
			moveXYZ(xyz, xyz, Mat(oR.t()), Mat(-oR.t()*ot));

			//moveXYZ(xyz,xyz,Mat::eye(3,3,CV_64F),Mat(-ot));

			//project 3D point image
			if (viewSW == 0)//image view
			{
				if (renderOpt > 0)
					projectImagefromXYZ(image2, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(image2, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}
			else//depth map view
			{

				Mat dispC;
				if (viewSW == 1)
					cvtColor(dshow2, dispC, CV_GRAY2BGR);
				else
					cv::applyColorMap(dshow2, dispC, 2);

				if (renderOpt > 0)
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}

			//post filter for rendering image
			if (renderOpt > 2)fillSmallHole(destImage, destImage);
			if (renderOpt > 1)
			{
				Mat gray, mask;
				cvtColor(destImage, gray, CV_BGR2GRAY);
				compare(gray, 0, mask, cv::CMP_EQ);
				medianBlur(destImage, view, 2 + 1);
				destImage.copyTo(view, ~mask);
			}
			else destImage.copyTo(view);

			if (isWrite)imwrite(format("out%4d.jpg", count++), view);

			if (isDrawLine)
			{
				Point2d ptf;
				projectPointSimple(look, R, t, k, ptf);
				circle(view, Point(ptf), 7, CV_RGB(0, 255, 0), CV_FILLED);
				line(view, Point(0, image.rows / 2), Point(image.cols, image.rows / 2), CV_RGB(255, 0, 0));
				line(view, Point(image.cols / 2, 0), Point(image.cols / 2, image.rows), CV_RGB(255, 0, 0));
			}
			double fps = 1000.0 / tm.getTime();
			putText(view, format("%.02f fps", fps), Point(30, 30), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));

			if (isLookat)
				putText(view, format("Look at: Free", bps), Point(30, 60), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));
			else
				putText(view, format("Look at: Fix", bps), Point(30, 60), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));

			//show image
			Mat msk, msk2;
			compare(view, Scalar(0), msk, CMP_EQ);
			compare(view2, Scalar(0), msk2, CMP_EQ);

			Mat view3;
			alphaBlend(view, view2, va / 100.0, view3);

			view2.copyTo(view3, msk);
			view.copyTo(view3, msk2);

			if (va == 0)view.copyTo(view3);
			if (va == 100)view2.copyTo(view3);
			imshow(wname, view3);
			key = waitKey(1);


			if (key == 'o')
			{
				renderOpt++;
				if (renderOpt > 3)renderOpt = 0;

			}
			if (key == 'g')
			{
				isDrawLine = isDrawLine ? false : true;
			}
			if (key == 'l')
			{
				isLookat = (isLookat) ? false : true;
			}
			if (key == 's')
			{
				isWrite = true;
			}
			if (key == 'd')
			{
				depth2XYZ(srcDepth, focal);
			}
			if (key == 'r')
			{
				pt.x = image.cols / 2;
				pt.y = image.rows / 2;

				z = initz;
				pitch = initpitch;
				yaw = inityaw;

				setTrackbarPos("z", wname, z);
				setTrackbarPos("pitch", wname, pitch);
				setTrackbarPos("yaw", wname, yaw);
			}

			if (key == 'v')
			{
				viewSW++;
				if (viewSW > 2) viewSW = 0;
				setTrackbarPos("sw", wname, viewSW);

			}
			num_loop++;
			if (loopcount > 0 && num_loop > loopcount) break;
		}
	}

	void PointCloudShow::loop(Mat& image_, Mat& srcDisparity_, float disp_amp, float focal, float baseline, int loopcount)
	{
		Mat srcDisparity = srcDisparity_.clone();
		Mat image;
		if (image_.channels() == 3)image = image_;
		else cvtColor(image_, image, CV_GRAY2BGR);

		Mat dshow;
		srcDisparity.convertTo(dshow, CV_8U, 1 / disp_amp);
		namedWindow(wname);

		const int xmax = (int)(baseline * 16);
		const int ymax = (int)(baseline * 16);
		const int initx = xmax / 2;
		const int inity = ymax / 2;
		const int initz = (int)(baseline * 16);
		const int initpitch = 90;
		const int inityaw = 90;

		if (!isInit)
		{
			pt = Point((image.cols - 1) / 2, (image.rows - 1) / 2);
			x = initx;
			y = initx;
			z = initz;

			pitch = initpitch;
			yaw = inityaw;

			loolatx = (image.cols - 1) / 2;
			loolaty = (image.rows - 1) / 2;

			renderOpt = 2;
			viewSW = 0;

			br = 5;
			bth = 10;
			maxr = 1;

			isDrawLine = true;
			isWrite = false;
			isLookat = false;

			fillOcclusion(srcDisparity, 0);
			Mat tr = srcDisparity.t();
			fillOcclusion(tr, 0);
			transpose(tr, srcDisparity);

			disparity2XYZ(srcDisparity, disp_amp, focal, baseline);

			look = get3DPointfromXYZ(xyz, image.size(), pt);

			isInit = true;
		}

		cv::setMouseCallback(wname, (MouseCallback)onMouse, (void*)&pt);

		createTrackbar("x", wname, &x, xmax);
		createTrackbar("y", wname, &y, ymax);
		createTrackbar("z", wname, &z, 8000);
		createTrackbar("pitch", wname, &pitch, 180);
		createTrackbar("yaw", wname, &yaw, 180);
		createTrackbar("look at x", wname, &loolatx, image.cols - 1);
		createTrackbar("look at y", wname, &loolaty, image.rows - 1);

		createTrackbar("render Opt", wname, &renderOpt, 3);
		createTrackbar("sw", wname, &viewSW, 2);

		createTrackbar("brad", wname, &br, 20);
		createTrackbar("bth", wname, &bth, 200);
		createTrackbar("maxr", wname, &maxr, 5);


		Mat destImage(image.size(), CV_8UC3);	//rendered image
		Mat view;//drawed image

		Mat k = Mat::eye(3, 3, CV_64F)*focal;
		k.at<double>(0, 2) = (image.cols - 1)*0.5;
		k.at<double>(1, 2) = (image.rows - 1)*0.5;
		k.at<double>(2, 2) = 1.0;

		fillOcclusion(srcDisparity, 0);
		Mat tr = srcDisparity.t();
		fillOcclusion(tr, 0);
		transpose(tr, srcDisparity);

		disparity2XYZ(srcDisparity, disp_amp, focal, baseline);

		int count = 0;
		CalcTime tm("total");
		int key = 0;

		int num_loop = 0;

		while (key != 'q')
		{
			double bps=0.0;
			//from mouse input
			x = (int)(xmax*(double)pt.x / (double)(image.cols - 1) + 0.5);
			y = (int)(ymax*(double)pt.y / (double)(image.rows - 1) + 0.5);
			setTrackbarPos("x", wname, x);
			setTrackbarPos("y", wname, y);

			tm.start();

			//{
			//	//CalcTime t(" depth2xyz projection");
			//	reprojectXYZ(depthF,xyz,focal_length);
			//}

			Mat R = Mat::eye(3, 3, CV_64F);
			Mat t = Mat::zeros(3, 1, CV_64F);
			t.at<double>(0, 0) = x - initx;
			t.at<double>(1, 0) = y - inity;
			t.at<double>(2, 0) = -z + initz;

			Point3d srcview = Point3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
			if (isLookat)
			{
				look = get3DPointfromXYZ(xyz, image.size(), Point(loolatx, loolaty));
			}
			lookat(look, srcview, R);
			Mat r;
			eular2rot(pitch - 90.0, 0.0, yaw - 90, r);
			R = r*R;

			//project 3D point image
			if (viewSW == 0)//image view
			{
				if (renderOpt > 0)
					projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(image, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}
			else//depth map view
			{
				Mat dispC;
				if (viewSW == 1)
					cvtColor(dshow, dispC, CV_GRAY2BGR);
				else
					cv::applyColorMap(dshow, dispC, 2);

				if (renderOpt > 0)
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), true);
				else
					projectImagefromXYZ(dispC, destImage, xyz, R, t, k, Mat(), Mat(), false);
			}

			//post filter for rendering image
			if (renderOpt > 2)fillSmallHole(destImage, destImage);
			if (renderOpt > 1)
			{
				Mat gray, mask;
				cvtColor(destImage, gray, CV_BGR2GRAY);
				compare(gray, 0, mask, cv::CMP_EQ);
				medianBlur(destImage, view, 2 + 1);
				destImage.copyTo(view, ~mask);
			}
			else destImage.copyTo(view);

			if (isWrite)imwrite(format("out%4d.jpg", count++), view);

			if (isDrawLine)
			{
				Point2d ptf;
				projectPointSimple(look, R, t, k, ptf);
				circle(view, Point(ptf), 7, CV_RGB(0, 255, 0), CV_FILLED);
				line(view, Point(0, image.rows / 2), Point(image.cols, image.rows / 2), CV_RGB(255, 0, 0));
				line(view, Point(image.cols / 2, 0), Point(image.cols / 2, image.rows), CV_RGB(255, 0, 0));
			}
			double fps = 1000.0 / tm.getTime();
			putText(view, format("%.02f fps", fps), Point(30, 30), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));

			if (isLookat)
				putText(view, format("Look at: Free", bps), Point(30, 60), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));
			else
				putText(view, format("Look at: Fix", bps), Point(30, 60), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));

			//show image
			imshow(wname, view);
			key = waitKey(1);

			if (key == 'o')
			{
				renderOpt++;
				if (renderOpt > 3)renderOpt = 0;

			}
			if (key == 'g')
			{
				isDrawLine = isDrawLine ? false : true;
			}
			if (key == 'l')
			{
				isLookat = (isLookat) ? false : true;
			}
			if (key == 's')
			{
				isWrite = true;
			}
			if (key == 'd')
			{
				disparity2XYZ(srcDisparity, disp_amp, focal, baseline);
			}
			if (key == 'r')
			{
				pt.x = image.cols / 2;
				pt.y = image.rows / 2;

				z = initz;
				pitch = initpitch;
				yaw = inityaw;

				setTrackbarPos("z", wname, z);
				setTrackbarPos("pitch", wname, pitch);
				setTrackbarPos("yaw", wname, yaw);
			}

			if (key == 'v')
			{
				viewSW++;
				if (viewSW > 2) viewSW = 0;
				setTrackbarPos("sw", wname, viewSW);

			}
			num_loop++;
			if (loopcount > 0 && num_loop > loopcount) break;
		}
	}

}