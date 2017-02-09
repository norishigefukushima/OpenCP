#include "opticalFlow.hpp"
#include "minmaxfilter.hpp"
#include "shiftImage.hpp"
#include "alphaBlend.hpp"
#include <opencv2/optflow.hpp>

using namespace std;
using namespace cv;

namespace cp
{

	inline bool isFlowCorrect(Point2f u)
	{
		return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
	}

	static Vec3b computeColor(float fx, float fy)
	{
		static bool first = true;

		// relative lengths of color transitions:
		// these are chosen based on perceptual similarity
		// (e.g. one can distinguish more shades between red and yellow
		//  than between yellow and green)
		const int RY = 15;
		const int YG = 6;
		const int GC = 4;
		const int CB = 11;
		const int BM = 13;
		const int MR = 6;
		const int NCOLS = RY + YG + GC + CB + BM + MR;
		static Vec3i colorWheel[NCOLS];

		if (first)
		{
			int k = 0;

			for (int i = 0; i < RY; ++i, ++k)
				colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

			for (int i = 0; i < YG; ++i, ++k)
				colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

			for (int i = 0; i < GC; ++i, ++k)
				colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

			for (int i = 0; i < CB; ++i, ++k)
				colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

			for (int i = 0; i < BM; ++i, ++k)
				colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

			for (int i = 0; i < MR; ++i, ++k)
				colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

			first = false;
		}

		const float rad = sqrt(fx * fx + fy * fy);
		const float a = atan2(-fy, -fx) / (float)CV_PI;

		const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
		const int k0 = static_cast<int>(fk);
		const int k1 = (k0 + 1) % NCOLS;
		const float f = fk - k0;

		Vec3b pix;

		for (int b = 0; b < 3; b++)
		{
			const float col0 = colorWheel[k0][b] / 255.f;
			const float col1 = colorWheel[k1][b] / 255.f;

			float col = (1 - f) * col0 + f * col1;

			if (rad <= 1)
				col = 1 - rad * (1 - col); // increase saturation with radius
			else
				col *= .75; // out of range

			pix[2 - b] = static_cast<uchar>(255.f * col);
		}

		return pix;
	}

	void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion)
	{
		dst.create(flow.size(), CV_8UC3);
		dst.setTo(Scalar::all(0));

		// determine motion range:
		float maxrad = maxmotion;

		if (maxmotion <= 0)
		{
			maxrad = 1.f;
			for (int y = 0; y < flow.rows; ++y)
			{
				for (int x = 0; x < flow.cols; ++x)
				{
					Point2f u = flow(y, x);

					if (!isFlowCorrect(u))
						continue;

					maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
				}
			}
		}

		for (int y = 0; y < flow.rows; ++y)
		{
			for (int x = 0; x < flow.cols; ++x)
			{
				Point2f u = flow(y, x);

				if (isFlowCorrect(u))
					dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
			}
		}
	}

	void divideFlow(Mat& flow, Mat& xflow, Mat& yflow)
	{
		vector<Mat> div;

		split(flow, div);
		vector<Mat>::const_iterator it = div.begin();
		Mat xtemp, ytemp;
		xtemp = *it++;
		ytemp = *it;
		xflow = xtemp.clone();
		yflow = ytemp.clone();
	}

	void mergeFlow(Mat& flow, Mat& xflow, Mat& yflow)
	{
		if (flow.empty()) flow.create(xflow.rows, xflow.cols, CV_32FC2);
		vector<Mat> merge;

		merge.push_back(xflow);
		merge.push_back(yflow);
		Mat mixed(flow.size(), flow.type());
		int fromTo[] = { 0, 0, 1, 1 };

		vector<Mat> mix;
		mix.push_back(mixed);

		mixChannels(merge, mix, fromTo, 2);

		flow = mixed.clone();
	}

	void readFlowFile(const char *filename, Mat& img)
	{
		FILE *stream = fopen(filename, "rb");

		int width, height;
		float tag;

		if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
			(int)fread(&width, sizeof(int), 1, stream) != 1 ||
			(int)fread(&height, sizeof(int), 1, stream) != 1)
			cout << "ReadFlowFile: problem reading file " << filename << endl;

		img.create(height, width, CV_32FC2);
		int nBands = 2;
		int n = nBands * width * height;

		float* ptr = new float[n];
		if ((int)fread(ptr, sizeof(float), n, stream) != n)
			cout << "ReadFlowFile" << filename << " : file is too short" << endl;

		if (fgetc(stream) != EOF)
			cout << "ReadFlowFile" << filename << ": file is too long" << endl;

		Mat flow(height, width, CV_32FC2, ptr);
		img = flow.clone();

		fclose(stream);
	}

	void XYtoPC(Mat& xflow, Mat& yflow, Mat& dist, Mat& theta)
	{
		if (dist.empty()) dist.create(xflow.rows, xflow.cols, CV_32F);
		if (theta.empty()) theta.create(xflow.rows, xflow.cols, CV_32F);

		Mat xsqr = xflow.mul(xflow);
		Mat ysqr = yflow.mul(yflow);
		Mat tan = yflow / xflow;

		sqrt((xsqr + ysqr), dist);

		for (int y = 0; y < xflow.rows; y++) {
			for (int x = 0; x < xflow.cols; x++) {
				theta.at<float>(y, x) = atan(tan.at<float>(y, x));
				if (xflow.at<float>(y, x) < 0) theta.at<float>(y, x) += CV_PI;
			}
		}
	}

	void PCtoXY(Mat& xflow, Mat& yflow, Mat& dist, Mat& theta)
	{
		if (xflow.empty()) xflow.create(dist.rows, dist.cols, CV_32F);
		if (yflow.empty()) yflow.create(dist.rows, dist.cols, CV_32F);

		for (int y = 0; y < xflow.rows; y++) {
			for (int x = 0; x < xflow.cols; x++) {
				xflow.at<float>(y, x) = dist.at<float>(y, x)*cos(theta.at<float>(y, x));
				yflow.at<float>(y, x) = dist.at<float>(y, x)*sin(theta.at<float>(y, x));
			}
		}
	}

	/*
	void CVR(CostVolumeRefinement2 cvr, Mat& src, Mat& guide, Mat& weight, Mat& dst,
	int dtrunc, int metric, int r, int Filter, int iter_cvr, int sig_s, int sig_c, int sig_d, int eps, int amp)
	{
	double min, max;

	Mat in = src.clone();
	in *= amp;
	minMaxLoc(in, &min, &max);
	in -= min;

	//	cvr.chooseFilter(in, guide, weight, dst, dtrunc, metric, r, Filter, iter_cvr, sig_s, sig_c, sig_d, eps*0.01);
	dst.convertTo(dst, CV_32F, 1.0 / 16.0);

	dst += min;
	dst /= amp;
	}
	*/

	void OpticalFlowEval(Mat& flow, Mat& gt, double RA, double RE, double& AE, double& EE)
	{
		Mat xflow, yflow, xgt, ygt;
		divideFlow(flow, xflow, yflow);
		divideFlow(gt, xgt, ygt);

		Mat flowL; sqrt((1 + xflow.mul(xflow) + yflow.mul(yflow)), flowL);
		Mat gtL; sqrt((1 + xgt.mul(xgt) + ygt.mul(ygt)), gtL);
		Mat Dots = 1 + xflow.mul(xgt) + yflow.mul(ygt);
		Mat subx; absdiff(xflow, xgt, subx);
		Mat suby; absdiff(yflow, ygt, suby);
		Mat sqEE = subx.mul(subx) + suby.mul(suby);

		Mat Amask = Mat::zeros(flow.size(), CV_8U);
		Mat Emask = Mat::zeros(flow.size(), CV_8U);
		for (int y = 0; y < flow.rows; y++) {
			for (int x = 0; x < flow.cols; x++) {
				double L = (double)(flowL.at<float>(y, x)*gtL.at<float>(y, x));
				double arc = (double)((Dots.at<float>(y, x)));
				double angle = (arc / L < 1 && arc / L > -1) ? acos((arc / L))*(180.0 / CV_PI) : 0;
				double endpt = sqrt((double)sqEE.at<float>(y, x));

				if (RA == 0.0) {
					AE += angle;
				}
				else if (angle > RA) {
					AE += 100;
					Amask.at<uchar>(y, x) = 255;
				}
				if (RE == 0.0) {
					EE += endpt;
				}
				else if (endpt > RE) {
					EE += 100;
					Emask.at<uchar>(y, x) = 255;
				}
			}
			//if((y%10) == 0) cout << "y : " << y << ", EE : " << EE << endl;
		}
		AE /= flow.rows*flow.cols;
		EE /= flow.rows*flow.cols;
	}


	OpticalFlowBM::OpticalFlowBM()
	{
		;
	}
	void OpticalFlowBM::cncheck(Mat& srcx, Mat& srcy, Mat& destx, Mat& desty, int thresh, int invalid)
	{
		short inv = invalid;
		for (int j = 0; j < srcx.rows; j++)
		{
			short* sx = srcx.ptr<short>(j);
			short* sy = srcy.ptr<short>(j);
			for (int i = 0; i < srcx.cols; i++)
			{
				int dx = i + sx[i];
				int dy = j + sy[i];

				if (dy >= 0 && dy<srcx.rows &&
					dx >= 0 && dx<srcx.cols)
				{
					int x = destx.at<short>(dy, dx);
					int y = desty.at<short>(dy, dx);
					if (abs(sx[i] - x)>thresh || abs(sy[i] - y)>thresh)
					{
						destx.at<short>(dy, dx) = inv;
						desty.at<short>(dy, dx) = inv;
						sx[i] = inv;
						sy[i] = inv;

					}
				}

			}
		}
	}

	template <class T>
	static void fillOcclusionAbs_(Mat& src, const T invalidvalue, const T maxval)
	{
		const int MAX_LENGTH = (int)(src.cols*0.5);
		//#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			T* s = src.ptr<T>(j);

			s[0] = maxval;
			s[src.cols - 1] = maxval;

			for (int i = 1; i < src.cols - 1; i++)
			{
				if (s[i] == invalidvalue)
				{
					int t = i;
					do
					{
						t++;
						if (t > src.cols - 1)break;
					} while (s[t] == invalidvalue);

					T dd;
					if (abs(s[i - 1]) < abs(s[t]))
						dd = s[i - 1];
					else
						dd = s[t];

					if (t - i > MAX_LENGTH)
					{
						for (int n = 0; n < src.cols; n++)
						{
							s[n] = invalidvalue;
						}
					}
					else
					{
						for (; i < t; i++)
						{
							s[i] = dd;
						}
					}
				}
			}
			s[0] = s[1];
			s[src.cols - 1] = s[src.cols - 2];
		}
	}
	void OpticalFlowBM::operator()(Mat& curr, Mat& next, Mat& dstx, Mat& dsty, Size ksize, int minx, int maxx, int miny, int maxy, int bd)
	{
		int invalid = 1024;

		int ss = 20;
		int sd = 2;
		int sobelclip = 10;
		int a = 9;

		Mat cim;
		Mat nim;
		copyMakeBorder(curr, cim, bd, bd, bd, bd, BORDER_REPLICATE);
		copyMakeBorder(next, nim, bd, bd, bd, bd, BORDER_REPLICATE);

		Mat destx = Mat::zeros(cim.size(), CV_16S);
		Mat desty = Mat::zeros(cim.size(), CV_16S);
		Mat odestx = Mat::zeros(cim.size(), CV_16S);
		Mat odesty = Mat::zeros(cim.size(), CV_16S);
		int rangex = maxx - minx + 1;
		int rangey = maxy - miny + 1;
		int searchsize = rangex*rangey;

		cost.resize(searchsize);
		ocost.resize(searchsize);

		int cn = curr.channels();
		vector<Mat> v0;
		vector<Mat> v1;
		split(cim, v0);
		split(nim, v1);

		vector<Mat> s0(3);
		vector<Mat> s1(3);

		Mat temp;
		for (int i = 0; i < cn; i++)
		{
			Sobel(v0[i], temp, CV_16S, 1, 0);
			max(temp, sobelclip, temp);
			temp += sobelclip;
			temp.convertTo(s0[i], CV_8U);

			Sobel(v1[i], temp, CV_16S, 1, 0);
			max(temp, sobelclip, temp);
			temp += sobelclip;
			temp.convertTo(s1[i], CV_8U);
		}



		Mat maxcost = Mat::zeros(cim.size(), CV_32S);
		maxcost.setTo(INT_MAX);
		Mat omaxcost = Mat::zeros(cim.size(), CV_32S);
		omaxcost.setTo(INT_MAX);
		Mat ccost;
		Mat occost;

#pragma omp parallel for
		for (int j = 0; j < rangey; j++)
		{
			Mat diff;
			Mat odiff;
			for (int i = 0; i < rangex; i++)
			{
				int count = j*rangex + i;
				cost[count] = Mat::zeros(cim.size(), CV_32S);
				ocost[count] = Mat::zeros(cim.size(), CV_32S);

				for (int c = 0; c < cn; c++)
				{
					warpShift(v1[c], diff, (i + minx), (j + miny), BORDER_REPLICATE);
					absdiff(diff, v0[c], diff);
					add(diff, cost[count], cost[count], noArray(), CV_32S);

					warpShift(v0[c], odiff, -(i + minx), -(j + miny), BORDER_REPLICATE);
					absdiff(odiff, v1[c], odiff);
					add(odiff, ocost[count], ocost[count], noArray(), CV_32S);

					warpShift(s1[c], diff, (i + minx), (j + miny), BORDER_REPLICATE);
					absdiff(diff, s0[c], diff);
					add(a*diff, cost[count], cost[count], noArray(), CV_32S);

					warpShift(s0[c], odiff, -(i + minx), -(j + miny), BORDER_REPLICATE);
					absdiff(odiff, s1[c], odiff);
					add(a*odiff, ocost[count], ocost[count], noArray(), CV_32S);
				}

				blur(cost[count], cost[count], ksize);
				//guidedFilter(cost[count],cim,cost[count],7,0.1);
				blur(ocost[count], ocost[count], ksize);
			}
		}
		int count = 0;
		Mat mask;
		for (int j = 0; j < rangey; j++)
		{
			for (int i = 0; i < rangex; i++)
			{
				//Mat cshow;cost[count].convertTo(cshow,CV_8U);imshow("c",cshow);waitKey(0);
				maxcost.copyTo(ccost);
				omaxcost.copyTo(occost);
				//showMatInfo(maxcost);
				//	showMatInfo(cost[count]);

				min(maxcost, cost[count], maxcost);
				compare(ccost, maxcost, mask, CMP_NE);
				destx.setTo(-(i + minx), mask);
				desty.setTo(-(j + miny), mask);

				min(omaxcost, ocost[count], omaxcost);
				compare(occost, omaxcost, mask, CMP_NE);
				odestx.setTo(-(i + minx), mask);
				odesty.setTo(-(j + miny), mask);

				count++;
			}
		}


		cncheck(destx, desty, odestx, odesty, 8, invalid);
		Mat dstx_, dsty_;
		Mat(destx(Rect(bd, bd, curr.cols, curr.rows))).copyTo(dstx_);
		Mat(desty(Rect(bd, bd, curr.cols, curr.rows))).copyTo(dsty_);

		filterSpeckles(dstx_, invalid, ss, sd, buffSpeckle);
		filterSpeckles(dsty_, invalid, ss, sd, buffSpeckle);

		fillOcclusionAbs_<short>(dstx_, invalid, 1000);
		fillOcclusionAbs_<short>(dsty_, invalid, 1000);

		dstx_.convertTo(dstx, CV_32F);
		dsty_.convertTo(dsty, CV_32F);
	}


	void chooseFlow(Mat& prevC1, Mat& nextC1, Mat& prevC3, Mat& nextC3, Mat& flow, int method)
	{
		if (flow.empty()) flow.create(Size(prevC1.cols, prevC1.rows), CV_32FC2);

		Mat dstflow, dstorg, xdst, ydst, xflow, yflow;
		Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();

		switch (method) {
		case 0:
			tvl1->calc(prevC1, nextC1, flow);
			break;
		case 1:
			optflow::calcOpticalFlowSF(prevC3, nextC3, flow, 3, 5, 3);
			break;
		case 2:
			calcOpticalFlowFarneback(prevC1, nextC1, flow, 0.7, 6, 12, 1, 5, 1.1, 0);
			break;
		case 3:
			OpticalFlowBM obm;
			int maxflow = 15;
			obm(prevC3, nextC3, xflow, yflow, Size(21, 21), -maxflow, maxflow, -maxflow, maxflow, maxflow + 10);

			mergeFlow(flow, xflow, yflow);
			Mat v; drawOpticalFlow(flow, v);
			guiAlphaBlend(prevC3, v);
			break;
		}
	}

	void makeEdgeWeight(Mat& guide, Mat& weight)
	{
		Mat gray, edge;
		guide.convertTo(gray, CV_8U);
		Canny(gray, edge, 200, 200, 3, true);
		edge.convertTo(edge, CV_32F);
		Mat emask; compare(edge, 255, emask, CMP_EQ);
		edge = 1;
		edge.setTo(0, emask);
		weight = edge.clone();

		//double a, b;
		//bilateralWeightMap(gray, weight, Size(7,7), 5, 10);
		//minMaxLoc(weight, &a, &b);
		//weight -= a;
		//weight *= (1.f/b);

		imshow("weight", weight);
	}

	float getMaxFlowRadius(Mat& src)
	{
		float maxrad = 1.0;
		float* s = src.ptr<float>(0);


		for (int i = 0; i < src.size().area(); i++)
		{
			Point2f u = Point2f(s[2 * i], s[2 * i + 1]);
			maxrad = max(maxrad, u.x * u.x + u.y * u.y);
		}

		return sqrt(maxrad);
	}

	void maxminFilter(Mat& src, Mat& dst)
	{
		if (dst.empty()) dst.create(src.size(), CV_8U);

		Mat max, min, temp;
		maxFilter(src, max, Size(3, 3));
		minFilter(src, min, Size(3, 3));
		Mat difmax; absdiff(src, max, difmax);
		Mat difmin; absdiff(src, min, difmin);
		Mat dif; absdiff(max, min, dif);

		float th = 3.0;
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				if (dif.at<float>(i, j) < th)
					dst.at<uchar>(i, j) = 127;
				else if (difmax.at<float>(i, j) < difmin.at<float>(i, j))
					dst.at<uchar>(i, j) = 255;
				else
					dst.at<uchar>(i, j) = 0;
			}
	}

	void flowCorrect(Mat& before, Mat& after)
	{
		Mat xbf, xaf, ybf, yaf;
		divideFlow(before, xbf, ybf);
		divideFlow(after, xaf, yaf);

		Mat xdif; absdiff(xbf, xaf, xdif);
		Mat ydif; absdiff(ybf, yaf, ydif);

		float th = 2.0;
		Mat xl, yl;
		compare(xdif, th, xl, CMP_GE);
		compare(ydif, th, yl, CMP_GE);

		float th_f = 0.05;
		Mat mask; compare(xl, yl, mask, CMP_NE);
		Mat xcopy; copyMakeBorder(xaf, xcopy, 1, 1, 1, 1, BORDER_REPLICATE);
		Mat ycopy; copyMakeBorder(yaf, ycopy, 1, 1, 1, 1, BORDER_REPLICATE);
		for (int i = 0; i < xbf.rows; i++) {
			for (int j = 0; j<xbf.cols; j++) {
				if (mask.at<uchar>(i, j) == 255) {
					if (xl.at<uchar>(i, j) > yl.at<uchar>(i, j)) {
						for (int x = 0; x < 3; x++)
							for (int y = 0; y < 3; y++)
								if ((xcopy.at<float>(i + y, j + x) >= xaf.at<float>(i, j) - th_f && xcopy.at<float>(i + y, j + x) <= xaf.at<float>(i, j) + th_f)
									&& (ycopy.at<float>(i + y, j + x) <= yaf.at<float>(i, j) - th_f || ycopy.at<float>(i + y, j + x) >= yaf.at<float>(i, j) + th_f))
									yaf.at<float>(i, j) = ycopy.at<float>(i + y, j + x);
					}
					else {
						for (int x = 0; x < 3; x++)
							for (int y = 0; y < 3; y++)
								if ((ycopy.at<float>(i + y, j + x) >= yaf.at<float>(i, j) - th_f && ycopy.at<float>(i + y, j + x) <= yaf.at<float>(i, j) + th_f)
									&& (xcopy.at<float>(i + y, j + x) <= xaf.at<float>(i, j) - th_f || xcopy.at<float>(i + y, j + x) >= xaf.at<float>(i, j) + th_f))
									xaf.at<float>(i, j) = xcopy.at<float>(i + y, j + x);
					}
				}
			}
		}
		mergeFlow(after, xaf, ybf);
	}

	template <class T>
	void dilateFlow_(Mat& src, Mat& mag, Mat& dest, Size size)
	{
		if (dest.empty()) dest.create(src.size(), src.type());
		Mat im, mim;
		const int hh = size.height / 2;
		const int hw = size.width / 2;
		const int width = src.cols;
		const int height = src.rows;

		copyMakeBorder(mag, mim, hh, hh, hw, hw, BORDER_REPLICATE);
		copyMakeBorder(src, im, hh, hh, hw, hw, BORDER_REPLICATE);

		for (int j = 0; j < height; j++)
		{
			T* d = dest.ptr<T>(j);
			for (int i = 0; i < width; i++)
			{
				T argmax_u = (T)0;
				T argmax_v = (T)0;
				T maxval = (T)0;
				for (int l = -hh; l <= hh; l++)
				{
					T* s = im.ptr<T>(j + l + hh) + 2 * hw;
					T* m = mim.ptr<T>(j + l + hh) + hw;

					for (int k = -hw; k <= hw; k++)
					{
						if (maxval < m[i + k])
						{
							maxval = m[i + k];
							argmax_u = s[2 * (i + k)];
							argmax_v = s[2 * (i + k) + 1];
						}
					}
				}
				d[2 * i] = argmax_u;
				d[2 * i + 1] = argmax_v;
			}
		}
	}

	void dilateFlow(Mat& src, Mat& dest, Size size)
	{
		vector<Mat> v;
		split(src, v);
		Mat mag;
		magnitude(v[0], v[1], mag);
		if (src.depth() == CV_8U)
		{
			dilateFlow_<uchar>(src, mag, dest, size);
		}
		else if (src.depth() == CV_32F)
		{
			dilateFlow_<float>(src, mag, dest, size);
		}
		else if (src.depth() == CV_16S)
		{
			;
		}

		//merge(v,dest);
	}
}