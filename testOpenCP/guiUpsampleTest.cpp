#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

class VisualizeKernel
{
public:
	string wname;
	static void onMouse(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	virtual void filter(Mat& src, Mat& guide, Mat& dest) = 0;
	virtual void setUpTrackbar() = 0;

	void showProfile(Mat& src, Point pt)
	{
		Mat show(Size(src.cols, 255), CV_8U);
		show.setTo(255);
		for (int i = 0; i < src.cols - 1; i++)
		{
			line(show, Point(i, src.at<float>(pt.y, i)),
				Point(i + 1, src.at<float>(pt.y, i + 1)), COLOR_BLACK);
		}
		flip(show, show, 0);
		imshow("plofile", show);
	}

	void run(Mat& src, const int maxKenelPlots = 1, Point pt = Point(0, 0), string winname = "viz")
	{
		wname = winname;
		namedWindow(wname);
		int ptindex = 0;
		setUpTrackbar();
		if (maxKenelPlots >= 2)
		{
			createTrackbar("pt_index", wname, &ptindex, maxKenelPlots);
		}
		int a = 20; createTrackbar("alpha", wname, &a, 100);
		int base = 64; createTrackbar("base", wname, &base, 128);
		int noise = 0; createTrackbar("noise", wname, &noise, 100);

		vector<Point> pts(maxKenelPlots);
		if (pt.x == 0 && pt.y == 0)
		{
			pt = Point(src.cols / 2, src.rows / 2);
			for (int i = 0; i < maxKenelPlots; i++)
			{
				pts[i] = Point(src.cols / 2, src.rows / 2);
			}
		}
		cv::setMouseCallback(wname, (MouseCallback)onMouse, (void*)&pt);

		Mat srcf; src.convertTo(srcf, CV_32F);
		Mat srcc; cvtColor(src, srcc, COLOR_GRAY2BGR);
		Mat dest;
		Mat show;
		Mat point = Mat::ones(src.size(), CV_32F);

		int key = 0;
		while (key != 'q')
		{
			point.setTo(FLT_EPSILON);
			for (int i = 0; i < maxKenelPlots; i++)
			{
				point.at <float>(pts[i]) = 25500.0;
			}
			pts[ptindex] = pt;

			Mat nf;
			addNoise(srcf, nf, noise);
			filter(nf, nf, dest);
			dest.convertTo(show, CV_8U);
			imshow("filtered", show);
			filter(point, nf, dest);

			double minv, maxv;
			minMaxLoc(dest, &minv, &maxv);
			dest = (255.0 / maxv * dest)*((255. - base) / 255.0) + base;

			//normalize(dest, dest, 255, 0, NORM_MINMAX);

			showProfile(dest, pt);
			dest.convertTo(show, CV_8U);

			applyColorMap(show, show, 2);
			alphaBlend(srcc, show, a*0.01, show);

			imshow(wname, show);
			key = waitKey(1);
		}
		destroyAllWindows();
	}
};

class JointBilateralFilter_32f_Invoker : public cv::ParallelLoopBody
{
public:
	JointBilateralFilter_32f_Invoker(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, Mat& mask_) :
		temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs_src(_space_ofs), space_ofs_jnt(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), mask(&mask_)
	{
	}
	virtual void operator() (const Range& range) const
	{
		int i, cn = dest->channels();
		int cnj = guide->channels();
		Size size = dest->size();

		for (i = range.start; i != range.end; i++)
		{
			const float* jptr = guide->ptr<float>(i + radiusV) + radiusH*cnj;
			const float* sptr = temp->ptr<float>(i + radiusV) + radiusH*cn;
			float* dptr = dest->ptr<float>(i);
			uchar* msk = mask->ptr<uchar>(i);

			if (cn == 1 && cnj == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					if (*msk++ != 0)
					{
						float sum = 0.f, wsum = 0.f;
						float val0 = jptr[j];
						for (int k = 0; k < maxk; k++)
						{
							int val = jptr[j + space_ofs_src[k]];
							//float w = space_weight[k] * color_weight[cvRound(abs(val - val0))];
							float w = color_weight[cvRound(abs(val - val0))];

							//float val = jptr[j + space_ofs_src[k]];
							float vals = sptr[j + space_ofs_src[k]];
							//float w = space_weight[k] * color_weight[cvRound(std::abs(val - val0))];
							sum += vals*w;
							wsum += w;
						}
						dptr[j] = sum / wsum;
					}
				}
			}
			else if (cn == 3 && cnj == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					if (*msk++ != 0)
					{
						float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;
						const float b0j = jptr[j], g0j = jptr[j + 1], r0j = jptr[j + 2];

						for (int k = 0; k < maxk; k++)
						{
							const float* jptr_k = jptr + j + space_ofs_src[k];
							const float bj = jptr_k[0], gj = jptr_k[1], rj = jptr_k[2];
							const float* sptr_k = sptr + j + space_ofs_src[k];
							const float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

							float w = space_weight[k]
								* color_weight[cvRound(std::abs(bj - b0j) + std::abs(gj - g0j) + std::abs(rj - r0j))];
							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[j] = sum_b*wsum;
						dptr[j + 1] = sum_g*wsum;
						dptr[j + 2] = sum_r*wsum;
					}
				}
			}
			else if (cn == 1 && cnj == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					if (*msk++ != 0)
					{
						float sum_b = 0, wsum = 0;
						float b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];
						for (int k = 0; k < maxk; k++)
						{
							const float* sptr_k = jptr + j + space_ofs_jnt[k];
							float val = *(sptr + l + space_ofs_src[k]);
							float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
							float w = space_weight[k] * color_weight[cvRound(std::abs(b - b0) +
								std::abs(g - g0) + std::abs(r - r0))];
							sum_b += val*w;
							wsum += w;
						}
						dptr[l] = sum_b / wsum;
					}
				}
			}
			else if (cn == 3 && cnj == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					if (*msk++ != 0)
					{
						float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;
						const float val0 = jptr[l];
						for (int k = 0; k < maxk; k++)
						{
							float val = jptr[l + space_ofs_jnt[k]];
							const float* sptr_k = sptr + j + space_ofs_src[k];

							float w = space_weight[k] * color_weight[cvRound(std::abs(val - val0))];
							sum_b += sptr_k[0] * w; sum_g += sptr_k[1] * w; sum_r += sptr_k[2] * w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[j] = sum_b*wsum;
						dptr[j + 1] = sum_g*wsum;
						dptr[j + 2] = sum_r*wsum;
					}
				}
			}
		}
	}
private:
	const Mat *temp;
	const Mat *weightMap;
	Mat *dest;
	const Mat* guide;
	int radiusH, radiusV, maxk, *space_ofs_src, *space_ofs_jnt;
	float *space_weight, *color_weight;
	Mat* mask;
};

class JointBilateralFilter_8u_Invoker : public cv::ParallelLoopBody
{
public:
	JointBilateralFilter_8u_Invoker(Mat& _dest, const Mat& _temp, const Mat& _guide, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, int* _space_guide_ofs, float *_space_weight, float *_color_weight, Mat& mask_) :
		temp(&_temp), dest(&_dest), guide(&_guide), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs_src(_space_ofs), space_ofs_jnt(_space_guide_ofs), space_weight(_space_weight), color_weight(_color_weight), mask(&mask_)
	{
	}
	virtual void operator() (const Range& range) const
	{
		int i, cn = dest->channels();
		int cnj = guide->channels();
		Size size = dest->size();

		for (i = range.start; i != range.end; i++)
		{
			const uchar* jptr = guide->ptr<uchar>(i + radiusV) + radiusH*cnj;
			const uchar* sptr = temp->ptr<uchar>(i + radiusV) + radiusH*cn;
			uchar* dptr = dest->ptr<uchar>(i);
			uchar* msk = mask->ptr<uchar>(i);

			if (cn == 1 && cnj == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					if (*msk++ != 0)
					{
						float sum = 0.f, wsum = 0.f;
						//const int val0 = jptr[j];
						const int val0 = sptr[j];
						for (int k = 0; k < maxk; k++)
						{
							int val = jptr[j + space_ofs_src[k]];
							//float w = space_weight[k] * color_weight[cvRound(abs(val - val0))];
							float w = color_weight[cvRound(abs(val - val0))];
							sum += w*sptr[j + space_ofs_src[k]];
							wsum += w;
						}
						dptr[j] = cvRound(sum / wsum);
					}
				}
			}
			else if (cn == 3 && cnj == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					if (*msk++ != 0)
					{
						float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;
						const int b0j = jptr[j], g0j = jptr[j + 1], r0j = jptr[j + 2];

						for (int k = 0; k < maxk; k++)
						{
							const uchar* jptr_k = jptr + j + space_ofs_src[k];
							const int bj = jptr_k[0], gj = jptr_k[1], rj = jptr_k[2];
							const uchar* sptr_k = sptr + j + space_ofs_src[k];
							const int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

							float w = space_weight[k]
								* color_weight[cvRound(std::abs(bj - b0j) + std::abs(gj - g0j) + std::abs(rj - r0j))];
							sum_b += b*w;
							sum_g += g*w;
							sum_r += r*w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[j] = cvRound(sum_b*wsum);
						dptr[j + 1] = cvRound(sum_g*wsum);
						dptr[j + 2] = cvRound(sum_r*wsum);
					}
				}
			}
			else if (cn == 1 && cnj == 3)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					if (*msk++ != 0)
					{
						float sum_b = 0, wsum = 0;
						const int  b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];
						for (int k = 0; k < maxk; k++)
						{
							const uchar* sptr_k = jptr + j + space_ofs_jnt[k];
							int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
							float w = space_weight[k] * color_weight[cvRound(std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0))];
							sum_b += *(sptr + l + space_ofs_src[k])*w;
							wsum += w;
						}
						dptr[l] = cvRound(sum_b / wsum);
					}
				}
			}
			else if (cn == 3 && cnj == 1)
			{
				for (int j = 0, l = 0; j < size.width * 3; j += 3, l++)
				{
					if (*msk++ != 0)
					{
						float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;
						const int val0 = jptr[l];
						for (int k = 0; k < maxk; k++)
						{
							int val = jptr[l + space_ofs_jnt[k]];
							const uchar* sptr_k = sptr + j + space_ofs_src[k];

							float w = space_weight[k] * color_weight[cvRound(std::abs(val - val0))];
							sum_b += sptr_k[0] * w; sum_g += sptr_k[1] * w; sum_r += sptr_k[2] * w;
							wsum += w;
						}
						wsum = 1.f / wsum;
						dptr[j] = cvRound(sum_b*wsum);
						dptr[j + 1] = cvRound(sum_g*wsum);
						dptr[j + 2] = cvRound(sum_r*wsum);
					}
				}
			}
		}
	}
private:
	const Mat *temp;
	const Mat *weightMap;
	Mat *dest;
	const Mat* guide;
	int radiusH, radiusV, maxk, *space_ofs_src, *space_ofs_jnt;
	float *space_weight, *color_weight;
	Mat* mask;
};



void rapidDetailPreservingBase_(const Mat& src, const Mat& joint, Mat& dst, Size kernelSize, double lambda, int borderType, Mat& mask)
{
	if (kernelSize.area() == 1) { src.copyTo(dst); return; }
	if (dst.empty())dst = Mat::zeros(src.size(), src.type());
	if (dst.type() != src.type() || dst.size() != src.size())dst.create(src.size(), src.type());

	Size size = src.size();
	if (dst.empty())dst = Mat::zeros(src.size(), src.type());

	CV_Assert(src.depth() == joint.depth());

	const int cn = src.channels();
	const int cnj = joint.channels();

	int radiusH = kernelSize.width / 2;
	int radiusV = kernelSize.height / 2;

	Mat jim;
	Mat sim;
	copyMakeBorder(joint, jim, radiusV, radiusV, radiusH, radiusH, borderType);
	copyMakeBorder(src, sim, radiusV, radiusV, radiusH, radiusH, borderType);

	vector<float> _color_weight(cnj * 256);
	vector<float> _space_weight(kernelSize.area());
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];

	vector<int> _space_ofs_src(kernelSize.area());
	vector<int> _space_ofs_jnt(kernelSize.area());
	int* space_ofs_src = &_space_ofs_src[0];
	int* space_ofs_jnt = &_space_ofs_jnt[0];

	// initialize color-related bilateral filter coefficients
	for (int i = 0; i < 256 * cnj; i++)
	{
		color_weight[i] = max((float)std::pow(i / (double)(255 * cnj), lambda), FLT_EPSILON);
		//color_weight[i] = 1.0-max((float)std::pow(i / (double)(255 * cnj), lambda),FLT_EPSILON);
		//cout << color_weight[i] << endl;
	}
	//getchar();

	int maxk = 0;
	// initialize space-related bilateral filter coefficients
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			//if (r > max(radiusV, radiusH)) continue;
			//space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs_jnt[maxk] = (int)(i*jim.cols*cnj + j*cnj);
			space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
		}
	}

	if (src.depth() == CV_8U)
	{
		JointBilateralFilter_8u_Invoker body(dst, sim, jim, radiusH, radiusV, maxk, space_ofs_src, space_ofs_jnt, space_weight, color_weight, mask);
		parallel_for_(Range(0, size.height), body);
	}
	else if (src.depth() == CV_32F)
	{
		JointBilateralFilter_32f_Invoker body(dst, sim, jim, radiusH, radiusV, maxk, space_ofs_src, space_ofs_jnt, space_weight, color_weight, mask);
		parallel_for_(Range(0, size.height), body);
	}
}

void rapidDetailPreservingBase(const Mat& src, const Mat& joint, Mat& dst, Size kernelSize, double lambda, int borderType, InputArray mask_)
{
	Mat mask = mask_.getMat();
	if (mask.empty())
		mask = Mat::ones(src.size(), CV_8U);
	else
		src.copyTo(dst);
	rapidDetailPreservingBase_(src, joint, dst, kernelSize, lambda, borderType, mask);
}

void downSample(Mat& src, Mat& dest, int step)
{
	dest.create(Size(src.cols / step, src.rows / step), src.type());
	uchar* d = dest.ptr<uchar>(0);
	for (int j = 0; j < src.rows; j += step)
	{
		uchar* sp = src.ptr<uchar>(j);
		for (int i = 0; i < src.cols; i += step)
		{
			*d++ = sp[i];
		}
	}
}

void downSampleBox(Mat& src, Mat& dest, int step, int r)
{
	Mat b;
	boxFilter(src, b, -1, Size(2 * r + 1, 2 * r + 1));
	downSample(b, dest, step);
}

void downSampleGauss(Mat& src, Mat& dest, int step, int r, double sigma=0.0)
{
	Mat b;
	if(sigma ==0.0) sigma = r / 3.0;
	GaussianBlur(src, b, Size(2 * r + 1, 2 * r + 1), sigma, sigma, BORDER_REPLICATE);

	downSample(b, dest, step);
}

void downSampleRDP(Mat& src, Mat& dest, int step, int r, double lambda)
{
	Mat b;
	Mat g;
	double sigma = r / 3.0;
	GaussianBlur(src, g, Size(2 * r + 1, 2 * r + 1), sigma, sigma, BORDER_REPLICATE);
	rapidDetailPreservingBase(src, g, b, Size(2 * r + 1, 2 * r + 1), lambda, BORDER_REPLICATE, Mat());
	downSample(b, dest, step);
}


void downSampleBilateral(Mat& src, Mat& dest, int step, int r, double sigmaC, double sigmaS)
{
	Mat b;
	Mat g; 
	double sigma = r / 3.0;
	GaussianBlur(src, g, Size(2 * r + 1, 2 * r + 1), sigma, sigma, BORDER_REPLICATE);
	bilateralFilter(src, b, 2 * r + 1, sigmaC, sigmaS, BORDER_REPLICATE);
	downSample(b, dest, step);
}

void downSampleArea(Mat& src, Mat& dest, int step)
{
	dest.create(Size(src.cols / step, src.rows / step), src.type());
	uchar* d = dest.ptr<uchar>(0);
	const float div = 1.f / (step*step);
	for (int j = 0; j < src.rows; j += step)
	{
		for (int i = 0; i < src.cols; i += step)
		{
			int v = 0;
			for (int l = 0; l < step; l++)
			{
				uchar* sp = src.ptr<uchar>(j + l) + i;
				for (int k = 0; k < step; k++)
				{
					v += sp[k];
				}
			}
			*d++ = saturate_cast<uchar>(v*div + 0.5f);
		}
	}
}



void hblend(Mat& src, Mat& dest, float a)
{
	Mat s;
	warpShift(src, s, 1, 1, BORDER_REFLECT);
	addWeighted(src, 1.0-a, s, a, 0.0, dest);
}


class VizKernelRDP : public VisualizeKernel
{
public:

	int fa;
	int r;
	int lambda;
	int sigma;
	int sw;
	void setUpTrackbar()
	{
		r = 10; createTrackbar("r", wname, &r, 50);
		lambda = 10; createTrackbar("lambda*0.01", wname, &lambda, 200);
		sigma = 10; createTrackbar("sigma*0.01", wname, &sigma, 200);
	}

	void filter(Mat& src, Mat& guide, Mat& dest)
	{
		src.copyTo(dest);
		Mat joint = guide.clone();
		GaussianBlur(guide, joint, Size(2 * r + 1, 2 * r + 1), sigma*0.1);
		rapidDetailPreservingBase(src, joint, dest, Size(2 * r + 1, 2 * r + 1), lambda*0.01, BORDER_REPLICATE, Mat());
	}
};

void guiUpsampleTest(Mat& src_)
{
	
	Mat src;
	if (src_.channels() == 3)cvtColor(src_, src, COLOR_BGR2GRAY);
	else src = src_;

	VizKernelRDP rpd;
	rpd.run(src);

	//src_.copyTo(src);
	Mat dest = src.clone();

	string wname = "upsample";
	namedWindow(wname);

	int a = 0; createTrackbar("a", wname, &a, 100);
	int r = 1; createTrackbar("r", wname, &r, 10);

	int sw = 6; createTrackbar("sw:up", wname, &sw, 10);
	int swd = 8; createTrackbar("sw:down", wname, &swd, 10);
	int acubic = 100; createTrackbar("acubic", wname, &acubic, 200);

	int br = 1; createTrackbar("br", wname, &br, 10);
	int bsigma = 50; createTrackbar("bsigma", wname, &bsigma, 1000);
	int csigma = 330; createTrackbar("csigma", wname, &csigma, 3000);

	int lambda = 50; createTrackbar("lambda", wname, &lambda, 200);
	int mx = 0; createTrackbar("x", wname, &mx, 10);
	

	int key = 0;
	Mat show;
	ConsoleImage ci;
	string downsampleMethod;
	string upsampleMethod;
	while (key != 'q')
	{
		int s = (int)pow(2, r);
		Mat sub;
		//resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_AREA);

		switch (swd)
		{
		case 0:
			resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_NEAREST);
			downsampleMethod = "NN";
			break;

		case 1:
			resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_AREA);
			downsampleMethod = "Area";
			break;

		case 2:
			resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_LINEAR);
			downsampleMethod = "Linear";
			break;

		case 3:
			resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_CUBIC);
			downsampleMethod = "Cubic";
			break;

		case 4:
			resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_LANCZOS4);
			downsampleMethod = "LANCZOS";
			break;

		case 5:
			downSample(src, sub, s);
			downsampleMethod = "MyNN";
			break;

		case 6:
			downSampleBox(src, sub, s, br);
			downsampleMethod = "MyBox";
			break;

		case 7:
			downSampleGauss(src, sub, s, br, bsigma/100.0);
			downsampleMethod = "MyGauss";
			break;

		case 8:
			downSampleRDP(src, sub, s, br, lambda*0.01);
			downsampleMethod = "RDP";
			//downSampleBilateral(src, sub, s, br,csigma/10.0,bsigma / 100.0);
			//downsampleMethod = "MyBilateral";
			break;

		case 9:
			downSampleArea(src, sub, s);
			downsampleMethod = "MyArea";
			break;

		default:
			resize(src, sub, Size(), 1.0 / s, 1.0 / s, INTER_NEAREST);
			downsampleMethod = "NN";
			break;
		}
		CalcTime t;
		{
			dest.setTo(0);
			t.start();
			switch (sw)
			{
			case 0:
				resize(sub, dest, Size(), s, s, INTER_NEAREST);
				upsampleMethod = "NN";
				break;

			case 1:
				resize(sub, dest, Size(), s, s, INTER_LINEAR);
				upsampleMethod = "Linear";
				break;

			case 2:
				resize(sub, dest, Size(), s, s, INTER_CUBIC);
				upsampleMethod = "Cubic";
				break;

			case 3:
				resize(sub, dest, Size(), s, s, INTER_LANCZOS4);
				upsampleMethod = "LANCZOS";
				break;

			case 4:
				nnUpsample(sub, dest);
				upsampleMethod = "myNN";
				break;

			case 5:
				linearUpsample(sub, dest);
				upsampleMethod = "myLinear";
				break;

			case 6:
				cubicUpsample(sub, dest, -acubic / 100.0);
				upsampleMethod = "myCubic";
				break;

			default:
				resize(sub, dest, Size(), s, s, INTER_NEAREST);
				upsampleMethod = "NN";
				break;
			}
		}
		hblend(dest, dest, mx*0.1);
		
		ci(format("Up   : %s", upsampleMethod));
		ci(format("Down : %s", downsampleMethod));
		ci(format("PSNR : %f", calcImageQualityMetric(dest, src, cp::IQM_PSNR, 5)));
		ci(format("time : %f", t.getTime()));
		ci.show();

		if (key == 'f')
		{
			a = (a != 0) ? 0 : 100;
			setTrackbarPos("a", wname, a);
		}
		alphaBlend(src, dest, a / 100.0, show);

		imshow(wname, show);
		imshow("sub", sub);
		key = waitKey(1);
	}
	destroyWindow(wname);
}