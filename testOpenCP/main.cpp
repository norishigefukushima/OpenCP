#include <opencp.hpp>
#include "test.hpp"
#include "fmath.hpp"
using namespace std;
#include "sugimoto2017.hpp"

void circleFilter(Mat& src, Mat& dest, int r)
{
	dest.create(src.size(), src.type());
	Mat img;
	copyMakeBorder(src, img, r, r, r, r, BORDER_REPLICATE);

	int ksize = (2 * r + 1)*(2 * r + 1);
	vector<float> weight(ksize);
	float total = 0.f;
	int area = 0;
	for (int l = -r; l <= r; l++)
	{
		for (int k = -r; k <= r; k++)
		{
			float p = sqrt(k*k + l*l);
			if (p > r) continue;
			float v = 1.f;
			weight[area++] = v;
			total += v;
		}
	}

	for (int l = -r, idx = 0; l <= r; l++)
	{
		for (int k = -r; k <= r; k++)
		{
			weight[idx++] /= area;
		}
	}

#pragma omp parallel for
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			float total = 0.f;
			for (int l = -r, index = 0; l <= r; l++)
			{
				for (int k = -r; k <= r; k++)
				{
					float p = sqrt(k*k + l*l);
					if (p > r) continue;
					total += weight[index++] * img.at<float>(j + l + r, i + k + r);
				}
			}
			dest.at<float>(j, i) = total;
		}

	}
}


class VisualizeDenormalKernel
{
public:
	string wname;
	static void onMouse(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == CV_EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	
	int r;
	int sigma_s;
	int sigma_r;
	int sw;
	int amp;

	float expr2(float a, float isigma)
	{
		return std::exp((a*a)*isigma);
	}
	


	float expr(float a, float sigma)
	{
		return std::exp((a*a)/-(2.f*sigma*sigma));
	}
	float exps(float a, float b, float sigma)
	{
		return std::exp((a*a+b*b) / -(2.f*sigma*sigma));
	}
	
	void filter(Mat& src, Mat& guide, Mat& dest, Point pt = Point(0, 0))
	{
		dest.create(src.size(), CV_32F);
		dest.setTo(0);
		const float sigma_range = sigma_r / 10.f;
		const float sigma_space = sigma_s / 10.f;

		Mat im; copyMakeBorder(src, im, r, r, r, r, BORDER_REPLICATE);
		Mat g; copyMakeBorder(guide, g, r, r, r, r, BORDER_REPLICATE);
		const int D = 2 * r + 1;

		const float cgb = g.at<float>(pt.y + r, 3 * (pt.x + r) + 0);
		const float cgg = g.at<float>(pt.y + r, 3 * (pt.x + r) + 1);
		const float cgr = g.at<float>(pt.y + r, 3 * (pt.x + r) + 2);

		const __m256 cmg = _mm256_set1_ps(g.at<float>(pt.y + r, 3 * (pt.x + r) + 0));
		const __m256 cmb = _mm256_set1_ps(g.at<float>(pt.y + r, 3 * (pt.x + r) + 1));
		const __m256 cmr = _mm256_set1_ps(g.at<float>(pt.y + r, 3 * (pt.x + r) + 2));
		const int CV_DECL_ALIGNED(32) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

		float CV_DECL_ALIGNED(32) buff[8];

		const float isigma_range = 1.f / -(2.f*sigma_range*sigma_range);

		vector<float> EXPS(D*D);
		vector<float> EXPR(256);
		
		for (int j = 0,n=0; j < D; j++)
		{
			float q = j - r;
			for (int i = 0; i < D; i++)
			{
				float p = i - r;
				//EXPS[n] = fmath::exp((p*p + q*q) / (-2.f*sigma_space*sigma_space))*(float)amp;
				EXPS[n] = max(FLT_EPSILON, fmath::exp((p*p + q*q) / (-2.f*sigma_space*sigma_space)));

			}
		}

		for (int i = 0; i < 256; i++)
		{
			EXPR[i] = max(FLT_EPSILON, fmath::exp((i*i)*isigma_range));
		}

		for (int j = 0, n=0; j < D ; j++)
		{
			float q = j - r;

			float* gg = g.ptr<float>(pt.y + j);
			float* imm = im.ptr<float>(pt.y + j);
			for (int i = 0; i < D ; i++)
			{
				float p = i - r;

				_MM_SET_EXCEPTION_STATE(0);
				
				/*
				float w = exps(p, q, sigma_space);
				w *= expr(gg[3 * (i + pt.x) + 0] - cgb, sigma_range);
				w *= expr(gg[3 * (i + pt.x) + 1] - cgg, sigma_range);
				w *= expr(gg[3 * (i + pt.x) + 2] - cgr, sigma_range);
				*/
				/*
				float w = exps(p, q, sigma_space)
					*expr(gg[3 * (i + pt.x) + 0] - cgb, sigma_range);
					*expr(gg[3 * (i + pt.x) + 1] - cgg, sigma_range)
					*expr(gg[3 * (i + pt.x) + 2] - cgr, sigma_range);
					*/
					//*imm[3 * j + 0];
				
				/*
				float w = exps(p, q, sigma_space)
					*expr2(gg[3 * (i + pt.x) + 0] - cgb, isigma_range)
					*expr2(gg[3 * (i + pt.x) + 1] - cgg, isigma_range)
					*expr2(gg[3 * (i + pt.x) + 2] - cgr, isigma_range);
				*/
				
				float w = EXPS[n++]
					*EXPR[abs((int)(gg[3 * (i + pt.x) + 0] - cgb))]
					*EXPR[abs((int)(gg[3 * (i + pt.x) + 1] - cgg))]
					*EXPR[abs((int)(gg[3 * (i + pt.x) + 2] - cgr))];
				
				/*
				__m256 mg = _mm256_loadu_ps(gg + 3 * (i + pt.x) + 0);
				__m256 mb = _mm256_loadu_ps(gg + 3 * (i + pt.x) + 1);
				__m256 mr = _mm256_loadu_ps(gg + 3 * (i + pt.x) + 2);
				

				mg = _mm256_and_ps(_mm256_sub_ps(mg, cmg), *(const __m256*)v32f_absmask);
				_mm256_store_ps(buff, mg);
				float w = EXPR[buff[0]] * EXPR[buff[0]] * EXPR[buff[0]];
				*/	
				
				const unsigned int mxcsr = _mm_getcsr();

				if (mxcsr & 0b10)
				{
					//cout<<"denormal"<<endl;
					dest.at <float>(pt.y + j - r, pt.x + i - r) = 255.f;
				}
				else
				{
					//cout << "normal" << endl;
					//dest.at <float>(pt.y + j-r, pt.x + i-r) = w;
				}				
			}
		}
	}
	
	void setUpTrackbar()
	{
		sw = 0; createTrackbar("sw", wname, &sw, 5);
		amp = 1; createTrackbar("amp", wname, &amp, 1024);
		r = 25; createTrackbar("r", wname, &r, 200);
		sigma_s = 120; createTrackbar("sigmaS", wname, &sigma_s, 1000);
		sigma_r = 30; createTrackbar("sigmaR", wname, &sigma_r, 1000);
	}

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
		if (maxKenelPlots >= 2)
		{
			createTrackbar("pt_index", wname, &ptindex, maxKenelPlots);
		}
		int a = 0; createTrackbar("a", wname, &a, 100);
		setUpTrackbar();

		int key = 0;
		Mat dest;
		Mat show;
		Mat point = Mat::ones(src.size(), CV_32F);

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
		Mat srcc;
		if (src.channels() != 3)
			cvtColor(src, srcc, COLOR_GRAY2BGR);
		else
			src.copyTo(srcc);
		while (key != 'q')
		{
			point.setTo(FLT_EPSILON);
			for (int i = 0; i < maxKenelPlots; i++)
			{
				point.at <float>(pts[i]) = 25500.0;
			}
			pts[ptindex] = pt;

			/*
			filter(srcf, srcf, dest, pt);
			dest.convertTo(show, CV_8U);
			imshow("image", show);
			*/

			filter(point, srcf, dest, pt);
			normalize(dest, dest, 255, 0, NORM_MINMAX);

			//showProfile(dest, pt);
			dest.convertTo(show, CV_8U);
			
			applyColorMap(show, show, 2);
			alphaBlend(srcc, show, a*0.01, show);

			imshow(wname, show);
			key = waitKey(1);
		}
	}
};

void guiMedianFilterTest(Mat& src, string wname="Median")
{
	namedWindow(wname);
	int r = 1; createTrackbar("r", wname, &r, 50);
	int sigma = 0; createTrackbar("sigma", wname, &sigma, 100);
	int sp = 10; createTrackbar("s/p", wname, &sp, 100);
	int key = 0;
	Mat dest;
	Mat noise;
	while (key != 'q')
	{
		addNoise(src, noise, sigma, sp / 100.0);
		medianBlur(noise, dest, r * 2 + 1);
		imshow(wname, dest);
		key = waitKey(1);
		cout << PSNR(src, noise) << "," << PSNR(src, dest) << endl;
	}
}

void matmul()
{
	int size = 1024+64;
	Mat a(size, size, CV_32F);
	Mat b(size, size, CV_32F);
	Mat c(size, size, CV_32F);
	randu(a, 0.f, 255.f);
	randu(b, 0.f, 255.f);

	int iter = 100;

	CalcTime t("flops", TIME_SEC);
	for (int i = 0; i < iter; i++)
	{
		cv::gemm(a, b, 1.0, NULL, 0.0, c);
		//c = a*b;
	}
	double f =  iter/ t.getTime();
	cout << "GFLOPS " << 2.0 * ((double)size*size*size) * f /(1000*1000*1000) << endl;
}

void shiftImage(Mat& src, Mat& dest, int shiftx)
{
	if (shiftx < 0)
	{
		
		Mat im;
		copyMakeBorder(src, im, 0, 0, 0, -shiftx, BORDER_REPLICATE);
		Mat(im(Rect(-shiftx, 0, src.cols, src.rows))).copyTo(dest);
	}
	else
	{
		
		Mat im;
		copyMakeBorder(src, im, 0, 0, shiftx, 0,BORDER_REPLICATE);
		Mat(im(Rect(0, 0, src.cols, src.rows))).copyTo(dest);
	}
}

void stereotest()
{
	Mat leftim = imread("army09_gray.png", 0);
	Mat rightim = imread("army07_gray.png", 0);
	//guiShift(leftim, rightim);

	
	Mat disp = Mat::zeros(leftim.size(), CV_8U);

	int mind = 0;
	int disparity_range = 16;
	const int r = 2;
	const int D = 2 * r + 1;

	Mat shift;
	vector<Mat> diff(disparity_range);
	for (int i = 0; i < disparity_range; i++)
	{
		shiftImage(rightim, shift, i+mind);
		absdiff(leftim, shift, diff[i]);
		boxFilter(diff[i], diff[i], CV_8U, Size(D, D));
	}
	
	for (int i = 0; i < leftim.size().area(); i++)
	{
		uchar emin = 255;
		int argmin = 0;
		for (int d = 0; d < disparity_range; d++)
		{
			uchar e = diff[d].at<uchar>(i);
			if (e < emin)
			{
				emin = e;
				argmin = d;
			}
		}
		disp.at<uchar>(i) = argmin;
	}
	
	imshow("disp", disp * 20);

	int line = 30;
	Mat DSI = Mat::zeros(Size(leftim.cols, disparity_range), CV_8U);
	for (int i = 0; i < disparity_range; i++)
	{
		Mat a = DSI.row(i);
		diff[i].row(line).copyTo(a);
	}
	Mat DSIresize;
	resize(DSI, DSIresize, Size(), 1.0, 5, CV_INTER_NN);
	imshow("DSI", DSIresize);

	waitKey();
}

void splitmergeTest(Mat& src)
{
	int iter = 10;
	Mat dest(src.size(), src.type());
	for(int i=0;i<iter;i++)
	{
		CalcTime t("base");
		vector<Mat> dst(3);
		
		split(src, dst);
		merge(dst, dest);
	}
	Mat dest2(src.size(), src.type());
	for (int i = 0; i<iter; i++)
	{
		CalcTime t("my");
		vector<Mat> dst;
		
		splitConvert(src, dst);
		//split(src, dst);
		merge(dst, dest);
		//mergeConvert(dst, dest2, false);
	}
	cp::guiAlphaBlend(dest, dest2);
}

int main(int argc, char** argv)
{
	
	//stereotest();
	//matmul(); return 0;
	Mat left = imread("img/stereo/Reindeer/view1.png", 0);
	Mat right = imread("img/stereo/Reindeer/view5.png", 0);
	Mat dmap = imread("img/stereo/Reindeer/sgbm.png", 0);
	Mat img = imread("img/lenna.png");
	Mat a;
	resize(img, a, Size(513, 513));
	splitmergeTest(a); return 0;
	//Mat img = imread("img/Kodak/kodim07.png");
	//Mat img = imread("img/b.png");
	
	//guiUpsampleTest(img);return 0;
	//guiDomainTransformFilterTest(img);
	//guiMedianFilterTest(img);
	//VisualizeDenormalKernel vdk;
	//vdk.run(img);
	//return 0;
	//VizKernel vk;
	//vk.run(img, 2);
	//HazeRemove2 hz;
	//Mat haze = imread("img/haze/swans.png");
	//Mat haze = imread("img/haze/canyon.png");
	
	//hz.gui(haze, "haze");
	//guiStereoSGBMTest(left, right, 96);

	//iirGuidedFilterTest2(img); return 0;
	//iirGuidedFilterTest1(dmap, left); return 0;
	//iirGuidedFilterTest(); return 0;
	//iirGuidedFilterTest(left); return 0;
	//fitPlaneTest(); return 0;
	//guiWeightMapTest(); return 0;
	//guiStereo(); return 0;
	//guiPlotTest(); return 0;
	//zoom(argc, argv);return 0;

	//guiGeightedJointBilateralFilterTest();
	//Mat haze = imread("img/haze/haze2.jpg"); guiHazeRemoveTest(haze);
	//Mat fuji = imread("img/fuji.png"); guiDenoiseTest(fuji);
	//Mat ff3 = imread("img/pixelart/ff3.png");

	Mat src = imread("img/lenna.png",0);
	//Mat src = imread("img/Kodak/kodim07.png",0);
	guiIterativeBackProjectionTest(src);
	//Mat src = imread("img/Kodak/kodim15.png",0);
	
	//Mat src = imread("img/cave-flash.png");
	//Mat src = imread("img/feathering/toy.png");
	//Mat src = imread("Clipboard01.png");

	//timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");
	//Mat src = imread("img/teddy_disp1.png");
	//Mat src_ = imread("img/stereo/Art/view1.png",0);
	//	Mat src;
	//	copyMakeBorder(src_,src,0,1,0,1,BORDER_REPLICATE);

	//Mat src = imread("img/lenna.png", 0);


	


	//Mat src = imread("img/stereo/Dolls/view1.png");
	//guiDenoiseTest(src);
	guiBilateralFilterTest(src);
	Mat ref = imread("img/stereo/Dolls/view6.png");
	//guiColorCorrectionTest(src, ref); return 0;
	//Mat src = imread("img/flower.png");
	//guiAnalysisImage(src);
	Mat dst = src.clone();
	//paralleldenoise(src, dst, 5);
	Mat disp = imread("img/stereo/Dolls/disp1.png",0 );
	//	Mat src;
	Mat dest;


	//guiCrossBasedLocalFilter(src); return 0;
	//guiHistgramTest(src);
	//Mat src = imread("img/kodim22.png");
	//Mat src = imread("img/teddy_view1.png");

	//eraseBoundary(src,10);
	Mat mega;
	resize(src, mega, Size(1024, 1024));
	//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiDualBilateralFilterTest(src,disp);
	//guiGausianFilterTest(src); return 0;

	//guiCoherenceEnhancingShockFilter(src, dest);
	
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	//guiDisparityPlaneFitSLICTest(src, ref, disp); return 0;
	getPSNRRealtimeO1BilateralFilterKodak();
	guiRealtimeO1BilateralFilterTest(src); return 0;
	
	Mat flashImg = imread("img/flash/cave-flash.png");
	Mat noflashImg = imread("img/flash/cave-noflash.png");
	Mat noflashImgGray; cvtColor(noflashImg, noflashImgGray, COLOR_BGR2GRAY);
	Mat flashImgGray; cvtColor(flashImg, flashImgGray, COLOR_BGR2GRAY);
	Mat fmega, nmega;
	resize(flashImgGray, fmega, Size(1024, 1024));
	resize(noflashImg, nmega, Size(1024, 1024));
	
	guiSLICTest(src);
	//guiEdgePresevingFilterOpenCV(src);

	//guiJointRealtimeO1BilateralFilterTest(noflashImgGray, flashImgGray); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImg, flashImgGray); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImgGray, flashImg); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImg, flashImg); return 0;

	//guiWeightedHistogramFilterTest(noflashImgGray, flashImg); return 0;
	//guiRealtimeO1BilateralFilterTest(noflashImgGray); return 0;
	//guiRealtimeO1BilateralFilterTest(src); return 0;
	//guiDMFTest(nmega, nmega, fmega); return 0;
	//guiGausianFilterTest(src); return 0;


	//guiAlphaBlend(ff3,ff3);
	//guiJointNearestFilterTest(ff3);
	//guiViewSynthesis();
	
	//guiSeparableBilateralFilterTest(src);
	//guiBilateralFilterSPTest(mega);
	//guiRecursiveBilateralFilterTest(mega);
	//fftTest(src);

	Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);

	Mat flash = imread("img/cave-flash.png");
	Mat noflash = imread("img/cave-noflash.png");
	Mat disparity = imread("img/teddy_disp1.png", 0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiCodingDistortionRemoveTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	

	//application 
	//guiDetailEnhancement(src);
	//guiGuidedFilterTest(mega);
	//	guiDomainTransformFilterTest(mega);
	return 0;
}