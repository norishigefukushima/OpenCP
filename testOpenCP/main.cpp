#include <opencp.hpp>
#include "test.hpp"
using namespace std;
using namespace cv;
using namespace cp;

void testIsSame()
{
	Mat a(256, 256, CV_8U);
	Mat b(256, 256, CV_8U);
	randu(a, 0, 255);
	randu(b, 0, 255);
	isSame(a, b);
	isSame(a, a);
	isSame(a, b, 10);
	isSame(a, a, 10);
}

string getInformation()
{
	string ret = "version: " + cv::getVersionString() + "\n";
	ret += "==============\n";
	if (cv::useOptimized()) ret += "cv::useOptimized: true\n";
	else ret += "cv::useOptimized: false\n";
	if (cv::ipp::useIPP()) ret += "cv::ipp::useIPP: true\n";
	else ret += "cv::ipp::useIPP: true\n";
	ret += cv::ipp::getIppVersion() + "\n";
	ret += format("cv::getNumberOfCPUs = %d\n", cv::getNumberOfCPUs());
	ret += format("cv::getNumThreads = %d\n", cv::getNumThreads());
	ret += getCPUFeaturesLine() + "\n";
	ret += "==============\n";

	return ret;
}

template<typename T>
void detailBoostLinear_(Mat& src, Mat& smooth, Mat& dest, const double boost)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(smptr[i] + boost * (imptr[i] - smptr[i]));
	}
}

template<typename T>
void detailBoostGauss_(Mat& src, Mat& smooth, Mat& dest, const double boost, const double sigma)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		const double sub = imptr[i] - smptr[i];
		dptr[i] = saturate_cast<T>(smptr[i] + boost * exp(sub * sub / (-2.0 * sigma * sigma)) * (sub));
	}
}

inline double contrastLinear_(double val, double a, double c)
{
	return a * (val - c) + c;
}

inline double contrastGauss_(double val, double a, double c, double coeff)
{
	double sub = (val - c);
	return val - a * exp(coeff * sub * sub) * sub;
}

inline double contrastGamma_(double val, double gamma)
{
	return pow(val / 255.0, gamma) * 255.0;
}


template<typename T>
void baseContrastLinear_(Mat& src, Mat& smooth, Mat& dest, const double a, const double c, const int method)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(contrastLinear_(smptr[i], a, c) + (imptr[i] - smptr[i]));
	}
}

template<typename T>
void baseContrastGauss_(Mat& src, Mat& smooth, Mat& dest, const double a, const double c, const double sigma, const int method)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	const double coeff = -1.0 / (2.0 * sigma * sigma);
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(contrastGauss_(smptr[i], a, c, coeff) + (imptr[i] - smptr[i]));
		//dptr[i] = saturate_cast<T>(contrastLinear_(smptr[i], a, c) + (imptr[i] - smptr[i]));
	}
}

template<typename T>
void baseContrastGamma_(Mat& src, Mat& smooth, Mat& dest, const double gamma)
{
	T* imptr = src.ptr<T>();
	T* smptr = smooth.ptr<T>();
	T* dptr = dest.ptr<T>();
	const int size = src.size().area();
	for (int i = 0; i < size; i++)
	{
		dptr[i] = saturate_cast<T>(contrastGamma_(smptr[i], gamma) + (imptr[i] - smptr[i]));
		//dptr[i] = saturate_cast<T>(contrastLinear_(smptr[i], a, c) + (imptr[i] - smptr[i]));
	}
}

void baseContrast(cv::InputArray src, cv::InputArray smooth, cv::OutputArray dest, const double a, const double c, const double sigma, const int method)
{
	dest.create(src.size(), src.type());
	Plot pt;
	pt.setXRange(0, 256);
	pt.setYRange(0, 256);
	const double coeff = -1.0 / (2.0 * sigma * sigma);
	for (int i = 0; i < 256; i++)
	{
		pt.push_back(i, contrastLinear_(i, a, c), 0);
		pt.push_back(i, contrastGauss_(i, a, c, coeff), 1);
		pt.push_back(i, contrastGamma_(i, a), 2);

	}
	pt.plot("contrast", false);

	Mat im = src.getMat();
	Mat sm = smooth.getMat();
	Mat dst = dest.getMat();
	if (method == 0)
	{
		if (im.depth() == CV_8U)  baseContrastGamma_<uchar>(im, sm, dst, a);
		if (im.depth() == CV_32F) baseContrastGamma_<float>(im, sm, dst, a);
		if (im.depth() == CV_64F) baseContrastGamma_<double>(im, sm, dst, a);
	}
	else if (method == 1)
	{
		if (im.depth() == CV_8U)  baseContrastLinear_<uchar>(im, sm, dst, a, c, method);
		if (im.depth() == CV_32F) baseContrastLinear_<float>(im, sm, dst, a, c, method);
		if (im.depth() == CV_64F) baseContrastLinear_<double>(im, sm, dst, a, c, method);
	}
	else if (method == 2)
	{
		if (im.depth() == CV_8U)  baseContrastGauss_<uchar>(im, sm, dst, a, c, sigma, method);
		if (im.depth() == CV_32F) baseContrastGauss_<float>(im, sm, dst, a, c, sigma, method);
		if (im.depth() == CV_64F) baseContrastGauss_<double>(im, sm, dst, a, c, sigma, method);
	}
	else if (method == 3)
	{
		if (im.depth() == CV_8U)  baseContrastGamma_<uchar>(im, sm, dst, a);
		if (im.depth() == CV_32F) baseContrastGamma_<float>(im, sm, dst, a);
		if (im.depth() == CV_64F) baseContrastGamma_<double>(im, sm, dst, a);
	}
}

void detailBoost(cv::InputArray src, cv::InputArray smooth, cv::OutputArray dest, const double boost, const double sigma, const int method)
{
	dest.create(src.size(), src.type());

	Mat im = src.getMat();
	Mat sm = smooth.getMat();
	Mat dst = dest.getMat();
	if (method == 0)
	{
		if (im.depth() == CV_8U)  detailBoostGauss_<uchar>(im, sm, dst, boost, sigma);
		if (im.depth() == CV_32F) detailBoostGauss_<float>(im, sm, dst, boost, sigma);
		if (im.depth() == CV_64F) detailBoostGauss_<double>(im, sm, dst, boost, sigma);
	}
	else if (method == 1)
	{
		if (im.depth() == CV_8U) detailBoostLinear_<uchar>(im, sm, dst, boost);
		if (im.depth() == CV_32F) detailBoostLinear_<float>(im, sm, dst, boost);
		if (im.depth() == CV_64F) detailBoostLinear_<double>(im, sm, dst, boost);
	}
}

void detailTest()
{
	string wname = "enhance";
	namedWindow(wname);
	int boost = 100; createTrackbar("boost", wname, &boost, 300);
	int center = 128; createTrackbar("center", wname, &center, 255);
	int sigma = 30; createTrackbar("sigma", wname, &sigma, 300);
	int sr = 30; createTrackbar("sr", wname, &sr, 300);
	int ss = 5; createTrackbar("ss", wname, &ss, 20);
	int method = 0; createTrackbar("method", wname, &method, 2);
	//Mat src = imread("img/flower.png", 0);
	Mat src = imread("img/lenna.png", 0);
	//addNoise(src, src, 5);
	Mat smooth, smooth2;

	int key = 0;
	Mat show, show2;
	cp::ConsoleImage ci;
	while (key != 'q')
	{
		const int d = 2 * (ss * 3) + 1;
		cv::bilateralFilter(src, smooth, d, sr, ss);
		edgePreservingFilter(src, smooth2, 1, ss, sr / 255.0);
		//cp::highDimensionalGaussianFilterPermutohedralLattice(src, smooth2, sr, ss);

		detailBoost(src, smooth, show, boost * 0.01, sigma, method);
		detailBoost(src, smooth2, show2, boost * 0.01, sigma, method);
		//baseContrast(src, smooth, show, boost * 0.01, center, sigma, 0);
		//baseContrast(src, smooth2, show2, boost * 0.01, center, sigma, 0);

		ci("PSNR smooth  %f", getPSNR(smooth, smooth2));
		ci("PSNR enhance %f", getPSNR(show, show2));
		ci.show();
		imshow(wname, show);
		imshow(wname + "2", show2);

		key = waitKey(1);
		if (key == 'c')guiAlphaBlend(show, show2);
	}
}

template<int postprocess>
void rangeBlurFilter_(const Mat& src, Mat& dst, const int r, const float sigma_range, const float sigma_space, const int borderType)
{
	const float lastexp = -1.f;
	dst.create(src.size(), src.type());
	Mat srcf;
	if (src.depth() == CV_32F)srcf = src;
	else src.convertTo(srcf, CV_32F);

	Mat destf;
	if (src.depth() == CV_32F)destf = dst;
	else destf.create(src.size(), CV_32F);

	cv::Mat ave, stddev;
	if (postprocess == 1)//using stddev
	{
		meanStdFilter(srcf, ave, stddev, r);
	}
	else
	{
		blur(srcf, ave, Size(2 * r + 1, 2 * r + 1));
	}

	Mat im;
	copyMakeBorder(srcf, im, r, r, r, r, borderType);

	const int d = (2 * r + 1) * (2 * r + 1);
	vector<float> rangeTable(256);
	float* rweight = &rangeTable[0];
	vector<float> space(d);
	vector<int> offset(d);

	const float coeff_r = -1.f / (2.f * sigma_range * sigma_range);
	for (int i = 0; i < 256; i++)
	{
		rangeTable[i] = exp(i * i * coeff_r);
	}

	const float coeff_s = -1.f / (2.f * sigma_space * sigma_space);
	float wsum = 0.f;
	if (sigma_space <= 0)
	{
		for (int j = -r, idx = 0; j <= r; j++)
		{
			for (int i = -r; i <= r; i++)
			{
				double dis = double(i * i + j * j);
				offset[idx] = im.cols * j + i;
				float v = 1.f;
				wsum += v;
				space[idx] = v;
				idx++;
			}
		}
	}
	else
	{
		for (int j = -r, idx = 0; j <= r; j++)
		{
			for (int i = -r; i <= r; i++)
			{
				double dis = double(i * i + j * j);
				offset[idx] = im.cols * j + i;
				float v = (float)exp(dis * coeff_s);
				wsum += v;
				space[idx] = v;
				idx++;
			}
		}
	}

	for (int k = 0; k < d; k++)
	{
		space[k] /= wsum;
	}

#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < src.rows; j++)
	{
		float* dptr = destf.ptr<float>(j);
		const float* sptr = im.ptr<float>(j + r) + r;
		const float* aptr = ave.ptr<float>(j);

		for (int i = 0; i < src.cols; i += 32)
		{
			const float* si = sptr + i;
			const __m256 ma0 = _mm256_lddqu_ps(aptr + i);
			const __m256 ma1 = _mm256_lddqu_ps(aptr + i + 8);
			const __m256 ma2 = _mm256_lddqu_ps(aptr + i + 16);
			const __m256 ma3 = _mm256_lddqu_ps(aptr + i + 24);

			__m256 mv0 = _mm256_setzero_ps();
			__m256 mv1 = _mm256_setzero_ps();
			__m256 mv2 = _mm256_setzero_ps();
			__m256 mv3 = _mm256_setzero_ps();
			__m256 mw0 = _mm256_setzero_ps();
			__m256 mw1 = _mm256_setzero_ps();
			__m256 mw2 = _mm256_setzero_ps();
			__m256 mw3 = _mm256_setzero_ps();
			for (int k = 0; k < d; k++)
			{
				const __m256 mr0 = _mm256_lddqu_ps(si + offset[k] + 0);
				const __m256 mr1 = _mm256_lddqu_ps(si + offset[k] + 8);
				const __m256 mr2 = _mm256_lddqu_ps(si + offset[k] + 16);
				const __m256 mr3 = _mm256_lddqu_ps(si + offset[k] + 24);
				__m256 mlw0 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(_mm256_sub_ps(mr0, ma0))), 4));
				__m256 mlw1 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(_mm256_sub_ps(mr1, ma1))), 4));
				__m256 mlw2 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(_mm256_sub_ps(mr2, ma2))), 4));
				__m256 mlw3 = _mm256_mul_ps(_mm256_set1_ps(space[k]), _mm256_i32gather_ps(rweight, _mm256_cvtps_epi32(_mm256_abs_ps(_mm256_sub_ps(mr3, ma3))), 4));
				mv0 = _mm256_fmadd_ps(mlw0, mr0, mv0);
				mv1 = _mm256_fmadd_ps(mlw1, mr1, mv1);
				mv2 = _mm256_fmadd_ps(mlw2, mr2, mv2);
				mv3 = _mm256_fmadd_ps(mlw3, mr3, mv3);
				mw0 = _mm256_add_ps(mlw0, mw0);
				mw1 = _mm256_add_ps(mlw1, mw1);
				mw2 = _mm256_add_ps(mlw2, mw2);
				mw3 = _mm256_add_ps(mlw3, mw3);
			}

			if constexpr (postprocess == 0)
			{
				_mm256_storeu_ps(dptr + i + 0, _mm256_div_ps(mv0, mw0));
				_mm256_storeu_ps(dptr + i + 8, _mm256_div_ps(mv1, mw1));
				_mm256_storeu_ps(dptr + i + 16, _mm256_div_ps(mv2, mw2));
				_mm256_storeu_ps(dptr + i + 24, _mm256_div_ps(mv3, mw3));
			}
			else if constexpr (postprocess == 1)
			{
				const float* s = stddev.ptr<float>(j, i);

				_mm256_storeu_ps(dptr + i + 0, _mm256_mul_ps(_mm256_loadu_ps(s), _mm256_div_ps(mv0, mw0)));
				_mm256_storeu_ps(dptr + i + 8, _mm256_mul_ps(_mm256_loadu_ps(s + 8), _mm256_div_ps(mv1, mw1)));
				_mm256_storeu_ps(dptr + i + 16, _mm256_mul_ps(_mm256_loadu_ps(s + 16), _mm256_div_ps(mv2, mw2)));
				_mm256_storeu_ps(dptr + i + 24, _mm256_mul_ps(_mm256_loadu_ps(s + 24), _mm256_div_ps(mv3, mw3)));
			}
			else if constexpr (postprocess == 2)
			{
				const float* s = stddev.ptr<float>(j, i);

				_mm256_storeu_ps(dptr + i + 0, _mm256_mul_ps(_mm256_set1_ps(lastexp), _mm256_div_ps(mv0, _mm256_mul_ps(_mm256_loadu_ps(s), mw0))));
				_mm256_storeu_ps(dptr + i + 8, _mm256_mul_ps(_mm256_set1_ps(lastexp), _mm256_div_ps(mv1, _mm256_mul_ps(_mm256_loadu_ps(s + 8), mw1))));
				_mm256_storeu_ps(dptr + i + 16, _mm256_mul_ps(_mm256_set1_ps(lastexp), _mm256_div_ps(mv2, _mm256_mul_ps(_mm256_loadu_ps(s + 16), mw2))));
				_mm256_storeu_ps(dptr + i + 24, _mm256_mul_ps(_mm256_set1_ps(lastexp), _mm256_div_ps(mv3, _mm256_mul_ps(_mm256_loadu_ps(s + 24), mw3))));
			}

		}
	}

	if (srcf.depth() != CV_32F) destf.convertTo(dst, src.type());
}

void rangeBlurFilter(const Mat& src, Mat& dst, const int r, const float sigma_range, const float sigma_space = -1.f, const int postProcessType = 0, const int borderType = cv::BORDER_DEFAULT)
{
	if (postProcessType == 0)rangeBlurFilter_<0>(src, dst, r, sigma_range, sigma_space, borderType);
	if (postProcessType == 1)rangeBlurFilter_<1>(src, dst, r, sigma_range, sigma_space, borderType);
	if (postProcessType == 2)rangeBlurFilter_<2>(src, dst, r, sigma_range, sigma_space, borderType);
}

template<int postprocess>
void rangeBlurFilterRef_(const Mat& src, Mat& dst, const int r, const float sigma)
{
	CV_Assert(src.depth() == CV_32F);

	dst.create(src.size(), src.type());
	cv::Mat ave, stddev;
	meanStdFilter(src, ave, stddev, r);
	const float coeff = 1.f / (-2.f * sigma * sigma);
	for (int j = r; j < src.rows - r; j++)
	{
		float* dptr = dst.ptr<float>(j);
		const float* sptr = src.ptr<float>(j);
		const float* avep = ave.ptr<float>(j);

		for (int i = r; i < src.cols - r; i++)
		{
			const float a = avep[i];
			const float s = sptr[i];
			float w = 0.f;
			for (int l = -r; l <= r; l++)
			{
				const float* kptr = src.ptr<float>(j + l);//kernel
				for (int k = -r; k <= r; k++)
				{
					const float vvv = kptr[i + k] - a;
					w += exp(vvv * vvv * coeff);
				}
			}
			float v = s * w;
			if constexpr (postprocess == 0) dptr[i] = v / w;
			else if constexpr (postprocess == 1) dptr[i] = v / w * stddev.at<float>(j, i);
			else if constexpr (postprocess == 2) dptr[i] = exp(-1.f * v / (w * stddev.at<float>(j, i)));
		}
	}
}

void rangeBlurFilterRef(const Mat& src, Mat& dst, const int r, const float sigma_range, const int postProcessType = 0)
{
	if (postProcessType == 0)rangeBlurFilterRef_<0>(src, dst, r, sigma_range);
	if (postProcessType == 1)rangeBlurFilterRef_<1>(src, dst, r, sigma_range);
	if (postProcessType == 2)rangeBlurFilterRef_<2>(src, dst, r, sigma_range);
}

int main(int argc, char** argv)
{
	/*__m256i a = _mm256_set_step_epi32(0);
	__m256i b = _mm256_set_step_epi32(8);
	__m256i c = _mm256_set_step_epi32(16);
	__m256i d = _mm256_set_step_epi32(24);

	__m256i w = _mm256_cmpgt_epi32(a, _mm256_set1_epi32(2));
	print_int(w);
	print_int(_mm256_andnot_si256(w, b));

	//print_uchar(_mm256_packus_epi16(_mm256_packus_epi32(a, b), _mm256_packus_epi32(c, d)));
	return 0;*/
	/*
	cout << getInformation() << endl; return 0;
	cout << cv::getBuildInformation() << endl;
	cv::ipp::setUseIPP(false);
	cv::setUseOptimized(false);
	*/
	//testUnnormalizedBilateralFilter(); return 0;
	//testMultiScaleFilter(); return 0;

	//testIsSame(); return 0;


	//detailTest(); return 0;
#pragma region setup
	//Mat img = imread("img/lenna.png");
	Mat img = imread("img/Kodak/kodim07.png");
	Mat imgg; cvtColor(img, imgg, COLOR_BGR2GRAY);

	Mat aa = convert(imgg, CV_32F);
	Mat t0, t1;
	//rangeBlurFilterRef(aa, t0, 5, 3);
	//rangeBlurFilter(aa, t1, 5, 3);
	//guiAlphaBlend(convert(t0,CV_8U), convert(t1,CV_8U));
	//Mat img = imread("img/cameraman.png",0);
	//Mat img = imread("img/barbara.png", 0);
	//filter2DTest(img); return 0;

#pragma endregion


#pragma region core
	//guiPixelizationTest();
	//testStreamConvert8U(); return 0;
	//testKMeans(img); return 0;
	//testTiling(img); return 0;
	//copyMakeBorderTest(img); return 0;
	//testSplitMerge(img); return 0;
	//consoleImageTest(); return 0;
	//testConcat(); return 0;
	//testsimd(); return 0;

	//testHistogram(); return 0;
	//testPlot(); return 0;
	//testPlot2D(); return 0;

	//guiHazeRemoveTest();

	//testCropZoom(); return 0;
	//testAddNoise(img); return 0;
	//testLocalPSNR(img); return 0;
	//testPSNR(img); return 0;
	//resize(img, a, Size(513, 513));
	//testHistgram(img);
	//testRGBHistogram();
	//testRGBHistogram2();
	//testTimer(img);
	//testMatInfo(); return 0;
	//testStat(); return 0;
	//testDestinationTimePrediction(img); return 0;
	//testAlphaBlend(left, right);
	//testAlphaBlendMask(left, right);
	//guiDissolveSlide(left, dmap);
	//guiLocalDiffHistogram(img);
	//guiContrast(img);
	//guiContrast(guiCropZoom(img));
	//testVideoSubtitle();
#pragma endregion

#pragma region imgproc
	//guiCvtColorPCATest(); return 0;
#pragma endregion

#pragma region stereo
	//testStereoBase(); return 0;
	//testCVStereoBM(); return 0;
	//testCVStereoSGBM(); return 0;
#pragma endregion

#pragma region filter
	//testGuidedImageFilter(Mat(), Mat()); return 0;
	//highDimentionalGaussianFilterTest(img); return 0;
	//highDimentionalGaussianFilterHSITest(); return 0;
	guiDenoiseTest(img);
	//testWeightedHistogramFilterDisparity(); return 0;
	//testWeightedHistogramFilter();return 0;
#pragma endregion 

	//guiUpsampleTest(img); return 0;
	//guiDomainTransformFilterTest(img);
	//guiMedianFilterTest(img);
	//VisualizeDenormalKernel vdk;
	//vdk.run(img);
	//return 0;
	//VizKernel vk;
	//vk.run(img, 2);


	//guiShift(left,right); return 0;
	//
	//iirGuidedFilterTest2(img); return 0;
	//iirGuidedFilterTest1(dmap, left); return 0;
	//iirGuidedFilterTest(); return 0;
	//iirGuidedFilterTest(left); return 0;
	//fitPlaneTest(); return 0;
	//guiWeightMapTest(); return 0;


	//guiGeightedJointBilateralFilterTest();
	//guiHazeRemoveTest();
	//Mat ff3 = imread("img/pixelart/ff3.png");

	Mat src = imread("img/lenna.png");

	//Mat src = imread("img/Kodak/kodim07.png",0);
	//guiIterativeBackProjectionTest(src);
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
	//Mat disp = imread("img/stereo/Dolls/disp1.png", 0);
	//	Mat src;
	Mat dest;

	//guiCrossBasedLocalFilter(src); return 0;


	//eraseBoundary(src,10);
	//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiDualBilateralFilterTest(src,disp);
	//guiGausianFilterTest(src); return 0;

	//guiCoherenceEnhancingShockFilter(src, dest);

	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	//guiDisparityPlaneFitSLICTest(src, ref, disp); return 0;
	//getPSNRRealtimeO1BilateralFilterKodak();
	//guiRealtimeO1BilateralFilterTest(src); return 0;

	Mat flashImg = imread("img/flash/cave-flash.png");
	Mat noflashImg = imread("img/flash/cave-noflash.png");
	Mat noflashImgGray; cvtColor(noflashImg, noflashImgGray, COLOR_BGR2GRAY);
	Mat flashImgGray; cvtColor(flashImg, flashImgGray, COLOR_BGR2GRAY);
	Mat fmega, nmega;
	resize(flashImgGray, fmega, Size(1024, 1024));
	resize(noflashImg, nmega, Size(1024, 1024));

	//guiEdgePresevingFilterOpenCV(src);
	//guiSLICTest(src);


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

	//Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);
	//Mat disparity = imread("img/teddy_disp1.png", 0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiCodingDistortionRemoveTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	//application 
	//guiDetailEnhancement(src);
	//guiDomainTransformFilterTest(mega);
	return 0;
}