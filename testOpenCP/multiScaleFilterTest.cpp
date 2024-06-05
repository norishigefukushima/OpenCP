#include <opencp.hpp>
#include <multiscalefilter/MultiScaleFilter.hpp>

using namespace std;
using namespace cv;
using namespace cp;

struct MouseMSFParameter
{
	cv::Rect pt;
	std::string wname;
	MouseMSFParameter(int x, int y, int width, int height, std::string name)
	{
		pt = cv::Rect(x, y, width, height);
		wname = name;
	}
};

static void guiMouseMSFOnMouse(int event, int x, int y, int flags, void* param)
{
	MouseMSFParameter* retp = (MouseMSFParameter*)param;

	if (flags == EVENT_FLAG_LBUTTON)
	{
		retp->pt.x = max(0, min(retp->pt.width - 1, x));
		retp->pt.y = max(0, min(retp->pt.height - 1, y));

		setTrackbarPos("x", retp->wname, x);
		setTrackbarPos("y", retp->wname, y);
	}
}

void testMultiScaleFilter()
{
	const bool isShowPSNR = true;
	//const bool isShowPSNR = false;

	//const bool isComputeReference = false;
	const bool isComputeReference = false;
	const bool isShowTileDiv = false;

	//const bool isDrawRemap = true;
	const bool isDrawRemap = false;

	constexpr bool isClassicScaleSpace = false;//false
	//constexpr bool isClassicScaleSpace = true;//true

	//const bool isPlotProfile = true;
	const bool isPlotProfile = false;

	//const bool isShowNaive = true;
	const bool isShowNaive = false;
	//const bool isShowCubic = true;
	const bool isShowCubic = false;
	//const bool isShowLinear = true;
	const bool isShowLinear = false;
	//const bool isShowFourier = true;
	const bool isShowFourier = false;

	constexpr bool isPyramid = true;//true
	//constexpr bool isPyramid = false;//false
	//constexpr bool isDoG = true;//true
	constexpr bool isDoG = false;//false

	Size tilediv = Size(4, 4);

	const string wname = "profile image";//profile image
	const string wnameImage = "image";//image
	namedWindow(wname);
	namedWindow(wnameImage);
	int showsw2 = 0; createTrackbar("show", wnameImage, &showsw2, 5);
	int dogapprox = 0; createTrackbar("dogapprox", wnameImage, &dogapprox, 1);
	int scaleMethodSW = 0; createTrackbar("scale", wnameImage, &scaleMethodSW, 5);

	static MouseMSFParameter param(512 / 2, 512 / 2, 512, 512, wname);
	setMouseCallback(wname, (MouseCallback)guiMouseMSFOnMouse, (void*)&param);

	int signal_w = 50;
	param.pt.x = 55;
	param.pt.y = 112;
	param.wname = wname;

	int showsw = 0; createTrackbar("showsw", wname, &showsw, 5);
	cv::createTrackbar("y", wname, &param.pt.y, 511);
	cv::createTrackbar("x", wname, &param.pt.x, 511);
	cv::createTrackbar("window_r", wname, &signal_w, 511);

	vector<Mat> img(100);
	char filename[38][50];

#pragma region read image
	sprintf(filename[0], "img/flower.png");
	//sprintf(filename[0], "img/lenna.png");
	//sprintf(filename[0], "img/kodak/kodim03.png");
	//sprintf(filename[0], "img/ex/image012.png");

	//for (int i = 1; i < 25; i++)
	for (int i = 1; i < 13; i++)
	{
		if (i <= 9)
			sprintf_s(filename[i], "img/ex/image00%d.png", i);
		else
			sprintf_s(filename[i], "img/ex/image0%d.png", i);

	}
	for (int i = 1; i <= 24; i++)
	{
		if (i <= 9)
			sprintf_s(filename[i + 12], "img/kodak/kodim0%d.png", i);
		else
			sprintf_s(filename[i + 12], "img/kodak/kodim%d.png", i);
	}
#pragma endregion

#pragma region window setup
	cv::namedWindow("trackbar");
	cv::moveWindow("trackbar", 300, 600);
	int im = 0; createTrackbar("img", "trackbar", &im, 36);
	int Color = 0; cv::createTrackbar("Color", "trackbar", &Color, 1);
	int sigma_r_ = 300; createTrackbar("sigma_range", "trackbar", &sigma_r_, 1000);
	int sigma_s_ = 10; createTrackbar("sigma_space", "trackbar", &sigma_s_, 200);
	int level = 2; cv::createTrackbar("level", "trackbar", &level, 10); setTrackbarMin("level", "trackbar", 1);
	int detail_param_ = 5; cv::createTrackbar("detail_param", "trackbar", &detail_param_, 10); cv::setTrackbarMin("detail_param", "trackbar", -5);
	int order = 20; cv::createTrackbar("order", "trackbar", &order, 300); cv::setTrackbarMin("order", "trackbar", 1);
	int wid = 100;  cv::createTrackbar("width", "trackbar", &wid, 512);
	int heigh = 100; cv::createTrackbar("height", "trackbar", &heigh, 512);
	int a = 160;  cv::createTrackbar("a", "trackbar", &a, 512);
	int b = 60; cv::createTrackbar("b", "trackbar", &b, 512);
	int isPlot = 0; cv::createTrackbar("isPlotRemap", "trackbar", &isPlot, 1);
	int WindowType = WindowType::GAUSS; cv::createTrackbar("WindowType", "trackbar", &WindowType, 3);
	//int periodMethod = LocalMultiScaleFilterFourier::Period::OPTIMIZE; 
	int periodMethod = LocalMultiScaleFilterFourier::Period::GAUSS_DIFF;
	cv::createTrackbar("PeriodMethod", "trackbar", &periodMethod, 2);
	int alpha_ = 5; cv::createTrackbar("alpha", "trackbar", &alpha_, 1000);
	int beta_ = 75; cv::createTrackbar("beta", "trackbar", &beta_, 1000);
	int Ssigma_ = 500; cv::createTrackbar("Ssigma", "trackbar", &Ssigma_, 1000);
	int pyramidComputeMethod = MultiScaleFilter::PyramidComputeMethod::IgnoreBoundary; cv::createTrackbar("pyramidMethod", "trackbar", &pyramidComputeMethod, 3);
	int cubic100 = 50; cv::createTrackbar("cubic", "trackbar", &cubic100, 200);
	int isFull = 0; //0: full computation for reference
	cv::createTrackbar("full", "trackbar", &isFull, 1);
	int adaptiveMethod = MultiScaleFilter::AdaptiveMethod::FIX;
	//int adaptiveMethod = MultiScaleFilter::AdaptiveMethod::ADAPTIVE;
	cv::createTrackbar("isAdaptive", "trackbar", &adaptiveMethod, 1);
	int adaptive_r = 255; cv::createTrackbar("adaptive_r", "trackbar", &adaptive_r, 512);
	int sr_amp = 10; cv::createTrackbar("sr_amp", "trackbar", &sr_amp, 30);
	int bt_amp = 20; cv::createTrackbar("bt_amp", "trackbar", &bt_amp, 100);
	int minmax = 1; cv::createTrackbar("minmax", "trackbar", &minmax, 1);
	int FourierSchedule = 1; cv::createTrackbar("FourierSchedule", "trackbar", &FourierSchedule, 5);
	int iteration = 1;  cv::createTrackbar("iteration", "trackbar", &iteration, 100);
	//omp_set_num_threads(1);
#pragma endregion

#pragma region variable
	MultiScaleGaussianFilter msgf;
	MultiScaleBilateralFilter msbf;

	LocalMultiScaleFilter lmsf;
	LocalMultiScaleFilterFull lmsffull;
	LocalMultiScaleFilterInterpolation lmsInter;
	TileLocalMultiScaleFilterInterpolation tlmsi;
	FastLLFReference fllfr;
	LocalMultiScaleFilterFourierReference llffr;
	LocalMultiScaleFilterFourier lmsFourier;
	TileLocalMultiScaleFilterFourier tllff;

	float Salpha, Sbeta, Ssigma;

	Mat src, srcf, dest;
	Mat destPyramidFull, destPyramidNaive, destPyramidNN, destPyramidLinear, destPyramidCubic, destPyramidFourier, destPyramidFLLFRef, destPyramidFourierRef;
	Mat destTilePyramidNN, destTilePyramidLinear, destTilePyramidCubic, destTilePyramidFourier;
	Mat destDoGNaive, destDoGNN, destDoGLinear, destDoGCubic, destDoGFourier;
	Mat destPyramid, destContrastPyramid, destDoG, destCoG, destDoBF, destCoBF;
	Mat destUNBF, destUNBFM;
	Mat fullLLFPrecomp;

	Mat showApprox;

	cp::Stat time_full_py, time_PyNnaive, time_PyNN, time_PyLinear, time_PyCubic, time_PyFourier, time_fllfref_py, time_FourierRef_py;
	cp::Stat time_TileNN_py, time_Tilelinear_py, time_Tilecubic_py, time_TileFourier_py;
	cp::Stat time_DoGNaive, time_DoGNN, time_DoGLinear, time_DoGCubic, time_DoGFourier;

	bool count_flag = false;
	int T_ = 440;
	vector<int> num(10); num[0] = 17; num[1] = 20; num[2] = 25; num[3] = 32; num[4] = 33; num[5] = 36; num[6] = 19; num[7] = 20; num[8] = 21; num[9] = 24;
	//9, 12, 13
	vector<float> order_count_compre(40), order_count_fastLLF(40);
	int count = 0, ct = 0;
	float time_compre = 0.f, time_linear = 0.f, time_tile = 0.f;

	Scalar mean, stddev;
#pragma endregion

	string EnhanceName[3];
	EnhanceName[0] = "NORMAL"; EnhanceName[1] = "ADAPTIVE"; EnhanceName[2] = "EDGE";
	int key = 0;
	cp::UpdateCheck uc(adaptiveMethod, pyramidComputeMethod, level, order, sigma_s_, periodMethod, Color, FourierSchedule);
	cp::UpdateCheck ucfull(im, Color, sigma_r_, sigma_s_, detail_param_, level, pyramidComputeMethod);
	cp::UpdateCheck ucim(im, Color);


	cp::ConsoleImage ci(Size(800, 1000), "info");

	cp::Plot profile(Size(512, 512));
	profile.setPlotTitle(0, "Input");
	profile.setPlotTitle(1, "Pyramid");
	profile.setPlotTitle(2, "DoG");
	profile.setPlotTitle(3, "LLF");
	profile.setPlotTitle(4, "LDF");
	profile.setPlotTitle(5, "DoBF");

	//omp_set_num_threads(1);
	while (key != 'q')
	{
#pragma region setup
		cp::Timer time_total("total", 0, false);

		if (ucim.isUpdate(im, Color))
		{
			src = imread(filename[im], Color);
			if (src.empty())
			{
				std::cout << "file open fail " << im << std::endl;
			}
			resize(src, src, Size(768, 512));
			//resize(src, src, Size(512, 768));
			//resize(src, src, Size(512, 512));
			//resize(src, src, Size(256, 256));
			//resize(src, src, Size(128, 128));
			//resize(src, src, Size(), 0.25, 0.125);
			src.convertTo(srcf, CV_32F);
			//max(srcf, 80, srcf);
			fullLLFPrecomp = srcf.clone();
		}
		float boost = float(detail_param_);
		float sigma_r = sigma_r_ * 0.1f;
		float sigma_s = (sigma_s_ == 0) ? sqrt(2.f) : sigma_s_ * 0.1f;
		Salpha = alpha_ * 0.1f;
		Sbeta = beta_ * 0.01f;
		Ssigma = Ssigma_ * 0.1f;
		//Ssigma = log(2.5);
#pragma endregion

#pragma region main processing

		if (isClassicScaleSpace)
		{
			msgf.setPyramidComputeMethod(MultiScaleFilter::PyramidComputeMethod::OpenCV);
			//msgf.setPyramidComputeMethod(MultiScaleFilter::PyramidComputeMethod::IgnoreBoundary);
			//msf.filter(srcf, dest, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
			msgf.filter(srcf, dest, -1.f, sigma_s, boost, level, MultiScaleFilter::Pyramid);
			cp::streamCopy(dest, destPyramid);

			msgf.filter(srcf, dest, -1.f, sigma_s, boost, level, MultiScaleFilter::ContrastPyramid);
			cp::streamCopy(dest, destContrastPyramid);
			//msf.filter(srcf, dest, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG);
			msgf.setDoGPyramidApprox(dogapprox == 1);
			msgf.filter(srcf, dest, -1.f, sigma_s, boost, level, MultiScaleFilter::DoG);
			cp::streamCopy(dest, destDoG);
			msgf.filter(srcf, dest, -1.f, sigma_s, boost, level, MultiScaleFilter::CoG);
			cp::streamCopy(dest, destCoG);
			msbf.filter(srcf, dest, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG);
			cp::streamCopy(dest, destDoBF);
			msbf.filter(srcf, dest, sigma_r, sigma_s, boost, level, MultiScaleFilter::CoG);
			cp::streamCopy(dest, destCoBF);

			if (showsw2 == 0) imshowScale(wnameImage, destPyramid);
			if (showsw2 == 1) imshowScale(wnameImage, destContrastPyramid);
			if (showsw2 == 2) imshowScale(wnameImage, destDoG);
			if (showsw2 == 3) imshowScale(wnameImage, destCoG);
			if (showsw2 == 4) imshowScale(wnameImage, destDoBF);
			if (showsw2 == 5) imshowScale(wnameImage, destCoBF);
			ci("PSNR py-dog %f", getPSNR(destPyramid, destDoG, 20));
		}

		if (isPyramid)
		{
#pragma region setting
			Mat adaptiveSigmaMap(src.size(), CV_32F, sigma_r);
			Mat adaptiveBoostMap(src.size(), CV_32F, boost);
			circle(adaptiveSigmaMap, src.size() / 2, adaptive_r, Scalar::all(sr_amp * 0.1 * sigma_r), cv::FILLED);
			circle(adaptiveBoostMap, src.size() / 2, adaptive_r, Scalar::all(bt_amp * 0.1 * boost), cv::FILLED);

			lmsf.setPyramidComputeMethod(MultiScaleFilter::PyramidComputeMethod(pyramidComputeMethod));
			lmsffull.setPyramidComputeMethod(MultiScaleFilter::PyramidComputeMethod(pyramidComputeMethod));
			lmsFourier.setPyramidComputeMethod(MultiScaleFilter::PyramidComputeMethod(pyramidComputeMethod));
			lmsInter.setPyramidComputeMethod(MultiScaleFilter::PyramidComputeMethod(pyramidComputeMethod));

			lmsf.setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod(minmax));
			lmsffull.setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod(minmax));
			lmsFourier.setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod(minmax));
			lmsInter.setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod(minmax));
			tllff.setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod(minmax));
			tlmsi.setRangeDescopeMethod(MultiScaleFilter::RangeDescopeMethod(minmax));

			lmsf.setAdaptive(adaptiveMethod, adaptiveSigmaMap, adaptiveBoostMap, level);
			lmsffull.setAdaptive(adaptiveMethod, adaptiveSigmaMap, adaptiveBoostMap, level);
			llffr.setAdaptive(adaptiveMethod, adaptiveSigmaMap, adaptiveBoostMap, level);
			lmsFourier.setAdaptive(adaptiveMethod, adaptiveSigmaMap, adaptiveBoostMap, level);
			tllff.setAdaptive(adaptiveMethod, tilediv, adaptiveSigmaMap, adaptiveBoostMap);

			lmsFourier.setIsPlot(isPlot == 1);
			if (FourierSchedule == 0)
			{
				lmsInter.setComputeScheduleMethod(false);
				tlmsi.setComputeScheduleMethod(false);
				lmsFourier.setComputeScheduleMethod(0, false, false);
				tllff.setComputeScheduleMethod(0, false, false);
			}
			if (FourierSchedule == 1)
			{
				lmsInter.setComputeScheduleMethod(true);
				tlmsi.setComputeScheduleMethod(true);
				lmsFourier.setComputeScheduleMethod(0, true, false);
				tllff.setComputeScheduleMethod(0, true, false);
			}
			if (FourierSchedule == 2)
			{
				lmsInter.setComputeScheduleMethod(true);
				tlmsi.setComputeScheduleMethod(true);
				lmsFourier.setComputeScheduleMethod(0, true, true);
				tllff.setComputeScheduleMethod(0, true, true);
			}
			if (FourierSchedule == 3)
			{
				lmsInter.setComputeScheduleMethod(false);
				tlmsi.setComputeScheduleMethod(false);
				lmsFourier.setComputeScheduleMethod(1, false, false);
				tllff.setComputeScheduleMethod(1, false, false);
			}
			if (FourierSchedule == 4)
			{
				lmsInter.setComputeScheduleMethod(true);
				tlmsi.setComputeScheduleMethod(true);
				lmsFourier.setComputeScheduleMethod(1, true, false);
				tllff.setComputeScheduleMethod(1, true, false);
			}
			if (FourierSchedule == 5)
			{
				lmsInter.setComputeScheduleMethod(true);
				tlmsi.setComputeScheduleMethod(true);
				lmsFourier.setComputeScheduleMethod(1, true, true);
				tllff.setComputeScheduleMethod(1, true, true);
			}

			lmsInter.setAdaptive(adaptiveMethod, adaptiveSigmaMap, adaptiveBoostMap, level);
			tlmsi.setAdaptive(adaptiveMethod, tilediv, adaptiveSigmaMap, adaptiveBoostMap);
			lmsFourier.setPeriodMethod(LocalMultiScaleFilterFourier::Period(periodMethod));
			tllff.setPeriodMethod(periodMethod);
			lmsInter.setCubicAlpha(-cubic100 * 0.01f);
			tlmsi.setCubicAlpha(-cubic100 * 0.01f);
#pragma endregion

#pragma region reference
			bool flag = (isFull == 1);
			//if (ucfull.isUpdate(im, Color, sigma_r_, sigma_s_, detail_param_, level, pyramidMethod) && flag)
			if (flag)
			{
				//cout<<"naive update"<<endl;
				int64 start = cv::getTickCount();
				lmsffull.filter(srcf, destPyramidFull, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
				int64 end = cv::getTickCount();
				time_full_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				destPyramidFull.copyTo(fullLLFPrecomp);
				//cp::imshowScale("full", destPyramidFull);
			}
			else
			{
				cp::streamCopy(fullLLFPrecomp, destPyramidFull);
			}

			if (ucfull.isUpdate(im, Color, sigma_r_, sigma_s_, detail_param_, level, MultiScaleFilter::Pyramid))//computing only update time for acceleration
			{
				//cp::Timer t("LLF pyramid naive");
				int64 start = cv::getTickCount();
				lmsf.filter(srcf, dest, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
				int64 end = cv::getTickCount();
				time_PyNnaive.push_back((end - start) * 1000 / cv::getTickFrequency());
				cp::streamCopy(dest, destPyramidNaive);
			}
#pragma endregion

#pragma region Gaussian Fourier pyramid
			if (isComputeReference)
			{
				//cp::Timer t("LLF Gaussian Fourier pyramid reference");
				int64 start = cv::getTickCount();
				llffr.filter(srcf, dest, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
				int64 end = cv::getTickCount();
				time_FourierRef_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				cp::streamCopy(dest, destPyramidFourierRef);
			}

			{
				//cp::Timer t("LLF Gaussian Fourier pyramid");
				lmsFourier.filter(srcf, destPyramidFourier, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					lmsFourier.filter(srcf, dest, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
					int64 end = cv::getTickCount();
					time_PyFourier.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destPyramidFourier);
				imshowScale("testFourier", destPyramidFourier);
			}

			{
				tllff.filter(tilediv, srcf, destTilePyramidFourier, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
				//cp::Timer t("Tile LLF Gaussian Fourier pyramid");
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					tllff.filter(tilediv, srcf, dest, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid);
					int64 end = cv::getTickCount();
					time_TileFourier_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destTilePyramidFourier);
			}
#pragma endregion

#pragma region Fast LLF
			if (isComputeReference)
			{
				//cp::Timer t("Fast LLF Reference");
				int64 start = cv::getTickCount();
				fllfr.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_NEAREST);
				int64 end = cv::getTickCount();
				time_fllfref_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				cp::streamCopy(dest, destPyramidFLLFRef);
			}

			{
				lmsInter.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_NEAREST);
				//cp::Timer t("LLF pyramid");
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					lmsInter.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_NEAREST);
					int64 end = cv::getTickCount();
					time_PyNN.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destPyramidNN);
			}
			{
				lmsInter.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_LINEAR);
				//cp::Timer t("LLF pyramid");
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					lmsInter.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_LINEAR);
					int64 end = cv::getTickCount();
					time_PyLinear.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destPyramidLinear);
			}
			{
				lmsInter.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_CUBIC);
				//cp::Timer t("LLF pyramid");
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					lmsInter.filter(srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_CUBIC);
					int64 end = cv::getTickCount();
					time_PyCubic.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destPyramidCubic);
			}

			{
				tlmsi.filter(tilediv, srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_NEAREST);
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					tlmsi.filter(tilediv, srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_NEAREST);
					int64 end = cv::getTickCount();
					time_TileNN_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destTilePyramidNN);
			}
			{
				tlmsi.filter(tilediv, srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_LINEAR);
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					tlmsi.filter(tilediv, srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_LINEAR);
					int64 end = cv::getTickCount();
					time_Tilelinear_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destTilePyramidLinear);
			}
			{
				tlmsi.filter(tilediv, srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_CUBIC);
				for (int i = 0; i < iteration; i++)
				{
					int64 start = cv::getTickCount();
					tlmsi.filter(tilediv, srcf, dest, 2 * order, sigma_r, sigma_s, boost, level, MultiScaleFilter::Pyramid, INTER_CUBIC);
					int64 end = cv::getTickCount();
					time_Tilecubic_py.push_back((end - start) * 1000 / cv::getTickFrequency());
				}
				cp::streamCopy(dest, destTilePyramidCubic);
			}
#pragma endregion

		}

		if (isDoG)
		{
			//DoG naive
			{
				//cp::Timer t("LLF DoGNaive", cp::TIME_MSEC, false);
				int64 start = cv::getTickCount();
				lmsf.filter(srcf, dest, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG);
				int64 end = cv::getTickCount();
				time_DoGNaive.push_back((end - start) * 1000 / cv::getTickFrequency());
				cp::streamCopy(dest, destDoGNaive);
			}

			{
				//cp::Timer t("Compressive Dog", cp::TIME_MSEC, false);
				int64 start = cv::getTickCount();
				lmsFourier.filter(srcf, destDoGFourier, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG);
				int64 end = cv::getTickCount();
				time_DoGFourier.push_back((end - start) * 1000 / cv::getTickFrequency());
			}
			{
				//cp::Timer t("LLF DoGLi", cp::TIME_MSEC, false);
				int64 start = cv::getTickCount();
				lmsInter.filter(srcf, destDoGNN, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG, INTER_NEAREST);
				int64 end = cv::getTickCount();
				time_DoGNN.push_back((end - start) * 1000 / cv::getTickFrequency());
			}
			{
				//cp::Timer t("LLF DoGLi", cp::TIME_MSEC, false);
				destDoGLinear.setTo(0);
				int64 start = cv::getTickCount();
				lmsInter.filter(srcf, destDoGLinear, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG, INTER_LINEAR);
				//destDoGLinear = srcf.clone();
				int64 end = cv::getTickCount();
				time_DoGLinear.push_back((end - start) * 1000 / cv::getTickFrequency());
			}

			{
				//cp::Timer t("LLF DoGLi", cp::TIME_MSEC, false);
				destDoGCubic.setTo(0);
				lmsInter.setCubicAlpha(-cubic100 * 0.01f);
				int64 start = cv::getTickCount();
				lmsInter.filter(srcf, destDoGCubic, order, sigma_r, sigma_s, boost, level, MultiScaleFilter::DoG, INTER_CUBIC);
				//destDoGLinear = srcf.clone();
				int64 end = cv::getTickCount();
				time_DoGCubic.push_back((end - start) * 1000 / cv::getTickFrequency());
			}

			cp::guiDiff(destDoGNaive, destDoGFourier, false);
			cv::imshow("DoGNaive", destDoGNaive / 255);
			//cv::imshow("destDoGNN", destDoGNN / 255);
			//cv::imshow("destDoGLinear", destDoGLinear / 255);
			//cv::imshow("DoGCubic", destDoGCubic / 255);
			cv::imshow("DoGFourier", destDoGFourier / 255);
		}
#pragma endregion

#pragma region console
		const int bb = 0;//(int)pow(2, level + 3);
		const int bb2 = (int)pow(2, level + 3);
		ci("# %d", time_DoGNaive.getSize());
		ci("Level %d sigma_s %5.2f py. r=%d", level, sigma_s, lmsf.getGaussianRadius(sigma_s));
		ci("sigma_r %5.2f boost %5.2f", sigma_r, boost);
		ci("order %d, #Py %d", order, 2 * order + 1);
		for (int l = 0; l < level + 1; l++)
		{
			ci("level %d, Size %04d %04d", l, lmsFourier.getLayerSize(l).width, lmsFourier.getLayerSize(l).height);
			//ci("level %d, Size %04d %04d", l, llff.getLayerSize(l).width-2, llff.getLayerSize(l).height - 2);
		}
		ci("PeriodName:" + lmsFourier.getPeriodName());
		ci(tlmsi.getComputeScheduleName());
		ci(tllff.getComputeScheduleName());
		ci("ADAPTIVE: %s", lmsFourier.getAdaptiveName().c_str());
		ci("Error: %f", lmsFourier.KernelError);
#pragma region PSNR
		ci("======PSNR======");
		if (isPyramid)
		{
			if (isShowPSNR)
			{
				if (isFull == 1)ci("Py.Full   | %6.2f dB", cp::getPSNR(destPyramidFull, destPyramidNaive, bb));
				ci("Py.Nearest| %6.2f dB", cp::getPSNR(destPyramidNN, destPyramidNaive, bb));
				ci("PyTNearest| %6.2f dB", cp::getPSNR(destTilePyramidNN, destPyramidNaive, bb));
				ci("Py.Linear | %6.2f dB", cp::getPSNR(destPyramidLinear, destPyramidNaive, bb));
				ci("PyTLinear | %6.2f dB", cp::getPSNR(destTilePyramidLinear, destPyramidNaive, bb));
				ci("Py.Cubic  | %6.2f dB", cp::getPSNR(destPyramidCubic, destPyramidNaive, bb));
				ci("PyTCubic  | %6.2f dB", cp::getPSNR(destTilePyramidCubic, destPyramidNaive, bb));
				ci("Py.Fourier| %6.2f dB", cp::getPSNR(destPyramidFourier, destPyramidNaive, bb));
				ci("PyTFourier| %6.2f dB", cp::getPSNR(destTilePyramidFourier, destPyramidNaive, bb));
				if (isComputeReference)
				{
					ci("FastLLFRef| %6.2f dB", cp::getPSNR(destPyramidFLLFRef, destPyramidNaive, bb2));
					ci("FourLLFRef| %6.2f dB", cp::getPSNR(destPyramidFourierRef, destPyramidNaive, bb2));
				}
				if (isShowNaive) cp::imshowScale("PyramidNaive", destPyramidNaive);
				if (isShowLinear) { cp::imshowScale("PyramidLinear", destPyramidLinear); cp::diffshow("PyLinear-PyNaive", destPyramidLinear, destPyramidNaive, 20.0); }
				if (isShowCubic) { cp::imshowScale("PyramidCubic", destPyramidCubic); cp::diffshow("PyCubic-PyNaive", destPyramidCubic, destPyramidNaive, 20.0); }
				if (isShowFourier) { cp::imshowScale("PyramidFourier", destPyramidFourier); cp::diffshow("PyFourier-PyNaive", destPyramidFourier, destPyramidNaive, 20.0); }
				if (isShowFourier) { cp::imshowScale("PyramidTileFourier", destTilePyramidFourier); cp::diffshow("PyTileFourier-PyNaive", destTilePyramidFourier, destPyramidNaive, 20.0); }
			}
			//{ cp::imshowScale("PyramidNearest", destPyramidNN); cp::diffshow("PyNearest-PyNaive", destPyramidNN, destPyramidNaive, 20.0); }
			//{ cp::imshowScale("PyramidTileNearest", destTilePyramidNN); cp::diffshow("PyTileNearest-PyNaive", destTilePyramidNN, destPyramidNaive, 20.0); }
			//{ cp::imshowScale("PyramidLinear", destPyramidLinear); cp::diffshow("PyTileLinear-PyNaive", destTilePyramidLinear, destPyramidNaive, 20.0); }
			//{ cp::imshowScale("PyramidTileLinear", destTilePyramidLinear); cp::diffshow("PyTileLinear-PyNaive", destTilePyramidLinear, destPyramidNaive, 20.0); }
			//cp::imshowScale("PyramidCubic", destPyramidCubic); cp::diffshow("PyCubic-PyNaive", destPyramidCubic, destPyramidNaive, 20.0);
			//cp::imshowScale("PyramidTileCubic", destTilePyramidCubic); cp::diffshow("PyTileCubic-PyNaive", destTilePyramidCubic, destPyramidNaive, 20.0);

			//{ cp::imshowScale("PyramidFourier", destPyramidFourier); cp::diffshow("PyFourier-PyNaive", destPyramidFourier, destPyramidNaive, 20.0); }
			//{ cp::imshowScale("PyramidFourierRef", destPyramidFourierRef); cp::diffshow("PyFourierRef-PyNaive", destPyramidFourierRef, destPyramidNaive, 20.0); }

		}
		if (isDoG)
		{
			ci("DoGNaive-DoGNN   | %5.2f dB", cp::getPSNR(destDoGNaive, destDoGNN, bb));
			ci("DoGNaive-DoGLin  | %5.2f dB", cp::getPSNR(destDoGNaive, destDoGLinear, bb));
			ci("DoGNaive-DoGCub  | %5.2f dB", cp::getPSNR(destDoGNaive, destDoGCubic, bb));
			ci("DoGNaive-DoGFouri| %5.2f dB", cp::getPSNR(destDoGNaive, destDoGFourier, bb));
			//ci("UNBF-DoGComp     | %5.2f dB", cp::getPSNR(destUNBFM, destDoGCompressive, bb));
		}
#pragma endregion
#pragma region Time
		ci("======TIME======");
		if (isPyramid)
		{
			if (isFull == 1)ci("Py Full   | %6.2f ms", time_full_py.getMedian());
			ci("Py.Naive  | %6.2f ms", time_PyNnaive.getMedian());
			ci("Py.Nearest| %6.2f ms", time_PyNN.getMedian());
			ci("PyTNearest| %6.2f ms", time_TileNN_py.getMedian());
			ci("Py.Linear | %6.2f ms", time_PyLinear.getMedian());
			ci("PyTLinear | %6.2f ms", time_Tilelinear_py.getMedian());
			ci("Py.Cubic  | %6.2f ms", time_PyCubic.getMedian());
			ci("PyTCubic  | %6.2f ms", time_Tilecubic_py.getMedian());
			ci("Py.Fourier| %6.2f ms", time_PyFourier.getMedian());
			ci("PyTFourier| %6.2f ms", time_TileFourier_py.getMedian());
			if (isComputeReference)
			{
				ci("FastLLFRef| %6.2f ms", time_fllfref_py.getMedian());
				ci("FourLLFRef| %6.2f ms", time_FourierRef_py.getMedian());
			}
		}
		if (isDoG)
		{
			ci("DoGnaive  | %6.2f ms", time_DoGNaive.getMedian());
			ci("DoGNN     | %6.2f ms", time_DoGNN.getMedian());
			ci("DoGLinear | %6.2f ms", time_DoGLinear.getMedian());
			ci("DoGCubic  | %6.2f ms", time_DoGCubic.getMedian());
			ci("DoGFourier| %6.2f ms", time_DoGFourier.getMedian());
			if (key == 'c')cp::guiAlphaBlend(destDoGNaive, destDoGFourier, "dogFourier   -dogNaive");
		}
#pragma endregion

#pragma endregion

#pragma region profile_plot
		if (isDrawRemap) msgf.drawRemap(false);

		if (isClassicScaleSpace)
		{
			if (isPlotProfile)
			{
				if (src.channels() == 1)
				{
					for (int i = param.pt.x - signal_w; i < param.pt.x + signal_w; i++)
					{
						const int line = param.pt.y;
						profile.push_back(i, srcf.at<float>(line, i), 0);
						//profile.push_back(i, destPyramid.at<float>(line, i), 1);
						//profile.push_back(i, destDoG.at<float>(line, i), 2);
						profile.push_back(i, destPyramidNaive.at<float>(line, i), 3);
						//profile.push_back(i, destPyramidCompressive.at<float>(line, i), 3);
						profile.push_back(i, destDoGNaive.at<float>(line, i), 4);
						profile.push_back(i, destDoBF.at<float>(line, i), 5);
					}
				}
				else
				{
					vector<Mat> gray(6);
					cvtColor(srcf, gray[0], COLOR_BGR2GRAY);
					cvtColor(destPyramid, gray[1], COLOR_BGR2GRAY);
					cvtColor(destDoG, gray[2], COLOR_BGR2GRAY);
					cvtColor(destPyramidNaive, gray[3], COLOR_BGR2GRAY);
					cvtColor(destDoGNaive, gray[4], COLOR_BGR2GRAY);
					cvtColor(destDoBF, gray[5], COLOR_BGR2GRAY);
					for (int i = param.pt.x - signal_w; i < param.pt.x + signal_w; i++)
					{
						const int line = param.pt.y;
						profile.push_back(i, gray[0].at<float>(line, i), 0);
						profile.push_back(i, gray[1].at<float>(line, i), 1);
						profile.push_back(i, gray[2].at<float>(line, i), 2);
						profile.push_back(i, gray[3].at<float>(line, i), 3);
						//profile.push_back(i, destPyramidCompressive.at<float>(line, i), 3);
						profile.push_back(i, gray[4].at<float>(line, i), 4);
						profile.push_back(i, gray[5].at<float>(line, i), 5);
					}
				}

				profile.plot("profile", false);
				profile.clear();
			}

			Mat show;
			Mat viz;
			string mes;
			if (showsw == 0)
			{
				viz = src;
				mes = "src";
			}
			if (showsw == 1)
			{
				viz = destPyramid;
				mes = "non edge preserve Pyramid";
			}
			if (showsw == 2)
			{
				viz = destDoG;
				mes = "non edge preserve DoG";
			}
			if (showsw == 3)
			{
				viz = destPyramidNaive;
				mes = "LLF";
			}
			if (showsw == 4)
			{
				viz = destDoGNaive;
				mes = "LDF";
			}
			if (showsw == 5)
			{
				viz = destDoBF;
				mes = "DoBF";
			}

			cv::displayOverlay(wname, mes, 1000);
			if (src.channels() == 1) cvtColor(viz, show, COLOR_GRAY2BGR);
			else show = viz.clone();
			cv::line(show, Point(param.pt.x - signal_w, param.pt.y), Point(param.pt.x + signal_w, param.pt.y), COLOR_RED, 2);
			cp::imshowScale(wname, show);
			Mat crop;
			cp::cropZoom(viz, crop, Rect(param.pt.x - signal_w, param.pt.y - signal_w, 2 * signal_w, 2 * signal_w));
			cp::imshowScale("crop src", crop);
		}
#pragma endregion

#pragma region key
		if (key == 'r' || uc.isUpdate(adaptiveMethod, pyramidComputeMethod, level, order, sigma_s_, periodMethod, Color, FourierSchedule))
		{
			time_PyNnaive.clear();
			time_PyNN.clear();
			time_PyLinear.clear();
			time_PyCubic.clear();
			time_PyFourier.clear();

			time_DoGNaive.clear();
			time_DoGNN.clear();
			time_DoGLinear.clear();
			time_DoGCubic.clear();
			time_DoGFourier.clear();

			time_fllfref_py.clear();
			time_FourierRef_py.clear();

			time_TileNN_py.clear();
			time_Tilelinear_py.clear();
			time_Tilecubic_py.clear();
			time_TileFourier_py.clear();
		}
		if (key == 't')
		{
			//cp::guiDiff(destDoGNaive, destDoGLinear);
			//Mat a = cp::convert(destDoGNaive, CV_8U);
			//Mat b = cp::convert(destDoGLinear, CV_8U);

			Mat a = cp::convert(destPyramidNaive, CV_8U);
			Mat b = cp::convert(destPyramidCubic, CV_8U);
			//Mat b = cp::convert(destPyramidCompressive, CV_8U);
			//Mat b = cp::convert(destPyramidFull, CV_8U);
			cp::guiAlphaBlend(a, b);
		}
		if (key == 'd')cp::guiDiff(destPyramidFourier, destPyramidNaive, "PyNN   -PyNaive");
		if (key == 'w')
		{
			cv::imwrite("S_TONE_Compressive.png", destPyramidFourier);
			cv::imwrite("S_TONE_Naive.png", destPyramidNaive);
			break;
		}
		if (key == 's')
		{
			Mat temp;
			resize(cv::Mat(srcf, cv::Rect(a, b, wid, heigh)), temp, Size(512, 512), 1);
			cv::imwrite("/save_img/cut_src.png", temp);
			resize(cv::Mat(destPyramidFourier, cv::Rect(a, b, wid, heigh)), temp, Size(512, 512), 1);
			cv::imwrite("/save_img/cut_compressive.png", temp);
			resize(cv::Mat(destPyramidLinear, cv::Rect(a, b, wid, heigh)), temp, Size(512, 512), 1);
			cv::imwrite("/save_img/cut_fastLLF.png", temp);
			break;
		}
		key = waitKey(1);

#pragma endregion

#pragma region show
		//show tile div
		if (isShowTileDiv) tllff.drawMinMax("minmax", srcf);

		string mes = "";
		if (isPyramid)
		{
			if (showsw2 == 0)
			{
				mes = "naive";
				cp::streamConvertTo8U(destPyramidNaive, showApprox);
			}
			if (showsw2 == 1)
			{
				mes = "Fourier";
				cp::streamConvertTo8U(destPyramidFourier, showApprox);
			}
			if (showsw2 == 2)
			{
				mes = "nearest";
				cp::streamConvertTo8U(destPyramidNN, showApprox);
			}
			if (showsw2 == 3)
			{
				mes = "linear";
				cp::streamConvertTo8U(destPyramidLinear, showApprox);
			}
			if (showsw2 == 4)
			{
				mes = "cubic";
				cp::streamConvertTo8U(destPyramidCubic, showApprox);
			}
			if (showsw2 == 5)
			{
				mes = "Full";
				cp::streamConvertTo8U(destPyramidFull, showApprox);
			}
			cv::imshow(wnameImage, showApprox);
		}

		cv::displayOverlay(wnameImage, mes, 1000);
		//cv::moveWindow(wnameImage, 50, 700);
		if (isPlotProfile) cv::moveWindow(wname, 50, 100);
		cv::moveWindow("crop src", 800, 100);

		cv::moveWindow("destPyramid", 600, 200);
		cv::moveWindow("destcomprePy", 1200, 200);
		cv::moveWindow("info", 600, 200);
		ci("Total     | %6.2f ms", time_total.getTime());
		ci.show();
#pragma endregion
	}
	return;
}

void testVizPyramid()
{
	const int size = 16;
	string wname = "pyramid";
	namedWindow(wname);
	int isLastConv = 0; createTrackbar("isLastConv", wname, &isLastConv, 1);
	int r = 2; createTrackbar("r", wname, &r, 5);
	int level = 4; createTrackbar("level", wname, &level, 5);
	int key = 0;
	cp::UpdateCheck uc(isLastConv, r, level);
	while (key != 'q')
	{
		if (uc.isUpdate(isLastConv, r, level))
		{
			cp::ComputePyramidSize cprySize(size, r, level, isLastConv == 1);
				imshow(wname, cprySize.vizPyramid());
				cprySize.print();
		}
		key = waitKey(1);
	}
}
