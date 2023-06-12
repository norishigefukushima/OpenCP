#include <opencp.hpp>
#include <spatialfilter/SpatialFilter.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cp;

void testSpatialFilter(Mat& src)
{
	string wname = "testSpatialFilter";
	ConsoleImage ci;
	namedWindow(wname);
	int algo = 10; createTrackbar("algorithm", wname, &algo, (int)SpatialFilterAlgorithm::BOX);	//(int)SpatialFilterAlgorithm::SIZE - 1
	int sigma10 = 20;  createTrackbar("sigma*0.1", wname, &sigma10, 200);
	int order = 5;  createTrackbar("order", wname, &order, 20);
	int type = 1;  createTrackbar("type", wname, &type, 2);
	int borderType = cv::BORDER_REFLECT; createTrackbar("border", wname, &borderType, 4);
	int key = 0;
	Mat show;
	Timer tfirst("", TIME_MSEC, false);
	Timer t("", TIME_MSEC, false);
	UpdateCheck ucs(sigma10, borderType);
	UpdateCheck uc(algo, order, type);
	Mat ref;

	while (key != 'q')
	{
		const bool isRefUpdate = ucs.isUpdate(sigma10, borderType);
		if (isRefUpdate)
		{
			SpatialFilter reff(SpatialFilterAlgorithm::FIR_KAHAN, CV_64F, SpatialKernel::GAUSSIAN, 0);
			reff.filter(src, ref, sigma10 * 0.1, 9, borderType);
		}
		if (uc.isUpdate(algo, order, type) || isRefUpdate)
		{
			tfirst.clearStat();
			t.clearStat();
		}

		SpatialFilterAlgorithm algorithm = SpatialFilterAlgorithm(algo);
		int desttype = (type == 0) ? CV_8U : (type == 1) ? CV_32F : CV_64F;
		
		SpatialFilter sf(algorithm, desttype, SpatialKernel::GAUSSIAN, 0);//const DCT_COEFFICIENTS dct_coeff = (option == 0) ? DCT_COEFFICIENTS::FULL_SEARCH_OPT : DCT_COEFFICIENTS::FULL_SEARCH_NOOPT;
		
		tfirst.start();
		sf.filter(src, show, sigma10 * 0.1, order, borderType);
		tfirst.pushLapTime();
		t.start();
		sf.filter(src, show, sigma10 * 0.1, order, borderType);
		t.pushLapTime();

		imshowScale(wname, show);

		ci("%d: sigma = %f", t.getStatSize(), sigma10 * 0.1);
		ci("algorith: " + cp::getAlgorithmName(algorithm));
		ci("order:  %d", cp::clipOrder(order, algorithm));
		ci("type: " + cp::getDepthName(desttype));
		ci("border: " + cp::getBorderName(borderType));
		ci("TIME: %f ms (1st)", tfirst.getLapTimeMedian());
		ci("TIME: %f ms (2nd)", t.getLapTimeMedian());
		ci("PSNR: %f dB", getPSNR(ref, show));
		ci.show();
		key = waitKey(1);
	}
}
