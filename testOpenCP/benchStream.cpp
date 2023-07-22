#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void benchStreamSet()
{
	string wname = "stream";
	cp::ConsoleImage ci(Size(700, 300));
	//namedWindow(wname);
	int iteration = 5; createTrackbar("iteration", "", &iteration, 1000);
	int size = 3;  createTrackbar("size", "", &size, 8);
	int th = 4;  createTrackbar("thread", "", &th, omp_get_max_threads());

	int key = 0;
	cp::Timer t("", TIME_MSEC, false);
	cv::Size imsize = Size(1024, 1024);
	Mat a(imsize, CV_32F);
	while (key != 'q')
	{
		const int w = 128 * (int)pow(2, size);
		const cv::Size imsize = Size(w, w);
		if (a.size() != imsize)a.create(imsize, CV_32F);
	
		omp_set_num_threads(th);
		const int thread_max = omp_get_max_threads();

		ci("Press ctrl+p for parameter setting");
		ci("Size %d x %d", w, w);
		ci("threads %d", thread_max);
		
		auto benchSet = [&](int iteration, bool parallel, int unroll)
		{
			t.clearStat();
			for (int i = 0; i < iteration; i++)
			{
				t.start();
				streamSet(a, 0.f, parallel, unroll);
				t.pushLapTime();
			}
		};

		benchSet(iteration, false, 1);
		ci("streamSet S1 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, false, 2);
		ci("streamSet S2 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, false, 4);
		ci("streamSet S4 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, false, 8);
		ci("streamSet S8 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 1);
		ci("streamSet P1 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 2);
		ci("streamSet P2 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 4);
		ci("streamSet P4 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 8);
		ci("streamSet P8 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);

		ci.show();
		key = waitKey(1);
	}

	//_mm_free(data_a);
	//_mm_free(data_b);
}