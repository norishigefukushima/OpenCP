#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testKMeans(Mat& src)
{
	string wname = "KMeans";
	Mat data, dataTr;

	namedWindow(wname);
	int ch = 7; createTrackbar("channel", wname, &ch, 128);
	setTrackbarMin("channel", wname, 1);
	int size = 16; createTrackbar("size", wname, &size, 100);
	setTrackbarMin("size", wname, 3);
	const int pad = 2;
	vector<Mat> vdata;
	vector<Mat> vdata2(ch);
	src.convertTo(data, CV_32F);
	split(data, vdata);
	for (int i = 0; i < ch; i++)vdata2[i] = vdata[i % src.channels()].clone();
	for (int i = 0; i < ch; i++)
	{
		vdata2[i].create(Size((int)pow(2, size) + pad, 1), CV_32F);
		randn(vdata2[i], 0, 5);
	}
	merge(vdata2, data);

	//data = data.reshape(1, src.size().area());
	data = data.reshape(1, vdata2[0].cols);
	transpose(data, dataTr);

	int k = 5; createTrackbar("K", wname, &k, 100);
	setTrackbarMin("K", wname, 1);
	int iterations = 10; createTrackbar("iteration", wname, &iterations, 100);
	int attempts = 1;
	Mat label, centroids;
	cv::TermCriteria criteria(cv::TermCriteria::COUNT, iterations, 1);

	int trials = 10; createTrackbar("trials", wname, &trials, 100);

	int key = 0;
	vector<Stat> st(10);
	ConsoleImage ci(Size(640, 480), wname);
	UpdateCheck uc(ch, size);
	UpdateCheck uc2(k, iterations);
	cp::Timer t("", TIME_MSEC);
	cp::Timer ttotal("", TIME_MSEC);
	cp::KMeans km;
	cp::KMeans km2;

	Mat srcf; src.convertTo(srcf, CV_32F);
	resize(srcf, srcf, Size(), 0.125, 0.125);
	srcf = srcf.reshape(1, srcf.cols*srcf.rows);
	km.gui(srcf, k, label, criteria, attempts, cv::KMEANS_PP_CENTERS, centroids, KMeans::MeanFunction::GaussInv, KMeans::Schedule::SoA_KND);

	while (key != 'q')
	{
		if (uc.isUpdate(ch, size))
		{
			vector<Mat> vdata2(ch);
			//for (int i = 0; i < ch; i++)vdata2[i] = vdata[i % src.channels()].clone();

			for (int i = 0; i < ch; i++)
			{
				vdata2[i].create(Size((int)pow(2, size) + pad, 1), CV_32F);
				randn(vdata2[i], 0, 5);
			}

			merge(vdata2, data);
			//data = data.reshape(1, src.size().area());
			data = data.reshape(1, vdata2[0].cols);
			transpose(data, dataTr);
			print_matinfo(data);
			for (int i = 0; i < st.size(); i++)st[i].clear();
		}
		if (uc2.isUpdate(k, iterations))
		{
			for (int i = 0; i < st.size(); i++)st[i].clear();
		}

		ttotal.start();
		int index = 0;
		//#define INIT
#ifdef INIT
		//int method = cv::KMEANS_RANDOM_CENTERS;
		cv::kmeans(data, k, label, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, centroids);
		int method = cv::KMEANS_USE_INITIAL_LABELS;
		Mat label2 = label.clone();

		label2.copyTo(label);
		cv::kmeans(data, k, label, criteria, attempts, method, centroids);
		Mat ans = centroids.clone();
		cout << "=====ans=====" << endl;
		cout << ans << endl;
		cout << "=============" << endl;
		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			cv::kmeans(data, k, label, criteria, attempts, method, centroids);
			//cout << i<<" "; print_mat(centroids);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cv::kmeans        "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

		//cout << "cv: "; print_mat(centroids);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			cp::kmeans(data, k, label, criteria, attempts, method, centroids);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cp::kmeans2       "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::Auto);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cp::kmeans auto "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::AoS_NKD);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cp::kmeans AoS_NKD "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::AoS_KND);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cp::kmeans AoS_KND "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::SoA_KND);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cp::kmeans SoA_KND "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

		t.start();
		for (int i = 0; i < trials; i++)
		{
			label2.copyTo(label);
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::SoA_NKD);
		}
		st[index++].push_back(t.getTime() / trials);
		cout << "cp::kmeans SoA_NKD "; if (!cp::isSameMat(ans, centroids)) { cout << centroids << endl; }

#else
		int method = cv::KMEANS_RANDOM_CENTERS;
		t.start();
		Mat label2 = label.clone();
		for (int i = 0; i < trials; i++)
		{
			cv::kmeans(data, k, label, criteria, attempts, method, centroids);
		}
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			cp::kmeans(data, k, label, criteria, attempts, method, centroids);
		}
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::Auto);
		}
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::AoS_NKD);
		}
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			km2.clustering(data, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::AoS_KND);
		}
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			km.clustering(dataTr, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::SoA_KND);
		}
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
		{
			km.clustering(dataTr, k, label, criteria, attempts, method, centroids, KMeans::MeanFunction::Mean, KMeans::Schedule::SoA_NKD);
		}
		st[index++].push_back(t.getTime() / trials);
#endif
#ifdef KPP
		t.start();
		for (int i = 0; i < trials; i++)
			cv::kmeans(data, k, label, criteria, attempts, cv::KMEANS_PP_CENTERS, centroids);
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
			cp::kmeans(data, k, label, criteria, attempts, cv::KMEANS_PP_CENTERS, centroids);
		st[index++].push_back(t.getTime() / trials);

		t.start();
		for (int i = 0; i < trials; i++)
			km.clustering(data, k, label, criteria, attempts, cv::KMEANS_PP_CENTERS, centroids);
		st[index++].push_back(t.getTime() / trials);
#endif
		st[index++].push_back(ttotal.getTime());

		index = 0;

		ci("global:trials %d", st[0].getSize());
		ci("local :trials %d", trials);
		ci("ch %d, size %d, sizeSqrt %d", ch, (int)pow(2, size), (int)sqrt(pow(2, size)));
		ci("kmeans iteration %d", iterations);
		ci("cv::kmeans      %f ms", st[index++].getMean());
		ci("cp::kmeans      %f ms", st[index++].getMean());
		ci("Kmeans Auto     %f ms", st[index++].getMean());
		ci("Kmeans AoSKND   %f ms", st[index++].getMean());
		ci("Kmeans AoSNKD   %f ms", st[index++].getMean());
		ci("Kmeans SoAKND   %f ms", st[index++].getMean());
		ci("Kmeans SoAKND   %f ms", st[index++].getMean());
		ci("Total           %f ms", st[index++].getMean());
#ifdef KPP
		ci("km.clutering++ %f ms", st[index++].getMean());
		ci("cv::Kmeans++   %f ms", st[index++].getMean());
		ci("cp::Kmeans++   %f ms", st[index++].getMean());
		ci("km.clutering++ %f ms", st[index++].getMean());
#endif
		ci.show();
		key = waitKey(1);
		if (key == 't')ci.push();
	}

}