#include "stat.hpp"
#include "histogram.hpp"
#include "matinfo.hpp"

using namespace std;
using namespace cv;

namespace cp
{

	double Stat::getMin()
	{
		if (num_data == 0)return 0.0;
		double minv = DBL_MAX;
		for (int i = 0; i < num_data; i++)
		{
			minv = min(minv, data[i]);
		}
		return minv;
	}

	double Stat::getMax()
	{
		if (num_data == 0)return 0.0;
		double maxv = DBL_MIN;
		for (int i = 0; i < num_data; i++)
		{
			maxv = max(maxv, data[i]);
		}
		return maxv;
	}

	double Stat::getMean()
	{
		if (num_data == 0)return 0.0;
		double sum = 0.0;
		for (int i = 0; i < num_data; i++)
		{
			sum += data[i];
		}
		return sum / (double)num_data;
	}

	double Stat::getStd()
	{
		if (num_data == 0)return 0.0;
		double std = 0.0;
		double mean = getMean();
		for (int i = 0; i < num_data; i++)
		{
			std += (mean - data[i])*(mean - data[i]);
		}
		return sqrt(std / (double)num_data);
	}

	double Stat::getMedian()
	{
		if (num_data == 0)return 0.0;
		vector<double> v;
		vector<double> s;
		for (int i = 0; i < data.size(); i++)
		{
			s.push_back(data[i]);
		}
		cv::sort(s, v, cv::SORT_ASCENDING);
		return v[num_data / 2];
	}

	void Stat::push_back(double val)
	{
		data.push_back(val);
		num_data = (int)data.size();
	}

	void Stat::clear()
	{
		data.clear();
		num_data = 0;
	}

	void Stat::print()
	{
		cout << "mean " << getMean() << endl;
		cout << "min  " << getMin() << endl;
		cout << "med  " << getMedian() << endl;
		cout << "max  " << getMax() << endl;
		cout << "std  " << getStd() << endl;
	}

	void Stat::drawDistribution(string wname, int div)
	{
		double min = getMin();
		double max = getMax();
		drawDistribution(wname, div, min, max);
	}

	void Stat::drawDistribution(string wname, int div, double minv, double maxv)
	{
		if (data.size() == 1)return;
		double med = getMedian();
		double mean = getMean();

		vector<int> hist(div);
		for (int i = 0; i < div; i++)hist[i] = 0;

		double interval = (maxv - minv) / (div - 1);
		for (int i = 0; i < data.size(); i++)
		{
			double n = min(max((data[i] - minv), 0.0), maxv - minv);
			int v = (int)(n / interval + 0.5);
			hist[v]++;
		}

		int medv = (int)((med - minv) / interval);
		int meanv = (int)((mean - minv) / interval);

		int hmax = 0;
		for (int i = 0; i < div; i++)
		{
			if (hmax < hist[i])hmax = hist[i];
		}

		Mat draw_ = Mat::zeros(Size(div, 256), CV_8UC3);

		/*double hmaxl = log(hmax+1);
		for (int i = 0; i < div; i++)
		{
			int h = (1.0 - (double)log(hist[i]+1) / (double)hmaxl)*(draw.rows - 1);

			cv::line(draw, Point(i, h), Point(i, draw.rows - 1), cv::Scalar(230, 230, 230));
		}
		Mat text();
		addText(text, "mean", Point(10, 20), "Consolas", 20, cv::Scalar::all(255))
	*/

		for (int i = 0; i < div; i++)
		{
			if (i % (div / 10) == 0)cv::line(draw_, Point(i, 0), Point(i, draw_.rows - 1), cv::Scalar::all(50));
			int h = int((1.0 - (double)hist[i] / (double)hmax)*draw_.rows - 1);
			cv::line(draw_, Point(i, h), Point(i, draw_.rows - 1), cv::Scalar(230, 230, 230));

			if (i == meanv) cv::line(draw_, Point(i, h), Point(i, draw_.rows - 1), cv::Scalar(255, 0, 0));
			if (i == medv)cv::line(draw_, Point(i, h), Point(i, draw_.rows - 1), cv::Scalar(0, 0, 255));
			if (i == medv && i == meanv)cv::line(draw_, Point(i, h), Point(i, draw_.rows - 1), cv::Scalar(128, 0, 128));
		}
		Mat draw;
		int amp = (int)ceil(256.0 / div);
		
		resize(draw_, draw, Size(), amp, 1, INTER_NEAREST);
		Mat text_img = Mat::zeros(Size(draw.cols, 30), CV_8UC3);

		string text = format("ave%f", mean);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("med%f", med);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("min%f", minv);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("max%f", maxv);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);
		

		//static int idx = 0;
		//imwrite(format("%04d.png", idx++), draw);
		//show();
		imshow(wname, draw);
	}
}