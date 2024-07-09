#include "stat.hpp"
#include "histogram.hpp"
#include "matinfo.hpp"
#include "plot.hpp"

#include "debugcp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	int Stat::getSize()
	{
		return (int)data.size();
	}

	double Stat::getMin()
	{
		if (getSize() == 0) return 0.0;
		double minv = DBL_MAX;
		const int size = getSize();
		for (int i = 0; i < size; i++)
		{
			minv = min(minv, data[i]);
		}
		return minv;
	}

	double Stat::getMax()
	{
		if (getSize() == 0) return 0.0;
		double maxv = -DBL_MAX;
		const int size = getSize();
		for (int i = 0; i < size; i++)
		{
			maxv = max(maxv, data[i]);
		}
		return maxv;
	}

	double Stat::getSum()
	{
		if (getSize() == 0) return 0.0;
		double sum = 0.0;
		const int size = getSize();
		for (int i = 0; i < size; i++)
		{
			sum += data[i];
		}
		return sum;
	}

	double Stat::getMean()
	{
		if (getSize() == 0) return 0.0;
		return getSum() / (double)getSize();
	}

	double Stat::getVar()
	{
		if (getSize() == 0)return 0.0;
		double std = 0.0;
		double mean = getMean();
		for (int i = 0; i < getSize(); i++)
		{
			std += (mean - data[i]) * (mean - data[i]);
		}
		return std / (double)getSize();
	}

	double Stat::getStd()
	{
		if (getSize() == 0) return 0.0;
		return sqrt(getVar());
	}

	double Stat::getMedian()
	{
		if (getSize() == 0) return 0.0;
		vector<double> v;
		cv::sort(data, v, cv::SORT_ASCENDING);
		return v[getSize() / 2];
	}

	vector<int> Stat::getSortIndex(int flags)
	{
		vector<int> v;
		if (getSize() == 0) return v;
		cv::sortIdx(data, v, flags);
		return v;
	}

	void Stat::pop_back()
	{
		data.pop_back();
	}

	void Stat::push_back(double val)
	{
		data.push_back(val);
	}

	void Stat::clear()
	{
		data.clear();
	}

	void Stat::print()
	{
		cout << "samples" << getSize() << endl;
		cout << "sum " << getSum() << endl;
		cout << "mean " << getMean() << endl;
		cout << "min  " << getMin() << endl;
		cout << "med  " << getMedian() << endl;
		cout << "max  " << getMax() << endl;
		cout << "std  " << getStd() << endl;
		cout << "var  " << getVar() << endl;
	}

	void Stat::drawDistributionStep(string wname, const float step)
	{
		double min = getMin();
		double max = getMax();
		const int div = (int)ceil((max - min) / step);
		drawDistribution(wname, div, min, max);
	}

	void Stat::drawDistributionSigmaClip(string wname, const int div, const float sigmaclip)
	{
		const double ave = getMean();
		const double std = getStd();
		double domain = std * sigmaclip;
		const double min = ave - domain;
		const double max = ave + domain;
		drawDistribution(wname, div, min, max);
	}

	void Stat::drawDistributionStepSigma(string wname, const float step, const float sigma)
	{
		const double ave = getMean();
		const double std = getStd();
		double domain = (int)ceil(std * sigma / step) * step;
		const double min = ave - domain;
		const double max = ave + domain;
		const int div = int(2.0 * domain / step) + 1;
		drawDistribution(wname, div, min, max);
	}

	void Stat::drawDistribution(string wname, int div)
	{
		double min = getMin();
		double max = getMax();
		drawDistribution(wname, div, min, max);
	}

	int size_ceil(int val, int target)
	{
		return (val % target == 0) ? val : (val / target + 1) * target;
	}

	void Stat::drawDistribution(string wname, int div, double minv, double maxv)
	{
		if (div <= 2) return;
		if (data.size() == 1) return;
		namedWindow(wname);
		const int tsize = max(256, div);
		const int size = size_ceil(div, tsize);
		const int lw = tsize / div;

		double stmax = getMax();
		double stmin = getMin();
		if (stmax == stmin) return;

		const double stmed = getMedian();
		const double stmean = getMean();
		const double ststd = getStd();
		Mat draw_ = Mat::zeros(Size(div, 256), CV_8UC3);
		vector<int> hist(div);
		vector<double> histval(div);
		const double range = maxv - minv;
		const double interval = range / (div - 1);
		for (int i = 0; i < div; i++)
		{
			hist[i] = 0;
			histval[i] = i * interval + minv;
		}

		for (int i = 0; i < data.size(); i++)
		{
			const double n = min(max((data[i] - minv), 0.0), range);
			const int v = cvRound(n / interval);
			hist[v]++;
		}

		const int medv = cvRound((stmed - minv) / interval);
		const int meanv = cvRound((stmean - minv) / interval);

		int hmax = 0;
		for (int i = 0; i < div; i++)
		{
			hmax = max(hmax, hist[i]);
		}

		for (int i = 0; i < div; i++)
		{
			const int x = i;
			if (i % (div / 10) == 0) cv::line(draw_, Point(x, 0), Point(x, draw_.rows - 1), cv::Scalar::all(50));
			int h = int((1.0 - (double)hist[i] / (double)hmax) * draw_.rows - 1);
			cv::line(draw_, Point(x, h), Point(x, draw_.rows - 1), cv::Scalar(230, 230, 230));

			if (i == meanv) cv::line(draw_, Point(x, h), Point(x, draw_.rows - 1), COLOR_RED);
			if (i == medv) cv::line(draw_, Point(x, h), Point(x, draw_.rows - 1), COLOR_BLUE);
			if (i == medv && i == meanv)cv::line(draw_, Point(x, h), Point(x, draw_.rows - 1), cv::Scalar(128, 0, 128));
		}
		Mat draw;
		resize(draw_, draw, Size(div * lw, 256), 0, 0, INTER_NEAREST);
		for (int i = 0; i < div - 1; i++)
		{
			const int x = i * lw + int(lw * 0.5);
			const int xp = (i + 1) * lw + int(lw * 0.5);
			const double v = (histval[i] + 0) - histval[meanv];
			const double vp = (histval[i + 1] - histval[meanv]);
			int y = draw.rows - 1 - cvRound((draw.rows - 1) * exp(v * v / (-2.0 * ststd * ststd)));
			int yp = draw.rows - 1 - cvRound((draw.rows - 1) * exp(vp * vp / (-2.0 * ststd * ststd)));
			line(draw, Point(x, y), Point(xp, yp), COLOR_GRAY150);
			//line(draw, Point(x, y), Point(xp, yp), COLOR_RED);
		}
		Mat text_img = Mat::zeros(Size(draw.cols, 30), CV_8UC3);

		string text = format("ave%f", stmean);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		line(text_img, Point(0, 20), Point(10, 20), COLOR_RED, 2);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("med%f", stmed);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		line(text_img, Point(0, 20), Point(10, 20), COLOR_BLUE, 2);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("min%f", minv);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("max%f", maxv);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("std%f", ststd);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		text_img.setTo(0);
		text = format("step%f", interval * 10);
		putText(text_img, text, Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_WHITE);
		vconcat(draw, text_img, draw);

		//static int idx = 0;
		//imwrite(format("%04d.png", idx++), draw);
		//show();
		imshow(wname, draw);
	}

	void Stat::drawPlofilePlot(string wname, const double amp)
	{
		cp::Plot pt;
		pt.setIsDrawMousePosition(false);
		pt.setPlotTitle(0, "plofile");
		//pt.setKey(cp::Plot::NOKEY);
		pt.setPlotSymbolALL(0);
		double maxv = -DBL_MAX;
		double minv = DBL_MAX;
		for (int i = 0; i < getSize(); i++)
		{
			const double v = data[i] * amp;
			maxv = max(maxv, v);
			minv = min(minv, v);
			pt.push_back(i, v);
		}
		pt.push_back_HLine(maxv, 1);
		pt.push_back_HLine(minv, 2);
		pt.setPlotTitle(1, "max " + to_string(maxv));
		pt.setPlotTitle(2, "min " + to_string(minv));
		pt.plot(wname, false);
	}

}