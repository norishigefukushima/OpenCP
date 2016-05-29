#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{


	Stat::Stat(){ ; }
	Stat::~Stat(){ ; }

	double Stat::getMin()
	{
		double minv = DBL_MAX;
		for (int i = 0; i < num_data; i++)
		{
			minv = min(minv, data[i]);
		}
		return minv;
	}

	double Stat::getMax()
	{
		double maxv = DBL_MIN;
		for (int i = 0; i < num_data; i++)
		{
			maxv = max(maxv, data[i]);
		}
		return maxv;
	}

	double Stat::getMean()
	{
		double sum = 0.0;
		for (int i = 0; i < num_data; i++)
		{
			sum += data[i];
		}
		return sum / (double)num_data;
	}

	double Stat::getStd()
	{
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
		if (data.size() == 0) return 0.0;
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
	
	void Stat::show()
	{
		cout << "mean " << getMean() << endl;
		cout << "min  " << getMin() << endl;
		cout << "med  " << getMedian() << endl;
		cout << "max  " << getMax() << endl;
		cout << "std  " << getStd() << endl;
	}

	void Stat::showDistribution()
	{
		Mat hist;
		
		drawHistogramImage(data, hist, CV_RGB(255, 255, 255));
		showMatInfo(hist);
		imshow("dist", hist);
	}
}