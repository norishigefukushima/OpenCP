#include "timer.hpp"
#include "debugcp.hpp"
using namespace std;
using namespace cv;

namespace cp
{
#pragma region Timer
	void Timer::start()
	{
		pre = getTickCount();
	}

	void Timer::clearStat()
	{
		pre = getTickCount();
		stat.clear();
	}

	int Timer::getAutoTimeMode()
	{
		if (cTime > 60.0 * 60.0 * 24.0)
		{
			return TIME_DAY;
		}
		else if (cTime > 60.0 * 60.0)
		{
			return TIME_HOUR;
		}
		else if (cTime > 60.0)
		{
			return TIME_MIN;
		}
		else if (cTime > 1.0)
		{
			return TIME_SEC;
		}
		else if (cTime > 1.0 / 1000.0)
		{
			return TIME_MSEC;
		}
		else if (cTime > 1.0 / 1000000.0)
		{
			return TIME_MICROSEC;
		}
		else
		{
			return TIME_NSEC;
		}
	}

	void Timer::convertTime(bool isPrint, std::string message)
	{
		int mode = timeMode;
		if (mode == TIME_AUTO)
		{
			mode = getAutoTimeMode();
			autoMode = mode;
		}

		switch (mode)
		{
		case TIME_SEC:
			cTime *= 1.0;
			break;
		case TIME_MIN:
			cTime /= (60.0);
			break;
		case TIME_HOUR:
			cTime /= (60 * 60);
			break;
		case TIME_DAY:
			cTime /= (60 * 60 * 24);
			break;
		case TIME_NSEC:
			cTime *= 1000000000.0;
			break;
		case TIME_MICROSEC:
			cTime *= 1000000.0;
			break;
		case TIME_MSEC:
		default:
			cTime *= 1000.0;
			break;
		}

		if (isPrint)
		{
			if (message == "")message = mes;
			switch (mode)
			{
			case TIME_NSEC:
				unit = "nsec";
			case TIME_MICROSEC:
				unit = "microsec";
			case TIME_MSEC:
				unit = "msec";
			case TIME_SEC:
				unit = "sec";
			case TIME_MIN:
				unit = "minute";
			case TIME_HOUR:
				unit = "hour";
			case TIME_DAY:
				unit = "day";
			default:
				unit = "";
				cout << message << ": error" << endl; break;
			}
			cout << message << ": " << cTime << " " + unit << endl;
		}
	}

	double Timer::getTime(bool isPrint, std::string message)
	{
		cTime = (getTickCount() - pre) / (getTickFrequency());
		convertTime(isPrint, message);
		return cTime;
	}

	double Timer::getpushLapTime(bool isPrint, string message)
	{
		double time = (getTickCount() - pre) / (getTickFrequency());
		countIndex++;
		if (countIndex >= countIgnoringThreshold)
		{
			if (countMax == 0)
			{
				stat.push_back(time);
			}
			else
			{
				if (stat.num_data < countMax)
				{
					stat.push_back(time);
				}
				else
				{
					countIndex = countIndex % countMax;
					stat.data[countIndex] = time;
				}
			}
		}

		cTime = time;
		convertTime(isPrint, message);

		start();
		return cTime;
	}

	double Timer::getLapTimeMedian(bool isPrint, string message)
	{
		cTime = stat.getMedian();
		convertTime(isPrint, message);
		return cTime;
	}

	double Timer::getLapTimeMean(bool isPrint, string message)
	{
		cTime = stat.getMean();
		convertTime(isPrint, message);
		return cTime;
	}

	std::string Timer::getUnit() { return unit; };

	int Timer::getStatSize()
	{
		return stat.num_data;
	}

	void Timer::drawDistribution(string wname, int div)
	{
		if (stat.num_data > 1)
			stat.drawDistribution(wname, div);
	}

	void Timer::drawDistribution(string wname, int div, double minv, double maxv)
	{
		if (stat.num_data > 1)
			stat.drawDistribution(wname, div, minv, maxv);
	}

	void Timer::setMessage(string& src)
	{
		mes = src;
	}

	void Timer::setMode(int mode)
	{
		timeMode = mode;
	}

	void Timer::setIsShow(const bool flag)
	{
		this->isShow = flag;
	}

	void Timer::setCountMax(const int value)
	{
		countMax = value;
	}

	void Timer::setIgnoringThreshold(const int value)
	{
		countIgnoringThreshold = value;
	}

	void Timer::init(string message, int mode, bool isShow)
	{
		this->isShow = isShow;
		timeMode = mode;
		setMessage(message);
		start();
	}

	Timer::Timer()
	{
		string t = "time ";
		init(t, TIME_AUTO, true);
	}

	Timer::Timer(char* message, int mode, bool isShow)
	{
		string m = message;
		init(m, mode, isShow);
	}

	Timer::Timer(string message, int mode, bool isShow)
	{
		init(message, mode, isShow);
	}

	Timer::~Timer()
	{
		if (isShow)	getTime(true);
	}
#pragma endregion

#pragma region DestinationTimePrediction
	int DestinationTimePrediction::getAutoTimeMode(double cTime)
	{
		if (cTime > 60.0 * 60.0 * 24.0)
		{
			return TIME_DAY;
		}
		else if (cTime > 60.0 * 60.0)
		{
			return TIME_HOUR;
		}
		else if (cTime > 60.0)
		{
			return TIME_MIN;
		}
		else if (cTime > 1.0)
		{
			return TIME_SEC;
		}
		else if (cTime > 1.0 / 1000.0)
		{
			return TIME_MSEC;
		}
		else if (cTime > 1.0 / 1000000.0)
		{
			return TIME_MICROSEC;
		}
		else
		{
			return TIME_NSEC;
		}
	}

	void DestinationTimePrediction::print(double time, string mes)
	{
		double cTime = time;

		switch (timeMode)
		{
		case TIME_NSEC:
			unit = "nsec"; break;
		case TIME_MICROSEC:
			unit = "microsec"; break;
		case TIME_MSEC:
			unit = "msec"; break;
		case TIME_SEC:
		default:
			unit = "sec"; break;
		case TIME_MIN:
			unit = "min"; break;
		case TIME_HOUR:
			unit = "hour"; break;
		case TIME_DAY:
			unit = "day"; break;
		}
		cout << mes << ": " << format("%.2f", cTime) << " " + unit;
	}

	double DestinationTimePrediction::cvtTick2Time(double tick, const bool isStateChange)
	{
		double cTime = tick / getTickFrequency();
		if (isStateChange)
			timeMode = getAutoTimeMode(cTime);

		switch (timeMode)
		{
		case TIME_NSEC:
			cTime *= 1000000000.0;
			break;
		case TIME_MICROSEC:
			cTime *= 1000000.0;
			break;
		case TIME_MSEC:
			cTime *= 1000.0;
			break;
		case TIME_SEC:
		default:
			cTime *= 1.0;
			break;
		case TIME_MIN:
			cTime /= (60.0);
			break;
		case TIME_HOUR:
			cTime /= (60 * 60);
			break;
		case TIME_DAY:
			cTime /= (60 * 60 * 24);
			break;
		}

		return cTime;
	}

	int64 DestinationTimePrediction::getTime(string mes)
	{
		int64 ret = (getTickCount() - startTime);
		double cTime = ret / (getTickFrequency());

		int timeMode = getAutoTimeMode(cTime);

		switch (timeMode)
		{
		case TIME_MSEC:
			cTime *= 1000.0;
			break;
		case TIME_SEC:
		default:
			cTime *= 1.0;
			break;
		case TIME_MIN:
			cTime /= (60.0);
			break;
		case TIME_HOUR:
			cTime /= (60 * 60);
			break;
		case TIME_DAY:
			cTime /= (60 * 60 * 24);
			break;
		}

		switch (timeMode)
		{
		case TIME_SEC:
		default:
			cout << mes << ": " << format("%.2f", cTime) << " sec                 ";
			break;
		case TIME_MIN:
			cout << mes << ": " << format("%.2f", cTime) << " minute              ";
			break;
		case TIME_HOUR:
			cout << mes << ": " << format("%.2f", cTime) << " hour                ";
			break;
		case TIME_DAY:
			cout << mes << ": " << format("%.2f", cTime) << " day                 ";
			break;
		case TIME_MSEC:
			cout << mes << ": " << format("%.2f", cTime) << " msec" << endl;
			break;
		}
		return ret;
	}

	double DestinationTimePrediction::predict_endstamp(const int idx, const int order)
	{
		double ret = 0.0;
		double per = (idx) / (double)(destCount);
		if (order == 1 || idx == 1)
		{
			ret = double(time_stamp[idx] - time_stamp[0]) / per;
		}
		else if (order == 2)
			//if (idx > 1)
		{
			Mat a(idx + 1, 3, CV_64F);
			Mat b(idx + 1, 1, CV_64F);
			a.at<double>(0, 0) = 1.0;
			a.at<double>(0, 1) = 0.0;
			a.at<double>(0, 2) = 0.0;
			b.at<double>(0, 0) = 0.0;
			for (int i = 1; i < idx + 1; i++)
			{
				const double t = (time_stamp[i] - time_stamp[0]);
				a.at<double>(i, 0) = 1.0;
				a.at<double>(i, 1) = i;
				a.at<double>(i, 2) = i * i;
				b.at<double>(i, 0) = t;
			}
		
			Mat w, u, vt;
			cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
			SVD::backSubst(w, u, vt, b, coefficients);

			double v = destCount;
			ret = coefficients.at<double>(2) * v*v + coefficients.at<double>(1) * v + coefficients.at<double>(0);
			//cout << per * 100 << ": ";
			//print(cvtTick2Time(val, false), "est2");
			//cout << endl;

		}
		return ret;
	}

	void DestinationTimePrediction::predict()
	{
		pCount++;
		/*if (pCount < 11)
		{
			int64 v = (int64)predict(pCount, 10);
			firstprediction = (v > 0) ? v : firstprediction;
		}
		else*/
		{
			predict(pCount);
		}
	}

	double DestinationTimePrediction::predict(int presentCount, int interval)
	{
		double ret = 0.0;
		int64 cstamp = getTickCount();
		time_stamp.push_back(cstamp);
		double per = (time_stamp.size() - 1) / (double)(destCount);
		//print_debug2(time_stamp.size(), destCount);
		double pred_stamp = predict_endstamp(time_stamp.size() - 1);
		double etime = cvtTick2Time(pred_stamp);
		double ctime = cvtTick2Time(cstamp - time_stamp[0], false);

		cout << format("%.3f %% computed, ", 100.0 * per);
		print(etime - ctime, "last ");
		print(etime, " estimated total");
		cout << endl;
		/*
		if ((presentCount % interval) == 0)
		{
			double per = (double)presentCount / destCount;
			//double ctime = (getTickCount()-preTime)/(getTickFrequency());
			int64 cstamp = getTickCount();
			double pret = ((double)destCount / (double)interval) * (cstamp - prestamp_for_prediction);

			pret = (destCount - presentCount) / (double)interval * (cstamp - prestamp_for_prediction);
			ret = pret;
			prestamp_for_prediction = cstamp;

			double cTime = pret / (getTickFrequency());

			int timeMode = getAutoTimeMode(cTime);

			switch (timeMode)
			{
			case TIME_SEC:
			default:
				cTime *= 1.0;
				break;
			case TIME_MIN:
				cTime /= (60.0);
				break;
			case TIME_HOUR:
				cTime /= (60 * 60);
				break;
			case TIME_DAY:
				cTime /= (60 * 60 * 24);
				break;
			}

			//cout << "\r";
			cout << "\n";

			string mes = format("%.3f %% computed, rest ", 100.0 * per);

			switch (timeMode)
			{
			case TIME_SEC:
			default:
				cout << mes << ": " << format("%.2f", cTime) << " sec                 ";
				break;
			case TIME_MIN:
				cout << mes << ": " << format("%.2f", cTime) << " minute              ";
				break;
			case TIME_HOUR:
				cout << mes << ": " << format("%.2f", cTime) << " hour                ";
				break;
			case TIME_DAY:
				cout << mes << ": " << format("%.2f", cTime) << " day                 ";
				break;
			}

			prestamp = cstamp;
		}
		*/

		return ret;
	}

	void DestinationTimePrediction::init(int DestinationCount)
	{
		pCount = 0;
		destCount = DestinationCount;
		startTime = getTickCount();
		time_stamp.push_back(startTime);

		prestamp_for_prediction = startTime;
		prestamp = startTime;

		firstprediction = 0;
	}

	DestinationTimePrediction::DestinationTimePrediction()
	{
		;
	}

	DestinationTimePrediction::DestinationTimePrediction(int DestinationCount)
	{
		init(DestinationCount);
	}

	DestinationTimePrediction::~DestinationTimePrediction()
	{
		int64 last_tick = getTime("actual ");
		//double time = cvtTick2Time((double)(time_stamp[0] - last_tick));
		//print(time, "Actual ");
		cout << endl;
		//print(time, "diff from 1st prediction ");
		cp::Plot pt;
		pt.setPlotTitle(0, "acctual");
		pt.setPlotTitle(1, "est o1");
		pt.setPlotTitle(2, "est o2");
		pt.setPlotTitle(3, "estfit o2");
		for (int j = 1; j < time_stamp.size(); j++)
		{
			for (int k = 1; k <= j; k++)
			{
				int64 stamp = time_stamp[k] - time_stamp[0];
				double v = cvtTick2Time(stamp, false);
				pt.push_back(k, v, 1);
				pt.push_back(k, v, 2);
			}
			double pred1 = predict_endstamp(j, 1);
			double pred2 = predict_endstamp(j, 2);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(pred1, false), 1);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(pred2, false), 2);

			for (int i = 1; i < time_stamp.size(); i++)
			{
				if (coefficients.size().area() == 3)
				{
					double v = i;
					double pred = coefficients.at<double>(2) * v * v + coefficients.at<double>(1) * v + coefficients.at<double>(0);
					pt.push_back(i, cvtTick2Time(pred, false), 3);
				}
			}
			for (int i = 1; i < time_stamp.size(); i++)
			{
				int64 stamp = time_stamp[i] - time_stamp[0];
				double v = cvtTick2Time(stamp, false);
				pt.push_back(i, v, 0);
			}
			pt.plot("pred", false);
			pt.clear();
			waitKey();
		}

	}
#pragma endregion
}