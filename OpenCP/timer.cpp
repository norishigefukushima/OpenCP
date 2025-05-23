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

	double Timer::getTimeNormalizeAmp(const int mode)
	{
		double ret = 0.0;
		switch (mode)
		{
		case TIME_MSEC:
		default:			ret = 1000.0; break;
		case TIME_MICROSEC: ret = 1000000.0; break;
		case TIME_NSEC:		ret = 1000000000.0; break;
		case TIME_SEC:		ret = 1.0; break;
		case TIME_MIN:		ret = 1.0 / 60.0; break;
		case TIME_HOUR:		ret = 1.0 / (60.0 * 60.0); break;
		case TIME_DAY:		ret = 1.0 / (60.0 * 60.0 * 24.0); break;
		}
		return ret;
	}

	void Timer::convertTime(bool isPrint, std::string message)
	{
		int mode = timeMode;
		if (mode == TIME_AUTO)
		{
			mode = getAutoTimeMode();
			autoMode = mode;
		}

		cTime *= getTimeNormalizeAmp(mode);

		if (isPrint)
		{
			if (message == "")message = mes;
			switch (mode)
			{
			case TIME_NSEC:
				unit = "nsec"; break;
			case TIME_MICROSEC:
				unit = "microsec"; break;
			case TIME_MSEC:
				unit = "msec"; break;
			case TIME_SEC:
				unit = "sec"; break;
			case TIME_MIN:
				unit = "minute"; break;
			case TIME_HOUR:
				unit = "hour"; break;
			case TIME_DAY:
				unit = "day"; break;
			default:
				unit = "";
				cout << message << ": error" << endl; break;
			}
			cout << message << ": " << cTime << " " + unit << endl;
		}
	}

	void Timer::pushLapTime()
	{
		const double time = double(getTickCount() - pre) / (getTickFrequency());
		countIndex++;
		if (countIndex >= countIgnoringThreshold)
		{
			if (countMax == 0)
			{
				stat.push_back(time);
			}
			else
			{
				if (stat.getSize() < countMax)
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
		start();
	}

	double Timer::getTime(bool isPrint, std::string message)
	{
		cTime = double(getTickCount() - pre) / (getTickFrequency());
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
				if (stat.getSize() < countMax)
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

	double Timer::getLapTimeMean(bool isPrint, string message)
	{
		cTime = stat.getMean();
		convertTime(isPrint, message);
		return cTime;
	}

	void Timer::setTrimRate(double rate)
	{
		stat.setTrimRate(rate);
	}

	double Timer::getLapTimeTrimMean(bool isPrint, string message)
	{
		cTime = stat.getTrimMean();
		convertTime(isPrint, message);
		return cTime;
	}

	double Timer::getLapTimeMedian(bool isPrint, string message)
	{
		cTime = stat.getMedian();
		convertTime(isPrint, message);
		return cTime;
	}

	double Timer::getLapTimeMin(bool isPrint, string message)
	{
		cTime = stat.getMin();
		convertTime(isPrint, message);
		return cTime;
	}

	double Timer::getLapTimeMax(bool isPrint, string message)
	{
		cTime = stat.getMax();
		convertTime(isPrint, message);
		return cTime;
	}

	std::string Timer::getUnit() { return unit; };

	int Timer::getStatSize()
	{
		return stat.getSize();
	}

	void Timer::drawDistribution(string wname, int div)
	{
		if (stat.getSize() > 1)
			stat.drawDistribution(wname, div);
	}


	void Timer::drawDistribution(string wname, int div, double minv, double maxv)
	{
		if (stat.getSize() > 1)
			stat.drawDistribution(wname, div, minv, maxv);
	}

	void Timer::drawDistributionSigmaClip(string wname, int div, double sigmaclip)
	{
		if (stat.getSize() > 1)
			stat.drawDistributionSigmaClip(wname, div, sigmaclip);
	}

	void Timer::drawPlofilePlot(string wname)
	{
		const int mode = (timeMode == TIME_AUTO) ? getAutoTimeMode() : timeMode;

		if (stat.getSize() > 1)
			stat.drawPlofilePlot(wname, getTimeNormalizeAmp(mode));
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
	int DestinationTimePrediction::getAutoTimeMode(const double cTime)
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

	void DestinationTimePrediction::printTime(const double time, const string mes)
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
		cout << mes << ": " << format("%5.2f", cTime) << " " + unit;
	}

	double DestinationTimePrediction::cvtTick2Time(double tick, const bool isStateChange)
	{
		double cTime = tick / getTickFrequency();
		if (isStateChange)
		{
			timeMode = getAutoTimeMode(cTime);
		}

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
		int64 ret = (getTickCount() - time_stamp[0]);
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

	inline int getOrder(int order, int index)
	{
		if (index == 0 && order == 1) return 0;
		else if (index == 1 && order == 2 || (index == 1 && order == 3)) return 1;
		else if (index == 2 && order == 3) return 2;
		else return order;
	}


	double DestinationTimePrediction::predict_endstamp(const int idx, const int order_, const bool isDiff)
	{
		int order = getOrder(order_, idx);
		double ret = 0.0;
		double per = (idx) / (double)(loopMax);
		if (order == 0)
		{
			ret = double(time_stamp[idx] - time_stamp[0]) / per;
		}
		if (order == 1)
		{
			Mat a(idx + 1, 2, CV_64F);
			Mat b(idx + 1, 1, CV_64F);
			a.at<double>(0, 0) = 1.0;
			a.at<double>(0, 1) = 0.0;
			b.at<double>(0, 0) = 0.0;
			if (isDiff)
			{
				const double normalize = double(time_stamp[idx] - time_stamp[idx - 1]);
				for (int i = 1; i < idx + 1; i++)
				{
					const double t = (time_stamp[i] - time_stamp[i - 1]) / normalize;
					a.at<double>(i, 0) = 1.0;
					a.at<double>(i, 1) = i;
					b.at<double>(i, 0) = t;
				}

				Mat w, u, vt;
				cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
				SVD::backSubst(w, u, vt, b, coefficients);
				coefficients *= normalize;
				ret = double(time_stamp[idx] - time_stamp[0]);
				for (int i = idx; i < loopMax; i++)
				{
					double dv = coefficients.at<double>(1) * i + coefficients.at<double>(0);
					ret += max(dv, 0.0);
				}
			}
			else
			{
				const double normalize = double(time_stamp[idx] - time_stamp[0]);
				for (int i = 1; i < idx + 1; i++)
				{
					const double t = (time_stamp[i] - time_stamp[0]) / normalize;
					a.at<double>(i, 0) = 1.0;
					a.at<double>(i, 1) = i;
					b.at<double>(i, 0) = t;
				}

				Mat w, u, vt;
				cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
				SVD::backSubst(w, u, vt, b, coefficients);

				double v = loopMax;
				coefficients *= normalize;
				ret = coefficients.at<double>(1) * v + coefficients.at<double>(0);
			}
		}
		else if (order == 2)
		{
			Mat a(idx + 1, 3, CV_64F);
			Mat b(idx + 1, 1, CV_64F);
			a.at<double>(0, 0) = 1.0;
			a.at<double>(0, 1) = 0.0;
			a.at<double>(0, 2) = 0.0;
			b.at<double>(0, 0) = 0.0;
			if (isDiff)
			{
				const double normalize = double(time_stamp[idx] - time_stamp[idx - 1]);
				for (int i = 1; i < idx + 1; i++)
				{
					const double t = (time_stamp[i] - time_stamp[i - 1]) / normalize;
					a.at<double>(i, 0) = 1.0;
					a.at<double>(i, 1) = i;
					a.at<double>(i, 2) = i * i;
					b.at<double>(i, 0) = t;
				}

				Mat w, u, vt;
				cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
				SVD::backSubst(w, u, vt, b, coefficients);
				coefficients *= normalize;
				ret = double(time_stamp[idx] - time_stamp[0]);
				for (int i = idx; i < loopMax; i++)
				{
					double dv = coefficients.at<double>(2) * i * i + coefficients.at<double>(1) * i + coefficients.at<double>(0);
					ret += max(dv, 0.0);
				}
			}
			else
			{
				const double normalize = double(time_stamp[idx] - time_stamp[0]);
				for (int i = 1; i < idx + 1; i++)
				{
					const double t = (time_stamp[i] - time_stamp[0]) / normalize;
					a.at<double>(i, 0) = 1.0;
					a.at<double>(i, 1) = i;
					a.at<double>(i, 2) = i * i;
					b.at<double>(i, 0) = t;
				}

				Mat w, u, vt;
				cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
				SVD::backSubst(w, u, vt, b, coefficients);

				double v = loopMax;
				coefficients *= normalize;
				ret = coefficients.at<double>(2) * v * v + coefficients.at<double>(1) * v + coefficients.at<double>(0);
			}
		}
		else if (order == 3)
		{
			Mat a(idx + 1, 4, CV_64F);
			Mat b(idx + 1, 1, CV_64F);
			a.at<double>(0, 0) = 1.0;
			a.at<double>(0, 1) = 0.0;
			a.at<double>(0, 2) = 0.0;
			a.at<double>(0, 3) = 0.0;
			b.at<double>(0, 0) = 0.0;

			if (isDiff)
			{
				const double normalize = double(time_stamp[idx] - time_stamp[idx - 1]);
				for (int i = 1; i < idx + 1; i++)
				{
					const double t = (time_stamp[i] - time_stamp[i - 1]) / normalize;
					a.at<double>(i, 0) = 1.0;
					a.at<double>(i, 1) = i;
					a.at<double>(i, 2) = i * i;
					a.at<double>(i, 3) = i * i * i;
					b.at<double>(i, 0) = t;
				}
				Mat w, u, vt;
				cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
				SVD::backSubst(w, u, vt, b, coefficients);
				coefficients *= normalize;
				ret = double(time_stamp[idx] - time_stamp[0]);
				for (int i = idx; i < loopMax; i++)
				{
					double dv = coefficients.at<double>(3) * i * i * i + coefficients.at<double>(2) * i * i + coefficients.at<double>(1) * i + coefficients.at<double>(0);
					ret += max(dv, 0.0);
				}
			}
			else
			{
				const double normalize = double(time_stamp[idx] - time_stamp[0]);
				for (int i = 1; i < idx + 1; i++)
				{
					const double t = (time_stamp[i] - time_stamp[0]) / normalize;
					a.at<double>(i, 0) = 1.0;
					a.at<double>(i, 1) = i;
					a.at<double>(i, 2) = i * i;
					a.at<double>(i, 3) = i * i * i;
					b.at<double>(i, 0) = t;
				}

				Mat w, u, vt;
				cv::SVDecomp(a, w, u, vt, SVD::FULL_UV);
				SVD::backSubst(w, u, vt, b, coefficients);

				double v = loopMax;
				coefficients *= normalize;
				ret = coefficients.at<double>(3) * v * v * v + coefficients.at<double>(2) * v * v + coefficients.at<double>(1) * v + coefficients.at<double>(0);
			}
		}
		return ret;
	}

	std::pair<double, double> DestinationTimePrediction::predict(const int order, const bool isDiff, const bool isPrint, const bool isParallel)
	{
		double per = 0.0;
		double pred_stamp = 0.0;
		double etime = 0.0;
		double ctime = 0.0;
		if (isParallel)
		{
#pragma omp critical
			{
				const int64 cstamp = getTickCount();
				time_stamp.push_back(cstamp);
				double per = (time_stamp.size()) / (double)(loopMax);
				//print_debug2(time_stamp.size(), destCount);
				pred_stamp = predict_endstamp((int)time_stamp.size() - 1, order, isDiff);
				etime = cvtTick2Time(pred_stamp);
				ctime = cvtTick2Time(double(cstamp - time_stamp[0]), false);

				if (isPrint)
				{
					cout << format("%4.1f %% %d/%d, ", 100.0 * per, (int)time_stamp.size() - 1, loopMax);
					printTime(ctime, "current");
					printTime(etime - ctime, " | last");
					printTime(etime, " | estimated ");
					cout << endl;
				}
			}
		}
		else
		{
			const int64 cstamp = getTickCount();
			time_stamp.push_back(cstamp);
			double per = (time_stamp.size()) / (double)(loopMax);
			//print_debug2(time_stamp.size(), destCount);
			pred_stamp = predict_endstamp((int)time_stamp.size() - 1, order, isDiff);
			etime = cvtTick2Time(pred_stamp);
			ctime = cvtTick2Time(double(cstamp - time_stamp[0]), false);

			if (isPrint)
			{
				cout << format("%4.1f %% %d/%d, ", 100.0 * per, (int)time_stamp.size() - 1, loopMax);
				printTime(ctime, "current");
				printTime(etime - ctime, " | last");
				printTime(etime, " | estimated ");
				cout << endl;
			}
		}
	
		std::pair<double, double> ret(ctime, etime);
		return ret;
	}

	void DestinationTimePrediction::init(const int loopMax, const int timeMode)
	{
		this->timeMode = timeMode;
		this->loopMax = loopMax;
		time_stamp.push_back(getTickCount());
	}

	DestinationTimePrediction::DestinationTimePrediction()
	{
		;
	}

	DestinationTimePrediction::DestinationTimePrediction(const int loopMax, int timeMode)
	{
		init(loopMax, timeMode);
	}

	DestinationTimePrediction::~DestinationTimePrediction()
	{
		/*
		//double time = cvtTick2Time((double)(time_stamp[0] - last_tick));
		//print(time, "Actual ");

		//print(time, "diff from 1st prediction ");
		cp::Plot pt;
		pt.setKey(cp::Plot::LEFT_TOP);
		pt.setPlotTitle(0, "acctual");
		pt.setPlotLineWidth(0, 3);
		pt.setPlotLineWidth(2, 2);
		pt.setPlotLineWidth(4, 2);
		pt.setPlotLineWidth(6, 2);
		pt.setPlotTitle(1, "est o1");
		pt.setPlotTitle(2, "est o1 diff");
		pt.setPlotTitle(3, "est o2");
		pt.setPlotTitle(4, "est o2 diff");
		pt.setPlotTitle(5, "est o3");
		pt.setPlotTitle(6, "est o3 diff");
		for (int j = 1; j < time_stamp.size(); j++)
		{
			for (int i = 1; i < time_stamp.size(); i++)
			{
				int64 stamp = time_stamp[i] - time_stamp[0];
				double v = cvtTick2Time(stamp, false);
				pt.push_back(i, v, 0);
			}

			for (int k = 1; k <= j; k++)
			{
				int64 stamp = time_stamp[k] - time_stamp[0];
				double v = cvtTick2Time(stamp, false);
				pt.push_back(k, v, 1);
				pt.push_back(k, v, 2);
				pt.push_back(k, v, 3);
				pt.push_back(k, v, 4);
				pt.push_back(k, v, 5);
				pt.push_back(k, v, 6);
			}

			pt.push_back(time_stamp.size() - 1, cvtTick2Time(predict_endstamp(j, 1, false), false), 1);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(predict_endstamp(j, 1, true), false), 2);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(predict_endstamp(j, 2, false), false), 3);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(predict_endstamp(j, 2, true), false), 4);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(predict_endstamp(j, 3, false), false), 5);
			pt.push_back(time_stamp.size() - 1, cvtTick2Time(predict_endstamp(j, 3, true), false), 6);

			pt.plot("pred", false);
			pt.clear();
			waitKey();
		}
		*/
	}
#pragma endregion
}