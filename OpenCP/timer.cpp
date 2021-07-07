#include "timer.hpp"

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
			cout << message << ": " << cTime << " "+unit << endl;
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

	void DestinationTimePrediction::tick2Time(double tick, string mes)
	{
		double cTime = tick / (getTickFrequency());

		int timeMode = getAutoTimeMode(cTime);

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

		switch (timeMode)
		{
		case TIME_NSEC:
			cout << mes << ": " << format("%.2f", cTime) << " nsec                 ";
			break;
		case TIME_MICROSEC:
			cout << mes << ": " << format("%.2f", cTime) << " microsec                 ";
			break;
		case TIME_MSEC:
			cout << mes << ": " << format("%.2f", cTime) << " msec                 ";
			break;
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

	
	void DestinationTimePrediction::predict()
	{
		pCount++;
		if (pCount < 11)
		{
			int64 v = (int64)predict(pCount, 10);
			firstprediction = (v > 0) ? v : firstprediction;
		}
		else
		{
			predict(pCount);
		}
	}

	double DestinationTimePrediction::predict(int presentCount, int interval)
	{
		double ret = 0.0;
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
		return ret;
	}

	void DestinationTimePrediction::init(int DestinationCount)
	{
		pCount = 0;
		destCount = DestinationCount;
		startTime = getTickCount();

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
		int64 a = getTime("actual ");
		tick2Time((double)(firstprediction - a), "diff from 1st prediction ");
	}
#pragma endregion
}