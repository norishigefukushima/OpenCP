#pragma once

#include "common.hpp"
#include "stat.hpp"

namespace cp
{
	enum
	{
		TIME_AUTO = 0,
		TIME_NSEC,
		TIME_MSEC,
		TIME_SEC,
		TIME_MIN,
		TIME_HOUR,
		TIME_DAY
	};
	class CP_EXPORT CalcTime
	{
		int64 pre;
		std::string mes;

		int timeMode;

		double cTime;
		bool _isShow;

		int autoMode;
		int autoTimeMode();
		cp::Stat stat;
		
		void convertTime(bool isPrint, std::string message);
	public:

		void init(std::string message, int mode, bool isShow);
	
		void setMode(int mode);
		void setMessage(std::string& src);

		void start();
		void restart();

		double getTime(bool isPrint=false, std::string message = "");
		double getLapTime(bool isPrint = false, std::string message="");
		double getLapTimeMedian(bool isPrint = false, std::string message = "");
		double getLapTimeMean(bool isPrint = false, std::string message = "");
		
		CalcTime(std::string message, int mode = TIME_AUTO, bool isShow = true);
		CalcTime(char* message, int mode = TIME_AUTO, bool isShow = true);
		CalcTime();

		~CalcTime();
	};

	class CP_EXPORT DestinationTimePrediction
	{
	public:
		int destCount;
		int pCount;
		int64 startTime;

		int64 firstprediction;

		int64 prestamp;
		int64 prestamp_for_prediction;

		void init(int DestinationCount);
		DestinationTimePrediction();
		DestinationTimePrediction(int DestinationCount);
		int autoTimeMode(double cTime);
		void tick2Time(double tick, std::string mes);
		int64 getTime(std::string mes);
		~DestinationTimePrediction();
		void predict();
		double predict(int presentCount, int interval = 500);
	};
}