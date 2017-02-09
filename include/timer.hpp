#pragma once

#include "common.hpp"

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
		std::vector<std::string> lap_mes;
	public:

		void start();
		void setMode(int mode);
		void setMessage(std::string& src);
		void restart();
		double getTime();
		void show();
		void show(std::string message);
		void lap(std::string message);
		void init(std::string message, int mode, bool isShow);

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