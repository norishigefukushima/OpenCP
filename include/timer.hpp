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

	/*
	Sample 1: compute time by using constructor and destructor for scope
	{	//must use scope
	Timer t;
	//some function
	}

	Sample 2: manually start timer
	Timer t;
	t.start()
	//some function
	t.getTime()

	Sample 3: skip start timer (start function is called by constructor) 
	Timer t;
	//some function
	t.getTime()

	Sample 4: compute mean or median value of trails
	Timer t;
	for(int i=0;i<loop;i++)
	{
		t.start()
		//some function
		t.getpushLapTime()
	}
	t.getLapTimeMean();
	t.getLapTimeMedian();
	t.clearStat();//clear stat and then re-compute computing time
	for(int i=0;i<loop;i++)
	{
		t.start()
		//some function
		t.getpushLapTime()
	}
	t.getLapTimeMean();
	t.getLapTimeMedian();
	*/
	class CP_EXPORT Timer
	{
		int64 pre;
		std::string mes;

		int timeMode;

		double cTime;
		bool _isShow;

		int autoMode;
		int autoTimeMode();
		cp::Stat stat;

		int countIgnoringThreshold;
		int countMax;
		int countIndex;

		void convertTime(bool isPrint, std::string message);
	public:

		void init(std::string message, int mode, bool isShow);

		void setMode(int mode);
		void setMessage(std::string& src);

		void start();
		void clearStat();

		void setCountMax(const int value);
		void setIgnoringThreshold(const int value);
		double getTime(bool isPrint = false, std::string message = "");
		double getpushLapTime(bool isPrint = false, std::string message = "");
		double getLapTimeMedian(bool isPrint = false, std::string message = "");
		double getLapTimeMean(bool isPrint = false, std::string message = "");
		int getStatSize();
		void drawDistribution(std::string wname = "Stat distribution", int div = 100);
		void drawDistribution(std::string wname, int div, double minv, double maxv);

		Timer(std::string message, int mode = TIME_AUTO, bool isShow = true);
		Timer(char* message, int mode = TIME_AUTO, bool isShow = true);
		Timer();

		~Timer();
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