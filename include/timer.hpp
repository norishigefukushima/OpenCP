#pragma once

#include "common.hpp"
#include "stat.hpp"

namespace cp
{
	enum
	{
		TIME_AUTO = 0,
		TIME_NSEC,
		TIME_MICROSEC,
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
		int64 pre = 0;
		std::string mes = "";
		std::string unit = "";
		int timeMode = 0;

		double cTime = 0.0;
		bool isShow = true;

		int autoMode = 0;
		int getAutoTimeMode();
		cp::Stat stat;

		int countIgnoringThreshold = 1;
		int countMax = 0;
		int countIndex = 0;

		double getTimeNormalizeAmp(const int unit);
		void convertTime(bool isShow, std::string message);
	public:

		void init(std::string message, int mode, bool isShow);

		void setMode(int mode);
		void setMessage(std::string& src);
		void setIsShow(const bool flag);

		void start();//call getTickCount();
		void clearStat();//clear Stat

		void setCountMax(const int value);//set ring buffer max (loop value max) for Stat. Default is infinity.
		void setIgnoringThreshold(const int value);//if(sample number < value) does not push the value into Stat for ignure cache 
		void pushLapTime();//push the value to Stat
		double getTime(bool isPrint = false, std::string message = "");//only getTickCount()
		double getpushLapTime(bool isPrint = false, std::string message = "");//getTickCount() and push the value to Stat

		double getLapTimeMean(bool isPrint = false, std::string message = "");//get mean value from Stat
		void setTrimRate(double rate);
		double getLapTimeTrimMean(bool isPrint = false, std::string message = "");//get mean value from Stat
		double getLapTimeMedian(bool isPrint = false, std::string message = "");//get median value from Stat
		double getLapTimeMin(bool isPrint = false, std::string message = "");//get min value from Stat
		double getLapTimeMax(bool isPrint = false, std::string message = "");//get max value from Stat

		std::string getUnit();//return string unit
		int getStatSize();//get the size of Stat
		void drawDistribution(std::string wname = "Stat distribution", int div = 100);
		void drawDistribution(std::string wname, int div, double minv, double maxv);
		void drawDistributionSigmaClip(std::string wname, int div, double sigmaclip);
		void drawPlofilePlot(std::string wname);


		Timer(std::string message, int mode = TIME_AUTO, bool isShow = true);
		Timer(char* message, int mode = TIME_AUTO, bool isShow = true);
		Timer();

		~Timer();
	};

	class CP_EXPORT DestinationTimePrediction
	{
		cv::Mat coefficients;
		int getAutoTimeMode(const double cTime);
		std::string unit = "";
		int timeMode = TIME_AUTO;
		std::vector<int64> time_stamp;
		int order = 1;
		int loopMax = 0;
		double predict_endstamp(const int idx, const int order = 1, const bool isDiff = true);
		int64 getTime(std::string mes);
		void printTime(const double time, const std::string mes);
		double cvtTick2Time(const double tick, const bool isStateChange = true);
	public:
		//order=0: average prediction, return pair(current, estimated)
		std::pair<double, double> predict(const int order = 0, const bool isDiff = false, const bool isPrint = true, const bool isParallel = false);

		void init(const int loopCountMax, const int timeMode = TIME_AUTO);
		DestinationTimePrediction();
		DestinationTimePrediction(const int loopCountMax, int timeMode = TIME_AUTO);
		~DestinationTimePrediction();
	};
}