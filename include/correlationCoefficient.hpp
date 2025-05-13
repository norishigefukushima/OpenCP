#pragma once

#include "common.hpp"
#include "plot.hpp"

namespace cp
{
	class CP_EXPORT PearsonCorrelationCoefficient
	{
		template<typename T>
		double mean(std::vector<T>& s);
		template<typename T>
		double covariance(std::vector<T>& s1, std::vector<T>& s2);
		template<typename T>
		double stddev(std::vector<T>& v1);
	public:
		double compute(std::vector<int>& v1, std::vector<int>& v2);
		double compute(std::vector<float>& v1, std::vector<float>& v2);
		double compute(std::vector<double>& v1, std::vector<double>& v2);
	};

	class CP_EXPORT KendallRankOrderCorrelationCoefficient
	{
		//not optimized but can be vectorized, not parallelization is off
		template<typename T>
		double body(std::vector<T>& x, std::vector<T>& y);
	public:

		double compute(std::vector<int>& x, std::vector<int>& y);
		double compute(std::vector<float>& x, std::vector<float >& y);
		double compute(std::vector<double>& x, std::vector<double>& y);
	};


	class CP_EXPORT SpearmanRankOrderCorrelationCoefficient
	{
		template<typename T>
		double rankTransformUsingAverageTieScore(std::vector<T>& src, std::vector<float>& dst);
		template<typename T>
		void rankTransformIgnoreTie(std::vector<T>& src, std::vector<float>& dst);
		template<typename T>
		void rankTransformBruteForce(std::vector<T>& src, std::vector<float>& dst);//not fast, and not used

		template<typename T>
		double computeRankDifference(std::vector<T>& Rsrc, std::vector<T>& Rref);

		template<typename T>
		double spearman_(std::vector<T>& v1, std::vector<T>& v2, const bool ignoreTie, const bool isPlot, const int plotIndex);

		template<typename T>
		void setPlotData(const std::vector<T>& v1, const std::vector<T>& v2, std::vector<cv::Point2d>& data);

		std::vector<size_t> indices;

		cp::Plot pt;
		std::vector<std::vector<cv::Point2d>> plotsRAW;
		std::vector<std::vector<cv::Point2d>> plotsRANK;
		double Tref = 0.0;
		std::vector<float> refRank;
		std::vector<float> srcRank;

		template<typename T>
		struct SpearmanOrder
		{
			T data;
			int order;
		};

		std::vector<SpearmanOrder<int>> sporder32i;
		std::vector<SpearmanOrder<float>> sporder32f;
		std::vector<SpearmanOrder<double>> sporder64f;
	public:
		SpearmanRankOrderCorrelationCoefficient();
		void setReference(std::vector<int>& ref, const bool ignoreTie = false);
		void setReference(std::vector<float>& ref, const bool ignoreTie = false);
		void setReference(std::vector<double>& ref, const bool ignoreTie = false);
		double computeUsingReference(std::vector<int>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<float>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<double>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<std::vector<int>>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<std::vector<float>>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<std::vector<double>>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double compute(std::vector<int>& v1, std::vector<int>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<int> thread safe). 
		double compute(std::vector<float>& v1, std::vector<float>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<float> thread safe). 
		double compute(std::vector<double>& v1, std::vector<double>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC ((vector<double>) thread safe). 
		std::vector<double> compute(std::vector<std::vector<int>>& v1, std::vector<std::vector<int>>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<int> thread safe). 
		std::vector<double> compute(std::vector<std::vector<float>>& v1, std::vector<std::vector<float>>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<float> thread safe). 
		std::vector<double> compute(std::vector<std::vector<double>>& v1, std::vector<std::vector<double>>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC ((vector<double>) thread safe). 
		void plot(const bool isWait = true, const double rawMin = 0.0, const double rawMax = 0.0, std::vector<std::string> labels = std::vector<std::string>());
		void plotwithAdditionalPoints(const std::vector<std::vector<cv::Point2d>>& additionalPoint, const bool isWait = true, const double rawMin = 0.0, const double rawMax = 0.0, std::vector<std::string> labels = std::vector<std::string>(), std::vector<std::string> xylabels = std::vector<std::string>());
		void plotwithAdditionalPoints(const std::vector<cv::Point2d>& additionalPoint, const bool isWait = true, const double rawMin = 0.0, const double rawMax = 0.0, std::vector<std::string> labels = std::vector<std::string>(), std::vector<std::string> xylabels = std::vector<std::string>());
		void plotwithAdditionalPoints(const cv::Point2d& additionalPoint, const bool isWait = true, const double rawMin = 0.0, const double rawMax = 0.0, std::vector<std::string> labels = std::vector<std::string>(), std::vector<std::string> xylabels = std::vector<std::string>());
	};
}
